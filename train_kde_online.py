"""
Fine-tuning BEV-only com KDE offline.

Apenas o BEV generator é atualizado. A policy está congelada e funciona
somente como driver para gerar observações diversas no ambiente.

O KDE, calculado offline sobre o dataset expert, pondera o loss do BEV:
ações raras (curvas, cruzamentos) recebem peso alto e contribuem mais
para o gradiente — ações comuns (retas) ficam com peso baixo.

Lógica por iteração (GRAD_ACCUM_STEPS passos):

  Fase 1 — Coleta (th.no_grad):
    obs_rgb → bev_gen.infer() → bev_bin → policy.forward() → action (+ruído)
    armazena na CPU: (image_input, gt_bev, action)

  Fase 2 — KDE weighting:
    kde_weights = compute_kde_weights(kde, all_actions)
    kde_weights.clamp_(max=KDE_WEIGHT_MAX_CLIP)

  Fase 3 — Update BEV (com grafo):
    pred = bev_gen.forward_train(image_input.to(GPU))
    loss = mean(kde_weights[i] * bev_gen.compute_loss(pred, gt_bev))
    loss.backward()
"""

import json
import numpy as np
import torch as th
import torch.nn.functional as F
import gym
import wandb
import tqdm
from pathlib import Path
from typing import Literal, Optional

from stable_baselines3.common.vec_env import SubprocVecEnv

from agent_policy import AgentPolicy
from carla_gym.envs import EndlessEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from data_collect import reward_configs, terminal_configs
from config.obs_config import get_obs_configs
from expert_dataset_def.kde_loader import load_kde, compute_kde_weights
from bev_generation.unet import Unet_BEVGenerator
from bev_generation.cvt_3ch import CVT_3chL1Generator
from bev_generation.cvt_6ch import CVT_6chVanilla
from bev_generation.IBEV_Generator import IBEVGenerator
from eval_agent import create_image_tensor

from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# Configuração
# ============================================================
BEV_ARC: Literal['unet', 'cvt', 'cvt_6ch'] = 'cvt'
DEVICE = 'cuda'

# Dataset expert: usado apenas para o fit do KDE (não entra no loop de treino)
DATASET_DIR = 'expert-data'
TRAIN_ROUTES = range(2, 10)
N_EPS = 1

TOWN_NAME = 'Town01'
TOTAL_STEPS = 500_000

# Janela de rollout: passos coletados antes de cada update do BEV generator.
# Maior janela → estimativa de densidade KDE mais representativa por update.
GRAD_ACCUM_STEPS = 16

LR_BEV = 1e-4
GRAD_CLIP = 1.0

# Ruído gaussiano adicionado às ações antes de env.step().
# Força o carro a posições ligeiramente não-usuais (mais curvas, cruzamentos).
ACTION_NOISE_STD = 0.05

# Clip máximo dos pesos KDE — evita que um único passo outlier domine o gradiente.
KDE_WEIGHT_MAX_CLIP = 5.0

CKPT_EVERY = 10_000
WANDB_PROJECT = f'kde-online-{BEV_ARC}'

# Pasta-raiz da BEV por arquitetura (usada para montar o dir de ckpt do gerador)
ARC_BEV_FOLDER = {
    'unet':    'bev_generation/unet',
    'cvt':     'bev_generation/cvt_3ch',
    'cvt_6ch': 'bev_generation/cvt_6ch',
}

# Caminho padrão dos pesos pré-treinados de cada gerador (registrado no JSON de info)
ARC_BEV_DEFAULT_CKPT = {
    'unet':    'bev_generation/unet/l1_50_no sigmoid_generator_49.pth',
    'cvt':     'bev_generation/cvt_3ch/ckpts/4_cam_l1/ckpt_49.pth',
    'cvt_6ch': 'bev_generation/cvt_6ch/ckpts/ckpt_vanilla_49.pth',
}

# ---- Checkpoints de origem ----
# Policy pré-treinada (congelada — não é atualizada durante o treino).
POLICY_PRETRAINED_CKPT: Optional[str] = 'ckpts/ckpt-real-bev/ckpt_latest.pth'

# BEV generator: None → usa pesos padrão de ARC_BEV_DEFAULT_CKPT (carregados em __init__).
BEV_GEN_CKPT: Optional[str] = None

# Para retomar treino anterior: aponte para o bev_latest.pth de uma run anterior.
RESUME_BEV_CKPT: Optional[str] = None  # ex: 'bev_generation/cvt_3ch/ckpts/finetune-kde/<run>/bev_latest.pth'

# Canais do obs['birdview'] usados como alvo para o BEV generator.
BEV_TARGET_CH = 3

env_configs = {
    'carla_map': TOWN_NAME,
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'ClearNoon',
}


# ============================================================
# Helpers
# ============================================================

def make_env():
    cfg = json.load(open("config/carla_config.json", "r"))
    obs_configs = get_obs_configs(BEV_ARC)
    env = EndlessEnv(
        obs_configs=obs_configs,
        reward_configs=reward_configs,
        terminal_configs=terminal_configs,
        host=cfg['host'],
        port=cfg['port'],
        seed=2021,
        no_rendering=False,
        **env_configs,
    )
    env = RlBirdviewWrapper(env, input_states=['rgb', 'traj', 'state', 'matrices'], acc_as_action=True)
    return env


def get_bev_generator(arc: str, device: str, model_path: Optional[str] = None) -> IBEVGenerator:
    kwargs = dict(device=device, use_eval=False)
    if model_path is not None:
        kwargs['model_path'] = model_path
    if arc == 'unet':
        return Unet_BEVGenerator(**kwargs)
    elif arc == 'cvt':
        return CVT_3chL1Generator(**kwargs)
    elif arc == 'cvt_6ch':
        return CVT_6chVanilla(**kwargs)
    raise ValueError(f"Unknown BEV architecture: {arc}")


def build_image_input(obs: dict, arc: str, device: str) -> dict:
    """Constrói o dict de entrada do BEV generator a partir do obs do VecEnv."""
    if arc == 'unet':
        return {'image': create_image_tensor(obs, unet=True).to(device)}
    elif arc == 'cvt':
        image = create_image_tensor(obs, unet=False, w_resize=256, h_resize=256).to(device)
    elif arc == 'cvt_6ch':
        image = create_image_tensor(obs, unet=False, w_resize=480, h_resize=224).to(device)
    else:
        raise ValueError(f"Unknown arc: {arc}")
    return {
        'image': image,
        'extrinsics': th.as_tensor(obs['extrinsics'], dtype=th.float32).to(device),
        'intrinsics': th.as_tensor(obs['intrinsics'], dtype=th.float32).to(device),
    }


def to_cpu(image_input: dict) -> dict:
    """Move um image_input dict para CPU (para armazenamento no rollout buffer)."""
    return {k: v.cpu() if isinstance(v, th.Tensor) else v for k, v in image_input.items()}


def to_device(image_input: dict, device: str) -> dict:
    """Move um image_input dict para o device alvo (para re-forward na fase de update)."""
    return {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in image_input.items()}


def binarize_for_policy(bev_raw: th.Tensor, arc: str) -> th.Tensor:
    """Converte saída raw do generator para formato {0, 255} que a policy espera."""
    threshold = 0.0 if arc == 'unet' else 0.5
    return (bev_raw.detach() > threshold).byte() * 255


def get_bev_target(obs: dict, device: str) -> th.Tensor:
    """Extrai o BEV ground-truth do obs e normaliza para [0, 1]."""
    real_bev = obs.get('birdview', obs.get('topdown'))
    target = th.as_tensor(real_bev, dtype=th.float32, device=device)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    return target[:, :BEV_TARGET_CH, :, :] / 255.0


# ============================================================
# Treino principal
# ============================================================

def train():
    run = wandb.init(project=WANDB_PROJECT, reinit=True)

    # Diretório de checkpoint do BEV generator
    bev_ckpt_dir = Path(ARC_BEV_FOLDER[BEV_ARC]) / 'ckpts' / 'finetune-kde' / run.name
    bev_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Ambiente ----
    env = SubprocVecEnv([make_env])
    action_low  = np.array([0.0, -1.0], dtype=np.float32)
    action_high = np.array([1.0,  1.0], dtype=np.float32)

    # ---- BEV generator (modo treino) ----
    bev_gen = get_bev_generator(BEV_ARC, device=DEVICE, model_path=BEV_GEN_CKPT)
    bev_gen.set_train()
    bev_optimizer = th.optim.Adam(bev_gen.parameters(), lr=LR_BEV)

    # ---- Policy (congelada — apenas para dirigir) ----
    observation_space = gym.spaces.Dict(
        birdview=gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8),
        state=gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32),
    )
    action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

    policy = AgentPolicy(
        observation_space=observation_space,
        action_space=action_space,
        policy_head_arch=[256, 256],
        features_extractor_entry_point='torch_layers:XtMaCNN',
        features_extractor_kwargs={'states_neurons': [256, 256]},
        distribution_entry_point='distributions:BetaDistribution',
        real_bev=True,
    )
    policy.to(DEVICE)
    policy.eval()

    # ---- Carrega checkpoints ----
    start_step = 0

    if RESUME_BEV_CKPT is not None:
        print(f"Retomando BEV gen de {RESUME_BEV_CKPT}")
        b_ckpt = th.load(RESUME_BEV_CKPT, map_location=DEVICE)
        bev_gen.generator.load_state_dict(b_ckpt['bev_generator_state_dict'])
        start_step = b_ckpt.get('total_steps', 0)
        print(f"  continuando do passo {start_step}")
    elif BEV_GEN_CKPT is None:
        print(f"BEV gen: pesos padrão ({ARC_BEV_DEFAULT_CKPT[BEV_ARC]})")

    if POLICY_PRETRAINED_CKPT is not None:
        print(f"Policy (congelada): {POLICY_PRETRAINED_CKPT}")
        ckpt = th.load(POLICY_PRETRAINED_CKPT, map_location=DEVICE)
        policy.load_state_dict(ckpt['policy_state_dict'])

    # JSON de rastreabilidade
    _bev_src = RESUME_BEV_CKPT or BEV_GEN_CKPT or ARC_BEV_DEFAULT_CKPT[BEV_ARC]
    with open(bev_ckpt_dir / 'info.json', 'w') as f:
        json.dump({
            'pretrained_bev':    _bev_src,
            'frozen_policy':     POLICY_PRETRAINED_CKPT or 'random_init',
            'arc':               BEV_ARC,
            'wandb_run':         run.name,
            'action_noise_std':  ACTION_NOISE_STD,
            'kde_weight_clip':   KDE_WEIGHT_MAX_CLIP,
        }, f, indent=2)

    # ---- KDE (fit único sobre dados expert) ----
    kde = load_kde(DATASET_DIR, routes=TRAIN_ROUTES, n_eps=N_EPS)

    # ---- Loop de treino ----
    obs = env.reset()
    total_steps = start_step

    print(f"Iniciando BEV fine-tuning KDE-online | arc={BEV_ARC} | total_steps={TOTAL_STEPS}")
    print(f"  GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS} | LR_BEV={LR_BEV} | ACTION_NOISE_STD={ACTION_NOISE_STD}")

    n_updates = (TOTAL_STEPS - start_step) // GRAD_ACCUM_STEPS
    pbar = tqdm.tqdm(total=n_updates, desc='bev-finetune', unit='upd', dynamic_ncols=True)

    while total_steps < TOTAL_STEPS:

        # ================================================================
        # Fase 1 — Coleta (sem grafo, armazena na CPU)
        # ================================================================
        rollout_image_inputs = []  # dict de tensors na CPU — re-enviado ao GPU na fase 3
        rollout_bev_targets  = []  # Tensor(1,3,192,192) normalizado [0,1] na CPU
        rollout_actions      = []  # ndarray(1,2) — ação ruidosa executada no ambiente

        with th.no_grad():
            for _ in range(GRAD_ACCUM_STEPS):
                image_input = build_image_input(obs, BEV_ARC, DEVICE)

                # Policy usa o BEV real do ambiente (para o qual foi treinada).
                # O BEV generator é treinado separadamente na Fase 3 — não precisa
                # do seu output aqui.
                actions, *_ = policy.forward(obs, deterministic=False, clip_action=True)

                # Ruído para forçar exploração de posições não-usuais
                noise = np.random.normal(0.0, ACTION_NOISE_STD, actions.shape).astype(np.float32)
                actions_noisy = np.clip(actions + noise, action_low, action_high)

                # gt_bev capturado antes do step (corresponde à obs atual)
                bev_target = get_bev_target(obs, device='cpu')

                obs, _, done, _ = env.step(actions_noisy)

                rollout_image_inputs.append(to_cpu(image_input))
                rollout_bev_targets.append(bev_target)
                rollout_actions.append(actions_noisy)

        # ================================================================
        # Fase 2 — KDE weighting (fora do grafo, em numpy)
        # ================================================================
        all_actions = th.as_tensor(
            np.concatenate(rollout_actions, axis=0), dtype=th.float32
        )  # (GRAD_ACCUM_STEPS, 2)
        kde_weights = compute_kde_weights(kde, all_actions)  # (GRAD_ACCUM_STEPS,)
        kde_weights.clamp_(max=KDE_WEIGHT_MAX_CLIP)
        # Re-normaliza após o clamp para manter a escala do loss estável
        kde_weights = kde_weights / (kde_weights.mean() + 1e-6)

        # ================================================================
        # Fase 3 — Update do BEV generator (com grafo, na GPU)
        # ================================================================
        bev_optimizer.zero_grad()

        weighted_losses = []
        raw_losses      = []

        for i in range(GRAD_ACCUM_STEPS):
            image_input_gpu = to_device(rollout_image_inputs[i], DEVICE)
            bev_target_gpu  = rollout_bev_targets[i].to(DEVICE)

            pred = bev_gen.forward_train(image_input_gpu)

            if pred.shape[-2:] != bev_target_gpu.shape[-2:]:
                bev_target_gpu = F.interpolate(bev_target_gpu, size=pred.shape[-2:], mode='nearest')

            raw_loss = bev_gen.compute_loss(pred, bev_target_gpu)
            raw_losses.append(raw_loss.detach())
            weighted_losses.append(kde_weights[i] * raw_loss)

        bev_loss_weighted = th.stack(weighted_losses).mean()
        bev_loss_weighted.backward()
        th.nn.utils.clip_grad_norm_(bev_gen.parameters(), GRAD_CLIP)
        bev_optimizer.step()

        total_steps += GRAD_ACCUM_STEPS

        # ================================================================
        # Log
        # ================================================================
        bev_loss_raw_val      = th.stack(raw_losses).mean().item()
        bev_loss_weighted_val = bev_loss_weighted.item()
        kde_w_std_val         = float(kde_weights.std().item())
        kde_w_max_val         = float(kde_weights.max().item())

        wandb.log({
            'bev_loss_weighted': bev_loss_weighted_val,
            'bev_loss_raw':      bev_loss_raw_val,
            'kde_weight_std':    kde_w_std_val,
            'kde_weight_max':    kde_w_max_val,
        }, step=total_steps)

        pbar.update(1)
        pbar.set_postfix(
            step=total_steps,
            bev_w=f'{bev_loss_weighted_val:.4f}',
            bev_r=f'{bev_loss_raw_val:.4f}',
            kde_std=f'{kde_w_std_val:.2f}',
            kde_max=f'{kde_w_max_val:.2f}',
        )

        # ================================================================
        # Checkpoint
        # ================================================================
        if total_steps % CKPT_EVERY < GRAD_ACCUM_STEPS:
            bev_ckpt = {
                'bev_generator_state_dict': bev_gen.generator.state_dict(),
                'total_steps': total_steps,
            }
            th.save(bev_ckpt, bev_ckpt_dir / f'bev_{total_steps}.pth')
            th.save(bev_ckpt, bev_ckpt_dir / 'bev_latest.pth')
            tqdm.tqdm.write(f"[ckpt {total_steps}] bev → {bev_ckpt_dir / 'bev_latest.pth'}")

    pbar.close()
    run.finish()
    env.close()


if __name__ == '__main__':
    train()
