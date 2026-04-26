"""
Fine-tuning online com KDE para reponderar ações raras.

Ambos os componentes (BEV generator + Policy) são atualizados exclusivamente
a partir da interação com a simulação EndlessEnv — sem dataset expert no loop.

Lógica de treino por iteração (GRAD_ACCUM_STEPS passos coletados antes de atualizar):

  BEV generator:
    obs_rgb → forward_train() → compara com BEV real do CARLA
    loss_bev = kde_weight * bev_loss   (kde_weight baseado na ação tomada)

  Policy (REINFORCE):
    a = policy(obs_com_bev_gerada) → env.step(a) → reward
    log_prob de `a` recomputado com grad via evaluate_actions()
    loss_policy = -mean(kde_weight * G_t * log_prob)   (G_t = retorno normalizado)

Ambos usam GRAD_ACCUM_STEPS como janela de rollout antes de cada update.

Para fine-tuning: defina POLICY_PRETRAINED_CKPT e/ou RESUME_FROM_CKPT no topo.
O BEV generator já carrega seus pesos pré-treinados em __init__ por padrão.
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

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

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

# Tamanho da janela de rollout: quantos passos acumular antes de cada update.
# Serve como gradient accumulation para AMBOS os modelos.
GRAD_ACCUM_STEPS = 8

LR_BEV = 1e-4
LR_POLICY = 1e-5
ENT_WEIGHT = 0.01
GRAD_CLIP = 1.0

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

# ---- Checkpoints de origem (fine-tuning) ----
# Policy pré-treinada (ex: saída do learn_bc.py). None → pesos aleatórios.
POLICY_PRETRAINED_CKPT: Optional[str] = 'ckpts/ckpt-cvt_3ch_L1/bc_ckpt_63_min_eval.pth'

# BEV generator: None → usa pesos padrão de ARC_BEV_DEFAULT_CKPT (carregados em __init__).
BEV_GEN_CKPT: Optional[str] = None

# ---- Retomar treino anterior deste script ----
# Aponte para os .pth salvos por uma run anterior; ignora POLICY/BEV_GEN_CKPT acima.
RESUME_POLICY_CKPT: Optional[str] = None  # ex: 'ckpts/finetune-kde/<run>/policy_latest.pth'
RESUME_BEV_CKPT:    Optional[str] = None  # ex: 'bev_generation/cvt_3ch/ckpts/finetune-kde/<run>/bev_latest.pth'

# Canais do obs['birdview'] usados como alvo para o BEV generator.
# O chauffeurnet retorna múltiplos canais; os 3 primeiros correspondem
# às camadas semânticas que os generators aprendem a predizer.
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


def binarize_for_policy(bev_raw: th.Tensor, arc: str) -> th.Tensor:
    """Converte saída raw do generator para formato {0, 255} que a policy espera."""
    threshold = 0.0 if arc == 'unet' else 0.5  # logits vs sigmoid
    return (bev_raw.detach() > threshold).byte() * 255


def get_bev_target(obs: dict, device: str) -> th.Tensor:
    """
    Extrai o BEV ground-truth do obs do VecEnv e normaliza para [0, 1].
    Usa os BEV_TARGET_CH primeiros canais do chauffeurnet birdview.
    """
    real_bev = obs.get('birdview', obs.get('topdown'))
    target = th.as_tensor(real_bev, dtype=th.float32, device=device)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    return (target[:, :BEV_TARGET_CH, :, :] / 255.0)


# ============================================================
# Treino principal
# ============================================================

def train():
    api_key = os.getenv('WANDB_API_KEY')
   
    run = wandb.init(project=WANDB_PROJECT, reinit=True)

    # Diretórios de checkpoint separados por componente
    policy_ckpt_dir = Path('ckpts') / 'finetune-kde' / run.name
    bev_ckpt_dir    = Path(ARC_BEV_FOLDER[BEV_ARC]) / 'ckpts' / 'finetune-kde' / run.name
    policy_ckpt_dir.mkdir(parents=True, exist_ok=True)
    bev_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Ambiente ----
    env = SubprocVecEnv([make_env])

    # ---- BEV generator (modo treino) ----
    bev_gen = get_bev_generator(BEV_ARC, device=DEVICE, model_path=BEV_GEN_CKPT)
    bev_gen.set_train()
    bev_optimizer = th.optim.Adam(bev_gen.parameters(), lr=LR_BEV)

    # ---- Policy ----
    observation_space = gym.spaces.Dict(
        birdview=gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8),
        state=gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32),
    )
    action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)

    policy = AgentPolicy(
        observation_space=observation_space,
        action_space=action_space,
        policy_head_arch=[256, 256],
        features_extractor_entry_point='torch_layers:XtMaCNN',
        features_extractor_kwargs={'states_neurons': [256, 256]},
        distribution_entry_point='distributions:BetaDistribution',
    )
    policy.to(DEVICE)
    policy_optimizer = th.optim.Adam(policy.parameters(), lr=LR_POLICY)

    # ---- Carrega checkpoints ----
    start_step = 0

    if RESUME_POLICY_CKPT is not None and RESUME_BEV_CKPT is not None:
        print(f"Retomando policy de {RESUME_POLICY_CKPT}")
        p_ckpt = th.load(RESUME_POLICY_CKPT, map_location=DEVICE)
        policy.load_state_dict(p_ckpt['policy_state_dict'])
        start_step = p_ckpt.get('total_steps', 0)

        print(f"Retomando BEV gen de {RESUME_BEV_CKPT}")
        b_ckpt = th.load(RESUME_BEV_CKPT, map_location=DEVICE)
        bev_gen.generator.load_state_dict(b_ckpt['bev_generator_state_dict'])
        print(f"  continuando do passo {start_step}")
    elif POLICY_PRETRAINED_CKPT is not None:
        print(f"Carregando policy pré-treinada de {POLICY_PRETRAINED_CKPT}")
        ckpt = th.load(POLICY_PRETRAINED_CKPT, map_location=DEVICE)
        policy.load_state_dict(ckpt['policy_state_dict'])

    # JSON de rastreabilidade: registra de onde os pesos vieram
    _policy_src = (RESUME_POLICY_CKPT or POLICY_PRETRAINED_CKPT or 'random_init')
    _bev_src    = (RESUME_BEV_CKPT    or BEV_GEN_CKPT            or ARC_BEV_DEFAULT_CKPT[BEV_ARC])
    with open(policy_ckpt_dir / 'info.json', 'w') as f:
        json.dump({'pretrained_policy': _policy_src, 'arc': BEV_ARC, 'wandb_run': run.name}, f, indent=2)
    with open(bev_ckpt_dir / 'info.json', 'w') as f:
        json.dump({'pretrained_bev': _bev_src, 'arc': BEV_ARC, 'wandb_run': run.name}, f, indent=2)

    # ---- KDE (fit único sobre dados expert) ----
    kde = load_kde(DATASET_DIR, routes=TRAIN_ROUTES, n_eps=N_EPS)

    # ---- Loop de treino ----
    obs = env.reset()
    total_steps = start_step

    print(f"Iniciando fine-tuning KDE-online | arc={BEV_ARC} | total_steps={TOTAL_STEPS}")
    print(f"  GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS} | LR_BEV={LR_BEV} | LR_POLICY={LR_POLICY}")

    n_updates = (TOTAL_STEPS - start_step) // GRAD_ACCUM_STEPS
    pbar = tqdm.tqdm(total=n_updates, desc='fine-tuning', unit='upd', dynamic_ncols=True)

    while total_steps < TOTAL_STEPS:

        # ================================================================
        # Coleta de rollout (GRAD_ACCUM_STEPS passos)
        # ================================================================
        rollout_bevs_gen    = []   # Tensor(1,C,H,W) com grad — para loss do BEV gen
        rollout_bev_targets = []   # Tensor(1,3,H,W) normalizado [0,1]
        rollout_bevs_bin    = []   # Tensor(1,3,H,W) float {0,255} detached — input da policy
        rollout_obs_states  = []   # Tensor(1,6) — estado para evaluate_actions
        rollout_actions     = []   # ndarray(1,2) — ação tomada (para log_prob com grad)
        rollout_rewards     = []   # float — reward da simulação

        for _ in range(GRAD_ACCUM_STEPS):
            # --- BEV generator forward COM gradiente ---
            image_input = build_image_input(obs, BEV_ARC, DEVICE)
            generated_bev = bev_gen.forward_train(image_input)   # raw output, com grad
            bev_target = get_bev_target(obs, DEVICE)

            if generated_bev.shape[-2:] != bev_target.shape[-2:]:
                bev_target = F.interpolate(bev_target, size=generated_bev.shape[-2:], mode='nearest')

            # --- Ação da policy (internamente usa no_grad) ---
            bev_bin = binarize_for_policy(generated_bev, BEV_ARC)  # detached byte {0,255}
            obs_for_policy = dict(obs)
            obs_for_policy['birdview'] = bev_bin
            actions, _, mu, sigma, _ = policy.forward(obs_for_policy, deterministic=False, clip_action=True)

            # --- Estado do obs antes do step (para evaluate_actions depois) ---
            obs_state = th.as_tensor(obs['state'], dtype=th.float32, device=DEVICE)

            # --- Passo no ambiente ---
            obs, reward, done, info = env.step(actions)

            # Armazena
            rollout_bevs_gen.append(generated_bev)
            rollout_bev_targets.append(bev_target)
            rollout_bevs_bin.append(bev_bin.float())
            rollout_obs_states.append(obs_state)
            rollout_actions.append(actions)
            rollout_rewards.append(float(np.mean(reward)))

        # Pesos KDE calculados para o rollout inteiro de uma vez.
        # Com num_envs=1, cada passo produz 1 ação — se normalizarmos passo a passo
        # o resultado é sempre 1.0 (único valor dividido por si mesmo).
        # Calculando em batch (GRAD_ACCUM_STEPS ações), a normalização compara
        # ações raras vs. comuns dentro da janela e produz pesos distintos.
        all_actions = th.as_tensor(
            np.concatenate(rollout_actions, axis=0), dtype=th.float32, device=DEVICE
        )  # (GRAD_ACCUM_STEPS, action_dim)
        rollout_kde_weights = compute_kde_weights(kde, all_actions)  # (GRAD_ACCUM_STEPS,)

        # ================================================================
        # Update do BEV generator
        # ================================================================
        bev_optimizer.zero_grad()

        bev_losses = []
        for i in range(GRAD_ACCUM_STEPS):
            raw_loss = bev_gen.compute_loss(rollout_bevs_gen[i], rollout_bev_targets[i])
            bev_losses.append(rollout_kde_weights[i] * raw_loss)

        bev_total_loss = th.stack(bev_losses).mean()
        bev_total_loss.backward()
        th.nn.utils.clip_grad_norm_(bev_gen.parameters(), GRAD_CLIP)
        bev_optimizer.step()

        # ================================================================
        # Update da policy (REINFORCE com retornos normalizados)
        # ================================================================
        rewards_arr = np.array(rollout_rewards, dtype=np.float32)
        std = rewards_arr.std()
        returns_arr = (rewards_arr - rewards_arr.mean()) / (std + 1e-8)

        policy_optimizer.zero_grad()

        policy_step_losses = []
        entropy_step_losses = []
        for i in range(GRAD_ACCUM_STEPS):
            obs_tensor = {
                'state': rollout_obs_states[i],
                'birdview': rollout_bevs_bin[i],
            }
            action_tensor = th.as_tensor(rollout_actions[i], dtype=th.float32, device=DEVICE)

            # log_prob COM gradiente pela policy (evaluate_actions não usa no_grad)
            log_probs, entropy_loss = policy.evaluate_actions(obs_tensor, action_tensor)

            ret   = th.as_tensor(returns_arr[i], dtype=th.float32, device=DEVICE)
            kde_w = rollout_kde_weights[i]   # escalar — peso do passo i dentro do rollout

            # REINFORCE: maximiza retorno KDE-ponderado
            step_loss = -(kde_w * ret * log_probs).mean()
            policy_step_losses.append(step_loss)
            entropy_step_losses.append(entropy_loss)
        
        policy_total_loss = (
            th.stack(policy_step_losses).mean()
            + ENT_WEIGHT * th.stack(entropy_step_losses).mean()
        )
        policy_total_loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        policy_optimizer.step()

        total_steps += GRAD_ACCUM_STEPS

        # ================================================================
        # Log
        # ================================================================
        mean_reward   = float(rewards_arr.mean())
        bev_loss_val  = bev_total_loss.item()
        pol_loss_val  = policy_total_loss.item()
        kde_w_std_val = float(rollout_kde_weights.std().item())   # std=0 → ações uniformes
        kde_w_max_val = float(rollout_kde_weights.max().item())   # ação mais rara do rollout

        wandb.log({
            'bev_loss':      bev_loss_val,
            'policy_loss':   pol_loss_val,
            'mean_reward':   mean_reward,
            'kde_weight_std': kde_w_std_val,
            'kde_weight_max': kde_w_max_val,
        }, step=total_steps)

        pbar.update(1)
        pbar.set_postfix(
            step=total_steps,
            bev=f'{bev_loss_val:.4f}',
            pol=f'{pol_loss_val:.4f}',
            rew=f'{mean_reward:.3f}',
            kde_std=f'{kde_w_std_val:.2f}',
            kde_max=f'{kde_w_max_val:.2f}',
        )

        # ================================================================
        # Checkpoint
        # ================================================================
        if total_steps % CKPT_EVERY < GRAD_ACCUM_STEPS:
            policy_ckpt = {'policy_state_dict': policy.state_dict(), 'total_steps': total_steps}
            bev_ckpt    = {'bev_generator_state_dict': bev_gen.generator.state_dict(), 'total_steps': total_steps}

            th.save(policy_ckpt, policy_ckpt_dir / f'policy_{total_steps}.pth')
            th.save(policy_ckpt, policy_ckpt_dir / 'policy_latest.pth')

            th.save(bev_ckpt, bev_ckpt_dir / f'bev_{total_steps}.pth')
            th.save(bev_ckpt, bev_ckpt_dir / 'bev_latest.pth')

            tqdm.tqdm.write(f"[ckpt {total_steps}] policy → {policy_ckpt_dir / 'policy_latest.pth'}")
            tqdm.tqdm.write(f"[ckpt {total_steps}] bev    → {bev_ckpt_dir / 'bev_latest.pth'}")

    pbar.close()
    run.finish()
    env.close()


if __name__ == '__main__':
    train()
