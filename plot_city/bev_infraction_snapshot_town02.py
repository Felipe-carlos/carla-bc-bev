"""
plot_city/bev_infraction_snapshot_town02.py

Single-pass BEV comparison at a chosen intersection of Town02.

Drives the expert (real-BEV) policy for `episode_step` steps in the target
episode, then feeds the captured observation to every BEV generator.

Notable scenarios
-----------------
ep0 (black circle) — western intersection (spawn x=104.6, y=187.6, yaw=-180°):
    kde and cvt_6ch go straight (west) instead of turning south.
    Failures at step 164 (x=42, y=188).  Capture default: step 148.

ep1 (black square) — eastern intersection (spawn x=-7.4, y=288.2, yaw=90°):
    kde and cvt_6ch go straight (north) instead of turning northwest.
    Failures at step 169/171 (x=46, y=306).  Capture at step 155.

Usage (from anywhere):
    python plot_city/bev_infraction_snapshot_town02.py
    python plot_city/bev_infraction_snapshot_town02.py --target-episode 0 --episode-step 148
    python plot_city/bev_infraction_snapshot_town02.py --target-episode 1 --episode-step 155
    python plot_city/bev_infraction_snapshot_town02.py --output-dir /custom/path
"""

import argparse
import json
import math
import queue
import sys
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch as th
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent_policy import AgentPolicy
from carla_gym.envs import EndlessEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from data_collect import reward_configs, terminal_configs
from config.obs_config import get_obs_configs
from eval_agent import create_image_tensor, prepare_image

from bev_generation.cvt_3ch import CVT_3chL1Generator
from bev_generation.unet import Unet_BEVGenerator
from bev_generation.cvt_6ch import CVT_6chVanilla
from bev_generation.cvt_6ch_vanilla_no_noise import CVT_6chVanillaNoNoise
from bev_generation.cvt_6ch_kde import CVT_6chKDE

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_DIR     = _REPO_ROOT / 'plot_city/bev_snapshots/town02_east'
CARLA_CFG_PATH = _REPO_ROOT / 'config/carla_config.json'
DEVICE         = 'cuda'

DRIVER_CKPT = _REPO_ROOT / 'ckpts/ckpt-real-bev/ckpt_latest.pth'

# ep1 (black square): eastern intersection — kde+cvt_6ch fail at step 169/171
DEFAULT_TARGET_EPISODE = 1
DEFAULT_EPISODE_STEP   = 155

ENV_CONFIG = {
    'carla_map': 'Town02',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers':  [0, 300],
    'weather_group':       'ClearNoon',
}

# (spawn_x, spawn_y, spawn_yaw)  — yaw in CARLA convention (0°=+x, 90°=+y)
_EP_SPAWNS = {
    0: ( 104.600, 187.600, -180.0),  # western intersection: kde+cvt_6ch go straight west
    1: (  -7.380, 288.223,   90.0),  # eastern intersection: kde+cvt_6ch go straight north
    2: ( 178.290, 306.503,    0.0),
    3: (   1.489, 105.992,  165.0),
    4: ( 193.736, 121.213,  -90.0),
}

GENERATORS = {
    'cvt_3ch':                  CVT_3chL1Generator,
    'unet':                     Unet_BEVGenerator,
    'cvt_6ch_vanilla':          CVT_6chVanilla,
    'cvt_6ch_vanilla_no_noise': CVT_6chVanillaNoNoise,
    'cvt_6ch_kde':              CVT_6chKDE,
}

GENERATOR_LABELS = {
    'cvt_3ch':                  'CVT 3ch L1',
    'unet':                     'UNet',
    'cvt_6ch_vanilla':          'CVT 6ch Vanilla',
    'cvt_6ch_vanilla_no_noise': 'CVT 6ch No Noise',
    'cvt_6ch_kde':              'CVT 6ch KDE',
    'real_bev':                 'Real BEV (GT)',
}
# ─────────────────────────────────────────────────────────────────────────────


# ── BEV inference helpers ─────────────────────────────────────────────────────

def _to_numpy(x) -> np.ndarray:
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _build_input(gen_name: str, obs: dict) -> dict:
    ext = th.as_tensor(obs['extrinsics'], dtype=th.float32).to(DEVICE)
    itr = th.as_tensor(obs['intrinsics'], dtype=th.float32).to(DEVICE)
    if gen_name == 'unet':
        return {'image': create_image_tensor(obs, unet=True).to(DEVICE)}
    if gen_name == 'cvt_3ch':
        return {
            'image':      create_image_tensor(obs, unet=False, w_resize=256, h_resize=256).to(DEVICE),
            'extrinsics': ext,
            'intrinsics': itr,
        }
    return {
        'image':      create_image_tensor(obs, unet=False, w_resize=480, h_resize=224).to(DEVICE),
        'extrinsics': ext,
        'intrinsics': itr,
    }


def run_generator(gen, gen_name: str, obs: dict) -> np.ndarray:
    inp = _build_input(gen_name, obs)
    bev = gen.infer(inp)
    return _to_numpy(bev)


# ── Colorization & IoU ────────────────────────────────────────────────────────

def colorize_bev(arr: np.ndarray) -> np.ndarray:
    """ch0→vermelho, ch2→verde, ch1→azul (last wins on overlap)."""
    out = np.zeros_like(arr)
    out[arr[:, :, 0] > 0] = [255,   0,   0]
    out[arr[:, :, 2] > 0] = [  0, 255,   0]
    out[arr[:, :, 1] > 0] = [  0,   0, 255]
    return out


def compute_iou(pred_np: np.ndarray, real_bev_np: np.ndarray) -> dict:
    p = th.as_tensor(pred_np[:, :3],     dtype=th.float32) > 127
    g = th.as_tensor(real_bev_np[:, :3], dtype=th.float32) > 127
    p, g = p.float(), g.float()
    inter = (p * g).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + g.sum(dim=(2, 3)) - inter
    iou   = (inter / (union + 1e-7))[0].tolist()
    return {'ch0': iou[0], 'ch1': iou[1], 'ch2': iou[2], 'mean': sum(iou) / 3}


# ── Image saving ──────────────────────────────────────────────────────────────

def _write_rgb(path: Path, img_hwc_rgb: np.ndarray):
    cv2.imwrite(str(path), cv2.cvtColor(img_hwc_rgb, cv2.COLOR_RGB2BGR))


def save_bev(gen_name: str, bev_np: np.ndarray, real_bev_np: np.ndarray,
             out_dir: Path) -> dict:
    d = out_dir / gen_name
    d.mkdir(parents=True, exist_ok=True)
    arr = bev_np[0].transpose(1, 2, 0).astype(np.uint8)
    _write_rgb(d / 'bev_generated_rgb.png', colorize_bev(arr))
    for i in range(3):
        cv2.imwrite(str(d / f'bev_generated_ch{i}.png'), arr[:, :, i])
    iou = compute_iou(bev_np, real_bev_np)
    with open(d / 'iou.json', 'w') as f:
        json.dump(iou, f, indent=2)
    return iou


def save_real_bev(real_bev_np: np.ndarray, out_dir: Path):
    d = out_dir / 'real_bev'
    d.mkdir(parents=True, exist_ok=True)
    arr = real_bev_np[0, :3].transpose(1, 2, 0).astype(np.uint8)
    _write_rgb(d / 'bev_real_rgb.png', colorize_bev(arr))
    for i in range(3):
        cv2.imwrite(str(d / f'bev_real_ch{i}.png'), arr[:, :, i])


def save_cameras(obs: dict, out_dir: Path):
    for key in ['left_rgb', 'central_rgb', 'right_rgb', 'rear_rgb']:
        img = prepare_image(obs.get(key))
        if img is not None:
            _write_rgb(out_dir / f'cam_{key}.png', img)


# ── Panel ─────────────────────────────────────────────────────────────────────

def build_panel(ious: dict, out_dir: Path):
    SZ  = 192
    BAR = 36
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = 0.42
    gen_order = list(GENERATORS.keys()) + ['real_bev']

    cols = []
    for name in gen_order:
        if name == 'real_bev':
            gen_path  = out_dir / 'real_bev' / 'bev_real_rgb.png'
            real_path = gen_path
        else:
            gen_path  = out_dir / name / 'bev_generated_rgb.png'
            real_path = out_dir / 'real_bev' / 'bev_real_rgb.png'

        if not gen_path.exists():
            continue

        gen_bgr  = cv2.imread(str(gen_path))
        real_bgr = cv2.imread(str(real_path))
        label    = GENERATOR_LABELS.get(name, name)
        iou      = ious.get(name, {})
        iou_str  = f"IoU {iou.get('mean', 1.0):.3f}" if name != 'real_bev' else 'GT Reference'

        col_h = BAR + SZ + BAR + SZ + BAR
        col   = np.zeros((col_h, SZ, 3), dtype=np.uint8)

        cv2.rectangle(col, (0, 0), (SZ, BAR), (50, 50, 50), -1)
        cv2.putText(col, label, (4, BAR - 10), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

        col[BAR:BAR + SZ, :] = real_bgr

        y1 = BAR + SZ
        cv2.rectangle(col, (0, y1), (SZ, y1 + BAR), (25, 25, 25), -1)
        cv2.putText(col, 'Real (top) / Gen (bottom)',
                    (4, y1 + BAR - 10), font, fs * 0.8, (160, 160, 160), 1, cv2.LINE_AA)

        y2 = BAR + SZ + BAR
        col[y2:y2 + SZ, :] = gen_bgr

        y3 = BAR + SZ + BAR + SZ
        cv2.rectangle(col, (0, y3), (SZ, y3 + BAR), (15, 15, 15), -1)
        cv2.putText(col, iou_str, (4, y3 + BAR - 10), font, fs, (200, 220, 255), 1, cv2.LINE_AA)

        cols.append(col)

    if cols:
        panel = np.hstack(cols)
        path  = out_dir / 'panel_comparison.png'
        cv2.imwrite(str(path), panel)
        print(f'\nPanel: {path}  ({panel.shape[1]}x{panel.shape[0]})')


# ── Map position capture ──────────────────────────────────────────────────────

def save_position_on_map(pos: dict, out_dir: Path,
                         radius_m: float = 300.0, img_size: int = 1024):
    import carla

    x, y = pos.get('x', 0.0), pos.get('y', 0.0)

    with open(out_dir / 'capture_info.json', 'w') as f:
        json.dump(pos, f, indent=2)
    print(f'Capture position  x={x:.2f}  y={y:.2f}  yaw={pos.get("yaw", 0.0):.1f}°'
          f'  step={pos.get("step")}')

    carla_cfg = json.load(open(CARLA_CFG_PATH))
    client = carla.Client(carla_cfg['host'], carla_cfg['port'])
    client.set_timeout(60.0)

    world = client.load_world('Town02')
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    fov = 60.0
    z   = radius_m / math.tan(math.radians(fov / 2))

    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(img_size))
    bp.set_attribute('image_size_y', str(img_size))
    bp.set_attribute('fov', str(fov))

    transform = carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=-90.0, yaw=0.0),
    )

    q = queue.Queue()
    cam = world.spawn_actor(bp, transform)
    cam.listen(q.put)
    world.tick()

    raw = None
    while raw is None or q.qsize() > 0:
        raw = q.get()

    arr = np.frombuffer(raw.raw_data, dtype=np.uint8).copy()
    arr = arr.reshape((raw.height, raw.width, 4))[:, :, :3].copy()

    cam.stop()
    cam.destroy()

    cx, cy = img_size // 2, img_size // 2
    r      = 22
    color  = (0, 0, 255)
    cv2.circle(arr, (cx, cy), r, color, 3)
    cv2.line(arr, (cx - r - 12, cy), (cx + r + 12, cy), color, 2)
    cv2.line(arr, (cx, cy - r - 12), (cx, cy + r + 12), color, 2)

    label = f'x={x:.1f}  y={y:.1f}  step={pos.get("step")}'
    cv2.putText(arr, label, (8, img_size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    path = out_dir / 'capture_location.png'
    cv2.imwrite(str(path), arr)
    print(f'Map crop saved:   {path}')


# ── Driver ────────────────────────────────────────────────────────────────────

def drive_to_capture(target_episode: int = DEFAULT_TARGET_EPISODE,
                     episode_step: int = DEFAULT_EPISODE_STEP) -> Optional[Tuple[dict, dict]]:
    """
    Drive the expert policy for `episode_step` steps in `target_episode` of
    Town02 and return (obs_snapshot, position_info).
    """
    if target_episode not in _EP_SPAWNS:
        raise ValueError(f'Unknown episode {target_episode}. Add its spawn to _EP_SPAWNS.')
    spawn_x, spawn_y, spawn_yaw = _EP_SPAWNS[target_episode]

    carla_cfg = json.load(open(CARLA_CFG_PATH))

    obs_space = gym.spaces.Dict(**{
        'birdview': gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8),
        'state':    gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32),
    })
    act_space = gym.spaces.Box(
        low=np.array([0., -1.]), high=np.array([1., 1.]), dtype=np.float32)

    policy = AgentPolicy(
        observation_space=obs_space,
        action_space=act_space,
        policy_head_arch=[256, 256],
        features_extractor_entry_point='torch_layers:XtMaCNN',
        features_extractor_kwargs={'states_neurons': [256, 256]},
        distribution_entry_point='distributions:BetaDistribution',
        real_bev=True,
    )
    saved = th.load(str(DRIVER_CKPT), map_location=DEVICE)
    policy.load_state_dict(saved['policy_state_dict'])
    policy = policy.eval().to(DEVICE)

    obs_cfgs = get_obs_configs('cvt_6ch')

    def make_env():
        import carla as _carla
        raw_env = EndlessEnv(
            obs_configs=obs_cfgs,
            reward_configs=reward_configs,
            terminal_configs=terminal_configs,
            host=carla_cfg['host'],
            port=carla_cfg['port'],
            seed=2021,
            no_rendering=False,
            **ENV_CONFIG,
        )
        spawn_t = _carla.Transform(
            _carla.Location(x=spawn_x, y=spawn_y, z=1.0),
            _carla.Rotation(yaw=spawn_yaw),
        )
        raw_env._task['ego_vehicles']['endless']['hero'] = False
        raw_env._task['ego_vehicles']['routes']['hero']  = [spawn_t]
        raw_env._shuffle_task = False
        return RlBirdviewWrapper(raw_env, input_states=['rgb', 'traj', 'state', 'matrices'],
                                 acc_as_action=True)

    env = SubprocVecEnv([make_env])
    obs = env.reset()
    last_info = None

    print(f'Driving {episode_step} steps from ep{target_episode} spawn '
          f'(x={spawn_x:.1f}, y={spawn_y:.1f}, yaw={spawn_yaw:.0f}°)...')

    for n_step in range(episode_step + 1):
        if n_step == episode_step:
            snapshot = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                        for k, v in obs.items()}
            env.close()
            pos = {}
            if last_info is not None:
                pos = {
                    'x':       float(last_info[0].get('ev_x', float('nan'))),
                    'y':       float(last_info[0].get('ev_y', float('nan'))),
                    'yaw':     float(last_info[0].get('ev_yaw', float('nan'))),
                    'episode': target_episode,
                    'step':    n_step,
                }
            return snapshot, pos

        actions, *_ = policy.forward(obs, deterministic=True, clip_action=True)
        obs, _, done, last_info = env.step(actions)

        if done[0]:
            print(f'Episode ended at step {n_step} before capture at {episode_step}.')
            env.close()
            return None

    env.close()
    return None


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-episode', type=int, default=DEFAULT_TARGET_EPISODE,
                        help='Which EndlessEnv episode to capture in (default: %(default)s)')
    parser.add_argument('--episode-step', type=int, default=DEFAULT_EPISODE_STEP,
                        help='Step within episode to capture (default: %(default)s)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    result = drive_to_capture(args.target_episode, args.episode_step)
    if result is None:
        sys.exit('Capture failed.')
    obs, pos = result

    real_bev_np = _to_numpy(obs['birdview'])

    save_cameras(obs, out_dir)
    save_real_bev(real_bev_np, out_dir)
    print('GT BEV saved.')

    ious = {}
    for gen_name, GenClass in GENERATORS.items():
        print(f'  [{gen_name}] loading generator...')
        gen    = GenClass(device=DEVICE)
        bev_np = run_generator(gen, gen_name, obs)
        iou    = save_bev(gen_name, bev_np, real_bev_np, out_dir)
        ious[gen_name] = iou
        print(f'    IoU  ch0={iou["ch0"]:.3f}  ch1={iou["ch1"]:.3f}  '
              f'ch2={iou["ch2"]:.3f}  mean={iou["mean"]:.3f}')
        del gen

    build_panel(ious, out_dir)

    if pos:
        save_position_on_map(pos, out_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
