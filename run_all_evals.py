import numpy as np
import torch as th
import gym
import json
import traceback
from pathlib import Path

from agent_policy import AgentPolicy
from carla_gym.envs import EndlessEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from data_collect import reward_configs, terminal_configs
from eval_agent import evaluate_policy
from eval_agent_infraction import (
    evaluate_policy_until_infraction,
    evaluate_policy_N_spawns,
    select_spread_spawns,
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from config.obs_config import get_obs_configs


TOWNS = ['Town01', 'Town02'] #town02 já foi avaliado — focar no town01 para análises mais detalhadas
SAVE_BEV_VIDEO = False
SAVE_TRAJ = True
DEVICE = 'cuda'

# ── Eval mode ────────────────────────────────────────────────────────────────
# N_SPAWN_EPISODES > 0  → evaluate_policy_N_spawns  (N episodes, each from a
#                         different spread spawn, stops at infraction or
#                         MAX_EPISODE_STEPS; all agents share the same spawns)
# N_SPAWN_EPISODES == 0 + EVAL_UNTIL_INFRACTION=True  → single infraction eval
# N_SPAWN_EPISODES == 0 + EVAL_UNTIL_INFRACTION=False → fixed 3000-step eval
N_SPAWN_EPISODES = 5
MAX_EPISODE_STEPS = 1500
EVAL_SEED = 2021                  # reused for every episode so routes are identical

EVAL_UNTIL_INFRACTION = False
MAX_INFRACTION_STEPS = 4000

if N_SPAWN_EPISODES > 0:
    PROGRESS_FILE = Path('eval_progress_spawns.json')
elif EVAL_UNTIL_INFRACTION:
    PROGRESS_FILE = Path('eval_progress_infraction.json')
else:
    PROGRESS_FILE = Path('eval_progress.json')

# ── Combinações sem temporal buffer ─────────────────────────────────────────
# arc → lista de checkpoints de policy a avaliar
ARC_TO_EVAL_NAMES = {
    'unet':          ['real-bev', 'unet'],        # já avaliado
    'cvt':           ['real-bev', 'cvt_3ch_L1'],  # já avaliado
    # 'cvt_finetuned': ['real-bev'],                # já avaliado
    'expert':        ['real-bev'],                # já avaliado
    'cvt_6ch':       ['real-bev','cvt_6ch_vanilla'],  # já avaliado
    'cvt_6ch_kde':   ['real-bev'],                # já avaliado
    'cvt_6ch_traj':  ['real-bev'],
    'cvt_6ch_traj_cmd':         ['real-bev'],
    'cvt_6ch_traj_cmd_kde':     ['real-bev'],
    'cvt_6ch_vanilla_no_noise': ['real-bev'],

}

# ── Combinações com temporal buffer ─────────────────────────────────────────
# Policy: ckpts_temporal/ckpt-<eval_name>/ckpt_latest.pth
ARC_TO_EVAL_NAMES_TEMPORAL = {
    'cvt':           ['cvt_3ch_L1'],   # já avaliado
    'cvt_finetuned': ['cvt_3ch_L1'],   # já avaliado
    'cvt_6ch':       ['cvt_3ch_L1'],   # já avaliado
    'cvt_6ch_kde':   ['cvt_3ch_L1'],   # já avaliado
    'cvt_6ch_traj':  ['cvt_3ch_L1'],
}


def _exp_key(town: str, eval_name: str, arc: str, temporal: bool) -> str:
    suffix = 'temporal' if temporal else 'standard'
    return f'{town}__{eval_name}__{arc}__{suffix}'


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {'completed': [], 'failed': []}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=4))


def mark_completed(progress: dict, town: str, eval_name: str, arc: str, temporal: bool) -> None:
    key = _exp_key(town, eval_name, arc, temporal)
    if key not in progress['completed']:
        progress['completed'].append(key)
    _save_progress(progress)


def mark_failed(progress: dict, town: str, eval_name: str, arc: str, temporal: bool, error: str) -> None:
    key = _exp_key(town, eval_name, arc, temporal)
    entry = {'key': key, 'error': error}
    progress['failed'].append(entry)
    _save_progress(progress)


def is_completed(progress: dict, town: str, eval_name: str, arc: str, temporal: bool) -> bool:
    return _exp_key(town, eval_name, arc, temporal) in progress['completed']


def make_env_maker(town_name: str, arc: str):
    obs_configs = get_obs_configs(arc)
    _env_configs = {
        'carla_map': town_name,
        'num_zombie_vehicles': [0, 150],
        'num_zombie_walkers': [0, 300],
        'weather_group': 'ClearNoon',
    }

    if arc in ('cvt_6ch_traj_cmd', 'cvt_6ch_traj_cmd_kde'):
        matrices_state = 'matrices_traj_cmd'
    elif arc == 'cvt_6ch_traj':
        matrices_state = 'matrices_traj'
    else:
        matrices_state = 'matrices'

    def _maker():
        cfg = json.load(open('config/carla_config.json', 'r'))
        env = EndlessEnv(
            obs_configs=obs_configs,
            reward_configs=reward_configs,
            terminal_configs=terminal_configs,
            host=cfg['host'],
            port=cfg['port'],
            seed=2021,
            no_rendering=False,
            **_env_configs,
        )
        env = RlBirdviewWrapper(
            env,
            input_states=['rgb', 'traj', 'state', matrices_state],
            acc_as_action=True,
        )
        return env

    return _maker


def build_policy(eval_name: str, temporal_buffer: bool = False) -> AgentPolicy:
    bev_channels = 9 if temporal_buffer else 3
    observation_space = gym.spaces.Dict(
        birdview=gym.spaces.Box(low=0, high=255, shape=(bev_channels, 192, 192), dtype=np.uint8),
        state=gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32),
    )
    action_space = gym.spaces.Box(
        low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32
    )
    policy_kwargs = {
        'observation_space': observation_space,
        'action_space': action_space,
        'policy_head_arch': [256, 256],
        'features_extractor_entry_point': 'torch_layers:XtMaCNN',
        'features_extractor_kwargs': {'states_neurons': [256, 256]},
        'distribution_entry_point': 'distributions:BetaDistribution',
        'real_bev': eval_name == 'real-bev',
    }
    policy = AgentPolicy(**policy_kwargs)
    policy.to(DEVICE)
    return policy


def run_single_eval(
    env, policy, town_name: str, eval_name: str, bev_arc: str,
    temporal_buffer: bool = False,
    spawn_points: list = None,
):
    ckpt_folder = 'ckpts_temporal' if temporal_buffer else 'ckpts'
    ckpt_path = Path(f'{ckpt_folder}/ckpt-{eval_name}/ckpt_latest.pth')
    saved = th.load(ckpt_path.as_posix(), map_location=DEVICE)
    policy.load_state_dict(saved['policy_state_dict'])

    video_folder = 'video_temporal' if temporal_buffer else 'video'
    video_path = Path(f'{video_folder}/{town_name.lower()}')
    video_path.mkdir(parents=True, exist_ok=True)
    eval_video_path = (video_path / f'{eval_name}_{bev_arc}.mp4').as_posix()

    bev_video_path = None
    if SAVE_BEV_VIDEO and not temporal_buffer:
        bev_video_path = Path(f'bev_video/{town_name.lower()}/{eval_name}_{bev_arc}.mp4')

    save_iou = not temporal_buffer and bev_arc != 'expert' and spawn_points is None
    iou_csv_path = None
    if save_iou:
        iou_dir = Path(f'eval/{town_name.lower()}')
        iou_dir.mkdir(parents=True, exist_ok=True)
        iou_csv_path = str(iou_dir / f'{eval_name}_{bev_arc}.csv')

    if spawn_points is not None:
        metrics_folder = 'eval_metrics_spawns'
    elif EVAL_UNTIL_INFRACTION:
        metrics_folder = 'eval_metrics_infraction'
    elif temporal_buffer:
        metrics_folder = 'eval_metrics_temporal'
    else:
        metrics_folder = 'eval_metrics'
    metrics_path = Path(f'{metrics_folder}/{town_name.lower()}')
    metrics_path.mkdir(parents=True, exist_ok=True)
    traj_output_path = str(metrics_path / 'trajectories' / f'{eval_name}_{bev_arc}.json') if SAVE_TRAJ else None

    if spawn_points is not None:
        avg_ep_stat, avg_route_completion, ep_events = evaluate_policy_N_spawns(
            env,
            policy,
            eval_video_path,
            spawn_points=spawn_points,
            max_episode_steps=MAX_EPISODE_STEPS,
            arc=bev_arc,
            video_save_dir=bev_video_path,
            temporal_buffer=temporal_buffer,
            save_traj=SAVE_TRAJ,
            traj_output_path=traj_output_path,
            eval_seed=EVAL_SEED,
        )
    elif EVAL_UNTIL_INFRACTION:
        avg_ep_stat, avg_route_completion, ep_events = evaluate_policy_until_infraction(
            env,
            policy,
            eval_video_path,
            max_eval_steps=MAX_INFRACTION_STEPS,
            arc=bev_arc,
            video_save_dir=bev_video_path,
            temporal_buffer=temporal_buffer,
            save_traj=SAVE_TRAJ,
            traj_output_path=traj_output_path,
        )
    else:
        avg_ep_stat, avg_route_completion, ep_events = evaluate_policy(
            env,
            policy,
            eval_video_path,
            arc=bev_arc,
            video_save_dir=bev_video_path,
            temporal_buffer=temporal_buffer,
            save_iou_csv=save_iou,
            iou_csv_path=iou_csv_path,
            save_traj=SAVE_TRAJ,
            traj_output_path=traj_output_path,
        )

    out_file = metrics_path / f'{eval_name}_{bev_arc}.json'
    with open(out_file, 'w') as f:
        json.dump(
            {'avg_ep_stat': avg_ep_stat, 'avg_route_completion': avg_route_completion},
            f,
            indent=4,
        )

    print(f'  -> saved metrics to {out_file}')
    env.reset()


def run_block(arc_map: dict, temporal_buffer: bool, progress: dict):
    label = 'TEMPORAL' if temporal_buffer else 'standard'
    total = sum(len(v) for v in arc_map.values()) * len(TOWNS)
    done = 0

    for town_name in TOWNS:
        for bev_arc, eval_names in arc_map.items():
            pending = [n for n in eval_names if not is_completed(progress, town_name, n, bev_arc, temporal_buffer)]
            if not pending:
                done += len(eval_names)
                print(f'\n[{label}] Skipping town={town_name}, arc={bev_arc} — all already done')
                continue

            print(f'\n[{label} {done}/{total}] Creating env — town={town_name}, arc={bev_arc}')
            env = SubprocVecEnv([make_env_maker(town_name, bev_arc)])

            # Select spread spawn points once per env — all models in this block
            # will use the exact same N spawns for a fair comparison.
            spawn_points = None
            if N_SPAWN_EPISODES > 0:
                candidates = env.env_method('get_spawn_candidates')[0]
                spawn_points = select_spread_spawns(candidates, N_SPAWN_EPISODES, seed=EVAL_SEED)
                print(f'  Spawn points selected ({N_SPAWN_EPISODES}):')
                for k, sp in enumerate(spawn_points):
                    print(f'    [{k}] x={sp["x"]:8.1f}  y={sp["y"]:8.1f}  yaw={sp["yaw"]:6.1f}')

            try:
                for eval_name in eval_names:
                    done += 1
                    if is_completed(progress, town_name, eval_name, bev_arc, temporal_buffer):
                        print(f'[{label} {done}/{total}] Skipping — already completed: policy={eval_name}, arc={bev_arc}, town={town_name}')
                        continue

                    print(
                        f'[{label} {done}/{total}] Evaluating — '
                        f'policy={eval_name}, arc={bev_arc}, town={town_name}'
                    )
                    try:
                        env.env_method('reset_rng', EVAL_SEED)
                        policy = build_policy(eval_name, temporal_buffer=temporal_buffer)
                        run_single_eval(
                            env, policy, town_name, eval_name, bev_arc,
                            temporal_buffer=temporal_buffer,
                            spawn_points=spawn_points,
                        )
                        mark_completed(progress, town_name, eval_name, bev_arc, temporal_buffer)
                    except Exception as e:
                        error_msg = traceback.format_exc()
                        print(f'  ERROR: {e}\n{error_msg}')
                        mark_failed(progress, town_name, eval_name, bev_arc, temporal_buffer, error_msg)
                        raise
            finally:
                env.close()
                print(f'  env closed — town={town_name}, arc={bev_arc}')


if __name__ == '__main__':
    progress = load_progress()
    n_done = len(progress['completed'])
    n_failed = len(progress['failed'])
    if n_done or n_failed:
        print(f'Resuming — {n_done} completed, {n_failed} previously failed')
        for f in progress['failed']:
            print(f"  [failed] {f['key']}")

    print('\n' + '='*60)
    print('  BLOCK 1 — standard evaluations')
    print('='*60)
    run_block(ARC_TO_EVAL_NAMES, temporal_buffer=False, progress=progress)

    # print('\n' + '='*60)
    # print('  BLOCK 2 — temporal buffer: cvt_6ch_traj')
    # print('='*60)
    # run_block(ARC_TO_EVAL_NAMES_TEMPORAL, temporal_buffer=True, progress=progress)
