"""
Infraction-based evaluations for the BC-BEV project.

evaluate_policy_until_infraction
    Runs a single continuous episode until the first infraction (or max_eval_steps).

evaluate_policy_N_spawns
    Runs N episodes, each from a different fixed spawn point, stopping each episode
    at the first infraction or max_episode_steps.  All agents use the same N spawns
    and the same per-episode seed, so trajectories are directly comparable.

select_spread_spawns
    Selects N spatially spread spawn points from the full candidate list using
    farthest-point sampling.
"""
import json
import time
import numpy as np
from pathlib import Path
from typing import Literal, Optional

import torch as th
from gym.wrappers.monitoring.video_recorder import ImageEncoder

from bev_generation.bev_buffer import TemporalBEVBuffer
from bev_generation.unet import Unet_BEVGenerator
from bev_generation.cvt_3ch import CVT_3chL1Generator
from bev_generation.cvt_3ch_finetuned import CVT_3chL1FinetunedGenerator
from bev_generation.cvt_6ch import CVT_6chVanilla
from bev_generation.cvt_6ch_kde import CVT_6chKDE
from bev_generation.cvt_6ch_traj import CVT_6chTraj
from bev_generation.cvt_6ch_traj_cmd import CVT_6chTrajCmd
from bev_generation.cvt_6ch_traj_cmd_kde import CVT_6chTrajCmdKDE
from bev_generation.cvt_6ch_vanilla_no_noise import CVT_6chVanillaNoNoise

from eval_agent import (
    compose_frame,
    create_image_tensor,
    create_image_tensor_4cam,
    get_avg_ep_stat,
    get_avg_route_completion,
)


def _build_bev_generator(arc: str, device: str):
    if arc == 'expert':
        return None
    mapping = {
        'unet':                   Unet_BEVGenerator,
        'cvt':                    CVT_3chL1Generator,
        'cvt_finetuned':          CVT_3chL1FinetunedGenerator,
        'cvt_6ch':                CVT_6chVanilla,
        'cvt_6ch_kde':            CVT_6chKDE,
        'cvt_6ch_traj':           CVT_6chTraj,
        'cvt_6ch_traj_cmd':       CVT_6chTrajCmd,
        'cvt_6ch_traj_cmd_kde':   CVT_6chTrajCmdKDE,
        'cvt_6ch_vanilla_no_noise': CVT_6chVanillaNoNoise,
    }
    return mapping[arc](device=device)


def _infer_bev(arc: str, bev_generator, obs, device: str):
    if arc == 'unet':
        return bev_generator.infer({'image': create_image_tensor(obs, unet=True).to(device)})
    if arc in ('cvt', 'cvt_finetuned'):
        return bev_generator.infer({
            'image': create_image_tensor(obs, unet=False, w_resize=256, h_resize=256).to(device),
            'extrinsics': th.as_tensor(obs['extrinsics'], dtype=th.float32).to(device),
            'intrinsics': th.as_tensor(obs['intrinsics'], dtype=th.float32).to(device),
        })
    if arc in ('cvt_6ch', 'cvt_6ch_kde', 'cvt_6ch_vanilla_no_noise'):
        return bev_generator.infer({
            'image': create_image_tensor(obs, unet=False, w_resize=480, h_resize=224).to(device),
            'extrinsics': th.as_tensor(obs['extrinsics'], dtype=th.float32).to(device),
            'intrinsics': th.as_tensor(obs['intrinsics'], dtype=th.float32).to(device),
        })
    if arc == 'cvt_6ch_traj':
        return bev_generator.infer({
            'image': create_image_tensor_4cam(obs, w_resize=480, h_resize=224).to(device),
            'extrinsics': th.as_tensor(obs['extrinsics'], dtype=th.float32).to(device),
            'intrinsics': th.as_tensor(obs['intrinsics'], dtype=th.float32).to(device),
            'traj': th.as_tensor(obs['traj'], dtype=th.float32).to(device),
        })
    if arc in ('cvt_6ch_traj_cmd', 'cvt_6ch_traj_cmd_kde'):
        return bev_generator.infer({
            'image': create_image_tensor_4cam(obs, w_resize=480, h_resize=224).to(device),
            'extrinsics': th.as_tensor(obs['extrinsics'], dtype=th.float32).to(device),
            'intrinsics': th.as_tensor(obs['intrinsics'], dtype=th.float32).to(device),
            'traj': th.as_tensor(obs['traj'], dtype=th.float32).to(device),
            'cmd': th.as_tensor(obs['cmd'], dtype=th.float32).to(device),
        })
    raise ValueError(f'Unknown arc: {arc}')


def evaluate_policy_until_infraction(
    env,
    policy,
    video_path,
    max_eval_steps: int = 5000,
    arc: Literal[
        'unet', 'cvt', 'cvt_finetuned', 'expert',
        'cvt_6ch', 'cvt_6ch_kde', 'cvt_6ch_traj',
        'cvt_6ch_traj_cmd', 'cvt_6ch_traj_cmd_kde',
        'cvt_6ch_vanilla_no_noise',
    ] = 'unet',
    video_save_dir: Optional[str] = None,
    temporal_buffer: bool = False,
    save_traj: bool = False,
    traj_output_path: Optional[str] = None,
):
    device = 'cuda'

    bev_generator = _build_bev_generator(arc, device)

    temporal_bev_buffer = None
    if temporal_buffer:
        temporal_bev_buffer = TemporalBEVBuffer(device=device, channels=3)
        print('Temporal buffer enabled: [t-2, t-1, t] concatenation')

    bev_video_out = Path(video_save_dir) if video_save_dir is not None else None
    if bev_video_out is not None:
        bev_video_out.parent.mkdir(parents=True, exist_ok=True)

    policy = policy.eval()
    t0 = time.time()
    for i in range(env.num_envs):
        env.set_attr('eval_mode', True, indices=i)
    obs = env.reset()

    if temporal_bev_buffer is not None:
        temporal_bev_buffer.reset()

    list_render = []
    bev_video_encoder = None

    ep_stat_buffer = []
    route_completion_buffer = []
    ep_events = {f'venv_{i}': [] for i in range(env.num_envs)}

    traj_log = []
    _episode_id = [0] * env.num_envs

    n_step = 0
    n_timeout = 0
    infraction_occurred = False
    steps_until_infraction = max_eval_steps

    print(f'Starting infraction eval — stopping at first infraction or {max_eval_steps} steps...')

    while n_step < max_eval_steps:
        real_bev = obs.get('birdview', obs.get('topdown', None))

        if arc != 'expert':
            bev = _infer_bev(arc, bev_generator, obs, device)

            if temporal_buffer and temporal_bev_buffer is not None:
                bev = temporal_bev_buffer.get_concat(bev)

            obs['birdview'] = bev

            if bev_video_out is not None:
                frame = compose_frame(obs, bev, real_bev)
                if bev_video_encoder is None:
                    bev_video_encoder = ImageEncoder(
                        output_path=str(bev_video_out),
                        frame_shape=frame.shape,
                        frames_per_sec=30,
                        output_frames_per_sec=30,
                    )
                bev_video_encoder.capture_frame(frame)

        actions, log_probs, mu, sigma, _ = policy.forward(obs, deterministic=True, clip_action=True)
        obs, reward, done, info = env.step(actions)

        for i in range(env.num_envs):
            env.set_attr('action_log_probs', log_probs[i], indices=i)
            env.set_attr('action_mu', mu[i], indices=i)
            env.set_attr('action_sigma', sigma[i], indices=i)

        list_render.append(env.render(mode='rgb_array'))

        if save_traj:
            for i in range(env.num_envs):
                is_infraction_terminal = (
                    bool(done[i])
                    and not bool(info[i]['timeout'])
                    and not bool(info[i]['route_completion']['is_route_completed'])
                )
                traj_log.append({
                    'episode': _episode_id[i],
                    'step': n_step,
                    'env_idx': i,
                    'x': info[i].get('ev_x', float('nan')),
                    'y': info[i].get('ev_y', float('nan')),
                    'yaw': info[i].get('ev_yaw', float('nan')),
                    'is_infraction_terminal': is_infraction_terminal,
                })
                if done[i]:
                    _episode_id[i] += 1

        n_step += 1

        if temporal_bev_buffer is not None and np.any(done):
            temporal_bev_buffer.reset()

        for i in np.where(done)[0]:
            if not info[i]['timeout']:
                ep_stat_buffer.append(info[i]['episode_stat'])
            route_completion_buffer.append(info[i]['route_completion'])
            ep_events[f'venv_{i}'].append(info[i]['episode_event'])
            n_timeout += int(info[i]['timeout'])

            is_infraction = (
                not bool(info[i]['timeout'])
                and not bool(info[i]['route_completion']['is_route_completed'])
            )
            if is_infraction:
                steps_until_infraction = n_step
                infraction_occurred = True

        if infraction_occurred:
            print(f'  Infraction at step {steps_until_infraction} — stopping.')
            break

    for ep_info in info:
        route_completion_buffer.append(ep_info['route_completion'])

    encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
    for im in list_render:
        encoder.capture_frame(im)
    encoder.close()

    if bev_video_encoder is not None:
        bev_video_encoder.close()

    if save_traj and traj_output_path and traj_log:
        Path(traj_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(traj_output_path, 'w') as f:
            json.dump(traj_log, f)
        print(f'Trajectory log saved: {traj_output_path} ({len(traj_log)} steps)')

    avg_ep_stat = get_avg_ep_stat(ep_stat_buffer, prefix='eval/')
    avg_route_completion = get_avg_route_completion(route_completion_buffer, prefix='eval/')

    avg_ep_stat['eval/eval_timeout'] = n_timeout
    avg_ep_stat['eval/steps_until_infraction'] = steps_until_infraction
    avg_ep_stat['eval/infraction_occurred'] = 1.0 if infraction_occurred else 0.0

    duration = time.time() - t0
    avg_ep_stat['time/t_eval'] = duration
    avg_ep_stat['time/fps_eval'] = n_step * env.num_envs / duration

    for i in range(env.num_envs):
        env.set_attr('eval_mode', False, indices=i)
    obs = env.reset()

    return avg_ep_stat, avg_route_completion, ep_events


# ─────────────────────────────────────────────────────────────────────────────
# N-spawn evaluation
# ─────────────────────────────────────────────────────────────────────────────

def select_spread_spawns(candidates: list, n: int, seed: int = 0) -> list:
    """
    Select n spawn points spread across the map using farthest-point sampling.

    candidates  list of dicts with keys x, y, z, yaw, road_id
                (returned by env.env_method('get_spawn_candidates')[0])
    n           number of points to select
    seed        controls which point is chosen first
    """
    points = np.array([[c['x'], c['y']] for c in candidates], dtype=np.float32)
    rng = np.random.RandomState(seed)

    first = int(rng.randint(len(points)))
    selected = [first]

    # distance from each candidate to the nearest already-selected point
    min_dists = np.linalg.norm(points - points[first], axis=1)

    for _ in range(n - 1):
        farthest = int(np.argmax(min_dists))
        selected.append(farthest)
        d = np.linalg.norm(points - points[farthest], axis=1)
        min_dists = np.minimum(min_dists, d)

    return [candidates[i] for i in selected]


def evaluate_policy_N_spawns(
    env,
    policy,
    video_path,
    spawn_points: list,
    max_episode_steps: int = 5000,
    arc: Literal[
        'unet', 'cvt', 'cvt_finetuned', 'expert',
        'cvt_6ch', 'cvt_6ch_kde', 'cvt_6ch_traj',
        'cvt_6ch_traj_cmd', 'cvt_6ch_traj_cmd_kde',
        'cvt_6ch_vanilla_no_noise',
    ] = 'unet',
    video_save_dir: Optional[str] = None,
    temporal_buffer: bool = False,
    save_traj: bool = False,
    traj_output_path: Optional[str] = None,
    eval_seed: int = 2021,
):
    """
    Run len(spawn_points) episodes, each from a fixed spawn, until infraction or
    max_episode_steps.

    Per-episode seed reset guarantees that all agents see the same route from
    each spawn (same np.random sequence for route target selection).

    Extra keys in avg_ep_stat:
        eval/episodes_summary  – list of per-episode dicts with spawn coords,
                                 steps_until_infraction, infraction_occurred
    """
    device = 'cuda'
    bev_generator = _build_bev_generator(arc, device)

    temporal_bev_buffer = None
    if temporal_buffer:
        temporal_bev_buffer = TemporalBEVBuffer(device=device, channels=3)
        print('Temporal buffer enabled: [t-2, t-1, t] concatenation')

    bev_video_out = Path(video_save_dir) if video_save_dir is not None else None
    if bev_video_out is not None:
        bev_video_out.parent.mkdir(parents=True, exist_ok=True)

    policy = policy.eval()
    t0 = time.time()

    list_render = []
    bev_video_encoder = None

    ep_stat_buffer = []
    route_completion_buffer = []
    ep_events = {f'venv_{i}': [] for i in range(env.num_envs)}

    traj_log = []
    episodes_summary = []

    n_timeout = 0
    total_steps = 0

    for ep_idx, spawn in enumerate(spawn_points):
        # ── fix spawn and reset RNG so all agents share identical route ──────
        env.env_method('set_start_spawn', spawn['x'], spawn['y'], spawn['z'], spawn['yaw'])
        env.env_method('reset_rng', eval_seed)

        for i in range(env.num_envs):
            env.set_attr('eval_mode', True, indices=i)
        obs = env.reset()

        if temporal_bev_buffer is not None:
            temporal_bev_buffer.reset()

        n_step = 0
        infraction_occurred = False
        steps_until_infraction = max_episode_steps
        done = np.array([False] * env.num_envs)
        info = None

        print(
            f'  Episode {ep_idx + 1}/{len(spawn_points)} — '
            f'spawn ({spawn["x"]:.1f}, {spawn["y"]:.1f}), '
            f'max {max_episode_steps} steps'
        )

        while n_step < max_episode_steps:
            real_bev = obs.get('birdview', obs.get('topdown', None))

            if arc != 'expert':
                bev = _infer_bev(arc, bev_generator, obs, device)

                if temporal_buffer and temporal_bev_buffer is not None:
                    bev = temporal_bev_buffer.get_concat(bev)

                obs['birdview'] = bev

                if bev_video_out is not None:
                    frame = compose_frame(obs, bev, real_bev)
                    if bev_video_encoder is None:
                        bev_video_encoder = ImageEncoder(
                            output_path=str(bev_video_out),
                            frame_shape=frame.shape,
                            frames_per_sec=30,
                            output_frames_per_sec=30,
                        )
                    bev_video_encoder.capture_frame(frame)

            actions, log_probs, mu, sigma, _ = policy.forward(obs, deterministic=True, clip_action=True)
            obs, reward, done, info = env.step(actions)

            for i in range(env.num_envs):
                env.set_attr('action_log_probs', log_probs[i], indices=i)
                env.set_attr('action_mu', mu[i], indices=i)
                env.set_attr('action_sigma', sigma[i], indices=i)

            list_render.append(env.render(mode='rgb_array'))

            if save_traj:
                for i in range(env.num_envs):
                    is_infraction_terminal = (
                        bool(done[i])
                        and not bool(info[i]['timeout'])
                        and not bool(info[i]['route_completion']['is_route_completed'])
                    )
                    traj_log.append({
                        'episode': ep_idx,
                        'step': n_step,
                        'env_idx': i,
                        'spawn_x': spawn['x'],
                        'spawn_y': spawn['y'],
                        'x': info[i].get('ev_x', float('nan')),
                        'y': info[i].get('ev_y', float('nan')),
                        'yaw': info[i].get('ev_yaw', float('nan')),
                        'is_infraction_terminal': is_infraction_terminal,
                    })

            n_step += 1
            total_steps += 1

            if temporal_bev_buffer is not None and np.any(done):
                temporal_bev_buffer.reset()

            for i in np.where(done)[0]:
                if not info[i]['timeout']:
                    ep_stat_buffer.append(info[i]['episode_stat'])
                route_completion_buffer.append(info[i]['route_completion'])
                ep_events[f'venv_{i}'].append(info[i]['episode_event'])
                n_timeout += int(info[i]['timeout'])

                is_infraction = (
                    not bool(info[i]['timeout'])
                    and not bool(info[i]['route_completion']['is_route_completed'])
                )
                if is_infraction:
                    steps_until_infraction = n_step
                    infraction_occurred = True

            if infraction_occurred:
                print(f'    Infraction at step {steps_until_infraction}')
                break

        # partial episode (max steps reached without done signal)
        if info is not None:
            for i in range(env.num_envs):
                if not done[i]:
                    route_completion_buffer.append(info[i]['route_completion'])

        episodes_summary.append({
            'episode_idx': ep_idx,
            'spawn_x': spawn['x'],
            'spawn_y': spawn['y'],
            'spawn_yaw': spawn['yaw'],
            'steps_until_infraction': steps_until_infraction,
            'infraction_occurred': infraction_occurred,
        })

    # ── save main render video (all episodes concatenated) ───────────────────
    if list_render:
        encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
        for im in list_render:
            encoder.capture_frame(im)
        encoder.close()

    if bev_video_encoder is not None:
        bev_video_encoder.close()

    if save_traj and traj_output_path and traj_log:
        Path(traj_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(traj_output_path, 'w') as f:
            json.dump(traj_log, f)
        print(f'Trajectory log saved: {traj_output_path} ({len(traj_log)} steps, {len(spawn_points)} episodes)')

    avg_ep_stat = get_avg_ep_stat(ep_stat_buffer, prefix='eval/')
    avg_route_completion = get_avg_route_completion(route_completion_buffer, prefix='eval/')

    avg_ep_stat['eval/eval_timeout'] = n_timeout
    avg_ep_stat['eval/episodes_summary'] = episodes_summary

    duration = time.time() - t0
    avg_ep_stat['time/t_eval'] = duration
    avg_ep_stat['time/fps_eval'] = (total_steps * env.num_envs / duration) if duration > 0 else 0.0

    for i in range(env.num_envs):
        env.set_attr('eval_mode', False, indices=i)
    env.reset()

    return avg_ep_stat, avg_route_completion, ep_events
