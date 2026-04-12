import numpy as np
import time
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from bev_generation.unet import Unet_BEVGenerator
from pathlib import Path
import cv2 
import torch as th
from typing import Literal



def create_image_tensor(obs, unet=False, w_resize=192, h_resize=192):

    def process_image(image: np.ndarray, traj=False):

        if image.ndim == 4:
            image = image[0]

        if not traj:
            image = image.transpose(1, 2, 0)
            image = cv2.resize(image, (w_resize, h_resize),interpolation=cv2.INTER_NEAREST)
            image = image.transpose(2, 0, 1)
        else:
            if image.ndim == 3:
                image = image[0]

            image = cv2.resize(image, (w_resize, h_resize),interpolation=cv2.INTER_NEAREST)
            image = image[None, :, :]

        return th.as_tensor(image, dtype=th.float32) / 255.0

    image_tensor_list = []

    

    if unet:
        traj_plot = process_image(obs['traj_plot'], traj=True)
        camera_order = ["central_rgb", "left_rgb", "right_rgb", "rear_rgb"]
    else:
        camera_order = ["left_rgb", "central_rgb", "right_rgb", "rear_rgb"]
        traj_plot = process_image(obs['traj_plot_rgb'], traj=True)

    for i in camera_order:
        image_tensor_list.append(process_image(obs[i]))

    image_tensor_list.append(traj_plot)

    images = th.cat(image_tensor_list, dim=0)

    return images.unsqueeze(0)

def evaluate_policy(env, policy, video_path, min_eval_steps=3000, arc:Literal['unet', 'cvt', 'expert']='unet'):
    device = 'cuda'
    if arc != 'expert':
        output_dir = Path('outputs')
        
        last_checkpoint_path = output_dir / 'checkpoint.txt'
        bev_generator = Unet_BEVGenerator(device=device)

    policy = policy.eval()
    t0 = time.time()
    for i in range(env.num_envs):
        env.set_attr('eval_mode', True, indices=i)
    obs = env.reset()
    

    list_render = []
    ep_stat_buffer = []
    route_completion_buffer = []
    ep_events = {}
    for i in range(env.num_envs):
        ep_events[f'venv_{i}'] = []

    n_step = 0
    n_timeout = 0
    env_done = np.array([False]*env.num_envs)
    # while n_step < min_eval_steps:
    while n_step < min_eval_steps or not np.all(env_done):
        
        if arc != 'expert':
            unet = arc == 'unet'
            image_input = {'image': create_image_tensor(obs,unet=unet).to(device)}
            
            bev = bev_generator.infer(image_input)
            obs['birdview'] = bev
        
        actions, log_probs, mu, sigma, _ = policy.forward(obs, deterministic=True, clip_action=True)
        obs, reward, done, info = env.step(actions)

        for i in range(env.num_envs):
            env.set_attr('action_log_probs', log_probs[i], indices=i)
            env.set_attr('action_mu', mu[i], indices=i)
            env.set_attr('action_sigma', sigma[i], indices=i)

        list_render.append(env.render(mode='rgb_array'))

        n_step += 1
        env_done |= done

        for i in np.where(done)[0]:
            if not info[i]['timeout']:
                ep_stat_buffer.append(info[i]['episode_stat'])
            if n_step < min_eval_steps or not np.all(env_done):
                route_completion_buffer.append(info[i]['route_completion'])
            ep_events[f'venv_{i}'].append(info[i]['episode_event'])
            n_timeout += int(info[i]['timeout'])

    for ep_info in info:
        route_completion_buffer.append(ep_info['route_completion'])

    # conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
    encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
    for im in list_render:
        encoder.capture_frame(im)
    encoder.close()

    avg_ep_stat = get_avg_ep_stat(ep_stat_buffer, prefix='eval/')
    avg_route_completion = get_avg_route_completion(route_completion_buffer, prefix='eval/')
    avg_ep_stat['eval/eval_timeout'] = n_timeout

    duration = time.time() - t0
    avg_ep_stat['time/t_eval'] = duration
    avg_ep_stat['time/fps_eval'] = n_step * env.num_envs / duration

    for i in range(env.num_envs):
        env.set_attr('eval_mode', False, indices=i)
    obs = env.reset()
    return avg_ep_stat, avg_route_completion, ep_events

def get_avg_ep_stat(ep_stat_buffer, prefix=''):
    avg_ep_stat = {}
    n_episodes = float(len(ep_stat_buffer))
    if n_episodes > 0:
        for ep_info in ep_stat_buffer:
            for k, v in ep_info.items():
                k_avg = f'{prefix}{k}'
                if k_avg in avg_ep_stat:
                    avg_ep_stat[k_avg] += v
                else:
                    avg_ep_stat[k_avg] = v

        for k in avg_ep_stat.keys():
            avg_ep_stat[k] /= n_episodes
    avg_ep_stat[f'{prefix}completed_n_episodes'] = n_episodes

    return avg_ep_stat


def get_avg_route_completion(ep_route_completion, prefix=''):
    avg_ep_stat = {}
    n_episodes = float(len(ep_route_completion))
    if n_episodes > 0:
        for ep_info in ep_route_completion:
            for k, v in ep_info.items():
                k_avg = f'{prefix}{k}'
                if k_avg in avg_ep_stat:
                    avg_ep_stat[k_avg] += v
                else:
                    avg_ep_stat[k_avg] = v

        for k in avg_ep_stat.keys():
            avg_ep_stat[k] /= n_episodes
    avg_ep_stat[f'{prefix}avg_n_episodes'] = n_episodes

    return avg_ep_stat
