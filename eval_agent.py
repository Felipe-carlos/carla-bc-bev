import numpy as np
import time
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from bev_generation.unet import Unet_BEVGenerator
from bev_generation.cvt_3ch import CVT_3chL1Generator 
from bev_generation.cvt_6ch import CVT_6chVanilla
from pathlib import Path
import cv2 
import torch as th
from typing import Literal, Optional



def create_image_tensor(obs, unet=False, w_resize=192, h_resize=192):
    
    def process_image(image: np.ndarray, traj=False):

        if image.ndim == 4:
            image = image[0]

        if not traj:
            image = image.transpose(1, 2, 0)
            image = cv2.resize(image, (w_resize, h_resize),interpolation=cv2.INTER_NEAREST)
            image = image.transpose(2, 0, 1)
        else:
                 
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = np.transpose(image, (1, 2, 0))  # CHW → HWC

            image = cv2.resize(image, (w_resize, h_resize), interpolation=cv2.INTER_NEAREST)

            
            if image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))
            else:
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
    if unet:
        images = th.cat(image_tensor_list, dim=0)
    else:
        images = th.stack(image_tensor_list, dim=0)

    return images.unsqueeze(0)

def _save_bev_images(bev, step_idx, save_dir: Path):
    """Função auxiliar para salvar as BEVs geradas em formato PNG."""
    if bev is None:
        return
        
    # Garante que é tensor e move para CPU
    bev_cpu = bev.detach().cpu() if isinstance(bev, th.Tensor) else th.as_tensor(bev).cpu()
    bev_np = bev_cpu.numpy()  # Shape esperado: (B, C, H, W)
    
    for b in range(bev_np.shape[0]):
        img = bev_np[b]
        
        # Converte CHW -> HWC se necessário
        if img.ndim == 3 and img.shape[0] <= 10:  # Heurística para dimensão de canais
            img = np.transpose(img, (1, 2, 0))
            
        # Normaliza [0, 1] -> [0, 255] e converte para uint8
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        
        # Ajusta formato para cv2.imwrite
        if img.shape[2] == 1:
            img = img.squeeze(-1)  # Escala de cinza
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR
            
        fname = f"bev_step_{step_idx:05d}_env{b}.png"
        cv2.imwrite(str(save_dir / fname), img)

def evaluate_policy(env, policy, video_path, min_eval_steps=3000, arc:Literal['unet', 'cvt', 'expert','cvt_6ch']='unet',bev_save_dir: Optional[str] = None):
    device = 'cuda'
    bev_save_path = Path(bev_save_dir) if bev_save_dir is not None else None
    if bev_save_path is not None:
        bev_save_path.mkdir(parents=True, exist_ok=True)
        print(f"📸 BEV images will be saved to: {bev_save_path}")

    if arc != 'expert':
        output_dir = Path('outputs')
        
        last_checkpoint_path = output_dir / 'checkpoint.txt'
        if arc == 'unet':
            bev_generator = Unet_BEVGenerator(device=device)
        elif arc == 'cvt':
            bev_generator = CVT_3chL1Generator(device=device)
        elif arc == 'cvt_6ch':
            bev_generator = CVT_6chVanilla(device=device)



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
    print(f'Starting evaluation for at least {min_eval_steps} steps or until all environments are done...')
    while n_step < min_eval_steps or not np.all(env_done):
        
        if arc != 'expert':
            if arc == 'unet':
                unet = True
                image_input = {'image': create_image_tensor(obs,unet=unet).to(device)}
                bev = bev_generator.infer(image_input)
                obs['birdview'] = bev
            elif arc == 'cvt':
                unet = False
                image_input = {
                    'image': create_image_tensor(obs,unet=unet,w_resize=256,h_resize=256).to(device),
                    'extrinsics': th.as_tensor(obs['extrinsics'], dtype=th.float32).to(device),
                    'intrinsics': th.as_tensor(obs['intrinsics'], dtype=th.float32).to(device),
                }
                bev = bev_generator.infer(image_input)
                obs['birdview'] = bev
            elif arc == 'cvt_6ch':
                unet = False
                image_input = {
                    'image': create_image_tensor(obs,unet=unet,w_resize=480,h_resize=224).to(device),
                    'extrinsics': th.as_tensor(obs['extrinsics'], dtype=th.float32).to(device),
                    'intrinsics': th.as_tensor(obs['intrinsics'], dtype=th.float32).to(device),
                }
                bev = bev_generator.infer(image_input)
                obs['birdview'] = bev
            # === NOVO: Salvar BEVs geradas ===
            if bev_save_path is not None:
                _save_bev_images(bev, n_step, bev_save_path)
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
