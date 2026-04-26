# modified from https://github.com/rohitrango/BC-regularized-GAIL/blob/master/a2c_ppo_acktr/algo/gail.py


from pathlib import Path
from typing import Union, Dict, Tuple, Any
from functools import partial
import time
import torch as th
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw
from config.obs_config import obs_configs
# from data_collect import obs_configs
import warnings
from .dataset_matrices_6ch import *
import os

warnings.filterwarnings("ignore")


def get_intrinsics(obs_configs, bev_resize):
    """
    Retorna lista de matrizes intrínsecas 3x3 para ['left_rgb', 'central_rgb', 'right_rgb', 'rear_rgb', 'bev']
    """
    stack_order = ['left_rgb', 'central_rgb', 'right_rgb', 'rear_rgb', 'bev']
    intrinsics = []

    for key in stack_order:
        if key in obs_configs['hero']:
            cam = obs_configs['hero'][key]
            fov = cam['fov']
            width = cam['width']
            height = cam['height']
            intrinsics.append(intrinsic_cam(fov, width=width, height=height))
        else:
            # BEV do traj
            bev_width = bev_resize 
            bev_height = bev_width
            pixels_per_meter = obs_configs['hero']['birdview']['pixels_per_meter'] * bev_resize / 192
            intrinsics.append(intrinsic_bev(bev_width, bev_height, pixels_per_meter))

    return intrinsics


def get_extrinsics(obs_configs, bev_resize):
    """
    Retorna lista de matrizes extrínsecas 4x4 para ['left_rgb', 'central_rgb', 'right_rgb', 'rear_rgb', 'bev']
    """
    stack_order = ['left_rgb', 'central_rgb', 'right_rgb', 'rear_rgb', 'bev']
    extrinsics = []

    for key in stack_order:
        ego_x, ego_y = bev_resize // 2, bev_resize - 40  # ego 40 px do fundo
        ppm = obs_configs['hero']['birdview']['pixels_per_meter'] * bev_resize / 192

        if key in obs_configs['hero']:
            cam = obs_configs['hero'][key]
            location = list(cam['location'])
            rotation = list(cam['rotation'])
            T = extrinsic_cam(location, rotation, ego_x, ego_y, bev_resize, bev_resize, ppm)
        else:
            # BEV virtual
            T = extrinsic_bev(ego_x, ego_y, bev_resize, bev_resize, ppm)
        extrinsics.append(T)

    return extrinsics

def traj_plotter_rgb(traj, w_resize , h_resize , img_path=None):
    radius = 10
    color = (255, 255, 255)
    scale = 500
    point_idx = -1
    img = Image.fromarray(np.zeros((h_resize, w_resize, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    while (point_idx + 1) * 2 <= len(traj):
        if point_idx < 0:
            x = traj[1] * scale
            y = -1 * traj[0] * scale
        elif point_idx == 0:
            x = 0
            y = 0
        else:
            x = traj[point_idx*2 + 1] * scale
            y = -1 * traj[point_idx*2] * scale
        x += w_resize / 2
        y += h_resize - 40
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        point_idx += 1
    image_array = np.transpose(img, [2, 0, 1])
    image_tensor = th.as_tensor(image_array)
    return image_tensor

class ExpertDataset(th.utils.data.Dataset):
    def __init__(self, dataset_directory, n_eps=1, routes=[i for i in range(10)], ep_start=0, unet=False,random_matriz=False,sizes=(256,256,256)):
        self.random_matriz = random_matriz
        self.dataset_path = Path(dataset_directory)
        self.length = 0
        self.get_idx = []
        self.trajs_states = []
        self.trajs_actions = []
        if unet:
            self.w_resize = 192
            self.h_resize = 192
            self.bev_resize = 192
        else:
            self.w_resize = sizes[0]#480
            self.h_resize = sizes[1]# #224
            self.bev_resize = sizes[2]# #200

        for route_idx in routes:
            for ep_idx in range(ep_start, ep_start + n_eps):
                route_path = self.dataset_path / ('route_%02d' % route_idx) / ('ep_%02d' % ep_idx)
                route_df = pd.read_json(route_path / 'episode.json')
                traj_length = len(os.listdir(route_path / "central_rgb")) -2 #route_df.shape[0]
                self.length += traj_length
                for step_idx in range(traj_length):
                    self.get_idx.append((route_idx, ep_idx, step_idx))
                    state_dict = {}
                    for state_key in route_df.columns:
                        state_dict[state_key] = route_df.iloc[step_idx][state_key]
                    self.trajs_states.append(state_dict)
                    self.trajs_actions.append(th.Tensor(route_df.iloc[step_idx]['actions']))

        self.trajs_actions = th.stack(self.trajs_actions)
        self.actual_obs = [None for _ in range(self.length)]

    def __len__(self):
        return self.length

    def process_image(self, image_path, birdview=False):
        image_array = Image.open(image_path)
        if birdview:
            image_array = image_array.resize((self.bev_resize, self.bev_resize), resample=Image.NEAREST)
        else:
            image_array = image_array.resize((self.w_resize, self.h_resize), resample=Image.NEAREST)
        image_array = np.transpose(image_array, [2, 0, 1])
        image_tensor = th.as_tensor(image_array.copy())
        image_tensor = image_tensor / 255.0
        return image_tensor

    def __getitem__(self, j):
        route_idx, ep_idx, step_idx = self.get_idx[j]
    
        # Load only the first time, images in uint8 are supposed to be light
        ep_dir = self.dataset_path / 'route_{:0>2d}/ep_{:0>2d}'.format(route_idx, ep_idx)
        masks_list = []
        for mask_index in range(1):
            mask_tensor = self.process_image(ep_dir / 'birdview_masks/{:0>4d}_{:0>2d}.png'.format(step_idx, mask_index), birdview=True)
            masks_list.append(mask_tensor)
        birdview = th.cat(masks_list)

        central_rgb = self.process_image(ep_dir / 'central_rgb/{:0>4d}.png'.format(step_idx))
        left_rgb = self.process_image(ep_dir / 'left_rgb/{:0>4d}.png'.format(step_idx))
        right_rgb = self.process_image(ep_dir / 'right_rgb/{:0>4d}.png'.format(step_idx))
        rear_rgb = self.process_image(ep_dir / 'rear_rgb/{:0>4d}.png'.format(step_idx)) ###
        state_dict = self.trajs_states[j]
        traj_plot_rgb = traj_plotter_rgb(state_dict['traj'], self.w_resize, self.h_resize) / 255.0

        images = th.stack([left_rgb, central_rgb, right_rgb, rear_rgb,traj_plot_rgb])
        # if self.random_matriz:
        #     extrinsics = EXTRINSICS
        #     intrinsics = INTRINSICS
        # else:
        extrinsics = get_extrinsics(obs_configs=obs_configs,bev_resize=self.bev_resize)
        intrinsics = get_intrinsics(obs_configs=obs_configs,bev_resize=self.bev_resize)
        extrinsics = th.Tensor(np.array(extrinsics))
        intrinsics = th.Tensor(np.array(intrinsics))
        obs_dict = {
            'bev': birdview,
            'image': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }


        return obs_dict
