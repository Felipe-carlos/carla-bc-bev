import torch.optim as optim
import numpy as np
import tqdm
import torch as th
from pathlib import Path
import wandb
import gym
import json


from agent_policy import AgentPolicy
from carla_gym.envs import EndlessEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from data_collect import reward_configs, terminal_configs
from eval_agent import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from config.obs_config import get_obs_configs


TOWN_NAME = 'Town01'
SAVE_BEV_VIDEO = True
eval_name = 'cvt_3ch_L1'
bev_arc = 'cvt'         ###['unet', 'cvt', 'expert','cvt_6ch']
temporal_buffer = False

env_configs = {
    'carla_map': TOWN_NAME,
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'ClearNoon'
}

def eval_bc(policy, device, env,eval_name, bev_arc='unet'):
    ckpt_folder = 'ckpts_temporal' if temporal_buffer else 'ckpts'
    ckpt_dir = Path(f'{ckpt_folder}/ckpt-{eval_name}')

    ckpt_path = (ckpt_dir / 'bc_ckpt_63_min_eval.pth').as_posix() ### era ckpt_latest.pth
    saved_variables = th.load(ckpt_path, map_location='cuda')

    policy.load_state_dict(saved_variables['policy_state_dict'])

    video_folder = 'video_temporal' if temporal_buffer else 'video'
    video_path = Path(f'{video_folder}/{TOWN_NAME.lower()}')
    video_path.mkdir(parents=True, exist_ok=True)

    bev_video_path = None
    if SAVE_BEV_VIDEO:
        eval_folder = 'eval_temporal' if temporal_buffer else 'eval'
        bev_video_path = Path(f'{eval_folder}/{bev_arc}/{TOWN_NAME.lower()}')    

    eval_video_path = (video_path / f'{eval_name}_{bev_arc}.mp4').as_posix()
    avg_ep_stat, avg_route_completion, ep_events = evaluate_policy(
        env,
        policy,
        eval_video_path,
        arc=bev_arc,
        video_save_dir=bev_video_path,
        temporal_buffer=temporal_buffer,
    )
  
    metrics_folder = 'eval_metrics_temporal' if temporal_buffer else 'eval_metrics'
    metrics_path = Path(f'{metrics_folder}/{TOWN_NAME.lower()}')
    metrics_path.mkdir(parents=True, exist_ok=True)
    with open(metrics_path / f'{eval_name}_{bev_arc}.json', 'w') as f:
        json.dump({
            'avg_ep_stat': avg_ep_stat,
            'avg_route_completion': avg_route_completion
        }, f, indent=4)
    env.reset()
    


def env_maker():
    cfg = json.load(open("config/carla_config.json", "r"))
    env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host=cfg['host'], port=cfg['port'],
                    seed=2021, no_rendering=False, **env_configs)
    env = RlBirdviewWrapper(env,input_states=['rgb', 'traj', 'state', 'matrices'], acc_as_action=True)
    return env

if __name__ == '__main__':

    obs_configs = get_obs_configs(bev_arc)

    env = SubprocVecEnv([env_maker])

    observation_space = {}
    shape = (9, 192, 192) if temporal_buffer else (3, 192, 192)
    observation_space['birdview'] = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)
    observation_space = gym.spaces.Dict(**observation_space)

    action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)

    
    # network
    policy_kwargs = {
        'observation_space': observation_space,
        'action_space': action_space,
        'policy_head_arch': [256, 256],
        'features_extractor_entry_point': 'torch_layers:XtMaCNN',
        'features_extractor_kwargs': {'states_neurons': [256,256]},
        'distribution_entry_point': 'distributions:BetaDistribution',
        'real_bev': eval_name == 'real-bev',
    }

    device = 'cuda'

    policy = AgentPolicy(**policy_kwargs)
    policy.to(device)

    batch_size = 24

    eval_bc(policy, device, env,eval_name=eval_name, bev_arc=bev_arc)