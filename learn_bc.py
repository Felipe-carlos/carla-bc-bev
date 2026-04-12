import torch.optim as optim
import numpy as np
import tqdm
import torch as th
from pathlib import Path
import wandb
import gym
import json
import os


from expert_dataset_def.expert_dataset import ExpertDataset
from agent_policy import AgentPolicy
from bev_generation.cvt_3ch import CVT_3chL1Generator
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv('WANDB_API_KEY')

def learn_bc(policy, device, expert_loader, eval_loader, resume_last_train):
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = output_dir / 'checkpoint.txt'

    bev_generator = CVT_3chL1Generator(device=device)
    project_name = f'bev_bc-{bev_generator.__name__()}'

    ckpt_dir = Path(f'ckpts/ckpt-{bev_generator.__name__()}')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume_last_train:
        with open(last_checkpoint_path, 'r') as f:
            wb_run_path = f.read()
        #wandb.login(key=API_KEY)
        api = wandb.Api()
        wandb_run = api.run(wb_run_path)
        wandb_run_id = wandb_run.id
        ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
        saved_variables = th.load(ckpt_path, map_location='cuda')
        train_kwargs = saved_variables['train_init_kwargs']
        start_ep = train_kwargs['start_ep']
        i_steps = train_kwargs['i_steps']
         
        policy.load_state_dict(saved_variables['policy_state_dict'])
        wandb.init(project=project_name, id=wandb_run_id, resume='must')
    else:
        run = wandb.init(project=project_name, reinit=True)
        with open(last_checkpoint_path, 'w') as log_file:
            log_file.write(wandb.run.path)
        start_ep = 0
        i_steps = 0

    video_path = Path('video')
    video_path.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(policy.parameters(), lr=1e-5)
    episodes = 400
    ent_weight = 0.01
    min_eval_loss = np.inf
    eval_step = int(1e5)
    steps_last_eval = 0

    
    for i_episode in tqdm.tqdm(range(start_ep, episodes),'Episode:'):
        total_loss = 0
        i_batch = 0
        policy = policy.train()
        # Expert dataset
        for expert_batch in tqdm.tqdm(expert_loader,'Batches:'):
            expert_obs_dict, expert_action = expert_batch
           
            bev = bev_generator.infer(expert_obs_dict)
                        
            obs_tensor_dict = {
                'state': expert_obs_dict['state'].float().to(device),
                'birdview': bev.to(device)
            }
            expert_action = expert_action.to(device)

            # Get BC loss
            alogprobs, entropy_loss = policy.evaluate_actions(obs_tensor_dict, expert_action)
            bcloss = -alogprobs.mean()

            loss = bcloss + ent_weight * entropy_loss
            total_loss += loss
            i_batch += 1
            i_steps += expert_obs_dict['state'].shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_eval_loss = 0
        i_eval_batch = 0
        for expert_batch in eval_loader:
            expert_obs_dict, expert_action = expert_batch
            bev = bev_generator.infer(expert_obs_dict)
            
            obs_tensor_dict = {
                'state': expert_obs_dict['state'].float().to(device),
                'birdview': bev.to(device)
            }
            expert_action = expert_action.to(device)

            # Get BC loss
            with th.no_grad():
                alogprobs, entropy_loss = policy.evaluate_actions(obs_tensor_dict, expert_action)
            bcloss = -alogprobs.mean()

            eval_loss = bcloss + ent_weight * entropy_loss
            total_eval_loss += eval_loss
            i_eval_batch += 1
        
        loss = total_loss / i_batch
        eval_loss = total_eval_loss / i_eval_batch
        wandb.log({
            'loss': loss,
            'eval_loss': eval_loss,
        }, step=i_steps)

        if min_eval_loss > eval_loss:
            ckpt_path = (ckpt_dir / f'bc_ckpt_{i_episode}_min_eval.pth').as_posix()
            th.save(
                {'policy_state_dict': policy.state_dict()},
               ckpt_path
            )
            min_eval_loss = eval_loss

        train_init_kwargs = {
            'start_ep': i_episode,
            'i_steps': i_steps
        } 
        ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
        th.save({'policy_state_dict': policy.state_dict(),
                 'train_init_kwargs': train_init_kwargs},
                ckpt_path)
        wandb.save(f'./{ckpt_path}')
    run = run.finish()


if __name__ == '__main__':
    resume_last_train = False

    observation_space = {}
    observation_space['birdview'] = gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8)
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
    }

    device = 'cuda'

    policy = AgentPolicy(**policy_kwargs)
    policy.to(device)

    batch_size = 1024

    gail_train_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'expert-data',
            routes=range(2, 10),
            n_eps=1,
            unet=False
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    
    gail_val_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'expert-data',
            routes=[0,1],
            n_eps=1,
            unet=False
            
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    learn_bc(policy, device, gail_train_loader, gail_val_loader, resume_last_train)