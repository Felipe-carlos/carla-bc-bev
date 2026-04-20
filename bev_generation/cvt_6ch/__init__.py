from bev_generation.IBEV_Generator import IBEVGenerator
from .model.model_builder import ModelBuilder
import torch 
import os

class CVT_6chVanilla(IBEVGenerator):
    def __init__(self, model_path:str=None ,device='cuda', use_eval=True):
        self.device = device
        if model_path is None:
            main = os.getcwd()
            model_path = os.path.join(main, 'bev_generation/cvt_6ch/ckpts/ckpt_vanilla_49.pth') 
        else:
            model_path = model_path
        

        self.generator = ModelBuilder()
        self.generator = self.generator.get_net()
        
        state_dict = torch.load(model_path)
        self.generator.load_state_dict(state_dict['network_state_dict'])
        self.generator.to(device)

        if use_eval:
            self.generator.eval()


    def __name__(self):
        return "cvt_6ch_vanilla"
        
    def infer(self, expert_obs_dict):
        """
        Inferência sem gradiente (eval mode)

        Returns:
            bev tensor (B, C, H, W)
            expert_obs_dict precisa conter as chaves  'image', 'extrinsics','intrinsics'
        """
        with torch.no_grad():
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in expert_obs_dict.items()}
            fake_birdview = self.generator(batch)
            # Resize: (4, 3, 200, 200) → (4, 3, 192, 192)
            fake_birdview = torch.nn.functional.interpolate(
            fake_birdview,
            size=(192, 192),
            mode='bilinear',   # melhor para imagens
            align_corners=False
            )

        fake_birdview = (fake_birdview > 0.5).byte()
        fake_birdview = fake_birdview[:, :3, :, :] * 255
        return fake_birdview

