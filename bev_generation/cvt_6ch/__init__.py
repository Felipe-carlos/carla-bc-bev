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

    def _to_device(self, obs_dict):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in obs_dict.items()}

    def infer(self, obs_dict):
        """No-gradient inference. Returns binarized BEV (B, 3, 192, 192) with values in {0, 255}."""
        with torch.no_grad():
            batch = self._to_device(obs_dict)
            out = self.generator(batch)
            out = torch.nn.functional.interpolate(out, size=(192, 192), mode='nearest')
        return (out > 0.5).byte()[:, :3, :, :] * 255

    def forward_train(self, obs_dict):
        """Forward with gradients. Returns sigmoid output (B, 6, 192, 192) in [0, 1]."""
        batch = self._to_device(obs_dict)
        out = self.generator(batch)
        return torch.nn.functional.interpolate(out, size=(192, 192), mode='nearest')

    def compute_loss(self, pred, target):
        """BCELoss on first 3 channels: pred is sigmoid output in [0, 1], target is in [0, 1]."""
        pred_3ch = pred[:, :3, :, :].clamp(1e-7, 1 - 1e-7)
        return torch.nn.functional.binary_cross_entropy(pred_3ch, target)

