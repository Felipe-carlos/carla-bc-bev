from bev_generation.IBEV_Generator import IBEVGenerator
from .model.model_builder import ModelBuilderTraj
import torch
import os


class CVT_6chTraj(IBEVGenerator):
    def __init__(self, model_path: str = None, device='cuda', use_eval=True):
        self.device = device
        if model_path is None:
            main = os.getcwd()
            model_path = os.path.join(main, 'bev_generation/cvt_6ch_traj/ckpts/ckpt_49.pth')

        self.generator = ModelBuilderTraj().get_net()

        state_dict = torch.load(model_path, map_location=device)
        self.generator.load_state_dict(state_dict['network_state_dict'])
        self.generator.to(device)

        if use_eval:
            self.generator.eval()

    def __name__(self):
        return "cvt_6ch_traj"

    def _to_device(self, obs_dict):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in obs_dict.items()}

    def infer(self, obs_dict):
        """No-gradient inference. Returns binarized BEV (B, 3, 192, 192) with values in {0, 255}."""
        with torch.no_grad():
            batch = self._to_device(obs_dict)
            out = self.generator(batch)
            out = torch.nn.functional.interpolate(out, size=(192, 192), mode='nearest')
        return (out > 0.0).byte()[:, :3, :, :] * 255

    def forward_train(self, obs_dict):
        batch = self._to_device(obs_dict)
        out = self.generator(batch)
        return torch.nn.functional.interpolate(out, size=(192, 192), mode='nearest')

    def compute_loss(self, pred, target):
        return torch.nn.functional.binary_cross_entropy_with_logits(pred[:, :3], target)
