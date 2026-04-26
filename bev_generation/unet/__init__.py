from bev_generation.IBEV_Generator import IBEVGenerator
from .unet_def import GeneratorUNet
import torch
import torch.nn.functional as F
import os


class Unet_BEVGenerator(IBEVGenerator):
    def __init__(self, model_path: str = None, device='cuda', use_eval=True):
        if model_path is None:
            main = os.getcwd()
            model_path = os.path.join(main, 'bev_generation/unet/l1_50_no sigmoid_generator_49.pth')
        self.device = device
        in_channels = 13
        self.generator = GeneratorUNet(in_channels=in_channels).to(device)

        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items()}
        self.generator.load_state_dict(new_state_dict)

        if use_eval:
            self.generator.eval()

    def __name__(self):
        return "unet"

    def infer(self, obs_dict):
        """No-gradient inference. Returns binarized BEV (B, 3, H, W) with values in {0, 255}."""
        with torch.no_grad():
            logits = self.generator(obs_dict['image'])
            bev = (logits > 0.0).byte() * 255  # logit > 0 ↔ sigmoid > 0.5
        return bev

    def forward_train(self, obs_dict):
        """Forward with gradients. Returns raw logits (B, 3, H, W)."""
        return self.generator(obs_dict['image'])

    def compute_loss(self, pred, target):
        """BCEWithLogitsLoss: pred is raw logits, target is in [0, 1]."""
        return F.binary_cross_entropy_with_logits(pred, target)

