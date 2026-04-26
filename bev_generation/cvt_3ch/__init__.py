from bev_generation.IBEV_Generator import IBEVGenerator
from .model.encoder import Encoder
from .model.decoder import Decoder
from .model.cvt import CrossViewTransformer
from .model.efficientnet import EfficientNetExtractor
import torch 
import os

class CVT_3chL1Generator(IBEVGenerator):
    def __init__(self, model_path:str=None ,device='cuda', use_eval=True):
        if model_path is None:
            main = os.getcwd()
            model_path = os.path.join(main, 'bev_generation/cvt_3ch/ckpts/4_cam_l1/ckpt_49.pth')
        else:
            model_path = model_path
        self.device = device
        self.image_height = 256  ##
        self.image_width = 256  ##
        self.bev_resolution = 256

        backbone = EfficientNetExtractor(
        layer_names=['reduction_4'],
        image_height=self.image_height,
        image_width=self.image_width,
        model_name='efficientnet-b4'
        )
        cross_view = {
            'heads': 4,
            'dim_head': 32,
            'qkv_bias': True,
            'skip': True,
            'no_image_features': False,
            'image_height': self.image_height,
            'image_width': self.image_width,
            'masks': False  # Habilita o uso de máscaras
        }
        bev_embedding = {
            'sigma': 1.0,
            'bev_height': self.bev_resolution,
            'bev_width': self.bev_resolution,
            'h_meters': 100.0,
            'w_meters': 100.0,
            'offset': 0.0,
            'decoder_blocks': [128, 128, 64]
        }
        encoder_dim = 128


        encoder = Encoder(
            backbone=backbone,
            cross_view=cross_view,
            bev_embedding=bev_embedding,
            dim=encoder_dim,
            middle=[2],
            scale=1.0
        )

        decoder = Decoder(
            dim=encoder_dim,
            blocks=[128, 128, 64],
            residual=True,
            factor=2
        )

        self.generator = CrossViewTransformer(
            encoder=encoder,
            decoder=decoder,
            dim_output=3,
            dim_last=64
        )
        state_dict = torch.load(model_path)
        self.generator.load_state_dict(state_dict['network_state_dict'])
        self.generator.to(device)

        if use_eval:
            self.generator.eval()


    def __name__(self):
        return "cvt_3ch_L1"

    def _to_device(self, obs_dict):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in obs_dict.items()}

    def infer(self, obs_dict):
        """No-gradient inference. Returns binarized BEV (B, 3, 192, 192) with values in {0, 255}."""
        with torch.no_grad():
            batch = self._to_device(obs_dict)
            out = self.generator(batch)
            out = torch.nn.functional.interpolate(out, size=(192, 192), mode='bilinear', align_corners=False)
        return (out > 0.5).byte() * 255

    def forward_train(self, obs_dict):
        """Forward with gradients. Returns sigmoid output (B, 3, 192, 192) in [0, 1]."""
        batch = self._to_device(obs_dict)
        out = self.generator(batch)
        return torch.nn.functional.interpolate(out, size=(192, 192), mode='bilinear', align_corners=False)

    def compute_loss(self, pred, target):
        """L1Loss: pred is sigmoid output in [0, 1], target is in [0, 1]."""
        return torch.nn.functional.l1_loss(pred, target)

