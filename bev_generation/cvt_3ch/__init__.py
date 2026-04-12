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

        if use_eval:
            self.generator.eval()


    def __name__(self):
        return "unet"
        
    def infer(self, expert_obs_dict):
        """
        Inferência sem gradiente (eval mode)

        Returns:
            bev tensor (B, C, H, W)
            expert_obs_dict precisa conter as chaves  'image', 'extrinsics','intrinsics'
        """
        with torch.no_grad():
            fake_birdview = self.generator(expert_obs_dict)
            # Resize: (4, 3, 256, 256) → (4, 3, 192, 192)
            fake_birdview = torch.nn.functional.interpolate(
            fake_birdview,
            size=(192, 192),
            mode='bilinear',   # melhor para imagens
            align_corners=False
            )

        fake_birdview = (fake_birdview > 0.5).byte()
        fake_birdview = fake_birdview * 255
        return fake_birdview

