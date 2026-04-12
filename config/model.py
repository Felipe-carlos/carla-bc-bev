from model.efficientnet import EfficientNetExtractor

class Config():
    def __init__(self):
        self.image_height = 224  ##
        self.image_width = 480  ##
        self.bev_resolution = 256


        self.backbone = EfficientNetExtractor(
        layer_names=['reduction_4'],
        image_height=self.image_height,
        image_width=self.image_width,
        model_name='efficientnet-b4'
        )
        self.cross_view = {
        'heads': 4,
        'dim_head': 32,
        'qkv_bias': True,
        'skip': True,
        'no_image_features': False,
        'image_height': self.image_height,
        'image_width': self.image_width,
        'masks': False  # Habilita o uso de máscaras
        }
        self.cross_view_masks = {
        'heads': 4,
        'dim_head': 32,
        'qkv_bias': True,
        'skip': True,
        'no_image_features': False,
        'image_height': self.image_height,
        'image_width': self.image_width,
        'masks': True  # Habilita o uso de máscaras
        }
        self.bev_embedding = {
        'sigma': 1.0,
        'bev_height': self.bev_resolution,
        'bev_width': self.bev_resolution,
        'h_meters': 100.0,
        'w_meters': 100.0,
        'offset': 0.0,
        'decoder_blocks': [128, 128, 64]
        }
        self.encoder_dim = 128
config = Config()