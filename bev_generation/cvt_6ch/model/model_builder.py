from .encoder import Encoder
from .decoder import Decoder
from .cvt import CrossViewTransformer
from .config import Config, ConfigLarge

class ModelBuilder():
    def __init__(self,masks=False,reduction=4,backbone=None,encoder=None,low_stride=False, decoder=None,dim_output=6):
        config = Config(masks=masks,reduction=reduction,low_stride=low_stride)
        backbone = backbone if backbone is not None else config.backbone
        if not encoder:
            encoder = Encoder(
                backbone=backbone,
                cross_view=config.cross_view,
                bev_embedding=config.bev_embedding,
                dim=config.encoder_dim,
                middle=[2],
                scale=1.0
            )
        

        if decoder is None:
            decoder = Decoder(
                dim=config.encoder_dim,
                blocks=[128, 128, 64],
                residual=True,
                factor=2
                )
        else:
            decoder = decoder
        

        self.network = CrossViewTransformer(
            encoder=encoder,
            decoder=decoder,
            dim_output=dim_output,
            dim_last=64
        )
    def get_net(self):
        return self.network
   
class ModelBuilderLarger():
    def __init__(self,masks=False,reduction=4,backbone=None,low_stride=False, decoder=None,dim_output=6):
        config = ConfigLarge(masks=masks,reduction=reduction,low_stride=low_stride)
        backbone = backbone if backbone is not None else config.backbone
        encoder = Encoder(
            backbone=backbone,
            cross_view=config.cross_view,
            bev_embedding=config.bev_embedding,
            dim=config.encoder_dim,
            middle=[2],
            scale=1.0
        )

        if decoder is None:
            decoder = Decoder(
                dim=config.encoder_dim,
                blocks=[256, 256, 128],
                residual=True,
                factor=2
                )
        else:
            decoder = decoder
        

        self.network = CrossViewTransformer(
            encoder=encoder,
            decoder=decoder,
            dim_output=dim_output,
            dim_last=64
        )
    def get_net(self):
        return self.network
   