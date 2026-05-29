from bev_generation.cvt_6ch.model.decoder import Decoder
from bev_generation.cvt_6ch.model.cvt import CrossViewTransformer
from bev_generation.cvt_6ch.model.config import Config
from .encoder import TrajCmdEncoder, N_WAYPOINTS, SIGMA_M


class ModelBuilderTrajCmd():
    def __init__(
        self,
        masks=False,
        reduction=4,
        backbone=None,
        low_stride=False,
        decoder=None,
        dim_output=6,
        n_waypoints=N_WAYPOINTS,
        sigma=SIGMA_M,
        n_cmd_classes=6,
    ):
        config = Config(masks=masks, reduction=reduction, low_stride=low_stride)
        backbone = backbone if backbone is not None else config.backbone

        encoder = TrajCmdEncoder(
            backbone=backbone,
            cross_view=config.cross_view,
            bev_embedding=config.bev_embedding,
            dim=config.encoder_dim,
            middle=[2],
            scale=1.0,
            n_waypoints=n_waypoints,
            sigma=sigma,
            n_cmd_classes=n_cmd_classes,
        )

        if decoder is None:
            decoder = Decoder(
                dim=config.encoder_dim,
                blocks=[128, 128, 64],
                residual=True,
                factor=2,
            )

        self.network = CrossViewTransformer(
            encoder=encoder,
            decoder=decoder,
            dim_output=dim_output,
            dim_last=64,
        )

    def get_net(self):
        return self.network
