import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import List

from bev_generation.cvt_6ch_traj.model.encoder import (
    TrajEncoder, traj_to_bev_signal, N_WAYPOINTS, SIGMA_M,
)


class TrajCmdEncoder(TrajEncoder):
    """TrajEncoder + FiLM conditioning of the navigation command on backbone features."""

    def __init__(
        self,
        backbone,
        cross_view: dict,
        bev_embedding: dict,
        dim: int = 128,
        middle: List[int] = [2],
        scale: float = 1.0,
        n_waypoints: int = N_WAYPOINTS,
        sigma: float = SIGMA_M,
        n_cmd_classes: int = 6,
    ):
        super().__init__(
            backbone=backbone,
            cross_view=cross_view,
            bev_embedding=bev_embedding,
            dim=dim,
            middle=middle,
            scale=scale,
            n_waypoints=n_waypoints,
            sigma=sigma,
        )
        self.cmd_emb = nn.Embedding(n_cmd_classes, dim)

        feat_dims = [self.down(torch.zeros(s)).shape[1] for s in self.backbone.output_shapes]
        self.cmd_film = nn.ModuleList([nn.Linear(dim, 2 * fd) for fd in feat_dims])

        # Zero-init: model starts as identity w.r.t. cmd and learns to use it gradually
        for layer in self.cmd_film:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        if 'cmd' in batch:
            cmd_idx = batch['cmd'].argmax(dim=-1)    # (B,)
            cmd_feat = self.cmd_emb(cmd_idx)          # (B, dim)
            modulated = []
            for feat, film_layer in zip(features, self.cmd_film):
                params = film_layer(cmd_feat)         # (B, 2*feat_dim)
                gamma, beta = params.chunk(2, dim=-1) # (B, feat_dim) each
                gamma = gamma[:, None].expand(-1, n, -1).reshape(b * n, -1, 1, 1)
                beta  =  beta[:, None].expand(-1, n, -1).reshape(b * n, -1, 1, 1)
                modulated.append((1.0 + gamma) * feat + beta)
            features = modulated

        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)

        if 'traj' in batch:
            traj_signal = traj_to_bev_signal(
                batch['traj'], self.waypoint_mlp, self.bev_grid,
                n_waypoints=self._n_waypoints, sigma=self.sigma, dim=self._dim,
            )
            x = x + traj_signal

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)

        return x
