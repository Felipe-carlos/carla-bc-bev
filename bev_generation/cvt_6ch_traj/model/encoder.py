import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List

from bev_generation.cvt_6ch.model.encoder import (
    Encoder, BEVEmbedding, generate_grid, get_view_matrix,
    Normalize, ResNetBottleNeck,
)

N_WAYPOINTS = 5
SIGMA_M = 6.0  # σ in meters — ≈ 2 grid cells/σ at 256px/100m resolution


class WaypointMLP(nn.Module):
    """Maps each waypoint (x, y) in meters to a dim-dimensional feature vector."""
    def __init__(self, in_dim: int = 2, hidden: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, wp: torch.Tensor) -> torch.Tensor:
        return self.net(wp)


def build_bev_grid(
    h: int, w: int,
    bev_h: int, bev_w: int,
    h_meters: float, w_meters: float,
    offset: float = 0.0,
) -> torch.Tensor:
    """
    Returns (2, h, w) grid of ego-frame coordinates in meters, mirroring
    the internal BEVEmbedding.grid layout. Register as a non-persistent buffer.
    """
    xs = torch.linspace(0, 1, w) * bev_w
    ys = torch.linspace(0, 1, h) * bev_h
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), 0)   # (2, h, w)
    grid = F.pad(grid, (0, 0, 0, 0, 0, 1), value=1)                 # (3, h, w)

    sh = bev_h / h_meters
    sw = bev_w / w_meters
    V = torch.tensor([
        [ 0., -sw,               bev_w / 2.],
        [-sh,  0., bev_h * offset + bev_h / 2.],
        [ 0.,  0.,                          1.],
    ])
    grid = V.inverse() @ rearrange(grid, 'd h w -> d (h w)')         # (3, h*w)
    grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)             # (3, h, w)
    return grid[:2]                                                   # (2, h, w)


def traj_to_bev_signal(
    traj_norm: torch.Tensor,   # (B, n_waypoints*2) — waypoints ÷ 100 m
    mlp: WaypointMLP,
    bev_grid: torch.Tensor,    # (2, H, W) — ego-frame meters buffer
    n_waypoints: int = N_WAYPOINTS,
    sigma: float = SIGMA_M,
    dim: int = 128,
) -> torch.Tensor:
    """
    Converts raw trajectory into a (B, dim, H, W) signal to add to the BEV prior.
    For each waypoint: feat = MLP(wp_meters), gaussian = Gaussian_2D(wp, grid),
    signal += feat[:, :, None, None] * gaussian[:, None].
    """
    B = traj_norm.shape[0]
    H, W = bev_grid.shape[1], bev_grid.shape[2]

    wp_m = traj_norm.view(B, n_waypoints, 2) * 100.0                  # (B, 5, 2) in meters
    feats = mlp(wp_m.reshape(-1, 2)).view(B, n_waypoints, dim)         # (B, 5, dim)

    gx = bev_grid[0]   # (H, W)
    gy = bev_grid[1]   # (H, W)

    signal = torch.zeros(B, dim, H, W, device=traj_norm.device, dtype=traj_norm.dtype)
    for i in range(n_waypoints):
        wx = wp_m[:, i, 0]   # (B,)
        wy = wp_m[:, i, 1]   # (B,)
        dist_sq = (gx[None] - wx[:, None, None]) ** 2 + \
                  (gy[None] - wy[:, None, None]) ** 2          # (B, H, W)
        gaussian_i = torch.exp(-dist_sq / (2 * sigma ** 2))   # (B, H, W)
        signal = signal + feats[:, i, :, None, None] * gaussian_i[:, None]

    return signal   # (B, dim, H, W)


class TrajEncoder(Encoder):
    """Encoder subclass that injects trajectory waypoints into the BEV prior."""

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
    ):
        super().__init__(
            backbone=backbone,
            cross_view=cross_view,
            bev_embedding=bev_embedding,
            dim=dim,
            middle=middle,
            scale=scale,
        )
        self._dim = dim
        self._n_waypoints = n_waypoints
        self.sigma = sigma

        self.waypoint_mlp = WaypointMLP(in_dim=2, hidden=64, out_dim=dim)

        bev_grid = build_bev_grid(
            h=bev_embedding['bev_height'] // (2 ** len(bev_embedding['decoder_blocks'])),
            w=bev_embedding['bev_width']  // (2 ** len(bev_embedding['decoder_blocks'])),
            bev_h=bev_embedding['bev_height'],
            bev_w=bev_embedding['bev_width'],
            h_meters=bev_embedding['h_meters'],
            w_meters=bev_embedding['w_meters'],
            offset=bev_embedding.get('offset', 0.0),
        )
        self.register_buffer('bev_grid', bev_grid, persistent=False)   # (2, 32, 32)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()         # (dim, H_prior, W_prior)
        x = repeat(x, '... -> b ...', b=b)         # (B, dim, H_prior, W_prior)

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
