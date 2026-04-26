from pathlib import Path
import numpy as np
import pandas as pd
import torch as th
from sklearn.neighbors import KernelDensity


def load_kde(dataset_dir: str, routes, n_eps: int = 1, bandwidth: float = 0.2) -> KernelDensity:
    """
    Fits a KernelDensity on all expert actions loaded from episode.json files.
    Call once at training startup and reuse the returned object — do not refit in the loop.
    """
    dataset_path = Path(dataset_dir)
    all_actions = []

    for route_idx in routes:
        for ep_idx in range(n_eps):
            json_path = dataset_path / f'route_{route_idx:02d}' / f'ep_{ep_idx:02d}' / 'episode.json'
            if not json_path.exists():
                continue
            df = pd.read_json(json_path)
            for i in range(len(df)):
                all_actions.append(df.iloc[i]['actions'])

    if not all_actions:
        raise ValueError(f"No expert actions found in '{dataset_dir}'. Check dataset_dir and routes.")

    actions_np = np.array(all_actions, dtype=np.float64)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(actions_np)
    print(f"[KDE] Fitted on {len(all_actions)} expert actions from '{dataset_dir}'.")
    return kde


def compute_kde_weights(kde: KernelDensity, actions: th.Tensor, epsilon: float = 1e-6) -> th.Tensor:
    """
    Inverse-density importance weights for a batch of actions.
    Rare actions (low density) receive high weight; common actions receive low weight.
    Weights are normalized to mean=1 so loss magnitude stays stable across batches.

    Args:
        kde:     Fitted KernelDensity object.
        actions: Tensor (B, action_dim) in the policy action space.
        epsilon: Numerical stability constant.

    Returns:
        weights: Float tensor (B,) on the same device as `actions`.
    """
    device = actions.device
    actions_np = actions.detach().cpu().numpy().astype(np.float64)
    log_density = kde.score_samples(actions_np)
    density = np.exp(log_density)
    weights = 1.0 / (density + epsilon)
    weights = weights / (weights.mean() + epsilon)
    return th.as_tensor(weights, dtype=th.float32).to(device)
