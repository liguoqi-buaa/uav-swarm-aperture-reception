from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn


class PhysicsInformedSTGNN(nn.Module):
    """Lightweight ST-GNN block used by the simulation pipeline.

    The module follows the paper design: spatial graph aggregation, temporal
    gated modeling, optional orbital-physics prior fusion, and H-step gradient
    output. The default controller uses the analytical physics predictor for
    reproducible table generation; this model is provided for PyTorch training
    and edge-inference replacement when training data or weights are available.
    """

    def __init__(
        self,
        node_feature_dim: int,
        physics_prior_dim: int,
        hidden_dim: int = 48,
        prediction_horizon: int = 5,
    ) -> None:
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.self_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.prior_proj = nn.Linear(physics_prior_dim, hidden_dim) if physics_prior_dim > 0 else None
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_horizon * 3),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        physics_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict future potential-field gradients.

        Args:
            node_features: Tensor shaped ``[batch, time, nodes, features]``.
            adjacency: Tensor shaped ``[batch, time, nodes, nodes]``.
            physics_prior: Optional tensor shaped ``[batch, prior_features]``.

        Returns:
            Tensor shaped ``[batch, horizon, nodes, 3]``.
        """
        batch, time_steps, nodes, _features = node_features.shape
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor_features = torch.matmul(adjacency, node_features) / degree
        x = torch.relu(self.self_proj(node_features) + self.neighbor_proj(neighbor_features))
        x = x.permute(0, 2, 1, 3).reshape(batch * nodes, time_steps, -1)
        _, hidden = self.temporal(x)
        hidden = hidden[-1].reshape(batch, nodes, -1)
        if self.prior_proj is not None and physics_prior is not None:
            hidden = hidden + torch.relu(self.prior_proj(physics_prior)).unsqueeze(1)
        gradient = self.output(hidden).reshape(batch, nodes, self.prediction_horizon, 3)
        return gradient.permute(0, 2, 1, 3).contiguous()


def model_size_mb(model: nn.Module) -> float:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    return total_params * 4.0 / (1024.0 * 1024.0)
