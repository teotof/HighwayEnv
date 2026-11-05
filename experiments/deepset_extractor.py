import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DeepSetExtractor(BaseFeaturesExtractor):
    """
    Permutation-invariant neighbour encoder.

    Each neighbour is embedded with the same network (phi), pooled with a sum,
    then combined with the ego features through a second network (rho).
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        hidden: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        if len(observation_space.shape) != 2:
            raise ValueError(f"Expected obs shape (vehicles, features), got {observation_space.shape}")

        vehicles_count, feat_dim = observation_space.shape
        if vehicles_count < 2:
            raise ValueError("vehicles_count must be >= 2 for DeepSets over neighbours.")

        self.vehicles_count = vehicles_count
        self.feat_dim = feat_dim

        self.phi = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden + feat_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() != 3:
            raise ValueError(f"Expected 3D obs (B,V,F), got {observations.shape}")

        ego = observations[:, 0, :]      # (B, F)
        nei = observations[:, 1:, :]     # (B, N, F)

        presence = (nei[:, :, 0] > 0.5).float()
        nei = nei * presence.unsqueeze(-1)

        h = self.phi(nei)                # (B, N, H)
        pooled = h.sum(dim=1)            # (B, H)

        features = torch.cat([ego, pooled], dim=1)
        return self.rho(features)
