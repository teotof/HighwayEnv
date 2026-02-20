import math
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class KinematicAttentionExtractor(BaseFeaturesExtractor):
    """
    Expects observations shaped either:
      - (batch, vehicles_count, features_dim)  e.g. (B, 5, 5)
      - or (batch, vehicles_count*features_dim) (flattened), which we reshape.
    Row 0 = ego, rows 1.. = neighbour "slots".
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        d_model: int = 32,
        mode: str = "learned",
        x_index: int = 1,
    ):
        super().__init__(observation_space, features_dim)

        assert len(observation_space.shape) in (2, 1), f"Unexpected obs shape: {observation_space.shape}"

        if len(observation_space.shape) == 2:
            self.vehicles_count, self.feat_dim = observation_space.shape
        else:
            # Flattened case: vehicles_count * feat_dim
            # If use a flat obs space -> set manually
            raise ValueError("Flattened observation_space not supported in this extractor.")

        self.n_nei = self.vehicles_count - 1
        if self.n_nei <= 0:
            raise ValueError("vehicles_count must be >= 2 for attention over neighbours.")

        self.mode = str(mode).lower()
        if self.mode not in {"learned", "uniform", "oracle"}:
            raise ValueError(f"Unknown attention mode '{mode}'. Use learned|uniform|oracle.")
        self.x_index = int(x_index)

        # Embed each vehicle feature vector -> d_model
        self.embed = nn.Linear(self.feat_dim, d_model)

        # Single-head scaled dot-product attention: q from ego, k/v from neighbours
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        # Final projection to SB3 features_dim
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

        self.last_attention = None  # (batch, n_nei)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Observations: (B, V, F)
        if observations.dim() == 2:
            # If feed flattened obs by mistake, this will fail loudly
            raise ValueError(f"Expected 3D obs (B,V,F), got {observations.shape}")

        x = observations
        ego = x[:, 0, :]           # (B, F)
        nei = x[:, 1:, :]          # (B, N, F)

        ego_e = self.embed(ego)    # (B, D)
        nei_e = self.embed(nei)    # (B, N, D)

        # presence == 1 means slot is occupied, 0 means empty
        presence = nei[:, :, 0]  # (B, N)
        mask = presence > 0.5
        mask_f = mask.float()

        # Empty neighbour slots should not contribute to keys/values.
        nei_e = nei_e * mask_f.unsqueeze(-1)

        if self.mode == "uniform":
            attn = mask_f
            denom = attn.sum(dim=1, keepdim=True)
            attn = torch.where(denom > 0, attn / denom, torch.zeros_like(attn))

        elif self.mode == "oracle":
            # Leader heuristic: nearest in front (min positive dx), else nearest by |dx|.
            dx = nei[:, :, self.x_index]  # (B, N)
            dx_valid = dx.masked_fill(~mask, float("inf"))

            front = dx_valid > 0
            front_dx = torch.where(front, dx_valid, torch.full_like(dx_valid, float("inf")))
            has_front = torch.isfinite(front_dx).any(dim=1)

            idx_front = torch.argmin(front_dx, dim=1)          # (B,)
            idx_abs = torch.argmin(torch.abs(dx_valid), dim=1) # (B,)
            leader_idx = torch.where(has_front, idx_front, idx_abs)

            attn = torch.zeros_like(dx_valid)
            has_valid = mask.any(dim=1)
            if has_valid.any():
                row_idx = torch.nonzero(has_valid, as_tuple=False).squeeze(1)
                attn[row_idx, leader_idx[row_idx]] = 1.0

        else:
            q = self.to_q(ego_e)       # (B, D)
            k = self.to_k(nei_e)       # (B, N, D)

            # Scores: (B, N)
            scores = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(k.shape[-1])
            scores = scores.masked_fill(~mask, -1e9)

            attn = torch.softmax(scores, dim=1)
            # Renormalize so fully-masked rows become all-zeros.
            attn = attn * mask_f
            denom = attn.sum(dim=1, keepdim=True)
            attn = torch.where(denom > 0, attn / denom, torch.zeros_like(attn))

        # Need value projection for all modes.
        v = self.to_v(nei_e)       # (B, N, D)

        # Context: (B, D)
        context = torch.einsum("bn,bnd->bd", attn, v)

        self.last_attention = attn.detach()  # store for evaluation/visualisation

        # Combine ego embedding + neighbour context
        features = torch.cat([ego_e, context], dim=1)  # (B, 2D)
        return self.out(features)
