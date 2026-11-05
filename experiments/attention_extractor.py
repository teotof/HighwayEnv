import math
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class KinematicAttentionEncoder(nn.Module):
    def __init__(
        self,
        obs_shape,
        features_dim: int = 128,
        d_model: int = 32,
        mode: str = "learned",
        x_index: int = 1,
        vx_index: int = 3,
        oracle_rule: str = "leader_x",
        ttc_eps: float = 1e-6,
    ):
        super().__init__()

        if len(obs_shape) != 2:
            raise ValueError(f"Expected obs shape (vehicles, features), got {obs_shape}")

        self.vehicles_count, self.feat_dim = obs_shape

        self.n_nei = self.vehicles_count - 1
        if self.n_nei <= 0:
            raise ValueError("vehicles_count must be >= 2 for attention over neighbours.")

        self.mode = str(mode).lower()
        if self.mode not in {"learned", "uniform", "oracle"}:
            raise ValueError(f"Unknown attention mode '{mode}'. Use learned|uniform|oracle.")
        self.x_index = int(x_index)
        self.vx_index = int(vx_index)
        self.oracle_rule = str(oracle_rule).lower()
        if self.oracle_rule not in {"leader_x", "ttc"}:
            raise ValueError(f"Unknown oracle_rule '{oracle_rule}'. Use leader_x|ttc.")
        self.ttc_eps = float(ttc_eps)

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
        if observations.dim() == 2:
            raise ValueError(f"Expected 3D obs (B,V,F), got {observations.shape}")

        x = observations
        ego = x[:, 0, :]
        nei = x[:, 1:, :]

        ego_e = self.embed(ego)
        nei_e = self.embed(nei)

        presence = nei[:, :, 0]
        mask = presence > 0.5
        mask_f = mask.float()

        nei_e = nei_e * mask_f.unsqueeze(-1)

        if self.mode == "uniform":
            attn = mask_f
            denom = attn.sum(dim=1, keepdim=True)
            attn = torch.where(denom > 0, attn / denom, torch.zeros_like(attn))

        elif self.mode == "oracle":
            dx = nei[:, :, self.x_index]
            dx_valid = dx.masked_fill(~mask, float("inf"))
            front = dx_valid > 0
            front_dx = torch.where(front, dx_valid, torch.full_like(dx_valid, float("inf")))
            has_front = torch.isfinite(front_dx).any(dim=1)
            idx_front = torch.argmin(front_dx, dim=1)
            idx_abs = torch.argmin(torch.abs(dx_valid), dim=1)
            fallback_idx = torch.where(has_front, idx_front, idx_abs)

            if self.oracle_rule == "ttc":
                dvx = nei[:, :, self.vx_index]
                closing = mask & ((dx * dvx) < 0)
                ttc = torch.abs(dx) / (torch.abs(dvx) + self.ttc_eps)
                ttc = ttc.masked_fill(~closing, float("inf"))
                has_ttc = torch.isfinite(ttc).any(dim=1)
                idx_ttc = torch.argmin(ttc, dim=1)
                oracle_idx = torch.where(has_ttc, idx_ttc, fallback_idx)
            else:
                oracle_idx = fallback_idx

            attn = torch.zeros_like(dx_valid)
            has_valid = mask.any(dim=1)
            if has_valid.any():
                row_idx = torch.nonzero(has_valid, as_tuple=False).squeeze(1)
                attn[row_idx, oracle_idx[row_idx]] = 1.0

        else:
            q = self.to_q(ego_e)
            k = self.to_k(nei_e)

            scores = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(k.shape[-1])
            scores = scores.masked_fill(~mask, -1e9)

            attn = torch.softmax(scores, dim=1)
            attn = attn * mask_f
            denom = attn.sum(dim=1, keepdim=True)
            attn = torch.where(denom > 0, attn / denom, torch.zeros_like(attn))

        v = self.to_v(nei_e)

        context = torch.einsum("bn,bnd->bd", attn, v)

        self.last_attention = attn.detach()

        features = torch.cat([ego_e, context], dim=1)
        return self.out(features)


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
        vx_index: int = 3,
        oracle_rule: str = "leader_x",
        ttc_eps: float = 1e-6,
    ):
        super().__init__(observation_space, features_dim)

        assert len(observation_space.shape) in (2, 1), f"Unexpected obs shape: {observation_space.shape}"

        if len(observation_space.shape) == 2:
            obs_shape = observation_space.shape
        else:
            raise ValueError("Flattened observation_space not supported in this extractor.")

        self.encoder = KinematicAttentionEncoder(
            obs_shape=obs_shape,
            features_dim=features_dim,
            d_model=d_model,
            mode=mode,
            x_index=x_index,
            vx_index=vx_index,
            oracle_rule=oracle_rule,
            ttc_eps=ttc_eps,
        )

    @property
    def last_attention(self):
        return self.encoder.last_attention

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)
