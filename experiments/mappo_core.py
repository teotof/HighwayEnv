from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from experiments.attention_extractor import KinematicAttentionEncoder


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y <= 1e-8:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)


@dataclass
class MAPPOConfig:
    obs_dim: int
    state_dim: int
    action_dim: int
    num_agents: int
    obs_shape: tuple[int, ...] | None = None
    policy_kind: str = "baseline"
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    num_minibatches: int = 4
    clip_vloss: bool = True
    target_kl: float | None = None
    features_dim: int = 128
    d_model: int = 32
    attn_mode: str = "learned"
    x_index: int = 1
    vx_index: int = 3
    oracle_rule: str = "leader_x"
    ttc_eps: float = 1e-6


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(obs_dim, hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.net = _mlp(state_dim + num_agents, hidden_dim, 1)

    def forward(self, states: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        one_hot_ids = F.one_hot(agent_ids.long(), num_classes=self.num_agents).float()
        critic_in = torch.cat([states, one_hot_ids], dim=-1)
        return self.net(critic_in).squeeze(-1)


class MAPPOPolicy(nn.Module):
    expects_matrix_obs = False

    def __init__(
        self,
        config: MAPPOConfig,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()
        self.config = config
        self.actor = GaussianActor(config.obs_dim, config.action_dim, config.hidden_dim)
        self.critic = CentralizedCritic(
            config.state_dim,
            config.num_agents,
            config.hidden_dim,
        )

        self.register_buffer(
            "action_low",
            torch.as_tensor(action_low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(action_high, dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def act(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        agent_ids: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.actor.distribution(obs)
        raw_actions = dist.mean if deterministic else dist.sample()
        clipped_actions = torch.clamp(raw_actions, self.action_low, self.action_high)
        log_probs = dist.log_prob(raw_actions).sum(dim=-1)
        values = self.critic(states, agent_ids)
        return raw_actions, clipped_actions, log_probs, values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        agent_ids: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.actor.distribution(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropies = dist.entropy().sum(dim=-1)
        values = self.critic(states, agent_ids)
        return log_probs, entropies, values


class MAPPOAttentionPolicy(nn.Module):
    expects_matrix_obs = True

    def __init__(
        self,
        config: MAPPOConfig,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()
        if config.obs_shape is None:
            raise ValueError("MAPPOAttentionPolicy requires config.obs_shape.")

        self.config = config
        self.encoder = KinematicAttentionEncoder(
            obs_shape=config.obs_shape,
            features_dim=config.features_dim,
            d_model=config.d_model,
            mode=config.attn_mode,
            x_index=config.x_index,
            vx_index=config.vx_index,
            oracle_rule=config.oracle_rule,
            ttc_eps=config.ttc_eps,
        )
        self.actor_mean = nn.Linear(config.features_dim, config.action_dim)
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        self.critic = CentralizedCritic(
            config.state_dim,
            config.num_agents,
            config.hidden_dim,
        )

        self.register_buffer(
            "action_low",
            torch.as_tensor(action_low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(action_high, dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def last_attention(self):
        return self.encoder.last_attention

    def distribution(self, obs: torch.Tensor) -> Normal:
        features = self.encoder(obs)
        mean = self.actor_mean(features)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        agent_ids: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_actions = dist.mean if deterministic else dist.sample()
        clipped_actions = torch.clamp(raw_actions, self.action_low, self.action_high)
        log_probs = dist.log_prob(raw_actions).sum(dim=-1)
        values = self.critic(states, agent_ids)
        return raw_actions, clipped_actions, log_probs, values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        agent_ids: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropies = dist.entropy().sum(dim=-1)
        values = self.critic(states, agent_ids)
        return log_probs, entropies, values


def save_mappo_checkpoint(
    path: str | Path,
    *,
    policy: MAPPOPolicy,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(policy.config),
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_mappo_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    config = MAPPOConfig(**checkpoint["config"])
    metadata = checkpoint.get("metadata", {})
    action_low = np.asarray(metadata["action_low"], dtype=np.float32)
    action_high = np.asarray(metadata["action_high"], dtype=np.float32)
    policy_kind = metadata.get("policy_kind", getattr(config, "policy_kind", "baseline"))
    if policy_kind == "attn":
        policy = MAPPOAttentionPolicy(
            config,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
    else:
        policy = MAPPOPolicy(
            config,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy, metadata
