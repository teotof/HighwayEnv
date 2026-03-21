from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_seeds(raw: str) -> list[int]:
    if not raw:
        return [0, 1, 2]
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    return seeds if seeds else [0, 1, 2]


def mappo_run_name(
    scenario_name: str,
    exp_version: str,
    controlled_vehicles: int,
    seed: int,
) -> str:
    return (
        f"mappo_{scenario_name}_{exp_version}_"
        f"cv{controlled_vehicles}_seed{seed}"
    )


def mappo_attention_run_name(
    scenario_name: str,
    exp_version: str,
    controlled_vehicles: int,
    attn_mode: str,
    seed: int,
) -> str:
    return (
        f"mappo_attn_{attn_mode}_{scenario_name}_{exp_version}_"
        f"cv{controlled_vehicles}_seed{seed}"
    )


def resolve_mappo_model_path(
    model_dir: str | Path,
    scenario_name: str,
    exp_version: str,
    controlled_vehicles: int,
    seed: int,
    *,
    model_kind: str = "baseline",
    attn_mode: str = "learned",
) -> Path:
    model_dir = Path(model_dir)
    model_kind = model_kind.strip().lower()

    if model_kind == "baseline":
        run_name = mappo_run_name(
            scenario_name,
            exp_version,
            controlled_vehicles,
            seed,
        )
    elif model_kind == "attn":
        run_name = mappo_attention_run_name(
            scenario_name,
            exp_version,
            controlled_vehicles,
            attn_mode,
            seed,
        )
    else:
        raise ValueError(f"Unknown MAPPO model_kind '{model_kind}'. Use baseline|attn.")

    return model_dir / f"{run_name}.pt"


def flatten_obs_batch(obs_batch: np.ndarray) -> np.ndarray:
    obs_batch = np.asarray(obs_batch, dtype=np.float32)
    if obs_batch.ndim < 2:
        raise ValueError("Expected batched observations with at least 2 dimensions.")
    return obs_batch.reshape(obs_batch.shape[0], -1)


def build_centralized_inputs(
    obs_batch: np.ndarray,
    num_agents: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_obs = flatten_obs_batch(obs_batch)
    num_slots, obs_dim = local_obs.shape
    if num_slots % num_agents != 0:
        raise ValueError("Number of agent slots must be divisible by num_agents.")

    num_worlds = num_slots // num_agents
    per_world_obs = local_obs.reshape(num_worlds, num_agents, obs_dim)
    world_states = per_world_obs.reshape(num_worlds, num_agents * obs_dim)
    state_batch = np.repeat(world_states, num_agents, axis=0)
    agent_ids = np.tile(np.arange(num_agents, dtype=np.int64), num_worlds)
    return local_obs, state_batch.astype(np.float32), agent_ids


def extract_slot_rewards(
    rewards: np.ndarray,
    infos: list[dict],
    *,
    reward_mode: str,
) -> np.ndarray:
    reward_mode = reward_mode.strip().lower()
    rewards = np.asarray(rewards, dtype=np.float32)

    if reward_mode == "team":
        return rewards

    if reward_mode != "agent":
        raise ValueError(f"Unknown reward_mode '{reward_mode}'. Use team|agent.")

    slot_rewards = []
    for reward, info in zip(rewards, infos):
        agent_index = info.get("agent_index")
        agents_rewards = info.get("agents_rewards")
        if agent_index is None or agents_rewards is None:
            slot_rewards.append(float(reward))
            continue
        slot_rewards.append(float(agents_rewards[agent_index]))
    return np.asarray(slot_rewards, dtype=np.float32)


def world_metrics_from_slots(
    slot_values: np.ndarray,
    num_agents: int,
) -> np.ndarray:
    slot_values = np.asarray(slot_values)
    if slot_values.shape[0] % num_agents != 0:
        raise ValueError("slot_values length must be divisible by num_agents.")
    return slot_values.reshape(-1, num_agents)
