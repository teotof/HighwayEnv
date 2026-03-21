from __future__ import annotations

import os

import numpy as np
import torch

from experiments.mappo_core import load_mappo_checkpoint
from experiments.mappo_utils import build_centralized_inputs
from experiments.marl_utils import make_shared_world_env


def auto_max_steps(env) -> int:
    cfg = env.unwrapped.config
    duration_s = float(cfg.get("duration", 40))
    policy_hz = float(cfg.get("policy_frequency", 1))
    return max(1, int(round(duration_s * policy_hz)))


def predict_mappo_actions(
    policy,
    obs: tuple[np.ndarray, ...],
    *,
    deterministic: bool = True,
) -> tuple[np.ndarray, ...]:
    obs_batch = np.stack(obs, axis=0)
    local_obs, states, agent_ids = build_centralized_inputs(
        obs_batch,
        num_agents=len(obs),
    )

    device = policy.device
    with torch.no_grad():
        _, clipped_actions, _, _ = policy.act(
            torch.as_tensor(local_obs, dtype=torch.float32, device=device),
            torch.as_tensor(states, dtype=torch.float32, device=device),
            torch.as_tensor(agent_ids, dtype=torch.long, device=device),
            deterministic=deterministic,
        )

    actions_np = clipped_actions.cpu().numpy()
    return tuple(actions_np[agent_index] for agent_index in range(len(obs)))


def eval_mappo_model(
    model_path: str,
    *,
    scenario_name: str,
    controlled_vehicles: int,
    n_eval_episodes: int,
    max_steps: int,
    seed: int,
    device: str = "cpu",
) -> dict[str, float]:
    env = make_shared_world_env(
        scenario_name=scenario_name,
        controlled_vehicles=controlled_vehicles,
    )
    policy, _ = load_mappo_checkpoint(model_path, device=device)
    eval_max_steps = max_steps if max_steps > 0 else auto_max_steps(env)

    world_reward_returns = []
    joint_returns = []
    agent_return_means = []
    lengths = []
    world_crashes = 0
    agent_crashes = 0
    world_successes = 0

    for episode in range(n_eval_episodes):
        obs, info = env.reset(seed=seed * 1000 + episode)
        terminated = truncated = False
        world_reward_return = 0.0
        agent_returns = np.zeros((controlled_vehicles,), dtype=np.float32)
        episode_steps = 0

        while not (terminated or truncated) and episode_steps < eval_max_steps:
            action = predict_mappo_actions(policy, obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            world_reward_return += float(reward)

            if "agents_rewards" in info:
                agent_returns += np.asarray(info["agents_rewards"], dtype=np.float32)
            else:
                agent_returns += float(reward)

            episode_steps += 1

        world_reward_returns.append(world_reward_return)
        joint_returns.append(float(agent_returns.sum()))
        agent_return_means.append(float(agent_returns.mean()))
        lengths.append(episode_steps)

        crashed = [
            bool(getattr(vehicle, "crashed", False))
            for vehicle in env.unwrapped.controlled_vehicles
        ]
        world_crashes += int(any(crashed))
        agent_crashes += int(sum(crashed))

        if bool(info.get("is_success", False)):
            world_successes += 1

        if episode % 5 == 0:
            print(
                f"[{os.path.basename(model_path)}] seed={seed} "
                f"ep={episode}/{n_eval_episodes}",
                flush=True,
            )

    env.close()

    return {
        "mean_world_reward_return": float(np.mean(world_reward_returns)),
        "std_world_reward_return": float(np.std(world_reward_returns)),
        "mean_joint_return": float(np.mean(joint_returns)),
        "mean_agent_return": float(np.mean(agent_return_means)),
        "mean_len": float(np.mean(lengths)),
        "world_crash_rate": float(world_crashes / n_eval_episodes),
        "agent_crash_rate": float(
            agent_crashes / max(n_eval_episodes * controlled_vehicles, 1)
        ),
        "world_success_rate": float(world_successes / n_eval_episodes),
        "mean_team_return": float(np.mean(world_reward_returns)),
    }


def summarize_mappo_results(label: str, results: list[dict[str, float]]) -> None:
    metrics = [
        ("mean_world_reward_return", "world_reward_return"),
        ("mean_joint_return", "joint_return"),
        ("mean_agent_return", "agent_return"),
        ("world_crash_rate", "world_crash_rate"),
        ("agent_crash_rate", "agent_crash_rate"),
        ("world_success_rate", "world_success_rate"),
        ("mean_len", "mean_len"),
    ]

    print(f"\n{label}")
    for metric_key, metric_label in metrics:
        values = [result[metric_key] for result in results]
        print(f"  {metric_label}: {np.mean(values):.3f} +/- {np.std(values):.3f}")
