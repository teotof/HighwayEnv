from __future__ import annotations

from functools import partial

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from stable_baselines3.common.monitor import Monitor

from experiments.scenarios import get_scenario
from experiments.wrappers import ShuffleNeighboursObs, StopGoLeaderWrapper


WRAPPER_MAP = {
    None: None,
    "ShuffleNeighboursObs": ShuffleNeighboursObs,
    "StopGoLeaderWrapper": StopGoLeaderWrapper,
}


def shared_run_name(
    scenario_name: str,
    exp_version: str,
    controlled_vehicles: int,
    seed: int,
) -> str:
    return (
        f"ppo_shared_{scenario_name}_{exp_version}_"
        f"cv{controlled_vehicles}_seed{seed}"
    )


def make_shared_world_env(
    scenario_name: str,
    controlled_vehicles: int,
    render_mode: str | None = None,
    monitor_path: str | None = None,
):
    scenario = get_scenario(
        scenario_name,
        controlled_vehicles=controlled_vehicles,
        multi_agent=True,
    )
    env = gym.make(
        scenario["env_id"],
        config=scenario["config"],
        render_mode=render_mode,
    )

    wrapper_name = scenario.get("wrapper")
    wrapper_kwargs = scenario.get("wrapper_kwargs", {})
    wrapper_class = WRAPPER_MAP.get(wrapper_name)
    if wrapper_class is not None:
        env = wrapper_class(env, **wrapper_kwargs)

    if monitor_path:
        env = Monitor(env, filename=monitor_path)

    return env


def make_shared_world_env_factory(
    scenario_name: str,
    controlled_vehicles: int,
    render_mode: str | None = None,
    monitor_path: str | None = None,
):
    return partial(
        make_shared_world_env,
        scenario_name=scenario_name,
        controlled_vehicles=controlled_vehicles,
        render_mode=render_mode,
        monitor_path=monitor_path,
    )


def predict_shared_actions(model, obs: tuple[np.ndarray, ...], deterministic: bool = True):
    obs_batch = np.stack(obs, axis=0)
    actions, _ = model.predict(obs_batch, deterministic=deterministic)
    return tuple(actions[agent_index] for agent_index in range(len(obs)))


def sample_shared_actions(action_space: gym.Space):
    if not isinstance(action_space, gym.spaces.Tuple):
        raise ValueError("Expected a Tuple action space for shared-policy MARL.")
    return tuple(space.sample() for space in action_space.spaces)
