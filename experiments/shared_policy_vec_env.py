from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.vector.vector_env import AutoresetMode
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.util import dict_to_obs, obs_space_info


def _split_batched_info_value(value: Any, index: int) -> Any:
    if isinstance(value, dict):
        return _split_batched_info_dict(value, index)
    if isinstance(value, np.ndarray):
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _split_batched_info_dict(info: dict[str, Any], index: int) -> dict[str, Any]:
    world_info: dict[str, Any] = {}
    for key, value in info.items():
        if key.startswith("_"):
            continue

        mask = info.get(f"_{key}")
        if mask is not None and not bool(np.asarray(mask)[index]):
            continue

        world_info[key] = _split_batched_info_value(value, index)
    return world_info


class SharedPolicyVecEnv(VecEnv):
    """
    Expose one PPO slot per controlled vehicle while stepping each multi-agent
    world only once.

    The wrapped vector env batches worlds. Each world itself exposes a tuple of
    agent observations/actions through highway-env's MultiAgent* config. This
    adapter flattens that structure into the VecEnv API expected by SB3.
    """

    actions: np.ndarray

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        *,
        num_agents: int | None = None,
        vectorization_mode: str = "sync",
        copy: bool = True,
        shared_memory: bool = False,
    ) -> None:
        vectorization_mode = vectorization_mode.lower()
        if vectorization_mode == "sync":
            self.world_env = SyncVectorEnv(
                env_fns,
                copy=copy,
                autoreset_mode=AutoresetMode.DISABLED,
            )
        elif vectorization_mode == "async":
            self.world_env = AsyncVectorEnv(
                env_fns,
                copy=copy,
                shared_memory=shared_memory,
                autoreset_mode=AutoresetMode.DISABLED,
            )
        else:
            raise ValueError(
                f"Unknown vectorization_mode '{vectorization_mode}'. Use sync|async."
            )

        world_obs_space = self.world_env.single_observation_space
        world_action_space = self.world_env.single_action_space
        if not isinstance(world_obs_space, spaces.Tuple):
            raise ValueError(
                "SharedPolicyVecEnv expects a Tuple observation space from "
                "MultiAgentObservation."
            )
        if not isinstance(world_action_space, spaces.Tuple):
            raise ValueError(
                "SharedPolicyVecEnv expects a Tuple action space from "
                "MultiAgentAction."
            )

        self.num_worlds = self.world_env.num_envs
        self.num_agents = num_agents or len(world_obs_space.spaces)
        if len(world_obs_space.spaces) != self.num_agents:
            raise ValueError(
                "num_agents does not match the MultiAgentObservation tuple length."
            )
        if len(world_action_space.spaces) != self.num_agents:
            raise ValueError(
                "num_agents does not match the MultiAgentAction tuple length."
            )

        self.agent_observation_space = world_obs_space.spaces[0]
        self.agent_action_space = world_action_space.spaces[0]

        if any(space != self.agent_observation_space for space in world_obs_space.spaces):
            raise ValueError("All agent observation spaces must be identical.")
        if any(space != self.agent_action_space for space in world_action_space.spaces):
            raise ValueError("All agent action spaces must be identical.")

        super().__init__(
            num_envs=self.num_worlds * self.num_agents,
            observation_space=self.agent_observation_space,
            action_space=self.agent_action_space,
        )

        self.keys, shapes, dtypes = obs_space_info(self.observation_space)
        self.buf_obs = OrderedDict(
            [
                (
                    key,
                    np.zeros((self.num_envs, *tuple(shapes[key])), dtype=dtypes[key]),
                )
                for key in self.keys
            ]
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.metadata = self.world_env.metadata

    def _slot_index(self, world_index: int, agent_index: int) -> int:
        return world_index * self.num_agents + agent_index

    def _world_seeds(self) -> list[int | None]:
        return [
            self._seeds[self._slot_index(world_index, 0)]
            for world_index in range(self.num_worlds)
        ]

    def _world_options(self) -> dict[str, Any] | None:
        world_options = [
            self._options[self._slot_index(world_index, 0)]
            for world_index in range(self.num_worlds)
        ]
        non_empty = [options for options in world_options if options]
        if not non_empty:
            return None
        if any(options != non_empty[0] for options in non_empty[1:]):
            raise ValueError(
                "SharedPolicyVecEnv only supports one reset options dict per reset."
            )
        return deepcopy(non_empty[0])

    def _save_world_obs(self, world_obs: tuple[Any, ...]) -> None:
        for world_index in range(self.num_worlds):
            for agent_index in range(self.num_agents):
                slot_index = self._slot_index(world_index, agent_index)
                self._save_obs(slot_index, world_obs[agent_index][world_index])

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]  # type: ignore[index]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, deepcopy(self.buf_obs))

    def _to_world_actions(self, actions: np.ndarray) -> tuple[np.ndarray, ...]:
        world_actions = []
        for agent_index in range(self.num_agents):
            batch = [
                actions[self._slot_index(world_index, agent_index)]
                for world_index in range(self.num_worlds)
            ]
            world_actions.append(np.asarray(batch))
        return tuple(world_actions)

    def reset(self) -> VecEnvObs:
        world_obs, world_infos = self.world_env.reset(
            seed=self._world_seeds(),
            options=self._world_options(),
        )
        self._save_world_obs(world_obs)

        for world_index in range(self.num_worlds):
            info = _split_batched_info_dict(world_infos, world_index)
            for agent_index in range(self.num_agents):
                slot_index = self._slot_index(world_index, agent_index)
                self.reset_infos[slot_index] = {
                    **deepcopy(info),
                    "world_index": world_index,
                    "agent_index": agent_index,
                }

        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        world_obs, world_rewards, world_terminated, world_truncated, world_infos = (
            self.world_env.step(self._to_world_actions(self.actions))
        )
        world_dones = np.logical_or(world_terminated, world_truncated)

        self.buf_rews = np.repeat(world_rewards.astype(np.float32), self.num_agents)
        self.buf_dones = np.repeat(world_dones, self.num_agents)

        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for world_index in range(self.num_worlds):
            world_info = _split_batched_info_dict(world_infos, world_index)
            for agent_index in range(self.num_agents):
                slot_index = self._slot_index(world_index, agent_index)
                slot_info = {
                    **deepcopy(world_info),
                    "world_index": world_index,
                    "agent_index": agent_index,
                    "TimeLimit.truncated": bool(
                        world_truncated[world_index] and not world_terminated[world_index]
                    ),
                }
                if world_dones[world_index]:
                    slot_info["terminal_observation"] = deepcopy(
                        world_obs[agent_index][world_index]
                    )
                infos[slot_index] = slot_info

        if np.any(world_dones):
            reset_obs, reset_infos = self.world_env.reset(
                options={"reset_mask": world_dones}
            )
            world_obs = reset_obs

            for world_index in range(self.num_worlds):
                if not world_dones[world_index]:
                    continue
                info = _split_batched_info_dict(reset_infos, world_index)
                for agent_index in range(self.num_agents):
                    slot_index = self._slot_index(world_index, agent_index)
                    self.reset_infos[slot_index] = {
                        **deepcopy(info),
                        "world_index": world_index,
                        "agent_index": agent_index,
                    }

        self._save_world_obs(world_obs)
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(infos),
        )

    def close(self) -> None:
        self.world_env.close()

    def get_images(self) -> Sequence[np.ndarray | None]:
        images = self.world_env.render()
        if images is None:
            return [None for _ in range(self.num_worlds)]
        return list(images)

    def _slot_indices(self, indices: VecEnvIndices = None) -> list[int]:
        return list(self._get_indices(indices))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        slot_indices = self._slot_indices(indices)
        world_values = list(self.world_env.get_attr(attr_name))
        return [world_values[slot_index // self.num_agents] for slot_index in slot_indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        world_values = list(self.world_env.get_attr(attr_name))
        for slot_index in self._slot_indices(indices):
            world_values[slot_index // self.num_agents] = value
        self.world_env.set_attr(attr_name, world_values)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> list[Any]:
        slot_indices = self._slot_indices(indices)
        world_results = list(self.world_env.call(method_name, *method_args, **method_kwargs))
        return [world_results[slot_index // self.num_agents] for slot_index in slot_indices]

    def env_is_wrapped(
        self,
        wrapper_class: type[gym.Wrapper],
        indices: VecEnvIndices = None,
    ) -> list[bool]:
        return [False for _ in self._slot_indices(indices)]
