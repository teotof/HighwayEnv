from __future__ import annotations

import numpy as np
from gymnasium.envs.registration import register, registry

from highway_env import utils
from highway_env.envs.exit_env import ExitEnv
from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.merge_env import MergeEnv
from highway_env.vehicle.controller import ControlledVehicle


class ContinuousHighwayEnv(HighwayEnv):
    """
    HighwayEnv variant with continuous-control-friendly rewards.

    The stock HighwayEnv uses normalized rewards, which makes standing still far
    less costly than intended once raw continuous control is enabled. This
    variant keeps the road/traffic model but uses unnormalized rewards plus
    explicit reverse/standstill penalties.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "normalize_reward": False,
                "offroad_terminal": True,
                "reverse_penalty": -4.0,
                "stopped_penalty": -0.5,
                "stop_speed": 1.0,
            }
        )
        return cfg

    def _reward(self, action) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward *= rewards["on_road_reward"]
        return float(reward)

    def _rewards(self, action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        reverse_ratio = np.clip(
            max(-forward_speed, 0.0) / max(abs(self.config["reward_speed_range"][0]), 1.0),
            0.0,
            1.0,
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": float(lane / max(len(neighbours) - 1, 1)),
            "high_speed_reward": float(np.clip(scaled_speed, 0, 1)),
            "reverse_penalty": float(reverse_ratio),
            "stopped_penalty": float(abs(forward_speed) < self.config["stop_speed"]),
            "on_road_reward": float(self.vehicle.on_road),
        }


class ContinuousExitEnv(ExitEnv):
    """
    ExitEnv variant for raw continuous control.

    Key differences from the stock env:
    - reward can be left unnormalized
    - reward uses forward speed
    - episode terminates on successful exit
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "normalize_reward": False,
                "offroad_terminal": True,
                "reverse_penalty": -4.0,
                "stopped_penalty": -0.5,
                "stop_speed": 1.0,
            }
        )
        return cfg

    def _reward(self, action) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config.get("normalize_reward", False):
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["goal_reward"]],
                [0, 1],
            )
            reward = np.clip(reward, 0, 1)
        reward *= rewards["on_road_reward"]
        return float(reward)

    def _rewards(self, action) -> dict[str, float]:
        lane_index = (
            self.vehicle.target_lane_index
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index
        )
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        reverse_ratio = np.clip(
            max(-forward_speed, 0.0) / max(abs(self.config["reward_speed_range"][0]), 1.0),
            0.0,
            1.0,
        )
        right_lane = 0.0
        if isinstance(lane_index[-1], int):
            right_lane = lane_index[-1] / max(self.config["lanes_count"], 1)
        return {
            "collision_reward": float(self.vehicle.crashed),
            "goal_reward": float(self._is_success()),
            "high_speed_reward": float(np.clip(scaled_speed, 0, 1)),
            "right_lane_reward": float(right_lane),
            "reverse_penalty": float(reverse_ratio),
            "stopped_penalty": float(abs(forward_speed) < self.config["stop_speed"]),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        return (
            self.vehicle.crashed
            or self._is_success()
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )


class ContinuousMergeEnv(MergeEnv):
    """
    MergeEnv variant for raw continuous control.

    The stock MergeEnv always normalizes rewards and was designed around discrete
    meta-actions. This variant keeps the merge interaction but makes the reward
    consistent with continuous longitudinal control.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "normalize_reward": False,
                "offroad_terminal": True,
                "reverse_penalty": -2.0,
                "stopped_penalty": -0.4,
                "stop_speed": 1.0,
            }
        )
        return cfg

    def _reward(self, action) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config.get("normalize_reward", False):
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"] + self.config["merging_speed_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return float(reward)

    def _rewards(self, action) -> dict[str, float]:
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        reverse_ratio = np.clip(
            max(-forward_speed, 0.0) / max(abs(self.config["reward_speed_range"][0]), 1.0),
            0.0,
            1.0,
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": float(self.vehicle.lane_index[2] / 1),
            "high_speed_reward": float(np.clip(scaled_speed, 0, 1)),
            "merging_speed_reward": float(
                sum(
                    (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                    for vehicle in self.road.vehicles
                    if vehicle.lane_index == ("b", "c", 2)
                    and isinstance(vehicle, ControlledVehicle)
                )
            ),
            "reverse_penalty": float(reverse_ratio),
            "stopped_penalty": float(abs(forward_speed) < self.config["stop_speed"]),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        return (
            self.vehicle.crashed
            or bool(self.vehicle.position[0] > 370)
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]


def _register_once(env_id: str, entry_point: str) -> None:
    if env_id not in registry:
        register(id=env_id, entry_point=entry_point)


_register_once("continuous-highway-v0", "experiments.custom_envs:ContinuousHighwayEnv")
_register_once("continuous-exit-v0", "experiments.custom_envs:ContinuousExitEnv")
_register_once("continuous-merge-v0", "experiments.custom_envs:ContinuousMergeEnv")
