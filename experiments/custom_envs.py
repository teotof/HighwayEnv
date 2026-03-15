from __future__ import annotations

import numpy as np
from gymnasium.envs.registration import register, registry

from highway_env import utils
from highway_env.envs.exit_env import ExitEnv
from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.merge_env import MergeEnv
from highway_env.vehicle.controller import ControlledVehicle


def _mean_reward_dict(reward_dicts: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([reward_dict[name] for reward_dict in reward_dicts]))
        for name in reward_dicts[0]
    }


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
        if len(self.controlled_vehicles) <= 1:
            return self._agent_reward(self.vehicle)
        return float(
            np.mean(
                [self._agent_reward(vehicle) for vehicle in self.controlled_vehicles]
            )
        )

    def _rewards(self, action) -> dict[str, float]:
        if len(self.controlled_vehicles) <= 1:
            return self._agent_rewards(self.vehicle)
        return _mean_reward_dict(
            [self._agent_rewards(vehicle) for vehicle in self.controlled_vehicles]
        )

    def _agent_rewards(self, vehicle) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = (
            vehicle.target_lane_index[2]
            if isinstance(vehicle, ControlledVehicle)
            else vehicle.lane_index[2]
        )
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        reverse_ratio = np.clip(
            max(-forward_speed, 0.0) / max(abs(self.config["reward_speed_range"][0]), 1.0),
            0.0,
            1.0,
        )
        return {
            "collision_reward": float(vehicle.crashed),
            "right_lane_reward": float(lane / max(len(neighbours) - 1, 1)),
            "high_speed_reward": float(np.clip(scaled_speed, 0, 1)),
            "reverse_penalty": float(reverse_ratio),
            "stopped_penalty": float(abs(forward_speed) < self.config["stop_speed"]),
            "on_road_reward": float(vehicle.on_road),
        }

    def _agent_reward(self, vehicle) -> float:
        rewards = self._agent_rewards(vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward *= rewards["on_road_reward"]
        return float(reward)

    def _agent_is_terminated(self, vehicle) -> bool:
        return vehicle.crashed or (
            self.config["offroad_terminal"] and not vehicle.on_road
        )

    def _is_terminated(self) -> bool:
        if len(self.controlled_vehicles) <= 1:
            return super()._is_terminated()
        return any(
            self._agent_is_terminated(vehicle) for vehicle in self.controlled_vehicles
        )

    def _info(self, obs, action=None) -> dict:
        info = super()._info(obs, action)
        if len(self.controlled_vehicles) > 1:
            info["agents_rewards"] = tuple(
                self._agent_reward(vehicle) for vehicle in self.controlled_vehicles
            )
            info["agents_terminated"] = tuple(
                self._agent_is_terminated(vehicle)
                for vehicle in self.controlled_vehicles
            )
            info["agents_crashed"] = tuple(
                bool(vehicle.crashed) for vehicle in self.controlled_vehicles
            )
        return info


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
        vehicles = self.controlled_vehicles or [self.vehicle]
        reward = float(np.mean([self._agent_reward(vehicle) for vehicle in vehicles]))
        if self.config.get("normalize_reward", False):
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["goal_reward"]],
                [0, 1],
            )
            reward = np.clip(reward, 0, 1)
        return float(reward)

    def _rewards(self, action) -> dict[str, float]:
        if len(self.controlled_vehicles) <= 1:
            return self._agent_rewards(self.vehicle)
        return _mean_reward_dict(
            [self._agent_rewards(vehicle) for vehicle in self.controlled_vehicles]
        )

    def _agent_rewards(self, vehicle) -> dict[str, float]:
        lane_index = (
            vehicle.target_lane_index
            if isinstance(vehicle, ControlledVehicle)
            else vehicle.lane_index
        )
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
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
            "collision_reward": float(vehicle.crashed),
            "goal_reward": float(self._agent_is_success(vehicle)),
            "high_speed_reward": float(np.clip(scaled_speed, 0, 1)),
            "right_lane_reward": float(right_lane),
            "reverse_penalty": float(reverse_ratio),
            "stopped_penalty": float(abs(forward_speed) < self.config["stop_speed"]),
            "on_road_reward": float(vehicle.on_road),
        }

    def _agent_reward(self, vehicle) -> float:
        rewards = self._agent_rewards(vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward *= rewards["on_road_reward"]
        return float(reward)

    def _agent_is_success(self, vehicle) -> bool:
        lane_index = (
            vehicle.target_lane_index
            if isinstance(vehicle, ControlledVehicle)
            else vehicle.lane_index
        )
        return lane_index == (
            "1",
            "2",
            self.config["lanes_count"],
        ) or lane_index == ("2", "exit", 0)

    def _agent_has_failed(self, vehicle) -> bool:
        return vehicle.crashed or (
            self.config["offroad_terminal"] and not vehicle.on_road
        )

    def _agent_is_terminated(self, vehicle) -> bool:
        return self._agent_has_failed(vehicle) or self._agent_is_success(vehicle)

    def _is_terminated(self) -> bool:
        if len(self.controlled_vehicles) <= 1:
            return (
                self.vehicle.crashed
                or self._agent_is_success(self.vehicle)
                or (self.config["offroad_terminal"] and not self.vehicle.on_road)
            )
        return any(
            self._agent_has_failed(vehicle) for vehicle in self.controlled_vehicles
        ) or all(
            self._agent_is_success(vehicle) for vehicle in self.controlled_vehicles
        )

    def _info(self, obs, action=None) -> dict:
        info = super()._info(obs, action)
        if len(self.controlled_vehicles) <= 1:
            info["is_success"] = self._agent_is_success(self.vehicle)
            return info

        info["is_success"] = all(
            self._agent_is_success(vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_success"] = tuple(
            self._agent_is_success(vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_rewards"] = tuple(
            self._agent_reward(vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminated(vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_crashed"] = tuple(
            bool(vehicle.crashed) for vehicle in self.controlled_vehicles
        )
        return info


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
                "controlled_vehicles": 1,
            }
        )
        return cfg

    def _reward(self, action) -> float:
        vehicles = self.controlled_vehicles or [self.vehicle]
        reward = float(np.mean([self._agent_reward(vehicle) for vehicle in vehicles]))
        if self.config.get("normalize_reward", False):
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"] + self.config["merging_speed_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        return float(reward)

    def _rewards(self, action) -> dict[str, float]:
        if len(self.controlled_vehicles) <= 1:
            return self._agent_rewards(self.vehicle)
        return _mean_reward_dict(
            [self._agent_rewards(vehicle) for vehicle in self.controlled_vehicles]
        )

    def _agent_rewards(self, vehicle) -> dict[str, float]:
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        reverse_ratio = np.clip(
            max(-forward_speed, 0.0) / max(abs(self.config["reward_speed_range"][0]), 1.0),
            0.0,
            1.0,
        )
        return {
            "collision_reward": float(vehicle.crashed),
            "right_lane_reward": float(vehicle.lane_index[2] / 1),
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
            "on_road_reward": float(vehicle.on_road),
        }

    def _agent_reward(self, vehicle) -> float:
        rewards = self._agent_rewards(vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward *= rewards["on_road_reward"]
        return float(reward)

    def _agent_has_finished(self, vehicle) -> bool:
        return bool(vehicle.position[0] > 370)

    def _agent_has_failed(self, vehicle) -> bool:
        return vehicle.crashed or (
            self.config["offroad_terminal"] and not vehicle.on_road
        )

    def _agent_is_terminated(self, vehicle) -> bool:
        return self._agent_has_failed(vehicle) or self._agent_has_finished(vehicle)

    def _is_terminated(self) -> bool:
        if len(self.controlled_vehicles) <= 1:
            return (
                self.vehicle.crashed
                or self._agent_has_finished(self.vehicle)
                or (self.config["offroad_terminal"] and not self.vehicle.on_road)
            )
        return any(
            self._agent_has_failed(vehicle) for vehicle in self.controlled_vehicles
        ) or all(
            self._agent_has_finished(vehicle) for vehicle in self.controlled_vehicles
        )

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _make_vehicles(self) -> None:
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        controlled_vehicles = int(self.config.get("controlled_vehicles", 1))

        if controlled_vehicles <= 1:
            ego_vehicle = self.action_type.vehicle_class(
                road,
                road.network.get_lane(("a", "b", 1)).position(30.0, 0.0),
                speed=30.0,
            )
            road.vehicles.append(ego_vehicle)
            self.vehicle = ego_vehicle

            for position, speed in [(90.0, 29.0), (70.0, 31.0), (5.0, 31.5)]:
                lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
                position = lane.position(
                    position + self.np_random.uniform(-5.0, 5.0), 0.0
                )
                speed += self.np_random.uniform(-1.0, 1.0)
                road.vehicles.append(other_vehicles_type(road, position, speed=speed))

            merging_vehicle = other_vehicles_type(
                road,
                road.network.get_lane(("j", "k", 0)).position(110.0, 0.0),
                speed=20.0,
            )
            merging_vehicle.target_speed = 30.0
            road.vehicles.append(merging_vehicle)
            return

        self.controlled_vehicles = []
        controlled_positions = [30.0 + 28.0 * idx for idx in range(controlled_vehicles)]
        for idx, longitudinal in enumerate(controlled_positions):
            lane_id = 1 if idx % 2 == 0 else 0
            lane = road.network.get_lane(("a", "b", lane_id))
            controlled_vehicle = self.action_type.vehicle_class(
                road,
                lane.position(longitudinal, 0.0),
                speed=30.0,
            )
            road.vehicles.append(controlled_vehicle)
            self.controlled_vehicles.append(controlled_vehicle)

        for position, speed in [(145.0, 29.0), (120.0, 31.0), (90.0, 31.5), (60.0, 30.0)]:
            lane = road.network.get_lane(("a", "b", int(self.np_random.integers(2))))
            candidate_position = lane.position(
                position + self.np_random.uniform(-5.0, 5.0), 0.0
            )
            if any(
                np.linalg.norm(candidate_position - vehicle.position) < 20.0
                for vehicle in self.controlled_vehicles
            ):
                continue

            candidate_speed = speed + self.np_random.uniform(-1.0, 1.0)
            road.vehicles.append(
                other_vehicles_type(road, candidate_position, speed=candidate_speed)
            )

        merging_vehicle = other_vehicles_type(
            road,
            road.network.get_lane(("j", "k", 0)).position(110.0, 0.0),
            speed=20.0,
        )
        merging_vehicle.target_speed = 30.0
        road.vehicles.append(merging_vehicle)

    def _info(self, obs, action=None) -> dict:
        info = super()._info(obs, action)
        if len(self.controlled_vehicles) > 1:
            info["agents_rewards"] = tuple(
                self._agent_reward(vehicle) for vehicle in self.controlled_vehicles
            )
            info["agents_finished"] = tuple(
                self._agent_has_finished(vehicle)
                for vehicle in self.controlled_vehicles
            )
            info["agents_terminated"] = tuple(
                self._agent_is_terminated(vehicle)
                for vehicle in self.controlled_vehicles
            )
            info["agents_crashed"] = tuple(
                bool(vehicle.crashed) for vehicle in self.controlled_vehicles
            )
        return info


def _register_once(env_id: str, entry_point: str) -> None:
    if env_id not in registry:
        register(id=env_id, entry_point=entry_point)


_register_once("continuous-highway-v0", "experiments.custom_envs:ContinuousHighwayEnv")
_register_once("continuous-exit-v0", "experiments.custom_envs:ContinuousExitEnv")
_register_once("continuous-merge-v0", "experiments.custom_envs:ContinuousMergeEnv")
