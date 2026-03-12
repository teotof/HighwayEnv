from __future__ import annotations

import math

from experiments import custom_envs  # noqa: F401  # register custom env ids

OBS_FEATURES = ["presence", "x", "y", "vx", "vy"]


def make_continuous_action(*, longitudinal: bool, lateral: bool) -> dict:
    action = {
        "type": "ContinuousAction",
        "longitudinal": longitudinal,
        "lateral": lateral,
        "speed_range": [0.0, 30.0],
        "acceleration_range": [-3.0, 3.0],
    }
    if lateral:
        action["steering_range"] = [-math.pi / 8, math.pi / 8]
    return action


def make_kinematics_observation(obs_vehicles: int) -> dict:
    return {
        "type": "Kinematics",
        "vehicles_count": obs_vehicles,
        "features": OBS_FEATURES,
        "absolute": False,
    }


def make_cf_config(
    obs_vehicles: int,
    *,
    duration: int,
    traffic_vehicles: int,
    vehicles_density: float,
) -> dict:
    return {
        "duration": duration,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "lanes_count": 1,
        "controlled_vehicles": 1,
        "initial_lane_id": 0,
        "ego_spacing": 2.0,
        "vehicles_count": traffic_vehicles,
        "vehicles_density": vehicles_density,
        "collision_reward": -10.0,
        "right_lane_reward": 0.0,
        "high_speed_reward": 1.0,
        "lane_change_reward": 0.0,
        "reward_speed_range": [8.0, 30.0],
        "normalize_reward": False,
        "offroad_terminal": True,
        "reverse_penalty": -4.0,
        "stopped_penalty": -0.5,
        "stop_speed": 1.0,
        "action": make_continuous_action(longitudinal=True, lateral=False),
        "observation": make_kinematics_observation(obs_vehicles),
    }


def make_merge_config(obs_vehicles: int, *, duration: int) -> dict:
    return {
        "duration": duration,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "collision_reward": -10.0,
        "right_lane_reward": 0.0,
        "high_speed_reward": 0.8,
        "reward_speed_range": [8.0, 30.0],
        "merging_speed_reward": -1.0,
        "lane_change_reward": 0.0,
        "normalize_reward": False,
        "offroad_terminal": True,
        "reverse_penalty": -4.0,
        "stopped_penalty": -0.4,
        "stop_speed": 1.0,
        "action": make_continuous_action(longitudinal=True, lateral=False),
        "observation": make_kinematics_observation(obs_vehicles),
    }


def make_exit_config(
    obs_vehicles: int,
    *,
    duration: int,
    traffic_vehicles: int,
    vehicles_density: float,
) -> dict:
    return {
        "duration": duration,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "lanes_count": 4,
        "controlled_vehicles": 1,
        "vehicles_count": traffic_vehicles,
        "vehicles_density": vehicles_density,
        "collision_reward": -10.0,
        "goal_reward": 8.0,
        "high_speed_reward": 0.5,
        "right_lane_reward": 0.3,
        "reward_speed_range": [10.0, 30.0],
        "normalize_reward": False,
        "offroad_terminal": True,
        "reverse_penalty": -4.0,
        "stopped_penalty": -0.5,
        "stop_speed": 1.0,
        "action": make_continuous_action(longitudinal=True, lateral=True),
        "observation": make_kinematics_observation(obs_vehicles),
    }


SCENARIOS = {
    # Scenario 0: one-lane car-following with a small neighbour set.
    "cf_v2": {
        "env_id": "continuous-highway-v0",
        "config": make_cf_config(
            obs_vehicles=5,
            duration=25,
            traffic_vehicles=12,
            vehicles_density=1.0,
        ),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 1: one-lane car-following with more observed neighbours.
    "cf_obs15": {
        "env_id": "continuous-highway-v0",
        "config": make_cf_config(
            obs_vehicles=15,
            duration=30,
            traffic_vehicles=24,
            vehicles_density=1.1,
        ),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 2: same car-following task, but neighbour slots are shuffled.
    "cf_obs15_shuffle": {
        "env_id": "continuous-highway-v0",
        "config": make_cf_config(
            obs_vehicles=15,
            duration=30,
            traffic_vehicles=24,
            vehicles_density=1.1,
        ),
        "wrapper": "ShuffleNeighboursObs",
        "wrapper_kwargs": {"shuffle_each_step": True},
    },
    # Scenario 3: stop-and-go lead vehicle in one-lane car-following.
    "cf_obs15_stopgo": {
        "env_id": "continuous-highway-v0",
        "config": make_cf_config(
            obs_vehicles=15,
            duration=35,
            traffic_vehicles=24,
            vehicles_density=1.1,
        ),
        "wrapper": "StopGoLeaderWrapper",
        "wrapper_kwargs": {"period": 20, "v_low": 8.0, "v_high": 24.0},
    },
    # Scenario 4: longitudinal merge negotiation with no reverse/standstill exploit.
    "merge_dense": {
        "env_id": "continuous-merge-v0",
        "config": make_merge_config(obs_vehicles=15, duration=25),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 5: goal-based lane-change/exit task, not free-form lane looping.
    "highway_lanechange_dense": {
        "env_id": "continuous-exit-v0",
        "config": make_exit_config(
            obs_vehicles=15,
            duration=30,
            traffic_vehicles=28,
            vehicles_density=1.6,
        ),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 6: harder one-lane car-following with more distractors.
    "cf_obs20_dense": {
        "env_id": "continuous-highway-v0",
        "config": make_cf_config(
            obs_vehicles=20,
            duration=35,
            traffic_vehicles=30,
            vehicles_density=1.3,
        ),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 7: same as cf_obs20_dense but shuffled each step.
    "cf_obs20_dense_shuffle": {
        "env_id": "continuous-highway-v0",
        "config": make_cf_config(
            obs_vehicles=20,
            duration=35,
            traffic_vehicles=30,
            vehicles_density=1.3,
        ),
        "wrapper": "ShuffleNeighboursObs",
        "wrapper_kwargs": {"shuffle_each_step": True},
    },
    # Scenario 8: harder stop-and-go car-following with 20 observed vehicles.
    "cf_obs20_dense_stopgo": {
        "env_id": "continuous-highway-v0",
        "config": make_cf_config(
            obs_vehicles=20,
            duration=40,
            traffic_vehicles=30,
            vehicles_density=1.3,
        ),
        "wrapper": "StopGoLeaderWrapper",
        "wrapper_kwargs": {"period": 20, "v_low": 6.0, "v_high": 24.0},
    },
    # Scenario 9: harder goal-based lane change with more observed vehicles.
    "highway_lanechange_dense_obs20": {
        "env_id": "continuous-exit-v0",
        "config": make_exit_config(
            obs_vehicles=20,
            duration=35,
            traffic_vehicles=36,
            vehicles_density=1.8,
        ),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 10: same task, but neighbour slots are shuffled.
    "highway_lanechange_dense_obs20_shuffle": {
        "env_id": "continuous-exit-v0",
        "config": make_exit_config(
            obs_vehicles=20,
            duration=35,
            traffic_vehicles=36,
            vehicles_density=1.8,
        ),
        "wrapper": "ShuffleNeighboursObs",
        "wrapper_kwargs": {"shuffle_each_step": True},
    },
}
