def make_cf_config(obs_vehicles: int, duration: int = 100):
    return {
        "duration": duration,
        "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": False},
        "observation": {
            "type": "Kinematics",
            "vehicles_count": obs_vehicles,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
        },
    }

SCENARIOS = {
    # Scenario 0: current
    "cf_v2": {
        "end_id": "highway-v0",
        "config": make_cf_config(obs_vehicles=5, duration=100),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 1: more neighbours
    "cf_obs15": {
        "end_id": "highway-v0",
        "config": make_cf_config(obs_vehicles=15, duration=100),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 2: shuffled neighbour ordering
    "cf_obs15_shuffle": {
        "end_id": "highway-v0",
        "config": make_cf_config(obs_vehicles=15, duration=100),
        "wrapper": "ShuffleNeighboursObs",  # resolved in train scripts
        "wrapper_kwargs": {"shuffle_each_step": True},
    },
    # Scenario 3: stop-and-go lead vehicle
    "cf_obs15_stopgo": {
        "end_id": "highway-v0",
        "config": make_cf_config(obs_vehicles=15, duration=140),
        "wrapper": "StopGoLeaderWrapper",
        "wrapper_kwargs": {"period": 25, "v_low": 10.0, "v_high": 30.0},
    },
    # Scenario 4: merge dense
    "merge_dense": {
    "env_id": "merge-v0",
    "config": {
        "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": False},
        "observation": {"type": "Kinematics", "vehicles_count": 15, "features": ["presence","x","y","vx","vy"], "absolute": False},
        "duration": 140,
    },
    "wrapper": None,
    "wrapper_kwargs": {},
    },
    # Scenario 5: lanechange dense
    "highway_lanechange_dense": {
    "env_id": "highway-v0",
    "config": {
      "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": True},
      "observation": {"type": "Kinematics", "vehicles_count": 15, "features": ["presence","x","y","vx","vy"], "absolute": False}, "duration": 140,
    },
    "wrapper": None,
    "wrapper_kwargs": {},
    },
}
