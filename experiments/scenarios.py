def make_cf_config(obs_vehicles: int):
    # Matches current config, only parameterising vehicles_count
    return {
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
        "config": make_cf_config(obs_vehicles=5),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 1: more neighbours
    "cf_obs15": {
        "config": make_cf_config(obs_vehicles=15),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 2: shuffled neighbour ordering
    "cf_obs15_shuffle": {
        "config": make_cf_config(obs_vehicles=15),
        "wrapper": "ShuffleNeighboursObs",  # resolved in train scripts
        "wrapper_kwargs": {"shuffle_each_step": True},
    },
    # Scenario 3: stop-and-go lead vehicle
    "cf_obs15_stopgo": {
        "config": make_cf_config(obs_vehicles=15),
        "wrapper": "StopGoLeaderWrapper",
        "wrapper_kwargs": {"period": 25, "v_low": 10.0, "v_high": 30.0},
    },
}
