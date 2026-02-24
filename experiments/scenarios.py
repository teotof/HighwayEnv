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


def make_highway_continuous_config(
    obs_vehicles: int,
    *,
    duration: int = 100,
    lateral: bool = False,
    traffic_vehicles: int | None = None,
    vehicles_density: float | None = None,
    lanes_count: int | None = None,
):
    """
    Highway-v0 config with continuous control and kinematics observations.
    Adds optional traffic density knobs used for Step-C "attention-favourable" setups.
    """
    cfg = make_cf_config(obs_vehicles=obs_vehicles, duration=duration)
    cfg["action"]["lateral"] = lateral

    if traffic_vehicles is not None:
        cfg["vehicles_count"] = traffic_vehicles
    if vehicles_density is not None:
        cfg["vehicles_density"] = vehicles_density
    if lanes_count is not None:
        cfg["lanes_count"] = lanes_count

    return cfg

SCENARIOS = {
    # Scenario 0: current
    "cf_v2": {
        "env_id": "highway-v0",
        "config": make_cf_config(obs_vehicles=5, duration=100),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 1: more neighbours
    "cf_obs15": {
        "env_id": "highway-v0",
        "config": make_cf_config(obs_vehicles=15, duration=100),
        "wrapper": None,
        "wrapper_kwargs": {},
    },
    # Scenario 2: shuffled neighbour ordering
    "cf_obs15_shuffle": {
        "env_id": "highway-v0",
        "config": make_cf_config(obs_vehicles=15, duration=100),
        "wrapper": "ShuffleNeighboursObs",  # resolved in train scripts
        "wrapper_kwargs": {"shuffle_each_step": True},
    },
    # Scenario 3: stop-and-go lead vehicle
    "cf_obs15_stopgo": {
        "env_id": "highway-v0",
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

    # Scenario 6: car following with 20 observed neighbours
    # Step-C Scenario A: more observed neighbours + denser traffic (sorted order)
    # Use as train side of a shape-matched generalization test against cf_obs20_dense_shuffle
    "cf_obs20_dense": {
        "env_id": "highway-v0",
        "config": make_highway_continuous_config(
            obs_vehicles=20,
            duration=120,
            lateral=False,
            traffic_vehicles=80,
            vehicles_density=2.0,
            lanes_count=4,
        ),
        "wrapper": None,
        "wrapper_kwargs": {},
    },

    # Scenario 7:
    # Step-C Scenario B: same as cf_obs20_dense, but neighbour order is shuffled each step
    "cf_obs20_dense_shuffle": {
        "env_id": "highway-v0",
        "config": make_highway_continuous_config(
            obs_vehicles=20,
            duration=120,
            lateral=False,
            traffic_vehicles=80,
            vehicles_density=2.0,
            lanes_count=4,
        ),
        "wrapper": "ShuffleNeighboursObs",
        "wrapper_kwargs": {"shuffle_each_step": True},
    },

    # Scenario 8:
    # Step-C Scenario C: denser traffic + more neighbours + stop-go perturbation (dynamic relevance)
    "cf_obs20_dense_stopgo": {
        "env_id": "highway-v0",
        "config": make_highway_continuous_config(
            obs_vehicles=20,
            duration=160,
            lateral=False,
            traffic_vehicles=80,
            vehicles_density=2.0,
            lanes_count=4,
        ),
        "wrapper": "StopGoLeaderWrapper",
        "wrapper_kwargs": {"period": 20, "v_low": 8.0, "v_high": 32.0},
    },

    # Scenario 9:
    # Step-C Scenario D (sorted): interaction-heavy lane-change with more observed vehicles
    # Useful as a train-side counterpart for the shuffled version below
    "highway_lanechange_dense_obs20": {
        "env_id": "highway-v0",
        "config": make_highway_continuous_config(
            obs_vehicles=20,
            duration=160,
            lateral=True,
            traffic_vehicles=90,
            vehicles_density=2.0,
            lanes_count=4,
        ),
        "wrapper": None,
        "wrapper_kwargs": {},
    },

    # Scenario 10:
    # Step-C Scenario E: interaction-heavy + more neighbours + shuffled ordering
    "highway_lanechange_dense_obs20_shuffle": {
        "env_id": "highway-v0",
        "config": make_highway_continuous_config(
            obs_vehicles=20,
            duration=160,
            lateral=True,
            traffic_vehicles=90,
            vehicles_density=2.0,
            lanes_count=4,
        ),
        "wrapper": "ShuffleNeighboursObs",
        "wrapper_kwargs": {"shuffle_each_step": True},
    },
}
