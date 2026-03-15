import numpy as np
import gymnasium as gym


class ShuffleNeighboursObs(gym.ObservationWrapper):
    """
    Keeps row 0 (ego) fixed, randomly permutes rows 1..N (neighbour slots).
    Works with obs shaped (vehicles_count, features_dim), e.g. (15, 5).
    """
    def __init__(self, env, shuffle_each_step: bool = True):
        super().__init__(env)
        self.shuffle_each_step = shuffle_each_step
        self.rng = np.random.default_rng(0)
        self._perm = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._perm = None
        obs, info = self.env.reset(seed=seed, options=options)
        if not self.shuffle_each_step:
            self._perm = self._make_perm(obs)
        return self.observation(obs), info

    def _make_perm(self, obs):
        n = obs.shape[0] - 1
        perm = self.rng.permutation(n)
        return perm

    def _shuffle_matrix(self, obs):
        if obs.ndim != 2:
            return obs  # fail-safe
        ego = obs[:1, :]
        nei = obs[1:, :]
        if nei.shape[0] <= 1:
            return obs
        perm = self._perm if (self._perm is not None) else self._make_perm(obs)
        nei_shuf = nei[perm, :]
        return np.concatenate([ego, nei_shuf], axis=0)

    def observation(self, obs):
        if isinstance(obs, tuple):
            return tuple(self._shuffle_matrix(agent_obs) for agent_obs in obs)
        return self._shuffle_matrix(obs)


class StopGoLeaderWrapper(gym.Wrapper):
    """
    Periodically changes the target_speed of the closest front vehicle.
    This creates stop-and-go behaviour in front of ego.
    """
    def __init__(self, env, period: int = 25, v_low: float = 10.0, v_high: float = 30.0):
        super().__init__(env)
        self.period = period
        self.v_low = v_low
        self.v_high = v_high
        self.t = 0
        self.rng = np.random.default_rng(0)
        self.leaders = []

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        obs, info = self.env.reset(seed=seed, options=options)
        self.leaders = self._find_leaders()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += 1

        # refresh leaders occasionally (traffic changes)
        if self.t % (self.period * 4) == 0:
            self.leaders = self._find_leaders()

        # every `period` steps, flip leader target speed
        if self.t % self.period == 0:
            updated_ids = set()
            for leader in self.leaders:
                if leader is None:
                    continue
                leader_id = id(leader)
                if leader_id in updated_ids:
                    continue
                updated_ids.add(leader_id)
                new_v = float(self.rng.uniform(self.v_low, self.v_high))
                # Try common vehicle attribute used in highway-env behaviour models
                if hasattr(leader, "target_speed"):
                    leader.target_speed = new_v
                elif hasattr(leader, "speed"):
                    leader.speed = new_v

        return obs, reward, terminated, truncated, info

    def _find_leaders(self):
        env = self.env.unwrapped
        if not hasattr(env, "road") or not hasattr(env.road, "vehicles"):
            return []
        vehicles = env.road.vehicles
        if not vehicles:
            return []

        egos = getattr(env, "controlled_vehicles", None) or [getattr(env, "vehicle", None)]
        leaders = []

        for ego in egos:
            if ego is None or not hasattr(ego, "position"):
                leaders.append(None)
                continue

            ego_x = float(ego.position[0])
            best = None
            best_dx = None

            for vehicle in vehicles:
                if vehicle is ego or not hasattr(vehicle, "position"):
                    continue
                dx = float(vehicle.position[0]) - ego_x
                if dx > 0.0 and (best_dx is None or dx < best_dx):
                    best_dx = dx
                    best = vehicle

            leaders.append(best)

        return leaders
