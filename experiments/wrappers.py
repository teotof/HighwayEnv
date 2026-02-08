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

    def observation(self, obs):
        if obs.ndim != 2:
            return obs  # fail-safe
        ego = obs[:1, :]
        nei = obs[1:, :]
        if nei.shape[0] <= 1:
            return obs
        perm = self._perm if (self._perm is not None) else self._make_perm(obs)
        nei_shuf = nei[perm, :]
        return np.concatenate([ego, nei_shuf], axis=0)


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
        self.leader = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        obs, info = self.env.reset(seed=seed, options=options)
        self.leader = self._find_leader()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += 1

        # refresh leader occasionally (traffic changes)
        if self.t % (self.period * 4) == 0:
            self.leader = self._find_leader()

        # every `period` steps, flip leader target speed
        if self.leader is not None and (self.t % self.period == 0):
            new_v = float(self.rng.uniform(self.v_low, self.v_high))
            # Try common vehicle attribute used in highway-env behaviour models
            if hasattr(self.leader, "target_speed"):
                self.leader.target_speed = new_v
            elif hasattr(self.leader, "speed"):
                # fallback (less realistic, but keeps the idea)
                if hasattr(self.leader, "target_speed"):
                    self.leader.target_speed = new_v
                self.leader.speed = new_v

        return obs, reward, terminated, truncated, info

    def _find_leader(self):
        env = self.env.unwrapped
        if not hasattr(env, "road") or not hasattr(env.road, "vehicles"):
            return None
        vehicles = env.road.vehicles
        if not vehicles:
            return None

        ego = getattr(env, "vehicle", None)
        if ego is None or not hasattr(ego, "position"):
            return None

        ego_x = float(ego.position[0])
        best = None
        best_dx = None

        for v in vehicles:
            if v is ego or not hasattr(v, "position"):
                continue
            dx = float(v.position[0]) - ego_x
            if dx > 0.0 and (best_dx is None or dx < best_dx):
                best_dx = dx
                best = v

        return best
