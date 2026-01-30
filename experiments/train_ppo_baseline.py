# Reproduced from https://github.com/Farama-Foundation/HighwayEnv.git

import gymnasium as gym
import highway_env  # registers envs

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.scenarios import SCENARIOS
from experiments.wrappers import ShuffleNeighboursObs, StopGoLeaderWrapper

ENV_ID = "highway-v0"

SCENARIO_NAME = "cf_obs15_shuffle"
ENV_CONFIG = SCENARIOS[SCENARIO_NAME]["config"]

# Wrapper selection (0 or 1 wrapper. simple for SubprocVecEnv on Windows)
WRAPPER_NAME = SCENARIOS[SCENARIO_NAME]["wrapper"]
WRAPPER_KWARGS = SCENARIOS[SCENARIO_NAME]["wrapper_kwargs"]

WRAPPER_MAP = {
    None: None,
    "ShuffleNeighboursObs": ShuffleNeighboursObs,
    "StopGoLeaderWrapper": StopGoLeaderWrapper,
}
WRAPPER_CLASS = WRAPPER_MAP[WRAPPER_NAME]

EXP_ID = f"{SCENARIO_NAME}_v1"
BASE_RUN_NAME = f"ppo_baseline_{EXP_ID}"

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4

# Parallel environments (Must be under __main__)
if __name__ == "__main__":
    for seed in SEEDS:
        run_name = f"{BASE_RUN_NAME}_seed{seed}"

        # Sanity Check
        test_env = gym.make(ENV_ID, config=ENV_CONFIG)
        obs, _ = test_env.reset(seed=seed)
        print(run_name, "obs shape:", obs.shape, "action space:", test_env.action_space)
        test_env.close()

        env = make_vec_env(
            ENV_ID,
            n_envs=N_ENVS,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={"config": ENV_CONFIG},
            monitor_dir=f"runs/monitor/{run_name}",
            seed=seed,
            wrapper_class=WRAPPER_CLASS,
            wrapper_kwargs=WRAPPER_KWARGS,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log="runs/tb",
            device="cpu",
            seed=seed,
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=run_name)
        model.save(f"runs/models/{run_name}")
        env.close()
