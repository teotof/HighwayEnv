# Reproduced from https://github.com/Farama-Foundation/HighwayEnv.git

import gymnasium as gym
import highway_env  # registers envs

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_ID = "highway-v0"

ENV_CONFIG = {
    "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": False},
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
    },
}

RUN_NAME = "ppo_baseline_cf_v1"  # This has to change per run

# Parallel environments (Must be under __main__)
if __name__ == "__main__":

    # Sanity check. Confirm config is applied
    test_env = gym.make(ENV_ID, config=ENV_CONFIG)
    obs, _ = test_env.reset()
    print("obs shape:", obs.shape)
    print("action space:", test_env.action_space)
    test_env.close()

    env = make_vec_env(
        ENV_ID,
        n_envs=4,                 # 4 or 8
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": ENV_CONFIG},
        monitor_dir=f"runs/monitor/{RUN_NAME}",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="runs/tb",
        device="cpu",
    )

    model.learn(total_timesteps=50_000, tb_log_name=RUN_NAME)
    model.save(f"runs/models/{RUN_NAME}")
