import gymnasium as gym
import highway_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from attention_extractor import KinematicAttentionExtractor
from experiments.scenarios import SCENARIOS
from experiments.wrappers import ShuffleNeighboursObs, StopGoLeaderWrapper

ENV_ID = "highway-v0"

import os
SCENARIO_NAME = os.getenv("SCENARIO_NAME", "cf_v2")
scenario = SCENARIOS[SCENARIO_NAME]
ATTN_MODE = os.getenv("ATTN_MODE", "learned")
X_INDEX = int(os.getenv("X_INDEX", "1"))
VX_INDEX = int(os.getenv("VX_INDEX", "3"))
ORACLE_RULE = os.getenv("ORACLE_RULE", "leader_x")
TTC_EPS = float(os.getenv("TTC_EPS", "1e-6"))

ENV_ID = scenario["env_id"]
ENV_CONFIG = scenario["config"]

# Wrapper selection (0 or 1 wrapper. simple for SubprocVecEnv on Windows)
WRAPPER_NAME = scenario.get("wrapper", None)
WRAPPER_KWARGS = scenario.get("wrapper_kwargs", {})

WRAPPER_MAP = {
    None: None,
    "ShuffleNeighboursObs": ShuffleNeighboursObs,
    "StopGoLeaderWrapper": StopGoLeaderWrapper,
}
WRAPPER_CLASS = WRAPPER_MAP.get(WRAPPER_NAME)

EXP_VERSION = os.getenv("EXP_VERSION", "v2")

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "500000"))
N_ENVS = int(os.getenv("N_ENVS", "4"))

if __name__ == "__main__":
    for seed in SEEDS:
        run_name = f"ppo_attn_{ATTN_MODE}_{SCENARIO_NAME}_{EXP_VERSION}_seed{seed}"

        # Sanity Check
        test_env = gym.make(ENV_ID, config=ENV_CONFIG)
        if WRAPPER_CLASS is not None:
            test_env = WRAPPER_CLASS(test_env, **WRAPPER_KWARGS)
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

        policy_kwargs = dict(
            features_extractor_class=KinematicAttentionExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                d_model=32,
                mode=ATTN_MODE,
                x_index=X_INDEX,
                vx_index=VX_INDEX,
                oracle_rule=ORACLE_RULE,
                ttc_eps=TTC_EPS,
            ),
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log="runs/tb",
            device="cpu",
            seed=seed,
            policy_kwargs=policy_kwargs,
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=run_name)
        model.save(f"runs/models/{run_name}")
        env.close()
