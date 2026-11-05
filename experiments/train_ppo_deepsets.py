import os

import gymnasium as gym
import highway_env  # registers envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from deepset_extractor import DeepSetExtractor
from experiments.scenarios import SCENARIOS
from experiments.wrappers import ShuffleNeighboursObs, StopGoLeaderWrapper

SCENARIO_NAME = os.getenv("SCENARIO_NAME", "cf_v2")
scenario = SCENARIOS[SCENARIO_NAME]

ENV_ID = scenario["env_id"]
ENV_CONFIG = scenario["config"]

WRAPPER_NAME = scenario.get("wrapper", None)
WRAPPER_KWARGS = scenario.get("wrapper_kwargs", {})

WRAPPER_MAP = {
    None: None,
    "ShuffleNeighboursObs": ShuffleNeighboursObs,
    "StopGoLeaderWrapper": StopGoLeaderWrapper,
}
WRAPPER_CLASS = WRAPPER_MAP.get(WRAPPER_NAME)

EXP_VERSION = os.getenv("EXP_VERSION", "v9")
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "500000"))
N_ENVS = int(os.getenv("N_ENVS", "4"))
DEEPSETS_HIDDEN = int(os.getenv("DEEPSETS_HIDDEN", "64"))


def parse_seeds(raw: str):
    if not raw:
        return [0, 1, 2]
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    return seeds if seeds else [0, 1, 2]


SEEDS = parse_seeds(os.getenv("SEEDS", "0,1,2"))


if __name__ == "__main__":
    for seed in SEEDS:
        run_name = f"ppo_deepsets_{SCENARIO_NAME}_{EXP_VERSION}_seed{seed}"

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
            features_extractor_class=DeepSetExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                hidden=DEEPSETS_HIDDEN,
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
