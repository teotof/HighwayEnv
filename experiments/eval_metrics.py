import os
import numpy as np
import gymnasium as gym
import highway_env  # registers envs
from stable_baselines3 import PPO

from experiments.scenarios import SCENARIOS
from experiments.wrappers import ShuffleNeighboursObs, StopGoLeaderWrapper

# Which models to load (training) vs which env to test in (evaluation)
TRAIN_SCENARIO_NAME = os.getenv("TRAIN_SCENARIO_NAME", os.getenv("SCENARIO_NAME", "cf_obs15_shuffle"))
EVAL_SCENARIO_NAME  = os.getenv("EVAL_SCENARIO_NAME", TRAIN_SCENARIO_NAME)

EXP_VERSION = os.getenv("EXP_VERSION", "v2")  # must match training run names
TRAIN_EXP_ID = f"{TRAIN_SCENARIO_NAME}_{EXP_VERSION}"

BASELINE_PREFIX = f"runs/models/ppo_baseline_{TRAIN_EXP_ID}_seed"
ATTN_PREFIX     = f"runs/models/ppo_attn_{TRAIN_EXP_ID}_seed"

# Build evaluation env from EVAL_SCENARIO_NAME
scenario_eval = SCENARIOS[EVAL_SCENARIO_NAME]
ENV_ID = scenario_eval["env_id"]
ENV_CONFIG = scenario_eval["config"]

WRAPPER_NAME = scenario_eval.get("wrapper", None)
WRAPPER_KWARGS = scenario_eval.get("wrapper_kwargs", {})

WRAPPER_MAP = {
    None: None,
    "ShuffleNeighboursObs": ShuffleNeighboursObs,
    "StopGoLeaderWrapper": StopGoLeaderWrapper,
}
WRAPPER_CLASS = WRAPPER_MAP.get(WRAPPER_NAME)

N_EVAL_EPISODES = int(os.getenv("N_EVAL_EPISODES", "50"))
SEEDS = [0, 1, 2]


def eval_one(model_path: str, seed: int):
    env = gym.make(ENV_ID, config=ENV_CONFIG)
    if WRAPPER_CLASS is not None:
        env = WRAPPER_CLASS(env, **WRAPPER_KWARGS)

    model = PPO.load(model_path, device="cpu")

    returns = []
    lengths = []
    crashes = 0

    for ep in range(N_EVAL_EPISODES):
        obs, info = env.reset(seed=seed * 1000 + ep)
        terminated = truncated = False
        ep_return = 0.0
        ep_len = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

        returns.append(ep_return)
        lengths.append(ep_len)

         # highway-env sets a "crashed" flag on the ego vehicle
        crashed = bool(env.unwrapped.vehicle.crashed)
        crashes += int(crashed)

    env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_len": float(np.mean(lengths)),
        "crash_rate": float(crashes / N_EVAL_EPISODES),
    }


def summarize(label: str, results: list[dict]):
    mean_returns = [r["mean_return"] for r in results]
    crash_rates = [r["crash_rate"] for r in results]
    mean_lens = [r["mean_len"] for r in results]

    print(f"\n{label}")
    print(f"  return:     {np.mean(mean_returns):.3f} ± {np.std(mean_returns):.3f}")
    print(f"  crash_rate: {np.mean(crash_rates):.3f} ± {np.std(crash_rates):.3f}")
    print(f"  ep_len:     {np.mean(mean_lens):.3f} ± {np.std(mean_lens):.3f}")


if __name__ == "__main__":
    print(f"Loading models trained on: {TRAIN_SCENARIO_NAME} (EXP_VERSION={EXP_VERSION})")
    print(f"Evaluating in env: {EVAL_SCENARIO_NAME} -> {ENV_ID}")

    baseline_results = []
    attn_results = []

    for seed in SEEDS:
        baseline_path = f"{BASELINE_PREFIX}{seed}.zip"
        attn_path = f"{ATTN_PREFIX}{seed}.zip"

        baseline_results.append(eval_one(baseline_path, seed))
        attn_results.append(eval_one(attn_path, seed))

    summarize("BASELINE", baseline_results)
    summarize("ATTENTION", attn_results)
