import os
import numpy as np
import gymnasium as gym
import highway_env  # registers envs
from stable_baselines3 import PPO

from attention_extractor import KinematicAttentionExtractor  # noqa: F401
from deepset_extractor import DeepSetExtractor  # noqa: F401

from experiments.scenarios import SCENARIOS
from experiments.wrappers import ShuffleNeighboursObs, StopGoLeaderWrapper

# Which models to load (training) vs which env to test in (evaluation)
TRAIN_SCENARIO_NAME = os.getenv("TRAIN_SCENARIO_NAME", os.getenv("SCENARIO_NAME", "cf_obs15_shuffle"))
EVAL_SCENARIO_NAME  = os.getenv("EVAL_SCENARIO_NAME", TRAIN_SCENARIO_NAME)

EXP_VERSION = os.getenv("EXP_VERSION", "v2")  # must match training run names
TRAIN_EXP_ID = f"{TRAIN_SCENARIO_NAME}_{EXP_VERSION}"
ATTN_MODE = os.getenv("ATTN_MODE", "").strip()
MODEL_KIND = os.getenv("MODEL_KIND", "").strip().lower()

if not MODEL_KIND:
    MODEL_KIND = "attn"

BASELINE_PREFIX = f"runs/models/ppo_baseline_{TRAIN_EXP_ID}_seed"
if MODEL_KIND == "attn":
    if ATTN_MODE:
        VARIANT_PREFIX = f"runs/models/ppo_attn_{ATTN_MODE}_{TRAIN_EXP_ID}_seed"
        VARIANT_LABEL = f"ATTENTION ({ATTN_MODE})"
    else:
        # Backward-compatible path for older runs without mode in name.
        VARIANT_PREFIX = f"runs/models/ppo_attn_{TRAIN_EXP_ID}_seed"
        VARIANT_LABEL = "ATTENTION"
elif MODEL_KIND == "deepsets":
    VARIANT_PREFIX = f"runs/models/ppo_deepsets_{TRAIN_EXP_ID}_seed"
    VARIANT_LABEL = "DEEPSETS"
else:
    raise ValueError(f"Unknown MODEL_KIND '{MODEL_KIND}'. Use attn|deepsets.")

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
MAX_STEPS = int(os.getenv("MAX_STEPS", "0"))  # 0 means "auto from env config"
SEEDS = [0, 1, 2]


def auto_max_steps(env) -> int:
    cfg = env.unwrapped.config
    duration_s = float(cfg.get("duration", 40))
    policy_hz = float(cfg.get("policy_frequency", 1))
    return max(1, int(round(duration_s * policy_hz)))


def eval_one(model_path: str, seed: int):
    env = gym.make(ENV_ID, config=ENV_CONFIG)
    if WRAPPER_CLASS is not None:
        env = WRAPPER_CLASS(env, **WRAPPER_KWARGS)
    
    # Safety cap so eval cannot get stuck if terminated/truncated never becomes True
    if MAX_STEPS > 0:
        max_steps = MAX_STEPS
    else:
        max_steps = auto_max_steps(env)

    model = PPO.load(model_path, device="cpu")

    returns = []
    lengths = []
    crashes = 0

    for ep in range(N_EVAL_EPISODES):
        obs, info = env.reset(seed=seed * 1000 + ep)
        terminated = truncated = False
        ep_return = 0.0
        ep_len = 0

        while not (terminated or truncated) and ep_len < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

        returns.append(ep_return)
        lengths.append(ep_len)

        crashed = False
        if hasattr(env.unwrapped, "vehicle") and env.unwrapped.vehicle is not None:
            crashed = bool(getattr(env.unwrapped.vehicle, "crashed", False))
        crashes += int(crashed)

        if ep % 5 == 0:
            print(f"[{os.path.basename(model_path)}] seed={seed} ep={ep}/{N_EVAL_EPISODES}", flush=True)

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
    print(f"Model kind: {MODEL_KIND}")
    if MODEL_KIND == "attn":
        print(f"Attention mode: {ATTN_MODE if ATTN_MODE else 'legacy-name'}")

    baseline_results = []
    variant_results = []

    for seed in SEEDS:
        baseline_path = f"{BASELINE_PREFIX}{seed}.zip"
        variant_path = f"{VARIANT_PREFIX}{seed}.zip"

        baseline_results.append(eval_one(baseline_path, seed))
        variant_results.append(eval_one(variant_path, seed))

    summarize("BASELINE", baseline_results)
    summarize(VARIANT_LABEL, variant_results)
