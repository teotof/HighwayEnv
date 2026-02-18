import os
import time
import numpy as np
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO

from experiments.scenarios import SCENARIOS
from experiments.wrappers import ShuffleNeighboursObs, StopGoLeaderWrapper

SCENARIO_NAME = os.getenv("SCENARIO_NAME", "cf_v2")
MODEL_PATH = os.getenv("MODEL_PATH", "runs/models/ppo_attn_cf_v2_v2_seed0.zip")
SEED = int(os.getenv("SEED", "0"))
STEPS = int(os.getenv("STEPS", "1000"))

scenario = SCENARIOS[SCENARIO_NAME]
ENV_ID = scenario["env_id"]
ENV_CONFIG = scenario["config"]

WRAPPER_MAP = {
    None: None,
    "ShuffleNeighboursObs": ShuffleNeighboursObs,
    "StopGoLeaderWrapper": StopGoLeaderWrapper,
}
WRAPPER_CLASS = WRAPPER_MAP.get(scenario.get("wrapper", None))
WRAPPER_KWARGS = scenario.get("wrapper_kwargs", {})

def main():
    env = gym.make(ENV_ID, config=ENV_CONFIG, render_mode=None)
    if WRAPPER_CLASS is not None:
        env = WRAPPER_CLASS(env, **WRAPPER_KWARGS)

    obs, info = env.reset(seed=SEED)
    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()
    attn_log = []
    dx_log = []
    presence_log = []

    for t in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract attention weights (batch=1)
        attn = model.policy.features_extractor.last_attention
        # Log neighbour relative x positions for context (slots 1..4)
        # features: [presence, x, y, vx, vy] => x is column 1
        if attn is not None:
            attn_log.append(attn.squeeze(0).cpu().numpy())      # (N,)
            dx_log.append(obs[1:, 1].copy())                    # (N,) neighbour x
            presence_log.append(obs[1:, 0].copy())              # (N,) presence

        time.sleep(1/30)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    attn_log = np.array(attn_log)
    dx_log = np.array(dx_log)
    presence_log = np.array(presence_log)

    out_dir = f"runs/attn_logs/{SCENARIO_NAME}/seed{SEED}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/attention.npy", attn_log)
    np.save(f"{out_dir}/dx.npy", dx_log)
    np.save(f"{out_dir}/presence.npy", presence_log)
    print("Saved:", attn_log.shape, dx_log.shape)

if __name__ == "__main__":
    main()
