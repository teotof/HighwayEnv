import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from experiments.scenarios import SCENARIOS
from experiments.wrappers import ShuffleNeighboursObs, StopGoLeaderWrapper

SCENARIO_NAME = os.getenv("SCENARIO_NAME", "cf_v2")
EXP_VERSION = os.getenv("EXP_VERSION", "v2")
SEED = int(os.getenv("SEED", "0"))

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    f"runs/models/ppo_attn_{SCENARIO_NAME}_{EXP_VERSION}_seed{SEED}.zip"
)
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
    model = PPO.load(MODEL_PATH, device="cpu")

    print("Env obs shape:", obs.shape)
    print("Model obs space:", model.observation_space)

    assert obs.shape == model.observation_space.shape, (
        f"Mismatch: env obs {obs.shape} vs model {model.observation_space.shape}. "
        f"MODEL_PATH={MODEL_PATH}"
    )

    attn_log, dx_log, presence_log = [], [], []

    for t in range(STEPS):
        obs_in = obs

        action, _ = model.predict(obs_in, deterministic=True)

        fe = model.policy.features_extractor
        attn = getattr(fe, "last_attention", None)

        if attn is None:
            attn_log.append(np.full((obs_in.shape[0]-1,), np.nan, dtype=np.float32))
        else:
            attn_log.append(attn.squeeze(0).cpu().numpy())

        dx_log.append(obs_in[1:, 1].copy())
        presence_log.append(obs_in[1:, 0].copy())

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()   # do not reseed every time

    env.close()

    attn_log = np.array(attn_log)
    dx_log = np.array(dx_log)
    presence_log = np.array(presence_log)

    out_dir = f"runs/attn_logs/{SCENARIO_NAME}/seed{SEED}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/attention.npy", attn_log)
    np.save(f"{out_dir}/dx.npy", dx_log)
    np.save(f"{out_dir}/presence.npy", presence_log)
    print("Saved:", attn_log.shape, dx_log.shape, presence_log.shape)

if __name__ == "__main__":
    main()
