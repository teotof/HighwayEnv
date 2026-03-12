import os
import numpy as np
import gymnasium as gym
import highway_env  # registers highway/merge env IDs in gymnasium
from stable_baselines3 import PPO

from experiments.scenarios import SCENARIOS
from experiments.wrappers import ShuffleNeighboursObs, StopGoLeaderWrapper

SCENARIO_NAME = os.getenv("SCENARIO_NAME", "cf_v2")
EXP_VERSION = os.getenv("EXP_VERSION", "v2")
SEED = int(os.getenv("SEED", "0"))
ATTN_MODE = os.getenv("ATTN_MODE", "").strip()

if ATTN_MODE:
    default_model_path = f"runs/models/ppo_attn_{ATTN_MODE}_{SCENARIO_NAME}_{EXP_VERSION}_seed{SEED}.zip"
else:
    # Backward-compatible default for older runs.
    default_model_path = f"runs/models/ppo_attn_{SCENARIO_NAME}_{EXP_VERSION}_seed{SEED}.zip"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    default_model_path
)
STEPS = int(os.getenv("STEPS", "0"))

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
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    env = gym.make(ENV_ID, config=ENV_CONFIG, render_mode=None)
    if WRAPPER_CLASS is not None:
        env = WRAPPER_CLASS(env, **WRAPPER_KWARGS)

    obs, info = env.reset(seed=SEED)
    model = PPO.load(MODEL_PATH, device="cpu")

    env_obs_shape = tuple(obs.shape)
    model_obs_shape = tuple(model.observation_space.shape)

    print(f"SCENARIO_NAME={SCENARIO_NAME} SEED={SEED} ATTN_MODE={ATTN_MODE if ATTN_MODE else 'legacy-name'}")
    print(f"Resolved MODEL_PATH={os.path.abspath(MODEL_PATH)}")
    print("Env obs shape:", env_obs_shape)
    print("Model obs shape:", model_obs_shape)

    max_steps = STEPS
    if max_steps <= 0:
        cfg = env.unwrapped.config
        max_steps = max(1, int(round(float(cfg.get("duration", 40)) * float(cfg.get("policy_frequency", 1)))))
    print("Attention log steps:", max_steps)

    if env_obs_shape != model_obs_shape:
        raise ValueError(
            "Observation mismatch: "
            f"env obs {env_obs_shape} vs model obs {model_obs_shape}. "
            f"MODEL_PATH={MODEL_PATH}. "
            "This usually means the wrong checkpoint was loaded for the scenario."
        )

    attn_log, dx_log, vx_log, presence_log = [], [], [], []

    for t in range(max_steps):
        obs_in = obs

        action, _ = model.predict(obs_in, deterministic=True)

        fe = model.policy.features_extractor
        attn = getattr(fe, "last_attention", None)

        if attn is None:
            attn_log.append(np.full((obs_in.shape[0]-1,), np.nan, dtype=np.float32))
        else:
            attn_log.append(attn.squeeze(0).cpu().numpy())

        dx_log.append(obs_in[1:, 1].copy())
        vx_log.append(obs_in[1:, 3].copy())
        presence_log.append(obs_in[1:, 0].copy())

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()   # do not reseed every time

    env.close()

    attn_log = np.array(attn_log)
    dx_log = np.array(dx_log)
    vx_log = np.array(vx_log)
    presence_log = np.array(presence_log)

    if ATTN_MODE:
        out_dir = f"runs/attn_logs/{ATTN_MODE}/{SCENARIO_NAME}/seed{SEED}"
    else:
        out_dir = f"runs/attn_logs/{SCENARIO_NAME}/seed{SEED}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/attention.npy", attn_log)
    np.save(f"{out_dir}/dx.npy", dx_log)
    np.save(f"{out_dir}/vx.npy", vx_log)
    np.save(f"{out_dir}/presence.npy", presence_log)
    print("Saved:", attn_log.shape, dx_log.shape, vx_log.shape, presence_log.shape)

if __name__ == "__main__":
    main()
