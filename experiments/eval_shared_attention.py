import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_extractor import KinematicAttentionExtractor  # noqa: F401
from experiments.marl_utils import (
    predict_shared_actions,
    resolve_shared_model_path,
    make_shared_world_env,
)
from experiments.shared_eval_utils import auto_max_steps


SCENARIO_NAME = os.getenv("SCENARIO_NAME", "cf_v2")
EXP_VERSION = os.getenv("EXP_VERSION", "v10")
SEED = int(os.getenv("SEED", "0"))
ATTN_MODE = os.getenv("ATTN_MODE", "learned").strip().lower()
CONTROLLED_VEHICLES = int(os.getenv("CONTROLLED_VEHICLES", "2"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "runs/models"))
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(
        resolve_shared_model_path(
            MODEL_DIR,
            SCENARIO_NAME,
            EXP_VERSION,
            CONTROLLED_VEHICLES,
            SEED,
            model_kind="attn",
            attn_mode=ATTN_MODE,
        )
    ),
)
STEPS = int(os.getenv("STEPS", "0"))


def output_dir() -> Path:
    return (
        Path("runs/attn_logs_shared")
        / ATTN_MODE
        / SCENARIO_NAME
        / f"cv{CONTROLLED_VEHICLES}"
        / f"seed{SEED}"
    )


def main():
    model_path = Path(MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = make_shared_world_env(
        scenario_name=SCENARIO_NAME,
        controlled_vehicles=CONTROLLED_VEHICLES,
    )
    obs, info = env.reset(seed=SEED)
    model = PPO.load(str(model_path), device="cpu")
    env_obs_shape = tuple(obs[0].shape)
    model_obs_shape = tuple(model.observation_space.shape)

    print(
        f"SCENARIO_NAME={SCENARIO_NAME} SEED={SEED} "
        f"ATTN_MODE={ATTN_MODE} CONTROLLED_VEHICLES={CONTROLLED_VEHICLES}"
    )
    print(f"Resolved MODEL_PATH={model_path.resolve()}")
    print("Per-agent env obs shape:", env_obs_shape)
    print("Model obs shape:", model_obs_shape)

    if env_obs_shape != model_obs_shape:
        raise ValueError(
            "Observation mismatch: "
            f"env obs {env_obs_shape} vs model obs {model_obs_shape}. "
            f"MODEL_PATH={model_path}."
        )

    max_steps = STEPS if STEPS > 0 else auto_max_steps(env)
    print("Attention log steps:", max_steps)

    attn_log, dx_log, vx_log, presence_log = [], [], [], []

    for _ in range(max_steps):
        obs_batch = np.stack(obs, axis=0)
        action = predict_shared_actions(model, obs, deterministic=True)

        fe = model.policy.features_extractor
        attn = getattr(fe, "last_attention", None)
        if attn is None:
            attn_log.append(
                np.full(
                    (CONTROLLED_VEHICLES, obs_batch.shape[1] - 1),
                    np.nan,
                    dtype=np.float32,
                )
            )
        else:
            attn_log.append(attn.cpu().numpy())

        dx_log.append(obs_batch[:, 1:, 1].copy())
        vx_log.append(obs_batch[:, 1:, 3].copy())
        presence_log.append(obs_batch[:, 1:, 0].copy())

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    attn_arr = np.asarray(attn_log)
    dx_arr = np.asarray(dx_log)
    vx_arr = np.asarray(vx_log)
    presence_arr = np.asarray(presence_log)

    np.save(out_dir / "attention.npy", attn_arr)
    np.save(out_dir / "dx.npy", dx_arr)
    np.save(out_dir / "vx.npy", vx_arr)
    np.save(out_dir / "presence.npy", presence_arr)
    print("Saved:", attn_arr.shape, dx_arr.shape, vx_arr.shape, presence_arr.shape)


if __name__ == "__main__":
    main()
