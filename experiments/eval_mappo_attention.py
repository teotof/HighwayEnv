import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mappo_core import load_mappo_checkpoint  # noqa: E402
from experiments.mappo_eval_utils import predict_mappo_actions, auto_max_steps  # noqa: E402
from experiments.mappo_utils import resolve_mappo_model_path  # noqa: E402
from experiments.marl_utils import make_shared_world_env  # noqa: E402


SCENARIO_NAME = os.getenv("SCENARIO_NAME", "cf_v2")
EXP_VERSION = os.getenv("EXP_VERSION", "v11")
SEED = int(os.getenv("SEED", "0"))
ATTN_MODE = os.getenv("ATTN_MODE", "learned").strip().lower()
CONTROLLED_VEHICLES = int(os.getenv("CONTROLLED_VEHICLES", "2"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "runs/models"))
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(
        resolve_mappo_model_path(
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
DEVICE = os.getenv("DEVICE", "cpu")


def output_dir() -> Path:
    return (
        Path("runs/attn_logs_mappo")
        / EXP_VERSION
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
    policy, metadata = load_mappo_checkpoint(str(model_path), device=DEVICE)

    if not getattr(policy, "expects_matrix_obs", False):
        raise ValueError(f"Checkpoint is not an attention MAPPO policy: {model_path}")

    env_obs_shape = tuple(obs[0].shape)
    print(
        f"SCENARIO_NAME={SCENARIO_NAME} SEED={SEED} "
        f"ATTN_MODE={ATTN_MODE} CONTROLLED_VEHICLES={CONTROLLED_VEHICLES}"
    )
    print(f"Resolved MODEL_PATH={model_path.resolve()}")
    print("Per-agent env obs shape:", env_obs_shape)
    print("Checkpoint policy_kind:", metadata.get("policy_kind"))

    max_steps = STEPS if STEPS > 0 else auto_max_steps(env)
    print("Attention log steps:", max_steps)

    attn_log, dx_log, vx_log, presence_log = [], [], [], []

    for _ in range(max_steps):
        obs_batch = np.stack(obs, axis=0)
        action = predict_mappo_actions(policy, obs, deterministic=True)

        attn = getattr(policy, "last_attention", None)
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
