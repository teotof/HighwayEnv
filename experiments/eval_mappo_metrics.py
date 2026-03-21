import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mappo_eval_utils import eval_mappo_model, summarize_mappo_results  # noqa: E402
from experiments.mappo_utils import parse_seeds, resolve_mappo_model_path  # noqa: E402


TRAIN_SCENARIO_NAME = os.getenv("TRAIN_SCENARIO_NAME", os.getenv("SCENARIO_NAME", "cf_v2"))
EVAL_SCENARIO_NAME = os.getenv("EVAL_SCENARIO_NAME", TRAIN_SCENARIO_NAME)
EXP_VERSION = os.getenv("EXP_VERSION", "v11")
CONTROLLED_VEHICLES = int(os.getenv("CONTROLLED_VEHICLES", "2"))
SEEDS = parse_seeds(os.getenv("SEEDS", "0,1,2"))
N_EVAL_EPISODES = int(os.getenv("N_EVAL_EPISODES", "50"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "0"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "runs/models"))
ATTN_MODE = os.getenv("ATTN_MODE", "learned").strip().lower()
DEVICE = os.getenv("DEVICE", "cpu")


if __name__ == "__main__":
    print(f"Train scenario: {TRAIN_SCENARIO_NAME}")
    print(f"Eval scenario: {EVAL_SCENARIO_NAME}")
    print(f"EXP_VERSION={EXP_VERSION}")
    print(f"CONTROLLED_VEHICLES={CONTROLLED_VEHICLES}")
    print(f"SEEDS={SEEDS}")
    print(f"N_EVAL_EPISODES={N_EVAL_EPISODES}")
    print(f"ATTN_MODE={ATTN_MODE}")
    print(f"DEVICE={DEVICE}")

    baseline_results = []
    attention_results = []

    for seed in SEEDS:
        baseline_path = resolve_mappo_model_path(
            MODEL_DIR,
            TRAIN_SCENARIO_NAME,
            EXP_VERSION,
            CONTROLLED_VEHICLES,
            seed,
            model_kind="baseline",
        )
        attn_path = resolve_mappo_model_path(
            MODEL_DIR,
            TRAIN_SCENARIO_NAME,
            EXP_VERSION,
            CONTROLLED_VEHICLES,
            seed,
            model_kind="attn",
            attn_mode=ATTN_MODE,
        )

        baseline_results.append(
            eval_mappo_model(
                str(baseline_path),
                scenario_name=EVAL_SCENARIO_NAME,
                controlled_vehicles=CONTROLLED_VEHICLES,
                n_eval_episodes=N_EVAL_EPISODES,
                max_steps=MAX_STEPS,
                seed=seed,
                device=DEVICE,
            )
        )
        attention_results.append(
            eval_mappo_model(
                str(attn_path),
                scenario_name=EVAL_SCENARIO_NAME,
                controlled_vehicles=CONTROLLED_VEHICLES,
                n_eval_episodes=N_EVAL_EPISODES,
                max_steps=MAX_STEPS,
                seed=seed,
                device=DEVICE,
            )
        )

    summarize_mappo_results("MAPPO BASELINE", baseline_results)
    summarize_mappo_results(f"MAPPO ATTENTION ({ATTN_MODE})", attention_results)
