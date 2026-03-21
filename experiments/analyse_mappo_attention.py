import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyse_attention import compute_leader_idx, compute_ttc_idx, entropy  # noqa: E402


SCENARIO_NAME = os.getenv("SCENARIO_NAME", "cf_v2")
EXP_VERSION = os.getenv("EXP_VERSION", "v11")
SEED = int(os.getenv("SEED", "0"))
ATTN_MODE = os.getenv("ATTN_MODE", "learned").strip().lower()
CONTROLLED_VEHICLES = int(os.getenv("CONTROLLED_VEHICLES", "2"))


def summarize_block(label: str, attn: np.ndarray, dx: np.ndarray, vx: np.ndarray, pres: np.ndarray):
    argmax_idx = np.argmax(attn, axis=1)
    leader_idx = compute_leader_idx(dx, pres)
    hit_rate = (argmax_idx == leader_idx).mean()
    leader_weight = attn[np.arange(attn.shape[0]), leader_idx].mean()
    ent = entropy(attn).mean()

    print(label)
    print("  T:", attn.shape[0], "N:", attn.shape[1])
    print("  argmax==leader hit rate:", hit_rate)
    print("  mean attention on leader:", leader_weight)
    print("  mean attention entropy:", ent)

    ttc_idx, has_closing, min_ttc = compute_ttc_idx(dx, vx, pres)
    ttc_hit_rate = (argmax_idx == ttc_idx).mean()
    ttc_weight = attn[np.arange(attn.shape[0]), ttc_idx].mean()
    closing_frac = has_closing.mean()
    finite_ttc = np.isfinite(min_ttc)
    mean_min_ttc = (
        float(np.nanmean(min_ttc[finite_ttc])) if finite_ttc.any() else float("nan")
    )

    print("  argmax==ttc_critical hit rate:", ttc_hit_rate)
    print("  mean attention on ttc_critical:", ttc_weight)
    print("  fraction steps with closing TTC candidate:", closing_frac)
    print("  mean min TTC (closing steps only):", mean_min_ttc)


def main():
    base = (
        Path("runs/attn_logs_mappo")
        / EXP_VERSION
        / ATTN_MODE
        / SCENARIO_NAME
        / f"cv{CONTROLLED_VEHICLES}"
        / f"seed{SEED}"
    )

    attn = np.load(base / "attention.npy")
    dx = np.load(base / "dx.npy")
    vx = np.load(base / "vx.npy")
    pres = np.load(base / "presence.npy")

    if attn.ndim != 3:
        raise ValueError(
            f"Expected MAPPO attention log with shape (T, A, N), got {attn.shape}."
        )

    steps, agents, neighbours = attn.shape
    print(f"steps={steps} agents={agents} neighbours={neighbours}")
    print(f"log_dir={base}")

    summarize_block(
        "AGGREGATE",
        attn.reshape(-1, neighbours),
        dx.reshape(-1, neighbours),
        vx.reshape(-1, neighbours),
        pres.reshape(-1, neighbours),
    )

    for agent_index in range(agents):
        summarize_block(
            f"AGENT {agent_index}",
            attn[:, agent_index, :],
            dx[:, agent_index, :],
            vx[:, agent_index, :],
            pres[:, agent_index, :],
        )


if __name__ == "__main__":
    main()
