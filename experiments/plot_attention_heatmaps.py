import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyse_attention import entropy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a two-panel Chapter 5 attention heatmap figure."
    )
    parser.add_argument(
        "--cf-log-dir",
        default="runs/attn_logs/v12/learned/cf_obs15/seed0",
        help="Single-agent car-following attention log directory.",
    )
    parser.add_argument(
        "--highway-log-dir",
        default="runs/attn_logs/v12/learned/highway_lanechange_dense_obs20/seed0",
        help="Single-agent lane-change attention log directory.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=180,
        help="How many logged timesteps to show from the start of each run.",
    )
    parser.add_argument(
        "--output",
        default="runs/ch5_figures/ch5_attention_heatmaps.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_attention(log_dir: Path, steps: int):
    attn = np.load(log_dir / "attention.npy")
    steps = min(steps, attn.shape[0])
    attn = attn[:steps]
    mean_weights = attn.mean(axis=0)
    slot_order = np.argsort(mean_weights)[::-1]
    attn_sorted = attn[:, slot_order].T
    ent = float(entropy(attn).mean())
    return attn_sorted, slot_order, ent


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cf_dir = Path(args.cf_log_dir)
    highway_dir = Path(args.highway_log_dir)

    cf_attn, cf_order, cf_entropy = load_attention(cf_dir, args.steps)
    hw_attn, hw_order, hw_entropy = load_attention(highway_dir, args.steps)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    panels = [
        (
            axes[0],
            cf_attn,
            cf_order,
            "cf_obs15 learned attention",
            cf_entropy,
        ),
        (
            axes[1],
            hw_attn,
            hw_order,
            "highway_lanechange_dense_obs20 learned attention",
            hw_entropy,
        ),
    ]

    vmax = max(float(cf_attn.max()), float(hw_attn.max()))

    for ax, data, order, title, ent in panels:
        image = ax.imshow(
            data,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"{title}\nslots sorted by mean weight, mean entropy = {ent:.2f}", fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Neighbour slot")
        tick_count = min(6, len(order))
        tick_idx = np.linspace(0, len(order) - 1, tick_count, dtype=int)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([f"s{order[i] + 1}" for i in tick_idx])

    cbar = fig.colorbar(image, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Attention weight")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
