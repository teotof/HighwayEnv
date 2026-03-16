import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create simple plots from saved attention logs.")
    parser.add_argument("--scenario", required=True, help="Scenario name under runs/attn_logs/<mode>/")
    parser.add_argument("--seed", type=int, default=0, help="Seed number")
    parser.add_argument("--mode", default="learned", help="Attention mode folder name")
    parser.add_argument("--start", type=int, default=0, help="Start timestep for the heatmap")
    parser.add_argument("--window", type=int, default=60, help="Number of timesteps to plot in the heatmap")
    parser.add_argument(
        "--bar-timestep",
        type=int,
        default=-1,
        help="Timestep for the bar chart. Use -1 to auto-pick a representative step.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/attn_plots",
        help="Output directory for generated images",
    )
    return parser.parse_args()


def infer_leader_index(dx_t: np.ndarray, presence_t: np.ndarray) -> int:
    valid = presence_t > 0.5
    dx_valid = dx_t.copy()
    dx_valid[~valid] = np.inf
    front_dx = np.where(dx_valid > 0, dx_valid, np.inf)
    if np.isfinite(front_dx).any():
        return int(np.argmin(front_dx))
    return int(np.argmin(np.abs(dx_valid)))


def pick_representative_timestep(attn: np.ndarray, dx: np.ndarray, presence: np.ndarray) -> int:
    leader_idx = np.array([infer_leader_index(dx[t], presence[t]) for t in range(attn.shape[0])])
    argmax_idx = np.nanargmax(attn, axis=1)
    leader_w = attn[np.arange(attn.shape[0]), leader_idx]
    hits = np.where(argmax_idx == leader_idx)[0]
    if len(hits) == 0:
        return int(np.nanargmax(np.nanmax(attn, axis=1)))
    target = float(np.median(leader_w[hits]))
    best = min(hits, key=lambda t: abs(float(leader_w[t]) - target))
    return int(best)


def save_bar_chart(out_path: Path, attn_t: np.ndarray, presence_t: np.ndarray, leader_idx: int, timestep: int, scenario: str):
    slots = np.arange(1, len(attn_t) + 1)
    colors = np.where(presence_t > 0.5, "#4C78A8", "#D9D9D9")
    colors[leader_idx] = "#F58518"

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(slots, attn_t, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Attention Weights at One Timestep ({scenario}, t={timestep})")
    ax.set_xlabel("Neighbour slot")
    ax.set_ylabel("Attention weight")
    ax.set_xticks(slots)
    ax.set_ylim(0, max(0.65, float(np.nanmax(attn_t)) + 0.05))
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.99,
        0.95,
        f"Leader slot: {leader_idx + 1}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.3"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(out_path: Path, attn: np.ndarray, presence: np.ndarray, start: int, end: int, scenario: str):
    window_attn = attn[start:end].T.copy()
    window_presence = presence[start:end].T
    window_attn[window_presence <= 0.5] = np.nan

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#EFEFEF")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    im = ax.imshow(
        window_attn,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        origin="upper",
        vmin=0.0,
        vmax=max(0.6, float(np.nanmax(window_attn))),
    )
    ax.set_title(f"Attention Heatmap Over Time ({scenario}, t={start}..{end - 1})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neighbour slot")
    ax.set_yticks(np.arange(window_attn.shape[0]))
    ax.set_yticklabels(np.arange(1, window_attn.shape[0] + 1))
    tick_count = min(7, max(2, end - start))
    x_ticks = np.linspace(0, end - start - 1, num=tick_count, dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(start + int(x)) for x in x_ticks])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    base = Path(args.out_dir)
    base.mkdir(parents=True, exist_ok=True)

    log_dir = Path("runs") / "attn_logs" / args.mode / args.scenario / f"seed{args.seed}"
    attn = np.load(log_dir / "attention.npy")
    dx = np.load(log_dir / "dx.npy")
    presence = np.load(log_dir / "presence.npy")

    start = max(0, args.start)
    end = min(attn.shape[0], start + max(1, args.window))
    timestep = args.bar_timestep if args.bar_timestep >= 0 else pick_representative_timestep(attn, dx, presence)
    timestep = max(0, min(attn.shape[0] - 1, timestep))
    leader_idx = infer_leader_index(dx[timestep], presence[timestep])

    stem = f"{args.mode}_{args.scenario}_seed{args.seed}"
    heatmap_path = base / f"{stem}_heatmap_t{start}_{end}.png"
    bar_path = base / f"{stem}_bar_t{timestep}.png"

    save_heatmap(heatmap_path, attn, presence, start, end, args.scenario)
    save_bar_chart(bar_path, attn[timestep], presence[timestep], leader_idx, timestep, args.scenario)

    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved bar chart: {bar_path}")
    print(f"Representative timestep: {timestep}")
    print(f"Leader slot at that timestep: {leader_idx + 1}")


if __name__ == "__main__":
    main()
