import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_log_dir(root: Path, attn_mode: str, scenario: str, controlled_vehicles: int, seed: int) -> Path:
    log_dir = root / "runs" / "attn_logs_shared" / attn_mode / scenario / f"cv{controlled_vehicles}" / f"seed{seed}"
    if not log_dir.exists():
        raise FileNotFoundError(f"Shared attention log directory not found: {log_dir}")
    return log_dir


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or window >= x.shape[0]:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="valid")


def entropy(attn: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return -(attn * np.log(attn + eps)).sum(axis=-1)


def leader_index(dx: np.ndarray, presence: np.ndarray) -> np.ndarray:
    ahead = np.where((presence > 0.5) & (dx > 0.0), dx, np.inf)
    leader = np.argmin(ahead, axis=-1)
    no_leader = ~np.isfinite(np.min(ahead, axis=-1))
    leader = leader.astype(np.int64)
    leader[no_leader] = -1
    return leader


def plot_summary(attn: np.ndarray, dx: np.ndarray, presence: np.ndarray, out_path: Path, heatmap_steps: int, movavg: int) -> None:
    steps, agents, neighbours = attn.shape
    show_agents = min(agents, 2)
    heatmap_steps = min(heatmap_steps, steps)
    ent = entropy(attn)
    leader = leader_index(dx, presence)

    leader_w = np.zeros((steps, agents), dtype=np.float32)
    for agent in range(agents):
        valid = leader[:, agent] >= 0
        idx = leader[valid, agent]
        leader_w[valid, agent] = attn[valid, agent, idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    if show_agents >= 1:
        im0 = axes[0].imshow(attn[:heatmap_steps, 0, :].T, aspect="auto", origin="lower", cmap="magma")
        axes[0].set_title(f"Agent 0 Attention Heatmap (first {heatmap_steps} steps)")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Neighbour slot")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    else:
        axes[0].axis("off")

    if show_agents >= 2:
        im1 = axes[1].imshow(attn[:heatmap_steps, 1, :].T, aspect="auto", origin="lower", cmap="magma")
        axes[1].set_title(f"Agent 1 Attention Heatmap (first {heatmap_steps} steps)")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Neighbour slot")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        mean_slot = attn.mean(axis=0)[0]
        axes[1].bar(np.arange(neighbours), mean_slot)
        axes[1].set_title("Mean Attention by Neighbour Slot")
        axes[1].set_xlabel("Neighbour slot")
        axes[1].set_ylabel("Mean attention")

    for agent in range(agents):
        series = moving_average(leader_w[:, agent], movavg)
        axes[2].plot(series, label=f"agent {agent}")
    axes[2].axhline(1.0 / neighbours, color="black", linestyle="--", linewidth=1, label="uniform")
    axes[2].set_title(f"Leader Weight Moving Average (window={movavg})")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Attention on nearest-front vehicle")
    axes[2].legend()

    for agent in range(agents):
        series = moving_average(ent[:, agent], movavg)
        axes[3].plot(series, label=f"agent {agent}")
    axes[3].axhline(math.log(neighbours), color="black", linestyle="--", linewidth=1, label="uniform entropy")
    axes[3].set_title(f"Attention Entropy Moving Average (window={movavg})")
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Entropy")
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_dx_scatter(attn: np.ndarray, dx: np.ndarray, presence: np.ndarray, out_path: Path, scatter_points: int) -> None:
    valid = presence > 0.5
    attn_flat = attn[valid]
    dx_flat = dx[valid]
    if attn_flat.size == 0:
        raise ValueError("No valid neighbours found in shared attention log")

    if attn_flat.size > scatter_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(attn_flat.size, size=scatter_points, replace=False)
        attn_flat = attn_flat[idx]
        dx_flat = dx_flat[idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(dx_flat, attn_flat, s=8, alpha=0.25)
    ax.set_title("Attention vs Longitudinal Distance")
    ax.set_xlabel("dx to neighbour")
    ax.set_ylabel("Attention weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--controlled-vehicles", type=int, required=True)
    parser.add_argument("--attn-mode", default="learned")
    parser.add_argument("--root", default=".")
    parser.add_argument("--heatmap-steps", type=int, default=300)
    parser.add_argument("--movavg", type=int, default=50)
    parser.add_argument("--scatter-points", type=int, default=4000)
    args = parser.parse_args()

    root = Path(args.root)
    log_dir = load_log_dir(root, args.attn_mode, args.scenario, args.controlled_vehicles, args.seed)

    attn = np.load(log_dir / "attention.npy")
    dx = np.load(log_dir / "dx.npy")
    presence = np.load(log_dir / "presence.npy")

    plot_summary(
        attn=attn,
        dx=dx,
        presence=presence,
        out_path=log_dir / "shared_attention_summary.png",
        heatmap_steps=args.heatmap_steps,
        movavg=args.movavg,
    )
    plot_dx_scatter(
        attn=attn,
        dx=dx,
        presence=presence,
        out_path=log_dir / "shared_attention_dx_scatter.png",
        scatter_points=args.scatter_points,
    )

    print(f"Saved plots to {log_dir}")


if __name__ == "__main__":
    main()
