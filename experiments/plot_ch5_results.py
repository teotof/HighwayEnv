import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SINGLE_AGENT_RESULTS = {
    "cf_obs15": {
        "ordered": {
            "Baseline": (31.01, 0.17),
            "Learned": (30.46, 0.61),
            "Uniform": (30.19, 0.22),
            "DeepSets": (31.07, 0.30),
        },
        "shuffled": {
            "Baseline": (1.67, 0.18),
            "Learned": (30.46, 0.61),
            "Uniform": (30.19, 0.22),
            "DeepSets": (31.07, 0.30),
        },
    },
    "cf_obs20_dense": {
        "ordered": {
            "Baseline": (5.20, 0.50),
            "Learned": (6.48, 1.27),
            "Uniform": (5.92, 0.65),
            "DeepSets": (7.68, 0.92),
        },
        "shuffled": {
            "Baseline": (-0.42, 0.18),
            "Learned": (6.48, 1.27),
            "Uniform": (5.92, 0.65),
            "DeepSets": (7.68, 0.92),
        },
    },
    "highway_lanechange_dense_obs20": {
        "ordered": {
            "Baseline": (59.19, 4.32),
            "Learned": (62.50, 7.45),
            "Uniform": (47.80, 12.82),
            "DeepSets": (57.09, 5.69),
        },
        "shuffled": {
            "Baseline": (21.93, 11.41),
            "Learned": (62.50, 7.45),
            "Uniform": (47.80, 12.82),
            "DeepSets": (57.09, 5.69),
        },
    },
}


COLORS = {
    "Baseline": "#4C78A8",
    "Learned": "#F58518",
    "Uniform": "#54A24B",
    "DeepSets": "#B279A2",
}

SCENARIO_LABELS = {
    "cf_obs15": "cf_obs15",
    "cf_obs20_dense": "cf_obs20_dense",
    "highway_lanechange_dense_obs20": "highway_lanechange_dense_obs20",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the Chapter 5 ordered-vs-shuffled robustness figure."
    )
    parser.add_argument(
        "--output",
        default="runs/ch5_figures/ch5_shuffle_returns.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = list(SINGLE_AGENT_RESULTS.keys())
    methods = ["Baseline", "Learned", "Uniform", "DeepSets"]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(16, 4.8), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]

    width = 0.36
    x = np.arange(len(methods))

    for ax, scenario in zip(axes, scenarios):
        ordered_means = [SINGLE_AGENT_RESULTS[scenario]["ordered"][m][0] for m in methods]
        ordered_stds = [SINGLE_AGENT_RESULTS[scenario]["ordered"][m][1] for m in methods]
        shuffled_means = [SINGLE_AGENT_RESULTS[scenario]["shuffled"][m][0] for m in methods]
        shuffled_stds = [SINGLE_AGENT_RESULTS[scenario]["shuffled"][m][1] for m in methods]

        for idx, method in enumerate(methods):
            ax.bar(
                x[idx] - width / 2,
                ordered_means[idx],
                width,
                yerr=ordered_stds[idx],
                color=COLORS[method],
                alpha=0.95,
                capsize=3,
            )
            ax.bar(
                x[idx] + width / 2,
                shuffled_means[idx],
                width,
                yerr=shuffled_stds[idx],
                color=COLORS[method],
                alpha=0.35,
                hatch="//",
                capsize=3,
            )

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Mean return")
        ax.grid(axis="y", alpha=0.25)

    ordered_proxy = plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.9)
    shuffled_proxy = plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.35, hatch="//")
    fig.legend(
        [ordered_proxy, shuffled_proxy],
        ["Ordered", "Shuffled"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Single-agent return under ordered and shuffled neighbour order", y=1.08)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
