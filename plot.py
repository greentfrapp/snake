import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

fig_folder = Path("figures")
fig_folder.mkdir(parents=True, exist_ok=True)


metric_map = {
    "avg score": "Averaged Test Score",
    "max avg": "Max Averaged Test Score",
    "max single": "Max Single Test Score",
}

# Graphs

for metric, pretty_metric in metric_map.items():

    x_axis_title = "Iterations (millions)"
    y_axis_title = pretty_metric

    vanilla_c = "#e74c3c"  # red
    curr_c = "#2ecc71"  # green
    plot_params = {
        "linewidth": 2,
        "alpha": 0.5,
        "zorder": 1,
    }
    scatter_params = {
        "edgecolors": "black",
        "linewidths": 1,
        "s": 50,
        "zorder": 2,
    }

    for s in range(4, 11):

        fig, ax = plt.subplots()

        title = f"S = {s}"

        for n in range(5):

            vanilla_log = f"logs/log_S_{s}_{n}.csv"
            curriculum_log = f"logs/log_S_{s}_{n}_curr.csv"

            vanilla_df = pd.read_csv(vanilla_log)
            curriculum_df = pd.read_csv(curriculum_log)

            ax.plot(
                vanilla_df["iteration"] / 1e6,
                vanilla_df[metric],
                color=vanilla_c,
                **plot_params
            )
            ax.plot(
                curriculum_df["iteration"] / 1e6,
                curriculum_df[metric],
                color=curr_c,
                **plot_params
            )

            ax.scatter(
                np.array([vanilla_df["iteration"][len(vanilla_df) - 1]]) / 1e6,
                np.array([vanilla_df[metric][len(vanilla_df) - 1]]),
                color=vanilla_c,
                marker='X' if vanilla_df["max single"][len(vanilla_df) - 1] ==
                vanilla_df["max single"][len(vanilla_df) - 2] else 'o',
                **scatter_params
            )

            ax.scatter(
                np.array([curriculum_df["iteration"][len(curriculum_df) - 1]]) / 1e6,
                np.array([curriculum_df[metric][len(curriculum_df) - 1]]),
                color=curr_c,
                marker='X' if curriculum_df["max single"][len(curriculum_df) - 1] ==
                curriculum_df["max single"][len(curriculum_df) - 2] else 'o',
                **scatter_params
            )

        ax.set_title(title)
        ax.set_xlabel(x_axis_title)
        ax.set_ylabel(y_axis_title)
        custom_lines = [
            Line2D([0], [0], color=vanilla_c, lw=4),
            Line2D([0], [0], color=curr_c, lw=4),
            Line2D([0], [0], marker="X", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=8, mew=1.5),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=8, mew=1.5),
        ]
        ax.legend(
            custom_lines,
            [
                "Vanilla",
                "Curriculum",
                "Did not achieve max score",
                "Achieved max score"
            ],
            loc="lower right",
        )
        plt.savefig(
            fig_folder / f"{metric.replace(' ', '_')}_{s}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

# Bar Charts

for metric in ["max avg", "max single"]:
    x_axis_title = "S (size of grid)"
    y_axis_title = "Average " + metric_map[metric]

    vanilla_c = "#e74c3c"  # red
    curr_c = "#2ecc71"  # green
    plot_params = {
        "linewidth": 2,
        "alpha": 0.5,
        "zorder": 1,
    }
    scatter_params = {
        "edgecolors": "black",
        "linewidths": 1,
        "s": 50,
        "zorder": 2,
    }

    vanilla_ys = []
    curr_ys = []
    xs = []

    for s in range(4, 11):

        vanilla_scores = []
        curr_scores = []

        for n in range(5):

            vanilla_log = f"logs/log_S_{s}_{n}.csv"
            curriculum_log = f"logs/log_S_{s}_{n}_curr.csv"

            vanilla_df = pd.read_csv(vanilla_log)
            curriculum_df = pd.read_csv(curriculum_log)

            vanilla_scores.append(vanilla_df[metric][len(vanilla_df) - 1])
            curr_scores.append(curriculum_df[metric][len(curriculum_df) - 1])

        vanilla_ys.append(np.mean(vanilla_scores))
        curr_ys.append(np.mean(curr_scores))
        xs.append(s)

    x = np.arange(len(xs))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(
        x - width / 2,
        vanilla_ys, width,
        label="Vanilla",
        color=vanilla_c
    )
    rects2 = ax.bar(
        x + width / 2,
        curr_ys, width,
        label="Curriculum",
        color=curr_c
    )

    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_title(y_axis_title + " Against Size of Grid")
    ax.set_xticks(x)
    ax.set_xticklabels(xs)
    ax.legend()
    plt.savefig(
        fig_folder / f"{metric.replace(' ', '_')}_bars.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
