import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.maps import get_config


def plot_grid_and_path(grid, start, goal, path=None, title="A* Path Planning", ax=None):
    """Plot a grid, start/goal markers, and an optional path."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(grid, cmap="gray_r")

    if path:
        x, y = zip(*path)
        ax.plot(y, x, linewidth=2)

    ax.scatter(start[1], start[0], marker="o", s=100, label="start")
    ax.scatter(goal[1], goal[0], marker="x", s=100, label="goal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_paths_for_map(map_name, results):
    """Plot one path panel per heuristic for a single map."""
    grid, start, goal = get_config(map_name)
    map_results = results[results["map_name"] == map_name]

    n = len(map_results)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, map_results.iterrows()):
        if row["path_found"]:
            status = "found"
        elif row["stopped_by_budget"]:
            status = "budget"
        else:
            status = "failed"

        title = f"{row['heuristic']} | {status} | expanded: {row['expanded_nodes']}"
        plot_grid_and_path(grid, start, goal, path=row["path"], title=title, ax=ax)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Paths on map: {map_name}", y=1.02)
    plt.tight_layout()
    return fig, axes


def _add_value_labels(ax, values):
    """Add compact labels above numeric bars."""
    for patch, value in zip(ax.patches, values):
        if pd.isna(value):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )


def plot_metric_bars(results, map_name, max_expansions=None):
    """Plot expanded nodes, runtime, path quality, and budget status."""
    map_results = results[results["map_name"] == map_name].reset_index(drop=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.reshape(-1)

    metrics = [
        ("expanded_nodes", "Expanded Nodes", "nodes"),
        ("runtime_ms", "Runtime", "milliseconds"),
        ("path_cost", "Path Cost", "geometric cost"),
        ("optimality_gap", "Optimality Gap", "cost difference"),
    ]

    for ax, (column, title, ylabel) in zip(axes, metrics):
        values = pd.to_numeric(map_results[column], errors="coerce")
        ax.bar(map_results["heuristic"], values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        _add_value_labels(ax, values)

        if column == "optimality_gap":
            finite_values = values.dropna()
            if len(finite_values) > 0 and np.allclose(finite_values, 0.0):
                ax.text(
                    0.5,
                    0.9,
                    "all successful paths optimal",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=9,
                )

    x = np.arange(len(map_results))
    width = 0.35
    axes[4].bar(x - width / 2, map_results["path_found"].astype(int), width, label="path_found")
    axes[4].bar(
        x + width / 2,
        map_results["stopped_by_budget"].astype(int),
        width,
        label="stopped_by_budget",
    )
    axes[4].set_title("Search Status")
    axes[4].set_ylabel("true = 1")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(map_results["heuristic"], rotation=30)
    axes[4].set_ylim(0, 1.2)
    axes[4].legend()

    axes[5].axis("off")
    suffix = f" | max_expansions={max_expansions}" if max_expansions is not None else ""
    fig.suptitle(f"Metrics on map: {map_name}{suffix}", y=1.02)
    plt.tight_layout()
    return fig, axes
