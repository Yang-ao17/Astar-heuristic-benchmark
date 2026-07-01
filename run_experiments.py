from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd

from src.astar import astar
from src.heuristics import HEURISTICS, h_zero
from src.maps import get_config
from src.metrics import compute_path_cost, path_quality_metrics
from src.visualization import plot_metric_bars, plot_paths_for_map


RESULT_COLUMNS = [
    "map_name",
    "heuristic",
    "path_found",
    "stopped_by_budget",
    "valid_path",
    "path_cost",
    "optimal_cost",
    "optimality_gap",
    "is_optimal",
    "expanded_nodes",
    "runtime_ms",
]


def run_benchmark(map_names, heuristics=HEURISTICS, max_expansions=3000):
    """Run the budgeted A* heuristic benchmark."""
    records = []

    for map_name in map_names:
        grid, start, goal = get_config(map_name)

        # Use unbudgeted h_zero as the Dijkstra-like optimal baseline.
        optimal_path, _, _ = astar(grid, start, goal, h_zero, max_expansions=None)
        optimal_cost = compute_path_cost(optimal_path)

        for heuristic_name, heuristic_fn in heuristics.items():
            start_time = time.perf_counter()
            path, expanded_nodes, stopped_by_budget = astar(
                grid,
                start,
                goal,
                heuristic_fn,
                max_expansions=max_expansions,
            )
            runtime_ms = (time.perf_counter() - start_time) * 1000

            record = {
                "map_name": map_name,
                "heuristic": heuristic_name,
                "stopped_by_budget": stopped_by_budget,
                "expanded_nodes": expanded_nodes,
                "runtime_ms": runtime_ms,
                "path": path,
            }
            record.update(path_quality_metrics(path, grid, start, goal, optimal_cost))
            records.append(record)

    return pd.DataFrame(records)


def print_summary(results):
    """Print a short human-readable summary per map."""
    for map_name, group in results.groupby("map_name"):
        print(f"\nMap: {map_name}")

        valid_successes = group[group["path_found"] & group["valid_path"]]
        found_names = ", ".join(valid_successes["heuristic"]) or "none"
        print(f"  Valid paths found: {found_names}")

        stopped = group[group["stopped_by_budget"]]
        stopped_names = ", ".join(stopped["heuristic"]) or "none"
        print(f"  Stopped by budget: {stopped_names}")

        if len(valid_successes) > 0:
            fewest_idx = valid_successes["expanded_nodes"].idxmin()
            fewest = valid_successes.loc[fewest_idx]
            print(
                "  Fewest expanded nodes among valid successes: "
                f"{fewest['heuristic']} ({fewest['expanded_nodes']} nodes)"
            )
        else:
            print("  Fewest expanded nodes among valid successes: none")

        suboptimal = group[group["optimality_gap"].fillna(0) > 1e-9]
        suboptimal_names = ", ".join(suboptimal["heuristic"]) or "none"
        print(f"  Suboptimal heuristics: {suboptimal_names}")


def save_outputs(results, map_names, results_dir, max_expansions):
    """Save benchmark results and figures."""
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "benchmark_results.csv"
    results.drop(columns=["path"]).to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    for map_name in map_names:
        fig, _ = plot_paths_for_map(map_name, results)
        path_fig = results_dir / f"{map_name}_paths.png"
        fig.savefig(path_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, _ = plot_metric_bars(results, map_name, max_expansions=max_expansions)
        metric_fig = results_dir / f"{map_name}_metrics.png"
        fig.savefig(metric_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved figures to {path_fig} and {metric_fig}")


def main():
    map_names = ["maze", "trap"]
    max_expansions = 3000
    results_dir = Path("results")

    results = run_benchmark(map_names, HEURISTICS, max_expansions=max_expansions)

    print(results[RESULT_COLUMNS].to_string(index=False))
    print_summary(results)
    save_outputs(results, map_names, results_dir, max_expansions)


if __name__ == "__main__":
    main()
