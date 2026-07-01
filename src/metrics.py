import numpy as np


def is_valid_path(path, grid, start, goal):
    """Check whether a path is a valid 8-neighbor grid path."""
    if path is None or len(path) == 0:
        return False

    if path[0] != start or path[-1] != goal:
        return False

    for cell in path:
        row, col = cell
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            return False
        if grid[cell] == 1:
            return False

    for current, next_cell in zip(path, path[1:]):
        dr = next_cell[0] - current[0]
        dc = next_cell[1] - current[1]
        row_diff = abs(dr)
        col_diff = abs(dc)

        if row_diff > 1 or col_diff > 1 or (row_diff == 0 and col_diff == 0):
            return False

        # Diagonal moves are allowed, but not through obstacle corners.
        if row_diff == 1 and col_diff == 1:
            if grid[current[0] + dr, current[1]] == 1:
                return False
            if grid[current[0], current[1] + dc] == 1:
                return False

    return True


def compute_path_cost(path):
    """Sum cardinal and diagonal movement costs for an 8-neighbor path."""
    if path is None:
        return None

    cost = 0.0
    for current, next_cell in zip(path, path[1:]):
        row_diff = abs(current[0] - next_cell[0])
        col_diff = abs(current[1] - next_cell[1])

        if row_diff + col_diff == 1:
            cost += 1.0
        elif row_diff == 1 and col_diff == 1:
            cost += np.sqrt(2)
        else:
            return None

    return cost


def path_quality_metrics(path, grid, start, goal, optimal_cost):
    """Compute path quality metrics for one A* run."""
    found = path is not None
    valid = is_valid_path(path, grid, start, goal)
    cost = compute_path_cost(path)
    gap = cost - optimal_cost if cost is not None and optimal_cost is not None else None

    return {
        "path_found": found,
        "valid_path": valid,
        "path_cost": cost,
        "optimal_cost": optimal_cost,
        "optimality_gap": gap,
        "is_optimal": np.isclose(gap, 0.0) if gap is not None else False,
    }
