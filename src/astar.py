from heapq import heappop, heappush

import numpy as np


def get_neighbors(grid, node):
    """Yield valid 8-neighbor moves as (neighbor, move_cost) pairs."""
    row, col = node

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue

            neighbor = (row + dr, col + dc)
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            if grid[neighbor] == 1:
                continue

            is_diagonal = dr != 0 and dc != 0
            if is_diagonal:
                # Prevent diagonal corner-cutting through blocked cardinal cells.
                if grid[row + dr, col] == 1 or grid[row, col + dc] == 1:
                    continue
                move_cost = np.sqrt(2)
            else:
                move_cost = 1.0

            yield neighbor, move_cost


def astar(grid, start, goal, heuristic_fn, max_expansions=None):
    """Run budgeted A* search on a grid."""
    # open_set is a priority queue of nodes to explore next.
    # The heap always pops the node with the smallest f-score.
    open_set = []
    heappush(open_set, (heuristic_fn(start, goal), 0.0, start))

    # came_from stores the best previous node for each visited node.
    # It is used to reconstruct the final path after reaching the goal.
    came_from = {}

    # g_score[node] is the best known movement cost from start to node.
    # Cardinal moves cost 1.0 and diagonal moves cost sqrt(2).
    g_score = {start: 0.0}

    expanded_nodes = 0

    while open_set:
        _, current_g, current = heappop(open_set)

        # Skip old heap entries that are worse than a path found later.
        if current_g > g_score[current]:
            continue

        expanded_nodes += 1

        # max_expansions is a practical search budget. If the search expands
        # more nodes than the budget allows, we stop and report failure.
        if max_expansions is not None and expanded_nodes > max_expansions:
            return None, expanded_nodes, True

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], expanded_nodes, False

        for neighbor, move_cost in get_neighbors(grid, current):
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, np.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                # f = g + h:
                # g is the cost so far, h is the heuristic estimate to goal.
                h = heuristic_fn(neighbor, goal)
                f = tentative_g + h
                heappush(open_set, (f, tentative_g, neighbor))

    return None, expanded_nodes, False
