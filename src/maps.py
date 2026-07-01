import numpy as np


def get_config(name):
    """Return (grid, start, goal) for a named benchmark map."""
    if name == "small":
        grid = np.zeros((10, 10))
        grid[3, 1:7] = 1
        grid[6, 1:5] = 1
        start = (2, 2)
        goal = (9, 9)

    elif name == "medium":
        grid = np.zeros((20, 20))
        grid[5, 2:15] = 1
        grid[10, 5:18] = 1
        grid[15, 0:10] = 1
        start = (0, 7)
        goal = (19, 5)

    elif name == "large":
        grid = np.zeros((100, 100))
        grid[30, 10:90] = 1
        grid[60, 20:80] = 1
        grid[30, 45:55] = 0
        grid[60, 50:60] = 0
        grid[20:80, 70] = 1
        grid[40:50, 70] = 0
        start = (20, 0)
        goal = (99, 99)

    elif name == "maze":
        grid = np.zeros((64, 64))
        grid[[0, -1], :] = 1
        grid[:, [0, -1]] = 1

        for col in range(4, 60, 4):
            grid[1:63, col] = 1
            if (col // 4) % 2 == 1:
                grid[1:8, col] = 0
            else:
                grid[56:63, col] = 0

        start = (1, 1)
        goal = (62, 62)
        grid[start] = 0
        grid[goal] = 0

    elif name == "trap":
        grid = np.zeros((20, 20))
        grid[[0, -1], :] = 1
        grid[:, [0, -1]] = 1

        obstacles = [
            (1, 12),
            (2, 8), (2, 13),
            (3, 2), (3, 9),
            (4, 8), (4, 12), (4, 16),
            (5, 8), (5, 11), (5, 13), (5, 17),
            (6, 8),
            (7, 3), (7, 6), (7, 10), (7, 12), (7, 17),
            (9, 2), (9, 13), (9, 16),
            (10, 7), (10, 12), (10, 14),
            (11, 1), (11, 2), (11, 3), (11, 14),
            (12, 8), (12, 17),
            (13, 7), (13, 9), (13, 12),
            (14, 9), (14, 11), (14, 16),
            (15, 2), (15, 15),
            (16, 3), (16, 13),
            (17, 5), (17, 14),
            (18, 4), (18, 5), (18, 9), (18, 16),
        ]
        for obstacle in obstacles:
            grid[obstacle] = 1

        start = (1, 1)
        goal = (18, 18)
        grid[start] = 0
        grid[goal] = 0

    else:
        raise ValueError("Choose one of: 'small', 'medium', 'large', 'maze', or 'trap'.")

    return grid, start, goal
