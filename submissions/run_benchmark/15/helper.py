def _evaluation_heuristic(my_position, enemy_position, map_state):
    # FIX: Calculate actual shortest maze walking distance instead of Manhattan distance
    return -_bfs_maze_distance(my_position, enemy_position, map_state)


def _bfs_maze_distance(start, goal, map_state):
    if start == goal:
        return 0
    queue = [(start, 0)]
    visited = {start}
    while len(queue) > 0:
        curr, dist = queue.pop(0)
        if abs(curr[0] - goal[0]) + abs(curr[1] - goal[1]) < 2:
            return dist + 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr[0] + dr, curr[1] + dc
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1]:
                if map_state[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
    return 100

def _manhattan(my_position, enemy_position):
    return abs(my_position[0] - enemy_position[0]) + abs(my_position[1] - enemy_position[1])