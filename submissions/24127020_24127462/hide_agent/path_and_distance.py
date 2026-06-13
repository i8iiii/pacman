from collections import deque
import numpy as np
from .helpers import (
   _apply_move,
   _get_neighbors,
   _is_valid_move,
   _is_straight_corridor,
   _is_valid_position,
   _translate_move,
)

def _bfs_distances(start: tuple, map_state: np.ndarray) -> dict[tuple, int]:
   dist = {start: 0}
   queue = deque([start])
   while queue:
      pos = queue.popleft()
      for nb in _get_neighbors(pos, map_state):
         if nb not in dist:
            dist[nb] = dist[pos] + 1
            queue.append(nb)
   return dist


def _bfs_path(start: tuple, dest: tuple, map_state: np.ndarray) -> list[tuple]:
   """Shortest path [start … dest]. Returns [] if unreachable."""
   if start == dest:
      return [start]
   parent = {}
   visited = {start}
   queue = deque([start])
   while queue:
      pos = queue.popleft()
      for nb in _get_neighbors(pos, map_state):
         if nb not in visited:
               visited.add(nb)
               parent[nb] = pos
               if nb == dest:
                  path, cur = [], dest
                  while cur != start:
                     path.append(cur)
                     cur = parent[cur]
                  path.append(start)
                  path.reverse()
                  return path
               queue.append(nb)
   return []
