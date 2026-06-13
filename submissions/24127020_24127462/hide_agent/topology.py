from collections import deque
from .helpers import (
   _apply_move,
   _get_neighbors,
   _is_valid_move,
   _is_straight_corridor,
   _is_valid_position,
   _translate_move,
)
import numpy as np


def _local_space_score(
   start: tuple[int, int], map_state: np.ndarray, radius: int = 6
) -> float:
   visited = {start}
   queue = deque([(start, 0)])
   count = 0

   while queue:
      pos, depth = queue.popleft()

      if depth >= radius:
         continue

      for nb in _get_neighbors(pos, map_state):
         if nb not in visited:
               visited.add(nb)
               queue.append((nb, depth + 1))
               count += 1

   return count


def _topology_score(pos: tuple, map_state: np.ndarray) -> float:
   neighbors = _get_neighbors(pos, map_state)
   degree = len(neighbors)

   if degree <= 1:
      return -200.0

   if degree == 2:
      dr1 = neighbors[0][0] - pos[0]
      dc1 = neighbors[0][1] - pos[1]
      dr2 = neighbors[1][0] - pos[0]
      dc2 = neighbors[1][1] - pos[1]
      if dr1 == -dr2 and dc1 == -dc2:
         return 0.0  # straight corridor
      return 10.0  # corner

   if degree == 3:
      return 25.0

   return 30.0


from pathlib import Path



