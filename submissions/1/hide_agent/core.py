import numpy as np
from functools import lru_cache
from collections import deque
from . import helpers

def bfs(start: tuple, destination: tuple, map_state: np.ndarray) -> list[tuple]:
   """
   Return the shortest path from start to destination, inclusive.

   Returns:
      [start, ..., destination]

   Returns:
      [] if no path exists.
   """
   if tuple(start) == tuple(destination):
      return [start]

   queue = deque([start])
   parent = {start: None}

   while queue:
      current = queue.popleft()

      for neighbor in helpers.get_neighbors(current, map_state):
         if neighbor in parent:
            continue

         parent[neighbor] = current

         if neighbor == destination:
            path = [destination]

            while parent[path[-1]] is not None:
               path.append(parent[path[-1]])

            path.reverse()
            return path

         queue.append(neighbor)

   return []

def enemy_next_position(my_pos: tuple, enemy: tuple, map_state: np.ndarray) -> tuple:
   path = bfs(enemy, my_pos, map_state)

   if len(path) < 2:
      return enemy

   # Always take the first step
   next_pos = path[1]

   # No second step available
   if len(path) < 3:
      return next_pos

   d1 = (
      path[1][0] - path[0][0],
      path[1][1] - path[0][1],
   )

   d2 = (
      path[2][0] - path[1][0],
      path[2][1] - path[1][1],
   )

   # Second step only if continuing straight
   if d1 == d2:
      return path[2]

   return next_pos

def simulate(my_pos: tuple, threat: tuple, map_state: np.ndarray, turns=3):
   pacman = threat

   for _ in range(turns):
      pacman = enemy_next_position(
         my_pos,
         pacman,
         map_state,
      )

   return len(bfs(my_pos, pacman, map_state)) - 1