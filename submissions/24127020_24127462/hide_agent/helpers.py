from environment import Move
import numpy as np

# BASIC HELPERS

def is_valid_position(pos: tuple, map_state):
   """Check if position is valid (not wall or unseen boundaries)."""
   row, col = pos
   height, width = map_state.shape
   
   # Check bounds
   if row < 0 or row >= height or col < 0 or col >= width:
      return False
   
   # Check not a wall 
   # (Optional: treat unseen (-1) carefully based on your strategy!)
   return map_state[row, col] == 0

def translate_move(current: tuple, next_pos: tuple) -> Move:
   dr = next_pos[0] - current[0]
   dc = next_pos[1] - current[1]
   return {
      (-1, 0): Move.UP,
      (1, 0): Move.DOWN,
      (0, -1): Move.LEFT,
      (0, 1): Move.RIGHT,
   }.get((dr, dc), Move.STAY)

def apply_move(pos: tuple, move) -> tuple:
   """Apply a move to a position, return new position."""
   delta_row, delta_col = move.value
   return (pos[0] + delta_row, pos[1] + delta_col)

def get_neighbors(pos: tuple, map_state: np.ndarray) -> list[tuple[int, int]]:
   """Get all valid neighboring positions and their moves."""
   neighbors = []
   
   for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
      next_pos = apply_move(pos, move)
      if is_valid_position(next_pos, map_state):
         neighbors.append(next_pos)
   
   return neighbors

# CUSTOM HELPERS
def list_dead_end_cells(map_state: np.ndarray) -> list[tuple]:
   """
   Return all cells that belong to dead-end corridors.
   Junction cells are not included.
   """
   dead_end_cells = set()

   rows, cols = map_state.shape

   for row in range(rows):
      for col in range(cols):
         start = (row, col)

         if not is_valid_position(start, map_state):
            continue

         if len(get_neighbors(start, map_state)) != 1:
            continue

         previous = None
         current = start

         while current not in dead_end_cells:
            dead_end_cells.add(current)

            neighbors = get_neighbors(current, map_state)

            # Terminal dead end
            if len(neighbors) == 1:
               next_cells = neighbors

            # Corridor cell
            elif len(neighbors) == 2:
               next_cells = [
                  cell for cell in neighbors
                  if cell != previous
               ]

            # Reached gate/junction
            else:
               break

            if not next_cells:
               break

            previous, current = current, next_cells[0]

   return list(dead_end_cells)

def build_dead_end_exit_map(map_state: np.ndarray) -> dict[tuple, tuple]:
   dead_end_cells = set(list_dead_end_cells(map_state))
   exit_map = {}

   for cell in dead_end_cells:
      # Walk corridor until we hit the junction
      prev, cur = None, cell
      while cur in dead_end_cells:
         neighbors = [n for n in get_neighbors(cur, map_state) if n != prev]
         if not neighbors:
               break
         prev, cur = cur, neighbors[0]
      exit_map[cell] = cur  # cur is now the junction

   return exit_map

