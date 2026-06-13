from environment import Move
import numpy as np

def _is_valid_position(pos: tuple, map_state: np.ndarray) -> bool:
   row, col = pos
   height, width = map_state.shape
   if row < 0 or row >= height or col < 0 or col >= width:
      return False
   return map_state[row, col] == 0


def _is_valid_move(pos: tuple, move: Move, map_state: np.ndarray) -> bool:
   return _is_valid_position(_apply_move(pos, move), map_state)

def _apply_move(pos: tuple, move: Move) -> tuple:
   row, col = pos
   deltas = {
      Move.UP: (-1, 0),
      Move.DOWN: (1, 0),
      Move.LEFT: (0, -1),
      Move.RIGHT: (0, 1),
   }
   dr, dc = deltas.get(move, (0, 0))
   return (row + dr, col + dc)

def _translate_move(current: tuple, next_pos: tuple) -> Move:
   dr = next_pos[0] - current[0]
   dc = next_pos[1] - current[1]
   return {
      (-1, 0): Move.UP,
      (1, 0): Move.DOWN,
      (0, -1): Move.LEFT,
      (0, 1): Move.RIGHT,
   }.get((dr, dc), Move.STAY)


def _get_neighbors(pos: tuple, map_state: np.ndarray) -> list[tuple]:
   return [
      _apply_move(pos, move)
      for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
      if _is_valid_position(_apply_move(pos, move), map_state)
   ]

def _is_straight_corridor(pos: tuple, neighbors: list) -> bool:
   """True if degree-2 tile is a straight corridor (not a corner)."""
   dr1 = neighbors[0][0] - pos[0]
   dc1 = neighbors[0][1] - pos[1]
   dr2 = neighbors[1][0] - pos[0]
   dc2 = neighbors[1][1] - pos[1]
   return dr1 == -dr2 and dc1 == -dc2
