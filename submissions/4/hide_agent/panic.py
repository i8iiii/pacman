# hide_agent/panic.py

from environment import Move
from debug import debug

from .core import (
   INF,
   manhattan,
   next_position,
   is_valid_position,
   is_capture,
   legal_ghost_moves,
   legal_pacman_actions,
   apply_pacman_action,
   capture_turn_distance,
   exit_count,
   dead_end_depth,
)

PANIC_TURNS = 2

def should_panic(
   pacman_pos: tuple[int, int],
   ghost_pos: tuple[int, int],
   map_state,
   pacman_speed: int,
) -> bool:
   if is_capture(pacman_pos, ghost_pos):
      return True

   turns = capture_turn_distance(
      pacman_pos,
      ghost_pos,
      map_state,
      pacman_speed,
   )

   return turns <= PANIC_TURNS

# hide_agent/panic.py

from environment import Move

from .core import (
   INF,
   next_position,
   legal_ghost_moves,
   bfs_distances,
)

def choose_move(map_state, ghost_pos: tuple, pacman_pos: tuple) -> Move:
   pacman_pos = (int(pacman_pos[0]), int(pacman_pos[1]))
   ghost_pos = (int(ghost_pos[0]), int(ghost_pos[1]))

   distances = bfs_distances(pacman_pos, map_state)

   best_move = Move.STAY
   best_distance = -1

   for move in legal_ghost_moves(ghost_pos, map_state):
      next_pos = next_position(ghost_pos, move)
      distance = distances.get(next_pos, INF)

      debug.candidate(move, next_pos, distance=distance)

      if distance > best_distance:
         best_distance = distance
         best_move = move

   best_position = next_position(ghost_pos, best_move)
   debug.decision(best_move, best_position, "greatest maze distance")
   return best_move
