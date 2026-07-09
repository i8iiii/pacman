from collections import deque

from environment import Move

from . import core

CAPTURE_SCORE = 1_000_000

W_CAPTURE_TURNS = 1200
W_MAZE_DISTANCE = 40
W_TRAP = 700
W_TOPOLOGY = 0.05

STAY_PENALTY = 200
DANGER_STAY_PENALTY = 2500
LOST_TEMPO_PENALTY = 1300

REVERSAL_PENALTY = 0

APPROACH_PENALTY = 900
AWAY_BONUS = 300

SAFE_AREA_DEPTH = 5
SAFE_AREA_BONUS = 180
LOW_SAFE_AREA_PENALTY = 1200
MIN_SAFE_AREA = 4

SIMULATION_DEPTH = 2

try:
   from debug import debug
except Exception:
   debug = None


def log(message: str) -> None:
   if debug is None:
      return

   try:
      debug.log(message)
   except Exception:
      pass

def cached_bfs_distances(
   start: tuple[int, int],
   map_state,
   distance_cache: dict,
) -> dict:
   if start not in distance_cache:
      distance_cache[start] = core.bfs_distances(start, map_state)

   return distance_cache[start]


def cached_capture_turn_distance(
   pacman_pos: tuple[int, int],
   ghost_pos: tuple[int, int],
   map_state,
   pacman_speed: int,
   capture_cache: dict,
) -> int:
   key = (pacman_pos, ghost_pos, pacman_speed)

   if key not in capture_cache:
      capture_cache[key] = core.capture_turn_distance(
         pacman_pos,
         ghost_pos,
         map_state,
         pacman_speed,
      )

   return capture_cache[key]


def simulate_pacman_turn(
   pacman_pos: tuple[int, int],
   ghost_pos: tuple[int, int],
   map_state,
   topology_map: dict,
   pacman_speed: int,
   depth: int,
   cache: dict,
   distance_cache: dict,
   capture_cache: dict,
) -> float:
   """
   Pacman moves next.
   Pacman chooses the action that gives the highest Pacman utility.
   """

   if core.is_capture(pacman_pos, ghost_pos):
      return CAPTURE_SCORE

   key = ("P", pacman_pos, ghost_pos, depth)
   if key in cache:
      return cache[key]

   if depth <= 0:
      value = evaluate_pacman_utility(
         pacman_pos,
         ghost_pos,
         map_state,
         pacman_speed,
         topology_map,
         distance_cache,
         capture_cache,
      )
      cache[key] = value
      return value

   best_utility = -core.INF

   for pacman_action in core.legal_pacman_actions(
      pacman_pos,
      map_state,
      pacman_speed,
      allow_stay=False,
   ):
      new_pacman_pos = core.apply_pacman_action(
         pacman_pos,
         pacman_action,
         map_state,
      )

      if core.is_capture(new_pacman_pos, ghost_pos):
         utility = CAPTURE_SCORE
      else:
         utility = simulate_ghost_turn(
            new_pacman_pos,
            ghost_pos,
            map_state,
            topology_map,
            pacman_speed,
            depth,
            cache,
            distance_cache,
            capture_cache,
         )

      if utility > best_utility:
         best_utility = utility

   cache[key] = best_utility
   return best_utility

def evaluate_pacman_utility(
   pacman_pos: tuple[int, int],
   ghost_pos: tuple[int, int],
   map_state,
   pacman_speed: int,
   topology_map: dict,
   distance_cache: dict | None = None,
   capture_cache: dict | None = None,
) -> float:
   """
   Higher score = better for Pacman.
   Lower score = better for Ghost.
   """

   if core.is_capture(pacman_pos, ghost_pos):
      return CAPTURE_SCORE

   if capture_cache is None:
      capture_turns = core.capture_turn_distance(
         pacman_pos,
         ghost_pos,
         map_state,
         pacman_speed,
      )
   else:
      capture_turns = cached_capture_turn_distance(
         pacman_pos,
         ghost_pos,
         map_state,
         pacman_speed,
         capture_cache,
      )

   if distance_cache is None:
      pacman_distances = core.bfs_distances(pacman_pos, map_state)
   else:
      pacman_distances = cached_bfs_distances(
         pacman_pos,
         map_state,
         distance_cache,
      )

   maze_distance = pacman_distances.get(ghost_pos, core.INF)

   exits = core.exit_count(ghost_pos, map_state)

   topology_score = 0
   if topology_map is not None and ghost_pos in topology_map:
      topology_score = topology_map[ghost_pos]["score"]

   utility = 0

   if capture_turns <= 1:
      utility += 500_000
   elif capture_turns == 2:
      utility += 30_000
   elif capture_turns == 3:
      utility += 8_000
   elif capture_turns == 4:
      utility += 2_000
   else:
      utility -= W_CAPTURE_TURNS * min(capture_turns, 6)

   utility -= W_MAZE_DISTANCE * min(maze_distance, 40)
   utility += W_TRAP * (4 - exits)

   # Topology is only a tie-breaker now.
   utility -= W_TOPOLOGY * topology_score

   return utility

def simulate_ghost_turn(
   pacman_pos: tuple[int, int],
   ghost_pos: tuple[int, int],
   map_state,
   topology_map: dict,
   pacman_speed: int,
   depth: int,
   cache: dict,
   distance_cache: dict,
   capture_cache: dict,
) -> float:
   """
   Ghost moves next.
   Ghost chooses the action that gives the lowest Pacman utility.
   """

   if core.is_capture(pacman_pos, ghost_pos):
      return CAPTURE_SCORE

   key = ("G", pacman_pos, ghost_pos, depth)
   if key in cache:
      return cache[key]

   if depth <= 0:
      value = evaluate_pacman_utility(
         pacman_pos,
         ghost_pos,
         map_state,
         pacman_speed,
         topology_map,
         distance_cache,
         capture_cache,
      )
      cache[key] = value
      return value

   best_utility = core.INF

   for ghost_move in core.legal_ghost_moves(ghost_pos, map_state):
      new_ghost_pos = core.next_position(ghost_pos, ghost_move)

      if not core.is_valid_position(new_ghost_pos, map_state):
         continue

      if core.is_capture(pacman_pos, new_ghost_pos):
         utility = CAPTURE_SCORE
      else:
         utility = simulate_pacman_turn(
            pacman_pos,
            new_ghost_pos,
            map_state,
            topology_map,
            pacman_speed,
            depth - 1,
            cache,
            distance_cache,
            capture_cache,
         )

      if utility < best_utility:
         best_utility = utility

   cache[key] = best_utility
   return best_utility


def predicted_safe_area(
   ghost_pos: tuple[int, int],
   pacman_pos: tuple[int, int],
   map_state,
   pacman_speed: int,
   max_depth: int = SAFE_AREA_DEPTH,
   capture_cache: dict | None = None,
) -> int:
   """
   Count how much future escape space Ghost has after Pacman's predicted move.

   A cell is considered useful if Ghost can reach it before Pacman can
   threaten its capture zone.
   """

   if not core.is_valid_position(ghost_pos, map_state):
      return 0

   safe_cells = 0

   queue = deque([(ghost_pos, 0)])
   visited = {ghost_pos}

   while queue:
      current, ghost_steps = queue.popleft()

      if capture_cache is None:
         capture_turns = core.capture_turn_distance(
            pacman_pos,
            current,
            map_state,
            pacman_speed,
         )
      else:
         capture_turns = cached_capture_turn_distance(
            pacman_pos,
            current,
            map_state,
            pacman_speed,
            capture_cache,
         )

      if capture_turns > ghost_steps + 1:
         safe_cells += 1

      if ghost_steps >= max_depth:
         continue

      for new_pos in core.legal_ghost_positions(current, map_state):
         if new_pos != current and new_pos not in visited:
            visited.add(new_pos)
            queue.append((new_pos, ghost_steps + 1))

   return safe_cells

def choose_move(
   ghost_pos: tuple[int, int],
   pacman_pos: tuple[int, int],
   map_state,
   topology_map: dict,
   pacman_speed: int = 2,
   previous_position: tuple[int, int] | None = None,
) -> Move:
   """
   Ghost chooses the move that minimizes Pacman's best response.
   """

   log("[CONTROL-SIM]")
   log(f" Pacman={pacman_pos}, Ghost={ghost_pos}")

   candidates = []
   best_non_stay_capture_turns = -1
   simulation_cache = {}
   distance_cache = {}
   capture_cache = {}

   for ghost_move in core.legal_ghost_moves(ghost_pos, map_state):
      new_ghost_pos = core.next_position(ghost_pos, ghost_move)

      if not core.is_valid_position(new_ghost_pos, map_state):
         continue

      worst_utility = -core.INF
      best_pacman_action = None
      best_pacman_pos = pacman_pos

      for pacman_action in core.legal_pacman_actions(
         pacman_pos,
         map_state,
         pacman_speed,
         allow_stay=False,
      ):
         new_pacman_pos = core.apply_pacman_action(
               pacman_pos,
               pacman_action,
               map_state,
         )

         if core.is_capture(new_pacman_pos, new_ghost_pos):
            utility = CAPTURE_SCORE
         else:
            utility = simulate_ghost_turn(
               new_pacman_pos,
               new_ghost_pos,
               map_state,
               topology_map,
               pacman_speed,
               SIMULATION_DEPTH,
               simulation_cache,
               distance_cache,
               capture_cache,
            )

         if utility > worst_utility:
               worst_utility = utility
               best_pacman_action = pacman_action
               best_pacman_pos = new_pacman_pos

      response_capture_turns = cached_capture_turn_distance(
         best_pacman_pos,
         new_ghost_pos,
         map_state,
         pacman_speed,
         capture_cache,
      )

      topo = 0
      if topology_map is not None and new_ghost_pos in topology_map:
         topo = topology_map[new_ghost_pos]["score"]

      if ghost_move != Move.STAY:
         best_non_stay_capture_turns = max(
               best_non_stay_capture_turns,
               response_capture_turns,
         )

      candidates.append({
         "move": ghost_move,
         "position": new_ghost_pos,
         "topology": topo,
         "pacman_action": best_pacman_action,
         "pacman_position": best_pacman_pos,
         "capture_turns": response_capture_turns,
         "worst_utility": worst_utility,
      })
   if not candidates:
      return Move.STAY
   
   best_move = Move.STAY
   best_position = ghost_pos
   best_final_score = core.INF

   for candidate in candidates:
      final_score = candidate["worst_utility"]

      if candidate["move"] == Move.STAY:
         final_score += STAY_PENALTY

         if candidate["capture_turns"] <= 5:
               final_score += DANGER_STAY_PENALTY

         if candidate["capture_turns"] <= best_non_stay_capture_turns:
               final_score += LOST_TEMPO_PENALTY

      pacman_distances = cached_bfs_distances(
         candidate["pacman_position"],
         map_state,
         distance_cache,
      )

      current_distance_after_pacman = pacman_distances.get(
         ghost_pos,
         core.INF,
      )

      candidate_distance_after_pacman = pacman_distances.get(
         candidate["position"],
         core.INF,
      )

      direction_delta = (
         candidate_distance_after_pacman
         - current_distance_after_pacman
      )

      if direction_delta < 0:
         final_score += APPROACH_PENALTY * abs(direction_delta)

      elif direction_delta > 0:
         final_score -= AWAY_BONUS * min(direction_delta, 2)

      safe_area = predicted_safe_area(
         candidate["position"],
         candidate["pacman_position"],
         map_state,
         pacman_speed,
         capture_cache=capture_cache,
      )

      final_score -= SAFE_AREA_BONUS * safe_area

      if safe_area < MIN_SAFE_AREA:
         final_score += LOW_SAFE_AREA_PENALTY

      reversal = (
         candidate["move"] != Move.STAY
         and previous_position is not None
         and candidate["position"] == previous_position
      )

      if reversal:
         final_score += REVERSAL_PENALTY

      log(
         f"   Candidate {candidate['move']}, "
         f"to={candidate['position']}, "
         f"topology={candidate['topology']}, "
         f"pacman_best={candidate['pacman_action']}->{candidate['pacman_position']}, "
         f"capture_turns={candidate['capture_turns']}, "
         f"direction_delta={direction_delta}, "
         f"safe_area={safe_area}, "
         # f"reversal={reversal}, "
         f"worst_utility={candidate['worst_utility']}, "
         f"final={final_score}"
      )

      if final_score < best_final_score:
         best_final_score = final_score
         best_move = candidate["move"]
         best_position = candidate["position"]

   log(
      f"   Chosen move={best_move}, "
      f"from={ghost_pos}, "
      f"to={best_position}, "
      f"score={best_final_score}"
   )

   log(
      f"   Cache sizes: "
      f"sim={len(simulation_cache)}, "
      f"bfs={len(distance_cache)}, "
      f"capture={len(capture_cache)}"
   )

   return best_move