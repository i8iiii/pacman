from collections import deque

from environment import Move

from . import core

CAPTURE_SCORE = 1_000_000

W_CAPTURE_TURNS = 1200
W_MAZE_DISTANCE = 40
W_TRAP = 300
W_TOPOLOGY = 0.05

STAY_PENALTY = 200
DANGER_STAY_PENALTY = 2500
LOST_TEMPO_PENALTY = 1300

REVERSAL_PENALTY = 0

APPROACH_PENALTY = 900
AWAY_BONUS = 300

SAFE_AREA_BONUS = 180
LOW_SAFE_AREA_PENALTY = 1200
MIN_SAFE_AREA = 4

SIMULATION_DEPTH = 2
SAFE_AREA_DEPTH = 3
__TIMEOUT__ = 850

from debug import debug
from time import perf_counter

__START__ = 0.0

def reset_timer():
   global __START__
   __START__ = perf_counter()

def get_run_time():
   return (perf_counter() - __START__) * 1000

def timed_out() -> bool:
   return get_run_time() >= __TIMEOUT__

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

   if depth <= 0 or timed_out():
      value = evaluate_pacman_utility(
         pacman_pos,
         ghost_pos,
         map_state,
         pacman_speed,
         topology_map,
         distance_cache,
         capture_cache,
      )
      if not timed_out():
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

   if depth <= 0 or timed_out():
      value = evaluate_pacman_utility(
         pacman_pos,
         ghost_pos,
         map_state,
         pacman_speed,
         topology_map,
         distance_cache,
         capture_cache,
      )
      if not timed_out():
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
   max_depth: int,
   capture_cache: dict | None = None,
   blocked_pos: tuple[int, int] | None = None,
) -> int:
   if not core.is_valid_position(ghost_pos, map_state):
      return 0

   safe_cells = 0
   queue = deque([(ghost_pos, 0)])
   visited = {ghost_pos}

   if blocked_pos is not None:
      visited.add(blocked_pos)

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

      # Do not count or travel through an unsafe cell.
      if capture_turns <= ghost_steps + 1:
         continue

      safe_cells += 1

      if ghost_steps >= max_depth:
         continue

      for new_pos in core.legal_ghost_positions(current, map_state):
         if new_pos != current and new_pos not in visited:
            visited.add(new_pos)
            queue.append((new_pos, ghost_steps + 1))

   return safe_cells

DEAD_END_ESCAPE_RESERVE = 2


def decisive_dead_end_move(
   ghost_pos: tuple[int, int],
   pacman_pos: tuple[int, int],
   map_state,
   pacman_speed: int,
   previous_position: tuple[int, int] | None = None,
) -> Move | None:
   """
   Return a decisive move while inside a dead end.

   None:
      Ghost is not inside a dead end. Use normal control logic.

   Move toward mouth:
      Escaping has enough time margin, or escape already started.

   STAY:
      The exit route does not have enough margin. Do not wait and
      then perform a late escape directly toward Pacman.
   """

   dead_end_mouth = find_dead_end_mouth(
      ghost_pos,
      map_state,
   )

   if dead_end_mouth is None:
      return None

   mouth_distances = core.bfs_distances(
      dead_end_mouth,
      map_state,
   )

   current_depth = mouth_distances.get(
      ghost_pos,
      core.INF,
   )

   escape_move = Move.STAY
   escape_position = ghost_pos
   escape_depth = current_depth

   # Find the move that directly reduces distance to the mouth.
   for move in core.legal_ghost_moves(
      ghost_pos,
      map_state,
   ):
      if move == Move.STAY:
         continue

      new_position = core.next_position(
         ghost_pos,
         move,
      )

      new_depth = mouth_distances.get(
         new_position,
         core.INF,
      )

      if new_depth < escape_depth:
         escape_move = move
         escape_position = new_position
         escape_depth = new_depth

   if escape_move == Move.STAY:
      log(
         f"[DEAD-END] mouth={dead_end_mouth}, "
         f"depth={current_depth}, "
         f"decision={Move.STAY}, "
         f"reason=no-exit-move"
      )
      return Move.STAY

   pacman_turns_to_mouth = core.capture_turn_distance(
      pacman_pos,
      dead_end_mouth,
      map_state,
      pacman_speed,
   )

   escape_slack = (
      pacman_turns_to_mouth
      - current_depth
   )

   # If the previous position was deeper in this same dead end,
   # the Ghost has already committed to escaping.
   continuing_escape = False
   previous_depth = None

   if previous_position is not None:
      previous_depth = mouth_distances.get(
         previous_position,
         core.INF,
      )

      continuing_escape = (
         previous_depth > current_depth
      )

   # Escaping has merit only when the Ghost can reach the mouth
   # with enough time left to clear the junction.
   escape_has_merit = (
      escape_slack >= DEAD_END_ESCAPE_RESERVE
   )

   if continuing_escape or escape_has_merit:
      decision = escape_move
      reason = (
         "continuing-escape"
         if continuing_escape
         else "escape-has-merit"
      )
   else:
      decision = Move.STAY
      reason = "escape-not-viable"

   log(
      f"[DEAD-END] "
      f"mouth={dead_end_mouth}, "
      f"depth={current_depth}, "
      f"pacman_turns_to_mouth={pacman_turns_to_mouth}, "
      f"slack={escape_slack}, "
      f"reserve={DEAD_END_ESCAPE_RESERVE}, "
      f"previous_depth={previous_depth}, "
      f"continuing={continuing_escape}, "
      f"escape_has_merit={escape_has_merit}, "
      f"escape_move={escape_move}"
      f"->{escape_position}, "
      f"decision={decision}, "
      f"reason={reason}"
   )

   return decision

def find_dead_end_mouth(
   pos: tuple[int, int],
   map_state,
) -> tuple[int, int] | None:
   """
   Return the sole junction connecting the current dead-end branch
   to the rest of the map.

   Return None when the position is:
   - already at a junction;
   - in a corridor connecting multiple junctions;
   - inside a loop.
   """

   if not core.is_valid_position(pos, map_state):
      return None

   if core.exit_count(pos, map_state) >= 3:
      return None

   queue = deque([pos])
   visited = {pos}

   boundary_junctions = set()
   has_endpoint = False

   while queue:
      current = queue.popleft()
      degree = core.exit_count(current, map_state)

      if degree <= 1:
         has_endpoint = True

      for move in core.PACMAN_MOVES:
         new_pos = core.next_position(
            current,
            move,
         )

         if not core.is_valid_position(
            new_pos,
            map_state,
         ):
            continue

         if core.exit_count(new_pos, map_state) >= 3:
            boundary_junctions.add(new_pos)
            continue

         if new_pos not in visited:
            visited.add(new_pos)
            queue.append(new_pos)

   if (
      has_endpoint
      and len(boundary_junctions) == 1
   ):
      return next(iter(boundary_junctions))

   return None


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

   log(
      f"[CONTROL-SIM] "
      f"Pacman={pacman_pos}, "
      f"Ghost={ghost_pos}"
   )

   dead_end_move = decisive_dead_end_move(
      ghost_pos,
      pacman_pos,
      map_state,
      pacman_speed,
      previous_position,
   )

   if dead_end_move is not None:
      return dead_end_move
   
   candidates = []

   simulation_cache = {}
   distance_cache = {}
   capture_cache = {}

   for ghost_move in core.legal_ghost_moves(
      ghost_pos,
      map_state,
   ):
      new_ghost_pos = core.next_position(
         ghost_pos,
         ghost_move,
      )

      if not core.is_valid_position(
         new_ghost_pos,
         map_state,
      ):
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

         if core.is_capture(
            new_pacman_pos,
            new_ghost_pos,
         ):
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

      response_capture_turns = (
         cached_capture_turn_distance(
            best_pacman_pos,
            new_ghost_pos,
            map_state,
            pacman_speed,
            capture_cache,
         )
      )

      topology_score = 0

      if (
         topology_map is not None
         and new_ghost_pos in topology_map
      ):
         topology_score = (
            topology_map[new_ghost_pos]["score"]
         )

      candidates.append({
         "move": ghost_move,
         "position": new_ghost_pos,
         "topology": topology_score,
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
   best_rank = None

   remaining_ms = max(
      0,
      __TIMEOUT__ - get_run_time(),
   )

   if remaining_ms < 30:
      safe_area_depth = 1
   elif remaining_ms < 80:
      safe_area_depth = min(
         2,
         SAFE_AREA_DEPTH,
      )
   else:
      safe_area_depth = SAFE_AREA_DEPTH

   log(
      f"   Shared safe-area depth="
      f"{safe_area_depth}"
   )

   # Calculate each candidate's raw safe area once.
   for candidate in candidates:
      blocked_pos = None

      if candidate["move"] != Move.STAY:
         blocked_pos = ghost_pos

      candidate["safe_area"] = predicted_safe_area(
         candidate["position"],
         candidate["pacman_position"],
         map_state,
         pacman_speed,
         max_depth=safe_area_depth,
         capture_cache=capture_cache,
         blocked_pos=blocked_pos,
      )

   stay_candidate = next(
      candidate
      for candidate in candidates
      if candidate["move"] == Move.STAY
   )

   stay_safe_area = stay_candidate["safe_area"]

   moving_safe_areas = [
      candidate["safe_area"]
      for candidate in candidates
      if candidate["move"] != Move.STAY
   ]

   best_moving_safe_area = max(
      moving_safe_areas,
      default=stay_safe_area,
   )

   stay_scored_safe_area = best_moving_safe_area

   # Detect whether the Ghost is currently inside a real dead end.
   dead_end_mouth = find_dead_end_mouth(
      ghost_pos,
      map_state,
   )

   dead_end_distances = {}
   current_dead_end_depth = None
   current_dead_end_slack = None
   must_exit_dead_end = False

   if dead_end_mouth is not None:
      dead_end_distances = cached_bfs_distances(
         dead_end_mouth,
         map_state,
         distance_cache,
      )

      current_dead_end_depth = (
         dead_end_distances.get(
            ghost_pos,
            core.INF,
         )
      )

      pacman_turns_to_mouth = (
         cached_capture_turn_distance(
            stay_candidate["pacman_position"],
            dead_end_mouth,
            map_state,
            pacman_speed,
            capture_cache,
         )
      )

      current_dead_end_slack = (
         pacman_turns_to_mouth
         - current_dead_end_depth
      )

      must_exit_dead_end = (
         current_dead_end_slack <= 1
      )

   # Precalculate movement and dead-end data.
   for candidate in candidates:
      pacman_distances = cached_bfs_distances(
         candidate["pacman_position"],
         map_state,
         distance_cache,
      )

      current_maze_distance = pacman_distances.get(
         ghost_pos,
         core.INF,
      )

      candidate_maze_distance = pacman_distances.get(
         candidate["position"],
         core.INF,
      )

      current_manhattan = core.manhattan(
         candidate["pacman_position"],
         ghost_pos,
      )

      candidate_manhattan = core.manhattan(
         candidate["pacman_position"],
         candidate["position"],
      )

      candidate["maze_distance"] = (
         candidate_maze_distance
      )

      candidate["manhattan"] = (
         candidate_manhattan
      )

      candidate["direction_delta"] = (
         candidate_maze_distance
         - current_maze_distance
      )

      candidate["manhattan_delta"] = (
         candidate_manhattan
         - current_manhattan
      )

      candidate["is_approach"] = (
         candidate["direction_delta"] < 0
         or candidate["manhattan_delta"] < 0
      )

      candidate["is_away"] = (
         candidate["direction_delta"] > 0
         and candidate["manhattan_delta"] > 0
      )

      if dead_end_mouth is not None:
         candidate_dead_end_depth = (
            dead_end_distances.get(
               candidate["position"],
               core.INF,
            )
         )

         candidate["dead_end_depth"] = (
            candidate_dead_end_depth
         )

         candidate["toward_dead_end_exit"] = (
            candidate["move"] != Move.STAY
            and candidate_dead_end_depth
            < current_dead_end_depth
         )

         candidate["deeper_into_dead_end"] = (
            candidate["move"] != Move.STAY
            and candidate_dead_end_depth
            > current_dead_end_depth
         )

         candidate_pacman_exit_turns = (
            cached_capture_turn_distance(
               candidate["pacman_position"],
               dead_end_mouth,
               map_state,
               pacman_speed,
               capture_cache,
            )
         )

         candidate["dead_end_slack"] = (
            candidate_pacman_exit_turns
            - candidate_dead_end_depth
         )

      else:
         candidate["dead_end_depth"] = None
         candidate["toward_dead_end_exit"] = False
         candidate["deeper_into_dead_end"] = False
         candidate["dead_end_slack"] = None

   # A move deeper into the current dead end does not count
   # as a useful non-approaching alternative.
   has_non_approaching_move = any(
      candidate["move"] != Move.STAY
      and not candidate["is_approach"]
      and not candidate["deeper_into_dead_end"]
      for candidate in candidates
   )

   viable_exit_move_exists = any(
      candidate["toward_dead_end_exit"]
      and candidate["worst_utility"] < CAPTURE_SCORE
      for candidate in candidates
   )

   valid_non_stay_capture_turns = [
      candidate["capture_turns"]
      for candidate in candidates
      if (
         candidate["move"] != Move.STAY
         and not candidate["deeper_into_dead_end"]
      )
   ]

   best_valid_non_stay_capture_turns = max(
      valid_non_stay_capture_turns,
      default=-1,
   )

   # While the exit is not urgent, staying is a legitimate
   # alternative to walking toward Pacman.
   stay_is_valid_alternative = (
      dead_end_mouth is not None
      and not must_exit_dead_end
      and stay_candidate["capture_turns"] > 1
   )

   for candidate in candidates:
      final_score = candidate["worst_utility"]

      if candidate["move"] == Move.STAY:
         final_score += STAY_PENALTY

         # Equal capture turns do not justify the large STAY penalties.
         stay_not_safer = (
            candidate["capture_turns"]
            < best_valid_non_stay_capture_turns
         )

         if (
            candidate["capture_turns"] <= 5
            and stay_not_safer
         ):
            final_score += DANGER_STAY_PENALTY

         if stay_not_safer:
            final_score += LOST_TEMPO_PENALTY

      direction_delta = (
         candidate["direction_delta"]
      )

      manhattan_delta = (
         candidate["manhattan_delta"]
      )

      is_approach = candidate["is_approach"]
      is_away = candidate["is_away"]

      urgent_exit_move = (
         must_exit_dead_end
         and candidate["toward_dead_end_exit"]
      )

      forced_approach = (
         candidate["move"] != Move.STAY
         and is_approach
         and (
            urgent_exit_move
            or (
               dead_end_mouth is None
               and not has_non_approaching_move
            )
         )
      )

      # A move toward the mouth is exempt only when
      # leaving the dead end has become urgent.
      if is_approach and not forced_approach:
         final_score += APPROACH_PENALTY

      elif is_away:
         final_score -= AWAY_BONUS * min(
            direction_delta,
            manhattan_delta,
            2,
         )

      safe_area = candidate["safe_area"]

      if candidate["move"] == Move.STAY:
         scored_safe_area = best_moving_safe_area
      else:
         scored_safe_area = safe_area

      final_score -= (
         SAFE_AREA_BONUS
         * scored_safe_area
      )

      if scored_safe_area <= MIN_SAFE_AREA:
         final_score += LOW_SAFE_AREA_PENALTY

      capture_improved = (
         candidate["capture_turns"]
         > stay_candidate["capture_turns"]
      )

      safe_area_improved = (
         scored_safe_area
         > stay_scored_safe_area
      )

      utility_improved = (
         candidate["worst_utility"]
         < stay_candidate["worst_utility"]
      )

      capture_not_worse = (
         candidate["capture_turns"]
         >= stay_candidate["capture_turns"]
      )

      safe_area_not_worse = (
         scored_safe_area
         >= stay_scored_safe_area
      )

      utility_has_real_support = (
         utility_improved
         and capture_not_worse
         and safe_area_not_worse
      )

      has_future_merit = (
         capture_improved
         or safe_area_improved
         or utility_has_real_support
      )

      meritless_approach = (
         candidate["move"] != Move.STAY
         and is_approach
         and not has_future_merit
      )

      blocked_approach = (
         meritless_approach
         and not forced_approach
         and (
            has_non_approaching_move
            or stay_is_valid_alternative
         )
      )

      # When exiting is urgent, STAY and deeper moves lose
      # to a viable move toward the mouth.
      dead_end_delay = (
         must_exit_dead_end
         and viable_exit_move_exists
         and not candidate["toward_dead_end_exit"]
      )

      # Do not walk back deeper into the dead end when STAY
      # preserves more escape slack.
      dead_end_retreat = (
         dead_end_mouth is not None
         and candidate["deeper_into_dead_end"]
         and candidate["dead_end_slack"] is not None
         and current_dead_end_slack is not None
         and candidate["dead_end_slack"]
            < current_dead_end_slack
         and stay_candidate["capture_turns"] > 1
      )

      candidate_maze_distance = (
         candidate["maze_distance"]
      )

      candidate_manhattan = (
         candidate["manhattan"]
      )

      candidate_rank = (
         1 if dead_end_delay else 0,
         1 if dead_end_retreat else 0,
         1 if blocked_approach else 0,
         final_score,
         -candidate["capture_turns"],
         -scored_safe_area,
         -candidate_maze_distance,
         -candidate_manhattan,
         -candidate["topology"],
      )

      log(
         f"   Candidate {candidate['move']}, "
         f"to={candidate['position']}, "
         f"topology={candidate['topology']}, "
         f"pacman_best={candidate['pacman_action']}"
         f"->{candidate['pacman_position']}, "
         f"capture_turns={candidate['capture_turns']}, "
         f"maze_distance={candidate_maze_distance}, "
         f"manhattan={candidate_manhattan}, "
         f"direction_delta={direction_delta}, "
         f"manhattan_delta={manhattan_delta}, "
         f"is_approach={is_approach}, "
         f"is_away={is_away}, "
         f"urgent_exit_move={urgent_exit_move}, "
         f"forced_approach={forced_approach}, "
         f"has_non_approaching_move="
         f"{has_non_approaching_move}, "
         f"stay_is_valid_alternative="
         f"{stay_is_valid_alternative}, "
         f"safe_area={safe_area}, "
         f"scored_safe_area={scored_safe_area}, "
         f"capture_improved={capture_improved}, "
         f"safe_area_improved={safe_area_improved}, "
         f"utility_improved={utility_improved}, "
         f"utility_supported="
         f"{utility_has_real_support}, "
         f"future_merit={has_future_merit}, "
         f"meritless_approach={meritless_approach}, "
         f"blocked_approach={blocked_approach}, "
         f"dead_end_mouth={dead_end_mouth}, "
         f"dead_end_depth="
         f"{candidate['dead_end_depth']}, "
         f"dead_end_slack="
         f"{candidate['dead_end_slack']}, "
         f"current_dead_end_slack="
         f"{current_dead_end_slack}, "
         f"must_exit_dead_end="
         f"{must_exit_dead_end}, "
         f"toward_dead_end_exit="
         f"{candidate['toward_dead_end_exit']}, "
         f"deeper_into_dead_end="
         f"{candidate['deeper_into_dead_end']}, "
         f"dead_end_delay={dead_end_delay}, "
         f"dead_end_retreat={dead_end_retreat}, "
         f"worst_utility="
         f"{candidate['worst_utility']}, "
         f"final={final_score}"
      )

      if (
         best_rank is None
         or candidate_rank < best_rank
      ):
         best_rank = candidate_rank
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