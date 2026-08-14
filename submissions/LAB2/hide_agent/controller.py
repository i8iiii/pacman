"""Simplified Hide controller: rotate closest hideouts with concealed routing."""

from pathlib import Path
from time import perf_counter

from environment import Move

from .concealment import rank_hideouts, scan_hideouts, visibility_footprints
from .diagnostics import DIAGNOSTICS_ENABLED, JsonlDiagnostics, MapDiagnostics
from .spatial import (
    RouteTarget,
    concealment_route,
    geometry_summary,
    route_is_structural,
    route_moves,
    structural_shortest_paths,
)


class HideController:
    """Rotate across nearby hideouts with simple deterministic behavior.

    Selection strategy: always prefer hideouts near the map border that is
    closest to the ghost's current position.  No detour stops are made en
    route — the ghost travels directly to each anchor hideout.
    """

    SCOUT = "SCOUT"
    HIDE = "HIDE"

    HOLD_TURNS = 10

    def __init__(
        self,
        log_path=None,
        map_text_path=None,
        map_jsonl_path=None,
        diagnostics_enabled=None,
        pacman_speed=2,
        capture_distance=2,
        observation_radius=5,
    ):
        debug_dir = Path(__file__).resolve().parent.parent / "debug"
        enabled = (
            DIAGNOSTICS_ENABLED
            if diagnostics_enabled is None
            else bool(diagnostics_enabled)
        )

        self._diagnostics = JsonlDiagnostics(
            log_path or debug_dir / "hide-agent.log",
            enabled=enabled,
        )
        self._map_diagnostics = MapDiagnostics(
            map_text_path or debug_dir / "hide-agent-map.txt",
            map_jsonl_path or debug_dir / "hide-agent-map.jsonl",
            enabled=enabled,
        )

        self._map_shape = None
        self._last_step_number = None
        self._ghost_spawn = None
        self._state = self.SCOUT

        self._pacman_speed = max(1, int(pacman_speed))
        self._capture_distance = max(1, int(capture_distance))
        self._observation_radius = max(0, int(observation_radius))

        self._visibility_footprints = {}
        self._hideout_candidates = ()

        self._anchor_hideout = None
        self._previous_anchor_hideout = None

        self._holding_at = None
        self._hold_remaining = 0

        self._active_target_kind = None
        self._active_target = None
        self._active_path = []

    def step(self, map_state, my_position, enemy_position, step_number):
        try:
            return self._step_internal(map_state, my_position, enemy_position, step_number)
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print("GHOST AGENT ERROR:", trace)
            self._diagnostics.write(
                "error",
                step_number=step_number,
                message=str(e),
                traceback=trace,
            )
            return Move.STAY

    def _step_internal(self, map_state, my_position, enemy_position, step_number):
        started = perf_counter()
        my_position = tuple(int(value) for value in my_position)
        enemy_position = (
            None
            if enemy_position is None
            else tuple(int(value) for value in enemy_position)
        )

        new_match = self._is_new_match(map_state, step_number)
        if new_match:
            self._start_match(map_state.shape, my_position, step_number)

        current_map = map_state.copy()
        self._diagnostics.write(
            "geometry_summary",
            step_number=step_number,
            **geometry_summary(
                current_map,
                my_position,
                enemy_position,
                pacman_speed=self._pacman_speed,
                capture_distance=self._capture_distance,
                observation_radius=self._observation_radius,
            ),
        )

        if new_match or not self._hideout_candidates:
            self._visibility_footprints = visibility_footprints(
                current_map,
                self._observation_radius,
            )
            self._hideout_candidates = scan_hideouts(
                current_map,
                observation_radius=self._observation_radius,
                ghost_spawn=self._ghost_spawn,
                pacman_speed=self._pacman_speed,
                footprints=self._visibility_footprints,
            )
            self._diagnostics.write(
                "hideout_scan",
                step_number=step_number,
                candidates=len(self._hideout_candidates),
            )

        move, reason = self._decide_move(
            current_map,
            my_position,
            enemy_position,
            step_number,
        )

        self._map_diagnostics.write_snapshot(
            step_number,
            current_map,
            hideout_candidates=self._hideout_candidates,
            selected_hideout=self._anchor_hideout,
            compromised_hideouts=(),
            pacman_belief=(),
        )

        runtime_ms = round((perf_counter() - started) * 1000.0, 3)
        self._diagnostics.write(
            "decision",
            step_number=step_number,
            move=move.name,
            reason=reason,
            state=self._state,
            runtime_ms=runtime_ms,
            anchor_hideout=(
                None
                if self._anchor_hideout is None
                else list(self._anchor_hideout)
            ),
            hold_remaining=self._hold_remaining,
        )

        self._last_step_number = step_number
        return move

    def _decide_move(self, current_map, my_position, enemy_position, step_number):
        self._ensure_anchor(
            current_map,
            my_position,
            enemy_position,
            step_number,
        )

        objective = self._anchor_hideout
        if objective is None:
            self._clear_route()
            self._state = self.SCOUT
            return Move.STAY, "no_hideout_available"

        self._synchronize_route(my_position)

        # If Pacman is visible, OR we are already holding here due to seeing Pacman,
        # override the objective to our current position so we activate HIDE mode immediately!
        if enemy_position is not None or self._holding_at == my_position:
            objective = my_position

        if my_position == objective:
            return self._handle_objective_arrival(
                current_map,
                my_position,
                enemy_position,
                step_number,
                objective,
            )

        self._holding_at = None
        self._hold_remaining = 0

        target_plan = self._plan_route(current_map, my_position, objective)
        move, reason = self._scout_move(
            current_map,
            my_position,
            target_plan,
            step_number,
        )
        self._state = self.SCOUT
        return move, reason

    def _handle_objective_arrival(
        self,
        current_map,
        my_position,
        enemy_position,
        step_number,
        objective,
    ):
        if self._holding_at != objective:
            self._holding_at = objective
            self._hold_remaining = self.HOLD_TURNS
            self._diagnostics.write(
                "hold_started",
                step_number=step_number,
                hideout=list(objective),
                hold_turns=self.HOLD_TURNS,
            )

        if self._hold_remaining > 0:
            self._hold_remaining -= 1
            self._clear_route()
            self._state = self.HIDE
            return Move.STAY, "hideout_hold"

        # Hold complete — rotate to a new anchor, excluding the current one.
        self._previous_anchor_hideout = self._anchor_hideout
        banned = (
            {self._previous_anchor_hideout}
            if self._previous_anchor_hideout is not None
            else set()
        )
        self._anchor_hideout = self._select_ranked_hideout(
            current_map,
            my_position,
            enemy_position,
            banned=banned,
        )
        self._holding_at = None
        self._hold_remaining = 0
        self._clear_route()

        self._diagnostics.write(
            "anchor_rotated",
            step_number=step_number,
            previous=(
                None
                if self._previous_anchor_hideout is None
                else list(self._previous_anchor_hideout)
            ),
            selected=(
                None
                if self._anchor_hideout is None
                else list(self._anchor_hideout)
            ),
        )

        self._state = self.SCOUT
        return Move.STAY, "rotate_anchor_after_hold"

    def _ensure_anchor(
        self,
        current_map,
        my_position,
        enemy_position,
        step_number,
    ):
        if self._anchor_hideout is not None:
            return

        banned = (
            {self._previous_anchor_hideout}
            if self._previous_anchor_hideout is not None
            else set()
        )
        self._anchor_hideout = self._select_ranked_hideout(
            current_map,
            my_position,
            enemy_position,
            banned=banned,
        )
        # If the ban excluded every option, retry without restrictions.
        if self._anchor_hideout is None and self._previous_anchor_hideout is not None:
            self._anchor_hideout = self._select_ranked_hideout(
                current_map,
                my_position,
                enemy_position,
                banned=set(),
            )

        self._diagnostics.write(
            "anchor_selected",
            step_number=step_number,
            selected=(
                None
                if self._anchor_hideout is None
                else list(self._anchor_hideout)
            ),
        )

    def _select_ranked_hideout(
        self,
        current_map,
        my_position,
        enemy_position,
        banned,
    ):
        effective_ban = [p for p in banned if p is not None]
        ranked = rank_hideouts(
            current_map,
            my_position,
            self._hideout_candidates,
            enemy_position=enemy_position,
            excluded_positions=effective_ban,
            observation_radius=self._observation_radius,
        )
        if not ranked and effective_ban:
            ranked = rank_hideouts(
                current_map,
                my_position,
                self._hideout_candidates,
                enemy_position=enemy_position,
                observation_radius=self._observation_radius,
            )
        return None if not ranked else tuple(ranked[0].position)

    def _plan_route(self, current_map, my_position, target):
        decision = concealment_route(
            current_map,
            my_position,
            target,
            self._visibility_footprints,
            active_road_visible=(),
        )
        path = tuple(decision.path)
        if path or tuple(my_position) == tuple(target):
            return RouteTarget(
                kind="anchor_hideout",
                position=tuple(target),
                path=path,
            )
        return None

    def _is_new_match(self, map_state, step_number):
        return (
            self._map_shape is None
            or self._map_shape != tuple(map_state.shape)
            or (
                self._last_step_number is not None
                and step_number <= self._last_step_number
            )
        )

    def _start_match(self, map_shape, ghost_spawn, step_number):
        self._map_shape = tuple(map_shape)
        self._ghost_spawn = tuple(ghost_spawn)
        self._state = self.SCOUT

        self._visibility_footprints = {}
        self._hideout_candidates = ()

        self._anchor_hideout = None
        self._previous_anchor_hideout = None

        self._holding_at = None
        self._hold_remaining = 0

        self._clear_route()

        self._diagnostics.reset()
        self._map_diagnostics.reset()
        self._diagnostics.write("match_start")
        self._diagnostics.write(
            "state_changed",
            step_number=step_number,
            previous_state=None,
            state=self.SCOUT,
            reason="match_start",
        )

    def _synchronize_route(self, my_position):
        if not self._active_path:
            return
        if tuple(my_position) == tuple(self._active_path[0]):
            self._active_path = self._active_path[1:]
            return
        if tuple(my_position) in self._active_path:
            index = self._active_path.index(tuple(my_position))
            self._active_path = self._active_path[index + 1 :]

    def _clear_route(self):
        self._active_target_kind = None
        self._active_target = None
        self._active_path = []

    def _scout_move(self, current_map, my_position, target_plan, step_number):
        if target_plan is None:
            self._clear_route()
            self._diagnostics.write(
                "scout_move",
                step_number=step_number,
                move=Move.STAY.name,
                target=None,
                reason="no_safe_route",
            )
            return Move.STAY, "no_safe_route"

        desired_path = list(target_plan.path)
        desired_moves = route_moves(my_position, desired_path)
        target_changed = (
            self._active_target != target_plan.position
            or self._active_target_kind != target_plan.kind
        )
        route_invalid = not route_is_structural(
            current_map,
            my_position,
            self._active_path,
        )
        route_changed = self._active_path != desired_path

        if self._active_target is None:
            event = "route_planned"
            reason = "target_selected"
        elif target_changed or route_invalid or route_changed:
            event = "route_replanned"
            if target_changed:
                reason = "target_changed"
            elif route_invalid:
                reason = "route_invalid"
            else:
                reason = "route_changed"
        else:
            event = None
            reason = None

        if event is not None:
            self._diagnostics.write(
                event,
                step_number=step_number,
                target=list(target_plan.position),
                target_kind=target_plan.kind,
                path=[list(my_position)] + [list(pos) for pos in desired_path],
                moves=[move.name for move in desired_moves],
                reason=reason,
            )

        self._active_target_kind = target_plan.kind
        self._active_target = target_plan.position
        self._active_path = desired_path

        if not desired_moves:
            move = Move.STAY
            reason = "target_has_no_legal_route"
        else:
            move = desired_moves[0]
            reason = "least_exposed_route"

        self._diagnostics.write(
            "scout_move",
            step_number=step_number,
            move=move.name,
            target=list(target_plan.position),
            target_kind=target_plan.kind,
            remaining_steps=len(desired_path),
        )
        return move, reason
