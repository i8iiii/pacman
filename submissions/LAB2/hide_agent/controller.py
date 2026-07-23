"""Hide controller with strategic concealment and P06-P09 escape behavior."""

from pathlib import Path
from time import perf_counter

from environment import Move

from .belief import PacmanBeliefTracker, choose_belief_hot_move
from .diagnostics import DIAGNOSTICS_ENABLED, JsonlDiagnostics, MapDiagnostics
from .escape import choose_visible_junction_escape
from .geometry import geometry_summary, is_capture
from .hideout import scan_hideouts, select_hideout, visibility_footprints
from .mobile_escape import choose_visible_mobile_escape
from .navigation import (
    RouteTarget,
    route_is_structural,
    route_moves,
)
from .pursuit import PursuitTracker
from .roads import (
    build_road_cycle,
    build_road_visibility,
    detect_major_roads,
    filter_hideout_candidates,
    road_thresholds,
    select_closest_safe_hideout,
)
from .topology import scan_campsites


class HideController:
    """Hide deeply before detection, then preserve the established escape logic."""

    SCOUT = "SCOUT"
    HIDE = "HIDE"
    HOT_UNSEEN = "HOT_UNSEEN"

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
        self._pacman_speed = max(1, int(pacman_speed))
        self._capture_distance = max(1, int(capture_distance))
        self._observation_radius = max(0, int(observation_radius))
        self._tactical_scan = None
        self._tactical_campsites = []
        self._hideout_candidates = ()
        self._selected_hideout = None
        self._compromised_hideouts = set()
        self._ghost_spawn = None
        self._arrival_logged_for = None
        self._major_roads = None
        self._road_schedule = None
        self._road_visibility = ()
        self._eligible_hideouts = ()
        self._road_excluded_hideouts = ()
        self._road_cycle = None
        self._active_road_stage = None
        self._active_road_ids = ()
        self._match_start_step = None
        self._state = None
        self._active_target_kind = None
        self._active_target = None
        self._active_path = []
        self._pursuit = PursuitTracker()
        self._belief = PacmanBeliefTracker()

    def step(self, map_state, my_position, enemy_position, step_number):
        step_started = perf_counter()
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

        if new_match or self._tactical_scan is None:
            self._major_roads = detect_major_roads(current_map)
            footprints = visibility_footprints(
                current_map,
                self._observation_radius,
            )
            self._road_visibility = build_road_visibility(
                self._major_roads,
                footprints,
            )
            self._road_cycle = build_road_cycle(
                self._road_visibility,
                self._ghost_spawn,
                current_map.shape,
                stage_turns=5,
            )
            self._active_road_stage = self._road_cycle.stage(0)
            self._active_road_ids = (
                self._active_road_stage.road_ids
            )
            self._tactical_scan, self._tactical_campsites = scan_campsites(
                current_map,
                pacman_speed=self._pacman_speed,
                capture_distance=self._capture_distance,
                observation_radius=self._observation_radius,
            )
            self._hideout_candidates = scan_hideouts(
                current_map,
                observation_radius=self._observation_radius,
                ghost_spawn=self._ghost_spawn,
                pacman_speed=self._pacman_speed,
                footprints=footprints,
            )
            (
                self._eligible_hideouts,
                self._road_excluded_hideouts,
            ) = filter_hideout_candidates(
                self._hideout_candidates,
                self._road_visibility,
                active_road_ids=self._active_road_ids,
            )
            self._diagnostics.write(
                "main_road_scan",
                step_number=step_number,
                shape=list(current_map.shape),
                thresholds=road_thresholds(current_map),
                road_count=len(self._major_roads),
                roads=[
                    road.to_log_record()
                    for road in self._major_roads
                ],
                road_visibility=[
                    record.to_log_record()
                    for record in self._road_visibility
                ],
                approach_road_ids=[
                    record.road.road_id
                    for record in self._road_visibility
                    if record.is_approach
                ],
                excluded_cell_count=len(
                    {
                        position
                        for record in self._road_visibility
                        if record.is_approach
                        for position in record.visible_cells
                    }
                ),
                excluded_hideout_count=len(
                    self._road_excluded_hideouts
                ),
            )
            class_counts = {}
            for candidate in self._hideout_candidates:
                class_counts[candidate.kind] = (
                    class_counts.get(candidate.kind, 0) + 1
                )
            footprints = [
                candidate.visibility_footprint
                for candidate in self._hideout_candidates
            ]
            self._diagnostics.write(
                "hideout_scan",
                step_number=step_number,
                candidates=len(self._hideout_candidates),
                classes=class_counts,
                entrance_hidden=sum(
                    int(candidate.entrance_hidden)
                    for candidate in self._hideout_candidates
                ),
                footprint_min=min(footprints) if footprints else None,
                footprint_max=max(footprints) if footprints else None,
            )

        if enemy_position is not None:
            self._compromise_selected_hideout(
                my_position,
                step_number,
            )

        target_plan = None
        selection = None
        previous_selected = (
            None
            if self._selected_hideout is None
            else self._selected_hideout.position
        )
        normal_unseen = enemy_position is None and not self._belief.active
        road_stage_changed = False
        if normal_unseen and self._road_cycle is not None:
            elapsed_turns = max(
                0,
                int(step_number) - int(self._match_start_step),
            )
            requested_index = self._road_cycle.requested_index(
                elapsed_turns
            )
            if requested_index != self._active_road_stage.index:
                requested_stage = self._road_cycle.stage(
                    requested_index
                )
                if requested_stage.road_ids:
                    self._active_road_stage = requested_stage
                    self._active_road_ids = requested_stage.road_ids
                    (
                        self._eligible_hideouts,
                        self._road_excluded_hideouts,
                    ) = filter_hideout_candidates(
                        self._hideout_candidates,
                        self._road_visibility,
                        active_road_ids=self._active_road_ids,
                    )
                    road_stage_changed = True

        if normal_unseen:
            eligible_positions = {
                candidate.position
                for candidate in self._eligible_hideouts
            }
            target_became_exposed = (
                road_stage_changed
                and previous_selected is not None
                and previous_selected not in eligible_positions
            )
            if target_became_exposed:
                selection = select_closest_safe_hideout(
                    current_map,
                    my_position,
                    self._eligible_hideouts,
                    self._compromised_hideouts,
                )
            else:
                selection = select_hideout(
                    current_map,
                    my_position,
                    self._eligible_hideouts,
                    self._compromised_hideouts,
                    preferred_position=previous_selected,
                )
            self._selected_hideout = selection.candidate

        if normal_unseen:
            if self._selected_hideout is not None:
                target_plan = RouteTarget(
                    kind="strategic_hideout",
                    position=self._selected_hideout.position,
                    path=selection.path,
                )
        selected_position = (
            None
            if self._selected_hideout is None
            else self._selected_hideout.position
        )
        current_hideout = self._at_selected_hideout(my_position)
        if current_hideout:
            target_plan = None

        if selected_position != previous_selected and selection is not None:
            self._arrival_logged_for = None
            self._diagnostics.write(
                "hideout_selected",
                step_number=step_number,
                selected=(
                    None
                    if selection.candidate is None
                    else selection.candidate.to_log_record()
                ),
                route_distance=selection.route_distance,
                rank=list(selection.rank),
                admitted_count=getattr(
                    selection,
                    "admitted_count",
                    None,
                ),
                rejections=dict(selection.rejections),
            )

        belief_update = None
        if enemy_position is None:
            belief_update = self._belief.observe_unseen(
                current_map,
                my_position,
                observation_radius=self._observation_radius,
                capture_distance=self._capture_distance,
                pacman_speed=self._pacman_speed,
            )
            if belief_update.rebuilt:
                self._write_belief_rebuilt(
                    belief_update,
                    step_number,
                )
        else:
            was_hot = (
                self._belief.active
                and self._belief.elapsed_unseen > 0
            )
            if was_hot:
                self._write_hot_unseen_exited(
                    step_number,
                    reason="new_sighting",
                )
                self._change_state(
                    (
                        self.HIDE
                        if current_hideout
                        else self.SCOUT
                    ),
                    step_number,
                    reason="new_sighting",
                )
            self._belief.record_visible(
                enemy_position,
                step_number,
            )

        self._map_diagnostics.write_snapshot(
            step_number,
            current_map,
            hideout_candidates=self._hideout_candidates,
            selected_hideout=selected_position,
            compromised_hideouts=self._compromised_hideouts,
            pacman_belief=self._belief.positions,
            road_visibility=self._road_visibility,
            road_excluded_hideouts=self._road_excluded_hideouts,
        )

        hot_result = None
        if enemy_position is None:
            hot_result = self._hot_unseen_move(
                current_map,
                my_position,
                step_number,
                belief_update,
                current_hideout,
            )

        if enemy_position is not None:
            if (
                self._pursuit.active is not None
                or self._pursuit.pending is not None
            ):
                invalidated = self._pursuit.invalidate("new_sighting")
                self._write_follower_invalidated(
                    invalidated,
                    step_number,
                )
                self._change_state(
                    (
                        self.HIDE if current_hideout else self.SCOUT
                    ),
                    step_number,
                    reason="new_sighting",
                )
            escape_decision = choose_visible_junction_escape(
                current_map,
                my_position,
                enemy_position,
                pacman_speed=self._pacman_speed,
                capture_distance=self._capture_distance,
            )
            if escape_decision is None:
                mobile_decision = choose_visible_mobile_escape(
                    current_map,
                    my_position,
                    enemy_position,
                    self._tactical_campsites,
                    pacman_speed=self._pacman_speed,
                    capture_distance=self._capture_distance,
                    observation_radius=self._observation_radius,
                )
                self._clear_route()
                self._write_visible_mobile_escape(
                    mobile_decision,
                    my_position,
                    enemy_position,
                    step_number,
                )
                move = mobile_decision.selected.move
                decision_reason = (
                    f"visible_mobile_escape_{mobile_decision.mode}"
                )
            else:
                self._clear_route()
                self._write_visible_escape(
                    escape_decision,
                    my_position,
                    enemy_position,
                    current_hideout,
                    step_number,
                )
                move = escape_decision.selected.move
                decision_reason = (
                    f"visible_escape_{escape_decision.mode}"
                )
            self._pursuit.record_visible_escape(
                current_map,
                my_position,
                enemy_position,
                move,
                step_number,
            )
        elif hot_result is not None:
            move, decision_reason = hot_result
        elif current_hideout:
            self._clear_route()
            self._change_state(
                self.HIDE,
                step_number,
                reason="hideout_reached",
            )
            if self._arrival_logged_for != selected_position:
                self._diagnostics.write(
                    "hideout_arrived",
                    step_number=step_number,
                    hideout=list(selected_position),
                )
                self._arrival_logged_for = selected_position
            move = Move.STAY
            decision_reason = "hideout_hold"
        else:
            self._change_state(
                self.SCOUT,
                step_number,
                reason="routing_to_hideout",
            )
            move, decision_reason = self._scout_move(
                current_map,
                my_position,
                target_plan,
                step_number,
            )

        runtime_ms = (perf_counter() - step_started) * 1000.0
        self._diagnostics.write(
            "decision",
            move=move.name,
            reason=decision_reason,
            runtime_ms=round(runtime_ms, 3),
            state=self._state,
            step_number=step_number,
        )
        self._last_step_number = step_number
        return move

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
        self._diagnostics.reset()
        self._map_diagnostics.reset()
        self._tactical_scan = None
        self._tactical_campsites = []
        self._hideout_candidates = ()
        self._selected_hideout = None
        self._compromised_hideouts = set()
        self._ghost_spawn = tuple(ghost_spawn)
        self._arrival_logged_for = None
        self._major_roads = None
        self._road_schedule = None
        self._road_visibility = ()
        self._eligible_hideouts = ()
        self._road_excluded_hideouts = ()
        self._road_cycle = None
        self._active_road_stage = None
        self._active_road_ids = ()
        self._match_start_step = int(step_number)
        self._state = self.SCOUT
        self._pursuit.reset()
        self._belief.reset()
        self._clear_route()
        self._diagnostics.write("match_start")
        self._diagnostics.write("match_state_reset")
        self._diagnostics.write(
            "state_changed",
            step_number=step_number,
            previous_state=None,
            state=self.SCOUT,
            reason="match_start",
        )

    def _at_selected_hideout(self, position):
        return (
            self._selected_hideout is not None
            and self._selected_hideout.position == tuple(position)
        )

    def _compromise_selected_hideout(
        self,
        my_position,
        step_number,
    ):
        if self._selected_hideout is None:
            return
        position = self._selected_hideout.position
        if position in self._compromised_hideouts:
            self._selected_hideout = None
            return

        reason = (
            "visible_at_hideout"
            if tuple(my_position) == position
            else "visible_en_route"
        )
        self._compromised_hideouts.add(position)
        self._diagnostics.write(
            "hideout_compromised",
            step_number=step_number,
            hideout=list(position),
            reason=reason,
            ghost_position=list(my_position),
        )
        self._selected_hideout = None
        self._arrival_logged_for = None
        self._clear_route()

    def _scout_move(
        self,
        current_map,
        my_position,
        target_plan,
        step_number,
    ):
        self._synchronize_route(my_position)

        if target_plan is None:
            if self._active_target is not None:
                self._diagnostics.write(
                    "route_replanned",
                    step_number=step_number,
                    previous_target=list(self._active_target),
                    previous_target_kind=self._active_target_kind,
                    target=None,
                    target_kind=None,
                    reason="no_safe_reachable_target",
                )
            self._clear_route()
            self._diagnostics.write(
                "scout_target",
                step_number=step_number,
                target=None,
                target_kind=None,
                reason="no_safe_reachable_target",
            )
            self._diagnostics.write(
                "scout_move",
                step_number=step_number,
                move=Move.STAY.name,
                target=None,
                target_kind=None,
                remaining_steps=0,
            )
            return Move.STAY, "no_safe_reachable_target"

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

        self._diagnostics.write(
            "scout_target",
            step_number=step_number,
            target=list(target_plan.position),
            target_kind=target_plan.kind,
            route_length=len(desired_path),
        )

        if self._active_target is None:
            self._write_route_event(
                "route_planned",
                step_number,
                my_position,
                target_plan,
                desired_path,
                desired_moves,
                reason="target_selected",
            )
        elif target_changed or route_invalid or route_changed:
            if target_changed:
                reason = "target_changed"
            elif route_invalid:
                reason = "route_invalid"
            else:
                reason = "route_changed"
            self._write_route_event(
                "route_replanned",
                step_number,
                my_position,
                target_plan,
                desired_path,
                desired_moves,
                reason=reason,
                previous_target=self._active_target,
                previous_target_kind=self._active_target_kind,
            )

        self._active_target_kind = target_plan.kind
        self._active_target = target_plan.position
        self._active_path = desired_path

        if not desired_moves:
            move = Move.STAY
            reason = "target_has_no_legal_route"
        else:
            move = desired_moves[0]
            reason = "hideout_route"

        self._diagnostics.write(
            "scout_move",
            step_number=step_number,
            move=move.name,
            target=list(target_plan.position),
            target_kind=target_plan.kind,
            remaining_steps=len(desired_path),
        )
        return move, reason

    def _write_route_event(
        self,
        event,
        step_number,
        my_position,
        target_plan,
        path,
        moves,
        reason,
        previous_target=None,
        previous_target_kind=None,
    ):
        fields = {
            "step_number": step_number,
            "target": list(target_plan.position),
            "target_kind": target_plan.kind,
            "path": [list(my_position)] + [
                list(position) for position in path
            ],
            "moves": [move.name for move in moves],
            "reason": reason,
        }
        if previous_target is not None:
            fields["previous_target"] = list(previous_target)
            fields["previous_target_kind"] = previous_target_kind
        self._diagnostics.write(event, **fields)

    def _write_visible_escape(
        self,
        decision,
        my_position,
        enemy_position,
        at_tactical_junction,
        step_number,
    ):
        self._diagnostics.write(
            "visible_at_junction",
            step_number=step_number,
            position=list(my_position),
            pacman_position=list(enemy_position),
            at_tactical_junction=at_tactical_junction,
            junction_type=decision.junction_type,
            approach_direction=decision.approach_direction.name,
            missing_direction=(
                None
                if decision.missing_direction is None
                else decision.missing_direction.name
            ),
            pacman_endpoints=[
                list(endpoint) for endpoint in decision.pacman_endpoints
            ],
            branches=[
                branch.to_log_record() for branch in decision.branches
            ],
        )
        self._diagnostics.write(
            "escape_branch_chosen",
            step_number=step_number,
            mode=decision.mode,
            move=decision.selected.move.name,
            endpoint=list(decision.selected.endpoint),
            equivalent_moves=[
                move.name for move in decision.equivalent_moves
            ],
            branch=decision.selected.to_log_record(),
        )

    def _write_visible_mobile_escape(
        self,
        decision,
        my_position,
        enemy_position,
        step_number,
    ):
        self._diagnostics.write(
            "visible_while_mobile",
            step_number=step_number,
            position=list(my_position),
            pacman_position=list(enemy_position),
            approach_direction=(
                None
                if decision.approach_direction is None
                else decision.approach_direction.name
            ),
        )
        selected = decision.selected
        self._diagnostics.write(
            "escape_target_chosen",
            step_number=step_number,
            mode=decision.mode,
            move=selected.move.name,
            endpoint=list(selected.endpoint),
            guaranteed_safe=selected.guaranteed_safe,
            immediate_rank=list(selected.immediate_rank),
            rank=list(selected.rank),
            equivalent_moves=[
                move.name for move in decision.equivalent_moves
            ],
            target=(
                None
                if selected.target is None
                else selected.target.to_log_record()
            ),
            target_reason=(
                "no_target_on_selected_continuation"
                if selected.target is None
                else selected.target.kind
            ),
        )

    def _hot_unseen_move(
        self,
        current_map,
        my_position,
        step_number,
        belief_update,
        current_hideout,
    ):
        if belief_update is None or belief_update.status == "inactive":
            return None

        update = self._pursuit.observe_unseen(
            current_map,
            my_position,
            observation_radius=self._observation_radius,
            capture_distance=self._capture_distance,
        )
        if update.entered:
            self._write_hot_unseen_entered(update, step_number)
        if update.status != "inactive":
            self._write_follower_updated(update, step_number)
        if update.status == "invalidated":
            self._write_follower_invalidated(update, step_number)

        next_belief = self._belief.predict_next(
            current_map,
            pacman_speed=self._pacman_speed,
        )
        if (
            current_hideout
            and not any(
                is_capture(
                    position,
                    my_position,
                    self._capture_distance,
                )
                for position in next_belief
            )
        ):
            if (
                self._pursuit.seed is not None
                or self._pursuit.active is not None
                or self._pursuit.pending is not None
            ):
                invalidated = self._pursuit.invalidate(
                    "belief_safe_hideout"
                )
                self._write_follower_invalidated(
                    invalidated,
                    step_number,
                )
            self._write_hot_unseen_exited(
                step_number,
                reason="belief_safe_hideout",
            )
            self._belief.reset()
            self._clear_route()
            self._change_state(
                self.HIDE,
                step_number,
                reason="belief_safe_hideout",
            )
            if self._arrival_logged_for != self._selected_hideout.position:
                self._diagnostics.write(
                    "hideout_arrived",
                    step_number=step_number,
                    hideout=list(self._selected_hideout.position),
                )
                self._arrival_logged_for = self._selected_hideout.position
            return Move.STAY, "hot_unseen_safe_hideout"

        strategic_hideouts = (
            ()
            if self._selected_hideout is None
            else (self._selected_hideout,)
        )

        likely_candidates = self._pursuit.project_hot_moves(
            current_map,
            my_position,
            strategic_hideouts,
            observation_radius=self._observation_radius,
            capture_distance=self._capture_distance,
        )
        previous_ghost_position = (
            self._belief.absence_history[-2]
            if len(self._belief.absence_history) >= 2
            else None
        )
        decision = choose_belief_hot_move(
            current_map,
            my_position,
            self._belief.positions,
            strategic_hideouts,
            observation_radius=self._observation_radius,
            capture_distance=self._capture_distance,
            pacman_speed=self._pacman_speed,
            likely_candidates=likely_candidates,
            previous_ghost_position=previous_ghost_position,
        )
        if decision.selected.likely_projection is not None:
            self._pursuit.stage_hot_candidate(
                decision.selected.likely_projection
            )

        self._clear_route()
        self._change_state(
            self.HOT_UNSEEN,
            step_number,
            reason=(
                "pursuit_entered"
                if update.entered
                else "pursuit_continued"
            ),
        )
        self._write_interceptor_updated(decision, step_number)
        self._write_hot_move(decision, step_number)
        return (
            decision.selected.move,
            f"hot_unseen_{decision.mode}",
        )

    def _write_hot_unseen_entered(self, update, step_number):
        seed = update.seed
        self._diagnostics.write(
            "hot_unseen_entered",
            step_number=step_number,
            seed_step_number=(
                None if seed is None else seed.step_number
            ),
            pacman_position=(
                None
                if seed is None
                else list(seed.pacman_position)
            ),
            ghost_departure_position=(
                None
                if seed is None
                else list(seed.ghost_position)
            ),
            escape_move=(
                None if seed is None else seed.escape_move.name
            ),
            expected_ghost_position=(
                None
                if seed is None
                else list(seed.expected_ghost_position)
            ),
            trail=[list(position) for position in update.trail],
            followers=[
                follower.to_log_record()
                for follower in update.followers
            ],
        )

    def _write_follower_updated(self, update, step_number):
        self._diagnostics.write(
            "follower_updated",
            step_number=step_number,
            status=update.status,
            previous_followers=[
                follower.to_log_record()
                for follower in update.previous_followers
            ],
            followers=[
                follower.to_log_record()
                for follower in update.followers
            ],
            removals=update.removals_log_record(),
        )

    def _write_hot_move(self, decision, step_number):
        selected = decision.selected
        likely = selected.likely_projection
        self._diagnostics.write(
            "hot_move",
            step_number=step_number,
            mode=decision.mode,
            move=selected.move.name,
            endpoint=list(selected.endpoint),
            guaranteed_safe=selected.guaranteed_safe,
            rank=list(selected.rank),
            equivalent_moves=[
                move.name for move in decision.equivalent_moves
            ],
            next_followers=[
                follower.to_log_record()
                for follower in (
                    () if likely is None else likely.next_followers
                )
            ],
            target=(
                None
                if selected.target is None
                else {
                    "position": list(selected.target.target),
                    "ghost_arrival": selected.target.ghost_arrival,
                    "pacman_threat_arrival": (
                        selected.target.pacman_threat_arrival
                    ),
                }
            ),
            capturing_endpoint_count=(
                selected.capturing_endpoint_count
            ),
            broad_belief_size=len(selected.next_belief),
        )

    def _write_belief_rebuilt(self, update, step_number):
        self._diagnostics.write(
            "belief_rebuilt",
            step_number=step_number,
            reason=update.rebuild_reason,
            elapsed_unseen=update.elapsed_unseen,
            belief_size=len(update.positions),
        )

    def _write_interceptor_updated(self, decision, step_number):
        selected = decision.selected
        assessment = selected.target
        if assessment is None and selected.interception.assessments:
            assessment = selected.interception.assessments[0]

        if assessment is None:
            fields = {
                "target": None,
                "route_junction": None,
                "ghost_arrival": None,
                "pacman_threat_arrival": None,
                "contested": None,
                "reason": "no_reachable_hideout",
            }
        elif assessment.first_contested_junction is not None:
            fields = {
                "target": list(assessment.target),
                "route_junction": list(
                    assessment.first_contested_junction
                ),
                "ghost_arrival": assessment.junction_ghost_arrival,
                "pacman_threat_arrival": (
                    assessment.junction_pacman_arrival
                ),
                "contested": True,
                "reason": assessment.reason,
            }
        else:
            fields = {
                "target": list(assessment.target),
                "route_junction": None,
                "ghost_arrival": assessment.ghost_arrival,
                "pacman_threat_arrival": (
                    assessment.pacman_threat_arrival
                ),
                "contested": assessment.contested,
                "reason": assessment.reason,
            }
        self._diagnostics.write(
            "interceptor_updated",
            step_number=step_number,
            **fields,
        )

    def _write_hot_unseen_exited(self, step_number, reason):
        self._diagnostics.write(
            "hot_unseen_exited",
            step_number=step_number,
            reason=reason,
        )

    def _write_follower_invalidated(self, update, step_number):
        self._diagnostics.write(
            "follower_invalidated",
            step_number=step_number,
            reason=update.reason,
            previous_followers=[
                follower.to_log_record()
                for follower in update.previous_followers
            ],
            removals=update.removals_log_record(),
        )

    def _synchronize_route(self, my_position):
        if self._active_target == my_position:
            self._clear_route()
            return
        if self._active_path and self._active_path[0] == my_position:
            self._active_path.pop(0)

    def _clear_route(self):
        self._active_target_kind = None
        self._active_target = None
        self._active_path = []

    def _change_state(self, new_state, step_number, reason):
        if self._state == new_state:
            return
        previous_state = self._state
        self._state = new_state
        self._diagnostics.write(
            "state_changed",
            step_number=step_number,
            previous_state=previous_state,
            state=new_state,
            reason=reason,
        )
