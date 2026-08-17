"""Strategic hideout analysis.

This module treats every non-wall value as structurally traversable.  In
particular, ``-1`` is usable for structural analysis without being converted
into remembered observation data.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .belief import pacman_turn_distances
from .spatial import (
    CARDINAL_MOVES,
    apply_move,
    has_line_of_sight,
    in_bounds,
    is_structurally_traversable,
    structural_shortest_paths,
)


Position = Tuple[int, int]
MIN_HIDEOUT_INSPECTION_DEPTH = 3


@dataclass(frozen=True)
class HideoutCandidate:
    """One structurally possible strategic waiting cell."""

    position: Position
    kind: str
    entrance: Optional[Position]
    gate_depth: int
    must_backtrack: bool
    entrance_hidden: bool = False
    inspection_depth: int = 0
    visibility_footprint: int = 0
    exposed_axes: int = 0
    spawn_discovery_distance: int = 0
    opposite_vertical_band: bool = False

    def to_log_record(self) -> dict:
        return {
            "position": list(self.position),
            "kind": self.kind,
            "entrance": None if self.entrance is None else list(self.entrance),
            "gate_depth": self.gate_depth,
            "must_backtrack": self.must_backtrack,
            "entrance_hidden": self.entrance_hidden,
            "inspection_depth": self.inspection_depth,
            "visibility_footprint": self.visibility_footprint,
            "exposed_axes": self.exposed_axes,
            "spawn_discovery_distance": self.spawn_discovery_distance,
            "opposite_vertical_band": self.opposite_vertical_band,
        }


def visibility_footprints(
    map_state: Sequence[Sequence[int]],
    observation_radius: int,
) -> Dict[Position, Tuple[Position, ...]]:
    """Return every structural cell's possible cardinal-line observers.

    The returned mapping and every observer tuple are coordinate-sorted so the
    result is stable across runs.
    """

    footprints: Dict[Position, Tuple[Position, ...]] = {}
    cells = sorted(_structural_cells(map_state))
    radius = max(0, int(observation_radius))

    for target in cells:
        observers = {target}
        for move in CARDINAL_MOVES:
            row, col = target
            for _ in range(radius):
                row += move.value[0]
                col += move.value[1]
                observer = (row, col)
                if not in_bounds(map_state, observer):
                    break
                if not is_structurally_traversable(map_state, observer):
                    break
                observers.add(observer)
        footprints[target] = tuple(sorted(observers))

    return footprints


def scan_hideouts(
    map_state: Sequence[Sequence[int]],
    observation_radius: int,
    pacman_spawn: Optional[Position] = None,
    ghost_spawn: Optional[Position] = None,
    pacman_speed: int = 2,
    footprints=None,
) -> Tuple[HideoutCandidate, ...]:
    """Classify every non-wall cell using junction branch structure.

    A branch gate is a degree-three-or-greater junction.  Removing that gate
    reveals which neighboring branches can reconnect without crossing it.
    Single-neighbor components are terminal and therefore require backtracking.
    """

    adjacency = _adjacency(map_state)
    if footprints is None:
        footprints = visibility_footprints(map_state, observation_radius)
    inferred_pacman_starts = _pacman_start_positions(
        map_state,
        adjacency,
        pacman_spawn,
        ghost_spawn,
    )
    spawn_distances = (
        pacman_turn_distances(
            map_state,
            inferred_pacman_starts,
            pacman_speed=pacman_speed,
        )
        if inferred_pacman_starts
        else {}
    )
    entrance_distance_cache = {}
    junctions = tuple(
        sorted(position for position, neighbors in adjacency.items() if len(neighbors) >= 3)
    )
    terminal_gates: Dict[Position, List[Position]] = {
        position: [] for position in adjacency
    }
    reconnecting_gates: Dict[Position, List[Position]] = {
        position: [] for position in adjacency
    }

    for gate in junctions:
        for component in _components_without(adjacency, gate):
            gate_neighbors = adjacency[gate].intersection(component)
            target = (
                terminal_gates
                if len(gate_neighbors) == 1
                else reconnecting_gates
            )
            for position in component:
                target[position].append(gate)

    candidates = []
    all_candidates = []
    for position in sorted(adjacency):
        terminal = terminal_gates[position]
        reconnecting = reconnecting_gates[position]
        if terminal:
            kind = "terminal"
            gates = terminal
        elif reconnecting:
            kind = "reconnecting"
            gates = reconnecting
        else:
            kind = "fallback"
            gates = []

        entrance = _nearest_gate(adjacency, position, gates)
        entrance_hidden = bool(
            entrance is not None
            and not has_line_of_sight(
                map_state,
                entrance,
                position,
                observation_radius,
            )
        )
        if entrance is not None and entrance not in entrance_distance_cache:
            entrance_distance_cache[entrance] = pacman_turn_distances(
                map_state,
                (entrance,),
                pacman_speed=pacman_speed,
            )
        inspection_depth = _observer_distance(
            entrance_distance_cache.get(entrance, {}),
            footprints[position],
        )
        spawn_discovery_distance = _observer_distance(
            spawn_distances,
            footprints[position],
        )
        opposite_vertical_band = _is_opposite_vertical_band(
            map_state,
            position,
            pacman_spawn,
            ghost_spawn,
        )
        
        has_vertical = any(neighbor[0] != position[0] for neighbor in adjacency[position])
        has_horizontal = any(neighbor[1] != position[1] for neighbor in adjacency[position])
        exposed_axes = int(has_vertical) + int(has_horizontal)

        candidate = HideoutCandidate(
            position=position,
            kind=kind,
            entrance=entrance,
            gate_depth=len(set(terminal)),
            must_backtrack=bool(terminal),
            entrance_hidden=entrance_hidden,
            inspection_depth=inspection_depth,
            visibility_footprint=len(footprints[position]),
            exposed_axes=exposed_axes,
            spawn_discovery_distance=spawn_discovery_distance,
            opposite_vertical_band=opposite_vertical_band,
        )
        all_candidates.append(candidate)
        if is_strategic_hideout(candidate):
            candidates.append(candidate)

    if not candidates:
        # Fallback to any non-fallback kind if no strict strategic hideouts exist
        candidates = [c for c in all_candidates if c.kind != "fallback"]
    if not candidates:
        # Ultimate fallback: all valid cells
        candidates = all_candidates

    return tuple(candidates)


def is_strategic_hideout(candidate) -> bool:
    """Return whether a candidate is deep enough for strategic hiding."""

    return (
        getattr(candidate, "kind", "fallback") != "fallback"
        and int(getattr(candidate, "inspection_depth", 0))
        >= MIN_HIDEOUT_INSPECTION_DEPTH
    )


def rank_hideouts(
    map_state,
    ghost_position: Position,
    candidates: Sequence[HideoutCandidate],
    *,
    enemy_position: Optional[Position] = None,
    possible_enemy_positions: Sequence[Position] = (),
    excluded_positions: Sequence[Position] = (),
    observation_radius: int = 5,
) -> Tuple[HideoutCandidate, ...]:
    """Return reachable strategic hideouts ordered by border proximity then safety.

    Only hideouts in the top or bottom edge zone (whichever is closest to the
    ghost vertically) are considered.  Middle-of-map candidates are excluded.
    Within the chosen zone candidates are sorted by distance to that edge
    ascending, then by exposed_axes ascending (fewer is better), then
    by fog-aware safety score descending.

    If no candidates survive the zone filter the full candidate list is used as
    a fallback so the agent always has somewhere to go.
    """

    ghost_distances, _ = structural_shortest_paths(map_state, ghost_position)
    excluded = {tuple(position) for position in excluded_positions}
    enemy_sources = _enemy_sources(
        map_state,
        enemy_position,
        possible_enemy_positions,
    )
    enemy_distances = _multi_source_distances(map_state, enemy_sources)
    rows, columns = map_state.shape
    max_corner = max(
        min(row, rows - 1 - row, column, columns - 1 - column)
        for row in range(rows)
        for column in range(columns)
    )
    disconnected_enemy_distance = rows * columns

    # Determine whether the ghost is closer to the top or bottom row border.
    gr, gc = ghost_position
    use_top = gr <= rows - 1 - gr  # True → prefer top edge; False → bottom edge

    # Edge zone = the outer third of the map on the preferred side.
    edge_zone = max(1, rows // 3)

    def _score_candidate(candidate):
        """Return (border_dist, exposed_dirs, safety_score) or None if invalid."""
        position = tuple(candidate.position)
        if (
            position in excluded
            or position not in ghost_distances
        ):
            return None

        corner_distance = min(
            position[0],
            rows - 1 - position[0],
            position[1],
            columns - 1 - position[1],
        )
        safety_score = ghost_distances[position] + max_corner - corner_distance
        if enemy_sources:
            enemy_distance = enemy_distances.get(
                position,
                disconnected_enemy_distance,
            )
            visibility_count = sum(
                1
                for source in enemy_sources
                if source != position
                and has_line_of_sight(
                    map_state,
                    source,
                    position,
                    observation_radius,
                )
            )
            safety_score += enemy_distance - 2 * visibility_count

        pr = position[0]
        border_dist = pr if use_top else (rows - 1 - pr)
        axes = candidate.exposed_axes
        return border_dist, axes, safety_score

    # First pass: only candidates inside the preferred edge zone.
    zone_scored = []
    all_scored = []
    for candidate in candidates:
        result = _score_candidate(candidate)
        if result is None:
            continue
        border_dist, axes, safety_score = result
        position = tuple(candidate.position)
        entry = (border_dist, axes, safety_score, position, candidate)
        all_scored.append(entry)
        pr = position[0]
        in_zone = pr < edge_zone if use_top else pr >= rows - edge_zone
        if in_zone:
            zone_scored.append(entry)

    # Use the zone-filtered list; fall back to all candidates if zone is empty.
    scored = zone_scored if zone_scored else all_scored

    # Primary sort: border_dist ascending (closer to preferred edge wins).
    # Secondary sort: exposed_axes ascending (fewer exposed axes wins).
    # Tertiary sort: safety_score descending (safer wins among equally-close).
    # Quaternary: position for determinism.
    scored.sort(key=lambda item: (item[0], item[1], -item[2], item[3]))
    return tuple(item[4] for item in scored)


def _enemy_sources(map_state, enemy_position, possible_enemy_positions):
    if enemy_position is not None:
        positions = (tuple(enemy_position),)
    else:
        positions = tuple(sorted({tuple(position) for position in possible_enemy_positions}))
    return tuple(
        position
        for position in positions
        if is_structurally_traversable(map_state, position)
    )


def _multi_source_distances(map_state, sources):
    distances = {position: 0 for position in sources}
    queue = deque(distances)

    while queue:
        position = queue.popleft()
        for move in CARDINAL_MOVES:
            neighbor = apply_move(position, move)
            if neighbor in distances or not is_structurally_traversable(map_state, neighbor):
                continue
            distances[neighbor] = distances[position] + 1
            queue.append(neighbor)

    return distances


def _structural_cells(map_state: Sequence[Sequence[int]]) -> Set[Position]:
    return {
        (row, col)
        for row, values in enumerate(map_state)
        for col, _ in enumerate(values)
        if is_structurally_traversable(map_state, (row, col))
    }


def _adjacency(map_state) -> Dict[Position, Set[Position]]:
    cells = _structural_cells(map_state)
    return {
        position: {
            apply_move(position, move)
            for move in CARDINAL_MOVES
            if apply_move(position, move) in cells
        }
        for position in cells
    }


def _components_without(
    adjacency: Dict[Position, Set[Position]],
    blocked: Position,
) -> Tuple[Set[Position], ...]:
    remaining = set(adjacency)
    remaining.discard(blocked)
    components = []

    while remaining:
        start = min(remaining)
        component = {start}
        queue = deque([start])
        remaining.remove(start)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor == blocked or neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        components.append(component)

    return tuple(components)


def _nearest_gate(
    adjacency: Dict[Position, Set[Position]],
    start: Position,
    gates: Sequence[Position],
) -> Optional[Position]:
    if not gates:
        return None
    distances = _graph_distances(adjacency, start)
    return min(gates, key=lambda gate: (distances.get(gate, 10**9), gate))


def _graph_distances(
    adjacency: Dict[Position, Set[Position]],
    start: Position,
) -> Dict[Position, int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency[current]):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


def _observer_distance(
    distances: Mapping[Position, int],
    observer_positions: Sequence[Position],
) -> int:
    if not distances:
        return 0
    reachable = [
        distances[position]
        for position in observer_positions
        if position in distances
    ]
    return min(reachable) if reachable else 10**9


def _is_opposite_vertical_band(
    map_state,
    position: Position,
    pacman_spawn: Optional[Position],
    ghost_spawn: Optional[Position],
) -> bool:
    middle = map_state.shape[0] / 2.0
    candidate_top = position[0] < middle
    if pacman_spawn is None:
        if ghost_spawn is None:
            return False
        pacman_top = not (ghost_spawn[0] < middle)
    else:
        pacman_top = pacman_spawn[0] < middle
    if candidate_top == pacman_top:
        return False
    if ghost_spawn is None:
        return True
    return candidate_top == (ghost_spawn[0] < middle)


def _pacman_start_positions(
    map_state,
    adjacency: Mapping[Position, Set[Position]],
    pacman_spawn: Optional[Position],
    ghost_spawn: Optional[Position],
) -> Tuple[Position, ...]:
    if pacman_spawn is not None:
        return (tuple(pacman_spawn),)
    if ghost_spawn is None:
        return ()

    rows = map_state.shape[0]
    ghost_starts_top = ghost_spawn[0] < rows / 2.0
    if ghost_starts_top:
        lower_bound = rows * 0.6
        inferred = [
            position for position in adjacency if position[0] >= lower_bound
        ]
    else:
        upper_bound = rows * 0.4
        inferred = [
            position for position in adjacency if position[0] < upper_bound
        ]
    return tuple(sorted(inferred))
