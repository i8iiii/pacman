"""Strategic hideout analysis.

This module treats every non-wall value as structurally traversable.  In
particular, ``-1`` is usable for structural analysis without being converted
into remembered observation data.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .geometry import (
    CARDINAL_MOVES,
    apply_move,
    has_line_of_sight,
    in_bounds,
    is_structurally_traversable,
)
from .belief import pacman_turn_distances
from .navigation import reconstruct_path, structural_shortest_paths


Position = Tuple[int, int]


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
            "spawn_discovery_distance": self.spawn_discovery_distance,
            "opposite_vertical_band": self.opposite_vertical_band,
        }


@dataclass(frozen=True)
class HideoutSelection:
    """A chosen strategic target plus auditable selector details."""

    candidate: Optional[HideoutCandidate]
    path: Tuple[Position, ...]
    route_distance: Optional[int]
    rank: Tuple[int, ...]
    admitted_count: int
    rejections: Mapping[str, int]


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
        candidates.append(
            HideoutCandidate(
                position=position,
                kind=kind,
                entrance=entrance,
                gate_depth=len(set(terminal)),
                must_backtrack=bool(terminal),
                entrance_hidden=entrance_hidden,
                inspection_depth=inspection_depth,
                visibility_footprint=len(footprints[position]),
                spawn_discovery_distance=spawn_discovery_distance,
                opposite_vertical_band=opposite_vertical_band,
            )
        )

    return tuple(candidates)


def select_hideout(
    map_state,
    ghost_position: Position,
    candidates: Sequence[HideoutCandidate],
    compromised: Sequence[Position],
    preferred_position: Optional[Position] = None,
    route_slack: int = 4,
) -> HideoutSelection:
    """Choose one reachable hideout deterministically.

    The active preferred target stays locked while reachable and uncompromised.
    Otherwise only the strongest available structural class competes, and its
    candidates must be no more than ``route_slack`` Ghost steps farther than
    the nearest candidate in that class.
    """

    distances, parents = structural_shortest_paths(map_state, ghost_position)
    compromised_positions = {tuple(position) for position in compromised}
    rejections = {
        "compromised": 0,
        "unreachable": 0,
        "lower_class": 0,
        "outside_route_window": 0,
    }
    reachable = []
    for candidate in candidates:
        if candidate.position in compromised_positions:
            rejections["compromised"] += 1
        elif candidate.position not in distances:
            rejections["unreachable"] += 1
        else:
            reachable.append(candidate)

    preferred_position = (
        None if preferred_position is None else tuple(preferred_position)
    )
    preferred = next(
        (
            candidate
            for candidate in reachable
            if candidate.position == preferred_position
        ),
        None,
    )
    if preferred is not None:
        return _selection_result(
            preferred,
            distances,
            parents,
            admitted_count=1,
            rejections=rejections,
        )

    if not reachable:
        return HideoutSelection(
            candidate=None,
            path=(),
            route_distance=None,
            rank=(),
            admitted_count=0,
            rejections=rejections,
        )

    best_class = max(_class_tier(candidate) for candidate in reachable)
    same_class = [
        candidate
        for candidate in reachable
        if _class_tier(candidate) == best_class
    ]
    rejections["lower_class"] = len(reachable) - len(same_class)
    nearest_distance = min(distances[candidate.position] for candidate in same_class)
    limit = nearest_distance + max(0, int(route_slack))
    admitted = [
        candidate
        for candidate in same_class
        if distances[candidate.position] <= limit
    ]
    rejections["outside_route_window"] = len(same_class) - len(admitted)
    selected = max(
        admitted,
        key=lambda candidate: _candidate_rank(
            candidate,
            distances[candidate.position],
        ),
    )
    return _selection_result(
        selected,
        distances,
        parents,
        admitted_count=len(admitted),
        rejections=rejections,
    )


def _structural_cells(map_state: Sequence[Sequence[int]]) -> Set[Position]:
    return {
        (row, col)
        for row, values in enumerate(map_state)
        for col, _ in enumerate(values)
        if is_structurally_traversable(map_state, (row, col))
    }


def _class_tier(candidate: HideoutCandidate) -> int:
    if candidate.kind == "terminal" and candidate.entrance_hidden:
        return 3
    if candidate.kind == "reconnecting" and candidate.entrance_hidden:
        return 2
    return 1


def _candidate_rank(
    candidate: HideoutCandidate,
    route_distance: int,
) -> Tuple[int, ...]:
    return hideout_quality_rank(candidate) + (
        -route_distance,
        -candidate.position[0],
        -candidate.position[1],
    )


def hideout_quality_rank(candidate: HideoutCandidate) -> Tuple[int, ...]:
    """Return static hideout quality without route or coordinate tie-breaks."""

    return (
        candidate.gate_depth,
        candidate.inspection_depth,
        -candidate.visibility_footprint,
        int(candidate.must_backtrack),
        candidate.spawn_discovery_distance,
        int(candidate.opposite_vertical_band),
    )


def _selection_result(
    candidate: HideoutCandidate,
    distances: Mapping[Position, int],
    parents: Mapping[Position, Optional[Position]],
    admitted_count: int,
    rejections: Mapping[str, int],
) -> HideoutSelection:
    route_distance = distances[candidate.position]
    return HideoutSelection(
        candidate=candidate,
        path=tuple(reconstruct_path(parents, candidate.position)),
        route_distance=route_distance,
        rank=_candidate_rank(candidate, route_distance),
        admitted_count=admitted_count,
        rejections=dict(rejections),
    )


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
