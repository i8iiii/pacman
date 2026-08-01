"""Strategic hideout analysis.

This module treats every non-wall value as structurally traversable.  In
particular, ``-1`` is usable for structural analysis without being converted
into remembered observation data.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from rl.hide_agent.spatial import (
    CARDINAL_MOVES,
    apply_move,
    has_line_of_sight,
    in_bounds,
    is_structurally_traversable,
)
from rl.hide_agent.belief import pacman_turn_distances
from rl.hide_agent.spatial import reconstruct_path, structural_shortest_paths


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
        candidate = HideoutCandidate(
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
        if is_strategic_hideout(candidate):
            candidates.append(candidate)

    return tuple(candidates)


def is_strategic_hideout(candidate) -> bool:
    """Return whether a candidate is deep enough for strategic hiding."""

    return (
        getattr(candidate, "kind", "fallback") != "fallback"
        and int(getattr(candidate, "inspection_depth", 0))
        >= MIN_HIDEOUT_INSPECTION_DEPTH
    )


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
        "unsafe": 0,
        "compromised": 0,
        "unreachable": 0,
        "lower_class": 0,
        "outside_route_window": 0,
    }
    reachable = []
    for candidate in candidates:
        if not is_strategic_hideout(candidate):
            rejections["unsafe"] += 1
        elif candidate.position in compromised_positions:
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


"""Structural campsite discovery and tactical confidence ranking."""

from collections import deque
from dataclasses import asdict, dataclass

from rl.hide_agent.spatial import (
    CARDINAL_MOVES,
    apply_move,
    has_line_of_sight,
    is_capture,
    is_observed_traversable,
    is_structurally_traversable,
    pacman_endpoints,
)


@dataclass(frozen=True)
class CampsiteCandidate:
    position: tuple
    observed: bool
    safe: bool
    confirmed_exits: int
    possible_exits: int
    structural_exits: int
    blind_capture_approaches: int
    unverified_capture_approaches: int
    occluding_exits: int
    independent_regions: int
    loop_exit_count: int
    warning_distance: int
    score: tuple

    def to_log_record(self):
        record = asdict(self)
        record["position"] = list(self.position)
        record["score"] = list(self.score)
        return record


def structural_neighbors(map_state, position, excluded=None):
    neighbors = []
    for move in CARDINAL_MOVES:
        neighbor = apply_move(position, move)
        if neighbor == excluded:
            continue
        if is_structurally_traversable(map_state, neighbor):
            neighbors.append(neighbor)
    return neighbors


def scan_campsites(
    map_state,
    pacman_speed=2,
    capture_distance=2,
    observation_radius=5,
):
    """Return scan statistics and every universal four-way safe campsite."""
    structural_cells = [
        (row, column)
        for row in range(map_state.shape[0])
        for column in range(map_state.shape[1])
        if is_structurally_traversable(map_state, (row, column))
    ]
    junction_count = 0
    t_junction_count = 0
    safe_campsites = []
    removal_regions = _removal_region_counts(map_state, structural_cells)

    for position in structural_cells:
        neighbors = structural_neighbors(map_state, position)
        if len(neighbors) < 3:
            continue

        junction_count += 1
        candidate = _evaluate_candidate(
            map_state,
            position,
            neighbors,
            pacman_speed,
            capture_distance,
            observation_radius,
            removal_regions[position],
        )
        if candidate.safe:
            safe_campsites.append(candidate)
        else:
            t_junction_count += 1

    scan_summary = {
        "junctions": junction_count,
        "safe_campsites": len(safe_campsites),
        "t_junctions": t_junction_count,
    }
    return scan_summary, safe_campsites


def _evaluate_candidate(
    map_state,
    position,
    neighbors,
    pacman_speed,
    capture_distance,
    observation_radius,
    independent_regions,
):
    confirmed_exits = sum(
        is_observed_traversable(map_state, neighbor) for neighbor in neighbors
    )
    possible_exits = len(neighbors) - confirmed_exits
    blind_approaches, unverified_approaches = _capture_approach_counts(
        map_state,
        position,
        pacman_speed,
        capture_distance,
        observation_radius,
    )
    occluding_exits = sum(
        _branch_reaches_occlusion(
            map_state,
            position,
            neighbor,
            observation_radius,
        )
        for neighbor in neighbors
    )
    loop_exit_count = (
        len(neighbors)
        if independent_regions == 1
        else max(0, len(neighbors) - independent_regions)
    )
    warning_distance = _warning_distance(map_state, position)
    observed = is_observed_traversable(map_state, position)
    safe = len(neighbors) == len(CARDINAL_MOVES)

    score = (
        -blind_approaches,
        occluding_exits,
        independent_regions,
        loop_exit_count,
        warning_distance,
        confirmed_exits,
        int(observed),
    )
    return CampsiteCandidate(
        position=position,
        observed=observed,
        safe=safe,
        confirmed_exits=confirmed_exits,
        possible_exits=possible_exits,
        structural_exits=len(neighbors),
        blind_capture_approaches=blind_approaches,
        unverified_capture_approaches=unverified_approaches,
        occluding_exits=occluding_exits,
        independent_regions=independent_regions,
        loop_exit_count=loop_exit_count,
        warning_distance=warning_distance,
        score=score,
    )


def _capture_approach_counts(
    map_state,
    campsite,
    pacman_speed,
    capture_distance,
    observation_radius,
):
    reach = max(1, int(pacman_speed)) + max(1, int(capture_distance)) - 1
    blind = 0
    unverified = 0

    row_start = max(0, campsite[0] - reach)
    row_end = min(map_state.shape[0], campsite[0] + reach + 1)
    column_start = max(0, campsite[1] - reach)
    column_end = min(map_state.shape[1], campsite[1] + reach + 1)

    for row in range(row_start, row_end):
        for column in range(column_start, column_end):
            approach = (row, column)
            if not is_structurally_traversable(map_state, approach):
                continue
            endpoints = pacman_endpoints(map_state, approach, pacman_speed)
            if not any(
                is_capture(endpoint, campsite, capture_distance)
                for endpoint in endpoints
            ):
                continue
            if not has_line_of_sight(
                map_state,
                campsite,
                approach,
                observation_radius,
            ):
                blind += 1
                if not is_observed_traversable(map_state, approach):
                    unverified += 1

    return blind, unverified


def _branch_reaches_occlusion(
    map_state,
    campsite,
    first_cell,
    observation_radius,
):
    max_depth = max(1, int(observation_radius)) + 1
    queue = deque([(first_cell, 1)])
    visited = {campsite, first_cell}

    while queue:
        current, depth = queue.popleft()
        if not has_line_of_sight(
            map_state,
            campsite,
            current,
            observation_radius,
        ):
            return True
        if depth >= max_depth:
            continue
        for neighbor in structural_neighbors(map_state, current):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, depth + 1))

    return False


def _removal_region_counts(map_state, structural_cells):
    """Count escape regions formed around every cell after removing that cell."""
    discovery = {}
    low_link = {}
    parent = {}
    region_counts = {}
    next_index = 0

    def visit(position):
        nonlocal next_index
        discovery[position] = next_index
        low_link[position] = next_index
        next_index += 1
        child_count = 0
        separating_children = 0

        for neighbor in structural_neighbors(map_state, position):
            if neighbor not in discovery:
                parent[neighbor] = position
                child_count += 1
                visit(neighbor)
                low_link[position] = min(low_link[position], low_link[neighbor])
                if (
                    parent[position] is not None
                    and low_link[neighbor] >= discovery[position]
                ):
                    separating_children += 1
            elif neighbor != parent[position]:
                low_link[position] = min(
                    low_link[position],
                    discovery[neighbor],
                )

        if parent[position] is None:
            region_counts[position] = child_count
        else:
            region_counts[position] = separating_children + 1

    for position in structural_cells:
        if position in discovery:
            continue
        parent[position] = None
        visit(position)

    return region_counts


def _warning_distance(map_state, campsite):
    ray_lengths = []

    for move in CARDINAL_MOVES:
        distance = 0
        while True:
            cell = apply_move(campsite, move, distance + 1)
            if not is_structurally_traversable(map_state, cell):
                break
            distance += 1
        if distance > 0:
            ray_lengths.append(distance)

    return min(ray_lengths) if ray_lengths else 0
