"""Structural campsite discovery and tactical confidence ranking."""

from collections import deque
from dataclasses import asdict, dataclass

from .geometry import (
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
