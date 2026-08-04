"""Topology-and-visibility area analysis for the LAB2 maze."""

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .spatial import traversable_neighbors, visibility_footprint


MAX_VIEWPOINTS_PER_AREA = 5
MIN_AREA_CELLS = 8
VISIBILITY_RADIUS = 5

_ANALYSIS_CACHE = {}


@dataclass(frozen=True)
class Gateway:
    area_a: int
    area_b: int
    connections: tuple


@dataclass(frozen=True)
class Area:
    area_id: int
    cells: frozenset
    centroid: tuple
    position_label: str
    viewpoints: tuple
    neighbors: tuple


@dataclass(frozen=True)
class AreaAnalysis:
    fingerprint: str
    shape: tuple
    areas: tuple
    cell_to_area: dict
    gateways: tuple
    analysis_seconds: float
    error: str = None

    def area_for(self, position):
        return self.cell_to_area.get(position)


class AreaAnalyzer:
    """Partition a maze into connected districts with bounded sight workload."""

    def analyze(self, map_state):
        topology = np.asarray(map_state) != 1
        fingerprint = _map_fingerprint(topology)
        cached = _ANALYSIS_CACHE.get(fingerprint)
        if cached is not None:
            return cached, True

        try:
            analysis = _analyze_topology(topology, fingerprint)
        except Exception as exc:
            analysis = _fallback_analysis(
                topology,
                fingerprint,
                f"{type(exc).__name__}: {exc}",
            )
        _ANALYSIS_CACHE[fingerprint] = analysis
        return analysis, False


def _analyze_topology(topology, fingerprint):
    started_at = perf_counter()
    nodes = tuple(
        tuple(int(value) for value in position)
        for position in np.argwhere(topology)
    )
    if not nodes:
        return AreaAnalysis(
            fingerprint=fingerprint,
            shape=tuple(int(value) for value in topology.shape),
            areas=(),
            cell_to_area={},
            gateways=(),
            analysis_seconds=perf_counter() - started_at,
            error=None,
        )

    original_adjacency = {
        node: set(traversable_neighbors(topology, node))
        for node in nodes
    }
    working_adjacency = {
        node: set(neighbors)
        for node, neighbors in original_adjacency.items()
    }
    footprints = {
        node: visibility_footprint(topology, node, VISIBILITY_RADIUS)
        for node in nodes
    }
    coverage_cache = {}

    while True:
        components = _connected_components(working_adjacency)
        coverage = {
            component: _bounded_coverage(
                component,
                footprints,
                MAX_VIEWPOINTS_PER_AREA,
                coverage_cache,
            )
            for component in components
        }
        oversized = [
            component
            for component in components
            if coverage[component] is None
        ]
        if not oversized:
            break

        centrality = {}
        for component in oversized:
            centrality.update(_edge_betweenness(working_adjacency, component))
        if not centrality:
            break

        boundary = min(
            centrality,
            key=lambda edge: (-centrality[edge], edge),
        )
        working_adjacency[boundary[0]].remove(boundary[1])
        working_adjacency[boundary[1]].remove(boundary[0])

    components = _merge_tiny_components(
        _connected_components(working_adjacency),
        original_adjacency,
        footprints,
        coverage_cache,
    )
    ordered_components = sorted(
        components,
        key=lambda component: (_centroid(component), min(component)),
    )

    cell_to_area = {}
    area_drafts = []
    for area_id, component in enumerate(ordered_components):
        viewpoints = _bounded_coverage(
            component,
            footprints,
            MAX_VIEWPOINTS_PER_AREA,
            coverage_cache,
        )
        if viewpoints is None:
            viewpoints = _greedy_coverage(component, footprints)
        centroid = _centroid(component)
        for cell in component:
            cell_to_area[cell] = area_id
        area_drafts.append(
            {
                "area_id": area_id,
                "cells": component,
                "centroid": centroid,
                "position_label": _position_label(
                    centroid,
                    topology.shape,
                ),
                "viewpoints": viewpoints,
            }
        )

    gateway_groups = {}
    for first, neighbors in original_adjacency.items():
        for second in neighbors:
            if first >= second:
                continue
            first_area = cell_to_area[first]
            second_area = cell_to_area[second]
            if first_area == second_area:
                continue
            area_a, area_b = sorted((first_area, second_area))
            connection = (first, second)
            if first_area != area_a:
                connection = (second, first)
            gateway_groups.setdefault((area_a, area_b), []).append(connection)

    gateways = tuple(
        Gateway(
            area_a=area_pair[0],
            area_b=area_pair[1],
            connections=tuple(sorted(connections)),
        )
        for area_pair, connections in sorted(gateway_groups.items())
    )
    neighbor_ids = {draft["area_id"]: set() for draft in area_drafts}
    for gateway in gateways:
        neighbor_ids[gateway.area_a].add(gateway.area_b)
        neighbor_ids[gateway.area_b].add(gateway.area_a)

    areas = tuple(
        Area(
            area_id=draft["area_id"],
            cells=draft["cells"],
            centroid=draft["centroid"],
            position_label=draft["position_label"],
            viewpoints=tuple(draft["viewpoints"]),
            neighbors=tuple(sorted(neighbor_ids[draft["area_id"]])),
        )
        for draft in area_drafts
    )
    return AreaAnalysis(
        fingerprint=fingerprint,
        shape=tuple(int(value) for value in topology.shape),
        areas=areas,
        cell_to_area=cell_to_area,
        gateways=gateways,
        analysis_seconds=perf_counter() - started_at,
        error=None,
    )


def _fallback_analysis(topology, fingerprint, error):
    started_at = perf_counter()
    cells = frozenset(
        tuple(int(value) for value in position)
        for position in np.argwhere(topology)
    )
    if not cells:
        areas = ()
        cell_to_area = {}
    else:
        footprints = {
            cell: visibility_footprint(topology, cell, VISIBILITY_RADIUS)
            for cell in cells
        }
        centroid = _centroid(cells)
        area = Area(
            area_id=0,
            cells=cells,
            centroid=centroid,
            position_label=_position_label(centroid, topology.shape),
            viewpoints=_greedy_coverage(cells, footprints),
            neighbors=(),
        )
        areas = (area,)
        cell_to_area = {cell: 0 for cell in cells}
    return AreaAnalysis(
        fingerprint=fingerprint,
        shape=tuple(int(value) for value in topology.shape),
        areas=areas,
        cell_to_area=cell_to_area,
        gateways=(),
        analysis_seconds=perf_counter() - started_at,
        error=error,
    )


def _connected_components(adjacency):
    remaining = set(adjacency)
    components = []
    while remaining:
        start = min(remaining)
        frontier = deque([start])
        component = {start}
        remaining.remove(start)
        while frontier:
            current = frontier.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                component.add(neighbor)
                frontier.append(neighbor)
        components.append(frozenset(component))
    return tuple(sorted(components, key=lambda item: (-len(item), min(item))))


def _edge_betweenness(adjacency, component):
    ordered_nodes = tuple(sorted(component))
    node_indices = {
        node: index
        for index, node in enumerate(ordered_nodes)
    }
    indexed_neighbors = tuple(
        tuple(
            node_indices[neighbor]
            for neighbor in sorted(adjacency[node])
            if neighbor in component
        )
        for node in ordered_nodes
    )
    scores = {}

    for source in range(len(ordered_nodes)):
        stack = []
        predecessors = [[] for _ in ordered_nodes]
        path_counts = [0.0] * len(ordered_nodes)
        path_counts[source] = 1.0
        distances = [-1] * len(ordered_nodes)
        distances[source] = 0
        frontier = deque([source])

        while frontier:
            current = frontier.popleft()
            stack.append(current)
            for neighbor in indexed_neighbors[current]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[current] + 1
                    frontier.append(neighbor)
                if distances[neighbor] == distances[current] + 1:
                    path_counts[neighbor] += path_counts[current]
                    predecessors[neighbor].append(current)

        dependencies = [0.0] * len(ordered_nodes)
        while stack:
            current = stack.pop()
            for predecessor in predecessors[current]:
                contribution = (
                    path_counts[predecessor]
                    / path_counts[current]
                    * (1.0 + dependencies[current])
                )
                edge = (
                    min(predecessor, current),
                    max(predecessor, current),
                )
                scores[edge] = scores.get(edge, 0.0) + contribution
                dependencies[predecessor] += contribution

    return {
        (
            ordered_nodes[edge[0]],
            ordered_nodes[edge[1]],
        ): score / 2.0
        for edge, score in scores.items()
    }


def _bounded_coverage(
    component,
    footprints,
    maximum_viewpoints,
    cache,
):
    cached = cache.get(component)
    if cached is not None or component in cache:
        return cached

    cells = tuple(sorted(component))
    bit_for_cell = {cell: 1 << index for index, cell in enumerate(cells)}
    full_mask = (1 << len(cells)) - 1
    candidates = []
    for viewpoint in cells:
        mask = 0
        for visible_cell in footprints[viewpoint] & component:
            mask |= bit_for_cell[visible_cell]
        candidates.append((viewpoint, mask))

    candidates = _remove_dominated_candidates(candidates)
    largest_footprint = max(mask.bit_count() for _, mask in candidates)
    if math.ceil(len(cells) / largest_footprint) > maximum_viewpoints:
        cache[component] = None
        return None

    covering_candidates = {bit: [] for bit in bit_for_cell.values()}
    for candidate_index, (_, mask) in enumerate(candidates):
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            covering_candidates[bit].append(candidate_index)
            remaining ^= bit

    for limit in range(1, maximum_viewpoints + 1):
        failed_states = set()
        result = _search_cover(
            full_mask,
            limit,
            candidates,
            covering_candidates,
            failed_states,
        )
        if result is not None:
            selected = tuple(sorted(candidates[index][0] for index in result))
            cache[component] = selected
            return selected

    cache[component] = None
    return None


def _search_cover(
    uncovered,
    remaining_choices,
    candidates,
    covering_candidates,
    failed_states,
):
    if uncovered == 0:
        return ()
    if remaining_choices == 0:
        return None

    state = (uncovered, remaining_choices)
    if state in failed_states:
        return None

    maximum_gain = max(
        (mask & uncovered).bit_count()
        for _, mask in candidates
    )
    if maximum_gain == 0:
        failed_states.add(state)
        return None
    if math.ceil(uncovered.bit_count() / maximum_gain) > remaining_choices:
        failed_states.add(state)
        return None

    uncovered_bits = []
    remaining = uncovered
    while remaining:
        bit = remaining & -remaining
        uncovered_bits.append(bit)
        remaining ^= bit
    pivot = min(
        uncovered_bits,
        key=lambda bit: len(covering_candidates[bit]),
    )
    options = sorted(
        covering_candidates[pivot],
        key=lambda index: (
            -(candidates[index][1] & uncovered).bit_count(),
            candidates[index][0],
        ),
    )
    for candidate_index in options:
        new_uncovered = uncovered & ~candidates[candidate_index][1]
        result = _search_cover(
            new_uncovered,
            remaining_choices - 1,
            candidates,
            covering_candidates,
            failed_states,
        )
        if result is not None:
            return (candidate_index,) + result

    failed_states.add(state)
    return None


def _remove_dominated_candidates(candidates):
    unique = {}
    for viewpoint, mask in candidates:
        unique.setdefault(mask, viewpoint)
    ordered = sorted(
        ((viewpoint, mask) for mask, viewpoint in unique.items()),
        key=lambda item: (-item[1].bit_count(), item[0]),
    )
    retained = []
    for viewpoint, mask in ordered:
        if any(mask | retained_mask == retained_mask for _, retained_mask in retained):
            continue
        retained.append((viewpoint, mask))
    return tuple(retained)


def _greedy_coverage(component, footprints):
    uncovered = set(component)
    selected = []
    while uncovered:
        viewpoint = min(
            component,
            key=lambda cell: (
                -len(footprints[cell] & uncovered),
                cell,
            ),
        )
        selected.append(viewpoint)
        uncovered -= footprints[viewpoint]
    return tuple(selected)


def _merge_tiny_components(
    components,
    original_adjacency,
    footprints,
    coverage_cache,
):
    components = list(components)
    changed = True
    while changed:
        changed = False
        for component in sorted(components, key=lambda item: (len(item), min(item))):
            if len(component) >= MIN_AREA_CELLS:
                continue
            neighbors = []
            for other in components:
                if other is component:
                    continue
                gateway_count = sum(
                    1
                    for cell in component
                    for neighbor in original_adjacency[cell]
                    if neighbor in other
                )
                if gateway_count:
                    neighbors.append((gateway_count, other))
            for _, other in sorted(
                neighbors,
                key=lambda item: (-item[0], len(item[1]), min(item[1])),
            ):
                merged = frozenset(component | other)
                if _bounded_coverage(
                    merged,
                    footprints,
                    MAX_VIEWPOINTS_PER_AREA,
                    coverage_cache,
                ) is None:
                    continue
                components.remove(component)
                components.remove(other)
                components.append(merged)
                changed = True
                break
            if changed:
                break
    return tuple(components)


def _centroid(component):
    return (
        sum(cell[0] for cell in component) / len(component),
        sum(cell[1] for cell in component) / len(component),
    )


def _position_label(centroid, shape):
    vertical = _axis_label(
        centroid[0],
        shape[0],
        ("TOP", "MIDDLE", "BOTTOM"),
    )
    horizontal = _axis_label(
        centroid[1],
        shape[1],
        ("LEFT", "MIDDLE", "RIGHT"),
    )
    if vertical == "MIDDLE" and horizontal == "MIDDLE":
        return "CENTER"
    return f"{vertical}_{horizontal}"


def _axis_label(coordinate, length, labels):
    normalized = coordinate / max(1, length - 1)
    if normalized < 1 / 3:
        return labels[0]
    if normalized < 2 / 3:
        return labels[1]
    return labels[2]


def _map_fingerprint(topology):
    digest = hashlib.sha1()
    digest.update(str(tuple(int(value) for value in topology.shape)).encode("ascii"))
    digest.update(np.asarray(topology, dtype=np.uint8).tobytes())
    return digest.hexdigest()[:16]
