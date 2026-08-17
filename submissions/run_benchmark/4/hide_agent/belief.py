"""Where Pacman could be, and how soon he could look at a given cell.

Hide only ever sees Pacman inside a wall-blocked cardinal cross, so for most of
a match ``enemy_position`` is ``None``.  :class:`PacmanBelief` keeps the set of
cells Pacman could currently occupy so the controller can reason about threat
while blind, instead of behaving as though Pacman does not exist.
"""

from collections import deque

from .spatial import (
    has_line_of_sight,
    is_structurally_traversable,
    pacman_endpoints,
)


UNREACHABLE = 10**9


def pacman_turn_distances(
    map_state,
    belief_positions,
    pacman_speed=2,
):
    """Return exact multi-source Pacman action-graph distances.

    Distance is counted in Pacman *turns*, not tiles: one turn expands to every
    endpoint reachable by a single straight action at ``pacman_speed``.
    """
    starts = tuple(
        sorted(
            {
                tuple(position)
                for position in belief_positions
                if is_structurally_traversable(map_state, position)
            }
        )
    )
    if not starts:
        return {}

    distances = {position: 0 for position in starts}
    queue = deque(starts)
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for endpoint in pacman_endpoints(
            map_state,
            current,
            speed=pacman_speed,
        ):
            endpoint = tuple(endpoint)
            if endpoint in distances:
                continue
            distances[endpoint] = next_distance
            queue.append(endpoint)
    return distances


def turns_until_observed(threat_distances, observers):
    """Return Pacman turns until any cell that can see the target is reached.

    ``observers`` is a visibility footprint: every cell with line of sight to
    the target.  Reaching one of them is what actually exposes the hideout, so
    this is the quantity a hold decision should be made against - not the turns
    needed to physically arrive at the target.
    """
    if not threat_distances:
        return UNREACHABLE
    reachable = [
        threat_distances[observer]
        for observer in observers
        if observer in threat_distances
    ]
    return min(reachable) if reachable else UNREACHABLE


class PacmanBelief:
    """The set of cells Pacman could occupy right now.

    A sighting collapses the set to a single cell.  Every blind turn expands it
    by one Pacman action and then removes whatever the Ghost can currently see,
    because Pacman standing there would have been visible.
    """

    def __init__(self, pacman_speed=2, observation_radius=5):
        self._pacman_speed = max(1, int(pacman_speed))
        self._observation_radius = max(0, int(observation_radius))
        self._positions = frozenset()
        self._last_seen = None
        self._turns_since_seen = None

    @property
    def positions(self):
        return self._positions

    @property
    def last_seen(self):
        return self._last_seen

    @property
    def turns_since_seen(self):
        return self._turns_since_seen

    def reset(self, map_state, ghost_position):
        """Seed the belief from the Arena's opposite-band spawn rule."""
        self._positions = frozenset(self._spawn_prior(map_state, ghost_position))
        self._last_seen = None
        self._turns_since_seen = None

    def update(self, map_state, ghost_position, enemy_position):
        if enemy_position is not None:
            self._positions = frozenset({tuple(enemy_position)})
            self._last_seen = tuple(enemy_position)
            self._turns_since_seen = 0
            return self._positions

        if not self._positions:
            self._positions = frozenset(
                self._spawn_prior(map_state, ghost_position)
            )

        expanded = set()
        for position in self._positions:
            expanded.update(
                pacman_endpoints(
                    map_state,
                    position,
                    speed=self._pacman_speed,
                )
            )

        pruned = {
            position
            for position in expanded
            if not has_line_of_sight(
                map_state,
                ghost_position,
                position,
                self._observation_radius,
            )
        }
        # An empty belief would assert Pacman is nowhere; keep the unpruned set
        # rather than claim the map is safe.
        self._positions = frozenset(pruned or expanded)
        if self._turns_since_seen is not None:
            self._turns_since_seen += 1
        return self._positions

    def _spawn_prior(self, map_state, ghost_position):
        """Environment.reset spawns Pacman in the row band opposite the Ghost."""
        rows, columns = map_state.shape
        ghost_in_top = int(ghost_position[0]) < rows / 2.0
        prior = [
            (row, column)
            for row in range(rows)
            for column in range(columns)
            if (row >= rows * 0.6 if ghost_in_top else row < rows * 0.4)
            and is_structurally_traversable(map_state, (row, column))
        ]
        if prior:
            return prior
        return [
            (row, column)
            for row in range(rows)
            for column in range(columns)
            if is_structurally_traversable(map_state, (row, column))
        ]
