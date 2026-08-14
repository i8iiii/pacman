"""Area-aware scouting freshness shared by SEARCHING and diagnostics."""

from collections import deque


BASE_EXPIRATION_TURNS = 12
MAX_EXPIRATION_TURNS = 96


class AreaFreshnessTracker:
    """Track monotonic cell expiration using dynamic area priorities."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Discard all state whose lifetime is one arena match."""
        self.priority_area_ids = ()
        self.area_belief_counts = {}
        self.area_distances = {}
        self.expiration_turns = {}
        self.expired_cells = set()

    def update(self, analysis, belief, visible_cells, step_number):
        """Refresh visible cells, then expire other cells by area urgency."""
        visible = {_position(cell) for cell in visible_cells}
        observed = {
            _position(cell): int(observed_step)
            for cell, observed_step in belief.last_observed_step.items()
        }
        self.expired_cells.intersection_update(observed)
        self.expired_cells.difference_update(visible)

        self._update_area_durations(analysis, belief.possible_positions)
        current_step = int(step_number)
        for cell, observed_step in observed.items():
            if cell in visible:
                continue
            area_id = analysis.cell_to_area.get(cell)
            if area_id is None:
                continue
            duration = self.expiration_turns.get(
                area_id,
                BASE_EXPIRATION_TURNS,
            )
            if max(0, current_step - observed_step) >= duration:
                self.expired_cells.add(cell)

    def fresh_cells(self, belief):
        """Return observed cells that have not expired since last visibility."""
        return frozenset(
            _position(cell)
            for cell in belief.last_observed_step
            if _position(cell) not in self.expired_cells
        )

    def required_cells_in_area(self, area, belief):
        """Return never-seen and expired cells belonging to one area."""
        observed = {_position(cell) for cell in belief.last_observed_step}
        cells = {_position(cell) for cell in area.cells}
        return frozenset(
            cell
            for cell in cells
            if cell not in observed or cell in self.expired_cells
        )

    def _update_area_durations(self, analysis, possible_positions):
        possible = {_position(cell) for cell in possible_positions}
        counts = {
            area.area_id: len(possible & {_position(cell) for cell in area.cells})
            for area in analysis.areas
        }
        self.area_belief_counts = counts
        if not counts:
            self.priority_area_ids = ()
            self.area_distances = {}
            self.expiration_turns = {}
            return

        maximum = max(counts.values())
        if maximum > 0:
            priorities = tuple(sorted(
                area_id
                for area_id, count in counts.items()
                if count == maximum
            ))
        else:
            # An empty belief is an inconsistent recovery state.  Treat every
            # area as equally urgent instead of preserving stale information.
            priorities = tuple(sorted(counts))

        areas_by_id = {area.area_id: area for area in analysis.areas}
        distances = {area_id: 0 for area_id in priorities}
        frontier = deque(priorities)
        while frontier:
            area_id = frontier.popleft()
            area = areas_by_id.get(area_id)
            if area is None:
                continue
            for neighbor_id in sorted(area.neighbors):
                if neighbor_id not in areas_by_id or neighbor_id in distances:
                    continue
                distances[neighbor_id] = distances[area_id] + 1
                frontier.append(neighbor_id)

        self.priority_area_ids = priorities
        self.area_distances = distances
        self.expiration_turns = {
            area_id: _expiration_for_distance(distances.get(area_id))
            for area_id in counts
        }


def _expiration_for_distance(distance):
    if distance is None:
        return MAX_EXPIRATION_TURNS
    capped_distance = min(3, max(0, int(distance)))
    return min(
        BASE_EXPIRATION_TURNS * (2 ** capped_distance),
        MAX_EXPIRATION_TURNS,
    )


def _position(value):
    return int(value[0]), int(value[1])
