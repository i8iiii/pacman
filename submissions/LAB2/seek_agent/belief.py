"""Set-based Ghost position belief and scouting-freshness tracking."""


class GhostBelief:
    """Track Ghost locations consistent with the seeker's observations.

    The belief is deliberately a set rather than a probability distribution.
    Callers provide visible cells explicitly because the observation map also
    contains globally known walls.
    """

    def __init__(self):
        self.possible_positions = set()
        self.last_observed_step = {}
        self._traversable_cells = frozenset()
        self._last_step_number = None

    def reset(self, map_state, pacman_position, visible_cells=(), step_number=0):
        """Seed the belief from the opposite spawn band for a new match."""
        step_number = int(step_number)
        pacman_position = _position(pacman_position)
        self._traversable_cells = _traversable_cells(map_state)
        self.last_observed_step = {}
        self._last_step_number = step_number

        visible_cells = self._record_visible(
            map_state,
            visible_cells,
            step_number,
        )
        height = _height(map_state)
        pacman_row = pacman_position[0]
        if pacman_row < height * 0.4:
            candidates = {
                cell for cell in self._traversable_cells
                if cell[0] > height * 0.6
            }
        elif pacman_row > height * 0.6:
            candidates = {
                cell for cell in self._traversable_cells
                if cell[0] < height * 0.4
            }
        else:
            candidates = set()

        if not candidates:
            candidates = set(self._traversable_cells - visible_cells)
        self.possible_positions = candidates - visible_cells
        return self.sorted_possible_positions()

    def update(self, map_state, visible_cells, enemy_position, step_number):
        """Advance unseen belief and apply this turn's visibility evidence."""
        step_number = int(step_number)
        self._traversable_cells = _traversable_cells(map_state)
        visible_cells = self._record_visible(
            map_state,
            visible_cells,
            step_number,
        )

        if enemy_position is not None:
            self.possible_positions = {_position(enemy_position)}
            self._last_step_number = step_number
            return self.sorted_possible_positions()

        if self._last_step_number is None:
            self._last_step_number = step_number
        elif step_number > self._last_step_number:
            for _ in range(step_number - self._last_step_number):
                self.possible_positions = self._expand_once(
                    self.possible_positions,
                )
            self._last_step_number = step_number
        # A controller reset owns decreasing step numbers.  Keep the previous
        # time anchor here instead of creating a negative elapsed interval.

        self.possible_positions.difference_update(visible_cells)
        return self.sorted_possible_positions()

    def sorted_possible_positions(self):
        """Return the current position set in deterministic order."""
        return tuple(sorted(self.possible_positions))

    def possible_positions_in_area(self, area):
        """Return possible Ghost cells belonging to ``area``, sorted."""
        return tuple(sorted(self.possible_positions & _area_cells(area)))

    def possible_positions_in(self, area):
        """Alias for :meth:`possible_positions_in_area`."""
        return self.possible_positions_in_area(area)

    def risk_fraction_for_area(self, area):
        """Return the share of remaining belief mass that lies in ``area``."""
        if not self.possible_positions:
            return 0.0
        return (
            len(self.possible_positions & _area_cells(area))
            / len(self.possible_positions)
        )

    def risk_fraction_in(self, area):
        """Alias for :meth:`risk_fraction_for_area`."""
        return self.risk_fraction_for_area(area)

    def never_observed_in_area(self, area):
        """Return traversable area cells that have never been visible."""
        return tuple(sorted(
            cell
            for cell in _area_cells(area) & self._traversable_cells
            if cell not in self.last_observed_step
        ))

    def never_observed_in(self, area):
        """Alias for :meth:`never_observed_in_area`."""
        return self.never_observed_in_area(area)

    def last_observed_age(self, cell, step_number=None):
        """Return a cell's non-negative observation age, or ``None`` if new."""
        observed_at = self.last_observed_step.get(_position(cell))
        if observed_at is None:
            return None
        if step_number is None:
            step_number = self._last_step_number
        if step_number is None:
            return None
        return max(0, int(step_number) - observed_at)

    def _record_visible(self, map_state, visible_cells, step_number):
        visible_traversable = {
            _position(cell)
            for cell in visible_cells
            if _is_traversable(map_state, _position(cell))
        }
        for cell in visible_traversable:
            self.last_observed_step[cell] = step_number
        return visible_traversable

    def _expand_once(self, positions):
        expanded = set()
        for row, column in positions:
            for candidate in (
                (row, column),
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if candidate in self._traversable_cells:
                    expanded.add(candidate)
        return expanded


def _position(value):
    return int(value[0]), int(value[1])


def _height(map_state):
    return int(map_state.shape[0])


def _traversable_cells(map_state):
    height, width = map_state.shape[:2]
    return frozenset(
        (row, column)
        for row in range(int(height))
        for column in range(int(width))
        if map_state[row, column] in (0, -1)
    )


def _is_traversable(map_state, cell):
    row, column = cell
    height, width = map_state.shape[:2]
    return (
        0 <= row < height
        and 0 <= column < width
        and map_state[row, column] in (0, -1)
    )


def _area_cells(area):
    cells = getattr(area, "cells", area)
    return {_position(cell) for cell in cells}
