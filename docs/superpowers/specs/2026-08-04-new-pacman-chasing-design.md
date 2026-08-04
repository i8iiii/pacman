# New Pacman Agent: Chasing-First Design

## Goal

Replace the existing Pacman seeker with a new implementation designed from
scratch. The first increment focuses on deterministic pursuit after the Ghost
has been seen. Search and post-contact investigation remain intentionally
minimal placeholders for later refinement.

The existing seeker implementation may be consulted only for the framework
syntax and workflow. None of its behavioral classes, state, or algorithms will
be reused. The existing `GhostAgent` entry point and `hide_agent` implementation
remain unchanged.

## Authoritative Game Rules

The implementation targets the LAB2 assignment contract:

- The evaluation observation is a 21 by 21 grid, but implementation logic will
  derive dimensions from `map_state.shape` rather than hard-code them.
- A cell value of `1` is a wall.
- Cell values `0` and `-1` are traversable for path calculation.
- A value of `0` means the cell is currently visible to Pacman.
- A value of `-1` means the cell is outside Pacman's current field of view; the
  Ghost may occupy it.
- Pacman sees along unobstructed cardinal rays up to five cells away.
- Pacman and Ghost choose their actions from the same pre-movement state and
  then move simultaneously.
- In LAB2 evaluation, Pacman may move up to two cells in one straight
  direction per turn. The implementation reads the configured speed supplied
  by the framework rather than hard-coding this value.
- Capture occurs when the final Manhattan distance is strictly less than two.
- A decision must stay within the one-second time limit and 128 MB memory limit.

## Behavioral States

The new agent has three explicit modes.

### SEARCHING

`SEARCHING` means Pacman has no actionable information about the Ghost's
whereabouts. For this increment, it is only a placeholder: Pacman selects a
legal random movement. Search quality, coverage, destination selection, and
anti-oscillation behavior are out of scope.

Seeing the Ghost immediately transitions to `CHASING`.

### CHASING

`CHASING` means the Ghost is visible in the current observation.

On every chasing turn, Pacman:

1. Stores the Ghost's current position as the latest last-seen position.
2. Stores the current step number with that observation.
3. Computes a fresh shortest path from Pacman's current position to the visible
   Ghost position, treating every non-wall cell as traversable.
4. Returns the first direction on that path.
5. Requests two movement steps only when the first two edges of the path use
   the same direction. Otherwise, it requests one step.

The path is recalculated every turn because the target moves and the current
observation changes. This increment pursues the Ghost's observed position; it
does not predict the Ghost's next action or attempt interception.

If the Ghost is no longer visible and a last-seen position exists, Pacman
transitions to `INVESTIGATING`.

### INVESTIGATING

`INVESTIGATING` means Pacman previously saw the Ghost, but it is not visible in
the current observation.

Pacman computes a shortest path to the saved last-seen position using the same
non-wall topology. If the Ghost becomes visible during this movement, Pacman
immediately transitions to `CHASING`, replaces the saved observation with the
new one, and chases the current position.

If Pacman reaches the last-seen position without seeing the Ghost, it returns
`STAY` on subsequent turns. This stationary behavior is an explicitly accepted
placeholder. Returning to search or performing a local search will be designed
in a later increment.

## State Transitions

```text
SEARCHING -- Ghost visible --> CHASING
CHASING -- Ghost hidden --> INVESTIGATING
INVESTIGATING -- Ghost visible --> CHASING
INVESTIGATING -- last-seen position reached --> INVESTIGATING and STAY
```

A new match clears the mode, saved Ghost observation, current path, and any
other match-scoped seeker state. A lower step number than the previous call is
treated as a new match.

## Path and Action Rules

Shortest paths operate on a dynamically sized grid and may traverse values `0`
and `-1`; only value `1` blocks movement. The first implementation should use a
simple deterministic unweighted shortest-path algorithm suitable for the small
grid and one-second limit.

Pacman may return either a single `Move` or `(Move, steps)`. The new agent will
return the tuple form consistently. Requested steps must be between one and the
configured `pacman_speed`; the path action must not cross a turn in the path or
exceed its available straight prefix.

If no path exists unexpectedly, Pacman returns `(Move.STAY, 1)` and records the
failure diagnostically. This fallback must not raise an exception from
`step()`.

## Diagnostics

Diagnostics are added with the behavior rather than retrofitted later. For each
decision, they must make it possible to reconstruct:

- the step number and active mode;
- Pacman's position and visible Ghost position;
- the saved last-seen position and observation step;
- the received observation and its visible-versus-hidden cells;
- the non-wall topology used for path calculation;
- the selected target and computed path;
- the returned direction and number of steps;
- mode transitions and their reasons;
- unexpected failures or missing paths;
- decision duration.

Diagnostic failures must never change or terminate agent behavior. Diagnostic
work must be switchable off for tournament execution so it does not threaten
the time or memory limits. The precise file organization and output details
will be chosen during implementation, when their first concrete use appears.

## Code Organization Constraints

New seeker behavior belongs under `submissions/LAB2/seek_agent`. The top-level
`submissions/LAB2/agent.py` remains the framework entry point and preserves the
existing Ghost integration.

No speculative module hierarchy will be created. Code will be separated only
as responsibilities appear during implementation. A function used by multiple
components will be moved into a distinct shared helper location; behavior that
has only one consumer stays with its owner.

## Testing

The first increment requires focused tests for:

- interpreting both `0` and `-1` as traversable and `1` as blocked;
- entering `CHASING` when the Ghost is visible;
- updating the saved Ghost position on every visible turn;
- shortest-path direction selection around walls;
- requesting two steps only for a two-cell straight path prefix;
- requesting one step when the path turns after its first edge;
- transitioning from `CHASING` to `INVESTIGATING` on lost visibility;
- returning to `CHASING` when the Ghost reappears;
- moving to the last-seen position while investigating;
- staying after reaching the last-seen position without reacquisition;
- clearing state when a new match begins;
- returning a safe action rather than raising when no path is available;
- diagnostics receiving the expected decision facts without affecting actions.

Tests will use small synthetic grids where possible instead of depending only
on the default arena map. At least one integration test will load the new agent
through the real `AgentLoader` and validate its returned action.

## Explicitly Deferred

This increment does not design or implement:

- systematic or intelligent search;
- random destination selection or exploration coverage;
- Ghost belief distributions;
- interception or future-move prediction;
- local search around the last-seen position;
- abandonment of a completed investigation;
- adversarial planning such as minimax;
- learning-based behavior.
