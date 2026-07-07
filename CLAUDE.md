# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pacman vs Ghost Arena — a university AI project where students implement search algorithm agents (Pacman seeker, Ghost hider) that compete on a grid maze. The framework runs games between student-submitted agents and evaluates them.

## Commands

```bash
# Install environment (uses uv + Python 3.14)
uv sync

# Run a game (from src/ directory)
cd src
python arena.py --seek <student_id> --hide <student_id>

# Quick run via shell script (from repo root)
./run_game.sh <seeker_id> <hider_id>

# Common arena options
python arena.py --seek <id> --hide <id> --no-viz          # no visualization (faster)
python arena.py --seek <id> --hide <id> --delay 1.0        # slow viz for debugging
python arena.py --seek <id> --hide <id> --max-steps 300    # longer game
python arena.py --seek <id> --hide <id> --start-mode stochastic  # random starts
python arena.py --seek <id> --hide <id> --capture-distance 3     # catch threshold
python arena.py --seek <id> --hide <id> --pacman-speed 2         # Pacman speed multiplier
python arena.py --seek <id> --hide <id> --pacman-obs-radius 5 --ghost-obs-radius 3  # fog of war
python arena.py --seek <id> --hide <id> --step-timeout 1.0       # per-step timeout
```

## Architecture

**Framework (`src/`):**
- `arena.py` — Game orchestrator. Parses CLI args, loads agents, runs the game loop, handles timeouts/errors. Entry point for all games.
- `environment.py` — Game state: map, positions, move execution, win conditions, fog-of-war observation system. Defines `Move` enum and `CellType` enum.
- `agent_interface.py` — Abstract base classes `PacmanAgent` and `GhostAgent` that student agents must inherit from.
- `agent_loader.py` — Dynamically loads student `agent.py` files via `importlib`, validates class names and inheritance, validates move types.
- `visualizer.py` — Terminal-based game visualization with ANSI colors.

**Student submissions (`submissions/<id>/agent.py`):**
- Each student folder contains an `agent.py` defining `PacmanAgent` and/or `GhostAgent` classes.
- Agents receive `(map_state, my_position, enemy_position, step_number)` each step and return a `Move` (or `(Move, steps)` tuple for Pacman with speed > 1).
- The loader adds the student's folder to `sys.path`, so helper modules alongside `agent.py` are importable.

**Key game mechanics:**
- Both agents move simultaneously each step.
- Map values: `0` = empty, `1` = wall, `-1` = unseen (fog of war).
- Pacman wins when Manhattan distance to Ghost < capture_distance_threshold (default 2 in arena CLI). Ghost wins if it survives max_steps.
- When `--pacman-speed N` is set, Pacman can return `(Move, steps)` where 1 ≤ steps ≤ N, moving multiple tiles in a straight line per turn (stops at walls/turns).
- Observation radius > 0 enables fog of war: cross-shaped vision from agent position, walls block sight lines. Enemy position is `None` when outside visible range.

## Agent Implementation Notes

- Class names must be exactly `PacmanAgent` and `GhostAgent` (case-sensitive).
- Must inherit from the base classes in `agent_interface.py`.
- `PacmanAgent.__init__` receives `pacman_speed` kwarg; `GhostAgent.__init__` receives no special kwargs.
- Returning invalid move types (string, None, wrong tuple) causes the agent to lose by default.
- Agent step timeout (default 1.0s in CLI) causes loss on timeout.
- Imports from `src/` use path manipulation: `sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))`.

## Dependencies

Managed via `uv` (see `pyproject.toml`). Key packages: numpy, scipy, scikit-learn, ortools, pyomo, cvxpy, cpmpy, minizinc. The `requirements.txt` is a simpler fallback. Torch is optional (used by some DQN-based agents but not in project dependencies).
