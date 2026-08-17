# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A course arena for a Pacman-vs-Ghost pursuit/evasion lab. `src/` is the instructor-provided
framework (do not change it to make an agent win). `submissions/<id>/` holds competing agents;
the active work is `submissions/LAB2/4/`. An agent ID is the **path under `submissions/`**, so
`--seek LAB2/4` loads `submissions/LAB2/4/agent.py`. Every submission lives in a numbered
subfolder (`LAB1/4`, `LAB2/4`, `run_benchmark/0`…); a bare `--seek LAB2` will not load.

## Commands

Run everything from `src/` (the framework imports `environment`/`agent_interface` as top-level
modules, so `src/` must be the working directory or on `sys.path`).

```powershell
cd src
$env:PYTHONIOENCODING = "utf-8"   # required on Windows — see below
python arena.py --seek LAB2/4 --hide LAB2/4 --pacman-obs-radius 5 --ghost-obs-radius 5
```

Common variants:

```powershell
# fast, no rendering — use this for batch/iteration work
python arena.py --seek LAB2/4 --hide LAB2/4 --pacman-obs-radius 5 --ghost-obs-radius 5 --no-viz

# random spawns instead of the map's fixed P/G markers
python arena.py --seek LAB2/4 --hide LAB2/4 --start-mode stochastic --no-viz

# slow playback for eyeballing behavior
python arena.py --seek LAB2/4 --hide LAB2/4 --delay 0.5

# pit LAB2 against a benchmark opponent
python arena.py --seek LAB2/4 --hide run_benchmark/11 --no-viz
```

`../run_game.sh <seeker_id> <hider_id> [arena options]` is a wrapper, but note it takes
**positional** IDs (not `--seek/--hide`) and prefers `conda run -n ml python`; on this machine
`.venv/Scripts/python.exe` is the working interpreter, so calling `arena.py` directly is usually
simpler.

There is no test suite and no linter config. Verification means running matches and reading
diagnostics.

### Tournament benchmark

`src/run_all_matches.py` plays every `run_benchmark/N` seeker against every `run_benchmark/N`
hider (skipping self-matches), `MATCHES_PER_PAIR = 3` games each, at radius 5 / stochastic
starts / no viz. It writes `src/results.csv` (per-matchup rows, then SEEKER and HIDER summary
blocks) and appends crashed games to `src/error_log.txt`.

```powershell
cd src
python run_all_matches.py
```

Two things to know before running it:

- `SEEKERS`/`HIDERS` at the top of the file are `[0, 2..16]` — 240 matchups × 3 = 720 games, a
  long run. Trim those lists to benchmark a subset.
- **`run_benchmark/4` is a snapshot copy of `LAB2/4`, not a link.** Editing `LAB2/4` does not
  change what the tournament plays; copy the packages over first, or add `LAB2/4` to the lists
  via `make_path`.
- Scoring convention: fewer steps is better for the seeker, more steps is better for the hider.

### CLI defaults that differ from the class defaults

`arena.py` defaults are `--capture-distance 2`, `--pacman-speed 2`, `--step-timeout 1.0`,
`--max-steps 200`, observation radii `0` (full visibility). The `Arena`/`Environment`
constructors default to `capture_distance_threshold=1, pacman_speed=1` — so a match constructed
in Python is *not* the same game as the CLI unless you pass those explicitly. Agents in LAB2
assume speed 2 / capture 2 / radius 5.

Capture is `manhattan_distance < capture_distance_threshold`, so `--capture-distance 2` means
capture on **adjacency or overlap** (distance ≤ 1), not distance 2. Under `--start-mode
stochastic`, Pacman spawns uniformly among empty cells in the bottom 40% of rows and the Ghost
in the top 40% — not anywhere on the map.

## The agent contract (what the framework enforces)

- `submissions/<id>/agent.py` must define `PacmanAgent` and/or `GhostAgent` subclassing the
  bases in `src/agent_interface.py`. `AgentLoader` inserts the submission folder into `sys.path`,
  so sibling packages (`seek_agent/`, `hide_agent/`) import as top-level.
- `PacmanAgent.__init__` receives `pacman_speed`; `GhostAgent.__init__` receives nothing. Any
  other constructor kwarg (diagnostics paths, `capture_distance`, `observation_radius`) is only
  reachable from a custom runner, never from `arena.py`.
- `step()` returns a `Move` for Ghost; `Move` **or** `(Move, steps)` for Pacman with
  `1 <= steps <= pacman_speed`. Exceeding the speed cap raises and forfeits the match.
- **Any exception or timeout inside `step()` forfeits the game instantly** — the arena awards
  the win to the opponent and prints a traceback to stderr. Both LAB2 controllers therefore wrap
  their whole step body in a try/except that degrades to `Move.STAY`.
- Both agents move **simultaneously**; you never see the opponent's current move.
- `map_state` is `0` empty, `1` wall, `-1` unseen. Walls are always truthful even under fog.
  LAB2 treats unseen as traversable everywhere via `topology = observation != 1` — an important
  invariant: pathfinding plans *through* fog, and belief tracking is what narrows it down.
- `enemy_position` is `None` whenever the enemy is outside the observer's cross-shaped
  line-of-sight (`Environment.get_visible_cells_cross`: 4 cardinal rays, blocked by walls — not
  a radius disc).
- Agents are long-lived across matches in a single process. Both LAB2 controllers detect a new
  match by `step_number <= self._last_step_number` (hide also checks map shape) and reset
  per-match state; keep that guard intact when adding state.

### Windows caveats

**Console encoding.** `arena.py` prints `✓` and `🏆`, which raises `UnicodeEncodeError` on the
default cp1252 console and kills the run *after* the agents load — it looks like an agent load
failure but is not. Set `$env:PYTHONIOENCODING = "utf-8"` before invoking `arena.py` directly
(`run_all_matches.py` already sets it for its subprocesses).

**No step timeout.** The per-step timeout uses `signal.SIGALRM`, which does not exist on Windows.
`arena.py` prints `WARNING: Step timeout requested but SIGALRM is unavailable` and disables the
timeout — so a slow agent that would forfeit on the grader's machine passes silently here.
`SearchPlanner` enforces its own `PLANNING_TIME_LIMIT_SECONDS = 0.35` budget instead; respect
that when adding planning work.

## LAB2 architecture

`submissions/LAB2/4/agent.py` is a thin adapter: `PacmanAgent` → `seek_agent.controller.SeekController`,
`GhostAgent` → `hide_agent.controller.HideController`. All logic lives in the two packages, and
`submissions/LAB2/4/README.md` is the detailed design document — read it before changing behavior.

**Seeker** (`seek_agent/`) is a three-mode state machine driven by the controller:
`CHASING` (ghost visible → shortest path) → `INVESTIGATING` (ghost just lost → probe last-seen
position and belief cells, bounded turns) → `SEARCHING` (no information). The supporting layers
are each single-responsibility and the controller is the only thing that owns mode transitions:
`areas.py` partitions the maze into connected areas (cached by map fingerprint), `belief.py`
maintains the possible-ghost-position set, `freshness.py` expires observed cells on a schedule
keyed to area-hop distance from the priority area, `search.py` owns SEARCHING's phase lifecycle
(select → transit → optional opportunistic sweep → sweep → complete), `routes.py` builds
in-area coverage routes, `spatial.py` is the shared BFS/visibility/move-delta layer.

**Hider** (`hide_agent/`) is `SCOUT` → `HIDE` when unseen, with two reactive overrides: visible
escape (`evasion.py`: junction-branch scoring first, then mobile escape) and `HOT_UNSEEN`
(`belief.py`: expand possible Pacman positions by Pacman's speed, prune by what the ghost can
currently see, then move against the whole belief set rather than a point estimate). On top sits
a migration layer (`relocation.py`) that detects major roads, models the seeker's repeating road
cycle, and walks the ghost through phases `TO_MIDDLE → MIDDLE_HOLD → TO_OPPOSITE →
DEEPEN_OPPOSITE → OPPOSITE_HOLD` across map bands. `concealment.py` scores hideout candidates;
a hideout the seeker ever sees is added to `_compromised_hideouts` for the rest of the match.

Both packages are written so diagnostics are pure observation — logging must never influence the
returned move.

## Diagnostics

Off by default; both write per-match files into a `debug/` folder next to `agent.py`
(`submissions/LAB2/4/debug/`).

```powershell
$env:PACMAN_SEEK_DIAGNOSTICS = "1"   # seeker: debug/seek-agent.jsonl + seek-agent-areas.txt
```

The hider has no env switch — set `DIAGNOSTICS_ENABLED = True` in `hide_agent/diagnostics.py`
(or pass `diagnostics_enabled=True` from a custom runner) to get `hide-agent.log`,
`hide-agent-map.txt`, `hide-agent-map.jsonl`. `seek-agent-areas.txt` is a custom rendered map
format (the `.vscode/` syntax highlighter for it was removed in `d3d5603`).

`.gitignore` still lists the pre-restructure path `submissions/LAB2/debug/`, so seeker
diagnostics under `LAB2/4/debug/` are **not** ignored — `run_benchmark/4/debug/*.jsonl` already
got committed this way. Check `git status` before committing after a diagnostics run.

## Other directories

- `submissions/LAB1/4/` — previous lab, kept for reference.
- `submissions/run_benchmark/{0,2..16}/` — other students' agents, the sparring pool for
  `run_all_matches.py` (there is no `1`).
- `submissions/LAB2/{seek,hide}_agent/` hold only stale `__pycache__` from before the move to
  `LAB2/4/`; ignore them.
- `src/tempCodeRunnerFile.py`, `src/results.csv`, `src/error_log.txt` are run artifacts, not
  framework code.
- `--sandbox` on `arena.py` swaps in an `instructors/agent_loader.py` that is not present in this
  repo; the flag will fail here.

## Local framework modification

`src/environment.py`'s classic map has been edited locally (`d3d5603`): the `P` and `G` start
markers were moved, so `--start-mode deterministic` here does not reproduce the instructor's
fixed spawns. Keep this in mind when a deterministic result disagrees with the grader.
