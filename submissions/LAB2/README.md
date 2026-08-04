# LAB2: Pacman Agent

GitHub repository: [i8iiii/pacman](https://github.com/i8iiii/pacman)

LAB2 contains a topology-aware Pacman seeker and a Ghost hider. The submission
entry points are defined in `agent.py`; the seeker implementation is organized
under `seek_agent/` and the hider under `hide_agent/`.

## Pacman behavior

Pacman operates in three modes:

- **SEARCHING:** partitions the maze into connected areas, prioritizes them
  using the current Ghost belief, plans coverage routes, and may sweep an area
  it crosses when the additional travel cost is worthwhile.
- **CHASING:** immediately follows the shortest traversable path to a visible
  Ghost and remembers its last observed position.
- **INVESTIGATING:** after losing sight of the Ghost, moves toward the last-seen
  position and checks the Ghost's possible current positions for up to seven
  turns before returning to SEARCHING.

Previously observed cells expire according to their area's graph distance from
the highest-priority area. The current expiration durations are:

| Area distance | Expiration |
| --- | ---: |
| Priority area | 12 turns |
| 1 area hop | 24 turns |
| 2 area hops | 48 turns |
| 3 or more area hops | 96 turns |

An expired cell remains expired until Pacman observes it again. Never-observed
cells are always eligible for scouting.

## Ghost behavior

The Hider combines strategic concealment with tactical escape behavior:

- **SCOUT:** analyzes the complete topology, selects a strategic hideout, and
  follows a concealed route toward it. Candidate scoring considers dead ends,
  entrances, inspection depth, visibility exposure, spawn distance, and
  whether Pacman could intercept the route.
- **HIDE:** stays at the selected hideout while it remains uncompromised.
- **Visible escape:** when Pacman is visible, evaluates safe branches at useful
  junctions. Away from those junctions, it selects a mobile escape using
  capture safety, worst-case distance, trapping risk, continuation depth, and
  available region size.
- **HOT_UNSEEN:** after Pacman disappears, expands a set of possible Pacman
  positions using Pacman's speed and removes positions contradicted by current
  visibility. Movement is then chosen against this belief rather than assuming
  a single Pacman location.

The Hider detects major roads and predicts a repeating road-search cycle. It
avoids hideouts exposed by the active road stage, gradually relocates between
map bands, and marks a hideout as compromised after Pacman discovers it.

## Running a match

Run commands from the repository's `src` directory:

```powershell
cd pacman/src
python arena.py --seek LAB2 --hide LAB2 --pacman-obs-radius 5 --ghost-obs-radius 5
```

Run a random-start match without visualization:

```powershell
python arena.py --seek LAB2 --hide LAB2 --pacman-obs-radius 5 --ghost-obs-radius 5 --start-mode stochastic --no-viz
```

Useful arena options include `--max-steps`, `--pacman-speed`, `--no-viz`, and
`--start-mode {deterministic,stochastic}`. Run `python arena.py --help` for the
complete list.

## Diagnostics

### Seeker diagnostics

Seeker diagnostics are disabled by default. Enable them in PowerShell before
running the arena:

```powershell
$env:PACMAN_SEEK_DIAGNOSTICS = "1"
```

Disable them again with:

```powershell
$env:PACMAN_SEEK_DIAGNOSTICS = "0"
```

Diagnostic files are recreated for each match under `submissions/LAB2/debug/`:

- `seek-agent.jsonl` contains structured turn-by-turn decisions.
- `seek-agent-areas.txt` contains the human-readable area map and freshness
  state.

Area-map symbols:

- `###` — wall
- `P` — current Pacman position
- `/` — observed and still-fresh cell
- `<n>` — cell in the current target area
- `n` — unseen or expired cell belonging to area `n`

The area descriptions also report possible Ghost cells, priority status,
area-hop distance, and expiration duration.

### Hider diagnostics

Hider diagnostics are also disabled by default. Enable them by setting
`DIAGNOSTICS_ENABLED = True` in `hide_agent/diagnostics.py`, or by constructing
`GhostAgent` with `diagnostics_enabled=True` in a custom runner.

The Hider recreates these files for every match:

- `debug/hide-agent.log` records state changes, hideout selection, belief
  updates, road stages, escape scoring, and final decisions.
- `debug/hide-agent-map.txt` provides readable map snapshots.
- `debug/hide-agent-map.jsonl` provides the same map state in structured form.

## Seeker module guide

- `controller.py` coordinates SEARCHING, CHASING, and INVESTIGATING.
- `belief.py` tracks possible current Ghost positions and observation times.
- `areas.py` partitions the topology and records area connections.
- `search.py` prioritizes areas and manages transit and sweep behavior.
- `routes.py` builds local area-coverage routes.
- `investigation.py` handles last-seen-position investigation.
- `freshness.py` owns the shared area-based expiration policy.
- `spatial.py` contains common movement, visibility, and pathfinding helpers.
- `diagnostics.py` writes the JSONL log and readable area map.

## Hider module guide

- `controller.py` coordinates SCOUT, HIDE, visible escape, and HOT_UNSEEN.
- `concealment.py` discovers and ranks strategic hideouts and campsites.
- `relocation.py` detects major roads, predicts road stages, and plans gradual
  movement between map regions.
- `evasion.py` scores junction and mobile escape choices while Pacman is
  visible.
- `belief.py` tracks possible unseen Pacman positions, pursuit direction, and
  interception risk.
- `spatial.py` contains structural movement, visibility, capture, and concealed
  routing helpers.
- `diagnostics.py` writes Hider event logs and synchronized map snapshots.
