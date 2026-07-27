# Hide-Agent Module Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the Hide-agent implementation into seven functional modules without changing runtime behavior.

**Architecture:** Merge modules by behavioral domain while preserving all consumed names and signatures. `controller.py` remains the only orchestrator, and dependencies continue to point from orchestration toward domain modules and the shared spatial foundation.

**Tech Stack:** Python 3.11-compatible source, `unittest`, NumPy, existing Arena interfaces.

## Global Constraints

- Do not change strategy, scoring, constants, state transitions, randomness, or public return values.
- Preserve `GhostAgent` and `HideController.step(...)`.
- Keep diagnostics disabled by default.
- Keep tests outside `submissions/LAB2`.
- Do not leave superseded modules, caches, or debug output in the final submission.

---

### Task 1: Add consolidation regression tests

**Files:**
- Create: `tests/test_hide_agent_consolidation.py`

**Interfaces:**
- Consumes: Arena `Move`, `Environment`, and `AgentLoader`.
- Produces: executable behavioral gates for the merged module paths and Hide entry point.

- [ ] **Step 1: Write the failing tests**

```python
import sys
import unittest
from pathlib import Path

PACMAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACMAN_ROOT / "src"))
sys.path.insert(0, str(PACMAN_ROOT / "submissions" / "LAB2"))

from environment import Environment, Move


class HideAgentConsolidationTests(unittest.TestCase):
    def test_merged_domain_apis_are_importable(self):
        from hide_agent.spatial import move_between, vertical_band
        from hide_agent.concealment import scan_campsites, scan_hideouts
        from hide_agent.relocation import build_road_cycle, migration_direction
        from hide_agent.evasion import (
            choose_visible_junction_escape,
            choose_visible_mobile_escape,
        )
        from hide_agent.belief import PacmanBeliefTracker, PursuitTracker

        self.assertEqual(move_between((1, 1), (1, 2)), Move.RIGHT)
        self.assertEqual(vertical_band((1, 1), 21), "top")
        self.assertTrue(callable(scan_campsites))
        self.assertTrue(callable(scan_hideouts))
        self.assertTrue(callable(build_road_cycle))
        self.assertTrue(callable(migration_direction))
        self.assertTrue(callable(choose_visible_junction_escape))
        self.assertTrue(callable(choose_visible_mobile_escape))
        self.assertIsInstance(PacmanBeliefTracker(), PacmanBeliefTracker)
        self.assertIsInstance(PursuitTracker(), PursuitTracker)

    def test_ghost_agent_returns_a_valid_move_without_debug_output(self):
        from agent import GhostAgent

        debug_dir = PACMAN_ROOT / "submissions" / "LAB2" / "debug"
        agent = GhostAgent()
        environment = Environment(
            pacman_speed=2,
            capture_distance_threshold=2,
        )
        observation, position, enemy = environment.get_observation("ghost", 5, 5)

        action = agent.step(observation, position, enemy, 1)

        self.assertIsInstance(action, Move)
        self.assertFalse(debug_dir.exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -B -m unittest tests.test_hide_agent_consolidation -v
```

Expected: error importing `hide_agent.spatial` because merged modules do not yet exist.

### Task 2: Merge spatial and concealment modules

**Files:**
- Create: `submissions/LAB2/hide_agent/spatial.py`
- Create: `submissions/LAB2/hide_agent/concealment.py`
- Delete: `submissions/LAB2/hide_agent/geometry.py`
- Delete: `submissions/LAB2/hide_agent/navigation.py`
- Delete: `submissions/LAB2/hide_agent/cross_map.py`
- Delete: `submissions/LAB2/hide_agent/hideout.py`
- Delete: `submissions/LAB2/hide_agent/topology.py`

**Interfaces:**
- `spatial.py` produces all former geometry, navigation, and cross-map names.
- `concealment.py` produces all former hideout and topology names.

- [ ] **Step 1: Move the three spatial implementations into `spatial.py`**

Keep function bodies unchanged. Remove the former `navigation.py` import from
`.geometry`, because those definitions now share a module.

- [ ] **Step 2: Move hideout and topology implementations into `concealment.py`**

Rewrite imports from `.geometry` and `.navigation` to `.spatial`. Preserve the
import of `pacman_turn_distances` from `.belief`.

- [ ] **Step 3: Update consumers**

Replace imports of `.geometry`, `.navigation`, `.cross_map`, `.hideout`, and
`.topology` with `.spatial` or `.concealment` in all remaining Hide modules.

- [ ] **Step 4: Run the focused tests**

Expected: the first import advances to the next missing merged module; no
syntax or circular-import error may occur.

### Task 3: Merge relocation, evasion, and belief modules

**Files:**
- Create: `submissions/LAB2/hide_agent/relocation.py`
- Create: `submissions/LAB2/hide_agent/evasion.py`
- Modify: `submissions/LAB2/hide_agent/belief.py`
- Delete: `submissions/LAB2/hide_agent/roads.py`
- Delete: `submissions/LAB2/hide_agent/migration.py`
- Delete: `submissions/LAB2/hide_agent/escape.py`
- Delete: `submissions/LAB2/hide_agent/mobile_escape.py`
- Delete: `submissions/LAB2/hide_agent/pursuit.py`

**Interfaces:**
- `relocation.py` produces all road and migration names.
- `evasion.py` produces all visible-junction and mobile-evasion names.
- `belief.py` produces both broad-belief and pursuit names.

- [ ] **Step 1: Merge roads and migration**

Rewrite spatial imports to `.spatial`, concealment imports to
`.concealment`, and remove migration's former import from `.roads`.

- [ ] **Step 2: Merge escape and mobile escape**

Rewrite geometry imports to `.spatial` and remove mobile escape's former
import from `.escape`.

- [ ] **Step 3: Append pursuit to belief**

Rewrite geometry and navigation imports to `.spatial`. Resolve private helper
name collisions without altering either call path.

- [ ] **Step 4: Update `controller.py` imports**

Import all domain APIs from `spatial`, `belief`, `concealment`, `relocation`,
and `evasion`.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Expected: both tests pass.

### Task 4: Final package and behavior verification

**Files:**
- Verify: `submissions/LAB2/agent.py`
- Verify: `submissions/LAB2/hide_agent/*.py`

- [ ] **Step 1: Scan for stale imports**

Run:

```powershell
rg -n "\.(geometry|navigation|cross_map|hideout|topology|roads|migration|escape|mobile_escape|pursuit)" submissions/LAB2
```

Expected: no Python import matches.

- [ ] **Step 2: Compile without retaining caches**

Run `python -B` imports for every final module and remove any pre-existing
`__pycache__` directories.

- [ ] **Step 3: Run the full test**

Run:

```powershell
python -B -m unittest tests.test_hide_agent_consolidation -v
```

Expected: 2 tests pass.

- [ ] **Step 4: Run a fog-of-war arena smoke match**

Run:

```powershell
python -B -X utf8 src/arena.py --seek LAB2 --hide LAB2 \
  --submissions-dir submissions --max-steps 50 --no-viz --delay 0 \
  --pacman-obs-radius 5 --ghost-obs-radius 5 \
  --pacman-speed 2 --capture-distance 2
```

Expected: match completes without an agent import, action, or runtime error.

- [ ] **Step 5: Verify final package**

Expected runtime files:

```text
agent.py
hide_agent/__init__.py
hide_agent/spatial.py
hide_agent/belief.py
hide_agent/concealment.py
hide_agent/relocation.py
hide_agent/evasion.py
hide_agent/controller.py
hide_agent/diagnostics.py
```

No debug directory or Python cache may exist.
