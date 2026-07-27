# Hide-Agent Module Consolidation Design

## Goal

Reduce the Hide-agent package from thirteen functional source modules to seven
without changing movement, scoring, timing, state transitions, diagnostics
defaults, or the public `GhostAgent` and `HideController` interfaces.

## Final Structure

```text
submissions/LAB2/
├── agent.py
└── hide_agent/
    ├── __init__.py
    ├── spatial.py
    ├── belief.py
    ├── concealment.py
    ├── relocation.py
    ├── evasion.py
    ├── controller.py
    └── diagnostics.py
```

## Merge Map

- `spatial.py`: `geometry.py`, `navigation.py`, and `cross_map.py`
- `concealment.py`: `hideout.py` and `topology.py`
- `relocation.py`: `roads.py` and `migration.py`
- `evasion.py`: `escape.py` and `mobile_escape.py`
- `belief.py`: existing `belief.py` and `pursuit.py`
- `controller.py`: remains the orchestration boundary
- `diagnostics.py`: remains optional and disabled by default

The superseded modules are deleted only after all imports point to the merged
modules.

## Dependency Direction

```text
agent.py
  -> controller.py
       -> spatial.py
       -> belief.py
       -> concealment.py -> belief.py
       -> relocation.py -> concealment.py
       -> evasion.py
       -> diagnostics.py
```

Dependencies must remain acyclic. Merged modules use direct local definitions
instead of importing their superseded source modules.

## Compatibility

- Preserve every externally consumed class, function name, signature, and
  return shape.
- Preserve `GhostAgent.step(map_state, my_position, enemy_position,
  step_number)`.
- Preserve `HideController.step(...)`.
- Do not change strategic constants, rankings, random-choice behavior, state
  transition rules, or diagnostics payloads.
- Keep diagnostics disabled by default.

Private helper collisions introduced by a merge are resolved by retaining one
helper only when implementations are identical. Otherwise, helpers receive
domain-specific private names and their original call sites are updated.

## Verification

Tests live outside the submission directory and cover:

- imports and representative APIs for every merged module;
- route and geometry behavior;
- hideout and campsite scanning;
- road-cycle and migration behavior;
- visible and mobile evasion;
- belief and pursuit state;
- `GhostAgent` loading and a valid Hide action.

The final gate also checks:

- all tests pass;
- all Python files compile;
- no imports reference superseded modules;
- the submission contains only the runtime entry point and eight package
  files;
- diagnostics remain disabled and create no debug output;
- a fog-of-war arena smoke match completes without an agent error.
