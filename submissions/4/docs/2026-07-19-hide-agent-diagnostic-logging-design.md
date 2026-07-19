# Hide Agent Diagnostic Logging Design

## Goal

Make the Hide agent's file log concise enough to scan turn by turn while preserving the evidence needed to diagnose incorrect move selection, panic behavior, dead-end handling, time-budget degradation, and unexpected exceptions.

This change is limited to the Hide agent in `submissions/4`, its debug utility, and tests dedicated to Hide-agent logging. It must not change move selection or touch the Seek agent.

## Log format

Each turn is one structured text block with consistent indentation and field names:

```text
[TURN 7] mode=CONTROL
  state: ghost=(4,15) pacman=(6,11) previous=(4,14)
  context: safe_depth=5 dead_end=none
  candidates:
    UP    -> (3,15) capture=2 maze=5 safe=2 utility=30006 final=30546 flags=away
    DOWN  -> (5,15) capture=2 maze=5 safe=2 utility=30039 final=31779 flags=blocked-approach
  decision: UP -> (3,15)
  reason: lowest-ranked viable candidate
  runtime: 34 ms
```

Panic turns use the same outer structure, with candidate maze distance instead of control-only scoring fields:

```text
[TURN 8] mode=PANIC
  state: ghost=(3,15) pacman=(6,13) previous=(4,15)
  candidates:
    UP    -> (2,15) distance=6
    DOWN  -> (4,15) distance=4
  decision: UP -> (2,15)
  reason: greatest maze distance
  runtime: 5 ms
```

Move names are rendered as `UP`, `DOWN`, `LEFT`, `RIGHT`, or `STAY`, without Python enum representations. Durations are integer milliseconds. Positions use compact tuple notation.

## Information retained

Control-mode logs retain the decision inputs that are useful when comparing candidates:

- Candidate move and destination.
- Predicted best Pacman response and resulting position.
- Capture-turn horizon, maze distance, safe area, topology score, raw utility, and final score.
- Active decision flags only, including away, approach, forced approach, blocked approach, urgent exit, dead-end entry, avoidable dead-end entry, deeper retreat, and dead-end delay.
- Current dead-end mouth, depth, slack, and whether exit is mandatory when applicable.
- Selected move, destination, selection reason, and runtime.

Panic-mode logs retain every legal move, destination, maze distance from Pacman, the selected move, and the reason for selection.

One initialization summary retains dead-end cell and branch counts. The full cell list is omitted because the separately generated topology map already contains map-level diagnostic detail.

If the controller lowers safe-area depth because the time budget is nearly exhausted, the turn context records the selected depth and a `time-budget-reduced` flag.

## Information removed

The logger will no longer emit:

- False-valued or `None` candidate flags.
- Cache sizes.
- The complete dead-end cell list.
- Safe-area proxy source arrays.
- Repeated global booleans on every candidate row.
- A separate `[STEP]` line after the turn decision.
- Embedded extra newlines or inconsistent labels and capitalization.

Values needed to understand ranking remain on the applicable candidate row. Removing a field from the log must not remove it from the decision algorithm.

## Components and data flow

The debug utility will provide small formatting helpers for moves, positions, key-value fields, and turn blocks. It remains responsible for writing `debug/log.txt` and clearing the file at agent initialization.

`GhostAgent.step` supplies turn-level context: step number, mode, starting positions, previous position, chosen move, and elapsed runtime. It opens the logical turn record before delegation and completes it after a move is returned.

`hide_agent.control` supplies the control candidate summaries, active flags, dead-end context, decision result, and decision reason. `hide_agent.panic` supplies panic candidate summaries and its distance-based reason.

Logging state is diagnostic only. No logging result is used by the decision algorithm, and logging failures continue to be non-fatal.

## Error handling

Unexpected exceptions crossing `GhostAgent.step` are appended as an `[ERROR]` section containing:

- Turn number and mode if known.
- Exception type and message.
- Full Python traceback.

After logging, the original exception is re-raised unchanged so arena behavior and error handling remain the same. Failure to write the diagnostic log must never replace or hide the original agent exception.

## Testing

Tests will use a temporary log destination or isolated writer so the tracked sample log is not modified during test execution. They will verify:

- A CONTROL turn has one header, compact candidate rows, one decision, one reason, and one runtime.
- A PANIC turn follows the same structure and reports candidate distances.
- Move enums and positions are normalized consistently.
- Only active flags are printed; false and `None` fields are absent.
- Cache sizes, safe-area source arrays, full dead-end cell lists, and redundant STEP records are absent.
- A reduced time budget produces the diagnostic flag.
- Exceptions include type, message, and traceback and are re-raised unchanged.
- For representative control and panic scenarios, move selection is identical before and after the logging refactor.

## Acceptance criteria

The change is complete when representative game turns produce the approved structured format, diagnostic tests pass, existing Hide-agent behavior tests pass, and no Seek-agent file has been modified.
