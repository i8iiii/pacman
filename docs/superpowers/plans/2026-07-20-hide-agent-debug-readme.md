# Hide Agent Debug README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concise quick-start guide for enabling and using the hider diagnostics.

**Architecture:** Create one Markdown file beside `debug.py`. It documents the existing switch and generated files without changing agent behavior.

**Tech Stack:** Markdown and the existing Python debug module.

## Global Constraints

- Create `submissions/4/debug/README.md`.
- Cover only the hider diagnostic workflow, `log.txt`, and `topology_map.txt`.
- Keep the instructions concise and simple.
- Do not document contest results or testing maps.

---

### Task 1: Add the Hide Agent Debug Quick Start

**Files:**
- Create: `submissions/4/debug/README.md`
- Reference: `submissions/4/debug/debug.py:6-7`
- Reference: `submissions/4/agent.py:291-300`

**Interfaces:**
- Consumes: `DEBUG_ENABLED` from `debug.py` and the existing `log.txt` and `topology_map.txt` writers.
- Produces: A standalone Markdown quick-start guide; no runtime interface changes.

- [ ] **Step 1: Confirm the documented switch and outputs**

Run:

```bash
rg -n "DEBUG_ENABLED|LOG_PATH" submissions/4/debug/debug.py
rg -n "write_topology_score_map" submissions/4/agent.py
```

Expected: `DEBUG_ENABLED = False`, `LOG_PATH` targets `log.txt`, and topology output is generated only when debugging is enabled.

- [ ] **Step 2: Create the concise README**

Create `submissions/4/debug/README.md` with exactly this content:

````markdown
# Hide Agent Debugging

## Enable debugging

Open `debug.py` and change:

```python
DEBUG_ENABLED = False
```

to:

```python
DEBUG_ENABLED = True
```

Run the hider normally. The `debug` folder will contain:

- `log.txt` — turn state, candidate moves, decisions, errors, and runtime.
- `topology_map.txt` — topology scores calculated for the current map.

Set `DEBUG_ENABLED = False` before contests to avoid logging overhead.
````

- [ ] **Step 3: Verify scope and wording**

Run:

```bash
rg -n "DEBUG_ENABLED|log\.txt|topology_map\.txt|before contests" submissions/4/debug/README.md
rg -n "all_vs_all|maps_for_testing|csv|xlsx" submissions/4/debug/README.md
```

Expected: The first command finds the switch, both diagnostic files, and the disable reminder. The second command returns no matches.

- [ ] **Step 4: Review the documentation diff**

Run:

```bash
git diff --check
git diff -- submissions/4/debug/README.md
```

Expected: No whitespace errors and only the approved concise README content.

- [ ] **Step 5: Commit**

```bash
git add submissions/4/debug/README.md
git commit -m "docs: add hide agent debug quick start"
```
