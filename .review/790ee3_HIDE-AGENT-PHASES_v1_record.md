# 批阅记录

- **源文件**：HIDE-AGENT-PHASES.md
- **源文件路径**：d:/FIT-HCMUS-K24/Y2/S2/Nhập Môn AI/LAB/pacman/submissions/LAB2/HIDE-AGENT-PHASES.md
- **源文件版本**：未知
- **批阅时间**：20260723_1207
- **批阅版本**：v1
- **批注数量**：0
  - 评论：0
  - 删除：0
  - 后插：0
  - 前插：0

---

## 操作指令

> 指令已按**从后往前**排列（倒序），请严格按照顺序从上到下逐条执行。
> 每条指令提供了「文本锚点」用于精确定位，请优先通过锚点文本匹配来确认目标位置，blockIndex 仅作辅助参考。

---

## 原始数据（JSON）

> 如需精确操作，可使用以下 JSON 数据。其中 `blockIndex` 是基于空行分割的块索引（从0开始），`startOffset` 是目标文本在块内的字符偏移量（从0开始），可用于区分同一块内的重复文本。

```json
{
  "fileName": "HIDE-AGENT-PHASES.md",
  "docVersion": "未知",
  "reviewVersion": 1,
  "annotationCount": 0,
  "rawMarkdown": "# Hide Agent: Implementation Phases\n\nStatus: **planning only — no implementation phase is active**.\n\n## Control rules\n\n- Only the user can start a phase by explicitly naming its ID, for example: `START P00`.\n- Only that phase may be implemented. Finishing it does not start the next phase.\n- Every phase must be tested and must produce its listed events in `debug/hide-agent.log` before it can be called complete.\n- After each phase, report the changed files, test result, and a short log excerpt, then stop and wait.\n- Do not commit unless the user explicitly asks for a commit.\n- If a chosen phase depends on unfinished work, explain the dependency and wait; do not implement it automatically.\n- Map knowledge and pursuit state exist only for the current game. The agent never reads previous logs or carries map knowledge into another game.\n\n## Phase board\n\n| ID | Status | Purpose | Completion check | Required log events |\n|---|---|---|---|---|\n| `P00` | Complete | Add a legal baseline Hide action and safe logging. | Every call returns a legal move; logging failure cannot crash the agent. | `match_start`, `decision` |\n| `P01` | Not started | Build match-local map memory from cells actually observed by Hide. | New observations merge correctly; a new game starts with empty memory; arbitrary map dimensions work. | `memory_reset`, `map_updated` |\n| `P02` | Not started | Implement exact movement, line-of-sight, and capture geometry for Hide and Pacman. | Tests cover walls, occlusion, diagonal blindness, Pacman's straight one/two-cell moves, and capture distance. | `geometry_summary` |\n| `P03` | Not started | Detect and rank campsite candidates from known topology. | Dead ends and easily guarded junctions are rejected; useful junctions with multiple escape directions are ranked. | `campsite_scan`, `campsite_selected` |\n| `P04` | Not started | Route through known cells while scouting for a campsite. | Hide follows legal known routes, explores a frontier when needed, and replans when new observations invalidate a route. | `scout_target`, `route_planned`, `route_replanned` |\n| `P05` | Not started | Add the no-sight controller: scout until a campsite is reached, then stay. | State changes between `SCOUT` and `CAMP` correctly; Hide does not camp at an unsafe or merely unknown cell. | `state_changed`, `scout_move`, `camp_hold` |\n| `P06` | Not started | Escape when Pacman is visible and Hide is already at a campsite. | Hide rejects immediately capturable moves, chooses an out-of-sight branch, and randomizes only between genuinely equivalent safe branches. | `visible_at_camp`, `escape_branch_chosen` |\n| `P07` | Not started | Escape when Pacman is visible and Hide is not at a campsite. | Hide first maximizes immediate safety and separation, then moves toward a reachable campsite or useful junction without entering a bad dead end. | `visible_while_mobile`, `escape_target_chosen` |\n| `P08` | Not started | Add the first `HOT_UNSEEN` pursuit model: assume Pacman follows from Hide's last departure direction. | After breaking sight, Hide keeps moving through another occlusion instead of stopping at the first corner; impossible follower predictions are discarded. | `hot_unseen_entered`, `follower_updated`, `hot_move`, `follower_invalidated` |\n| `P09` | Not started | Strengthen `HOT_UNSEEN` against intercepting or searching Pacmen. | Hide considers junction/campsite interception; if simple predictions fail, it keeps a reachable set of possible Pacman cells. `HOT_UNSEEN` ends only on a new sighting or arrival at a campsite safe from every plausible next-turn capture. | `interceptor_updated`, `belief_rebuilt`, `hot_unseen_exited` |\n| `P10` | Not started | Add emergency behavior and enforce the per-move time limit. | No-campsite, no-route, all-dangerous, exception, and deadline cases still return a legal move within budget. | `emergency`, `deadline_fallback`, `decision_error` |\n| `P11` | Not started | Validate and tune the completed strategy without adding unapproved features. | Seeded games on varied unknown maps against follower, interceptor, sweeper, belief-based, random, and A/B/C reference seekers report survival, wins, invalid moves, and timing. | `match_end`, `benchmark_summary` |\n\n## Dependency order\n\n`P00 → P01 → P02 → P03 → P04 → P05 → P06/P07 → P08 → P09 → P10 → P11`\n\n`P06` and `P07` are separate so each visible-escape situation can be tested and accepted independently.\n\n## Current gate\n\n`P00` is complete. No further code work is authorized. Wait for the user to\nselect another phase; do not begin `P01` automatically.\n",
  "annotations": []
}
```