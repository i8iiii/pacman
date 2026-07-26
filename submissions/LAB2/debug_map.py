"""
Debug tool: visualize the map with ghost hiding probability rankings.
Usage:
    cd submissions/LAB2
    python debug_map.py

Outputs:
    map_probability.txt  — ASCII map with probability rank numbers
    map_details.txt      — full table of all cells with scores
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('../../src').resolve()))
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
from environment import Environment
from agent import MapMemory, MapAnalyzer, GhostProbability, PathFinder, _MAP_CACHE, _fingerprint

# Get the default map
env = Environment()
full_map = env.map

# Run analysis
mm = MapMemory()
mm.update(full_map)

ma = MapAnalyzer()
analysis = ma.analyze(full_map)

pf = PathFinder(lambda: full_map)
gp = GhostProbability(ma, pf)

# Compute probability ranking from a neutral position (middle of map)
mid = (full_map.shape[0] // 2, full_map.shape[1] // 2)
ranked = gp.compute(mid)

# Build a lookup: cell -> rank (1 = highest probability)
rank_of = {}
for rank, cell in enumerate(ranked, 1):
    rank_of[cell] = rank

# Build probability scores with details
scored = []
dead_ends = analysis["dead_ends"]
corners = analysis["corners"]
pockets = analysis["pockets"]
mid_row = analysis["mid_row"]

pocket_of = {}
for pid, region in pockets.items():
    for cell in region:
        pocket_of[cell] = pid

for pos, _ in analysis["exit_counts"].items():
    score = 1.0
    if pos[0] < mid_row:
        score *= 3.0
    if pos in dead_ends:
        score *= 5.0
    if pos in corners:
        score *= 3.0
    if pos in pocket_of:
        score *= 2.0
    dist = abs(pos[0] - mid[0]) + abs(pos[1] - mid[1])
    score *= 1.0 + 1.0 / max(1, dist)
    scored.append((pos, score))

scored.sort(key=lambda x: x[1], reverse=True)

# ---- Write ASCII map with ranks ----
h, w = full_map.shape
with open('map_probability.txt', 'w') as f:
    f.write("Ghost Hiding Probability Map\n")
    f.write("=" * 60 + "\n")
    f.write("Legend:\n")
    f.write("  # = wall\n")
    f.write("  . = empty (low priority)\n")
    f.write("  D = dead end (top hiding spot)\n")
    f.write("  C = corner\n")
    f.write("  p = pocket\n")
    f.write("  U = upper half (no dead-end/corner/pocket)\n")
    f.write("  numbers 1-9 = top 9 ranked cells\n")
    f.write("=" * 60 + "\n\n")

    for r in range(h):
        line = ""
        for c in range(w):
            if full_map[r, c] == 1:
                line += "#"
            elif (r, c) in rank_of:
                rank = rank_of[(r, c)]
                if rank <= 9:
                    line += str(rank)
                elif (r, c) in dead_ends:
                    line += "D"
                elif (r, c) in corners:
                    line += "C"
                elif (r, c) in pocket_of:
                    line += "p"
                elif r < mid_row:
                    line += "U"
                else:
                    line += "."
            else:
                line += "."
        line += f"  {r}"
        f.write(line + "\n")

    f.write("\n")
    f.write("Top 20 hiding spots (rank, position, dead-end?, corner?, pocket?, upper-half?, score):\n")
    f.write("-" * 70 + "\n")
    for i, (pos, score) in enumerate(scored[:20], 1):
        is_de = "DE" if pos in dead_ends else "  "
        is_co = "CO" if pos in corners else "  "
        is_po = "PO" if pos in pocket_of else "  "
        is_up = "UP" if pos[0] < mid_row else "  "
        f.write(f"  {i:3d}. ({pos[0]:2d},{pos[1]:2d})  {is_de} {is_co} {is_po} {is_up}  score={score:.1f}\n")

print("Written: map_probability.txt")

# ---- Write detailed ranking file ----
with open('map_details.txt', 'w') as f:
    f.write("Full Cell Ranking by Ghost Hiding Probability\n")
    f.write("=" * 70 + "\n")
    f.write(f"{'Rank':>5}  {'Pos':>8}  {'Score':>8}  {'Exits':>5}  {'DeadEnd?':>8}  {'Corner?':>8}  {'Pocket?':>8}  {'Upper?':>6}\n")
    f.write("-" * 70 + "\n")

    exit_counts = analysis["exit_counts"]
    for rank, (pos, score) in enumerate(scored, 1):
        exits = exit_counts.get(pos, 0)
        is_de = "YES" if pos in dead_ends else ""
        is_co = "YES" if pos in corners else ""
        is_po = "YES" if pos in pocket_of else ""
        is_up = "YES" if pos[0] < mid_row else ""
        f.write(f"{rank:5d}  ({pos[0]:2d},{pos[1]:2d})  {score:8.1f}  {exits:5d}  {is_de:>8}  {is_co:>8}  {is_po:>8}  {is_up:>6}\n")

    f.write("\n")
    f.write(f"Total cells ranked: {len(scored)}\n")
    f.write(f"Dead ends: {len(dead_ends)}\n")
    f.write(f"Corners: {len(corners)}\n")
    f.write(f"Pocket regions: {len(pockets)}\n")
    f.write(f"Upper half boundary (mid_row): {mid_row}\n")

print("Written: map_details.txt")

# ---- Verify cache fingerprint ----
fp = _fingerprint(full_map)
print(f"\nMap fingerprint: {fp}")
print(f"Map shape: {h}x{w}")
print(f"Known cells: {(full_map == 0).sum()}")
print(f"Wall cells: {(full_map == 1).sum()}")
