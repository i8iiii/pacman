from pathlib import Path
from collections import deque

from . import core


TOPOLOGY_WEIGHTS = {
   "dead_end": -1200,
   "corridor": 100,
   "junction_3": 700,
   "junction_4": 900,

   "local_area": 25,

   "at_junction": 500,
   "junction_distance": -120,

   "long_corridor": -250,
   "far_from_junction": -1000,
}

from pathlib import Path

def get_map_path() -> Path:
   submission_dir = Path(__file__).resolve().parent.parent
   return submission_dir / "debug" / "map.txt"


def valid_neighbors(pos: tuple, map_state) -> list[tuple]:
   neighbors = []

   for move in core.PACMAN_MOVES:
      new_pos = core.next_position(pos, move)

      if core.is_valid_position(new_pos, map_state):
         neighbors.append((move, new_pos))

   return neighbors


def exit_count(pos: tuple, map_state) -> int:
   return len(valid_neighbors(pos, map_state))


def classify_cell(exits: int) -> str:
   if exits <= 1:
      return "DEAD_END"

   if exits == 2:
      return "CORRIDOR"

   return "JUNCTION"


def junction_distance(pos: tuple, map_state, max_depth: int = 8) -> int:
   if not core.is_valid_position(pos, map_state):
      return max_depth + 1

   if exit_count(pos, map_state) >= 3:
      return 0

   queue = deque([(pos, 0)])
   visited = {pos}

   while queue:
      current, depth = queue.popleft()

      if exit_count(current, map_state) >= 3:
         return depth

      if depth >= max_depth:
         continue

      for _, new_pos in valid_neighbors(current, map_state):
         if new_pos not in visited:
               visited.add(new_pos)
               queue.append((new_pos, depth + 1))

   return max_depth + 1


def local_area_size(pos: tuple, map_state, limit: int = 6) -> int:
   if not core.is_valid_position(pos, map_state):
      return 0

   queue = deque([(pos, 0)])
   visited = {pos}

   while queue:
      current, depth = queue.popleft()

      if depth >= limit:
         continue

      for _, new_pos in valid_neighbors(current, map_state):
         if new_pos not in visited:
               visited.add(new_pos)
               queue.append((new_pos, depth + 1))

   return len(visited)


def score_topology_cell(features: dict) -> tuple[int, dict]:
   exits = features["exits"]
   cell_type = features["type"]
   jdist = features["junction_distance"]
   area = features["local_area"]

   parts = {}

   if cell_type == "DEAD_END":
      parts["type"] = TOPOLOGY_WEIGHTS["dead_end"]

   elif cell_type == "CORRIDOR":
      parts["type"] = TOPOLOGY_WEIGHTS["corridor"]

   else:
      if exits >= 4:
         parts["type"] = TOPOLOGY_WEIGHTS["junction_4"]
      else:
         parts["type"] = TOPOLOGY_WEIGHTS["junction_3"]

   parts["local_area"] = min(area, 40) * TOPOLOGY_WEIGHTS["local_area"]

   if jdist == 0:
      parts["junction_distance"] = TOPOLOGY_WEIGHTS["at_junction"]
   else:
      parts["junction_distance"] = min(jdist, 8) * TOPOLOGY_WEIGHTS["junction_distance"]

   if cell_type == "CORRIDOR" and jdist >= 4:
      parts["long_corridor"] = (jdist - 3) * TOPOLOGY_WEIGHTS["long_corridor"]
   else:
      parts["long_corridor"] = 0

   if jdist >= 8:
      parts["far_from_junction"] = TOPOLOGY_WEIGHTS["far_from_junction"]
   else:
      parts["far_from_junction"] = 0

   total = sum(parts.values())

   return total, parts


def build_topology_score_map(map_state) -> dict:
   topology_map = {}

   height, width = map_state.shape

   for row in range(height):
      for col in range(width):
         pos = (row, col)

         if not core.is_valid_position(pos, map_state):
               continue

         neighbors = valid_neighbors(pos, map_state)
         exits = len(neighbors)
         cell_type = classify_cell(exits)
         jdist = junction_distance(pos, map_state)
         area = local_area_size(pos, map_state)

         features = {
               "position": pos,
               "type": cell_type,
               "exits": exits,
               "neighbors": neighbors,
               "junction_distance": jdist,
               "local_area": area,
         }

         score, parts = score_topology_cell(features)

         features["score"] = score
         features["score_parts"] = parts

         topology_map[pos] = features

   return topology_map

def write_topology_score_map(
   topology_map: dict,
   map_state,
   path: Path | None = None
) -> None:
   if path is None:
      path = get_map_path()

   path.parent.mkdir(parents=True, exist_ok=True)

   lines = []

   lines.append("=== TOPOLOGY SCORE MAP ===")
   for name, value in TOPOLOGY_WEIGHTS.items():
      lines.append(f"  {name}: {value}")
   lines.append("")

   height, width = map_state.shape

   for row in range(height):
      row_scores = []

      for col in range(width):
         pos = (row, col)

         if pos not in topology_map:
               row_scores.append("#####")
         else:
               score = topology_map[pos]["score"]
               row_scores.append(f"{score:5d}")

      lines.append(" ".join(row_scores))

   with open(path, "w", encoding="utf-8") as file:
      file.write("\n".join(lines))