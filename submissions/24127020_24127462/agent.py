import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent

from hide_agent import (
    Move,
    np,
    _apply_move,
    _is_valid_position,
    _topology_score,
    _local_space_score,
    _bfs_distances,
    _bfs_path,
    _get_neighbors,
    _is_straight_corridor,
    _translate_move,
    THREAT_TRIGGER,
    W_DIST,
    W_RATIO,
)

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A-Train"

        self._map_built = False
        self.last_known_enemy_pos = None
        self.safety_map: dict[tuple, float] = {}

    def step(self, map_state, my_position, enemy_position, step_number) -> Move:
        if not self._map_built:
            self._build_safety_map(map_state)
            # debugging
            self._dump_safety_map(map_state, Path(__file__).parent / "hide_agent" / "safety_map.txt")  
            self._map_built = True

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat = (
            enemy_position if enemy_position is not None else self.last_known_enemy_pos
        )

        # LAB1: 
        return self._best_move(map_state, my_position, threat)

    def _best_move(
        self, map_state: np.ndarray, my_pos: tuple, threat_pos: tuple
    ) -> Move:

        # TODO: Blind Version
        if my_pos is None or threat_pos is None:
            return Move.STAY

        my_dist = _bfs_distances(my_pos, map_state)
        seeker_dist = _bfs_distances(threat_pos, map_state)

        seeker_steps_to_me = seeker_dist.get(my_pos, 3636) / 2
        seeker_is_close = seeker_steps_to_me <=  THREAT_TRIGGER

        best_score = float("-inf")
        best_dest = None

        for tile in self.safety_map:
            if tile == my_pos:
                continue

            my_d = my_dist.get(tile)
            seek_d = seeker_dist.get(tile)

            if my_d is None or seek_d is None:
                continue

            # Seeker moves 2 cells
            seek_d = (seek_d + 1) // 2

            # Seeker can reach this tile first
            if not seeker_is_close and my_d >= seek_d:
                continue

            neighbors = _get_neighbors(tile, map_state)

            is_corner = len(neighbors) == 2 and not _is_straight_corridor(
                tile, neighbors
            )

            ratio = my_d / max(seek_d, 1)

            score = self.safety_map[tile] + W_DIST * seek_d - W_RATIO * ratio

            if seeker_is_close and is_corner:
                score += 30

            dx = tile[0] - my_pos[0]
            dy = tile[1] - my_pos[1]
            sx = threat_pos[0] - my_pos[0]
            sy = threat_pos[1] - my_pos[1]
            if dx * sx + dy * sy > 0:
                score -= 10    

            if score > best_score:
                best_score = score
                best_dest = tile

        # Fallback
        if best_dest is None:
            neighbors = _get_neighbors(my_pos, map_state)

            if not neighbors:
                return Move.STAY

            best_dest = max(neighbors, key=lambda pos: seeker_dist.get(pos, 0))

        path = _bfs_path(my_pos, best_dest, map_state)

        if len(path) < 2:
            return Move.STAY

        return _translate_move(my_pos, path[1])

    def _build_safety_map(self, map_state: np.ndarray) -> None:
        self.safety_map = {}

        for row in range(map_state.shape[0]):
            for col in range(map_state.shape[1]):
                pos = (row, col)

                if not _is_valid_position(pos, map_state):
                    continue

                topo = _topology_score(pos, map_state)
                space = _local_space_score(pos, map_state)

                self.safety_map[pos] = topo + space

    def _dump_safety_map(
        self, map_state: np.ndarray, filename=Path(__file__).parent / "safety_map.txt"
    ) -> None:
        with open(filename, "w") as f:
            for row in range(map_state.shape[0]):
                line = []
                for col in range(map_state.shape[1]):
                    pos = (row, col)
                    if not _is_valid_position(pos, map_state):
                        line.append("####")
                    else:
                        line.append(f"{ self.safety_map[pos]:4.0f}")
                f.write(" ".join(line) + "\n")
