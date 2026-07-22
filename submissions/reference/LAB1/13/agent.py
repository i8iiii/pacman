import os
import sys
from pathlib import Path

# Cho phep import cac lop interface cua framework tu thu muc src/.
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_interface import PacmanAgent as BasePacmanAgent  # noqa: E402
from agent_interface import GhostAgent as BaseGhostAgent    # noqa: E402
from environment import Move                                # noqa: E402

from search_utils import MazeGraph                          # noqa: E402
import adversarial                                          # noqa: E402

# Ngan sach thoi gian cho moi nuoc di. Gioi han cung cua de la 1.0s; chua bien
# an toan cho Colab cham + chi phi ngoai (validate, visualize).
# Co the ghi de bang bien moi truong AGENT_TIME_BUDGET khi chay benchmark.
TIME_BUDGET = float(os.environ.get("AGENT_TIME_BUDGET", "0.6"))
# Toc do Pacman ma Ghost gia dinh khi mo hinh hoa doi thu (mac dinh Arena = 2).
ASSUMED_PACMAN_SPEED = 2


class _GraphMixin:
    def _get_graph(self, map_state) -> MazeGraph:
        # Ban do tinh -> chi dung graph (kem cache BFS) mot lan.
        if getattr(self, "_graph", None) is None:
            self._graph = MazeGraph(map_state)
        return self._graph


class PacmanAgent(BasePacmanAgent, _GraphMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Arena truyen pacman_speed vao day; mac dinh 1 neu chay che do thuong.
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._graph = None
        self.last_seen_enemy = None  # phuc vu truong hop fog (enemy=None)

    def step(self, map_state, my_position, enemy_position, step_number):
        graph = self._get_graph(map_state)
        my_position = tuple(my_position)

        if enemy_position is not None:
            self.last_seen_enemy = tuple(enemy_position)

        target = self.last_seen_enemy
        if target is None:
            # Chua tung thay Ghost (chi xay ra khi fog): dung yen cho thong tin.
            return (Move.STAY, 1)

        try:
            action = adversarial.choose_pacman_action(
                graph, my_position, target, self.pacman_speed,
                time_budget=TIME_BUDGET,
            )
        except Exception:
            # An toan tuyet doi: khong bao gio de loi/timeout lam thua xu ep.
            action = graph.next_move_towards(my_position, target)

        if action is None:
            action = (Move.STAY, 1)
        return action


class GhostAgent(BaseGhostAgent, _GraphMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._graph = None
        self.last_seen_enemy = None

    def step(self, map_state, my_position, enemy_position, step_number):
        graph = self._get_graph(map_state)
        my_position = tuple(my_position)

        if enemy_position is not None:
            self.last_seen_enemy = tuple(enemy_position)

        threat = self.last_seen_enemy
        if threat is None:
            # Chua thay Pacman (fog): di toi o xa nhat theo heuristic de tan ra.
            return graph.farthest_move_from(
                my_position, (graph.h // 2, graph.w // 2)
            )

        try:
            move = adversarial.choose_ghost_move(
                graph, threat, my_position, ASSUMED_PACMAN_SPEED,
                time_budget=TIME_BUDGET,
            )
        except Exception:
            move = graph.farthest_move_from(my_position, threat)

        if move is None:
            move = Move.STAY
        return move
