"""
agent.py - V4 Aggressive
========================

Pacman:
- Biết vị trí G mỗi lượt nên lập tức truy đuổi và chặn, không chờ học xong.
- Ưu tiên speed=2, giảm số đường thoát và forced-capture pressure.
- Beam prediction top-2, tối đa 3 lượt.
- Khi khoảng cách không giảm, chuyển CUT-OFF để chiếm giao lộ phía trước.

Ghost:
- SAFE_EVADE / EXPAND / MAX_DELAY.
- Safe horizon làm lớp cứng, rollout ngắn làm tie-breaker.
- Phạt tiếp tục chạy vào choke point mà P có thể chiếm trước.
"""

from collections import deque
import importlib.util
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_interface import PacmanAgent as BasePacmanAgent  # noqa: E402
from agent_interface import GhostAgent as BaseGhostAgent    # noqa: E402
from environment import Move                                # noqa: E402

# ---------------------------------------------------------------------------
# Nạp module cục bộ để không bị sys.modules của submission khác ghi đè.
# ---------------------------------------------------------------------------
_SUBMISSION_DIR = Path(__file__).resolve().parent
_LOCAL_TAG = "".join(ch if ch.isalnum() else "_" for ch in str(_SUBMISSION_DIR))


def _load_local_module(logical_name: str, filename: str):
    module_key = f"_hide_seek_{_LOCAL_TAG}_{logical_name}"
    cached = sys.modules.get(module_key)
    if cached is not None:
        return cached

    module_path = _SUBMISSION_DIR / filename
    spec = importlib.util.spec_from_file_location(module_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_key, None)
        raise
    return module


_local_search_utils = _load_local_module("search_utils", "search_utils.py")
MazeGraph = _local_search_utils.MazeGraph
action_steps = _local_search_utils.action_steps
direction_between = _local_search_utils.direction_between
pacman_moves = _local_search_utils.pacman_moves

_previous_search_utils = sys.modules.get("search_utils")
sys.modules["search_utils"] = _local_search_utils
try:
    adversarial = _load_local_module("adversarial", "adversarial.py")
finally:
    if _previous_search_utils is None:
        sys.modules.pop("search_utils", None)
    else:
        sys.modules["search_utils"] = _previous_search_utils


TIME_BUDGET = min(0.90, max(0.15, float(os.environ.get("AGENT_TIME_BUDGET", "0.68"))))
ASSUMED_PACMAN_SPEED = max(1, int(os.environ.get("ASSUMED_PACMAN_SPEED", "2")))
DEFAULT_PACMAN_SPEED = max(1, int(os.environ.get("PACMAN_SPEED", str(ASSUMED_PACMAN_SPEED))))

OBSERVATION_COUNT = 3
PREDICTION_HORIZON = 3
BEAM_WIDTH = 2
MAX_PREDICTION_MISSES = 2
CLOSE_DISTANCE = 5

# Lab 2: enemy_position có thể là None do fog-of-war, không chỉ vì chưa từng
# thấy. Nếu không thấy Ghost quá STALE_THRESHOLD bước, vị trí last_seen_enemy
# coi như không còn đáng tin -> chuyển sang chế độ dò tìm (explore) thay vì
# tiếp tục đuổi theo một điểm có thể đã sai từ lâu.
STALE_THRESHOLD = 10

ENTER_DELAY = 0.20
EXIT_DELAY = 0.05
ENTER_EXPAND = -0.20
EXIT_EXPAND = -0.05


def _explore_toward_frontier(graph, pos, speed=1):
    """
    Lab 2: khi chưa/không còn tin vào vị trí đối thủ, chủ động di chuyển về
    phía 'biên khám phá' (frontier) gần nhất để mở rộng tầm nhìn thay vì
    đứng yên hoặc đuổi theo một điểm đã lỗi thời. Nếu bản đồ đã khám phá hết
    (không còn frontier), tuần tra tới ô đã biết xa nhất để tránh đứng yên
    một chỗ (dễ bị phục kích / bỏ lỡ cơ hội chạm mặt đối thủ).
    Trả về action đúng định dạng framework (Move hoặc (Move, steps)).
    """
    target = graph.nearest_frontier(pos)
    if target is None:
        target = graph.farthest_known_cell(pos)
    if target is None or target == pos:
        return Move.STAY

    best_action, best_d = Move.STAY, graph.dist(pos, target)
    if best_d < 0:
        return Move.STAY
    for action, npos in pacman_moves(graph, pos, max(1, int(speed))):
        d = graph.dist(npos, target)
        if 0 <= d < best_d:
            best_d, best_action = d, action
    return best_action


class _GraphMixin:
    """
    Lab 2: map_state mỗi bước chỉ là quan sát CỤC BỘ (nhiều ô -1). Bản đồ
    thật tĩnh suốt trận, nên ta build graph đúng MỘT lần (lần gọi đầu tiên
    của một agent instance = đầu trận, vì Arena tạo agent mới mỗi trận) rồi
    chỉ update() để hợp nhất quan sát mới — không bao giờ rebuild-from-
    scratch, tránh "quên" toàn bộ bản đồ đã khám phá chỉ vì nó tạm thời nằm
    ngoài tầm nhìn ở bước hiện tại.
    """

    def _get_graph(self, map_state):
        if getattr(self, "_graph", None) is None:
            self._graph = MazeGraph(map_state)
            self._on_new_graph()
        else:
            self._graph.update(map_state)
        return self._graph

    def _on_new_graph(self):
        return None


class PacmanAgent(BasePacmanAgent, _GraphMixin):
    """Seeker quyết liệt: tận dụng speed=2 để áp sát và chặn đường."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Aggressive Intercept Pacman V4"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", DEFAULT_PACMAN_SPEED)))
        self._graph = None
        self._map_signature = None
        self._reset_match_state()

    def _reset_match_state(self):
        self.last_seen_enemy = None
        self.last_seen_step = None
        self.prev_pac = None
        self.prev_ghost = None
        self.prev_ghost_direction = None
        self.distance_history = deque(maxlen=3)

        self.ghost_model = adversarial.GHOST_UNKNOWN
        self.distance_hits = 0
        self.space_hits = 0
        self.transition_observations = 0
        self.prediction_misses = 0

    def _on_new_graph(self):
        self._reset_match_state()

    def _reset_learning(self):
        self.ghost_model = adversarial.GHOST_UNKNOWN
        self.distance_hits = 0
        self.space_hits = 0
        self.transition_observations = 0
        self.prediction_misses = 0

    def _observe_ghost(self, graph, current_pac, current_ghost):
        if self.prev_pac is None or self.prev_ghost is None:
            return

        actual = tuple(current_ghost)
        if self.ghost_model != adversarial.GHOST_UNKNOWN:
            predicted = adversarial.ghost_model_best_positions(
                graph, self.prev_pac, self.prev_ghost, self.ghost_model
            )
            if actual in predicted:
                self.prediction_misses = 0
            else:
                self.prediction_misses += 1
                if self.prediction_misses >= MAX_PREDICTION_MISSES:
                    self._reset_learning()
            return

        legal = {pos for _, pos in adversarial.ghost_moves(graph, self.prev_ghost)}
        if actual not in legal or len(legal) < 2:
            return

        distance_set, space_set = adversarial.prediction_sets_for_observation(
            graph, self.prev_pac, self.prev_ghost
        )
        self.transition_observations += 1
        self.distance_hits += int(actual in distance_set)
        self.space_hits += int(actual in space_set)

        if self.transition_observations >= OBSERVATION_COUNT:
            if self.distance_hits > self.space_hits:
                self.ghost_model = adversarial.GHOST_DISTANCE
            elif self.space_hits > self.distance_hits:
                self.ghost_model = adversarial.GHOST_SPACE
            else:
                # Không ép phân loại; beam UNKNOWN vẫn hoạt động ngay.
                self.ghost_model = adversarial.GHOST_UNKNOWN
                self.transition_observations = 0
                self.distance_hits = 0
                self.space_hits = 0

    def _tail_chase_detected(self, ghost_direction):
        if len(self.distance_history) < 3 or ghost_direction is None:
            return False
        d0, d1, d2 = self.distance_history
        not_closing = d2 >= d1 and d1 >= d0 - 1
        same_direction = (
            self.prev_ghost_direction is not None
            and ghost_direction == self.prev_ghost_direction
        )
        return not_closing or same_direction

    @staticmethod
    def _prefer_full_speed(base_action, base_metrics, all_metrics, speed):
        """Đẩy P đi đủ speed khi pressure gần tương đương, tránh bước 1 chậm chạp."""
        if speed <= 1 or action_steps(base_action) == speed:
            return base_action
        full_speed = [m for m in all_metrics if m["steps"] == speed]
        if not full_speed:
            return base_action
        candidate = max(full_speed, key=lambda item: item["score"])
        if (
            candidate["capture_count"] >= base_metrics["capture_count"]
            and candidate["forced_next_count"] >= base_metrics["forced_next_count"]
            and candidate["safe_escape_count"] <= base_metrics["safe_escape_count"] + 1
            and candidate["worst_distance"] <= base_metrics["worst_distance"] + 1
            and candidate["next_pressure"] >= base_metrics["next_pressure"] - 0.15
        ):
            return candidate["action"]
        return base_action

    def _remember_state(self, pac, ghost, ghost_direction):
        self.prev_pac = tuple(pac)
        self.prev_ghost = tuple(ghost)
        self.prev_ghost_direction = ghost_direction

    def step(self, map_state, my_position, enemy_position, step_number):
        graph = self._get_graph(map_state)
        pac = tuple(my_position)

        if enemy_position is not None:
            self.last_seen_enemy = tuple(enemy_position)
            self.last_seen_step = step_number

        staleness = (
            (step_number - self.last_seen_step)
            if self.last_seen_step is not None else None
        )
        # Lab 2: chưa từng thấy Ghost, hoặc đã quá lâu không thấy -> vị trí
        # last_seen_enemy không còn đáng tin. Chủ động dò bản đồ (frontier)
        # thay vì đứng yên/đuổi theo điểm đã lỗi thời.
        if staleness is None or staleness > STALE_THRESHOLD:
            return _explore_toward_frontier(graph, pac, self.pacman_speed)

        ghost = self.last_seen_enemy

        ghost_direction = (
            direction_between(self.prev_ghost, ghost)
            if self.prev_ghost is not None
            else None
        )
        self._observe_ghost(graph, pac, ghost)

        try:
            distance = graph.dist(pac, ghost)
            self.distance_history.append(distance)

            # 1) Khi gần, dùng forced capture + Minimax ngắn, phạm vi mở tới 5.
            if 0 <= distance <= CLOSE_DISTANCE:
                action = adversarial.choose_capture_mode_action(
                    graph,
                    pac,
                    ghost,
                    self.pacman_speed,
                    time_budget=min(0.36, TIME_BUDGET * 0.55),
                )
                self._remember_state(pac, ghost, ghost_direction)
                return action

            # 2) Nền tảng quyết liệt: giảm escape set, ép bắt lượt kế.
            base_action, base_metrics, all_metrics = adversarial.choose_aggressive_pursuit(
                graph, pac, ghost, self.pacman_speed, return_details=True
            )
            chosen = self._prefer_full_speed(
                base_action, base_metrics, all_metrics, self.pacman_speed
            )

            force_cutoff = self._tail_chase_detected(ghost_direction)

            # 3) Không chờ phân loại: luôn beam-predict top-2; model chỉ điều chỉnh điểm.
            intercept_action, _target, _paths, coverage = (
                adversarial.choose_aggressive_intercept_action(
                    graph,
                    pac,
                    ghost,
                    self.pacman_speed,
                    model=self.ghost_model,
                    horizon=PREDICTION_HORIZON,
                    beam_width=BEAM_WIDTH,
                    ghost_direction=ghost_direction,
                    force_cutoff=force_cutoff,
                )
            )
            intercept_metrics = next(
                (m for m in all_metrics if m["action"] == intercept_action),
                None,
            )
            chosen_metrics = next(
                (m for m in all_metrics if m["action"] == chosen),
                base_metrics,
            )
            if adversarial.aggressive_intercept_passes(
                intercept_metrics,
                chosen_metrics,
                coverage=coverage,
                force_cutoff=force_cutoff,
            ):
                chosen = intercept_action

            self._remember_state(pac, ghost, ghost_direction)
            return chosen

        except Exception:
            self._remember_state(pac, ghost, ghost_direction)
            # Fallback vẫn tận dụng speed thay vì chỉ đi 1 ô.
            return graph.pacman_action_to_capture_zone(
                pac, ghost, self.pacman_speed, adversarial.CAPTURE_DIST
            )


class GhostAgent(BaseGhostAgent, _GraphMixin):
    """Hider: safe horizon + rollout + chống intercept."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Safe Rollout Ghost V4"
        self._graph = None
        self._map_signature = None

        real_speed = kwargs.get("pacman_speed", kwargs.get("enemy_speed"))
        try:
            real_speed = int(real_speed) if real_speed is not None else None
        except (TypeError, ValueError):
            real_speed = None
        self.enemy_speed = (
            max(real_speed, ASSUMED_PACMAN_SPEED)
            if real_speed is not None
            else ASSUMED_PACMAN_SPEED
        )
        self._reset_match_state()

    def _reset_match_state(self):
        self.last_seen_enemy = None
        self.prev_pac = None
        self.prev_ghost = None
        self.last_move = None
        self.p_quality = deque(maxlen=3)
        self.mode = adversarial.MODE_SAFE
        self.pending_mode = None
        self.pending_count = 0

    def _on_new_graph(self):
        self._reset_match_state()

    def _observe_pacman(self, graph, current_pac):
        if self.prev_pac is None or self.prev_ghost is None:
            return
        quality = adversarial.pacman_action_quality(
            graph,
            self.prev_pac,
            self.prev_ghost,
            current_pac,
            self.enemy_speed,
        )
        if quality is not None:
            self.p_quality.append(float(quality))

    def _p_strength(self):
        return 0.50 if not self.p_quality else sum(self.p_quality) / len(self.p_quality)

    def _request_mode(self, desired, immediate=False):
        if immediate:
            self.mode = desired
            self.pending_mode = None
            self.pending_count = 0
            return
        if desired == self.mode:
            self.pending_mode = None
            self.pending_count = 0
            return
        if self.pending_mode == desired:
            self.pending_count += 1
        else:
            self.pending_mode = desired
            self.pending_count = 1
        if self.pending_count >= 2:
            self.mode = desired
            self.pending_mode = None
            self.pending_count = 0

    def _update_mode(self, balance, best_horizon):
        if best_horizon <= 1:
            self._request_mode(adversarial.MODE_DELAY, immediate=True)
            return

        desired = self.mode
        if self.mode == adversarial.MODE_DELAY:
            if balance < EXIT_DELAY:
                desired = adversarial.MODE_EXPAND if balance < ENTER_EXPAND else adversarial.MODE_SAFE
        elif self.mode == adversarial.MODE_EXPAND:
            if balance > EXIT_EXPAND:
                desired = adversarial.MODE_DELAY if balance > ENTER_DELAY else adversarial.MODE_SAFE
        else:
            if balance > ENTER_DELAY:
                desired = adversarial.MODE_DELAY
            elif balance < ENTER_EXPAND:
                desired = adversarial.MODE_EXPAND
            else:
                desired = adversarial.MODE_SAFE
        self._request_mode(desired)

    def _remember_state(self, pac, ghost, move):
        self.prev_pac = tuple(pac)
        self.prev_ghost = tuple(ghost)
        self.last_move = move

    def step(self, map_state, my_position, enemy_position, step_number):
        graph = self._get_graph(map_state)
        ghost = tuple(my_position)

        if enemy_position is not None:
            self.last_seen_enemy = tuple(enemy_position)
        pac = self.last_seen_enemy

        if pac is None:
            frontier_set = set(graph.frontier_cells())
            candidates = [(Move.STAY, ghost)]
            candidates.extend((move, pos) for pos, move in graph.neighbors(ghost))
            move = max(
                candidates,
                key=lambda item: (
                    graph.cycle_potential(item[1], 5),
                    graph.local_space(item[1], 4),
                    graph.open_degree(item[1]),
                    int(item[1] in frontier_set),
                    int(item[0] != Move.STAY),
                ),
            )[0]
            self.last_move = move
            return move

        self._observe_pacman(graph, pac)

        try:
            options, summary = adversarial.analyze_ghost_options(
                graph,
                pac,
                ghost,
                self.enemy_speed,
                survival_depth=3,
                rollout_depth=4,
                time_budget=min(0.54, TIME_BUDGET * 0.80),
                last_move=self.last_move,
            )
            balance = self._p_strength() - summary["escape_strength"]
            self._update_mode(balance, summary["best_horizon"])
            move = adversarial.select_ghost_move(options, self.mode)
            self._remember_state(pac, ghost, move)
            return move

        except adversarial.SearchTimeout:
            move = adversarial.choose_ghost_one_step(
                graph, pac, ghost, self.enemy_speed, self.last_move
            )
            self._remember_state(pac, ghost, move)
            return move
        except Exception:
            move = graph.farthest_move_from(ghost, pac)
            self._remember_state(pac, ghost, move)
            return move
