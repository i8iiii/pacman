"""
adversarial.py
================
V4 Aggressive cho Hide-and-Seek Arena.

Pacman:
- Tận dụng speed=2 ngay từ đầu, không chờ đủ dữ liệu mới chặn.
- AGGRESSIVE PURSUIT: giảm số đường thoát, tăng capture coverage và ưu tiên
  action hai ô khi không làm mất cơ hội bắt.
- BEAM INTERCEPT: giữ hai đường chạy Ghost có khả năng nhất, nhìn tối đa 3 lượt.
- CUT-OFF: khi bị kéo theo đuôi, chiếm giao lộ/góc phía trước Ghost.
- CAPTURE: từ khoảng cách gần dùng tìm kiếm sâu ngắn nhưng không bị bộ lọc
  quá thận trọng giữ lại.

Ghost:
- Hard safety + safe horizon vẫn là nền tảng.
- Thêm deterministic rollout ngắn làm tie-breaker giữa các nước an toàn.
- Nhận biết nguy cơ Pacman chặn phía trước và cho phép đổi nhánh/quay đầu.
"""

import time

from environment import Move
from search_utils import (
    MazeGraph,
    action_move,
    action_steps,
    direction_between,
    ghost_moves,
    manhattan,
    opposite_move,
    pacman_moves,
)

CAPTURE_DIST = 2
INF = 10 ** 9
CAPTURE_SCORE = 10 ** 6

GHOST_DISTANCE = "DISTANCE_RUNNER"
GHOST_SPACE = "SPACE_RUNNER"
GHOST_UNKNOWN = "UNKNOWN"

MODE_EXPAND = "EXPAND"
MODE_SAFE = "SAFE_EVADE"
MODE_DELAY = "MAX_DELAY"


class SearchTimeout(Exception):
    pass


def _check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout


def is_capture(pac, ghost) -> bool:
    return manhattan(pac, ghost) < CAPTURE_DIST


def _safe_dist(graph, a, b):
    value = graph.dist(a, b)
    return value if value >= 0 else graph.h * graph.w


def _end_for_action(graph, pos, speed, action):
    for candidate, end in pacman_moves(graph, pos, speed):
        if candidate == action:
            return end
    return tuple(pos)


# ---------------------------------------------------------------------------
# Minimax ngắn cho trạng thái gần
# ---------------------------------------------------------------------------
def evaluate(graph: MazeGraph, pac, ghost, plies_used: int, speed: int = 2):
    if is_capture(pac, ghost):
        return CAPTURE_SCORE - plies_used

    d = _safe_dist(graph, pac, ghost)
    pressure = 0
    for _, npac in pacman_moves(graph, pac, speed):
        pressure += int(is_capture(npac, ghost))

    return (
        -14.0 * d
        + 24.0 * pressure
        - 1.5 * graph.local_space(ghost, 3)
        - 4.0 * graph.cycle_potential(ghost, 4)
        - 2.0 * graph.speed_denial_score(ghost, pac)
        + 4.0 * int(graph.is_corridor(ghost))
    )


def _value(graph, pac, ghost, depth, speed, alpha, beta, deadline, plies):
    _check_deadline(deadline)
    if is_capture(pac, ghost):
        return CAPTURE_SCORE - plies
    if depth <= 0:
        return evaluate(graph, pac, ghost, plies, speed)

    best = -INF
    p_candidates = pacman_moves(graph, pac, speed)
    p_candidates.sort(
        key=lambda item: (
            _safe_dist(graph, item[1], ghost),
            -action_steps(item[0]),
        )
    )

    for _, npac in p_candidates:
        _check_deadline(deadline)
        worst = INF
        g_candidates = ghost_moves(graph, ghost)
        g_candidates.sort(
            key=lambda item: (
                -_safe_dist(graph, npac, item[1]),
                -graph.cycle_potential(item[1], 4),
                -graph.local_space(item[1], 3),
            )
        )
        for _, ng in g_candidates:
            _check_deadline(deadline)
            value = (
                CAPTURE_SCORE - (plies + 1)
                if is_capture(npac, ng)
                else _value(
                    graph, npac, ng, depth - 1, speed,
                    alpha, beta, deadline, plies + 1,
                )
            )
            worst = min(worst, value)
            if worst <= alpha:
                break
        best = max(best, worst)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


def choose_pacman_action(graph, pac, ghost, speed, time_budget=0.30, max_depth=4):
    deadline = time.perf_counter() + max(0.02, time_budget)
    roots = pacman_moves(graph, pac, speed)
    roots.sort(
        key=lambda item: (
            _safe_dist(graph, item[1], ghost),
            -action_steps(item[0]),
        )
    )
    best_action = roots[0][0]

    for depth in range(1, max_depth + 1):
        try:
            current_action = None
            current_value = -INF
            alpha, beta = -INF, INF
            for action, npac in roots:
                _check_deadline(deadline)
                worst = INF
                for _, ng in ghost_moves(graph, ghost):
                    _check_deadline(deadline)
                    value = (
                        CAPTURE_SCORE
                        if is_capture(npac, ng)
                        else _value(
                            graph, npac, ng, depth - 1, speed,
                            alpha, beta, deadline, 1,
                        )
                    )
                    worst = min(worst, value)
                    if worst <= alpha:
                        break
                if worst > current_value:
                    current_value = worst
                    current_action = action
                alpha = max(alpha, current_value)
            if current_action is not None:
                best_action = current_action
            if current_value >= CAPTURE_SCORE - depth:
                break
        except SearchTimeout:
            break
    return best_action


# ---------------------------------------------------------------------------
# PACMAN: aggressive one-step + forced capture pressure
# ---------------------------------------------------------------------------
def _best_next_capture_coverage(graph, pac, ghost, speed):
    """Nước P tốt nhất ở lượt kế tiếp bắt được bao nhiêu phản ứng G."""
    responses = ghost_moves(graph, ghost)
    best_count = 0
    guaranteed = False
    for _, npac in pacman_moves(graph, pac, speed):
        count = sum(1 for _, ng in responses if is_capture(npac, ng))
        if count > best_count:
            best_count = count
        if count == len(responses):
            guaranteed = True
            break
    return best_count, len(responses), guaranteed


def pacman_action_metrics(graph, pac, ghost, speed, action_pair):
    action, npac = action_pair
    responses = ghost_moves(graph, ghost)

    capture_count = 0
    safe_escape_count = 0
    forced_next_count = 0
    next_pressure = 0.0
    distances = []
    escape_spaces = []

    for _, ng in responses:
        if is_capture(npac, ng):
            capture_count += 1
            distances.append(0)
            escape_spaces.append(0)
            next_pressure += 1.0
            continue

        safe_escape_count += 1
        distances.append(_safe_dist(graph, npac, ng))
        escape_spaces.append(graph.local_space(ng, 3))
        best_count, response_count, guaranteed_next = _best_next_capture_coverage(
            graph, npac, ng, speed
        )
        forced_next_count += int(guaranteed_next)
        if response_count:
            next_pressure += best_count / response_count

    response_count = len(responses)
    guaranteed = response_count > 0 and capture_count == response_count
    worst_distance = max(distances) if distances else 0
    average_distance = sum(distances) / len(distances) if distances else 0.0
    worst_escape = max(escape_spaces) if escape_spaces else 0
    mobility = graph.open_degree(npac)
    steps = action_steps(action)
    full_speed = int(speed > 1 and steps == speed)
    current_distance = _safe_dist(graph, pac, ghost)
    closing_gain = current_distance - worst_distance

    # Lexicographic, nhưng quyết liệt hơn V3:
    # forced-next và áp lực lượt sau đứng trước khoảng cách tuyệt đối.
    score = (
        int(guaranteed),
        capture_count,
        forced_next_count,
        round(next_pressure, 4),
        -safe_escape_count,
        -worst_distance,
        -average_distance,
        closing_gain,
        full_speed,
        steps,
        -worst_escape,
        mobility,
        int(action != Move.STAY),
    )

    return {
        "action": action,
        "end": npac,
        "guaranteed": guaranteed,
        "capture_count": capture_count,
        "response_count": response_count,
        "safe_escape_count": safe_escape_count,
        "forced_next_count": forced_next_count,
        "next_pressure": next_pressure,
        "worst_distance": worst_distance,
        "average_distance": average_distance,
        "worst_escape": worst_escape,
        "mobility": mobility,
        "steps": steps,
        "full_speed": bool(full_speed),
        "closing_gain": closing_gain,
        "score": score,
    }


def all_pacman_action_metrics(graph, pac, ghost, speed):
    return [
        pacman_action_metrics(graph, pac, ghost, speed, pair)
        for pair in pacman_moves(graph, pac, speed)
    ]


def choose_aggressive_pursuit(graph, pac, ghost, speed, return_details=False):
    metrics = all_pacman_action_metrics(graph, pac, ghost, speed)
    best = max(metrics, key=lambda item: item["score"])
    return (best["action"], best, metrics) if return_details else best["action"]


# Tên cũ để tương thích với phần đo năng lực P và các test cũ.
choose_safe_pursuit = choose_aggressive_pursuit


def _find_action_metrics(metrics, action):
    for item in metrics:
        if item["action"] == action:
            return item
    return None


# ---------------------------------------------------------------------------
# PACMAN: mô hình Ghost và beam prediction top-2
# ---------------------------------------------------------------------------
def _distance_runner_score(graph, pac, pos):
    return (
        _safe_dist(graph, pac, pos),
        graph.open_degree(pos),
        graph.local_space(pos, 2),
    )


def _space_runner_score(graph, pac, pos):
    return (
        3.0 * graph.open_degree(pos)
        + 0.45 * graph.local_space(pos, 3)
        + 1.25 * graph.cycle_potential(pos, 4)
        + 0.50 * _safe_dist(graph, pac, pos)
        - 1.50 * int(graph.is_corridor(pos))
    )


def ghost_model_best_positions(graph, pac, ghost, model):
    candidates = ghost_moves(graph, ghost)
    if model == GHOST_DISTANCE:
        values = [(_distance_runner_score(graph, pac, pos), pos) for _, pos in candidates]
    elif model == GHOST_SPACE:
        values = [(_space_runner_score(graph, pac, pos), pos) for _, pos in candidates]
    else:
        return {pos for _, pos in candidates}
    best = max(value for value, _ in values)
    return {pos for value, pos in values if value == best}


def prediction_sets_for_observation(graph, pac, ghost):
    return (
        ghost_model_best_positions(graph, pac, ghost, GHOST_DISTANCE),
        ghost_model_best_positions(graph, pac, ghost, GHOST_SPACE),
    )


def _ghost_evasion_numeric(
    graph, pac, ghost, move, ng, speed,
    model=GHOST_UNKNOWN, preferred_direction=None,
):
    p_candidates = pacman_moves(graph, pac, speed)
    capture_count = sum(1 for _, npac in p_candidates if is_capture(npac, ng))
    guaranteed_safe = capture_count == 0
    worst_distance = min(
        (_safe_dist(graph, npac, ng) for _, npac in p_candidates),
        default=0,
    )

    model_bonus = 0.0
    if model == GHOST_DISTANCE:
        model_bonus = 2.0 * _safe_dist(graph, pac, ng)
    elif model == GHOST_SPACE:
        model_bonus = 1.5 * graph.local_space(ng, 3) + 4.0 * graph.cycle_potential(ng, 4)
    else:
        model_bonus = (
            0.8 * _safe_dist(graph, pac, ng)
            + 0.7 * graph.local_space(ng, 3)
            + 2.0 * graph.cycle_potential(ng, 4)
        )

    direction = direction_between(ghost, ng)
    inertia = 4.0 if preferred_direction is not None and direction == preferred_direction else 0.0
    stay_penalty = 6.0 if move == Move.STAY else 0.0

    return (
        1000.0 * int(guaranteed_safe)
        - 140.0 * capture_count
        + 16.0 * worst_distance
        + 1.2 * graph.local_space(ng, 3)
        + 4.0 * graph.cycle_potential(ng, 4)
        + 2.5 * graph.speed_denial_score(ng, pac)
        + model_bonus
        + inertia
        - stay_penalty
    )


def predict_ghost_beam(
    graph, pac, ghost, speed,
    model=GHOST_UNKNOWN,
    horizon=3,
    beam_width=2,
    ghost_direction=None,
):
    """Giữ tối đa hai đường G mạnh nhất mỗi tầng, không cam kết vào một dự đoán."""
    states = [{
        "pac": tuple(pac),
        "ghost": tuple(ghost),
        "path": [],
        "directions": [],
        "value": 0.0,
    }]

    for _ in range(max(1, min(3, int(horizon)))):
        expanded = []
        for state in states:
            preferred = state["directions"][-1] if state["directions"] else ghost_direction
            for move, ng in ghost_moves(graph, state["ghost"]):
                value = _ghost_evasion_numeric(
                    graph,
                    state["pac"],
                    state["ghost"],
                    move,
                    ng,
                    speed,
                    model=model,
                    preferred_direction=preferred,
                )
                p_action = graph.pacman_action_to_capture_zone(
                    state["pac"], ng, speed, CAPTURE_DIST
                )
                next_p = _end_for_action(graph, state["pac"], speed, p_action)
                direction = direction_between(state["ghost"], ng)
                expanded.append({
                    "pac": next_p,
                    "ghost": ng,
                    "path": state["path"] + [ng],
                    "directions": state["directions"] + [direction],
                    "value": state["value"] + value,
                })

        # Loại trạng thái trùng vị trí cuối, giữ đường có value cao hơn.
        unique = {}
        for state in expanded:
            key = (state["pac"], state["ghost"])
            if key not in unique or state["value"] > unique[key]["value"]:
                unique[key] = state
        states = sorted(unique.values(), key=lambda item: item["value"], reverse=True)[:max(1, beam_width)]
        if not states:
            break

    return [state["path"] for state in states if state["path"]]


def _intercept_candidates(graph, ghost, paths, ghost_direction, force_cutoff):
    candidates = []
    for path in paths:
        for turn, pos in enumerate(path, start=1):
            candidates.append((pos, turn, "predicted"))

    if ghost_direction is not None:
        for pos in graph.forward_choke_points(ghost, ghost_direction, 7):
            candidates.append((pos, max(1, _safe_dist(graph, ghost, pos)), "forward"))

    if force_cutoff:
        for pos in graph.nearby_choke_points(ghost, 6)[:8]:
            candidates.append((pos, max(1, _safe_dist(graph, ghost, pos)), "cutoff"))

    return candidates


def choose_aggressive_intercept_action(
    graph, pac, ghost, speed,
    model=GHOST_UNKNOWN,
    horizon=3,
    beam_width=2,
    ghost_direction=None,
    force_cutoff=False,
):
    paths = predict_ghost_beam(
        graph, pac, ghost, speed,
        model=model,
        horizon=horizon,
        beam_width=beam_width,
        ghost_direction=ghost_direction,
    )
    if not paths:
        return None, None, [], 0

    best_target = None
    best_score = None
    best_coverage = 0

    for target, ghost_turns, kind in _intercept_candidates(
        graph, ghost, paths, ghost_direction, force_cutoff
    ):
        p_turns = graph.pacman_capture_turn_distance(
            pac, target, speed, CAPTURE_DIST
        )
        if p_turns < 0:
            continue

        coverage = 0
        for path in paths:
            if any(_safe_dist(graph, target, pos) <= 1 for pos in path):
                coverage += 1

        # Quyết liệt: cho phép đến muộn 1 lượt vì capture distance và speed=2.
        viable = p_turns <= ghost_turns + 1
        if not viable and not force_cutoff:
            continue

        kind_bonus = 2 if kind == "cutoff" else (1 if kind == "forward" else 0)
        score = (
            int(viable),
            coverage,
            kind_bonus,
            int(graph.is_junction(target)),
            -p_turns,
            -ghost_turns,
            -_safe_dist(graph, pac, target),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_target = target
            best_coverage = coverage

    if best_target is None:
        return None, None, paths, 0

    action = graph.pacman_action_to_capture_zone(
        pac, best_target, speed, CAPTURE_DIST
    )
    return action, best_target, paths, best_coverage


# Tên cũ để code ngoài vẫn gọi được.
def choose_intercept_action(graph, pac, ghost, speed, model, horizon=3):
    action, target, paths, _ = choose_aggressive_intercept_action(
        graph, pac, ghost, speed,
        model=model,
        horizon=horizon,
        beam_width=2,
    )
    return action, target, paths


def aggressive_intercept_passes(candidate, base, coverage=0, force_cutoff=False):
    if candidate is None:
        return False
    if base["guaranteed"] and not candidate["guaranteed"]:
        return False

    if candidate["capture_count"] > base["capture_count"]:
        return True
    if candidate["forced_next_count"] > base["forced_next_count"]:
        return True
    if candidate["safe_escape_count"] < base["safe_escape_count"]:
        return True

    max_detour = 4 if force_cutoff else 3
    if coverage >= 2 and candidate["worst_distance"] <= base["worst_distance"] + max_detour:
        return True
    if force_cutoff and candidate["worst_distance"] <= base["worst_distance"] + 4:
        return True
    if (
        candidate["full_speed"]
        and candidate["worst_distance"] <= base["worst_distance"] + 2
        and candidate["next_pressure"] >= base["next_pressure"] - 0.25
    ):
        return True
    return False


def choose_capture_mode_action(graph, pac, ghost, speed, time_budget=0.30):
    aggressive_action, aggressive, metrics = choose_aggressive_pursuit(
        graph, pac, ghost, speed, return_details=True
    )

    # Có bắt chắc hoặc ép bắt lượt kế: hành động ngay, không cần overthinking.
    if aggressive["guaranteed"] or aggressive["forced_next_count"] > 0:
        return aggressive_action

    try:
        deep_action = choose_pacman_action(
            graph, pac, ghost, speed,
            time_budget=time_budget,
            max_depth=4,
        )
    except Exception:
        return aggressive_action

    deep = _find_action_metrics(metrics, deep_action)
    if deep is None:
        return aggressive_action

    # Ít thận trọng hơn V3: chấp nhận Minimax nếu không giảm pressure rõ rệt.
    if deep["score"] >= aggressive["score"]:
        return deep_action
    if (
        deep["capture_count"] >= aggressive["capture_count"]
        and deep["next_pressure"] >= aggressive["next_pressure"]
        and deep["worst_distance"] <= aggressive["worst_distance"] + 2
    ):
        return deep_action
    return aggressive_action


# ---------------------------------------------------------------------------
# GHOST: safe horizon
# ---------------------------------------------------------------------------
def _survival_horizon_state(graph, pac, ghost, speed, depth, deadline, cache):
    _check_deadline(deadline)
    if depth <= 0 or is_capture(pac, ghost):
        return 0
    key = (pac, ghost, depth, speed)
    if key in cache:
        return cache[key]

    best = 0
    p_moves = pacman_moves(graph, pac, speed)
    for _, ng in ghost_moves(graph, ghost):
        _check_deadline(deadline)
        worst = depth
        for _, npac in p_moves:
            _check_deadline(deadline)
            value = 0 if is_capture(npac, ng) else 1 + _survival_horizon_state(
                graph, npac, ng, speed, depth - 1, deadline, cache
            )
            worst = min(worst, value)
            if worst <= best:
                break
        best = max(best, worst)
        if best >= depth:
            break
    cache[key] = best
    return best


def _forced_move_survival_horizon(graph, pac, ng, speed, depth, deadline, cache):
    if depth <= 0:
        return 0
    worst = depth
    for _, npac in pacman_moves(graph, pac, speed):
        _check_deadline(deadline)
        value = 0 if is_capture(npac, ng) else 1 + _survival_horizon_state(
            graph, npac, ng, speed, depth - 1, deadline, cache
        )
        worst = min(worst, value)
        if worst == 0:
            break
    return worst


def _intercept_risk(graph, pac, ghost, move, speed):
    if move == Move.STAY:
        return 0
    risks = 0
    for target in graph.forward_choke_points(ghost, move.value, 6):
        p_turns = graph.pacman_capture_turn_distance(pac, target, speed, CAPTURE_DIST)
        g_turns = _safe_dist(graph, ghost, target)
        if 0 <= p_turns <= g_turns + 1:
            risks += 1
    return risks


def ghost_move_metrics(
    graph, pac, ghost, speed, move_pair,
    survival_depth=3, deadline=None, horizon_cache=None,
    last_move=None,
):
    move, ng = move_pair
    p_candidates = pacman_moves(graph, pac, speed)
    capture_count = 0
    distances = []
    for _, npac in p_candidates:
        if is_capture(npac, ng):
            capture_count += 1
            distances.append(0)
        else:
            distances.append(_safe_dist(graph, npac, ng))

    if horizon_cache is None:
        horizon_cache = {}
    safe_horizon = _forced_move_survival_horizon(
        graph, pac, ng, speed, survival_depth, deadline, horizon_cache
    )

    reverse = int(last_move is not None and action_move(move) == opposite_move(last_move))
    return {
        "move": move,
        "end": ng,
        "guaranteed_safe": capture_count == 0,
        "capture_count": capture_count,
        "pacman_action_count": len(p_candidates),
        "worst_distance": min(distances) if distances else 0,
        "safe_horizon": safe_horizon,
        "space": graph.local_space(ng, 3),
        "degree": graph.open_degree(ng),
        "cycle": graph.cycle_potential(ng, 5),
        "speed_denial": graph.speed_denial_score(ng, pac),
        "corridor": graph.is_corridor(ng),
        "reverse": reverse,
        "intercept_risk": _intercept_risk(graph, pac, ghost, move, speed),
        "rollout": -INF,
    }


# ---------------------------------------------------------------------------
# GHOST: rollout ngắn, mô phỏng đúng action speed=2
# ---------------------------------------------------------------------------
def _rollout_pac_action(graph, pac, ghost, speed, policy, ghost_direction=None):
    if policy == "direct":
        return graph.pacman_action_to_capture_zone(pac, ghost, speed, CAPTURE_DIST)
    if policy == "cutoff" and ghost_direction is not None:
        targets = graph.forward_choke_points(ghost, ghost_direction, 6)
        viable = []
        for target in targets:
            turns = graph.pacman_capture_turn_distance(pac, target, speed, CAPTURE_DIST)
            if turns >= 0:
                viable.append((turns, target))
        if viable:
            target = min(viable)[1]
            return graph.pacman_action_to_capture_zone(pac, target, speed, CAPTURE_DIST)
    return choose_aggressive_pursuit(graph, pac, ghost, speed)


def _rollout_ghost_action(graph, pac, ghost, speed, last_move=None):
    options = []
    for move, ng in ghost_moves(graph, ghost):
        capture_count = 0
        distances = []
        for _, npac in pacman_moves(graph, pac, speed):
            if is_capture(npac, ng):
                capture_count += 1
                distances.append(0)
            else:
                distances.append(_safe_dist(graph, npac, ng))
        reverse = int(last_move is not None and action_move(move) == opposite_move(last_move))
        options.append((
            (
                int(capture_count == 0),
                -capture_count,
                min(distances) if distances else 0,
                graph.cycle_potential(ng, 4),
                graph.local_space(ng, 3),
                graph.speed_denial_score(ng, pac),
                -reverse,
                int(move != Move.STAY),
            ),
            move,
            ng,
        ))
    _, move, ng = max(options, key=lambda item: item[0])
    return move, ng


def ghost_rollout_value(
    graph, pac, root_ghost, speed,
    depth=4, deadline=None, last_move=None,
):
    policies = ("aggressive", "direct", "cutoff")
    values = []

    for policy in policies:
        p_sim = tuple(pac)
        g_sim = tuple(root_ghost)
        g_last = last_move
        survived = 0
        captured = False
        ghost_direction = None

        # Pacman phản ứng với root move.
        p_action = _rollout_pac_action(graph, p_sim, g_sim, speed, policy, ghost_direction)
        p_sim = _end_for_action(graph, p_sim, speed, p_action)
        if is_capture(p_sim, g_sim):
            values.append(-1000.0)
            continue

        survived += 1
        for _ in range(max(0, depth - 1)):
            _check_deadline(deadline)
            move, next_g = _rollout_ghost_action(graph, p_sim, g_sim, speed, g_last)
            ghost_direction = direction_between(g_sim, next_g)
            p_action = _rollout_pac_action(
                graph, p_sim, g_sim, speed, policy, ghost_direction
            )
            next_p = _end_for_action(graph, p_sim, speed, p_action)
            if is_capture(next_p, next_g):
                captured = True
                break
            p_sim, g_sim = next_p, next_g
            g_last = move
            survived += 1

        terminal = (
            100.0 * survived
            + 4.0 * _safe_dist(graph, p_sim, g_sim)
            + 3.0 * graph.cycle_potential(g_sim, 4)
            + graph.local_space(g_sim, 3)
        )
        if captured:
            terminal -= 120.0
        values.append(terminal)

    if not values:
        return -INF
    # Vừa chống P mạnh nhất, vừa không quá bi quan như pure minimax.
    return 0.70 * min(values) + 0.30 * (sum(values) / len(values))


def analyze_ghost_options(
    graph, pac, ghost, speed,
    survival_depth=3,
    time_budget=0.40,
    rollout_depth=4,
    last_move=None,
):
    deadline = time.perf_counter() + max(0.04, time_budget)
    cache = {}
    options = []

    for pair in ghost_moves(graph, ghost):
        _check_deadline(deadline)
        options.append(ghost_move_metrics(
            graph, pac, ghost, speed, pair,
            survival_depth=survival_depth,
            deadline=deadline,
            horizon_cache=cache,
            last_move=last_move,
        ))

    # Chỉ rollout các nước nền tảng tốt nhất để giữ thời gian.
    basic_key = lambda item: (
        int(item["guaranteed_safe"]),
        item["safe_horizon"],
        -item["capture_count"],
        -item["intercept_risk"],
        item["worst_distance"],
        item["cycle"],
        item["space"],
    )
    rollout_candidates = sorted(options, key=basic_key, reverse=True)[:3]
    for item in rollout_candidates:
        _check_deadline(deadline)
        item["rollout"] = ghost_rollout_value(
            graph, pac, item["end"], speed,
            depth=rollout_depth,
            deadline=deadline,
            last_move=item["move"],
        )

    safe_ratio = (
        sum(1 for item in options if item["guaranteed_safe"]) / len(options)
        if options else 0.0
    )
    best_horizon = max((item["safe_horizon"] for item in options), default=0)
    best_margin = max(
        (max(0, item["worst_distance"] - CAPTURE_DIST) for item in options),
        default=0,
    )
    best_space = max((item["space"] for item in options), default=0)
    max_space = max(1, graph.max_reachable_area(3))

    escape_strength = (
        0.45 * min(1.0, best_horizon / max(1, survival_depth))
        + 0.25 * safe_ratio
        + 0.20 * min(1.0, best_margin / 8.0)
        + 0.10 * min(1.0, best_space / max_space)
    )
    return options, {
        "safe_move_ratio": safe_ratio,
        "best_horizon": best_horizon,
        "best_margin": best_margin,
        "space_norm": min(1.0, best_space / max_space),
        "escape_strength": escape_strength,
    }


def select_ghost_move(options, mode=MODE_SAFE):
    if not options:
        return Move.STAY

    if mode == MODE_DELAY:
        key = lambda item: (
            item["safe_horizon"],
            int(item["guaranteed_safe"]),
            -item["capture_count"],
            item["rollout"],
            -item["intercept_risk"],
            item["speed_denial"],
            item["cycle"],
            item["worst_distance"],
            item["space"],
            -item["reverse"],
            int(item["move"] != Move.STAY),
        )
    elif mode == MODE_EXPAND:
        key = lambda item: (
            int(item["guaranteed_safe"]),
            item["safe_horizon"],
            item["rollout"],
            -item["intercept_risk"],
            item["cycle"],
            item["space"],
            item["degree"],
            item["worst_distance"],
            -item["capture_count"],
            -item["reverse"],
            int(item["move"] != Move.STAY),
        )
    else:
        key = lambda item: (
            int(item["guaranteed_safe"]),
            item["safe_horizon"],
            -item["capture_count"],
            item["rollout"],
            -item["intercept_risk"],
            item["worst_distance"],
            item["speed_denial"],
            item["cycle"],
            item["space"],
            -item["reverse"],
            int(item["move"] != Move.STAY),
        )
    return max(options, key=key)["move"]


def choose_ghost_one_step(graph, pac, ghost, speed, last_move=None):
    options = []
    for move, ng in ghost_moves(graph, ghost):
        capture_count = 0
        distances = []
        for _, npac in pacman_moves(graph, pac, speed):
            if is_capture(npac, ng):
                capture_count += 1
                distances.append(0)
            else:
                distances.append(_safe_dist(graph, npac, ng))
        options.append({
            "move": move,
            "end": ng,
            "guaranteed_safe": capture_count == 0,
            "capture_count": capture_count,
            "worst_distance": min(distances) if distances else 0,
            "space": graph.local_space(ng, 3),
            "degree": graph.open_degree(ng),
            "cycle": graph.cycle_potential(ng, 4),
            "speed_denial": graph.speed_denial_score(ng, pac),
            "safe_horizon": int(capture_count == 0),
            "rollout": 0.0,
            "intercept_risk": _intercept_risk(graph, pac, ghost, move, speed),
            "reverse": int(last_move is not None and action_move(move) == opposite_move(last_move)),
        })
    return select_ghost_move(options, MODE_SAFE)


# ---------------------------------------------------------------------------
# Đo năng lực Pacman đối thủ
# ---------------------------------------------------------------------------
def pacman_action_quality(graph, previous_pac, previous_ghost, current_pac, speed):
    metrics = all_pacman_action_metrics(graph, previous_pac, previous_ghost, speed)
    metrics.sort(key=lambda item: item["score"], reverse=True)
    actuals = [
        (index, item)
        for index, item in enumerate(metrics)
        if item["end"] == tuple(current_pac)
    ]
    if not actuals:
        return None

    rank, actual = min(actuals, key=lambda pair: pair[0])
    rank_quality = 1.0 if len(metrics) <= 1 else 1.0 - rank / (len(metrics) - 1)

    best_worst = min(item["worst_distance"] for item in metrics)
    span = max(1, max(item["worst_distance"] for item in metrics) - best_worst)
    distance_quality = 1.0 - min(1.0, (actual["worst_distance"] - best_worst) / span)

    best_escape = min(item["safe_escape_count"] for item in metrics)
    max_escape = max(item["safe_escape_count"] for item in metrics)
    escape_span = max(1, max_escape - best_escape)
    escape_quality = 1.0 - min(
        1.0,
        (actual["safe_escape_count"] - best_escape) / escape_span,
    )

    return 0.40 * rank_quality + 0.30 * distance_quality + 0.30 * escape_quality
