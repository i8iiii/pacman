import time

from environment import Move
from search_utils import (
    MazeGraph, manhattan, ghost_moves, pacman_moves,
)

# Ngưỡng bắt: Pacman thắng khi Manhattan(Pacman, Ghost) < CAPTURE_DIST.
# Trùng với mặc định của Arena (--capture-distance 2).
CAPTURE_DIST = 2

# Giá trị "vô cực" và phần thưởng khi bắt được.
INF = 10 ** 9
CAPTURE_SCORE = 10 ** 6

# Trọng số hàm đánh giá.
W_DIST = 10.0     # mỗi đơn vị khoảng cách thực
W_TRAP = 2.0      # phần thưởng cho Pacman khi Ghost ít đường thoát


def is_capture(pac, ghost) -> bool:
    return manhattan(pac, ghost) < CAPTURE_DIST


def evaluate(graph: MazeGraph, pac, ghost, plies_used: int) -> float:
    if is_capture(pac, ghost):
        return CAPTURE_SCORE - plies_used

    d = graph.dist(pac, ghost)
    if d < 0:  # không tới được nhau (hiếm) -> coi như rất xa
        d = graph.h * graph.w

    util = -W_DIST * d
    util += W_TRAP * (4 - graph.open_degree(ghost))  # Ghost càng bí, Pacman càng lợi
    return util


def _value(graph, pac, ghost, depth, pac_first, speed,
           alpha, beta, deadline, plies_used):
    if is_capture(pac, ghost):
        return CAPTURE_SCORE - plies_used
    if depth == 0 or time.time() > deadline:
        return evaluate(graph, pac, ghost, plies_used)

    if pac_first:
        # Pacman = MAX (ngoài), Ghost = MIN (trong)
        best = -INF
        for _, npac in pacman_moves(graph, pac, speed):
            worst = INF
            for _, ng in ghost_moves(graph, ghost):
                if is_capture(npac, ng):
                    v = CAPTURE_SCORE - (plies_used + 1)
                else:
                    v = _value(graph, npac, ng, depth - 1, pac_first, speed,
                               alpha, beta, deadline, plies_used + 1)
                if v < worst:
                    worst = v
                if worst <= alpha:        # cắt tỉa beta của lớp MAX bên ngoài
                    break
            if worst > best:
                best = worst
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
    else:
        # Ghost = MIN (ngoài), Pacman = MAX (trong)
        best = INF
        for _, ng in ghost_moves(graph, ghost):
            resp = -INF
            for _, npac in pacman_moves(graph, pac, speed):
                if is_capture(npac, ng):
                    v = CAPTURE_SCORE - (plies_used + 1)
                else:
                    v = _value(graph, npac, ng, depth - 1, pac_first, speed,
                               alpha, beta, deadline, plies_used + 1)
                if v > resp:
                    resp = v
                if resp >= beta:          # cắt tỉa alpha của lớp MIN bên ngoài
                    break
            if resp < best:
                best = resp
            if best < beta:
                beta = best
            if alpha >= beta:
                break
        return best


#  Quyết định cho Pacman (Seeker) — TỐI ĐA HOÁ utility
def choose_pacman_action(graph, pac, ghost, speed,
                         time_budget=0.8, max_depth=6):
    deadline = time.time() + time_budget
    root_actions = pacman_moves(graph, pac, speed)

    # Sắp xếp sơ bộ: thử nước tiến gần Ghost trước -> cắt tỉa tốt hơn.
    root_actions.sort(key=lambda am: graph.dist(am[1], ghost))

    best_action = root_actions[0][0]
    # Đào sâu dần; giữ kết quả của mức sâu cuối cùng hoàn tất kịp thời gian.
    for depth in range(1, max_depth + 1):
        cur_best_action, cur_best_val = None, -INF
        alpha, beta = -INF, INF
        completed = True
        for action, npac in root_actions:
            # Ghost đáp trả để TỐI THIỂU HOÁ utility của Pacman.
            worst = INF
            for _, ng in ghost_moves(graph, ghost):
                if is_capture(npac, ng):
                    v = CAPTURE_SCORE
                else:
                    v = _value(graph, npac, ng, depth - 1, True, speed,
                               alpha, beta, deadline, 1)
                worst = min(worst, v)
                if worst <= alpha:
                    break
            if worst > cur_best_val:
                cur_best_val, cur_best_action = worst, action
            alpha = max(alpha, cur_best_val)
            if time.time() > deadline:
                completed = False
                break
        if cur_best_action is not None and (completed or best_action is None):
            best_action = cur_best_action
        if not completed:
            break
        if cur_best_val >= CAPTURE_SCORE - max_depth:  # đã thấy đường bắt chắc chắn
            break

    return best_action


#  Quyết định cho Ghost (Hider) — TỐI THIỂU HOÁ utility
def choose_ghost_move(graph, pac, ghost, enemy_speed,
                      time_budget=0.8, max_depth=6):
    deadline = time.time() + time_budget
    root_moves = ghost_moves(graph, ghost)

    # Thử nước ra xa Pacman trước -> cắt tỉa tốt hơn.
    root_moves.sort(key=lambda mp: -graph.dist(mp[1], pac))

    best_move = root_moves[0][0]
    for depth in range(1, max_depth + 1):
        cur_best_move, cur_best_val = None, INF
        alpha, beta = -INF, INF
        completed = True
        for move, ng in root_moves:
            # Pacman đáp trả để TỐI ĐA HOÁ utility.
            resp = -INF
            for _, npac in pacman_moves(graph, pac, enemy_speed):
                if is_capture(npac, ng):
                    v = CAPTURE_SCORE
                else:
                    v = _value(graph, npac, ng, depth - 1, False, enemy_speed,
                               alpha, beta, deadline, 1)
                resp = max(resp, v)
                if resp >= beta:
                    break
            if resp < cur_best_val:
                cur_best_val, cur_best_move = resp, move
            beta = min(beta, cur_best_val)
            if time.time() > deadline:
                completed = False
                break
        if cur_best_move is not None and (completed or best_move is None):
            best_move = cur_best_move
        if not completed:
            break

    return best_move
