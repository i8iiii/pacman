from random import random
import random
import numpy as np
from environment import Move

def get_neighbors(pos: tuple, map_state: np.ndarray) -> list:
    """Lấy danh sách các ô hợp lệ xung quanh vị trí hiện tại."""
    neighbors = []
    height, width = map_state.shape
    
    # Thứ tự ưu tiên: Lên, Xuống, Trái, Phải
    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        delta_row, delta_col = move.value
        next_pos = (pos[0] + delta_row, pos[1] + delta_col)
        
        # Kiểm tra trong ranh giới và không phải tường
        if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width:
            if map_state[next_pos[0], next_pos[1]] == 0:
                neighbors.append((next_pos, move))
                
    return neighbors

def bfs(start: tuple, goal: tuple, map_state: np.ndarray) -> list:
    """Thuật toán BFS: Luôn tìm được đường ĐI NGẮN NHẤT."""
    queue = [(start, [])]
    visited = {start}
    
    while queue:
        current_pos, path = queue.pop(0) # FIFO (Lấy ra ở đầu)
        
        if current_pos == goal:
            return path
            
        for next_pos, move in get_neighbors(current_pos, map_state):
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, path + [move]))
                
    return []

def dfs(start: tuple, goal: tuple, map_state: np.ndarray) -> list:
    """
    Thuật toán DFS: Đi sâu vào tận cùng của một nhánh.
    Lưu ý: DFS sẽ tìm được đường, nhưng THƯỜNG KHÔNG PHẢI đường ngắn nhất.
    """
    stack = [(start, [])]
    visited = set()
    
    while stack:
        current_pos, path = stack.pop() # LIFO (Lấy ra ở cuối - đỉnh stack)
        
        if current_pos == goal:
            return path
            
        if current_pos not in visited:
            visited.add(current_pos)
            
            for next_pos, move in get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    stack.append((next_pos, path + [move]))
                    
    return []

def manhattan_heuristic(pos: tuple, goal: tuple) -> int:
    """Tính khoảng cách Manhattan."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def astar(start: tuple, goal: tuple, map_state: np.ndarray) -> list:
    """
    Thuật toán A* Search.
    Sử dụng List cơ bản làm Priority Queue để tuân thủ quy định của đồ án.
    """
    # frontier lưu trữ các tuple: (f_cost, g_cost, vị_trí, đường_đi)
    frontier = [(0, 0, start, [])]
    
    # Dictionary lưu chi phí g_cost nhỏ nhất để đi đến một ô
    g_costs = {start: 0}
    
    while frontier:
        # TÌM Ô CÓ CHI PHÍ f(n) NHỎ NHẤT THỦ CÔNG (Thay thế cho heapq)
        min_idx = 0
        for i in range(1, len(frontier)):
            if frontier[i][0] < frontier[min_idx][0]:
                min_idx = i
                
        # Lấy phần tử tốt nhất ra khỏi frontier
        f_cost, current_g, current_pos, path = frontier.pop(min_idx)
        
        # Đã tìm thấy mục tiêu
        if current_pos == goal:
            return path
            
        # Khám phá các ô lân cận
        for next_pos, move in get_neighbors(current_pos, map_state):
            new_g = current_g + 1 
            
            # Chỉ xét ô tiếp theo nếu chưa đi qua, hoặc tìm được đường mới ngắn hơn
            if next_pos not in g_costs or new_g < g_costs[next_pos]:
                g_costs[next_pos] = new_g
                
                h_cost = manhattan_heuristic(next_pos, goal)
                new_f = new_g + h_cost
                
                frontier.append((new_f, new_g, next_pos, path + [move]))
                
    return []

def greedy_search(start: tuple, goal: tuple, map_state: np.ndarray) -> list:
    """
    Thuật toán Tham lam (Greedy Seeker):
    Chỉ nhìn vào bước đi ngay trước mắt. Chọn ô có khoảng cách Manhattan 
    tới đích gần nhất mà không quan tâm nó có dẫn vào ngõ cụt hay không.
    """
    neighbors = get_neighbors(start, map_state)
    if not neighbors:
        return []
        
    best_move = None
    min_dist = float('inf')
    
    for next_pos, move in neighbors:
        # Tính khoảng cách từ ô lân cận tới Ghost
        dist = manhattan_heuristic(next_pos, goal)
        if dist < min_dist:
            min_dist = dist
            best_move = move
            
    # Trả về một mảng chứa 1 bước đi duy nhất để đồng bộ với cấu trúc path
    return [best_move] if best_move else []

def random_search(start: tuple, map_state: np.ndarray) -> list:
    """
    Thuật toán Ngẫu nhiên (Random Seeker):
    Nhắm mắt đi bừa vào một trong các ô trống xung quanh.
    """
    neighbors = get_neighbors(start, map_state)
    if not neighbors:
        return []
        
    # Chọn ngẫu nhiên 1 tuple (next_pos, move) từ danh sách
    _, random_move = random.choice(neighbors)
    
    return [random_move]
