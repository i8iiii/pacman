# belief.py - theo dõi / phỏng đoán vị trí đối phương khi mất dấu (enemy_position is None).
#
# ĐÂY LÀ BẢN TỐI GIẢN (placeholder): chỉ nhớ vị trí thấy lần cuối.
# Phần việc của 19127615 là nâng lên belief distribution 21x21 (predict + update 2 pha),
# GIỮ NGUYÊN 3 chữ ký dưới đây - agent.py gọi đúng theo interface này, không được đổi.
#   - EnemyTracker()
#   - update(self, map_state, my_pos, enemy_pos, step)  -> không trả về gì
#   - get_target(self, my_pos)                          -> (row, col) hoặc None
# Xem đặc tả đầy đủ ở labs/lab2/README.md §4 và các bản mẫu HideSeek/Blind/{B,C}/agent.py.


"""Theo dõi vị trí đối phương bằng một phân phối xác suất trên bản đồ.

Interface công khai của ``EnemyTracker`` được giữ nguyên để tương thích với
``agent.py``. Mỗi lần ``update`` thực hiện hai bước:

1. Predict: khuếch tán xác suất theo các nước đi hợp lệ (4 hướng + đứng yên).
2. Update: dùng quan sát hiện tại để đặt lại one-hot khi thấy đối phương,
   hoặc loại mọi ô đang nhìn thấy khi không thấy đối phương.

Theo framework của Lab 2, tường luôn có giá trị 1; ô -1 là ô trống ngoài tầm
nhìn. Vì vậy điều kiện ô đi được là ``map_state != 1``.
"""

import numpy as np


class EnemyTracker:
    def __init__(self):
        # Mảng xác suất cùng kích thước map; được tạo ở lần update đầu tiên.
        self.belief = None

        # Chỉ dùng để hỗ trợ debug/khôi phục; agent.py không truy cập trực tiếp.
        self.last_seen = None
        self.last_seen_step = -1

    @staticmethod
    def _as_map_array(map_state):
        """Chuyển map sang ndarray 2 chiều mà không sửa dữ liệu đầu vào."""
        arr = np.asarray(map_state)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError("map_state must be a non-empty 2D array")
        return arr

    def _set_uniform(self, map_array, unseen_only=False):
        """Đặt belief đều trên các ô hợp lệ.

        Khi ``unseen_only=True``, ưu tiên đúng đặc tả reset của README: chỉ
        phân phối trên ô chưa nhìn thấy (-1). Nếu không còn ô unseen (trạng
        thái mâu thuẫn hiếm gặp), fallback về mọi ô đi được để belief vẫn hợp
        lệ và agent không bị crash.
        """
        walkable = map_array != 1
        candidates = walkable & (map_array == -1) if unseen_only else walkable

        count = int(np.count_nonzero(candidates))
        if count == 0 and unseen_only:
            candidates = walkable
            count = int(np.count_nonzero(candidates))

        self.belief = np.zeros(map_array.shape, dtype=np.float64)
        if count > 0:
            self.belief[candidates] = 1.0 / float(count)

    def _ensure_initialized(self, map_array):
        """Khởi tạo hoặc tái khởi tạo nếu kích thước bản đồ thay đổi."""
        if self.belief is None or self.belief.shape != map_array.shape:
            self._set_uniform(map_array, unseen_only=False)

    def _predict(self, map_array):
        """Khuếch tán belief sang 4 ô kề hợp lệ và chính ô hiện tại.

        Cài đặt dùng phép dịch mảng NumPy thay cho vòng lặp nhiều tầng. Mỗi ô
        nguồn chia đều xác suất cho số đích hợp lệ của riêng nó, nên xác suất
        được bảo toàn trước pha quan sát.
        """
        walkable = map_array != 1

        # Số hành động hợp lệ tại mỗi ô nguồn: STAY + các hướng có đích đi được.
        degree = walkable.astype(np.float64)
        degree[1:, :] += walkable[:-1, :]   # đi UP
        degree[:-1, :] += walkable[1:, :]   # đi DOWN
        degree[:, 1:] += walkable[:, :-1]   # đi LEFT
        degree[:, :-1] += walkable[:, 1:]   # đi RIGHT

        clean = np.nan_to_num(
            np.asarray(self.belief, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        clean[~walkable] = 0.0
        clean[clean < 0.0] = 0.0

        share = np.zeros_like(clean)
        np.divide(clean, degree, out=share, where=degree > 0.0)

        predicted = np.zeros_like(clean)

        # STAY
        predicted += share

        # Mỗi phép gán dưới đây chuyển phần share của ô nguồn sang ô đích.
        predicted[:-1, :] += share[1:, :] * walkable[:-1, :]   # UP
        predicted[1:, :] += share[:-1, :] * walkable[1:, :]    # DOWN
        predicted[:, :-1] += share[:, 1:] * walkable[:, :-1]   # LEFT
        predicted[:, 1:] += share[:, :-1] * walkable[:, 1:]    # RIGHT

        predicted[~walkable] = 0.0
        self.belief = predicted

    def _normalize_or_reset(self, map_array, unseen_only_on_reset=True):
        """Làm sạch số học, chuẩn hóa tổng về 1 hoặc reset an toàn."""
        walkable = map_array != 1
        self.belief = np.nan_to_num(
            np.asarray(self.belief, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.belief[~walkable] = 0.0
        self.belief[self.belief < 0.0] = 0.0

        total = float(self.belief.sum())
        if np.isfinite(total) and total > 1e-15:
            self.belief /= total
        else:
            self._set_uniform(map_array, unseen_only=unseen_only_on_reset)

    def update(self, map_state, my_pos, enemy_pos, step):
        """Cập nhật belief; tuyệt đối không ném exception ra ngoài.

        ``my_pos`` hiện không cần cho phép tính Bayes, nhưng vẫn giữ trong chữ
        ký cố định để tương thích với agent.py. ``step`` được lưu khi nhìn thấy
        đối phương để hỗ trợ kiểm tra/debug.
        """
        previous = None if self.belief is None else self.belief.copy()
        previous_last_seen = self.last_seen
        previous_last_seen_step = self.last_seen_step

        try:
            map_array = self._as_map_array(map_state)
            self._ensure_initialized(map_array)

            # Pha 1 — Predict: đối phương có thể đi 1 ô hoặc đứng yên.
            self._predict(map_array)

            # Pha 2 — Update theo quan sát.
            if enemy_pos is not None:
                row = int(enemy_pos[0])
                col = int(enemy_pos[1])
                h, w = map_array.shape

                if not (0 <= row < h and 0 <= col < w):
                    raise ValueError("enemy_pos is outside the map")
                if map_array[row, col] == 1:
                    raise ValueError("enemy_pos cannot be on a wall")

                self.belief.fill(0.0)
                self.belief[row, col] = 1.0
                self.last_seen = (row, col)
                self.last_seen_step = int(step)
                return

            # Không thấy địch: mọi ô đang nhìn thấy chắc chắn không chứa địch.
            visible = map_array != -1
            self.belief[visible] = 0.0
            self._normalize_or_reset(map_array, unseen_only_on_reset=True)

        except Exception:
            # Ưu tiên giữ trạng thái belief hợp lệ trước đó. Nếu đây là lần đầu
            # và input lỗi, để belief=None để get_target() trả None an toàn.
            self.belief = previous
            self.last_seen = previous_last_seen
            self.last_seen_step = previous_last_seen_step
            return

    def get_target(self, my_pos):
        """Trả ô xác suất cao nhất; hòa thì chọn ô gần ``my_pos`` nhất."""
        try:
            if self.belief is None or self.belief.size == 0:
                return None

            belief = np.nan_to_num(
                np.asarray(self.belief, dtype=np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            max_prob = float(np.max(belief))
            if not np.isfinite(max_prob) or max_prob <= 0.0:
                return None

            # isclose giữ tie-break ổn định trước sai số dấu phẩy động rất nhỏ.
            candidates = np.argwhere(
                np.isclose(belief, max_prob, rtol=1e-12, atol=1e-15)
            )
            if candidates.size == 0:
                return None

            try:
                my_row = int(my_pos[0])
                my_col = int(my_pos[1])
            except Exception:
                my_row = 0
                my_col = 0

            # np.argwhere đã theo thứ tự hàng-cột; row/col tiếp tục làm tie-break
            # cuối để kết quả xác định, không nhảy ngẫu nhiên giữa các bước.
            best = min(
                candidates,
                key=lambda rc: (
                    abs(int(rc[0]) - my_row) + abs(int(rc[1]) - my_col),
                    int(rc[0]),
                    int(rc[1]),
                ),
            )
            return int(best[0]), int(best[1])

        except Exception:
            return self.last_seen
