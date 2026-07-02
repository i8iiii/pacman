# model.py — Nhóm 08
# Model lớn hơn (~19.5MB) để tận dụng giới hạn nộp bài 20MB
import torch
import torch.nn as nn
import torch.nn.functional as F


class PacmanNet(nn.Module):
    """
    CNN lớn cho DQN Pacman — tận dụng tối đa budget 20MB.
    Architecture: Conv(32) → Conv(64, stride=2) → FC(650) → 4 actions
    Số params: ~5.05M → kích thước ~19.3MB
    """
    def __init__(self, input_shape=(1, 21, 21), n_actions=4):
        super(PacmanNet, self).__init__()
        # Conv lớn hơn để extract features tốt hơn
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)   # → 32×21×21
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # → 64×11×11

        # 64 × ceil(21/2) × ceil(21/2) = 64 × 11 × 11 = 7744
        self.feature_size = 64 * 11 * 11  # = 7744

        # FC lớn: 7744 map features + 4 last_move → 650 hidden → 4 actions
        # 7748 × 650 ≈ 5.04M params → ~19.3MB file
        self.fc1 = nn.Linear(self.feature_size + 4, 650)
        self.fc2 = nn.Linear(650, 4)

    def forward(self, x, last_move_vec):
        # x: [B, 1, 21, 21]   last_move_vec: [B, 4]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)          # [B, 7744]
        combined = torch.cat((x, last_move_vec), dim=1)  # [B, 7748]
        x = F.relu(self.fc1(combined))      # [B, 650]
        return self.fc2(x)                  # [B, 4]