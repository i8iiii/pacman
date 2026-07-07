# model.py — CNN-DQN Model for Pacman Agent (Submission 1.2)
# Small 2-layer CNN feature extractor + DQN head
# Designed for CPU inference under 1ms

import torch
import torch.nn as nn
import torch.nn.functional as F


class PacmanCNN(nn.Module):
    """
    Small CNN-DQN for Pacman agent.

    Architecture:
        Input: map_state [B, 1, 21, 21] + last_move one-hot [B, 4]

        CNN Feature Extractor:
            Conv2d(1→32, 3×3, stride=1, pad=1) → ReLU  → [B, 32, 21, 21]
            Conv2d(32→64, 3×3, stride=2, pad=1) → ReLU  → [B, 64, 11, 11]
            Flatten                                        → [B, 7744]

        DQN Head:
            Linear(7744 + 4 → 256) → ReLU → Dropout(0.1)
            Linear(256 → 4)        → Linear (raw Q-values)

        Output: Q-values for [UP, DOWN, LEFT, RIGHT]
    """

    def __init__(self, input_shape=(1, 21, 21), n_actions=4):
        super(PacmanCNN, self).__init__()

        # CNN Feature Extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # 64 * 11 * 11 = 7744
        self.feature_size = 64 * 11 * 11

        # DQN Head
        self.fc1 = nn.Linear(self.feature_size + n_actions, 256)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(256, n_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m is self.fc2:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, last_move_vec):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        combined = torch.cat((x, last_move_vec), dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        q_values = self.fc2(x)
        return q_values
