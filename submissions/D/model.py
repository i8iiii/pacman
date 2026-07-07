# model.py — CNN-DQN Model
# Combines ideas from submission C (CNN + DQN) and submission 1 (game state evaluation)
# Architecture: CNN feature extractor → DQN head with ReLU and Linear activations
# Designed for CPU training

import torch
import torch.nn as nn
import torch.nn.functional as F


class PacmanCNNDQN(nn.Module):
    """
    CNN-DQN model for Pacman agent.

    Architecture overview:
    ─────────────────────
    Input: map_state [B, 1, 21, 21] + last_move one-hot [B, 4]

    CNN Feature Extractor (with ReLU activation):
        Conv2d(1→32, 3×3, stride=1, pad=1)  → ReLU → [B, 32, 21, 21]
        Conv2d(32→64, 3×3, stride=2, pad=1) → ReLU → [B, 64, 11, 11]
        Conv2d(64→128, 3×3, stride=2, pad=1)→ ReLU → [B, 128, 6, 6]

    DQN Head:
        Flatten CNN features (128*6*6=4608) + last_move (4) = 4612
        Linear(4612 → 512)  → ReLU
        Linear(512 → 128)   → ReLU
        Linear(128 → 4)     → Linear (identity, no activation — raw Q-values)

    Output: Q-values for 4 actions [UP, DOWN, LEFT, RIGHT]
    """

    def __init__(self, input_shape=(1, 21, 21), n_actions=4):
        super(PacmanCNNDQN, self).__init__()

        # ── CNN Feature Extractor ──
        # Layer 1: Extract low-level spatial features (edges, walls)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)   # → [B, 32, 21, 21]
        self.bn1 = nn.BatchNorm2d(32)

        # Layer 2: Downsample + extract mid-level features (corridors, junctions)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # → [B, 64, 11, 11]
        self.bn2 = nn.BatchNorm2d(64)

        # Layer 3: Further downsample + extract high-level features (strategic zones)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # → [B, 128, 6, 6]
        self.bn3 = nn.BatchNorm2d(128)

        # Compute flattened feature size: 128 * ceil(11/2) * ceil(11/2) = 128 * 6 * 6
        self.feature_size = 128 * 6 * 6  # = 4608

        # ── DQN Head ──
        # Concatenate CNN features (4608) + last_move one-hot (4) = 4612
        self.fc1 = nn.Linear(self.feature_size + n_actions, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_actions)  # Output: Q-values (linear activation)

        # Dropout for regularization during training
        self.dropout = nn.Dropout(p=0.1)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for ReLU layers, Xavier for the output layer."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m is self.fc3:
                    # Output layer: Xavier init for linear activation
                    nn.init.xavier_uniform_(m.weight)
                else:
                    # Hidden layers: Kaiming init for ReLU activation
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, last_move_vec):
        """
        Forward pass.

        Args:
            x:             Map state tensor [B, 1, 21, 21]
            last_move_vec: One-hot encoded last move [B, 4]

        Returns:
            Q-values for each action [B, 4]
        """
        # ── CNN Feature Extraction (ReLU activation) ──
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 32, 21, 21]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 64, 11, 11]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, 6, 6]

        # Flatten spatial features
        x = x.view(x.size(0), -1)            # [B, 4608]

        # ── Concatenate with last move info ──
        combined = torch.cat((x, last_move_vec), dim=1)  # [B, 4612]

        # ── DQN Head ──
        x = F.relu(self.fc1(combined))        # [B, 512] — ReLU activation
        x = self.dropout(x)
        x = F.relu(self.fc2(x))              # [B, 128] — ReLU activation
        x = self.dropout(x)

        # Output layer: Linear activation (identity) — raw Q-values
        # No activation applied here; this is standard for DQN
        # as Q-values can be any real number
        q_values = self.fc3(x)               # [B, 4] — Linear (no activation)

        return q_values


class PacmanCNNDQN_Small(nn.Module):
    """
    Smaller variant for faster training / deployment under file-size constraints.

    Architecture:
        Conv2d(1→32, 3×3, s=1, p=1) → ReLU
        Conv2d(32→64, 3×3, s=2, p=1) → ReLU
        FC(7744+4 → 256) → ReLU
        FC(256 → 4) → Linear

    ~2M params → ~7.6MB file
    """

    def __init__(self, input_shape=(1, 21, 21), n_actions=4):
        super(PacmanCNNDQN_Small, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.feature_size = 64 * 11 * 11  # = 7744

        self.fc1 = nn.Linear(self.feature_size + n_actions, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x, last_move_vec):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        combined = torch.cat((x, last_move_vec), dim=1)
        x = F.relu(self.fc1(combined))
        return self.fc2(x)  # Linear activation (identity)
