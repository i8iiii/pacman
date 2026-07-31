"""Tests for StateBuilder — multi-channel frame stacking for DRL Pacman Seeker."""

import sys
from pathlib import Path

PACMAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACMAN_ROOT / "src"))

import unittest
import numpy as np
from rl.config import Config
from rl.state_builder import StateBuilder


class TestStateBuilder(unittest.TestCase):
    """Verify shape, channel semantics, frame stacking, and last-known tracking."""

    def setUp(self):
        self.config = Config()
        # small map for easier assertions
        self.config.map_height = 5
        self.config.map_width = 5
        self.config.n_frames = 2
        self.config.n_channels = 6
        self.builder = StateBuilder(self.config)

        # test map:
        #   0  1  0  0  0
        #   0  1  0 -1  0
        #   0  0  0  0  0
        #   0 -1  0  1  0
        #   0  0  0  0  0
        self.sample_map = np.array([
            [0, 1, 0, 0, 0],
            [0, 1, 0,-1, 0],
            [0, 0, 0, 0, 0],
            [0,-1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ])

    # ------------------------------------------------------------------
    # shape tests
    # ------------------------------------------------------------------

    def test_output_shape_default_config(self):
        """With the real config (21x19, 4 frames, 6 channels) shape is (24,21,19)."""
        real_cfg = Config()
        sb = StateBuilder(real_cfg)
        sb.reset()
        big_map = np.full((21, 19), 0)
        result = sb.build(big_map, (0, 0), None, 0)
        self.assertEqual(result.shape, (24, 21, 19))

    def test_output_shape_small_config(self):
        """(n_frames * n_channels, H, W) = (12, 5, 5) for the small config."""
        self.builder.reset()
        result = self.builder.build(self.sample_map, (0, 0), None, 0)
        self.assertEqual(result.shape, (12, 5, 5),
                         f"Expected (12,5,5), got {result.shape}")

    # ------------------------------------------------------------------
    # channel content tests (single-frame buffer for simplicity)
    # ------------------------------------------------------------------

    def _build_single(self, my_pos, enemy_pos):
        """Build with a fresh builder so only one frame is in buffer."""
        sb = StateBuilder(self.config)
        sb.reset()
        return sb.build(self.sample_map, my_pos, enemy_pos, 0)

    def test_channel0_walls(self):
        """Channel 0 = 1 where map==1, else 0."""
        result = self._build_single((0, 0), None)
        # take the first frame (first 6 channels)
        frame = result[-6:]  # last frame appended
        ch0 = frame[0]  # walls channel
        expected_walls = (self.sample_map == 1).astype(np.float32)
        np.testing.assert_array_equal(ch0, expected_walls)

    def test_channel1_visible_empty(self):
        """Channel 1 = 1 where map==0, else 0."""
        result = self._build_single((0, 0), None)
        frame = result[-6:]
        ch1 = frame[1]
        expected_empty = (self.sample_map == 0).astype(np.float32)
        np.testing.assert_array_equal(ch1, expected_empty)

    def test_channel2_unseen(self):
        """Channel 2 = 1 where map==-1, else 0."""
        result = self._build_single((0, 0), None)
        frame = result[-6:]
        ch2 = frame[2]
        expected_unseen = (self.sample_map == -1).astype(np.float32)
        np.testing.assert_array_equal(ch2, expected_unseen)

    def test_channel3_pacman_position(self):
        """Channel 3 = 1 at pacman position, else 0."""
        my_pos = (2, 2)
        result = self._build_single(my_pos, None)
        frame = result[-6:]
        ch3 = frame[3]
        self.assertEqual(ch3[my_pos], 1.0)
        self.assertEqual(ch3.sum(), 1.0)

    def test_channel4_ghost_visible(self):
        """Channel 4 = 1 at enemy position when visible, else 0."""
        result = self._build_single((0, 0), (3, 3))
        frame = result[-6:]
        ch4 = frame[4]
        self.assertEqual(ch4[3, 3], 1.0)
        self.assertEqual(ch4.sum(), 1.0)

    def test_channel4_ghost_not_visible(self):
        """Channel 4 = all zeros when enemy_position is None."""
        result = self._build_single((0, 0), None)
        frame = result[-6:]
        ch4 = frame[4]
        np.testing.assert_array_equal(ch4, np.zeros((5, 5), dtype=np.float32))

    def test_channel5_last_known_ghost_persists(self):
        """Channel 5 retains last seen ghost position across builds."""
        sb = StateBuilder(self.config)
        sb.reset()
        # first build with visible enemy at (1, 1)
        sb.build(self.sample_map, (0, 0), (1, 1), 0)
        # second build with enemy=None; last_known should still be (1, 1)
        result = sb.build(self.sample_map, (0, 0), None, 1)
        frame = result[-6:]
        ch5 = frame[5]
        self.assertEqual(ch5[1, 1], 1.0)
        self.assertEqual(ch5.sum(), 1.0)

    def test_channel5_last_known_updates_when_visible(self):
        """Channel 5 updates when enemy is visible."""
        sb = StateBuilder(self.config)
        sb.reset()
        sb.build(self.sample_map, (0, 0), (1, 1), 0)
        result = sb.build(self.sample_map, (0, 0), (4, 4), 1)
        frame = result[-6:]
        ch5 = frame[5]
        self.assertEqual(ch5[4, 4], 1.0)
        self.assertEqual(ch5.sum(), 1.0)

    # ------------------------------------------------------------------
    # frame stacking tests
    # ------------------------------------------------------------------

    def test_buffer_starts_with_zeros(self):
        """Before any build(), reset() pre-fills buffer with zero frames."""
        sb = StateBuilder(self.config)
        sb.reset()
        result = sb.build(self.sample_map, (0, 0), None, 0)
        # first n_frames-1 frames should be all zeros
        first_frame = result[:6]  # n_frames=2, so first frame is at [0:6]
        np.testing.assert_array_equal(first_frame, np.zeros((6, 5, 5), dtype=np.float32))

    def test_buffer_only_keeps_n_frames(self):
        """After n_frames+2 builds, only the most recent n_frames are kept."""
        sb = StateBuilder(self.config)
        sb.reset()
        results = []
        positions = [(0, 0), (1, 1), (2, 2), (3, 3)]
        for i, pos in enumerate(positions):
            r = sb.build(self.sample_map, pos, None, i)
            results.append(r)
        # After 4 builds with n_frames=2, result should contain frames 2 and 3
        final = results[-1]
        # First frame in buffer should be frame from step 2
        first_stacked = final[:6]
        frame2_result = results[2][-6:]  # frame from step 2
        np.testing.assert_array_equal(first_stacked, frame2_result)

    def test_reset_clears_buffer_and_last_known(self):
        """reset() clears the frame buffer and last_known_enemy_pos."""
        sb = StateBuilder(self.config)
        sb.reset()
        sb.build(self.sample_map, (0, 0), (1, 1), 0)
        sb.reset()
        self.assertIsNone(sb.last_known_enemy_pos)
        # After reset, building without enemy should have all-zero ch5
        result = sb.build(self.sample_map, (0, 0), None, 0)
        frame = result[-6:]
        ch5 = frame[5]
        np.testing.assert_array_equal(ch5, np.zeros((5, 5), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
