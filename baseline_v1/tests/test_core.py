import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.data import to_local, to_global, features, split_trials
from runtime.model import PoseGRU
from metrics import score_trial

NAMES = ['Wrist'] + [f'{f}_{i}' for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'] for i in range(1, 5)]


class CoreTests(unittest.TestCase):
    def test_coordinate_roundtrip(self):
        y = np.random.default_rng(1).normal(size=(8, 63)).astype('float32')
        for wrist in [0, 5, 20]:
            np.testing.assert_allclose(to_global(to_local(y, wrist), wrist), y, atol=1e-6)

    def test_split_has_no_overlap_and_retains_conditions(self):
        items = [{'condition': str(i // 5)} for i in range(20)]
        tr, va = split_trials(items, 0.2, 42)
        self.assertFalse(set(tr) & set(va))
        self.assertEqual(sorted(tr + va), list(range(20)))
        self.assertEqual({items[i]['condition'] for i in tr}, {items[i]['condition'] for i in va})
        self.assertEqual((tr, va), split_trials(items, 0.2, 42))

    def test_perfect_motion_and_static_score(self):
        rng = np.random.default_rng(2)
        for y in [rng.normal(size=(12, 63)), np.tile(np.arange(63), (12, 1))]:
            score = score_trial(y, y, np.arange(12, dtype=np.int64) * 13510000, NAMES)
            self.assertAlmostEqual(score['hrfs_local'], 100, places=8)
            self.assertAlmostEqual(score['mpjpe'], 0)

    def test_translation_changes_wrist_not_local_pose(self):
        y = np.random.default_rng(5).normal(size=(12, 63))
        p = (y.reshape(-1, 21, 3) + [10, 0, 0]).reshape(-1, 63)
        score = score_trial(y, p, np.arange(12, dtype=np.int64) * 13510000, NAMES)
        self.assertAlmostEqual(score['pose'], 100, places=8)
        self.assertAlmostEqual(score['bone'], 100, places=8)
        self.assertAlmostEqual(score['wrist'], 100 * np.exp(-0.5), places=8)

    def test_full_preprocessing_and_model_are_causal(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = PoseGRU({'hidden': 8, 'latent': 4}, np.eye(4, 63, dtype='float32')).eval()
        x = np.random.default_rng(9).normal(size=(20, 512)).astype('float32')
        mean, std = np.zeros(512), np.ones(512)
        with torch.inference_mode():
            a = model(torch.from_numpy(features(x, mean, std))[None])[0].numpy()
            x[10:] += 100
            b = model(torch.from_numpy(features(x, mean, std))[None])[0].numpy()
        np.testing.assert_allclose(a[:, :10], b[:, :10], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
