import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.model import SharedGRU
from runtime.data import event_features, split_trials, to_local, to_global
from runtime.postprocess import postprocess, fit_bones
from metrics import score_trial

NAMES = ['Wrist'] + [f'{f}_{i}' for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'] for i in range(1, 5)]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_padding_does_not_change_valid_outputs(self):
        for architecture, bidir in [('shared_gru_v2', False), ('shared_gru_v2', True), ('shared_tcn_v2', False)]:
            c = dict(hidden=8, sessions=['a'], conditions=['x'], metadata=True, architecture=architecture,
                     bidirectional=bidir, shared_input=True, adapter_scale=.1)
            model = SharedGRU(c).eval()
            x, meta = torch.randn(1, 12, 1024), torch.randn(1, 12, 12)
            sid = torch.zeros(1, dtype=torch.long)
            with torch.inference_mode():
                a = model(x, meta, sid, sid)
                xp, mp = torch.cat([x, torch.zeros(1, 9, 1024)], 1), torch.cat([meta, torch.zeros(1, 9, 12)], 1)
                b = model(xp, mp, sid, sid, torch.tensor([12]))
            np.testing.assert_allclose(a.numpy(), b[:, :12].numpy(), atol=2e-6)

    def test_unidirectional_neural_path(self):
        model = SharedGRU(dict(hidden=8, sessions=['a'], conditions=['x'], metadata=False)).eval()
        x = torch.randn(1, 15, 1024)
        meta = torch.zeros(1, 15, 12)
        sid = torch.zeros(1, dtype=torch.long)
        with torch.inference_mode():
            a = model(x, meta, sid, sid)
            x[:, 8:] += 100
            b = model(x, meta, sid, sid)
        np.testing.assert_allclose(a[:, :8].numpy(), b[:, :8].numpy(), atol=1e-6)

    def test_event_features_use_relative_int64_timestamps(self):
        times = 1_800_000_000_000_000_000 + np.arange(20, dtype=np.int64) * 13_510_000
        a = event_features(dict(times=times, onset=3, reward=16))
        b = event_features(dict(times=times - times[0], onset=3, reward=16))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (20, 12))

    def test_split_disjoint(self):
        items = [dict(condition=str(i // 4)) for i in range(36)]
        a, b = split_trials(items, .2, 42)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(sorted(a + b), list(range(36)))

    def test_postprocess_preserves_perfect_fixed_bones(self):
        y = np.tile(np.arange(63, dtype='float32'), (20, 1))
        bones = fit_bones([dict(y=y)], NAMES)
        p = postprocess(y, bones, NAMES, 2, .5)
        np.testing.assert_allclose(p, y, atol=1e-5)
        s = score_trial(y, p, np.arange(20, dtype=np.int64) * 13_510_000, NAMES)
        self.assertAlmostEqual(s['hrfs_local'], 100, places=4)

    def test_coordinate_roundtrip(self):
        y = np.random.default_rng(7).normal(size=(11, 63)).astype('float32')
        np.testing.assert_allclose(to_global(to_local(y, 0), 0), y, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
