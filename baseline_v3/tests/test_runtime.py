import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.model import SharedGRU
from runtime.data import event_features, model_events, split_trials, to_local, to_global, features, model_target, decode_target
from runtime.postprocess import postprocess, fit_bones
from metrics import score_trial
from training_utils import mix_trial

NAMES = ['Wrist'] + [f'{f}_{i}' for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'] for i in range(1, 5)]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_padding_does_not_change_valid_outputs(self):
        for architecture, bidir in [('shared_gru_v2', False), ('shared_gru_v2', True), ('shared_tcn_v2', False), ('dual_gru_v3', False), ('stacked_dual_v3', False), ('attention_v3', False)]:
            c = dict(hidden=8, sessions=['a'], conditions=['x'], metadata=True, architecture=architecture,
                     bidirectional=bidir, shared_input=True, adapter_scale=.1, origin_head=True)
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

    def test_recording_clock_translation_and_extrapolation(self):
        times = np.int64(1_800_000_000_000_000_000) + np.arange(12, dtype=np.int64) * 13_510_000
        item = dict(times=times, onset=2, reward=9)
        stat = dict(clock_origin=np.array(times[0] - 100_000_000_000, dtype=np.int64), clock_scale=np.array(200.))
        c = dict(recording_clock=True, metadata=True, hidden=8, sessions=['a'], conditions=['x'], architecture='dual_gru_v3')
        a = model_events(item, stat, c)
        shift = times[0]
        b = model_events({**item, 'times': times - shift}, {**stat, 'clock_origin': stat['clock_origin'] - shift}, c)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (12, 16))
        model = SharedGRU(c).eval()
        with torch.inference_mode():
            out = model(torch.zeros(1, 12, 1024), torch.from_numpy(a)[None], torch.tensor([0]), torch.tensor([0]))
        self.assertTrue(torch.isfinite(out).all())
        far = model_events({**item, 'times': times + 10**15}, stat, c)
        self.assertTrue(np.isfinite(far).all())
        self.assertLessEqual(float(np.max(np.abs(far[:, 12:]))), 4.)

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

    def test_multiscale_features_are_trial_local_and_finite(self):
        x = np.ones((1, 512), dtype='float32')
        for mode, dim in [('ema', 1024), ('multiscale', 2048), ('lagged', 2560), ('wide', 2048)]:
            f = features(x, np.zeros(512), np.ones(512), mode)
            self.assertEqual(f.shape, (1, dim))
            self.assertTrue(np.isfinite(f).all())
            self.assertEqual(f.dtype, np.float32)

    def test_canonical_rotation_roundtrip(self):
        rng = np.random.default_rng(19)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        stat = {'rotation': rotation.astype('float32')}
        y = rng.normal(size=(20, 63)).astype('float32')
        np.testing.assert_allclose(decode_target(model_target(y, stat), stat), y, atol=1e-6)

    def test_projected_features_match_model_input(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(13, 512)).astype('float32')
        projection = np.eye(512, 32, dtype='float32')
        f = features(x, np.zeros(512), np.ones(512), 'multiscale', projection)
        self.assertEqual(f.shape, (13, 128))
        model = SharedGRU(dict(hidden=8, sessions=['a'], conditions=['x'], metadata=True,
                               architecture='dual_gru_v3', feature_dim=128, shared_input=True)).eval()
        sid = torch.zeros(1, dtype=torch.long)
        with torch.inference_mode():
            output = model(torch.from_numpy(f)[None], torch.zeros(1, 13, 12), sid, sid)
        self.assertEqual(tuple(output.shape), (1, 13, 63))
        self.assertTrue(torch.isfinite(output).all())

    def test_event_aligned_mixup(self):
        def trial(phase, offset):
            phase = torch.tensor(phase, dtype=torch.float32)
            meta = torch.zeros(len(phase), 12)
            meta[:, 2] = phase
            return ((phase + offset)[:, None], meta, (2*phase + offset)[:, None], 0, 1)
        a = trial([-1, 0, 1, 2], 0)
        b = trial([-1, -.5, 0, .5, 1, 1.5, 2], 4)
        mixed = mix_trial(a, b, .75)
        torch.testing.assert_close(mixed[0], a[0] + 1)
        torch.testing.assert_close(mixed[2], a[2] + 1)
        torch.testing.assert_close(mixed[1], a[1])
        with self.assertRaises(ValueError):
            mix_trial(a, (*b[:3], 1, 1), .5)

    def test_zero_initialized_experts_preserve_parent(self):
        config = dict(hidden=8, sessions=['a'], conditions=['x', 'y'], metadata=True,
                      architecture='dual_gru_v3', shared_input=True)
        parent = SharedGRU(config).eval()
        experts = SharedGRU({**config, 'condition_input': True, 'condition_head': True}).eval()
        result = experts.load_state_dict(parent.state_dict(), strict=False)
        self.assertEqual(set(result.missing_keys), {'condition_input_weight', 'condition_weight'})
        x, meta = torch.randn(1, 9, 1024), torch.randn(1, 9, 12)
        sid, cid = torch.tensor([0]), torch.tensor([1])
        with torch.inference_mode():
            torch.testing.assert_close(parent(x, meta, sid, cid), experts(x, meta, sid, cid))

    def test_vectorized_metric_matches_frozen_v2_formula(self):
        import importlib.util
        path = Path(__file__).resolve().parents[2] / 'baseline_v2' / 'metrics.py'
        spec = importlib.util.spec_from_file_location('frozen_metrics', path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        rng = np.random.default_rng(12)
        for length in (1, 2, 3, 30):
            y = rng.normal(size=(length, 63)).astype('float32')
            for pred in (y.copy(), y + rng.normal(size=y.shape), np.zeros_like(y)):
                times = np.arange(length, dtype=np.int64) * 13_510_000
                a, b = score_trial(y, pred, times, NAMES), reference.score_trial(y, pred, times, NAMES)
                for key in a:
                    self.assertAlmostEqual(a[key], b[key], places=10)


if __name__ == '__main__':
    unittest.main()
