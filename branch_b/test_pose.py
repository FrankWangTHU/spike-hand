import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from pose import fit_pose_pca
from runtime.model import SharedGRU, save_checkpoint, load_checkpoint


class PoseTests(unittest.TestCase):
    def test_pca_and_roundtrip(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        for k in (16, 32):
            x = np.random.default_rng(42).normal(size=(200, 60))
            mean, basis, report = fit_pose_pca([x], k)
            np.testing.assert_allclose(basis @ basis.T, np.eye(k), atol=1e-6)
            self.assertGreater(report['explained_variance_ratio'], 0)
            config = dict(hidden=8, sessions=['s'], conditions=[1], feature_dim=4,
                          metadata=False, architecture='stacked_dual_v3',
                          pose_components=k, pose_residual_scale=.2)
            model = SharedGRU(config).eval()
            model.pose_mean.copy_(torch.from_numpy(mean))
            model.pose_basis.copy_(torch.from_numpy(basis))
            self.assertFalse(model.pose_basis.requires_grad)
            inputs = torch.randn(2, 9, 4)
            sid = torch.zeros(2, dtype=torch.long)
            lengths = torch.tensor([5, 9])
            out = model(inputs, None, sid, sid, lengths)
            single = model(inputs[:1, :5], None, sid[:1], sid[:1])
            torch.testing.assert_close(out[:1, :5], single)
            out.square().mean().backward()
            self.assertGreater(model.factor_head.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.head.weight.grad[3:].abs().sum().item(), 0)
            self.assertIsNone(model.pose_basis.grad)
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
                path = Path(d) / 'checkpoint.pt'
                save_checkpoint(path, model, {}, config, {})
                restored, _, _, _ = load_checkpoint(path, 'cpu')
                restored.eval()
                torch.testing.assert_close(out, restored(inputs, None, sid, sid, lengths))

    def test_residual_bound_and_wrist_independence(self):
        config = dict(hidden=8, sessions=['s'], conditions=[1], feature_dim=4,
                      metadata=False, pose_components=16, pose_residual_scale=.2)
        model = SharedGRU(config).eval()
        with torch.no_grad():
            model.head.bias.fill_(100)
        sid = torch.zeros(1, dtype=torch.long)
        out = model(torch.randn(1, 5, 4), None, sid, sid)
        self.assertLessEqual(out[:, :, 3:].abs().max().item(), .200001)
        self.assertGreater(out[:, :, :3].min().item(), 90)


if __name__ == '__main__':
    unittest.main()

