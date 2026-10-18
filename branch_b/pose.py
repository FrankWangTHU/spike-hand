"""PCA of training-only, session-standardized wrist-relative finger coordinates."""
import numpy as np


def fit_pose_pca(poses, components):
    x = np.concatenate(poses).astype('float64')
    if x.ndim != 2 or x.shape[1] != 60 or not np.isfinite(x).all():
        raise ValueError('Expected finite (frames, 60) training poses')
    mean = x.mean(0)
    centered = x - mean
    values, vectors = np.linalg.eigh(centered.T @ centered / max(len(x)-1, 1))
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0)
    basis = vectors[:, order[:components]].T
    # Make signs deterministic across equivalent eigensolver outputs.
    basis *= np.where(basis[np.arange(components), np.abs(basis).argmax(1)] < 0, -1, 1)[:, None]
    return mean.astype('float32'), basis.astype('float32'), {
        'components': components, 'fit_frames': len(x),
        'space': 'session-standardized 60D wrist-relative pose; wrist excluded',
        'explained_variance_ratio': float(values[:components].sum()/max(values.sum(), 1e-20)),
        'residual_normalized_rms': float(np.sqrt(values[components:].sum()/60)),
    }
