import numpy as np
from scipy.ndimage import gaussian_filter1d
from .data import skeleton


def fit_bones(items, names):
    _, edges = skeleton(names)
    y = np.concatenate([t['y'] for t in items]).reshape(-1, 21, 3)
    return np.median(np.linalg.norm(y[:, edges[:, 0]] - y[:, edges[:, 1]], axis=-1), axis=0).astype('float32')


def postprocess(pred, bones, names, sigma=1.0, strength=0.5):
    p = gaussian_filter1d(pred, sigma=sigma, axis=0, mode='nearest') if sigma > 0 else pred.copy()
    p = p.reshape(-1, 21, 3)
    original = p.copy()
    _, edges = skeleton(names)
    for i, (parent, child) in enumerate(edges):
        vector = original[:, child] - original[:, parent]
        length = np.linalg.norm(vector, axis=-1, keepdims=True)
        target = (1 - strength) * length + strength * bones[i]
        p[:, child] = p[:, parent] + vector * target / np.maximum(length, 1e-6)
    return p.reshape(-1, 63).astype('float32')
