"""Local finite-data HRFS reproduction, not the private official scorer."""
import numpy as np
from runtime.data import skeleton


def score_trial(y, pred, times, names):
    if y.shape != pred.shape or y.ndim != 2 or y.shape[1] != 63 or len(y) == 0:
        raise ValueError('Expected matching nonempty (T,63) trajectories')
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Local scorer only supports finite trajectories')
    y, p = y.reshape(-1, 21, 3).astype('float64'), pred.reshape(-1, 21, 3).astype('float64')
    wrist, edges = skeleton(names)
    fingers = [j for j in range(21) if j != wrist]
    local_y = y[:, fingers] - y[:, wrist, None]
    local_p = p[:, fingers] - p[:, wrist, None]
    ew = np.linalg.norm(p[:, wrist] - y[:, wrist], axis=-1).mean()
    ep = np.linalg.norm(local_p - local_y, axis=-1).mean()
    a, b = edges.T
    ly = np.linalg.norm(y[:, a] - y[:, b], axis=-1)
    lp = np.linalg.norm(p[:, a] - p[:, b], axis=-1)
    eb = np.mean(np.abs(lp - ly) / np.maximum(ly, 1e-8))
    scores = {'wrist': float(100 * np.exp(-ew / 20)), 'pose': float(100 * np.exp(-ep / 15)),
              'bone': float(100 * np.exp(-5 * eb))}
    weights = {'wrist': 0.25, 'pose': 0.35, 'bone': 0.1}
    if len(y) >= 2:
        dt = np.diff(np.asarray(times, dtype=np.int64)).astype('float64') / 1e9
        dt = np.median(dt[np.isfinite(dt) & (dt > 0)]) if np.any(dt > 0) else 1 / 30
        vy = (np.diff(local_y, axis=0) / dt).reshape(len(y) - 1, -1)
        vp = (np.diff(local_p, axis=0) / dt).reshape(len(y) - 1, -1)
        ev = np.sqrt(np.mean((vp - vy) ** 2))
        correlations = np.array([], dtype='float64')
        if len(vy) >= 2:
            u, v = vy - vy.mean(0), vp - vp.mean(0)
            su, sv = vy.std(0), vp.std(0)
            active = (su > 1e-10) & (sv > 1e-10)
            numerator = (u[:, active] * v[:, active]).sum(0)
            denominator = np.sqrt((u[:, active] ** 2).sum(0) * (v[:, active] ** 2).sum(0))
            correlations = np.clip(numerator / denominator, -1, 1)
            identical_constant = (su <= 1e-10) & (sv <= 1e-10) & np.all(np.abs(vy - vp) <= 1e-10, axis=0)
            correlations = np.concatenate([correlations, np.ones(int(identical_constant.sum()))])
        corr = np.mean(correlations) if len(correlations) else 0.0
        scores['motion'] = float(50 * np.exp(-ev / 5) + 50 * max(0, corr))
        weights['motion'] = 0.2
    scores['hrfs_local'] = float(sum(scores[k] * v for k, v in weights.items()) / sum(weights.values()))
    scores['mpjpe'] = float(np.linalg.norm(p - y, axis=-1).mean())
    scores['rmse'] = float(np.sqrt(np.mean((p - y) ** 2)))
    return scores


def average(rows):
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in rows[0]}
