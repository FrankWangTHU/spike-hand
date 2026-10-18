from pathlib import Path
import re
import numpy as np


def session_paths(root, partition):
    paths = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / partition).is_dir())
    if not paths:
        raise ValueError(f'No {partition} sessions under {root}')
    return paths


def trial_paths(session, partition):
    paths = list((session / partition).glob('trial_*.npz'))
    if any(re.fullmatch(r'trial_\d+\.npz', p.name) is None for p in paths):
        raise ValueError(f'Invalid trial filename in {session}')
    paths.sort(key=lambda p: int(p.stem.split('_')[-1]))
    if not paths or [int(p.stem.split('_')[-1]) for p in paths] != list(range(1, len(paths) + 1)):
        raise ValueError(f'Expected consecutive trials 1..N in {session}')
    return paths


def load_names(session):
    values = np.load(session / 'keypoint_names.npy', allow_pickle=False)
    names = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in values.reshape(-1)]
    expected = ['Wrist'] + [f'{finger}_{i}' for finger in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'] for i in range(1, 5)]
    if len(names) != 21 or set(names) != set(expected):
        raise ValueError(f'Unexpected keypoint names: {session}')
    return names


def skeleton(names):
    wrist = names.index('Wrist')
    edges = []
    for finger in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
        chain = [wrist] + [names.index(f'{finger}_{i}') for i in range(1, 5)]
        edges.extend(zip(chain[:-1], chain[1:]))
    return wrist, np.asarray(edges)


def read_trial(path, labels=False):
    with np.load(path, allow_pickle=False) as d:
        tx, sbp = d['tx'].astype('float32'), d['sbp'].astype('float32')
        times = d['frame_timestamps_ns']
        if tx.ndim != 2 or tx.shape[1] != 256 or tx.shape != sbp.shape or len(tx) == 0:
            raise ValueError(f'Invalid TX/SBP shape: {path}')
        if times.shape != (len(tx),) or times.dtype != np.int64 or np.any(np.diff(times) <= 0):
            raise ValueError(f'Invalid timestamps: {path}')
        if not np.isfinite(tx).all() or not np.isfinite(sbp).all() or min(tx.min(), sbp.min()) < 0:
            raise ValueError(f'Invalid neural values: {path}')
        item = {'x': np.concatenate([np.log1p(tx), np.log1p(sbp)], axis=1), 'times': times,
                'condition': str(d['condition'].item()), 'path': str(path)}
        item['onset'], item['reward'] = int(d['Ton_frame_index']), int(d['reward_frame_index'])
        if not 0 <= item['onset'] < item['reward'] < len(tx):
            raise ValueError(f'Invalid event indices: {path}')
        if labels:
            y = d['keypoint'].astype('float32')
            if y.shape != (len(tx), 63) or not np.isfinite(y).all():
                raise ValueError(f'Invalid keypoints: {path}')
            item['y'] = y
    return item


def split_trials(items, fraction, seed):
    rng = np.random.default_rng(seed)
    val = []
    for condition in sorted({t['condition'] for t in items}):
        ids = [i for i, t in enumerate(items) if t['condition'] == condition]
        rng.shuffle(ids)
        count = min(len(ids) - 1, max(1, round(len(ids) * fraction))) if len(ids) > 1 else 0
        val.extend(ids[:count])
    if not val:
        raise ValueError('Not enough repeated conditions for trial-level validation')
    return [i for i in range(len(items)) if i not in val], sorted(val)


def to_local(y, wrist):
    p = y.reshape(-1, 21, 3).copy()
    root = p[:, wrist].copy()
    p -= root[:, None]
    p[:, wrist] = root
    return p.reshape(-1, 63)


def to_global(y, wrist):
    p = y.reshape(-1, 21, 3).copy()
    root = p[:, wrist].copy()
    p[:, wrist] = 0
    p += root[:, None]
    return p.reshape(-1, 63)


def model_target(y, stat):
    local = to_local(y, 0)
    if 'rotation' in stat:
        local = (local.reshape(-1, 21, 3) @ stat['rotation']).reshape(-1, 63)
    return local.astype('float32')


def decode_target(local, stat):
    if 'rotation' in stat:
        local = (local.reshape(-1, 21, 3) @ stat['rotation'].T).reshape(-1, 63)
    return to_global(local, 0).astype('float32')


def features(x, mean, std, mode='ema', projection=None):
    """Trial-local temporal features; only ema mode uses past neural frames alone."""
    z = np.clip((x - mean) / std, -10, 10).astype('float32')
    if projection is not None:
        z = z @ projection
    if mode in ('multiscale', 'wide'):
        from scipy.ndimage import gaussian_filter1d
        if mode == 'wide':
            return np.concatenate([z] + [gaussian_filter1d(z, s, axis=0, mode='nearest') for s in (3, 9, 27)], axis=1).astype('float32')
        g2 = gaussian_filter1d(z, 2, axis=0, mode='nearest')
        g6 = gaussian_filter1d(z, 6, axis=0, mode='nearest')
        slope = gaussian_filter1d(z, 2, axis=0, order=1, mode='nearest')
        return np.concatenate([z, g2, g6, slope], axis=1).astype('float32')
    if mode == 'lagged':
        from scipy.ndimage import gaussian_filter1d
        g = gaussian_filter1d(z, 2, axis=0, mode='nearest')
        indices = np.arange(len(z))
        return np.concatenate([g[np.clip(indices + lag, 0, len(z) - 1)] for lag in [-8, -4, 0, 4, 8]], axis=1)
    if mode != 'ema':
        raise ValueError(f'Unknown feature mode: {mode}')
    smooth = np.empty_like(z)
    smooth[0] = z[0]
    for i in range(1, len(z)):
        smooth[i] = 0.35 * z[i] + 0.65 * smooth[i - 1]
    return np.concatenate([z, smooth], axis=1)


def event_features(item):
    times, a, b = item['times'], item['onset'], item['reward']
    t = (times - times[a]).astype(np.float64) / 1e9
    duration = (times[b] - times[a]) / 1e9
    phase = np.where(t < 0, t, np.where(t <= duration, t / duration, 1 + t - duration))
    value = [t / 4, (t - duration) / 4, phase, np.full(len(t), duration / 4),
             (t >= 0).astype(float), (t >= duration).astype(float)]
    for k in (1, 2, 4):
        value.extend([np.sin(np.pi * k * phase), np.cos(np.pi * k * phase)])
    return np.stack(value, axis=1).astype('float32')


def model_events(item, stat, config):
    events = event_features(item)
    if config.get('recording_clock', False):
        # Subtract int64 nanoseconds before any floating-point conversion.
        elapsed = (np.int64(item['times'][0]) - stat['clock_origin']).astype(np.float64) / 1e9
        t = np.clip(elapsed / stat['clock_scale'] - 1., -2., 2.)
        clock = np.array([t, t*t, np.sin(np.pi*t), np.cos(np.pi*t)], dtype='float32')
        events = np.concatenate([events, np.broadcast_to(clock, (len(events), 4))], axis=1)
    return events


def transform_input(item, kind):
    if kind == 'raw':
        item['x'] = np.expm1(item['x']).astype('float32')
    elif kind == 'sqrt_tx':
        item['x'][:, :256] = np.sqrt(np.maximum(np.expm1(item['x'][:, :256]), 0))
    elif kind != 'log':
        raise ValueError(f'Unknown transform: {kind}')
    return item
