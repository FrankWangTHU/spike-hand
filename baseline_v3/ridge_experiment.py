"""Trial-disjoint kernel regression comparator; all statistics fit on training trials."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from runtime.data import session_paths, trial_paths, read_trial, transform_input, split_trials, to_local, to_global, load_names, event_features
from metrics import score_trial, average


def main():
    output = Path('experiments/kernel_ridge')
    output.mkdir(parents=True, exist_ok=True)
    params = [(w, a) for w in (.5, 1., 2.) for a in (.01, .1, 1.)]
    scores = {str(p): [] for p in params}
    predictions = {str(p): {} for p in params}
    for session in session_paths('../player_data/Data_heldin', 'heldin'):
        names = load_names(session)
        items = [transform_input(read_trial(p, True), 'raw') for p in trial_paths(session, 'heldin')]
        tr, va = split_trials(items, .2, 42)
        raw = np.concatenate([items[i]['x'] for i in tr])
        mean, std = raw.mean(0), np.maximum(raw.std(0), 1e-3)
        zs = [np.clip((t['x'] - mean) / std, -10, 10) for t in items]
        smooth = np.concatenate([gaussian_filter1d(zs[i], 6, axis=0) for i in tr])
        _, vec = eigh(smooth.T @ smooth, subset_by_index=[448, 511])
        conditions = sorted({t['condition'] for t in items})
        embeddings = []
        phases = []
        for t, z in zip(items, zs):
            z = z @ vec
            g = gaussian_filter1d(z, 3, axis=0)
            idx = np.arange(len(z))
            embeddings.append(np.concatenate([g[np.clip(idx + lag, 0, len(z)-1)] for lag in (-8, 0, 8)] + [gaussian_filter1d(z, 12, axis=0)], axis=1))
            phase = event_features(t)[:, 2:3]
            cond = np.eye(len(conditions))[conditions.index(t['condition'])]
            phases.append(np.concatenate([phase, np.tile(cond, (len(z), 1))], axis=1))
        x = np.concatenate([embeddings[i][::4] for i in tr]).astype('float64')
        m = np.concatenate([phases[i][::4] for i in tr])
        y = np.concatenate([to_local(items[i]['y'], 0)[::4] for i in tr])
        ym = y.mean(0)
        d = cdist(x, x, 'sqeuclidean')
        scale = max(float(np.median(d)), 1e-6)
        md = cdist(m, m, 'sqeuclidean')
        vx = np.concatenate([embeddings[i] for i in va])
        vm = np.concatenate([phases[i] for i in va])
        vd, vmd = cdist(vx, x, 'sqeuclidean') / scale, cdist(vm, m, 'sqeuclidean')
        for width in (.5, 1., 2.):
            kernel = np.exp(-d / (scale * width) - md)
            eigen, basis = eigh(kernel, check_finite=False)
            projection = basis.T @ (y - ym)
            cross = np.exp(-vd / width - vmd) @ basis
            for alpha in (.01, .1, 1.):
                pred = to_global((cross @ (projection / (eigen[:, None] + alpha)) + ym).astype('float32'), 0)
                rows, start = [], 0
                key = str((width, alpha))
                for i in va:
                    t = items[i]
                    p = pred[start:start + len(t['y'])]
                    start += len(p)
                    rows.append(score_trial(t['y'], p, t['times'], names))
                    predictions[key][f'{session.name}_{Path(t["path"]).stem}'] = p
                scores[key].append(average(rows))
        print(session.name, {k: round(v[-1]['hrfs_local'], 2) for k, v in scores.items()}, flush=True)
    results = [{'parameter': k, 'mean': average(v)} for k, v in scores.items()]
    results.sort(key=lambda r: r['mean']['hrfs_local'], reverse=True)
    (output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'validation_predictions.npz', **predictions[results[0]['parameter']])
    print('BEST', results[0], flush=True)


if __name__ == '__main__':
    main()
