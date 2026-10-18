"""Separate global wrist ensemble from the frozen relative-pose ensemble."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from runtime.data import session_paths, trial_paths, read_trial, split_trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    record = json.loads(Path(args.ensemble).read_text(encoding='utf-8'))[0]
    groups = {k: record['groups'][k] for k, v in record['group_weights'].items() if v > 0}
    names = list(groups)
    predictions = []
    for name, paths in groups.items():
        members = [dict(np.load(p, allow_pickle=False)) for p in paths]
        predictions.append({k: np.mean([m[k][:, :3] for m in members], axis=0) for k in members[0]})
    ys, ps, trial_weights, counts, expected = [], [], [], [], set()
    for session in session_paths('../player_data/Data_heldin', 'heldin'):
        items = [read_trial(p, True) for p in trial_paths(session, 'heldin')]
        _, va = split_trials(items, .2, 42)
        for i in va:
            t = items[i]
            key = f'{session.name}_{Path(t["path"]).stem}'
            expected.add(key)
            ys.append(t['y'][:, :3])
            ps.append(np.stack([gaussian_filter1d(m[key], record['sigma'], axis=0, mode='nearest') for m in predictions]))
            counts.append(len(t['y']))
            trial_weights.append(1 / len(va))
    if any(set(m) != expected for m in predictions):
        raise ValueError('Wrong validation split')
    y = np.concatenate(ys).astype('float64')
    predictions = np.concatenate(ps, axis=1).astype('float64')
    counts = np.asarray(counts)
    starts = np.r_[0, counts.cumsum()[:-1]]
    weights = np.asarray(trial_weights) / sum(trial_weights)

    def objective(w):
        residual = np.einsum('m,mtc->tc', w, predictions, optimize=False) - y
        distance = np.sqrt((residual**2).sum(-1) + 1e-12)
        trial_distance = np.add.reduceat(distance, starts) / counts
        score = 100 * np.exp(-trial_distance / 20)
        frame_grad = np.einsum('tc,mtc->mt', residual / distance[:, None], predictions, optimize=False)
        trial_grad = np.add.reduceat(frame_grad, starts, axis=1) / counts[None]
        return -float(weights @ score), (trial_grad @ (weights * score / 20))

    initial = np.array([record['group_weights'][k] for k in names])
    result = minimize(objective, initial, jac=True, method='SLSQP', bounds=[(0, 1)]*len(names),
                      constraints=[{'type': 'eq', 'fun': lambda x: x.sum() - 1., 'jac': lambda x: np.ones_like(x)}],
                      options={'maxiter': 80, 'ftol': 1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    best = -objective(result.x)[0]
    initial_score = -objective(initial)[0]
    output = {'source_ensemble': args.ensemble, 'wrist_group_weights': dict(zip(names, result.x.tolist())),
              'pose_group_weights': record['group_weights'], 'initial_wrist': initial_score, 'optimized_wrist': best,
              'expected_hrfs_gain': (best-initial_score)*.25/.9, 'iterations': int(result.nit),
              'selection_split': 42, 'optimizer': 'SLSQP nonnegative simplex; validation-only tuning'}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
