"""Small, global-weight ensemble search on development trials only."""
import argparse
import json
from pathlib import Path
import numpy as np
from runtime.data import session_paths, trial_paths, read_trial, load_names, split_trials
from runtime.postprocess import fit_bones, postprocess
from metrics import score_trial, average


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', required=True, help='JSON mapping group name to equal-weight prediction NPZ paths')
    p.add_argument('--start', nargs='+', required=True, help='name=weight entries summing to one')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--output', required=True)
    p.add_argument('--prune-tolerance', type=float, default=0., help='Maximum total development score loss allowed to remove members')
    p.add_argument('--selection-output', help='Save chosen/pruned ensemble separately from all evaluated candidates')
    args = p.parse_args()
    groups = json.loads(Path(args.groups).read_text(encoding='utf-8'))
    names = list(groups)
    predictions = []
    for name in names:
        members = [dict(np.load(path, allow_pickle=False)) for path in groups[name]]
        if any(set(m) != set(members[0]) for m in members):
            raise ValueError('Prediction trial sets differ')
        predictions.append({key: np.mean([m[key] for m in members], axis=0) for key in members[0]})
    prepared, expected = [], set()
    for session in session_paths('../player_data/Data_heldin', 'heldin'):
        keypoints = load_names(session)
        items = [read_trial(path, True) for path in trial_paths(session, 'heldin')]
        tr, va = split_trials(items, .2, 42)
        bones = fit_bones([items[i] for i in tr], keypoints)
        trials = []
        for i in va:
            t = items[i]
            key = f'{session.name}_{Path(t["path"]).stem}'
            expected.add(key)
            trials.append((t['y'], t['times'], np.stack([m[key] for m in predictions])))
        prepared.append((session.name, keypoints, bones, trials))
    if any(set(m) != expected for m in predictions):
        raise ValueError('Predictions do not match split 42 exactly')
    records, cache = [], {}
    def evaluate(weights):
        key = tuple(np.round(weights, 8))
        if key in cache:
            return cache[key]
        rows = []
        for sid, keypoints, bones, trials in prepared:
            scores = []
            for y, times, ps in trials:
                pred = np.einsum('m,mtc->tc', weights, ps, optimize=False).astype('float32')
                pred = postprocess(pred, bones, keypoints, 1., .5)
                scores.append(score_trial(y, pred, times, keypoints))
            rows.append({'session': sid, **average(scores)})
        result = {'sigma': 1., 'strength': .5, 'condition_bones': False,
                  'group_weights': dict(zip(names, weights.tolist())), 'groups': groups,
                  'mean': average([{k: v for k, v in row.items() if k != 'session'} for row in rows]), 'sessions': rows}
        records.append(result)
        cache[key] = result
        return result
    weights = np.zeros(len(names))
    for entry in args.start:
        name, value = entry.split('=')
        weights[names.index(name)] = float(value)
    if (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid starting weights')
    best = evaluate(weights)
    print('START', best['mean']['hrfs_local'], best['group_weights'], flush=True)
    for iteration in range(args.rounds):
        candidate = best
        for i in range(len(names)):
            for alpha in (.1, .2, .3):
                trial = (1 - alpha) * weights
                trial[i] += alpha
                result = evaluate(trial)
                if result['mean']['hrfs_local'] > candidate['mean']['hrfs_local']:
                    candidate = result
        if candidate['mean']['hrfs_local'] - best['mean']['hrfs_local'] < .02:
            break
        best = candidate
        weights = np.array([best['group_weights'][name] for name in names])
        print('ROUND', iteration + 1, best['mean']['hrfs_local'], best['group_weights'], flush=True)
    best = max(records, key=lambda row: row['mean']['hrfs_local'])
    if args.prune_tolerance > 0:
        floor = best['mean']['hrfs_local'] - args.prune_tolerance
        while True:
            weights = np.array([best['group_weights'][name] for name in names])
            if np.count_nonzero(weights) <= 1:
                break
            choices = []
            for i in np.flatnonzero(weights):
                trial = weights.copy()
                trial[i] = 0
                trial /= trial.sum()
                candidate = evaluate(trial)
                if candidate['mean']['hrfs_local'] >= floor:
                    choices.append(candidate)
            if not choices:
                break
            best = max(choices, key=lambda row: row['mean']['hrfs_local'])
            print('PRUNED', best['mean']['hrfs_local'], best['group_weights'], flush=True)
    if args.selection_output:
        Path(args.selection_output).write_text(json.dumps([best], indent=2), encoding='utf-8')
    records.sort(key=lambda row: row['mean']['hrfs_local'], reverse=True)
    Path(args.output).write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('Evaluated', len(records), 'development-only global weight settings', flush=True)


if __name__ == '__main__':
    main()
