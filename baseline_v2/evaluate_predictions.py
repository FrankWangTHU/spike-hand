import argparse
import json
from pathlib import Path
import numpy as np
from runtime.data import session_paths, trial_paths, read_trial, load_names, split_trials
from runtime.postprocess import fit_bones, postprocess
from metrics import score_trial, average


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', nargs='+', required=True)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', nargs='+', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--strength', type=float)
    args = parser.parse_args()
    predictions = [dict(np.load(p, allow_pickle=False)) for p in args.predictions]
    weights = np.asarray(args.weights if args.weights else [1 / len(predictions)] * len(predictions))
    if len(weights) != len(predictions) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        parser.error('Invalid weights')
    sessions = []
    for session in session_paths('../player_data/Data_heldin', 'heldin'):
        names = load_names(session)
        items = [read_trial(p, True) for p in trial_paths(session, 'heldin')]
        tr, va = split_trials(items, .2, args.split_seed)
        bones = fit_bones([items[i] for i in tr], names)
        selected = []
        for i in va:
            t = items[i]
            key = f'{session.name}_{Path(t["path"]).stem}'
            pred = sum(w * p[key] for w, p in zip(weights, predictions))
            selected.append((t, pred))
        sessions.append((session.name, names, bones, selected))
    results = []
    for sigma in [args.sigma] if args.sigma is not None else [0, 1, 2, 3]:
        for strength in [args.strength] if args.strength is not None else [0, .5, 1]:
            rows = []
            for sid, names, bones, selected in sessions:
                scores = [score_trial(t['y'], postprocess(pred, bones, names, sigma, strength), t['times'], names) for t, pred in selected]
                rows.append({'session': sid, **average(scores)})
            mean = average([{k: v for k, v in row.items() if k != 'session'} for row in rows])
            results.append({'sigma': sigma, 'strength': strength, 'mean': mean, 'sessions': rows})
            print(sigma, strength, mean['hrfs_local'], flush=True)
    results.sort(key=lambda row: row['mean']['hrfs_local'], reverse=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('BEST', {k: v for k, v in results[0].items() if k != 'sessions'}, flush=True)


if __name__ == '__main__':
    main()
