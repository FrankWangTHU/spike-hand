"""Ablation of training-only hand-pose subspace projection."""
import argparse
import json
from pathlib import Path
import numpy as np
from runtime.data import session_paths, trial_paths, read_trial, load_names, split_trials, to_local, to_global
from runtime.postprocess import fit_bones, postprocess
from metrics import score_trial, average


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    predictions = [dict(np.load(path, allow_pickle=False)) for path in args.predictions]
    settings = [(60, 0.)] + [(rank, blend) for rank in (6, 12, 24) for blend in (.5, 1.)]
    records = {str(s): [] for s in settings}
    for session in session_paths('../player_data/Data_heldin', 'heldin'):
        names = load_names(session)
        items = [read_trial(path, True) for path in trial_paths(session, 'heldin')]
        tr, va = split_trials(items, .2, 42)
        train = [items[i] for i in tr]
        bones = fit_bones(train, names)
        y = np.concatenate([to_local(t['y'], 0)[:, 3:] for t in train]).astype('float64')
        mean = y.mean(0)
        _, eigen = np.linalg.eigh((y - mean).T @ (y - mean))
        for rank, blend in settings:
            basis = eigen[:, -rank:]
            rows = []
            for i in va:
                t = items[i]
                key = f'{session.name}_{Path(t["path"]).stem}'
                pred = np.mean([v[key] for v in predictions], axis=0)
                local = to_local(pred, 0)
                local[:, 3:] += blend * ((local[:, 3:] - mean) @ basis @ basis.T + mean - local[:, 3:])
                pred = postprocess(to_global(local, 0), bones, names, 1., .5)
                rows.append(score_trial(t['y'], pred, t['times'], names))
            records[str((rank, blend))].append(average(rows))
    results = [{'setting': k, 'mean': average(v)} for k, v in records.items()]
    results.sort(key=lambda x: x['mean']['hrfs_local'], reverse=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    for r in results:
        print(r, flush=True)


if __name__ == '__main__':
    main()
