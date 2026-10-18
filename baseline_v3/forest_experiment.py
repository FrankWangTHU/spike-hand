"""Shared multi-output tree ensemble as a non-recurrent comparator."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import ExtraTreesRegressor
from runtime.data import session_paths, trial_paths, read_trial, transform_input, split_trials, to_local, to_global, load_names, event_features
from metrics import score_trial, average


def main():
    output = Path('experiments/shared_forest')
    output.mkdir(parents=True, exist_ok=True)
    sessions = session_paths('../player_data/Data_heldin', 'heldin')
    records, conditions = [], set()
    for session in sessions:
        items = [transform_input(read_trial(p, True), 'raw') for p in trial_paths(session, 'heldin')]
        tr, va = split_trials(items, .2, 42)
        x = np.concatenate([items[i]['x'] for i in tr])
        y = np.concatenate([to_local(items[i]['y'], 0) for i in tr])
        stat = dict(xmean=x.mean(0), xstd=np.maximum(x.std(0), 1e-3), ymean=y.mean(0), ystd=np.maximum(y.std(0), 1e-3))
        records.append((session.name, items, tr, va, stat, load_names(session)))
        conditions.update(t['condition'] for t in items)
    conditions = sorted(conditions)
    train_x, train_y, validation = [], [], []
    for sid, (session, items, tr, va, stat, names) in enumerate(records):
        for i, item in enumerate(items):
            z = np.clip((item['x'] - stat['xmean']) / stat['xstd'], -10, 10)
            identity = np.zeros(len(records) + len(conditions), dtype='float32')
            identity[sid] = 1
            identity[len(records) + conditions.index(item['condition'])] = 1
            x = np.concatenate([gaussian_filter1d(z, 3, axis=0), gaussian_filter1d(z, 12, axis=0),
                                event_features(item), np.tile(identity, (len(z), 1))], axis=1)
            if i in tr:
                train_x.append(x[::3])
                train_y.append(((to_local(item['y'], 0) - stat['ymean']) / stat['ystd'])[::3])
            else:
                validation.append((session, item, x, stat, names))
    x, y = np.concatenate(train_x), np.concatenate(train_y)
    print('Training rows:', x.shape, flush=True)
    model = ExtraTreesRegressor(n_estimators=96, max_depth=24, min_samples_leaf=10, max_features=.5,
                                n_jobs=4, random_state=42)
    model.fit(x, y)
    by_session, predictions = {}, {}
    for session, item, x, stat, names in validation:
        p = to_global((model.predict(x) * stat['ystd'] + stat['ymean']).astype('float32'), 0)
        predictions[f'{session}_{Path(item["path"]).stem}'] = p
        by_session.setdefault(session, []).append(score_trial(item['y'], p, item['times'], names))
    rows = [{'session': s, **average(v)} for s, v in by_session.items()]
    mean = average([{k: v for k, v in r.items() if k != 'session'} for r in rows])
    np.savez_compressed(output / 'validation_predictions.npz', **predictions)
    (output / 'validation.json').write_text(json.dumps({'mean': mean, 'sessions': rows, 'parameters': model.get_params()}, indent=2), encoding='utf-8')
    print(mean, flush=True)


if __name__ == '__main__':
    main()
