import json
from pathlib import Path
import numpy as np
from experiment_templates import read_all, resample
from metrics import score_trial, average


def fit_transform(source, target):
    source, target = source.reshape(-1, 3).astype('float64'), target.reshape(-1, 3).astype('float64')
    sm, tm = source.mean(0), target.mean(0)
    x, y = source - sm, target - tm
    u, singular, vt = np.linalg.svd(x.T @ y)
    r = u @ vt
    # Camera calibration may change handedness; allow an orthogonal reflection.
    scale = singular.sum() / np.square(x).sum()
    return r * scale, tm - sm @ (r * scale)


def transform(values, a, b):
    return (values.reshape(-1, 21, 3) @ a + b).reshape(-1, 63).astype('float32')


def main():
    sessions, grid = read_all()
    local = {}
    for s in sessions:
        local[s['name']] = {c: np.mean([s['items'][i]['aligned'] for i in s['tr'] if s['items'][i]['condition'] == c], axis=0)
                            for c in sorted({s['items'][i]['condition'] for i in s['tr']})}
    reference = local[sessions[0]['name']]
    maps = {}
    for s in sessions:
        template = local[s['name']]
        common = sorted(set(reference) & set(template))
        maps[s['name']] = fit_transform(np.concatenate([template[c] for c in common]), np.concatenate([reference[c] for c in common]))
    conditions = sorted(reference)
    global_templates = {c: np.mean([transform(s['items'][i]['aligned'], *maps[s['name']]) for s in sessions for i in s['tr'] if s['items'][i]['condition'] == c], axis=0) for c in conditions}
    results = {}
    for blend in [0, 0.25, 0.5, 0.75, 1.0]:
        rows = []
        for s in sessions:
            a, b = maps[s['name']]
            scores = []
            for i in s['va']:
                t = s['items'][i]
                c = t['condition']
                g = transform(global_templates[c], np.linalg.inv(a), -b @ np.linalg.inv(a))
                template = blend * local[s['name']][c] + (1 - blend) * g
                p = resample(template, grid, t['phase'])
                scores.append(score_trial(t['y'], p, t['times'], s['names']))
            rows.append(average(scores))
        results[str(blend)] = average(rows)
        print(blend, results[str(blend)], flush=True)
    (Path('experiments') / 'aligned_templates.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
