"""Evaluate public condition/event-aligned trajectory priors on fixed held-in splits."""
import sys
import json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline_v1'))
from runtime.data import session_paths, trial_paths, read_trial, load_names, split_trials
from metrics import score_trial, average


def phase(item):
    n = len(item['x'])
    a, b = item['onset'], item['reward']
    times = item['times']
    # Public trial metadata: onset and reward delimit the action interval.
    t = (times - times[a]).astype(np.float64) / 1e9
    duration = (times[b] - times[a]) / 1e9
    return np.where(t < 0, t, np.where(t <= duration, t / duration, 1 + t - duration))


def resample(values, old, new):
    return np.column_stack([np.interp(new, old, values[:, j]) for j in range(values.shape[1])]).astype('float32')


def read_all(seed=42):
    root = Path(__file__).resolve().parents[1] / 'player_data/Data_heldin'
    sessions = []
    grid = np.linspace(-0.6, 1.6, 256)
    for session in session_paths(root, 'heldin'):
        items = []
        for p in trial_paths(session, 'heldin'):
            item = read_trial(p, True)
            with np.load(p) as d:
                item.update(onset=int(d['Ton_frame_index']), reward=int(d['reward_frame_index']))
            item['phase'] = phase(item)
            item['aligned'] = resample(item['y'], item['phase'], grid)
            items.append(item)
        tr, va = split_trials(items, 0.2, seed)
        sessions.append({'name': session.name, 'names': load_names(session), 'items': items, 'tr': tr, 'va': va})
    return sessions, grid


def main():
    sessions, grid = read_all()
    conditions = sorted({t['condition'] for s in sessions for t in s['items']})
    global_templates = {c: np.mean([s['items'][i]['aligned'] for s in sessions for i in s['tr'] if s['items'][i]['condition'] == c], axis=0) for c in conditions}
    outputs = {}
    for blend in [0., 0.25, 0.5, 0.75, 1.]:
        rows = []
        for s in sessions:
            scores = []
            for i in s['va']:
                item = s['items'][i]
                matches = [s['items'][j]['aligned'] for j in s['tr'] if s['items'][j]['condition'] == item['condition']]
                local = np.mean(matches, axis=0) if matches else global_templates[item['condition']]
                template = blend * local + (1 - blend) * global_templates[item['condition']]
                p = resample(template, grid, item['phase'])
                scores.append(score_trial(item['y'], p, item['times'], s['names']))
            rows.append({'session': s['name'], **average(scores)})
        result = average([{k: v for k, v in r.items() if k != 'session'} for r in rows])
        outputs[str(blend)] = {'mean': result, 'sessions': rows}
        print(blend, result, flush=True)
    out = Path(__file__).resolve().parent / 'experiments'
    out.mkdir(exist_ok=True)
    (out / 'templates.json').write_text(json.dumps(outputs, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
