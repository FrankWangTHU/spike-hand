"""Train-only condition trajectory templates, aligned by public trial events."""
import json
from pathlib import Path
import numpy as np
from runtime.data import session_paths, trial_paths, read_trial, load_names, split_trials, event_features
from metrics import score_trial, average


def main():
    root = Path('experiments/phase_template')
    root.mkdir(exist_ok=True)
    predictions, rows = {}, []
    for session in session_paths('../player_data/Data_heldin', 'heldin'):
        names = load_names(session)
        items = [read_trial(p, True) for p in trial_paths(session, 'heldin')]
        tr, va = split_trials(items, .2, 42)
        scores = []
        for i in va:
            target = items[i]
            phase = event_features(target)[:, 2]
            members = [items[j] for j in tr if items[j]['condition'] == target['condition']]
            aligned = []
            for t in members:
                source_phase = event_features(t)[:, 2]
                aligned.append(np.stack([np.interp(phase, source_phase, t['y'][:, j]) for j in range(63)], axis=1))
            pred = np.mean(aligned, axis=0).astype('float32')
            predictions[f'{session.name}_{Path(target["path"]).stem}'] = pred
            scores.append(score_trial(target['y'], pred, target['times'], names))
        rows.append({'session': session.name, **average(scores)})
    mean = average([{k: v for k, v in r.items() if k != 'session'} for r in rows])
    np.savez_compressed(root / 'validation_predictions.npz', **predictions)
    (root / 'validation.json').write_text(json.dumps({'mean': mean, 'sessions': rows}, indent=2), encoding='utf-8')
    print(mean)


if __name__ == '__main__':
    main()
