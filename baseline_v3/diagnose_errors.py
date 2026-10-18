"""Descriptive development diagnostics; never changes the frozen selection."""
import json
from pathlib import Path
import numpy as np
from runtime.data import session_paths, trial_paths, read_trial, load_names, split_trials
from runtime.postprocess import fit_bones, postprocess
from metrics import score_trial, average


def main():
    plan = json.loads(Path('reports/selection.json').read_text(encoding='utf-8'))
    members = [dict(np.load(m['development_predictions'], allow_pickle=False)) for m in plan['members']]
    conditions, fingers, phases = {}, {}, {}
    for session in session_paths('../player_data/Data_heldin', 'heldin'):
        names = load_names(session)
        items = [read_trial(p, True) for p in trial_paths(session, 'heldin')]
        tr, va = split_trials(items, .2, 42)
        bones = fit_bones([items[i] for i in tr], names)
        finger_errors, phase_errors = {}, {}
        for i in va:
            t = items[i]
            key = f'{session.name}_{Path(t["path"]).stem}'
            p = sum(m['weight'] * predictions[key] for m, predictions in zip(plan['members'], members))
            p = postprocess(p, bones, names, plan['sigma'], plan['bone_strength'])
            conditions.setdefault(t['condition'], {}).setdefault(session.name, []).append(score_trial(t['y'], p, t['times'], names))
            truth, pred = t['y'].reshape(-1, 21, 3), p.reshape(-1, 21, 3)
            error = np.linalg.norm((pred[:, 1:] - pred[:, :1]) - (truth[:, 1:] - truth[:, :1]), axis=-1)
            for j, finger in enumerate(('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')):
                finger_errors.setdefault(finger, []).append(float(error[:, 4*j:4*j+4].mean()))
            wrist = np.linalg.norm(pred[:, 0] - truth[:, 0], axis=-1)
            for phase, section in [('before_onset', slice(0, t['onset'])), ('action', slice(t['onset'], t['reward']+1)), ('after_reward', slice(t['reward']+1, None))]:
                if len(wrist[section]):
                    phase_errors.setdefault(phase, []).append({'wrist_error': float(wrist[section].mean()), 'relative_pose_error': float(error[section].mean())})
        for key, values in finger_errors.items():
            fingers.setdefault(key, []).append(float(np.mean(values)))
        for key, values in phase_errors.items():
            phases.setdefault(key, []).append(average(values))
    rows = [{'condition': c, 'sessions': len(s), **average([average(v) for v in s.values()])} for c, s in conditions.items()]
    rows.sort(key=lambda r: r['hrfs_local'])
    report = {'split_seed': 42, 'aggregation': 'Equal trials within each session, then equal sessions; original coordinate units.',
              'conditions': rows, 'relative_finger_error': {k: float(np.mean(v)) for k, v in fingers.items()},
              'phase_error': {k: average(v) for k, v in phases.items()},
              'purpose': 'Descriptive analysis of the already selected development model; not further tuning.'}
    Path('reports/error_diagnosis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
