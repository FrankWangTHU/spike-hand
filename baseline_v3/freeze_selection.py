"""Freeze development-selected models before any v3 confirmation evaluation."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', required=True)
    parser.add_argument('--output', default='reports/selection.json')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Do not overwrite a frozen selection: ' + str(output))
    selected = json.loads(Path(args.ensemble).read_text(encoding='utf-8'))[0]
    old_paths = {'shrink_adapter': ('seed42', 'confirm_seed43'),
                 'shrink_seed7': ('seed7', 'confirm_seed43_seed7'),
                 'shared_no_adapter': ('no_adapter', 'confirm_seed43_noadapter')}
    members = []
    for group, weight in selected['group_weights'].items():
        if weight <= 0:
            continue
        paths = selected['groups'][group]
        for predictions in paths:
            source = Path(predictions).parent
            if source.name.endswith('_fp32'):
                source = source.with_name(source.name[:-5])
            settings = json.loads((source / 'config.json').read_text(encoding='utf-8'))
            if settings['args'].get('init_checkpoint') or settings['args'].get('refit'):
                raise ValueError('This pipeline expects from-scratch development sources')
            if group == 'v2':
                full, confirm = old_paths[source.name]
                refit = '../baseline_v2/final/' + full
                confirmation = '../baseline_v2/experiments/' + confirm
            else:
                refit, confirmation = 'final/' + source.name, 'experiments/confirm_v3/' + source.name
            members.append({'group': group, 'source': str(source), 'development_predictions': predictions,
                            'weight': weight / len(paths), 'new': group != 'v2', 'refit': refit,
                            'confirmation': confirmation, 'selected_epoch': settings['best_epoch'],
                            'source_sha256': {f: hashlib.sha256((source / f).read_bytes()).hexdigest() for f in ('checkpoint.pt', 'config.json')}})
    if not math.isclose(sum(m['weight'] for m in members), 1., abs_tol=1e-8):
        raise ValueError('Invalid total weight')
    files = ['train.py', 'training_utils.py', 'metrics.py', 'infer.py', 'package_submission.py',
             'retrain_selected.py', 'evaluate_predictions.py', 'finalize_v3.py']
    files += [str(p) for p in sorted(Path('runtime').glob('*.py'))]
    plan = {'frozen_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'development_source': args.ensemble, 'development_score': selected['mean'],
            'development_split_seed': 42, 'confirmation_split_seed': 43,
            'sigma': selected['sigma'], 'bone_strength': selected['strength'], 'members': members,
            'code_sha256': {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
            'protocol': 'No model, epoch, preprocessing or weight tuning on the alternate confirmation split.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print('FROZEN', selected['mean']['hrfs_local'], [(m['source'], m['selected_epoch'], m['weight']) for m in members], flush=True)


if __name__ == '__main__':
    main()
