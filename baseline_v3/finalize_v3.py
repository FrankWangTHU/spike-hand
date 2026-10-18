"""Execute a frozen selection without tuning on the confirmation split."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', default='reports/selection.json')
    parser.add_argument('--stage', choices=['development', 'confirm', 'refit', 'package'], required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.selection).read_text(encoding='utf-8'))
    for filename, expected in plan.get('code_sha256', {}).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen training/inference code changed: ' + filename)
    members = plan['members']
    for member in members:
        for filename in ('checkpoint.pt', 'config.json'):
            actual = hashlib.sha256((Path(member['source']) / filename).read_bytes()).hexdigest()
            if actual != member['source_sha256'][filename]:
                raise ValueError('Selected development source changed: ' + member['source'])
    def run(script, *options):
        command = [sys.executable, script, *map(str, options)]
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, check=True)
    weights = [m['weight'] for m in members]
    if args.stage in ('confirm', 'refit'):
        selected = [m['source'] for m in members if m['new']]
        root = 'experiments/confirm_v3' if args.stage == 'confirm' else 'final'
        run('retrain_selected.py', '--models', *selected, '--mode', args.stage,
            '--split-seed', 43, '--output-root', root)
    if args.stage in ('development', 'confirm'):
        field = 'development_predictions' if args.stage == 'development' else 'confirmation'
        paths = [m[field] if field == 'development_predictions' else str(Path(m[field]) / 'validation_predictions.npz') for m in members]
        run('evaluate_predictions.py', '--predictions', *paths, '--weights', *weights,
            '--split-seed', 42 if args.stage == 'development' else 43,
            '--sigma', plan['sigma'], '--strength', plan['bone_strength'],
            '--output', 'reports/' + args.stage + '.json')
    if args.stage == 'package':
        run('package_submission.py', '--members', *[m['refit'] for m in members], '--weights', *weights,
            '--sigma', plan['sigma'], '--bone-strength', plan['bone_strength'])


if __name__ == '__main__':
    main()
