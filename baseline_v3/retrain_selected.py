"""Freeze development-selected settings for confirmation or all-data refitting."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help='Completed development model directories')
    parser.add_argument('--mode', choices=['confirm', 'refit'], required=True)
    parser.add_argument('--split-seed', type=int, default=43)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    provenance = root / 'provenance.json'
    previous = json.loads(provenance.read_text(encoding='utf-8')) if provenance.exists() else []
    commands = []
    for member in args.models:
        source = Path(member)
        report = json.loads((source / 'config.json').read_text(encoding='utf-8'))
        settings = report['args']
        if settings.get('refit') or settings.get('init_checkpoint'):
            raise ValueError('Expected a development model trained from scratch')
        target = root / source.name
        command = [sys.executable, 'train.py', '--output', str(target), '--epochs', str(report['best_epoch']),
                   '--schedule-epochs', str(settings.get('schedule_epochs') or settings['epochs']),
                   '--split-seed', str(args.split_seed)]
        for key in ['data_dir', 'hidden', 'batch_size', 'seed', 'input_transform', 'adapter_scale', 'feature_mode',
                    'channel_dropout', 'motion_weight', 'input_components', 'mixup', 'lr']:
            if key in settings:
                command.extend(['--' + key.replace('_', '-'), str(settings[key])])
        for key in ['no_metadata', 'bidirectional', 'shared_input', 'tcn', 'dual', 'stacked_dual', 'attention', 'condition_head', 'condition_input', 'origin_head', 'recording_clock', 'canonical_target', 'score_loss', 'tf32']:
            if settings.get(key):
                command.append('--' + key.replace('_', '-'))
        command.append('--refit' if args.mode == 'refit' else '--last-epoch')
        record = {'source': str(source), 'source_checkpoint_sha256': hashlib.sha256((source / 'checkpoint.pt').read_bytes()).hexdigest(),
                  'command': command, 'mode': args.mode, 'status': 'running'}
        old = next((r for r in previous if r.get('source') == str(source)), None)
        if old and old.get('status') == 'complete' and all(old.get(k) == record[k] for k in ('source_checkpoint_sha256', 'command', 'mode')):
            saved = json.loads((target / 'config.json').read_text(encoding='utf-8'))
            complete = saved['best_epoch'] == report['best_epoch'] and (target / 'checkpoint.pt').exists()
            if args.mode == 'confirm':
                complete = complete and (target / 'validation.json').exists() and (target / 'validation_predictions.npz').exists()
            if complete:
                commands.append(old)
                provenance.write_text(json.dumps(commands + [r for r in previous if r.get('source') not in {c['source'] for c in commands}], indent=2), encoding='utf-8')
                print('Already completed matching frozen settings:', target, flush=True)
                continue
        commands.append(record)
        provenance.write_text(json.dumps(commands + [r for r in previous if r.get('source') not in {c['source'] for c in commands}], indent=2), encoding='utf-8')
        print('Running:', ' '.join(command), flush=True)
        with open(root / (source.name + '.log'), 'w', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        record['status'] = 'complete'
        provenance.write_text(json.dumps(commands + [r for r in previous if r.get('source') not in {c['source'] for c in commands}], indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
