"""Sequential B1/B2 development runs, with auditable matched baseline comparison."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    baseline = ROOT.parent / 'baseline_v3/experiments/stacked_large'
    reference = json.loads((baseline / 'config.json').read_text())
    source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [ROOT/'train.py', ROOT/'pose.py', ROOT/'metrics.py',
                        ROOT/'training_utils.py', *sorted((ROOT/'runtime').glob('*.py'))]}
    (ROOT / 'source_manifest.json').write_text(json.dumps(source, indent=2))
    results = {'baseline': json.loads((baseline/'validation.json').read_text())['mean'],
               'protocol': 'split42 development; same 220-epoch search as stacked_large; raw predictions',
               'experiments': {}}
    for label, k in [('B1', 16), ('B2', 32)]:
        output = ROOT / 'experiments' / label.lower()
        if output.exists():
            raise RuntimeError(f'Refusing to overwrite {output}')
        output.mkdir(parents=True)
        command = [sys.executable, '-u', 'train.py', '--output', str(output),
                   '--epochs', '220', '--hidden', '192', '--batch-size', '8',
                   '--input-transform', 'raw', '--shared-input', '--adapter-scale', '.1',
                   '--feature-mode', 'multiscale', '--stacked-dual', '--channel-dropout', '.25',
                   '--lr', '.0007', '--tf32', '--pose-components', str(k), '--device', 'cuda']
        (output/'command.json').write_text(json.dumps(command, indent=2))
        print(f'Starting {label}', flush=True)
        with (output/'train.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        config = json.loads((output/'config.json').read_text())
        if config['split'] != reference['split']:
            raise RuntimeError('Baseline split mismatch')
        ignored = {'output', 'data_dir', 'device', 'pose_components', 'pose_residual_scale'}
        for key, value in reference['args'].items():
            if key not in ignored and config['args'].get(key) != value:
                raise RuntimeError(f'Baseline configuration mismatch: {key}')
        metrics = json.loads((output/'validation.json').read_text())['mean']
        results['experiments'][label] = {
            'best_epoch': config['best_epoch'], 'metrics': metrics,
            'delta': {key: metrics[key]-value for key, value in results['baseline'].items()},
            'motion_decreased': metrics['motion'] < results['baseline']['motion'],
            'pca': json.loads((output/'pca_fit.json').read_text())['explained_variance_ratio'],
        }
        (ROOT/'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(results['experiments'][label]), flush=True)
    print('B1_B2_COMPLETE; review HRFS and Motion before B3', flush=True)


if __name__ == '__main__':
    main()
