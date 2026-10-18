"""Sequential capacity/regularization experiments on development split only."""
import subprocess
import sys
from pathlib import Path


JOBS = [
    ('stacked_large', ['--epochs', '220', '--hidden', '192', '--stacked-dual',
                       '--channel-dropout', '.25', '--lr', '.0007']),
    ('attention_large', ['--epochs', '220', '--hidden', '192', '--attention', '--condition-head',
                         '--channel-dropout', '.25', '--motion-weight', '.1', '--lr', '.0007']),
    ('dropout_strong', ['--epochs', '200', '--hidden', '128', '--dual',
                        '--channel-dropout', '.4', '--lr', '.001']),
    ('recording_clock', ['--epochs', '180', '--hidden', '128', '--dual',
                         '--recording-clock', '--channel-dropout', '.15', '--lr', '.001']),
]


if __name__ == '__main__':
    for name, options in JOBS:
        root = Path('experiments') / name
        if (root / 'validation.json').exists():
            print('Already finished:', name, flush=True)
            continue
        command = [sys.executable, 'train.py', '--output', str(root), '--batch-size', '8',
                   '--shared-input', '--adapter-scale', '.1', '--input-transform', 'raw',
                   '--feature-mode', 'multiscale', '--tf32', *options]
        print('Running:', ' '.join(command), flush=True)
        with open(name + '.log', 'w', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
