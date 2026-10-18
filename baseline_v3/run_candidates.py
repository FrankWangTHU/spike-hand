"""Run the remaining development candidates sequentially to fit the 8 GB GPU."""
import subprocess
import sys
from pathlib import Path

JOBS = [
    ('canonical_wide', ['--dual', '--feature-mode', 'wide', '--canonical-target']),
    ('dropout_dual', ['--dual', '--feature-mode', 'multiscale', '--channel-dropout', '.15']),
    ('lagged_uni_h128', ['--feature-mode', 'lagged']),
]


def main():
    for name, options in JOBS:
        root = Path('experiments') / name
        if (root / 'validation.json').exists():
            print('Already finished:', name, flush=True)
            continue
        args = [sys.executable, 'train.py', '--output', str(root), '--epochs', '180', '--hidden', '128',
                '--batch-size', '8', '--shared-input', '--adapter-scale', '.1', '--input-transform', 'raw', *options]
        print('Running:', ' '.join(args), flush=True)
        with open(name + '.log', 'w', encoding='utf-8') as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
