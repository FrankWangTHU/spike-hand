import subprocess
import sys
from pathlib import Path

JOBS = [
    ('conditional_experts', ['--epochs', '80', '--dual', '--feature-mode', 'multiscale', '--condition-input', '--condition-head',
                             '--channel-dropout', '.15', '--lr', '.0003', '--tf32',
                             '--init-checkpoint', 'experiments/dropout_dual/checkpoint.pt']),
    ('attention_full', ['--epochs', '160', '--attention', '--feature-mode', 'multiscale', '--condition-head',
                        '--channel-dropout', '.15', '--motion-weight', '.1', '--lr', '.0007', '--tf32']),
    ('origin_dual', ['--epochs', '180', '--dual', '--feature-mode', 'multiscale', '--origin-head',
                     '--channel-dropout', '.15', '--tf32']),
    ('mixup_finetune', ['--epochs', '50', '--dual', '--feature-mode', 'multiscale', '--channel-dropout', '.15',
                        '--mixup', '.5', '--lr', '.0002', '--init-checkpoint', 'experiments/dropout_dual/checkpoint.pt']),
]


def main():
    for name, options in JOBS:
        root = Path('experiments') / name
        if (root / 'validation.json').exists():
            print('Already finished:', name, flush=True)
            continue
        command = [sys.executable, 'train.py', '--output', str(root), '--hidden', '128', '--batch-size', '8',
                   '--shared-input', '--adapter-scale', '.1', '--input-transform', 'raw', *options]
        print('Running:', ' '.join(command), flush=True)
        with open(name + '.log', 'w', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
