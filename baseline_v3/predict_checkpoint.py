"""Re-evaluate a saved development checkpoint with submission precision."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from runtime.model import load_checkpoint, get_device
from runtime.data import read_trial, transform_input
from train import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    torch.set_num_threads(4)
    source, output = Path(args.model), Path(args.output)
    report = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    if report['args']['refit']:
        raise ValueError('A full-data refit has no untouched local validation split')
    device = get_device(args.device)
    model, stats, config, names = load_checkpoint(source / 'checkpoint.pt', device)
    root = Path(report['args']['data_dir'])
    items = []
    for session, split in report['split'].items():
        for filename in split['validation']:
            item = transform_input(read_trial(root / session / 'heldin' / filename, True), config['input_transform'])
            item['session'] = session
            items.append(item)
    output.mkdir(parents=True, exist_ok=True)
    print(evaluate(model, stats, config, names, items, device, output), flush=True)


if __name__ == '__main__':
    main()
