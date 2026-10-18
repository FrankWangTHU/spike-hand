"""Offline neural decoding with documented public metadata and an ensemble."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path
import numpy as np
import h5py
import torch
from runtime.data import session_paths, trial_paths, load_names, read_trial, transform_input
from runtime.model import load_checkpoint, get_device, predict
from runtime.postprocess import postprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = get_device(args.device)
    root = Path(args.model_dir)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    weights = np.asarray([m['weight'] for m in manifest['members']])
    if not len(weights) or np.any(weights < 0) or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid ensemble weights')
    members = [load_checkpoint(root / m['file'], device) for m in manifest['members']]
    sessions = session_paths(args.data_dir, 'heldout')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unexpected = {p.name for p in output.glob('*_pred.h5')} - {s.name + '_pred.h5' for s in sessions}
    if unexpected:
        raise ValueError(f'Unrelated files in output directory: {unexpected}')
    for session in sessions:
        names = load_names(session)
        if session.name not in manifest['sessions']:
            raise ValueError(f'No model for {session.name}')
        session_config = manifest['sessions'][session.name]
        if session_config['names'] != names:
            raise ValueError(f'Keypoint ordering mismatch: {session.name}')
        for _, _, config, saved_names in members:
            if session.name not in config['sessions'] or saved_names.get(session.name) != names:
                raise ValueError(f'Checkpoint session mismatch: {session.name}')
        predictions = []
        for path in trial_paths(session, 'heldout'):
            original = read_trial(path, labels=False)
            p = np.zeros((len(original['x']), 63), dtype=np.float32)
            for weight, (model, stats, config, _) in zip(weights, members):
                if original['condition'] not in config['conditions']:
                    raise ValueError(f'Unknown condition: {original["condition"]}')
                item = transform_input({**original, 'x': original['x'].copy()}, config.get('input_transform', 'log'))
                p += weight * predict(model, item, stats[session.name], config['sessions'].index(session.name),
                                      config['conditions'].index(item['condition']), device)
            bones = session_config.get('condition_bones', {}).get(original['condition'], session_config['bones'])
            p = postprocess(p, np.asarray(bones), names, manifest['sigma'], manifest['bone_strength'])
            if not np.isfinite(p).all():
                raise ValueError('Nonfinite model prediction')
            predictions.append(p)
        lengths = np.asarray([len(p) for p in predictions], dtype=np.int64)
        target = output / f'{session.name}_pred.h5'
        temporary = target.with_suffix('.h5.tmp')
        with h5py.File(temporary, 'w') as f:
            f.create_dataset('predicted_trajectory', data=np.concatenate(predictions).astype('float32'))
            f.create_dataset('trial_lengths', data=lengths)
            f.create_dataset('trial_indices', data=np.arange(1, len(lengths) + 1, dtype=np.int64))
            f.create_dataset('keypoint_names', data=np.array(names, dtype=h5py.string_dtype('utf-8')))
        temporary.replace(target)
        print(f'{session.name}: {len(lengths)} trials / {sum(lengths)} frames', flush=True)


if __name__ == '__main__':
    main()
