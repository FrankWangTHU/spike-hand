import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import numpy as np
import h5py
import torch
from runtime.data import session_paths, trial_paths, load_names, read_trial
from runtime.model import load_checkpoint, predict, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--model_dir', required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = p.parse_args()
    torch.set_num_threads(4)
    device = get_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sessions = session_paths(args.data_dir, 'heldout')
    unknown = {x.name for x in output.glob('*_pred.h5')} - {s.name + '_pred.h5' for s in sessions}
    if unknown:
        raise ValueError(f'Output directory has unrelated predictions: {unknown}')
    for session in sessions:
        model, stats, config, names = load_checkpoint(Path(args.model_dir) / session.name / 'checkpoint.pt', device)
        if names != load_names(session):
            raise ValueError(f'Keypoint order differs for {session.name}')
        predictions = []
        for trial in trial_paths(session, 'heldout'):
            predictions.append(predict(model, read_trial(trial), stats, config['wrist'], device))
        lengths = np.asarray([len(x) for x in predictions], dtype=np.int64)
        path = output / f'{session.name}_pred.h5'
        temporary = path.with_suffix('.h5.tmp')
        with h5py.File(temporary, 'w') as f:
            f.create_dataset('predicted_trajectory', data=np.concatenate(predictions).astype('float32'))
            f.create_dataset('trial_lengths', data=lengths)
            f.create_dataset('trial_indices', data=np.arange(1, len(lengths) + 1, dtype=np.int64))
            f.create_dataset('keypoint_names', data=np.array(names, dtype=h5py.string_dtype('utf-8')))
        temporary.replace(path)
        print(f'{session.name}: {len(lengths)} trials / {sum(lengths)} frames', flush=True)


if __name__ == '__main__':
    main()
