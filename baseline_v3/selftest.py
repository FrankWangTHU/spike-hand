"""Self-test the actual ZIP using public trials with keypoint removed."""
import argparse
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile
import numpy as np
import h5py
from runtime.data import session_paths, trial_paths, load_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', default='submission.zip')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--data-dir', default='../player_data/Data_heldin')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--include-longest', action='store_true', help='Also test the longest trial in each session')
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=False)
    extracted = work / 'extracted'
    with ZipFile(args.zip) as archive:
        if {p.split('/')[0] for p in archive.namelist()} != {'infer.py', 'requirements.txt', 'runtime', 'model'}:
            raise ValueError('Incorrect ZIP top level')
        archive.extractall(extracted)
    expected = {}
    for session in session_paths(args.data_dir, 'heldin'):
        target = work / 'input' / session.name
        (target / 'heldout').mkdir(parents=True)
        shutil.copyfile(session / 'keypoint_names.npy', target / 'keypoint_names.npy')
        paths = trial_paths(session, 'heldin')
        chosen = [paths[0], paths[-1]]
        if args.include_longest:
            def length(path):
                with np.load(path, allow_pickle=False) as d:
                    return len(d['frame_timestamps_ns'])
            longest = max(paths, key=length)
            if longest not in chosen:
                chosen.append(longest)
        lengths = []
        for i, path in enumerate(chosen, 1):
            with np.load(path, allow_pickle=False) as d:
                values = {k: d[k] for k in d.files if k != 'keypoint'}
            np.savez_compressed(target / 'heldout' / f'trial_{i:03}.npz', **values)
            lengths.append(len(values['tx']))
        expected[session.name] = (lengths, load_names(session))
    before = {p.relative_to(extracted): p.stat().st_mtime_ns for p in extracted.rglob('*') if p.is_file()}
    start = time.perf_counter()
    subprocess.run([sys.executable, str(extracted / 'infer.py'), '--data_dir', str(work / 'input'), '--model_dir', str(extracted / 'model'),
                    '--output_dir', str(work / 'output'), '--device', args.device], cwd=extracted, check=True)
    elapsed = time.perf_counter() - start
    after = {p.relative_to(extracted): p.stat().st_mtime_ns for p in extracted.rglob('*') if p.is_file()}
    if before != after:
        raise ValueError('Inference modified the submission directory')
    if {p.name for p in (work / 'output').iterdir()} != {f'{s}_pred.h5' for s in expected}:
        raise ValueError('Unexpected output files')
    for session, (lengths, names) in expected.items():
        with h5py.File(work / 'output' / f'{session}_pred.h5') as f:
            if set(f) != {'predicted_trajectory', 'trial_lengths', 'trial_indices', 'keypoint_names'}:
                raise ValueError('Invalid HDF5 keys')
            p = f['predicted_trajectory'][:]
            if p.shape != (sum(lengths), 63) or p.dtype != np.float32 or not np.isfinite(p).all():
                raise ValueError('Invalid trajectories')
            for key, values in [('trial_lengths', lengths), ('trial_indices', list(range(1, len(lengths) + 1)))]:
                np.testing.assert_array_equal(f[key][:], values)
                if f[key].dtype != np.int64:
                    raise ValueError('Invalid index dtype')
            if f['keypoint_names'].asstr()[:].tolist() != names:
                raise ValueError('Name order mismatch')
    report = {'sessions': len(expected), 'trials': sum(len(v[0]) for v in expected.values()), 'frames': sum(sum(v[0]) for v in expected.values()),
              'device': args.device, 'seconds': elapsed, 'purpose': 'Public label-stripped trial interface check, not a generalization score.',
              'zip_sha256': hashlib.sha256(Path(args.zip).read_bytes()).hexdigest(),
              'environment': {'python': platform.python_version(), **{p: version(p) for p in ('numpy', 'scipy', 'h5py', 'torch')}}}
    (work / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
