"""Build the ZIP and test its inference on label-stripped public trials."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import h5py
from runtime.data import session_paths, trial_paths, load_names, read_trial
from runtime.model import load_checkpoint, get_device

ROOT = Path(__file__).resolve().parent


def package(model_dir, data_dir, output):
    sessions = session_paths(data_dir, 'heldin')
    for session in sessions:
        _, stats, config, names = load_checkpoint(model_dir / session.name / 'checkpoint.pt', get_device('cpu'))
        if names != load_names(session) or config['wrist'] != names.index('Wrist'):
            raise ValueError(f'Model metadata mismatch: {session}')
        if not all(np.isfinite(value).all() for value in stats.values()):
            raise ValueError(f'Nonfinite statistics: {session}')
    if output.name != 'submission.zip':
        raise ValueError('Archive must be named submission.zip')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for name in ['infer.py', 'requirements.txt']:
            archive.write(ROOT / name, name)
        for path in sorted((ROOT / 'runtime').glob('*.py')):
            archive.write(path, 'runtime/' + path.name)
        for session in sessions:
            for name in ['checkpoint.pt', 'model_config.json']:
                path = model_dir / session.name / name
                archive.write(path, f'model/{session.name}/{name}')
        manifest = {'model': 'causal_pose_gru_pca_v1', 'sessions': {s.name: f'{s.name}/checkpoint.pt' for s in sessions}}
        archive.writestr('model/manifest.json', json.dumps(manifest, indent=2))
    if temporary.stat().st_size >= 2_000_000_000:
        temporary.unlink()
        raise ValueError('Archive exceeds 2 GB')
    temporary.replace(output)
    return len(sessions)


def selftest(archive_path, data_dir, work, device):
    # Use a fresh directory so a stale prediction cannot pass the checks.
    work.mkdir(parents=True, exist_ok=False)
    extracted = work / 'extracted'
    with ZipFile(archive_path) as archive:
        if {p.split('/')[0] for p in archive.namelist()} != {'infer.py', 'requirements.txt', 'runtime', 'model'}:
            raise ValueError('Wrong ZIP top-level entries')
        archive.extractall(extracted)
    expected = {}
    for session in session_paths(data_dir, 'heldin'):
        target = work / 'input' / session.name
        (target / 'heldout').mkdir(parents=True)
        shutil.copyfile(session / 'keypoint_names.npy', target / 'keypoint_names.npy')
        paths = trial_paths(session, 'heldin')
        chosen = [paths[0], paths[-1]] if len(paths) > 1 else paths
        expected[session.name] = []
        for i, path in enumerate(chosen, 1):
            with np.load(path, allow_pickle=False) as data:
                values = {k: data[k] for k in data.files if k != 'keypoint'}
                expected[session.name].append(len(values['tx']))
            np.savez_compressed(target / 'heldout' / f'trial_{i:03}.npz', **values)
    before = {p.relative_to(extracted): p.stat().st_mtime_ns for p in extracted.rglob('*') if p.is_file()}
    started = time.perf_counter()
    subprocess.run([sys.executable, str(extracted / 'infer.py'), '--data_dir', str(work / 'input'),
                    '--model_dir', str(extracted / 'model'), '--output_dir', str(work / 'output'),
                    '--device', device], cwd=extracted, check=True)
    elapsed = time.perf_counter() - started
    after = {p.relative_to(extracted): p.stat().st_mtime_ns for p in extracted.rglob('*') if p.is_file()}
    if before != after:
        raise ValueError('Inference changed files in submission directory')
    if {p.name for p in (work / 'output').iterdir()} != {f'{s}_pred.h5' for s in expected}:
        raise ValueError('Unexpected or missing output files')
    for session, lengths in expected.items():
        with h5py.File(work / 'output' / f'{session}_pred.h5') as f:
            if set(f) != {'predicted_trajectory', 'trial_lengths', 'trial_indices', 'keypoint_names'}:
                raise ValueError('Wrong HDF5 fields')
            pred = f['predicted_trajectory'][:]
            if pred.shape != (sum(lengths), 63) or pred.dtype != np.float32 or not np.isfinite(pred).all():
                raise ValueError(f'Invalid prediction: {session}')
            for key, values in [('trial_lengths', lengths), ('trial_indices', range(1, len(lengths) + 1))]:
                np.testing.assert_array_equal(f[key][:], values)
                if f[key].dtype != np.int64:
                    raise ValueError(f'Invalid dtype: {key}')
            if f['keypoint_names'].asstr()[:].tolist() != load_names(Path(data_dir) / session):
                raise ValueError(f'Incorrect keypoint names: {session}')
    return {'sessions': len(expected), 'trials': sum(len(v) for v in expected.values()),
            'frames': sum(sum(v) for v in expected.values()), 'device': device, 'seconds': elapsed,
            'purpose': 'Interface test using label-stripped public training trials; not generalization evaluation.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='../player_data/Data_heldin')
    p.add_argument('--model-dir', default='model')
    p.add_argument('--output', default='submission.zip')
    p.add_argument('--selftest-dir')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = p.parse_args()
    count = package(Path(args.model_dir), Path(args.data_dir), Path(args.output))
    print(f'Packaged {count} sessions in {args.output}', flush=True)
    if args.selftest_dir:
        report = selftest(Path(args.output).resolve(), Path(args.data_dir).resolve(), Path(args.selftest_dir).resolve(), args.device)
        (Path(args.selftest_dir) / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
