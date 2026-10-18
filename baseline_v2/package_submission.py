import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from runtime.data import session_paths, trial_paths, read_trial, load_names
from runtime.model import load_checkpoint, get_device
from runtime.postprocess import fit_bones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', nargs='+', required=True, help='Full-data refit directories')
    parser.add_argument('--weights', nargs='+', type=float)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--bone-strength', type=float, default=.5)
    parser.add_argument('--data-dir', default='../player_data/Data_heldin')
    parser.add_argument('--output', default='submission.zip')
    args = parser.parse_args()
    if args.sigma < 0 or not 0 <= args.bone_strength <= 1:
        parser.error('Invalid postprocessing settings')
    weights = np.asarray(args.weights if args.weights else [1 / len(args.members)] * len(args.members))
    if len(weights) != len(args.members) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        parser.error('Weights must be nonnegative, sum to one and match members')
    source = Path(__file__).resolve().parent
    sessions = session_paths(args.data_dir, 'heldin')
    manifest = {'version': 2, 'sigma': args.sigma, 'bone_strength': args.bone_strength, 'members': [], 'sessions': {}}
    for i, member in enumerate(args.members):
        root = Path(member)
        report = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        if not report['args']['refit']:
            raise ValueError(f'{root} is a validation model, not a full-data refit')
        if report['best_epoch'] != report['args']['epochs']:
            raise ValueError(f'{root} has not completed its requested refit epochs')
        _, _, config, names = load_checkpoint(root / 'checkpoint.pt', get_device('cpu'))
        if set(config['sessions']) != {s.name for s in sessions}:
            raise ValueError('Session coverage mismatch')
        for session in sessions:
            if names[session.name] != load_names(session):
                raise ValueError(f'Keypoint mismatch: {session}')
        manifest['members'].append({'file': f'member_{i}.pt', 'weight': float(weights[i]), 'epochs': report['best_epoch'],
                                    'architecture': config['architecture'], 'metadata': config['metadata']})
    for session in sessions:
        names = load_names(session)
        items = [read_trial(p, True) for p in trial_paths(session, 'heldin')]
        manifest['sessions'][session.name] = {'names': names, 'bones': fit_bones(items, names).tolist()}
    output = Path(args.output)
    if output.name != 'submission.zip':
        parser.error('Output must be named submission.zip')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for name in ['infer.py', 'requirements.txt']:
            archive.write(source / name, name)
        for path in sorted((source / 'runtime').glob('*.py')):
            archive.write(path, f'runtime/{path.name}')
        for i, member in enumerate(args.members):
            archive.write(Path(member) / 'checkpoint.pt', f'model/member_{i}.pt')
        archive.writestr('model/manifest.json', json.dumps(manifest, indent=2))
    if temporary.stat().st_size >= 2_000_000_000:
        temporary.unlink()
        raise ValueError('Submission exceeds 2 GB')
    temporary.replace(output)
    print(f'{output}: {output.stat().st_size:,} bytes; {len(sessions)} sessions; {len(args.members)} members')


if __name__ == '__main__':
    main()
