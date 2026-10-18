"""Assemble auditable results after validation, refit, packaging and self-tests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--development', required=True)
    p.add_argument('--confirmation', required=True)
    p.add_argument('--selected', nargs='+', required=True)
    p.add_argument('--selftests', nargs='+', required=True)
    p.add_argument('--comparisons', nargs='*', default=[])
    p.add_argument('--zip', default='submission.zip')
    args = p.parse_args()
    base = read('../baseline_v2/reports/summary.json')
    dev, confirm = read(args.development)[0], read(args.confirmation)[0]
    package = Path(args.zip)
    package_hash = hashlib.sha256(package.read_bytes()).hexdigest()
    selftests = [read(path) for path in args.selftests]
    if any(t['zip_sha256'] != package_hash for t in selftests):
        raise ValueError('Self-tests were not run against the current ZIP')
    comparisons = [read(path) for path in args.comparisons]
    if any(c['zip_sha256'] != package_hash or not c['passed'] for c in comparisons):
        raise ValueError('Inference comparisons do not validate the current ZIP')
    with ZipFile(package) as archive:
        manifest = json.loads(archive.read('model/manifest.json'))
    models = []
    for path in args.selected:
        source = Path(path)
        c = read(source / 'config.json')
        models.append({'directory': str(source), 'best_epoch': c['best_epoch'], 'args': c['args'],
                       'validation': read(source / 'validation.json')['mean']})
    report = {
        'target_platform_score': [58, 60],
        'v2_platform_score': {'value': 50.6, 'source': 'User reported; not independently verified.'},
        'v3_platform_score': None,
        'frozen_selection': read('reports/selection.json'),
        'best_searched_development': read('experiments/capacity_ensemble.json')[0]['mean'],
        'development_error_diagnosis': read('reports/error_diagnosis.json'),
        'development_split_seed': 42, 'confirmation_split_seed': 43,
        'v2_development': base['selected_ensemble_development'],
        'v2_confirmation': base['fixed_ensemble_confirmation'],
        'v3_development': dev['mean'], 'v3_confirmation': confirm['mean'],
        'development_gain': dev['mean']['hrfs_local'] - base['selected_ensemble_development']['hrfs_local'],
        'confirmation_gain': confirm['mean']['hrfs_local'] - base['fixed_ensemble_confirmation']['hrfs_local'],
        'selected_new_models': models,
        'manifest': manifest,
        'selftests': selftests,
        'inference_comparisons': comparisons,
        'package': {'path': str(package), 'bytes': package.stat().st_size,
                    'sha256': package_hash},
        'limitations': [
            'Local HRFS is a finite-data reproduction, not the private official scorer.',
            'Development split is used for model/checkpoint/ensemble selection.',
            'Confirmation settings are fixed before evaluation, but use another split of the same held-in data, not an independent new dataset.',
            'No v3 leaderboard result is available; 58-60 has not been demonstrated.',
            'The pipeline is offline and uses public condition, onset/reward timing, and future neural context.',
        ],
    }
    root = Path('reports')
    root.mkdir(exist_ok=True)
    (root / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fields = ['name', 'hrfs_local', 'wrist', 'pose', 'bone', 'motion', 'mpjpe', 'rmse']
    rows = [{'name': 'v2 ensemble / development', **report['v2_development']},
            {'name': 'v3 ensemble / development', **report['v3_development']},
            {'name': 'v2 ensemble / confirmation', **report['v2_confirmation']},
            {'name': 'v3 ensemble / confirmation', **report['v3_confirmation']}]
    for name in ['dual_multiscale_h128', 'canonical_wide', 'dropout_dual', 'lagged_uni_h128',
                 'attention_pca', 'attention_full', 'pca_dual', 'motion_finetune', 'conditional_experts',
                 'origin_dual', 'mixup_finetune', 'stacked_large', 'attention_large', 'dropout_strong', 'recording_clock',
                 'shared_forest', 'phase_template']:
        path = Path('experiments') / name / 'validation.json'
        if path.exists():
            rows.append({'name': name + ' / raw single model', **read(path)['mean']})
    kernel = Path('experiments/kernel_ridge/results.json')
    if kernel.exists():
        rows.append({'name': 'kernel_ridge / best development setting', **read(kernel)[0]['mean']})
    with (root / 'experiments.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: report[k] for k in ['development_gain', 'confirmation_gain', 'package']}, indent=2))


if __name__ == '__main__':
    main()
