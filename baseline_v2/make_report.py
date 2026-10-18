import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    baseline = json.loads((ROOT.parent / 'baseline_v1/reports/summary.json').read_text())
    final = json.loads((ROOT / 'experiments/ensemble_three.json').read_text())[0]
    confirm = json.loads((ROOT / 'experiments/confirm_ensemble.json').read_text())[0]
    comparisons = [{'name': 'v1 GRU + PCA', **baseline['mean']}]
    labels = {'shared_meta': 'Shared GRU / independent adapters', 'shared_raw': 'Raw neural features',
              'shared_channels': 'Shared channel projection', 'shared_bidir': 'Bidirectional / independent adapters',
              'shared_tcn': 'Shared TCN', 'shared_score': 'Score-oriented fine-tune', 'neural_only': 'No metadata ablation',
              'shrink_adapter': 'Reduced adapters, seed 42', 'shrink_seed7': 'Reduced adapters, seed 7',
              'shared_no_adapter': 'No input adapters', 'shared_bidir_channels': 'Larger bidirectional shared model'}
    for folder, label in labels.items():
        path = ROOT / 'experiments' / folder / 'validation.json'
        if path.exists():
            comparisons.append({'name': label, **json.loads(path.read_text())['mean']})
    comparisons.append({'name': 'Selected ensemble + postprocessing', **final['mean']})
    clean = [{k: r[k] for k in ('name', 'hrfs_local', 'mpjpe', 'wrist', 'pose', 'motion', 'bone')} for r in comparisons]
    with (out / 'experiments.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(clean[0]))
        writer.writeheader()
        writer.writerows(clean)
    summary = {'development_split_seed': 42, 'confirmation_split_seed': 43,
               'baseline_v1': baseline['mean'], 'selected_ensemble_development': final['mean'],
               'fixed_ensemble_confirmation': confirm['mean'],
               'members': ['shrink_adapter', 'shrink_seed7', 'shared_no_adapter'], 'weights': [1 / 3] * 3,
               'sigma': 1, 'bone_strength': .5,
               'limitations': ['Local HRFS reproduction, not an official leaderboard score.',
                              'Development split used for architecture and checkpoint selection.',
                              'Confirmation uses fixed final epochs and weights; it is an alternate split of the same held-in dataset, not a new independent dataset.',
                              'Published condition and event metadata plus symmetric smoothing make the complete pipeline offline.']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    rows = []
    for original, new, check in zip(baseline['sessions'], final['sessions'], confirm['sessions']):
        assert original['session'] == new['session'] == check['session']
        rows.append({'session': new['session'], 'v1': original['hrfs_local'], 'v2_development': new['hrfs_local'],
                     'v2_confirmation': check['hrfs_local'], 'v2_mpjpe': new['mpjpe']})
    with (out / 'sessions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1.5, 1]})
    y = np.arange(len(rows))
    a.barh(y - .18, [r['v1'] for r in rows], .36, color='#b9c6d1', label='v1')
    a.barh(y + .18, [r['v2_development'] for r in rows], .36, color='#2463a6', label='v2 ensemble')
    a.set_yticks(y, [r['session'] for r in rows])
    a.invert_yaxis()
    a.set_xlabel('Local HRFS (same development split)')
    a.set_title('Improvement across all 20 sessions')
    a.legend(loc='upper left', bbox_to_anchor=(0, -.10), ncol=2, frameon=False)
    values = [baseline['mean']['hrfs_local'], final['mean']['hrfs_local'], confirm['mean']['hrfs_local']]
    bars = b.bar(['v1\nSplit 42', 'v2\nSplit 42', 'v2\nSplit 43'], values, color=['#b9c6d1', '#2463a6', '#26816b'])
    b.bar_label(bars, fmt='%.2f', padding=5)
    b.axhline(50, color='#a8642c', linestyle='--', linewidth=1, label='Target: 50')
    b.set_ylim(0, 60)
    b.set_ylabel('Session-equal mean local HRFS')
    b.set_title('Development and fixed-setting confirmation')
    b.legend(frameon=False)
    fig.suptitle('Hand decoding v2 | shared neural mapping and three-model ensemble', fontsize=15)
    fig.text(.5, .01, 'Local validation only; official heldout scores are unknown. The complete pipeline uses offline public metadata.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .065, 1, .95])
    fig.savefig(out / 'comparison.png', dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
