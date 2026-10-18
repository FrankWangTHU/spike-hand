"""Export the completed validation run to CSV and a static diagnostic figure."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent / 'reports'
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    rows = summary['sessions']
    with (root / 'validation.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1.5, 1]})
    idx = np.arange(len(rows))
    a.barh(idx - 0.18, [r['hrfs_local'] for r in rows], height=0.36, color='#2463a6', label='Pose GRU + PCA')
    a.barh(idx + 0.18, [r['constant_hrfs'] for r in rows], height=0.36, color='#b7c4ce', label='Training mean pose')
    a.set_yticks(idx, [r['session'] for r in rows])
    a.invert_yaxis()
    a.set_xlabel('Local HRFS reproduction (higher is better)')
    a.legend(loc='upper left', bbox_to_anchor=(0, -0.10), ncol=2, frameon=False)
    a.set_title('Held-in trial validation by session')
    for row in rows:
        detail = json.loads((root / (row['session'] + '.json')).read_text(encoding='utf-8'))
        history = detail['history']
        b.plot([h['epoch'] for h in history], [h['hrfs_local'] for h in history], alpha=0.55, linewidth=1.1)
    b.set_xlabel('Epoch')
    b.set_ylabel('Validation local HRFS')
    b.set_title('Learning curves (one line per session)')
    fig.suptitle('Baseline v1 | condition-stratified trial split, seed 42', fontsize=15)
    fig.text(0.5, 0.01, 'One validation split used for checkpoint selection; not official leaderboard scores. Coordinates retain original units.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, 0.065, 1, 0.96])
    fig.savefig(root / 'validation.png', dpi=160)
    print(json.dumps(summary['mean'], indent=2))


if __name__ == '__main__':
    main()
