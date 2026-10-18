import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    def read(path):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if isinstance(value, list):
            value = value[0]
        return {r['session']: r for r in value['sessions']}
    a, b = read(args.baseline), read(args.candidate)
    if a.keys() != b.keys():
        raise ValueError('Session sets differ')
    rows = [{'session': s, 'baseline': a[s]['hrfs_local'], 'candidate': b[s]['hrfs_local'],
             'gain': b[s]['hrfs_local'] - a[s]['hrfs_local']} for s in sorted(a)]
    delta = np.array([r['gain'] for r in rows])
    rng = np.random.default_rng(2026)
    bootstrap = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(1)
    report = {'mean_gain': float(delta.mean()), 'improved_sessions': int((delta > 0).sum()),
              'session_bootstrap_95_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
              'caveat': 'Descriptive resampling across sessions; does not correct model-selection bias or establish leaderboard improvement.',
              'sessions': rows}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'sessions'})


if __name__ == '__main__':
    main()
