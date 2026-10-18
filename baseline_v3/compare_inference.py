"""Compare actual ZIP inference across devices/environments on identical inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    left, right = Path(args.left), Path(args.right)
    reports = [json.loads((root / 'report.json').read_text(encoding='utf-8')) for root in (left, right)]
    for key in ('zip_sha256', 'sessions', 'trials', 'frames'):
        if reports[0][key] != reports[1][key]:
            raise ValueError('Different test artifact or inputs: ' + key)
    files = sorted((left / 'output').glob('*_pred.h5'))
    if {f.name for f in files} != {f.name for f in (right / 'output').glob('*_pred.h5')}:
        raise ValueError('Session sets differ')
    rows, absolute_sum, count = [], 0., 0
    for path in files:
        with h5py.File(path) as a, h5py.File(right / 'output' / path.name) as b:
            for key in ('trial_lengths', 'trial_indices'):
                np.testing.assert_array_equal(a[key][:], b[key][:])
            np.testing.assert_array_equal(a['keypoint_names'].asstr()[:], b['keypoint_names'].asstr()[:])
            x, y = a['predicted_trajectory'][:], b['predicted_trajectory'][:]
        np.testing.assert_allclose(x, y, atol=.01, rtol=.0001)
        difference = np.abs(x.astype('float64') - y)
        rows.append({'session': path.name, 'max_absolute_difference': float(difference.max()),
                     'mean_absolute_difference': float(difference.mean())})
        absolute_sum += difference.sum()
        count += difference.size
    result = {'left': args.left, 'right': args.right, 'zip_sha256': reports[0]['zip_sha256'],
              'atol': .01, 'rtol': .0001, 'passed': True,
              'max_absolute_difference': max(r['max_absolute_difference'] for r in rows),
              'mean_absolute_difference': float(absolute_sum / count), 'sessions': rows}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'sessions'})


if __name__ == '__main__':
    main()
