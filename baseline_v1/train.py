"""Condition-stratified validation, followed by a fresh full-data refit."""
import argparse
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from runtime.data import session_paths, trial_paths, load_names, read_trial, split_trials, to_local, features
from runtime.model import PoseGRU, predict, get_device, save_checkpoint
from metrics import score_trial, average


def prepare(items, wrist, latent):
    x = np.concatenate([t['x'] for t in items]).astype('float64')
    y = np.concatenate([to_local(t['y'], wrist) for t in items]).astype('float64')
    stats = {'x_mean': x.mean(0).astype('float32'), 'x_std': np.maximum(x.std(0), 1e-3).astype('float32'),
             'y_mean': y.mean(0).astype('float32'), 'y_std': np.maximum(y.std(0), 1e-3).astype('float32')}
    z = (y - stats['y_mean']) / stats['y_std']
    _, vectors = np.linalg.eigh(z.T @ z / len(z))
    basis = vectors[:, -latent:][:, ::-1].T.copy().astype('float32')
    prepared = [(torch.from_numpy(features(t['x'], stats['x_mean'], stats['x_std'])),
                 torch.from_numpy((to_local(t['y'], wrist) - stats['y_mean']) / stats['y_std'])) for t in items]
    return stats, basis, prepared


def fit(items, validation, names, args, epochs, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_device(args.device)
    config = {'hidden': args.hidden, 'latent': args.latent, 'wrist': names.index('Wrist'),
              'feature_transform': 'log1p_standardize_clip10_ema035', 'version': 1}
    stats, basis, data = prepare(items, config['wrist'], args.latent)
    model = PoseGRU(config, basis).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Entire trials are padded only on the right; loss masks exclude padding.
    # The unidirectional GRU cannot let padding affect earlier valid outputs.
    best, best_epoch, best_state, stale, history = -float('inf'), 0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(data))
        loss_sum, nframes = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch = [data[i] for i in order[start:start + args.batch_size]]
            lengths = torch.tensor([len(x) for x, _ in batch], device=device)
            x = pad_sequence([x for x, _ in batch], batch_first=True).to(device)
            target = pad_sequence([y for _, y in batch], batch_first=True).to(device)
            mask = torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]
            optimizer.zero_grad(set_to_none=True)
            pred, factors = model(x)
            error = (pred - target).square().reshape(*mask.shape, 21, 3).mean(-1)
            wrist = config['wrist']
            fingers = [j for j in range(21) if j != wrist]
            position = (0.4 * error[:, :, wrist] + 0.6 * error[:, :, fingers].mean(-1))[mask].mean()
            latent_loss = (factors - target @ model.basis.T).square().mean(-1)[mask].mean()
            pair_mask = mask[:, 1:] & mask[:, :-1]
            delta = torch.diff(pred - target, dim=1)
            velocity = delta.square().mean(-1)[pair_mask].mean() if pair_mask.any() else pred.sum() * 0
            loss = position + 0.05 * latent_loss + 0.1 * velocity
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * lengths.sum().item()
            nframes += lengths.sum().item()
        row = {'epoch': epoch, 'train_loss': loss_sum / nframes}
        if validation:
            scores = [score_trial(t['y'], predict(model, t, stats, wrist, device), t['times'], names) for t in validation]
            row.update(average(scores))
            if row['hrfs_local'] > best:
                best, best_epoch, stale = row['hrfs_local'], epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        if validation and stale >= args.patience:
            break
    if validation:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return model, stats, config, history, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../player_data/Data_heldin')
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--sessions', nargs='*')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--hidden', type=int, default=96)
    parser.add_argument('--latent', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--no-refit', action='store_true')
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1 or not 1 <= args.latent <= 63 or min(args.epochs, args.hidden, args.batch_size, args.patience) < 1 or args.lr <= 0:
        parser.error('Invalid training parameters')
    torch.set_num_threads(4)
    out = Path(args.output_dir)
    reports = out / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    selected = session_paths(args.data_dir, 'heldin')
    if args.sessions:
        missing = set(args.sessions) - {p.name for p in selected}
        if missing:
            parser.error(f'Unknown sessions: {missing}')
        selected = [p for p in selected if p.name in args.sessions]
    summary = []
    for session in selected:
        started = time.perf_counter()
        names = load_names(session)
        items = [read_trial(p, labels=True) for p in trial_paths(session, 'heldin')]
        train_ids, val_ids = split_trials(items, args.val_fraction, args.seed)
        training, validation = [items[i] for i in train_ids], [items[i] for i in val_ids]
        print(f'{session.name}: train={len(training)}, val={len(validation)}', flush=True)
        model, stats, config, history, best_epoch = fit(training, validation, names, args, args.epochs, args.seed)
        save_checkpoint(out / 'validation_models' / session.name / 'checkpoint.pt', model, stats, config, names)
        results = []
        for t in validation:
            pred = predict(model, t, stats, config['wrist'], get_device(args.device))
            score = score_trial(t['y'], pred, t['times'], names)
            results.append({'trial': Path(t['path']).name, 'condition': t['condition'], **score})
        # A training-only mean pose is a useful sanity reference on this same split.
        mean_pose = np.concatenate([t['y'] for t in training]).mean(0)
        constant = average([score_trial(t['y'], np.broadcast_to(mean_pose, t['y'].shape), t['times'], names) for t in validation])
        metrics = average([{k: v for k, v in r.items() if k not in ('trial', 'condition')} for r in results])
        report = {'session': session.name, 'trials': len(items), 'frames': sum(len(t['x']) for t in items),
                  'train_trials': [Path(t['path']).name for t in training], 'val_trials': [Path(t['path']).name for t in validation],
                  'best_epoch': best_epoch, 'validation': metrics, 'constant_baseline': constant,
                  'trial_metrics': results, 'history': history, 'args': vars(args)}
        if not args.no_refit:
            print(f'{session.name}: refit all trials for {best_epoch} epochs', flush=True)
            model, stats, config, _, _ = fit(items, [], names, args, best_epoch, args.seed)
            target = out / 'model' / session.name
            save_checkpoint(target / 'checkpoint.pt', model, stats, config, names)
            (target / 'model_config.json').write_text(json.dumps({**config, 'epochs': best_epoch, 'names': names}, indent=2), encoding='utf-8')
        report['elapsed_seconds'] = time.perf_counter() - started
        (reports / f'{session.name}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        summary.append({'session': session.name, 'best_epoch': best_epoch, **metrics,
                        'constant_hrfs': constant['hrfs_local'], 'elapsed_seconds': report['elapsed_seconds']})
        (reports / 'summary.json').write_text(json.dumps({'sessions': summary, 'mean': average([{k: v for k, v in r.items() if k != 'session'} for r in summary])}, indent=2), encoding='utf-8')
        print(f'{session.name}: HRFS(local)={metrics["hrfs_local"]:.3f}, constant={constant["hrfs_local"]:.3f}, seconds={report["elapsed_seconds"]:.1f}', flush=True)
    if not args.no_refit:
        checkpoints = sorted((out / 'model').glob('*/checkpoint.pt'))
        (out / 'model' / 'manifest.json').write_text(json.dumps({'model': 'causal_pose_gru_pca_v1',
            'sessions': {p.parent.name: f'{p.parent.name}/checkpoint.pt' for p in checkpoints}}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
