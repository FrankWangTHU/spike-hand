import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from runtime.data import session_paths, trial_paths, load_names, read_trial, split_trials, to_local, features, event_features, skeleton, transform_input
from runtime.model import SharedGRU, get_device, predict, save_checkpoint, load_checkpoint
from metrics import score_trial, average


def evaluate(model, stats, config, names, items, device, output=None):
    by_session = {}
    predictions = {}
    for item in items:
        s = item['session']
        p = predict(model, item, stats[s], config['sessions'].index(s), config['conditions'].index(item['condition']), device)
        score = score_trial(item['y'], p, item['times'], names[s])
        by_session.setdefault(s, []).append({'trial': Path(item['path']).name, **score})
        predictions[f'{s}_{Path(item["path"]).stem}'] = p
    rows = [{'session': s, **average([{k: v for k, v in r.items() if k != 'trial'} for r in values])} for s, values in by_session.items()]
    mean = average([{k: v for k, v in r.items() if k != 'session'} for r in rows])
    if output:
        np.savez_compressed(output / 'validation_predictions.npz', **predictions)
        (output / 'validation.json').write_text(json.dumps({'mean': mean, 'sessions': rows, 'trials': by_session}, indent=2), encoding='utf-8')
    return mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='../player_data/Data_heldin')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=160)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--no-metadata', action='store_true')
    p.add_argument('--bidirectional', action='store_true')
    p.add_argument('--input-transform', choices=['log', 'raw', 'sqrt_tx'], default='log')
    p.add_argument('--shared-input', action='store_true')
    p.add_argument('--tcn', action='store_true')
    p.add_argument('--last-epoch', action='store_true', help='Select final epoch without using validation for checkpoint selection')
    p.add_argument('--score-loss', action='store_true')
    p.add_argument('--init-checkpoint')
    p.add_argument('--adapter-scale', type=float, default=1.0)
    p.add_argument('--refit', action='store_true')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--lr', type=float, default=0.001)
    args = p.parse_args()
    if min(args.epochs, args.hidden, args.batch_size) < 1 or args.lr <= 0 or args.adapter_scale < 0:
        p.error('Epochs, hidden, batch size and lr must be positive; adapter scale must be nonnegative')
    if args.tcn and args.bidirectional:
        p.error('--tcn and --bidirectional select different architectures')
    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training, validation, stats, names, split = [], [], {}, {}, {}
    for session in session_paths(args.data_dir, 'heldin'):
        names[session.name] = load_names(session)
        if names[session.name][0] != 'Wrist':
            raise ValueError('Expected wrist-first dataset')
        items = [transform_input(read_trial(path, True), args.input_transform) for path in trial_paths(session, 'heldin')]
        for item in items:
            item['session'] = session.name
        tr, va = split_trials(items, 0.2, args.split_seed)
        split[session.name] = {'train': [Path(items[i]['path']).name for i in tr], 'validation': [Path(items[i]['path']).name for i in va]}
        chosen = items if args.refit else [items[i] for i in tr]
        x = np.concatenate([t['x'] for t in chosen]).astype('float64')
        y = np.concatenate([to_local(t['y'], 0) for t in chosen]).astype('float64')
        stats[session.name] = {'x_mean': x.mean(0).astype('float32'), 'x_std': np.maximum(x.std(0), 1e-3).astype('float32'),
                               'y_mean': y.mean(0).astype('float32'), 'y_std': np.maximum(y.std(0), 1e-3).astype('float32')}
        training.extend(chosen)
        if not args.refit:
            validation.extend(items[i] for i in va)
    config = {'hidden': args.hidden, 'sessions': sorted(stats), 'conditions': sorted({t['condition'] for t in training}),
              'metadata': not args.no_metadata, 'bidirectional': args.bidirectional, 'input_transform': args.input_transform,
              'shared_input': args.shared_input, 'adapter_scale': args.adapter_scale,
              'architecture': 'shared_tcn_v2' if args.tcn else 'shared_gru_v2'}
    packed = []
    for t in training:
        stat = stats[t['session']]
        packed.append((torch.from_numpy(features(t['x'], stat['x_mean'], stat['x_std'])),
                       torch.from_numpy(event_features(t)),
                       torch.from_numpy((to_local(t['y'], 0) - stat['y_mean']) / stat['y_std']),
                       config['sessions'].index(t['session']), config['conditions'].index(t['condition'])))
    means = torch.from_numpy(np.stack([stats[s]['y_mean'] for s in config['sessions']])).to(device)
    scales = torch.from_numpy(np.stack([stats[s]['y_std'] for s in config['sessions']])).to(device)
    _, edges = skeleton(next(iter(names.values())))
    model = SharedGRU(config).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        initial['config'].setdefault('adapter_scale', 1.0)
        if initial['config'] != config:
            raise ValueError('Initial checkpoint configuration differs')
        for s in stats:
            for k in stats[s]:
                if not np.allclose(stats[s][k], initial['stats'][s][k].numpy()):
                    raise ValueError('Initial checkpoint statistics differ; possible split mismatch')
        model.load_state_dict(initial['state'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.1)
    history, best = [], -1
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(packed))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = [packed[i] for i in order[start:start + args.batch_size]]
            x = pad_sequence([b[0] for b in batch], batch_first=True).to(device)
            meta = pad_sequence([b[1] for b in batch], batch_first=True).to(device)
            y = pad_sequence([b[2] for b in batch], batch_first=True).to(device)
            sid = torch.tensor([b[3] for b in batch], device=device)
            cid = torch.tensor([b[4] for b in batch], device=device)
            lengths = torch.tensor([len(b[0]) for b in batch], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, meta, sid, cid, lengths)
            d = ((pred - y) * scales[sid, None]).reshape(*mask.shape, 21, 3)
            distances = torch.sqrt(d.square().sum(-1) + 1e-6)
            position = (0.4 * distances[:, :, 0] / 20 + 0.6 * distances[:, :, 1:].mean(-1) / 15)[mask].mean()
            delta = torch.diff(pred - y, dim=1)
            velocity = delta.square().mean(-1)[mask[:, 1:] & mask[:, :-1]].mean()
            # Compare bone lengths in the wrist-centered coordinate system.
            pl = (pred * scales[sid, None] + means[sid, None]).reshape(*mask.shape, 21, 3).clone()
            yl = (y * scales[sid, None] + means[sid, None]).reshape(*mask.shape, 21, 3).clone()
            pl[:, :, 0] = 0
            yl[:, :, 0] = 0
            pa, pb = edges.T
            lp = torch.linalg.vector_norm(pl[:, :, pa] - pl[:, :, pb], dim=-1)
            ly = torch.linalg.vector_norm(yl[:, :, pa] - yl[:, :, pb], dim=-1)
            bone = ((lp - ly).abs() / ly.clamp_min(1)).mean(-1)[mask].mean()
            loss = position + 0.2 * velocity + 0.05 * bone
            if args.score_loss and epoch > 5:
                trial_wrist = (distances[:, :, 0] * mask).sum(1) / lengths
                trial_pose = (distances[:, :, 1:].mean(-1) * mask).sum(1) / lengths
                trial_bone = (((lp - ly).abs() / ly.clamp_min(1)).mean(-1) * mask).sum(1) / lengths
                pair = (mask[:, 1:] & mask[:, :-1])[:, :, None]
                u = torch.diff(pl[:, :, 1:].flatten(2), dim=1)
                v = torch.diff(yl[:, :, 1:].flatten(2), dim=1)
                count = pair.sum(1).clamp_min(1)
                u = (u - (u * pair).sum(1)[:, None] / count[:, None]) * pair
                v = (v - (v * pair).sum(1)[:, None] / count[:, None]) * pair
                corr = ((u * v).sum(1) / torch.sqrt(u.square().sum(1) * v.square().sum(1) + 1e-6)).mean(1)
                fidelity = (.25 * torch.exp(-trial_wrist / 20) + .35 * torch.exp(-trial_pose / 15)
                            + .1 * torch.exp(-5 * trial_bone) + .1 * corr.clamp_min(0)).mean() / .9
                loss = 0.25 * position + 0.1 * velocity - fidelity
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'seconds': time.perf_counter() - started}
            if validation:
                row.update(evaluate(model, stats, config, names, validation, device))
            print(json.dumps(row), flush=True)
            history.append(row)
            (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            if args.refit or (args.last_epoch and epoch == args.epochs) or (not args.last_epoch and row['hrfs_local'] > best):
                best = row.get('hrfs_local', 0)
                save_checkpoint(output / 'checkpoint.pt', model, stats, config, names)
                (output / 'config.json').write_text(json.dumps({'args': vars(args), 'best_epoch': epoch, 'config': config, 'split': split}, indent=2), encoding='utf-8')
    if validation:
        model, stats, config, names = load_checkpoint(output / 'checkpoint.pt', device)
        print('BEST', evaluate(model, stats, config, names, validation, device, output), flush=True)


if __name__ == '__main__':
    main()
