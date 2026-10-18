import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from runtime.data import session_paths, trial_paths, load_names, read_trial, split_trials, to_local, model_target, features, model_events, skeleton, transform_input
from runtime.model import SharedGRU, get_device, predict, save_checkpoint, load_checkpoint
from metrics import score_trial, average
from training_utils import mix_trial


def evaluate(model, stats, config, names, items, device, output=None):
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
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
    torch.set_float32_matmul_precision(previous_precision)
    return mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='../../player_data/Data_heldin')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=160)
    p.add_argument('--schedule-epochs', type=int, help='Keep the development learning-rate schedule when refitting a selected earlier epoch')
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
    p.add_argument('--feature-mode', choices=['ema', 'multiscale', 'lagged', 'wide'], default='ema')
    p.add_argument('--dual', action='store_true')
    p.add_argument('--attention', action='store_true')
    p.add_argument('--stacked-dual', action='store_true')
    p.add_argument('--condition-head', action='store_true')
    p.add_argument('--condition-input', action='store_true')
    p.add_argument('--origin-head', action='store_true')
    p.add_argument('--recording-clock', action='store_true')
    p.add_argument('--canonical-target', action='store_true')
    p.add_argument('--channel-dropout', type=float, default=0.0)
    p.add_argument('--motion-weight', type=float, default=0.0)
    p.add_argument('--input-components', type=int, default=512)
    p.add_argument('--refit', action='store_true')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--tf32', action='store_true')
    p.add_argument('--mixup', type=float, default=0.0, help='Probability of event-aligned same-session/task trial mixup')
    p.add_argument('--phase-mode', choices=['none', 'wrist', 'all', 'go4'], default='none')
    p.add_argument('--phase-aux-weight', type=float, default=0.)
    p.add_argument('--pre-weight', type=float, default=1.0)
    args = p.parse_args()
    if args.phase_aux_weight < 0 or (args.phase_aux_weight and args.phase_mode != 'go4') or (args.phase_mode == 'go4' and (args.mixup or args.origin_head)):
        p.error('Invalid phase supervision configuration')
    if args.pre_weight <= 0 or (args.phase_mode != 'none' and args.no_metadata):
        p.error('Positive phase weight and metadata are required')
    if min(args.epochs, args.hidden, args.batch_size) < 1 or args.lr <= 0 or args.adapter_scale < 0:
        p.error('Epochs, hidden, batch size and lr must be positive; adapter scale must be nonnegative')
    if args.schedule_epochs is not None and args.schedule_epochs < args.epochs:
        p.error('--schedule-epochs cannot be shorter than --epochs')
    if sum([args.tcn, args.bidirectional, args.dual, args.attention, args.stacked_dual]) > 1:
        p.error('Select only one temporal architecture')
    if args.attention and args.hidden % 4:
        p.error('Attention hidden width must be divisible by four')
    if args.origin_head and args.no_metadata:
        p.error('Origin supervision uses public onset metadata')
    if args.recording_clock and args.no_metadata:
        p.error('Recording clock requires metadata')
    if not 0 <= args.channel_dropout < 1:
        p.error('--channel-dropout must be in [0,1)')
    if args.motion_weight < 0 or (args.motion_weight and args.canonical_target):
        p.error('Motion correlation weight must be nonnegative and uses original coordinate axes')
    if not 1 <= args.input_components <= 512:
        p.error('--input-components must be in [1,512]')
    if not 0 <= args.mixup <= 1:
        p.error('--mixup must be in [0,1]')
    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    if args.tf32 and device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training, validation, stats, names, split = [], [], {}, {}, {}
    reference = None
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
        if args.phase_mode == 'go4':
            # Only training trials obtain event supervision. Validation CSV events
            # are never attached to model inputs or used in the training loss.
            with (session/'heldin/taskinfos.csv').open(encoding='utf-8-sig',newline='') as handle:
                events={int(row['trial_index']):row for row in csv.DictReader(handle)}
            for item in chosen:
                row=events[int(Path(item['path']).stem.split('_')[1])]
                go=int(row['cue_on_frame'])-int(row['window_start_frame'])
                assert item['onset'] <= go <= item['reward']
                phase=np.full(len(item['x']),-1,dtype='float32')
                phase[item['onset']:go]=0
                phase[go:item['reward']+1]=1
                item['phase_label']=phase
        x = np.concatenate([t['x'] for t in chosen]).astype('float64')
        y = np.concatenate([to_local(t['y'], 0) for t in chosen]).astype('float64')
        rotation = None
        if args.canonical_target:
            shape = y.reshape(-1, 21, 3).mean(0)[1:]
            if reference is None:
                reference = shape.copy()
            u, _, vt = np.linalg.svd(shape.T @ reference)
            correction = np.eye(3)
            correction[-1, -1] = np.linalg.det(u @ vt)
            rotation = (u @ correction @ vt).astype('float32')
            y = (y.reshape(-1, 21, 3) @ rotation).reshape(-1, 63)
        stats[session.name] = {'x_mean': x.mean(0).astype('float32'), 'x_std': np.maximum(x.std(0), 1e-3).astype('float32'),
                               'y_mean': y.mean(0).astype('float32'), 'y_std': np.maximum(y.std(0), 1e-3).astype('float32')}
        if rotation is not None:
            stats[session.name]['rotation'] = rotation
        if args.recording_clock:
            starts = np.asarray([t['times'][0] for t in chosen], dtype=np.int64)
            stats[session.name]['clock_origin'] = np.asarray(starts.min(), dtype=np.int64)
            stats[session.name]['clock_scale'] = np.asarray(max(float(starts.max() - starts.min()) / 2e9, 1.))
        training.extend(chosen)
        if not args.refit:
            validation.extend(items[i] for i in va)
    if args.input_components < 512:
        from scipy.ndimage import gaussian_filter1d
        covariance = np.zeros((512, 512), dtype='float64')
        for t in training:
            stat = stats[t['session']]
            z = np.clip((t['x'] - stat['x_mean']) / stat['x_std'], -10, 10)
            z = gaussian_filter1d(z, 3, axis=0, mode='nearest')[::4].astype('float64')
            covariance += z.T @ z
        _, basis = np.linalg.eigh(covariance)
        projection = basis[:, -args.input_components:].astype('float32')
        for stat in stats.values():
            stat['input_projection'] = projection
    config = {'phase_mode': args.phase_mode, 'hidden': args.hidden, 'sessions': sorted(stats), 'conditions': sorted({t['condition'] for t in training}),
              'metadata': not args.no_metadata, 'bidirectional': args.bidirectional, 'input_transform': args.input_transform,
              'shared_input': args.shared_input, 'adapter_scale': args.adapter_scale,
              'canonical_target': args.canonical_target,
              'feature_mode': args.feature_mode, 'feature_dim': {'ema': 2, 'multiscale': 4, 'lagged': 5, 'wide': 4}[args.feature_mode] * args.input_components,
              'input_components': args.input_components,
              'condition_head': args.condition_head,
              'condition_input': args.condition_input,
              'origin_head': args.origin_head,
              'recording_clock': args.recording_clock,
              'architecture': 'stacked_dual_v3' if args.stacked_dual else ('attention_v3' if args.attention else ('dual_gru_v3' if args.dual else ('shared_tcn_v2' if args.tcn else 'shared_gru_v2')))}
    packed = []
    for t in training:
        stat = stats[t['session']]
        packed.append((torch.from_numpy(features(t['x'], stat['x_mean'], stat['x_std'], args.feature_mode, stat.get('input_projection'))),
                       torch.from_numpy(model_events(t, stat, config)),
                       torch.from_numpy((model_target(t['y'], stat) - stat['y_mean']) / stat['y_std']),
                       config['sessions'].index(t['session']), config['conditions'].index(t['condition'])))
        if args.phase_mode == 'go4':
            packed[-1]=(*packed[-1],torch.from_numpy(t['phase_label']))
    means = torch.from_numpy(np.stack([stats[s]['y_mean'] for s in config['sessions']])).to(device)
    mix_groups = {}
    for i, item in enumerate(packed):
        mix_groups.setdefault(item[3:5], []).append(i)
    scales = torch.from_numpy(np.stack([stats[s]['y_std'] for s in config['sessions']])).to(device)
    _, edges = skeleton(next(iter(names.values())))
    model = SharedGRU(config).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        initial['config'].setdefault('adapter_scale', 1.0)
        initial['config'].setdefault('feature_mode', 'ema')
        initial['config'].setdefault('feature_dim', 1024)
        initial['config'].setdefault('canonical_target', False)
        initial['config'].setdefault('input_components', 512)
        initial['config'].setdefault('condition_head', False)
        initial['config'].setdefault('origin_head', False)
        initial['config'].setdefault('condition_input', False)
        initial['config'].setdefault('recording_clock', False)
        comparable = dict(config)
        added = set()
        if args.phase_mode == 'go4':
            if initial['config'].get('phase_mode') != 'all':
                raise ValueError('go4 warm start requires the three-phase all model')
            comparable['phase_mode']='all'
            state=initial['state']
            for suffix in ['0.weight','0.bias','2.weight','2.bias']:
                state['phase_heads.3.'+suffix]=state['phase_heads.2.'+suffix].clone()
                state['phase_heads.2.'+suffix]=state['phase_heads.1.'+suffix].clone()
            added.update(['stage_classifier.weight','stage_classifier.bias'])
        for flag, parameter in [('condition_input', 'condition_input_weight'), ('condition_head', 'condition_weight')]:
            if config[flag] and not initial['config'][flag]:
                comparable[flag] = False
                added.add(parameter)
        if initial['config'] != comparable:
            raise ValueError('Initial checkpoint configuration differs')
        for s in stats:
            for k in stats[s]:
                if not np.allclose(stats[s][k], initial['stats'][s][k].numpy()):
                    raise ValueError('Initial checkpoint statistics differ; possible split mismatch')
        status = model.load_state_dict(initial['state'], strict=False)
        if set(status.missing_keys) != added or status.unexpected_keys:
            raise ValueError('Unexpected state mismatch when adding conditional experts')
        if args.phase_mode == 'go4':
            # Opposite perturbations cancel at initial q=.5, but avoid the
            # exact symmetry trap in the unsupervised matched-capacity control.
            with torch.no_grad():
                generator=torch.Generator(device='cpu').manual_seed(1701)
                noise=torch.randn(model.phase_heads[1][-1].weight.shape,generator=generator)*.001
                model.phase_heads[1][-1].weight.add_(noise.to(device))
                model.phase_heads[2][-1].weight.sub_(noise.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.schedule_epochs or args.epochs, eta_min=args.lr * 0.1)
    history, best = [], -1
    started = time.perf_counter()
    if args.init_checkpoint and validation and not args.last_epoch:
        initial_score=evaluate(model,stats,config,names,validation,device)
        best=initial_score['hrfs_local']
        history.append(dict(epoch=0,seconds=0.,**initial_score))
        save_checkpoint(output/'checkpoint.pt',model,stats,config,names)
        (output/'config.json').write_text(json.dumps({'args':vars(args),'best_epoch':0,'config':config,'split':split},indent=2),encoding='utf-8')
        print(json.dumps(history[-1]),flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(packed))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = [packed[i] for i in order[start:start + args.batch_size]]
            if args.mixup:
                for j, i in enumerate(order[start:start + args.batch_size]):
                    alternatives = [k for k in mix_groups[packed[i][3:]] if k != i]
                    if alternatives and np.random.random() < args.mixup:
                        partner = packed[np.random.choice(alternatives)]
                        batch[j] = mix_trial(packed[i], partner, float(np.random.beta(.4, .4)))
            x = pad_sequence([b[0] for b in batch], batch_first=True).to(device)
            if args.channel_dropout:
                keep = (torch.rand(len(batch), 1, args.input_components, device=device) >= args.channel_dropout).to(x.dtype)
                x = x * keep.repeat(1, 1, x.shape[-1] // args.input_components) / (1 - args.channel_dropout)
            meta = pad_sequence([b[1] for b in batch], batch_first=True).to(device)
            y = pad_sequence([b[2] for b in batch], batch_first=True).to(device)
            sid = torch.tensor([b[3] for b in batch], device=device)
            cid = torch.tensor([b[4] for b in batch], device=device)
            lengths = torch.tensor([len(b[0]) for b in batch], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
            optimizer.zero_grad(set_to_none=True)
            if args.phase_mode == 'go4':
                pred,phase_logits=model(x,meta,sid,cid,lengths,return_phase=True)
            elif args.origin_head:
                pred, origin = model(x, meta, sid, cid, lengths, return_origin=True)
            else:
                pred = model(x, meta, sid, cid, lengths)
            d = ((pred - y) * scales[sid, None]).reshape(*mask.shape, 21, 3)
            distances = torch.sqrt(d.square().sum(-1) + 1e-6)
            position = (0.4 * distances[:, :, 0] / 20 + 0.6 * distances[:, :, 1:].mean(-1) / 15)[mask].mean()
            if args.pre_weight != 1.0:
                phase_weights = torch.where(meta[:, :, 0] < 0, args.pre_weight, 1.0) * mask
                # Keep each trial's total weight equal to its original length.
                phase_weights = phase_weights * lengths[:, None] / phase_weights.sum(1, keepdim=True)
                frame_position = 0.4 * distances[:, :, 0] / 20 + 0.6 * distances[:, :, 1:].mean(-1) / 15
                position = (frame_position * phase_weights).sum() / mask.sum()
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
            if args.phase_aux_weight:
                labels=pad_sequence([b[5] for b in batch],batch_first=True,padding_value=-1).to(device)
                ce=torch.nn.functional.binary_cross_entropy_with_logits(phase_logits,labels.clamp_min(0),reduction='none')
                phase_terms=[]
                for label in [0,1]:
                    valid=(labels==label)&mask
                    count=valid.sum(1)
                    if (count>0).any():
                        phase_terms.append(((ce*valid).sum(1)/count.clamp_min(1))[count>0].mean())
                phase_loss=torch.stack(phase_terms).mean()
                loss=loss+args.phase_aux_weight*phase_loss
            if args.origin_head:
                resting = (meta[:, :, 2] < 0) & mask
                resting[:, 0] = True
                target_origin = (y[:, :, :3] * resting[:, :, None]).sum(1) / resting.sum(1)[:, None]
                origin_error = torch.linalg.vector_norm((origin - target_origin) * scales[sid, :3], dim=-1).mean() / 20
                loss = loss + .3 * origin_error
            if args.motion_weight:
                pair = (mask[:, 1:] & mask[:, :-1])[:, :, None]
                u = torch.diff(pl[:, :, 1:].flatten(2), dim=1)
                v = torch.diff(yl[:, :, 1:].flatten(2), dim=1)
                count = pair.sum(1).clamp_min(1)
                u = (u - (u * pair).sum(1)[:, None] / count[:, None]) * pair
                v = (v - (v * pair).sum(1)[:, None] / count[:, None]) * pair
                variance = v.square().sum(1)
                corr = (u * v).sum(1) / torch.sqrt(u.square().sum(1) * variance + 1e-6)
                valid = variance > 1e-6
                loss = loss - args.motion_weight * (corr * valid).sum() / valid.sum().clamp_min(1)
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
            if validation and (not args.last_epoch or epoch == args.epochs):
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
    # Release training graphs and CUDA allocations before interpreter teardown.
    import gc
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print('TRAINING_COMPLETE', flush=True)
