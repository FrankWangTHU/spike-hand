"""Independent multi-task temporal decoder. No external training code imports."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn
from torch.nn import functional as F
from runtime.data import session_paths, trial_paths, read_trial, transform_input, split_trials, load_names, to_local, to_global, event_features, skeleton
from metrics import score_trial, average


class Block(nn.Module):
    def __init__(self, h, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        self.conv = nn.Conv1d(h, h, 5, padding=2*dilation, dilation=dilation)
        self.ff = nn.Sequential(nn.Linear(h, h*2), nn.GELU(), nn.Dropout(.15), nn.Linear(h*2, h))

    def forward(self, x, mask):
        z = F.gelu(self.conv((self.norm(x)*mask).transpose(1, 2)).transpose(1, 2))
        return (x + self.ff(z))*mask


class Decoder(nn.Module):
    def __init__(self, ns, nc, h=96, manifold=False):
        super().__init__()
        self.session = nn.Embedding(ns, 12)
        self.condition = nn.Embedding(nc, 12)
        self.project = nn.Linear(512*2+12+24, h)
        self.blocks = nn.ModuleList([Block(h, d) for d in [1, 2, 4, 8, 16, 1]])
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 63))
        self.reconstruct = nn.Linear(h, 512)
        self.adapter = nn.Parameter(torch.zeros(ns, h, 63))
        self.bias = nn.Embedding(ns, 63)
        nn.init.zeros_(self.bias.weight)
        self.manifold = manifold
        if manifold:
            self.pose_encoder = nn.Sequential(nn.Linear(63,128),nn.GELU(),nn.Linear(128,32))
            self.latent_head = nn.Linear(h,32)
            self.pose_decoder = nn.Sequential(nn.Linear(32,128),nn.GELU(),nn.Linear(128,63))

    def forward(self, x, meta, sid, cid, mask, target=None):
        emb = torch.cat([self.session(sid), self.condition(cid)], -1)[:, None].expand(-1, x.shape[1], -1)
        z = F.gelu(self.project(torch.cat([x, meta, emb], -1)))*mask
        for block in self.blocks:
            z = block(z, mask)
        if self.manifold:
            latent = self.latent_head(z)
            result = self.pose_decoder(latent) + .1*torch.bmm(z,self.adapter[sid]) + self.bias(sid)[:,None]
            if target is not None:
                encoded = self.pose_encoder(target)
                penalty = ((self.pose_decoder(encoded)-target).square()*mask).sum()/mask.sum()/63
                penalty = penalty + .1*((latent-encoded.detach()).square()*mask).sum()/mask.sum()/32
                return result, self.reconstruct(z), penalty
            return result, self.reconstruct(z)
        return self.head(z) + .1*torch.bmm(z, self.adapter[sid]) + self.bias(sid)[:, None], self.reconstruct(z)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='experiments/denoising_tcn')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hidden', type=int, default=96)
    p.add_argument('--aux', type=float, default=.05)
    p.add_argument('--manifold', action='store_true')
    p.add_argument('--score', action='store_true')
    p.add_argument('--init-checkpoint')
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--schedule-epochs', type=int)
    p.add_argument('--last-epoch', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = Path(__file__).resolve().parent.parent/'player_data/Data_heldin'
    train, val, stats, names, splits = [], [], {}, {}, {}
    for session in session_paths(data, 'heldin'):
        items = [transform_input(read_trial(f, True), 'raw') for f in trial_paths(session, 'heldin')]
        tr, va = split_trials(items, .2, args.split_seed)
        sid = session.name
        names[sid] = load_names(session)
        assert names[sid][0] == 'Wrist'
        x = np.concatenate([items[i]['x'] for i in tr])
        y = np.concatenate([to_local(items[i]['y'], 0) for i in tr])
        stats[sid] = dict(xm=x.mean(0), xs=np.maximum(x.std(0), 1e-3), ym=y.mean(0), ys=np.maximum(y.std(0), 1e-3))
        splits[sid] = dict(train=[Path(items[i]['path']).name for i in tr], validation=[Path(items[i]['path']).name for i in va])
        for i, item in enumerate(items):
            item['session'] = sid
            (train if i in tr else val).append(item)
    sessions = sorted(stats); conditions = sorted({t['condition'] for t in train})
    device = torch.device(args.device)
    def prepare(t):
        stat = stats[t['session']]
        z = np.clip((t['x']-stat['xm'])/stat['xs'], -8, 8)
        # Trial-local filtering before decimation; no boundary crosses trials.
        smooth = gaussian_filter1d(z, 2, axis=0)
        wide = gaussian_filter1d(z, 8, axis=0)
        ix = np.unique(np.r_[np.arange(0, len(z), 3), len(z)-1])
        return dict(x=torch.tensor(np.concatenate([smooth, wide], -1)[ix]),
                    aux=torch.tensor(smooth[ix]), meta=torch.tensor(event_features(t)[ix]),
                    y=torch.tensor(((to_local(t['y'], 0)-stat['ym'])/stat['ys'])[ix]),
                    sid=sessions.index(t['session']), cid=conditions.index(t['condition']), ix=ix, item=t)
    packed = [prepare(t) for t in train]; vp = [prepare(t) for t in val]
    model = Decoder(len(sessions), len(conditions), args.hidden, args.manifold).to(device)
    if args.init_checkpoint:
        initial=torch.load(args.init_checkpoint,map_location='cpu',weights_only=True)
        assert initial['config']['splits']==splits, 'Warm start split mismatch'
        for s in stats:
            for k in stats[s]:
                assert np.allclose(stats[s][k],initial['stats'][s][k].numpy()), 'Warm start statistics mismatch'
        model.load_state_dict(initial['state'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.schedule_epochs or args.epochs, eta_min=args.lr*.1)
    means = torch.tensor(np.stack([stats[s]['ym'] for s in sessions]), device=device)
    scales = torch.tensor(np.stack([stats[s]['ys'] for s in sessions]), device=device)
    _, edges = skeleton(names[sessions[0]]); a, b = edges.T
    config = dict(args=vars(args), sessions=sessions, conditions=conditions, splits=splits)
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    started = time.perf_counter(); history=[]; best=-1
    for epoch in range(1, args.epochs+1):
        model.train(); losses=[]
        order=rng.permutation(len(packed))
        for start in range(0, len(order), 16):
            batch=[packed[i] for i in order[start:start+16]]
            def pad(k):
                return nn.utils.rnn.pad_sequence([t[k] for t in batch], batch_first=True).to(device)
            x, meta, y, aux = [pad(k) for k in ['x', 'meta', 'y', 'aux']]
            sid=torch.tensor([t['sid'] for t in batch], device=device)
            cid=torch.tensor([t['cid'] for t in batch], device=device)
            lens=torch.tensor([len(t['x']) for t in batch], device=device)
            mask=(torch.arange(x.shape[1], device=device)[None]<lens[:,None])[:,:,None]
            keep=(torch.rand(x.shape[0], 1, 512, device=device)>.15).float()
            noisy=x*keep.repeat(1,1,2)/.85
            manifold_loss = 0.
            if args.manifold:
                pred, recon, manifold_loss=model(noisy, meta, sid, cid, mask, target=y)
            else:
                pred, recon=model(noisy, meta, sid, cid, mask)
            delta=((pred-y)*scales[sid,None]).reshape(*pred.shape[:2],21,3)
            distances=(delta.square().sum(-1)+1e-6).sqrt()
            m=mask[:,:,0]
            wrist=(distances[:,:,0]*m).sum(1)/lens
            pose=(distances[:,:,1:].mean(-1)*m).sum(1)/lens
            pl=(pred*scales[sid,None]+means[sid,None]).reshape(*pred.shape[:2],21,3)
            yl=(y*scales[sid,None]+means[sid,None]).reshape(*pred.shape[:2],21,3)
            pl=torch.cat([torch.zeros_like(pl[:,:,:1]),pl[:,:,1:]],2)
            yl=torch.cat([torch.zeros_like(yl[:,:,:1]),yl[:,:,1:]],2)
            lp=(pl[:,:,a]-pl[:,:,b]).norm(dim=-1)
            ly=(yl[:,:,a]-yl[:,:,b]).norm(dim=-1)
            bone=(((lp-ly).abs()/ly.clamp_min(1)).mean(-1)*m).sum(1)/lens
            pair=(mask[:,1:] & mask[:,:-1])
            u=torch.diff(pl[:,:,1:].flatten(2),dim=1); v=torch.diff(yl[:,:,1:].flatten(2),dim=1)
            count=pair.sum(1).clamp_min(1)
            uc=(u-(u*pair).sum(1)[:,None]/count[:,None])*pair
            vc=(v-(v*pair).sum(1)[:,None]/count[:,None])*pair
            corr=(uc*vc).sum(1)/(uc.square().sum(1)*vc.square().sum(1)+1e-6).sqrt()
            loss=(.4*wrist/20+.6*pose/15+.08*bone).mean()-.12*corr.mean()
            if args.score:
                # Differentiable score surrogate: retain a small distance penalty
                # so difficult trials do not lose all learning signal.
                fidelity=(.25*torch.exp(-wrist/20)+.35*torch.exp(-pose/15)+.1*torch.exp(-5*bone)+.1*corr.mean(1).clamp_min(0)).mean()/.9
                loss=.15*(.4*wrist/20+.6*pose/15).mean()-fidelity
            loss=loss+.04*((u-v).square()*pair).sum()/pair.sum().clamp_min(1)/60
            loss=loss+args.aux*((recon-aux).square()*mask).sum()/mask.sum()/512
            loss=loss+.1*manifold_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); losses.append(loss.item())
        schedule.step()
        if epoch==args.epochs or (not args.last_epoch and (epoch==1 or epoch%10==0)):
            model.eval(); rows={}; predictions={}
            with torch.inference_mode():
                for t in vp:
                    pred,_=model(t['x'][None].to(device),t['meta'][None].to(device),torch.tensor([t['sid']],device=device),torch.tensor([t['cid']],device=device),torch.ones(1,len(t['x']),1,device=device))
                    stat=stats[t['item']['session']]
                    local=pred[0].cpu().numpy()*stat['ys']+stat['ym']
                    full=np.stack([np.interp(np.arange(len(t['item']['y'])),t['ix'],local[:,j]) for j in range(63)],-1).astype('float32')
                    full=to_global(full,0); s=t['item']['session']
                    rows.setdefault(s,[]).append(score_trial(t['item']['y'],full,t['item']['times'],names[s]))
                    predictions[s+'_'+Path(t['item']['path']).stem]=full
            mean=average([average(r) for r in rows.values()])
            row=dict(epoch=epoch,seconds=time.perf_counter()-started,loss=float(np.mean(losses)),**mean)
            history.append(row); print(json.dumps(row),flush=True)
            (out/'history.json').write_text(json.dumps(history,indent=2))
            if mean['hrfs_local']>best:
                best=mean['hrfs_local']
                torch.save(dict(state=model.state_dict(),stats={s:{k:torch.tensor(v) for k,v in st.items()} for s,st in stats.items()},config=config),out/'checkpoint.pt')
                np.savez_compressed(out/'validation_predictions.npz',**predictions)
                (out/'validation.json').write_text(json.dumps(dict(mean=mean,epoch=epoch,sessions={s:average(r) for s,r in rows.items()}),indent=2))
    print('COMPLETE',best,flush=True)


if __name__=='__main__':
    main()
