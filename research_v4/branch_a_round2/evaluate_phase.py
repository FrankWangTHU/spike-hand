"""Held-in phase-label audit; event truth is never passed to the network."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from runtime.model import load_checkpoint
from runtime.data import read_trial,transform_input,features,model_events
from metrics import average

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment');parser.add_argument('--checkpoint',default='checkpoint.pt');args=parser.parse_args()
    folder=ROOT/args.experiment
    model,stats,config,_=load_checkpoint(folder/args.checkpoint,torch.device('cpu'))
    if config['phase_mode']!='go4':raise ValueError('Expected go4 checkpoint')
    torch.set_num_threads(2);model.eval()
    manifest=json.loads((folder/'config.json').read_text())
    rows=[]
    for s,split in manifest['split'].items():
        data=REPO/'player_data/Data_heldin'/s/'heldin'
        with (data/'taskinfos.csv').open(encoding='utf-8-sig',newline='') as f:
            events={int(r['trial_index']):r for r in csv.DictReader(f)}
        def truth(file):
            r=events[int(Path(file).stem.split('_')[1])]
            return int(r['cue_on_frame'])-int(r['window_start_frame'])
        templates={};all_offsets=[]
        for file in split['train']:
            t=read_trial(data/file,False);g=truth(file)
            offset=float(t['times'][g]-t['times'][t['onset']])/1e9
            templates.setdefault(t['condition'],[]).append(offset);all_offsets.append(offset)
        for file in split['validation']:
            item=transform_input(read_trial(data/file,False),config['input_transform'])
            stat=stats[s]
            x=torch.from_numpy(features(item['x'],stat['x_mean'],stat['x_std'],config['feature_mode']))[None]
            meta=torch.from_numpy(model_events(item,stat,config))[None]
            _,logits=model(x,meta,torch.tensor([config['sessions'].index(s)]),torch.tensor([config['conditions'].index(item['condition'])]),return_phase=True)
            # Only after inference do we retrieve the ground-truth go event.
            on,rew=item['onset'],item['reward'];go=truth(file)
            q=logits[0,on:rew+1].sigmoid().numpy().astype('float64').clip(1e-7,1-1e-7)
            y=np.arange(on,rew+1)>=go
            recall=[float(((q>=.5)[y==label]==label).mean()) for label in [False,True] if (y==label).any()]
            prefix=np.r_[0,np.cumsum(np.log1p(-q)-np.log(q))]
            boundary=on+int(np.argmax(prefix[:-1]))
            offsets=templates.get(item['condition'],all_offsets)
            clock_seconds=np.median(offsets)
            times=(item['times']-item['times'][on])/1e9
            clock_boundary=int(np.argmin(np.abs(times-clock_seconds)));clock_boundary=max(on,min(rew,clock_boundary))
            clock_pred=np.arange(on,rew+1)>=clock_boundary
            clock_recall=[float((clock_pred[y==label]==label).mean()) for label in [False,True] if (y==label).any()]
            rows.append(dict(session=s,trial=file,balanced_accuracy=float(np.mean(recall)),
                             go_mae_seconds=float(abs(item['times'][boundary]-item['times'][go]))/1e9,
                             clock_balanced_accuracy=float(np.mean(clock_recall)),
                             clock_go_mae_seconds=float(abs(item['times'][clock_boundary]-item['times'][go]))/1e9))
    sessions={s:average([{k:v for k,v in r.items() if k not in ['session','trial']} for r in rows if r['session']==s]) for s in sorted({r['session'] for r in rows})}
    result=dict(checkpoint=args.checkpoint,mean=average(list(sessions.values())),sessions=sessions,trials=rows,
                notes='CPU inference, no go cue or keypoint input. Clock baseline uses only same-session/condition training median offset. Boundary is maximum likelihood single change point; not used in coordinate decoding.')
    (folder/('phase_audit.json' if args.checkpoint=='checkpoint.pt' else 'phase_audit_final.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result['mean'])

if __name__=='__main__':main()
