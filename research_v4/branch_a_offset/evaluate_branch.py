"""Frozen v3 comparison; CSV go cue is used for diagnostics only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from runtime.data import read_trial, load_names, skeleton
from metrics import score_trial, average

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent

def aggregate(rows):
    sessions = sorted({r['session'] for r in rows})
    return average([average([{k:v for k,v in r.items() if k != 'session'} for r in rows if r['session']==s]) for s in sessions])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--confirmation', action='store_true')
    parser.add_argument('--replace-stacked', action='store_true')
    args=parser.parse_args()
    spec=json.loads((ROOT/'reference_v3.json').read_text())
    cfg=json.loads((ROOT/args.experiment/'config.json' if args.experiment else ROOT/'reference_config.json').read_text())
    expected=43 if args.confirmation else 42
    assert cfg['args']['split_seed']==expected
    members=[]
    for member in spec['members']:
        path=(REPO/'baseline_v3'/member['confirmation']/'validation_predictions.npz') if args.confirmation else (REPO/'baseline_v3'/member['development_predictions'])
        members.append((member['weight'],np.load(path.resolve())))
    candidate=np.load(ROOT/args.experiment/'validation_predictions.npz') if args.experiment else None
    if args.replace_stacked:
        assert candidate is not None
        index=next(i for i,m in enumerate(spec['members']) if m['group']=='stacked_large')
        replaced_weight,replaced_predictions=members[index]
    trials=[]
    for session,split in cfg['split'].items():
        folder=REPO/'player_data/Data_heldin'/session
        names=load_names(folder); _,edges=skeleton(names)
        train=[read_trial(folder/'heldin'/f,True)['y'] for f in split['train']]
        y=np.concatenate(train).reshape(-1,21,3)
        bones=np.median(np.linalg.norm(y[:,edges[:,0]]-y[:,edges[:,1]],axis=-1),axis=0)
        with (folder/'heldin/taskinfos.csv').open(encoding='utf-8-sig',newline='') as f:
            events={int(r['trial_index']):r for r in csv.DictReader(f)}
        for file in split['validation']:
            item=read_trial(folder/'heldin'/file,True); key=session+'_'+Path(file).stem
            row=events[int(Path(file).stem.split('_')[1])]
            go=int(row['cue_on_frame'])-int(row['window_start_frame'])
            assert item['onset']<=go<=item['reward']
            base=sum(w*a[key] for w,a in members)
            new=candidate[key] if candidate is not None else base
            if args.replace_stacked:
                new=base+replaced_weight*(new-replaced_predictions[key])
            trials.append((session,item,names,edges,bones,go,base,new))
    results=[]
    for alpha in ([0,1] if candidate is not None else [0]):
        scores=[]; phases={k:[] for k in ['pre_ton','pre_go','execution','post_reward','first10','pre_ton_after10']}
        for s,t,names,edges,bones,go,base,new in trials:
            pred=gaussian_filter1d((1-alpha)*base+alpha*new,1,axis=0,mode='nearest').reshape(-1,21,3)
            old=pred.copy()
            for j,(a,b) in enumerate(edges):
                d=old[:,b]-old[:,a]; length=np.linalg.norm(d,axis=-1,keepdims=True)
                pred[:,b]=pred[:,a]+d*(.5*length+.5*bones[j])/np.maximum(length,1e-6)
            scores.append(dict(session=s,**score_trial(t['y'],pred.reshape(-1,63),t['times'],names)))
            y=t['y'].reshape(-1,21,3); ew=np.linalg.norm(pred[:,0]-y[:,0],axis=-1)
            ep=np.linalg.norm((pred[:,1:]-pred[:,:1])-(y[:,1:]-y[:,:1]),axis=-1).mean(-1)
            idx=np.arange(len(y)); on,rew=t['onset'],t['reward']
            masks={'pre_ton':idx<on,'pre_go':(idx>=on)&(idx<go),'execution':(idx>=go)&(idx<=rew),'post_reward':idx>rew,
                   'first10':idx<min(on,10),'pre_ton_after10':(idx>=10)&(idx<on)}
            for name,mask in masks.items():
                if mask.any(): phases[name].append(dict(session=s,wrist_error=float(ew[mask].mean()),pose_error=float(ep[mask].mean()),fraction=float(mask.mean())))
        sessions={s:average([{k:v for k,v in r.items() if k!='session'} for r in scores if r['session']==s]) for s in sorted({r['session'] for r in scores})}
        result=dict(alpha=alpha,mode='replace_stacked' if args.replace_stacked else 'append',mean=aggregate(scores),sessions=sessions,phases={k:aggregate(v) for k,v in phases.items()})
        if results:
            result['delta']=result['mean']['hrfs_local']-results[0]['mean']['hrfs_local']
            result['sessions_improved']=sum(v['hrfs_local']>results[0]['sessions'][s]['hrfs_local'] for s,v in sessions.items())
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='sessions'},ensure_ascii=False),flush=True)
    target=ROOT/args.experiment if args.experiment else ROOT
    (target/('replacement.json' if args.replace_stacked else 'comparison.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__': main()
