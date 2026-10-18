import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from runtime.data import read_trial, load_names, skeleton
from metrics import score_trial, average

p=argparse.ArgumentParser()
p.add_argument('--experiment',default='experiments/denoising_complete')
p.add_argument('--reference',default='reference_ensemble.json')
p.add_argument('--output-name',default='comparison.json')
p.add_argument('--weights',type=float,nargs='+',default=[0,.1,.25,.5,1])
args=p.parse_args()
root=Path(__file__).resolve().parent
repo=root.parent
experiment=root/args.experiment
cfg=json.loads((experiment/'config.json').read_text())
spec=json.loads((root/args.reference).read_text())[0]
groups={k:[np.load((repo/'baseline_v3'/path).resolve()) for path in paths] for k,paths in spec['groups'].items() if spec['group_weights'][k]}
candidate=np.load(experiment/'validation_predictions.npz')
trials=[]
for s, split in cfg['splits'].items():
    folder=repo/'player_data/Data_heldin'/s
    names=load_names(folder)
    _,edges=skeleton(names)
    train=[read_trial(folder/'heldin'/f,True) for f in split['train']]
    y=np.concatenate([t['y'] for t in train]).reshape(-1,21,3)
    bones=np.median(np.linalg.norm(y[:,edges[:,1]]-y[:,edges[:,0]],axis=-1),axis=0)
    for f in split['validation']:
        key=s+'_'+Path(f).stem
        t=read_trial(folder/'heldin'/f,True)
        base=sum(spec['group_weights'][k]*np.mean([a[key] for a in arrays],axis=0) for k,arrays in groups.items())
        trials.append((s,t,names,edges,bones,base,candidate[key]))
results=[]
for weight in args.weights:
    rows={}
    for s,t,names,edges,bones,base,new in trials:
        pred=gaussian_filter1d((1-weight)*base+weight*new,1,axis=0,mode='nearest').reshape(-1,21,3)
        original=pred.copy()
        for i,(a,b) in enumerate(edges):
            vector=original[:,b]-original[:,a]
            length=np.linalg.norm(vector,axis=-1,keepdims=True)
            pred[:,b]=pred[:,a]+vector*(.5*length+.5*bones[i])/np.maximum(length,1e-6)
        rows.setdefault(s,[]).append(score_trial(t['y'],pred.reshape(-1,63),t['times'],names))
    result=dict(candidate_weight=weight,mean=average([average(r) for r in rows.values()]),sessions={s:average(r) for s,r in rows.items()})
    results.append(result)
    print(weight,result['mean'],flush=True)
(experiment/args.output_name).write_text(json.dumps(results,indent=2))
