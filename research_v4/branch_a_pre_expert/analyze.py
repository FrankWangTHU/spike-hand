import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from train import blend_weight
from runtime.data import read_trial
from evaluate_branch import aggregate

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
folder=ROOT/'experiments/pre_ton'
cfg=json.loads((folder/'config.json').read_text())
expert=np.load(folder/'expert_predictions.npz')
a3=np.load(ROOT.parent/'branch_a/experiments/a3_all/validation_predictions.npz')
primary=np.load(folder/'validation_predictions.npz')
variants=[('primary',.2,False),('overlap_100ms',.1,False),('overlap_400ms',.4,False),('wrist_only',.2,True)]
results=[]
for name,width,wrist_only in variants:
    predictions={}; diagnostics=[]
    for session,split in cfg['split'].items():
        for file in split['validation']:
            key=session+'_'+Path(file).stem
            item=read_trial(REPO/'player_data/Data_heldin'/session/'heldin'/file,True)
            w=blend_weight(item['times'],item['onset'],width)[:,None]
            p=w*expert[key]+(1-w)*a3[key]
            if wrist_only:
                points=p.reshape(-1,21,3)
                old=a3[key].reshape(-1,21,3)
                p=(old-old[:,:1]+points[:,:1]).reshape(-1,63)
            assert np.isfinite(p).all()
            if name=='primary': np.testing.assert_array_equal(p,primary[key])
            t=(item['times']-item['times'][item['onset']])/1e9
            if not wrist_only: np.testing.assert_array_equal(p[t>=width],a3[key][t>=width])
            y=item['y'].reshape(-1,21,3); p3=p.reshape(-1,21,3); old=a3[key].reshape(-1,21,3)
            row={'session':session}
            for region,mask in [('pre',t<0),('boundary',abs(t)<=.2)]:
                if not mask.any():continue
                for label,v in [('expert_splice',p3),('a3',old)]:
                    row[f'{region}_{label}_wrist']=float(np.linalg.norm(v[:,0]-y[:,0],axis=-1)[mask].mean())
                    row[f'{region}_{label}_pose']=float(np.linalg.norm((v[:,1:]-v[:,:1])-(y[:,1:]-y[:,:1]),axis=-1).mean(-1)[mask].mean())
            dt=np.diff(item['times'])/1e9
            near=(abs(t[1:])<=.2)&(abs(t[:-1])<=.2)
            for label,v in [('expert_splice',p3),('a3',old)]:
                velocity_error=np.linalg.norm(np.diff(v[:,0]-y[:,0],axis=0)/dt[:,None],axis=-1)
                row[f'boundary_{label}_wrist_velocity_error']=float(velocity_error[near].mean())
            diagnostics.append(row); predictions[key]=p
    target=folder if name=='primary' else ROOT/'experiments'/name
    if name!='primary':
        target.mkdir(exist_ok=True)
        (target/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        np.savez_compressed(target/'validation_predictions.npz',**predictions)
        with (target/'replacement.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-X','utf8','evaluate_branch.py','--experiment',str(target),'--replace-stacked'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    comparisons=json.loads((target/'replacement.json').read_text())
    full=next(r for r in comparisons if r['alpha']==1)
    results.append(dict(name=name,half_overlap_seconds=width,wrist_only=wrist_only,full_replacement=full,raw_diagnostics=aggregate(diagnostics)))
summary=dict(best_epoch=cfg['best_epoch'],split=42,submissions_used=0,baseline_v3=52.06467483592634,baseline_a3=52.23365738873004,primary_predeclared='primary',sensitivity_variants_not_independent_confirmation=True,results=results)
(ROOT/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
files=['train.py','run.py','analyze.py','runtime/model.py']
hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
hashes['initial_A3']=hashlib.sha256((ROOT.parent/'branch_a/experiments/a3_all/checkpoint.pt').read_bytes()).hexdigest()
assert hashes['initial_A3']=='24f7758822264ede9fef9ab7f770b49d0dfe736046abc0268104fcef27a48844'
(ROOT/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print(json.dumps({r['name']:{'score':r['full_replacement']['mean']['hrfs_local'],'phase':r['full_replacement']['phases']['pre_ton'],'raw':r['raw_diagnostics']} for r in results},indent=2))
