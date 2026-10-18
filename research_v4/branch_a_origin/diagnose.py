import json
from pathlib import Path
import numpy as np
from runtime.data import read_trial
from evaluate_branch import aggregate
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
cfg=json.loads((ROOT.parent/'branch_a/experiments/a3_all/config.json').read_text())
sources={'a3':ROOT.parent/'branch_a/experiments/a3_all/validation_predictions.npz','pre_expert':ROOT.parent/'branch_a_pre_expert/experiments/pre_ton/validation_predictions.npz'}
results={}
for name,path in sources.items():
    predictions=np.load(path); rows=[]; details=[]
    for session,split in cfg['split'].items():
        for file in split['validation']:
            item=read_trial(REPO/'player_data/Data_heldin'/session/'heldin'/file,True)
            n=item['onset']
            if n<2:continue
            p=predictions[session+'_'+Path(file).stem][:n].reshape(-1,21,3).astype('float64')
            y=item['y'][:n].reshape(-1,21,3).astype('float64')
            e=p[:,0]-y[:,0]; bias=e.mean(0); residual=e-bias
            mse=float(np.mean(np.sum(e**2,axis=-1))); bias2=float(np.sum(bias**2)); centered=float(np.mean(np.sum(residual**2,axis=-1)))
            assert np.isclose(mse,bias2+centered,atol=1e-8)
            anchor=e[:min(5,n)].mean(0)
            target_motion=y[:,0]-y[:,0].mean(0)
            row=dict(session=session,wrist_mse=mse,bias_squared=bias2,centered_mse=centered,bias_fraction=bias2/max(mse,1e-12),wrist_error=float(np.linalg.norm(e,axis=-1).mean()),oracle_mean_aligned_error=float(np.linalg.norm(residual,axis=-1).mean()),oracle_first5_aligned_error=float(np.linalg.norm(e-anchor,axis=-1).mean()),true_motion_rms=float(np.sqrt(np.mean(np.sum(target_motion**2,axis=-1)))),pred_motion_rms=float(np.sqrt(np.mean(np.sum((p[:,0]-p[:,0].mean(0))**2,axis=-1)))))
            rows.append(row); details.append(dict(trial=file,frames=n,**row))
    mean=aggregate(rows)
    mean['bias_share_of_aggregated_mse']=mean['bias_squared']/mean['wrist_mse']
    mean['trials_bias_dominant']=sum(r['bias_fraction']>.5 for r in rows)
    mean['trial_count']=len(rows)
    results[name]=dict(mean=mean,trials=details)
output=dict(results=results,scope='Unprocessed single-model wrist predictions on split42, Ton-exclusive prefix.',warning='Oracle alignment uses validation labels only for error diagnosis. It is not a deployable prediction or a valid model score.')
(ROOT/'diagnosis.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps({k:v['mean'] for k,v in results.items()},indent=2))
