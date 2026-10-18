import json
from pathlib import Path
import numpy as np
from runtime.data import read_trial
ROOT=Path(__file__).resolve().parent
rows=[]
for folder in sorted((ROOT/'experiments').glob('fold*')):
    if not (folder/'complete.json').exists():continue
    cfg=json.loads((folder/'config.json').read_text())
    predictions=np.load(folder/'validation_predictions.npz')
    for session,split in cfg['split'].items():
        for file in split['validation']:
            t=read_trial(ROOT.parent.parent/'player_data/Data_heldin'/session/'heldin'/file,True)
            n=t['onset']
            if n==0:continue
            p=predictions[session+'_'+Path(file).stem]
            error=p[:n,:3].astype('float64')-t['y'][:n,:3]
            rows.append(dict(session=session,norm=float(np.linalg.norm(error.mean(0))),error=float(np.linalg.norm(error,axis=-1).mean()),frames=n))
if rows:
    result=dict(trials=len(rows),offset_norm_quantiles=np.quantile([r['norm'] for r in rows],[0,.5,.9,.95,1]).tolist(),pre_frames_quantiles=np.quantile([r['frames'] for r in rows],[0,.5,1]).tolist(),mean_trial_wrist_error=float(np.mean([r['error'] for r in rows])))
    (ROOT/'oof_health.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result)
