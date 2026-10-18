import json
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
from calibrator import Calibrator,features,correction_weight,apply_offset
from runtime.data import read_trial
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
outer=json.loads((ROOT.parent/'branch_a/experiments/a3_all/config.json').read_text())
folds=json.loads((ROOT/'folds.json').read_text())
rows=[];seen=set()
for fold,split in enumerate(folds):
    predictions=np.load(ROOT/'experiments'/f'fold{fold}'/'validation_predictions.npz')
    for s,partition in split.items():
        assert not set(partition['train'])&set(partition['validation'])
        assert not (set(partition['train'])|set(partition['validation']))&set(outer['split'][s]['validation'])
        for file in partition['validation']:
            key=s+'_'+Path(file).stem
            assert key not in seen;seen.add(key)
            item=read_trial(REPO/'player_data/Data_heldin'/s/'heldin'/file,True)
            n=item['onset']
            if n==0:continue
            p=predictions[key]
            numeric,neural=features(item,p)
            error=p[:n,:3].astype('float64')-item['y'][:n,:3]
            rows.append(dict(key=key,session=s,condition=item['condition'],fold=fold,numeric=numeric,neural=neural,offset=-error.mean(0),error=error,weight=correction_weight(item)[:n]))
assert len(seen)==sum(len(v['train']) for v in outer['split'].values())

def error_score(rows,offsets,strength):
    sessions={}
    for r,b in zip(rows,offsets):
        error=float(np.linalg.norm(r['error']+r['weight'][:,None]*strength*b,axis=-1).mean())
        sessions.setdefault(r['session'],[]).append(error)
    return float(np.mean([np.mean(v) for v in sessions.values()]))

# All choices below use only outer-training trials. This lightweight calibration
# CV is not a fully nested re-training of the upstream neural decoder.
choices=[]
for kind in ['group','metadata','neural']:
    options=[]
    for alpha in ([100.] if kind=='group' else [10.,100.,1000.]):
        held_predictions=np.zeros((len(rows),3))
        for fold in range(3):
            tr=[r for r in rows if r['fold']!=fold]
            ids=[i for i,r in enumerate(rows) if r['fold']==fold]
            model=Calibrator(kind,alpha).fit(tr)
            held_predictions[ids]=model.predict([rows[i] for i in ids])
        for strength in [0.,.25,.5,1.]:
            options.append(dict(kind=kind,alpha=alpha,strength=strength,inner_error=error_score(rows,held_predictions,strength)))
    best=min(options,key=lambda r:r['inner_error'])
    choices.append(best)
    (ROOT/f'{kind}_selection.json').write_text(json.dumps(options,indent=2),encoding='utf-8')

# Freeze all calibration choices before reading outer-validation coordinate labels.
(ROOT/'selection.json').write_text(json.dumps(dict(oof_trials=len(seen),calibration_trials=len(rows),choices=choices,selection_uses_outer_labels=False,half_overlap=.2),indent=2),encoding='utf-8')
outer_predictions=np.load(ROOT.parent/'branch_a/experiments/a3_all/validation_predictions.npz')
for choice in choices:
    kind=choice['kind'];folder=ROOT/'experiments'/('calibrated_'+kind)
    folder.mkdir(exist_ok=False)
    model=Calibrator(kind,choice['alpha']).fit(rows)
    with (folder/'calibrator.pkl').open('wb') as f:pickle.dump(dict(model=model,choice=choice),f)
    predictions={};offsets={}
    for s,split in outer['split'].items():
        for file in split['validation']:
            key=s+'_'+Path(file).stem
            item=read_trial(REPO/'player_data/Data_heldin'/s/'heldin'/file,False)
            assert 'y' not in item
            p=outer_predictions[key];numeric,neural=features(item,p)
            b=model.predict([dict(session=s,condition=item['condition'],numeric=numeric,neural=neural)])[0]
            corrected=apply_offset(item,p,b,choice['strength'])
            weight=correction_weight(item)
            np.testing.assert_array_equal(corrected[weight==0],p[weight==0])
            assert np.isfinite(corrected).all()
            predictions[key]=corrected;offsets[key]=b.tolist()
    np.savez_compressed(folder/'validation_predictions.npz',**predictions)
    (folder/'offsets.json').write_text(json.dumps(offsets,indent=2),encoding='utf-8')
    (folder/'config.json').write_text(json.dumps(outer,indent=2),encoding='utf-8')
    with (folder/'replacement.log').open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-X','utf8','evaluate_branch.py','--experiment',str(folder),'--replace-stacked'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',choice,flush=True)
