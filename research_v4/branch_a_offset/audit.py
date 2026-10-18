import json
from pathlib import Path
import numpy as np
import torch
from calibrated_runtime import CalibratedPredictor
from runtime.data import read_trial
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
outer=json.loads((ROOT.parent/'branch_a/experiments/a3_all/config.json').read_text())
folds=json.loads((ROOT/'folds.json').read_text())
seen=set()
for index,fold in enumerate(folds):
    cfg=json.loads((ROOT/'experiments'/f'fold{index}'/'config.json').read_text())
    assert cfg['best_epoch']==150 and cfg['args']['last_epoch'] and cfg['args']['init_checkpoint'] is None
    assert cfg['split']==fold
    for s,split in fold.items():
        assert not set(split['train'])&set(split['validation'])
        assert not (set(split['train'])|set(split['validation']))&set(outer['split'][s]['validation'])
        for file in split['validation']:
            key=(s,file);assert key not in seen;seen.add(key)
assert len(seen)==401
results={}
for kind in ['group','metadata','neural']:
    folder=ROOT/'experiments'/('calibrated_'+kind)
    saved=np.load(folder/'validation_predictions.npz')
    model=CalibratedPredictor(ROOT.parent/'branch_a/experiments/a3_all/checkpoint.pt',folder/'calibrator.pkl',torch.device('cuda'))
    errors=[]
    for s,split in outer['split'].items():
        file=split['validation'][0]
        item=read_trial(REPO/'player_data/Data_heldin'/s/'heldin'/file,False)
        assert 'y' not in item
        p=model.predict(item,s)
        target=saved[s+'_'+Path(file).stem]
        np.testing.assert_allclose(p,target,atol=1e-5,rtol=1e-6)
        errors.append(float(abs(p-target).max()))
    results[kind]=dict(trials=len(errors),max_abs_difference=max(errors),labels_read=False)
    del model
(ROOT/'runtime_audit.json').write_text(json.dumps(dict(oof_coverage=len(seen),folds_disjoint_from_outer_validation=True,runtime=results),indent=2),encoding='utf-8')
print(results)
