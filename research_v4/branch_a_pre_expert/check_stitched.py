import json
from pathlib import Path
import numpy as np
import torch
from runtime.data import read_trial,transform_input
from stitched_runtime import StitchedPredictor
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
for name,wrist in [('pre_ton',False),('pre_ton_wrist',True)]:
    folder=ROOT/'experiments'/name
    cfg=json.loads((folder/'config.json').read_text())
    model=StitchedPredictor(folder/'checkpoint.pt',ROOT.parent/'branch_a/experiments/a3_all/checkpoint.pt',torch.device('cuda'),wrist_only=wrist)
    saved=np.load(folder/'validation_predictions.npz')
    errors=[]
    for session,split in cfg['split'].items():
        file=split['validation'][0]
        item=transform_input(read_trial(ROOT.parent.parent/'player_data/Data_heldin'/session/'heldin'/file,False),'raw')
        assert 'y' not in item
        p=model.predict(item,session)
        target=saved[session+'_'+Path(file).stem]
        np.testing.assert_allclose(p,target,atol=1e-5,rtol=1e-6)
        errors.append(float(abs(p-target).max()))
    result=dict(trials=len(errors),labels_read=False,max_abs_difference=max(errors))
    (folder/'stitched_runtime_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(name,result)
    del model
