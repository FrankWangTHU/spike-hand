"""Checkpoint inference verification with no label access in prediction."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from runtime.model import load_checkpoint, predict
from runtime.data import read_trial, transform_input

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent

def main():
    p=argparse.ArgumentParser();p.add_argument('experiment');args=p.parse_args()
    folder=ROOT/args.experiment
    model,stats,config,names=load_checkpoint(folder/'checkpoint.pt',torch.device('cpu'))
    torch.set_num_threads(2)
    saved=np.load(folder/'validation_predictions.npz')
    manifest=json.loads((folder/'config.json').read_text())
    errors=[]; failures=[]; gpu_errors=[]
    gpu_model=None
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('highest')
        gpu_model,_,_,_=load_checkpoint(folder/'checkpoint.pt',torch.device('cuda'))
    for s,split in manifest['split'].items():
        file=split['validation'][0]
        item=transform_input(read_trial(REPO/'player_data/Data_heldin'/s/'heldin'/file,False),config['input_transform'])
        assert 'y' not in item
        pred=predict(model,item,stats[s],config['sessions'].index(s),config['conditions'].index(item['condition']),torch.device('cpu'))
        target=saved[s+'_'+Path(file).stem]
        assert pred.shape==target.shape and np.isfinite(pred).all()
        if not np.allclose(pred,target,atol=.01,rtol=.0001): failures.append(s)
        errors.append(float(np.max(np.abs(pred-target))))
        if gpu_model is not None:
            gpu=predict(gpu_model,item,stats[s],config['sessions'].index(s),config['conditions'].index(item['condition']),torch.device('cuda'))
            gpu_errors.append(float(np.max(np.abs(gpu-target))))
            assert np.allclose(gpu,target,atol=1e-5,rtol=1e-6)
    result=dict(trials=len(errors),max_abs_cpu_vs_saved=max(errors),labels_read_for_inference=False,strict_cpu_tolerance_failures=failures,
                max_abs_gpu_vs_saved=max(gpu_errors) if gpu_errors else None)
    (folder/'checkpoint_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result)

if __name__=='__main__':main()
