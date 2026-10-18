"""Verify checkpoint reload and inference with labels physically removed."""
import json
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--experiment',default='denoising_complete')
args=p.parse_args()
experiment=root/'experiments'/args.experiment
cfg=json.loads((experiment/'config.json').read_text())
expected=np.load(experiment/'validation_predictions.npz')
data=root/'selftest/input'; output=root/'selftest'/args.experiment
references={}
for s,split in cfg['splits'].items():
    source=root.parent/'player_data/Data_heldin'/s
    dest=data/s; (dest/'heldout').mkdir(parents=True,exist_ok=True)
    np.save(dest/'keypoint_names.npy',np.load(source/'keypoint_names.npy',allow_pickle=False))
    name=split['validation'][0]
    with np.load(source/'heldin'/name,allow_pickle=False) as f:
        np.savez_compressed(dest/'heldout/trial_001.npz',**{k:f[k] for k in f.files if k!='keypoint'})
    references[s]=expected[s+'_'+Path(name).stem]
subprocess.run([sys.executable,str(root/'predict.py'),'--checkpoint',str(experiment/'checkpoint.pt'),'--data_dir',str(data),'--output_dir',str(output),'--device','cpu'],check=True)
errors=[]
assert len(list(output.glob('*_pred.h5')))==20
for s,ref in references.items():
    with h5py.File(output/(s+'_pred.h5')) as f:
        p=f['predicted_trajectory'][:]
        assert p.shape==ref.shape and p.dtype==np.float32 and np.isfinite(p).all()
        assert f['trial_indices'][:].tolist()==[1]
        assert f['trial_lengths'][:].tolist()==[len(ref)]
        assert f['keypoint_names'].shape==(21,)
        errors.append(float(np.max(np.abs(p-ref))))
assert max(errors)<1e-4,errors
result=dict(sessions=20,labels_removed=True,max_reload_error=max(errors),passed=True)
(output/'report.json').write_text(json.dumps(result,indent=2))
print(result)
