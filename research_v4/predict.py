"""Research checkpoint inference; inputs may have no keypoint field."""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import h5py
from train_latent import Decoder
from runtime.data import read_trial, transform_input, event_features, to_global, session_paths, trial_paths, load_names


def load(path, device):
    c = torch.load(path, map_location='cpu', weights_only=True)
    cfg = c['config']
    model = Decoder(len(cfg['sessions']),len(cfg['conditions']),cfg['args']['hidden'],cfg['args'].get('manifold',False)).to(device)
    model.load_state_dict(c['state']); model.eval()
    stats={s:{k:v.numpy() for k,v in st.items()} for s,st in c['stats'].items()}
    return model,stats,cfg


@torch.inference_mode()
def predict(model, stats, cfg, session, item, device):
    stat=stats[session]
    z=np.clip((item['x']-stat['xm'])/stat['xs'],-8,8)
    x=np.concatenate([gaussian_filter1d(z,2,axis=0),gaussian_filter1d(z,8,axis=0)],-1)
    ix=np.unique(np.r_[np.arange(0,len(z),3),len(z)-1])
    p,_=model(torch.tensor(x[ix],device=device)[None],torch.tensor(event_features(item)[ix],device=device)[None],torch.tensor([cfg['sessions'].index(session)],device=device),torch.tensor([cfg['conditions'].index(item['condition'])],device=device),torch.ones(1,len(ix),1,device=device))
    local=p[0].cpu().numpy()*stat['ys']+stat['ym']
    full=np.stack([np.interp(np.arange(len(z)),ix,local[:,j]) for j in range(63)],-1).astype('float32')
    return to_global(full,0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data_dir',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu')
    args=p.parse_args(); torch.set_num_threads(4)
    device=torch.device('cuda' if args.device=='cuda' or args.device=='auto' and torch.cuda.is_available() else 'cpu')
    model,stats,cfg=load(args.checkpoint,device)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    for session in session_paths(args.data_dir,'heldout'):
        trials=[transform_input(read_trial(f,False),'raw') for f in trial_paths(session,'heldout')]
        predictions=[predict(model,stats,cfg,session.name,t,device) for t in trials]
        values=np.concatenate(predictions).astype('float32')
        assert np.isfinite(values).all()
        with h5py.File(out/(session.name+'_pred.h5'),'w') as f:
            f['predicted_trajectory']=values
            f['trial_lengths']=np.array([len(v) for v in predictions],dtype='int64')
            f['trial_indices']=np.arange(1,len(trials)+1,dtype='int64')
            f['keypoint_names']=np.array(load_names(session),dtype=h5py.string_dtype('utf-8'))
        print(session.name,values.shape,flush=True)


if __name__=='__main__':
    main()
