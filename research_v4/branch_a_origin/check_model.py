from pathlib import Path
import numpy as np
import torch
from runtime.model import SharedGRU
torch.set_num_threads(2)
root=Path(__file__).resolve().parent
state=torch.load(root.parent/'branch_a/experiments/a3_all/checkpoint.pt',map_location='cpu',weights_only=True)
config=state['config']
old=SharedGRU(config).eval(); old.load_state_dict(state['state'])
new=SharedGRU(dict(config,anchor_decomposition=True)).eval()
status=new.load_state_dict(state['state'],strict=False)
assert set(status.missing_keys)=={'anchor_head.0.weight','anchor_head.0.bias','anchor_head.2.weight','anchor_head.2.bias'}
torch.manual_seed(42)
x=torch.randn(2,60,config['feature_dim']); meta=torch.zeros(2,60,12)
meta[:,:,0]=(torch.arange(60)[None]/30-.5)/4
sid=torch.tensor([0,1]); cid=torch.tensor([0,1]); lengths=torch.tensor([60,43])
a=old(x,meta,sid,cid,lengths)
b,anchor=new(x,meta,sid,cid,lengths,return_origin=True)
torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
torch.testing.assert_close((b[:,:5,:3]-anchor[:,None]).mean(1),torch.zeros(2,3),atol=1e-6,rtol=0)
padded=new(torch.nn.functional.pad(x,(0,0,0,20)),torch.nn.functional.pad(meta,(0,0,0,20)),sid,cid,lengths)
for i,n in enumerate(lengths):torch.testing.assert_close(b[i,:n],padded[i,:n],atol=1e-5,rtol=1e-5)
(b[:,:10,:3].square().mean()+anchor.square().mean()).backward()
assert new.anchor_head[-1].weight.grad.abs().sum()>0
print('PASS initial equivalence, anchored relative displacement, padding, anchor gradients')
