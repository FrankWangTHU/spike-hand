import numpy as np
import torch
from runtime.model import SharedGRU

torch.set_num_threads(2)
c=dict(hidden=16,sessions=['s'],conditions=['c'],feature_dim=12,metadata=True,shared_input=True,
       adapter_scale=.1,architecture='stacked_dual_v3',phase_mode='all')
torch.manual_seed(42);old=SharedGRU(c).eval()
with torch.no_grad():
    for head in old.phase_heads:head[-1].weight.normal_(std=.1)
state={k:v.clone() for k,v in old.state_dict().items()}
for suffix in ['0.weight','0.bias','2.weight','2.bias']:
    state['phase_heads.3.'+suffix]=state['phase_heads.2.'+suffix].clone()
    state['phase_heads.2.'+suffix]=state['phase_heads.1.'+suffix].clone()
new=SharedGRU(dict(c,phase_mode='go4')).eval()
status=new.load_state_dict(state,strict=False)
assert set(status.missing_keys)=={'stage_classifier.weight','stage_classifier.bias'}
with torch.no_grad():
    noise=torch.randn(new.phase_heads[1][-1].weight.shape,generator=torch.Generator().manual_seed(1701))*.001
    new.phase_heads[1][-1].weight.add_(noise);new.phase_heads[2][-1].weight.sub_(noise)
x=torch.randn(2,30,12);meta=torch.randn(2,30,12);sid=torch.zeros(2,dtype=torch.long);lengths=torch.tensor([23,30])
ref=old(x,meta,sid,sid,lengths)
pred,logits=new(x,meta,sid,sid,lengths,return_phase=True)
assert torch.allclose(ref,pred,atol=1e-6)
assert torch.equal(logits,torch.zeros_like(logits))
assert torch.allclose(pred[:1,:23],new(x[:1,:23],meta[:1,:23],sid[:1],sid[:1]),atol=1e-6)
pred.square().mean().backward()
assert new.stage_classifier.weight.grad.abs().sum()>0,'unsupervised gate stuck in symmetry'
new.zero_grad();pred,logits=new(x,meta,sid,sid,lengths,return_phase=True)
torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.ones_like(logits)).backward()
assert new.stage_classifier.weight.grad.abs().sum()>0
print('PASS: warm-start equivalence, padding invariance, unsupervised and supervised gate gradients; forward takes no go-cue input')
