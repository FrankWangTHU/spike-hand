import numpy as np
import torch
from runtime.model import SharedGRU

def main():
    torch.set_num_threads(2)
    config=dict(hidden=16,sessions=['s'],conditions=['c'],feature_dim=12,metadata=True,
                shared_input=True,adapter_scale=.1,architecture='stacked_dual_v3',phase_mode='none')
    torch.manual_seed(42); base=SharedGRU(config).eval()
    x=torch.randn(1,30,12); meta=torch.randn(1,30,12); sid=torch.zeros(1,dtype=torch.long)
    for mode in ['wrist','all']:
        torch.manual_seed(42); model=SharedGRU(dict(config,phase_mode=mode)).eval()
        for name,value in base.state_dict().items(): assert torch.equal(value,model.state_dict()[name]),name
        ref=base(x,meta,sid,sid)
        assert torch.equal(ref,model(x,meta,sid,sid))
        loss=model(x,meta,sid,sid).square().mean(); loss.backward()
        assert any(h[-1].weight.grad.abs().sum()>0 for h in model.phase_heads)
        with torch.no_grad():
            for h in model.phase_heads: h[-1].weight.normal_(std=.01)
        y=model(x,meta,sid,sid)
        batch_x=torch.cat([torch.nn.functional.pad(x,(0,0,0,9)),torch.randn(1,39,12)])
        batch_m=torch.cat([torch.nn.functional.pad(meta,(0,0,0,9)),torch.randn(1,39,12)])
        z=model(batch_x,batch_m,torch.zeros(2,dtype=torch.long),torch.zeros(2,dtype=torch.long),torch.tensor([30,39]))
        assert torch.allclose(y,z[:1,:30],atol=1e-6)
        if mode=='wrist': assert torch.equal(ref[:,:,3:],y[:,:,3:])
    print('PASS: matched baseline initialization, zero residual equivalence, gradients, variable-length padding, wrist-only isolation')

if __name__=='__main__': main()
