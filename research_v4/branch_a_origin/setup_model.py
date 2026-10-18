from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=ROOT/'runtime/model.py'
s=p.read_text()
needle='    @staticmethod\n    def reverse_valid'
s=s.replace(needle,'''        if config.get('anchor_decomposition',False):
            with torch.random.fork_rng(devices=[]):
                self.anchor_head=nn.Sequential(nn.Linear(3*output_size,128),nn.GELU(),nn.Linear(128,3))
                nn.init.zeros_(self.anchor_head[-1].weight)
                nn.init.zeros_(self.anchor_head[-1].bias)

    @staticmethod
    def reverse_valid''')
needle='        if return_phase:\n'
s=s.replace(needle,'''        if self.config.get('anchor_decomposition',False):
            idx=torch.arange(z.shape[1],device=z.device)[None]
            valid=torch.ones(z.shape[:2],dtype=torch.bool,device=z.device) if lengths is None else idx<lengths[:,None]
            early=valid & (meta[:,:,0]<0)
            initial=early & (idx<5)
            initial[:,0]=True
            def average_features(values,mask):
                return (values*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
            base_anchor=average_features(result[:,:,:3],initial)
            relative=result[:,:,:3]-base_anchor[:,None]
            context=torch.cat([average_features(z,initial),average_features(z,early),average_features(z,valid)],-1)
            origin=base_anchor.detach()+self.anchor_head(context)
            result=torch.cat([origin[:,None]+relative,result[:,:,3:]],-1)
        if return_phase:
''')
p.write_text(s,encoding='utf-8')
s=(ROOT.parent/'branch_a_pre_expert/train_wrist.py').read_text(encoding='utf-8-sig')
s=s.replace("config = {'phase_mode': args.phase_mode", "config = {'anchor_decomposition': True, 'phase_mode': args.phase_mode")
s=s.replace('        added = set()', "        comparable.pop('anchor_decomposition')\n        added = {'anchor_head.0.weight','anchor_head.0.bias','anchor_head.2.weight','anchor_head.2.bias'}")
s=s.replace("            if args.phase_mode == 'go4':\n                pred,phase_logits", "            if config.get('anchor_decomposition'):\n                pred,origin=model(x,meta,sid,cid,lengths,return_origin=True)\n            elif args.phase_mode == 'go4':\n                pred,phase_logits")
needle='            if args.phase_aux_weight:\n'
s=s.replace(needle,'''            # Identifiable initial-position + displacement targets in local standardized coordinates.
            idx=torch.arange(x.shape[1],device=device)[None]
            initial=mask & (meta[:,:,0]<0) & (idx<5)
            initial[:,0]=True
            target_anchor=(y[:,:,:3]*initial[:,:,None]).sum(1)/initial.sum(1)[:,None]
            initial_error=torch.linalg.vector_norm((origin-target_anchor)*scales[sid,:3],dim=-1).mean()/20
            relative_error=torch.linalg.vector_norm(((pred[:,:,:3]-origin[:,None])-(y[:,:,:3]-target_anchor[:,None]))*scales[sid,None,:3],dim=-1)
            loss=loss+.5*initial_error+.5*weighted_mean(relative_error/20,weight)
            if args.phase_aux_weight:
''')
(ROOT/'train.py').write_text(s,encoding='utf-8')
s=(ROOT.parent/'branch_a_pre_expert/run.py').read_text().replace('experiments/pre_ton','experiments/anchor')
(ROOT/'run.py').write_text(s,encoding='utf-8')
