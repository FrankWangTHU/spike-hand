"""Two independently loaded models; inference needs neural data and public Ton only."""
import numpy as np
from runtime.model import load_checkpoint, predict

class StitchedPredictor:
    def __init__(self, expert_checkpoint, a3_checkpoint, device, wrist_only=False, half_overlap=.2):
        if half_overlap <= 0:
            raise ValueError('half_overlap must be positive')
        self.expert=load_checkpoint(expert_checkpoint,device)
        self.a3=load_checkpoint(a3_checkpoint,device)
        self.device=device
        self.wrist_only=wrist_only
        self.half_overlap=half_overlap

    def predict(self, item, session):
        outputs=[]
        for model,stats,config,_ in [self.expert,self.a3]:
            outputs.append(predict(model,item,stats[session],config['sessions'].index(session),config['conditions'].index(item['condition']),self.device))
        early,late=outputs
        t=(item['times']-item['times'][item['onset']]).astype('float64')/1e9
        z=np.clip((t+self.half_overlap)/(2*self.half_overlap),0,1)
        weight=(1-z*z*(3-2*z)).astype('float32')[:,None]
        result=weight*early+(1-weight)*late
        if self.wrist_only:
            old=late.reshape(-1,21,3)
            result=(old-old[:,:1]+result.reshape(-1,21,3)[:,:1]).reshape(-1,63)
        return result
