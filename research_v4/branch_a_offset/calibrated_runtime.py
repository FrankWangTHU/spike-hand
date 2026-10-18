import pickle
from calibrator import features,apply_offset
from runtime.data import transform_input
from runtime.model import load_checkpoint,predict

class CalibratedPredictor:
    """Input is an untransformed read_trial item (log1p neural channels)."""
    def __init__(self,a3_checkpoint,calibrator_path,device):
        self.model,self.stats,self.config,self.names=load_checkpoint(a3_checkpoint,device)
        with open(calibrator_path,'rb') as f:bundle=pickle.load(f)
        self.calibrator=bundle['model'];self.choice=bundle['choice'];self.device=device

    def predict(self,item,session):
        base_item=transform_input(dict(item),self.config['input_transform'])
        base=predict(self.model,base_item,self.stats[session],self.config['sessions'].index(session),self.config['conditions'].index(item['condition']),self.device)
        numeric,neural=features(item,base)
        offset=self.calibrator.predict([dict(session=session,condition=item['condition'],numeric=numeric,neural=neural)])[0]
        return apply_offset(item,base,offset,self.choice['strength'])
