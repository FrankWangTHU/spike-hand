"""Small train-only residual calibrators; no coordinate labels at inference."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

def features(item,prediction):
    n=item['onset']
    count=max(1,n)
    p=prediction[:count].reshape(-1,21,3)[:,0].astype('float64')
    x=item['x'][:count].astype('float64')  # read_trial's log1p TX and SBP
    middle=max(1,len(x)//2)
    neural=np.concatenate([x.mean(0),x[:middle].mean(0),x[middle:].mean(0) if middle<len(x) else x[-1],x.std(0)])
    numeric=np.concatenate([p.mean(0),p[:min(5,count)].mean(0),p[-1],p.std(0),p[-1]-p[0],[(item['times'][count-1]-item['times'][0])/1e9,count]])
    assert np.isfinite(neural).all() and np.isfinite(numeric).all()
    return numeric,neural

class Calibrator:
    def __init__(self,kind,alpha=100.):
        self.kind=kind;self.alpha=alpha

    def fit(self,rows):
        self.sessions=sorted({r['session'] for r in rows})
        self.conditions=sorted({r['condition'] for r in rows})
        target=np.stack([r['offset'] for r in rows])
        weights=np.array([len(rows)/(len(self.sessions)*sum(t['session']==r['session'] for t in rows)) for r in rows])
        self.limit=float(np.quantile(np.linalg.norm(target,axis=-1),.95))
        if self.kind=='group':
            self.global_mean=np.average(target,axis=0,weights=weights)
            self.session_shift={};self.group_shift={}
            for s in self.sessions:
                index=[i for i,r in enumerate(rows) if r['session']==s]
                self.session_shift[s]=(target[index]-self.global_mean).sum(0)/(len(index)+5)
                for c in self.conditions:
                    ids=[i for i in index if rows[i]['condition']==c]
                    if ids:self.group_shift[s,c]=(target[ids]-self.global_mean-self.session_shift[s]).sum(0)/(len(ids)+5)
        else:
            self.numeric_scaler=StandardScaler().fit(np.stack([r['numeric'] for r in rows]))
            if self.kind=='neural':
                neural=np.stack([r['neural'] for r in rows])
                self.neural_scaler=StandardScaler().fit(neural)
                self.pca=PCA(n_components=16,svd_solver='randomized',random_state=42,whiten=True).fit(self.neural_scaler.transform(neural))
            self.regressor=Ridge(alpha=self.alpha).fit(self.design(rows),target,sample_weight=weights)
        return self

    def design(self,rows):
        groups=np.zeros((len(rows),len(self.sessions)+len(self.conditions)))
        for i,r in enumerate(rows):
            if r['session'] in self.sessions:groups[i,self.sessions.index(r['session'])]=1
            if r['condition'] in self.conditions:groups[i,len(self.sessions)+self.conditions.index(r['condition'])]=1
        numeric=self.numeric_scaler.transform(np.stack([r['numeric'] for r in rows]))
        blocks=[groups,numeric]
        if self.kind=='neural':blocks.append(self.pca.transform(self.neural_scaler.transform(np.stack([r['neural'] for r in rows]))))
        return np.concatenate(blocks,axis=1)

    def predict(self,rows):
        if self.kind=='group':
            offsets=np.stack([self.global_mean+self.session_shift.get(r['session'],np.zeros(3))+self.group_shift.get((r['session'],r['condition']),np.zeros(3)) for r in rows])
        else:offsets=self.regressor.predict(self.design(rows))
        length=np.linalg.norm(offsets,axis=-1,keepdims=True)
        return offsets*np.minimum(1,self.limit/np.maximum(length,1e-8))

def correction_weight(item,half_width=.2):
    t=(item['times']-item['times'][item['onset']]).astype('float64')/1e9
    z=np.clip((t+half_width)/(2*half_width),0,1)
    return (1-z*z*(3-2*z)).astype('float32')

def apply_offset(item,prediction,offset,strength):
    if item['onset']==0:
        return prediction.copy()
    p=prediction.reshape(-1,21,3)
    return (p+correction_weight(item)[:,None,None]*(strength*np.asarray(offset,dtype='float32'))[None,None,:]).reshape(-1,63)
