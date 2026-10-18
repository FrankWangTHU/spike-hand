import numpy as np
from calibrator import Calibrator,features,apply_offset,correction_weight
rng=np.random.default_rng(42)
rows=[]
for i in range(60):
    rows.append(dict(session=str(i%3),condition=str(i%2),numeric=rng.normal(size=17),neural=rng.normal(size=80),offset=rng.normal(size=3)))
for kind in ['group','metadata','neural']:
    model=Calibrator(kind,100).fit(rows)
    p=model.predict(rows[:5]);assert p.shape==(5,3) and np.isfinite(p).all()
    # Inference rows have no residual targets or coordinate labels.
    clean=[{k:v for k,v in r.items() if k!='offset'} for r in rows[:5]]
    np.testing.assert_array_equal(p,model.predict(clean))
item=dict(onset=20,times=np.arange(60,dtype=np.int64)*20000000,x=rng.normal(size=(60,512)))
prediction=rng.normal(size=(60,63)).astype('float32')
p=apply_offset(item,prediction,[1,2,3],.5)
np.testing.assert_array_equal(p[correction_weight(item)==0],prediction[correction_weight(item)==0])
a=p.reshape(-1,21,3);b=prediction.reshape(-1,21,3)
np.testing.assert_allclose(a-a[:,:1],b-b[:,:1],atol=1e-6,rtol=1e-6)
np.testing.assert_array_equal(apply_offset(item,prediction,[1,2,3],0),prediction)
np.testing.assert_array_equal(apply_offset(dict(item,onset=0),prediction,[1,2,3],1),prediction)
print('PASS label-free correction, output shapes, zero-strength identity, unchanged post-window and relative pose')
