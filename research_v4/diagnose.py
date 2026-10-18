"""Frozen held-in validation error audit. Oracle edits are diagnostic ONLY."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from runtime.data import read_trial, load_names, skeleton, event_features
from metrics import score_trial, average

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis'; OUT.mkdir(exist_ok=True)
REPO=ROOT.parent
cfg=json.loads((ROOT/'experiments/tcn_pair/config.json').read_text())
spec=json.loads((ROOT/'reference_ensemble.json').read_text())[0]
groups={k:[np.load((REPO/'baseline_v3'/p).resolve()) for p in paths] for k,paths in spec['groups'].items() if spec['group_weights'][k]}
new=np.load(ROOT/'experiments/tcn_pair/validation_predictions.npz')
v3spec=json.loads((OUT/'v3_selection_snapshot.json').read_text())
v3members=[(m['weight'],np.load((REPO/'baseline_v3'/m['development_predictions']).resolve())) for m in v3spec['members']]

def post(p,bones,edges):
    p=gaussian_filter1d(p,1,axis=0,mode='nearest').reshape(-1,21,3).copy()
    old=p.copy()
    for i,(a,b) in enumerate(edges):
        d=old[:,b]-old[:,a]; n=np.linalg.norm(d,axis=-1,keepdims=True)
        p[:,b]=p[:,a]+d*(.5*n+.5*bones[i])/np.maximum(n,1e-6)
    return p

def summary(rows):
    sessions=sorted({r['session'] for r in rows})
    return average([average([{k:v for k,v in r.items() if isinstance(v,(float,int))} for r in rows if r['session']==s]) for s in sessions])

def corr(a,b):
    u=a-a.mean(0);v=b-b.mean(0)
    den=np.sqrt((u*u).sum(0)*(v*v).sum(0)); ok=den>1e-10
    return float(np.mean((u*v).sum(0)[ok]/den[ok])) if ok.any() else 0.

rows=[]; oracle={}; phases=[]; fingers=[]; conditions=[]; cases=[]; lags={str(i):[] for i in range(-8,9,2)}
training_counts=[]
for s,split in cfg['splits'].items():
    folder=REPO/'player_data/Data_heldin'/s; names=load_names(folder); _,edges=skeleton(names)
    train=[read_trial(folder/'heldin'/f,True) for f in split['train']]
    yy=np.concatenate([t['y'] for t in train]).reshape(-1,21,3)
    bones=np.median(np.linalg.norm(yy[:,edges[:,0]]-yy[:,edges[:,1]],axis=-1),axis=0)
    for c in sorted({t['condition'] for t in train}):
        training_counts.append(dict(session=s,condition=c,count=sum(t['condition']==c for t in train)))
    for f in split['validation']:
        t=read_trial(folder/'heldin'/f,True); key=s+'_'+Path(f).stem; y=t['y'].reshape(-1,21,3).astype('float64')
        base=sum(spec['group_weights'][k]*np.mean([a[key] for a in arr],0) for k,arr in groups.items())
        p=post(.75*base+.25*new[key],bones,edges).astype('float64')
        p3=post(sum(w*a[key] for w,a in v3members),bones,edges)
        def score(z):return score_trial(t['y'],z.reshape(-1,63),t['times'],names)
        def save_oracle(name,z):oracle.setdefault(name,[]).append(dict(session=s,**score(z)))
        save_oracle('current',p); save_oracle('latest_v3',p3)
        yl=y[:,1:]-y[:,:1];pl=p[:,1:]-p[:,:1]
        errw=np.linalg.norm(p[:,0]-y[:,0],axis=-1);errp=np.linalg.norm(pl-yl,axis=-1)
        dt=np.median(np.diff(t['times']))/1e9
        vy=np.diff(yl,axis=0).reshape(len(y)-1,-1)/dt;vp=np.diff(pl,axis=0).reshape(len(y)-1,-1)/dt
        vrmse=float(np.sqrt(np.mean((vy-vp)**2))); vc=corr(vy,vp)
        on,rew=t['onset'],t['reward']
        wrist_speed=np.r_[0,np.linalg.norm(np.diff(y[:,0],axis=0),axis=1)/dt]
        delta=y[:,0]-p[:,0]
        q=p+delta.mean(0)[None,None,:];save_oracle('oracle_constant_wrist_translation',q)
        q=p+delta[:max(on,1)].mean(0)[None,None,:];save_oracle('oracle_pre_onset_wrist_translation',q)
        q=p+delta[:,None,:];save_oracle('oracle_perfect_wrist',q)
        local_offset=(yl-pl).mean(0)
        q=p.copy();q[:,1:]+=local_offset;save_oracle('oracle_constant_local_shape_offset',q)
        q=p.copy();q[:,1:]=p[:,:1]+yl;save_oracle('oracle_perfect_local_pose',q)
        q=p.copy();q[:on]=y[:on];save_oracle('oracle_perfect_pre_onset_all',q)
        q=p.copy();q[:on]=.5*(y[:on]+p[:on]);save_oracle('oracle_half_pre_onset_error',q)
        pp=p.reshape(-1,63)
        lag_scores=[]
        for lag in range(-8,9,2):
            q=pp[np.clip(np.arange(len(y))+lag,0,len(y)-1)]
            sc=score(q);lags[str(lag)].append(dict(session=s,**sc));lag_scores.append((sc['hrfs_local'],lag))
        # Templates use only same-session/condition training labels, event-aligned.
        same=[u for u in train if u['condition']==t['condition']]
        phase=event_features(t)[:,2]
        template=np.mean([np.stack([np.interp(phase,event_features(u)[:,2],u['y'][:,j]) for j in range(63)],axis=-1) for u in same],axis=0).reshape(-1,21,3)
        save_oracle('train_only_event_template',template)
        tl=template[:,1:]-template[:,:1]
        wrist_relative_sse=float(np.sum((p[:,0]-y[:,0])**2)/max(np.sum((template[:,0]-y[:,0])**2),1e-9))
        pose_relative_sse=float(np.sum((pl-yl)**2)/max(np.sum((tl-yl)**2),1e-9))
        lowy=gaussian_filter1d(yl,3,axis=0);lowp=gaussian_filter1d(pl,3,axis=0)
        lowcorr=corr(np.diff(lowy,axis=0).reshape(len(y)-1,-1),np.diff(lowp,axis=0).reshape(len(y)-1,-1))
        highy=yl-lowy; highp=pl-lowp
        error=pl-yl;lowerr=gaussian_filter1d(error,3,axis=0)
        r=dict(session=s,trial=f,condition=t['condition'],**score(p),wrist_error=float(errw.mean()),pose_error=float(errp.mean()),
               velocity_rmse=vrmse,velocity_corr=vc,velocity_exp_score=float(50*np.exp(-vrmse/5)),velocity_corr_score=50*max(0,vc),
               velocity_rms_ratio=float(np.sqrt(np.mean(vp**2)/max(np.mean(vy**2),1e-12))),
               wrist_constant_offset_norm=float(np.linalg.norm(delta.mean(0))),wrist_bias_mse_fraction=float(np.sum(delta.mean(0)**2)/np.mean(np.sum(delta**2,axis=1))),
               pose_bias_mse_fraction=float(np.sum(error.mean(0)**2)/np.mean(np.sum(error**2,axis=(1,2)))),
               pose_lowpass_error_energy_ratio=float(np.sum(lowerr**2)/np.sum(error**2)),
               smoothed_velocity_corr=lowcorr,highpass_pose_corr=corr(highy.reshape(len(y),-1),highp.reshape(len(y),-1)),
               pre_fraction=on/len(y),best_oracle_lag=max(lag_scores)[1],
               pre_wrist_error_share=float(errw[:on].sum()/max(errw.sum(),1e-9)),
               pre_pose_error_share=float(errp[:on].sum()/max(errp.sum(),1e-9)),
               wrist_error_sse_over_template=wrist_relative_sse,pose_error_sse_over_template=pose_relative_sse,
               same_condition_train_trials=len(same))
        rows.append(r)
        for phase_name,mask in [('pre_onset',np.arange(len(y))<on),('action',(np.arange(len(y))>=on)&(np.arange(len(y))<=rew)),('post_reward',np.arange(len(y))>rew)]:
            if mask.any():phases.append(dict(session=s,phase=phase_name,wrist_error=float(errw[mask].mean()),pose_error=float(errp[mask].mean()),fraction=float(mask.mean()),true_wrist_speed=float(wrist_speed[mask].mean())))
        for j,name in enumerate(['Thumb','Index','Middle','Ring','Pinky']):
            fingers.append(dict(session=s,finger=name,error=float(errp[:,j*4:j*4+4].mean()),tip_error=float(errp[:,j*4+3].mean())))
        cases.append(dict(key=key,y=y,p=p,times=(t['times']-t['times'][0])/1e9,onset=on,reward=rew,score=r['hrfs_local']))

out=dict(model='frozen research_v4 52.0699 blend',validation_trials=len(rows),aggregation='Trials equally within sessions then sessions equally; phase subsets condition on presence.',
    mean=summary(rows),sessions={s:summary([r for r in rows if r['session']==s]) for s in cfg['sessions']},
    phases={k:summary([r for r in phases if r['phase']==k]) for k in ['pre_onset','action','post_reward']},
    fingers={k:summary([r for r in fingers if r['finger']==k]) for k in ['Thumb','Index','Middle','Ring','Pinky']},
    conditions={k:summary([r for r in rows if r['condition']==k]) for k in cfg['conditions']},
    diagnostic_counterfactuals={k:summary(v) for k,v in oracle.items()},global_lag_sweep={k:summary(v) for k,v in lags.items()},
    per_trial=rows,training_counts=training_counts,
    caveats=['Oracle corrections use validation truth ONLY for error attribution; not deployable predictions or achieved scores.','Label high-frequency variation is not proven to be measurement noise.','Diagnostic results do not establish causal mechanisms or expected leaderboard gains.'])
(OUT/'diagnosis.json').write_text(json.dumps(out,indent=2),encoding='utf8')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ordered=sorted(cases,key=lambda c:c['score'])
    selected=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    fig,axs=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for axrow,c in zip(axs,selected):
        for j in range(3):
            axrow[0].plot(c['times'],c['y'][:,0,j],label=f'True {"xyz"[j]}',lw=1.4)
            axrow[0].plot(c['times'],c['p'][:,0,j],'--',label=f'Pred {"xyz"[j]}',lw=1.2)
        y=c['y'][:,8]-c['y'][:,0];p=c['p'][:,8]-c['p'][:,0]
        axrow[1].plot(c['times'],np.linalg.norm(y,axis=1),label='True index-tip distance')
        axrow[1].plot(c['times'],np.linalg.norm(p,axis=1),'--',label='Predicted')
        for ax in axrow:
            ax.axvline(c['times'][c['onset']],color='gray',ls=':',label='Onset')
            ax.axvline(c['times'][c['reward']],color='black',ls=':',label='Reward')
            ax.set_xlabel('Seconds');ax.set_ylabel('Original coordinate units');ax.legend(fontsize=7,ncol=2)
        axrow[0].set_title(c['key']+f' | HRFS {c["score"]:.2f}')
        axrow[1].set_title('Index-tip distance from wrist')
    fig.savefig(OUT/'trajectory_examples.png',dpi=150)
    plt.close(fig)
except ImportError as e:
    print('Plot unavailable',e)
print(json.dumps({k:out[k] for k in ['mean','phases','fingers','diagnostic_counterfactuals']},indent=2),flush=True)
