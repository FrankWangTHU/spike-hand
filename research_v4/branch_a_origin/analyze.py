import hashlib
import json
from pathlib import Path
import numpy as np
from runtime.data import read_trial
from evaluate_branch import aggregate
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
folder=ROOT/'experiments/anchor'
cfg=json.loads((folder/'config.json').read_text())
baseline=next(r for r in json.loads((ROOT.parent/'branch_a/experiments/a3_all/replacement.json').read_text()) if r['alpha']==1)
candidate=next(r for r in json.loads((folder/'replacement.json').read_text()) if r['alpha']==1)
pred=np.load(folder/'expert_predictions.npz')
baseline_pred=np.load(ROOT.parent/'branch_a/experiments/a3_all/validation_predictions.npz')
rows=[]
for session,split in cfg['split'].items():
    for file in split['validation']:
        key=session+'_'+Path(file).stem
        item=read_trial(REPO/'player_data/Data_heldin'/session/'heldin'/file,True)
        n=item['onset']
        if n<2:continue
        target=item['y'][:n,:3].astype('float64')
        target_anchor=target[:min(5,n)].mean(0)
        row={'session':session}
        for label,data in [('a3',baseline_pred[key]),('new',pred[key])]:
            p=data[:n,:3].astype('float64'); anchor=p[:min(5,n)].mean(0)
            e=p-target; bias=e.mean(0)
            row[label+'_anchor_error']=float(np.linalg.norm(anchor-target_anchor))
            row[label+'_relative_error']=float(np.linalg.norm((p-anchor)-(target-target_anchor),axis=-1).mean())
            row[label+'_bias_norm']=float(np.linalg.norm(bias))
            row[label+'_wrist_error']=float(np.linalg.norm(e,axis=-1).mean())
        rows.append(row)
delta=np.array([candidate['sessions'][s]['hrfs_local']-baseline['sessions'][s]['hrfs_local'] for s in baseline['sessions']])
ci=np.quantile(np.random.default_rng(42).choice(delta,(10000,20)).mean(1),[.025,.975])
summary=dict(best_epoch=cfg['best_epoch'],submissions_used=0,development_split=42,baseline=baseline,candidate=candidate,delta_vs_a3=float(delta.mean()),sessions_improved_vs_a3=int((delta>0).sum()),descriptive_session_bootstrap95=ci.tolist(),raw_component_diagnostics=aggregate(rows),diagnostic_trials=len(rows),initial_diagnosis=json.loads((ROOT/'diagnosis.json').read_text())['results']['a3']['mean'])
(ROOT/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
provenance={name:digest(ROOT/name) for name in ['train.py','runtime/model.py','run.py','diagnose.py','analyze.py','check_model.py','stitched_runtime.py']}
provenance['initial_A3']=digest(ROOT.parent/'branch_a/experiments/a3_all/checkpoint.pt')
provenance['v3_zip']=digest(REPO/'baseline_v3/submission.zip')
assert provenance['initial_A3']=='24f7758822264ede9fef9ab7f770b49d0dfe736046abc0268104fcef27a48844'
assert provenance['v3_zip']=='1c4e29a7a62b311e0b8444af5f09bb947b16a86278d39e73390c198eabbd8372'
(ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
diagnosis=summary['initial_diagnosis']; components=summary['raw_component_diagnostics']
lines=['# Ton前位置偏移诊断与初始位置＋位移模型','', '本地划分42；401训练、169验证Trial，未提交平台。原A3与v3文件哈希未变。', '', '## 误差分解', '', '对每条Trial的Ton前腕部误差e(t)，令b=mean(e)，则mean(||e||²)=||b||²+mean(||e-b||²)。Trial先平均，再按Session等权汇总。诊断排除Ton前不足2帧的3条Trial，其余166条参与；数值与包含全部Trial的完整指标不可直接比较。', '', f"A3聚合平方误差中，整体偏移占{diagnosis['bias_share_of_aggregated_mse']:.2%}。120/166条Trial偏移占比超过50%。未经后处理的腕部平均距离误差{diagnosis['wrist_error']:.5f}，使用验证真值消除均值偏移后为{diagnosis['oracle_mean_aligned_error']:.5f}，仅用前5个有效点对齐后为{diagnosis['oracle_first5_aligned_error']:.5f}。这些是真值辅助诊断，不是可部署结果或模型得分。", '', '## 本轮结构与训练', '', '独立专家继续使用完整神经信号窗口。显式分解腕部p(t)=a+d(t)：将原解码器腕部输出减去前min(5,Ton索引)个点的均值，得到零初始均值的相对位移；起始位置头读取初始区间、Ton前、全窗口的GRU特征池化，预测起始位置修正。预测锚点=原解码器锚点（停止梯度）＋修正。新增头末层零初始化，初始输出保持A3等价。真实起始位置只在训练损失中使用，推理没有坐标或go cue输入。', '', '沿用前轮独立腕部专家：完整网络可训练，Ton附近±0.2秒smoothstep拼接，后段及手指相对姿态使用冻结A3；AdamW lr1e-4、60轮余弦、batch8、hidden192、channel dropout0.25。基础损失为早期腕部位置＋0.2倍标准化速度误差；新增0.5倍起始位置误差和0.5倍相对位移误差，距离项除以20。起始位置监督与相对位移监督共同固定分解含义。', '', f"固定拼接方式按完整轨迹原始本地分选择checkpoint，最佳epoch={cfg['best_epoch']}。不是直接按真值校正后的诊断分选模型。", '', '## 实测结果','', '| 指标 | A3 | 新模型 |','|---|---:|---:|',f"| 最终集成本地分 | {baseline['mean']['hrfs_local']:.8f} | {candidate['mean']['hrfs_local']:.8f} |",f"| 集成Ton前腕部误差 | {baseline['phases']['pre_ton']['wrist_error']:.6f} | {candidate['phases']['pre_ton']['wrist_error']:.6f} |"]
for label in ['anchor_error','relative_error','bias_norm','wrist_error']:
    lines.append(f"| 单专家原始 {label}（166条） | {components['a3_'+label]:.6f} | {components['new_'+label]:.6f} |")
lines+=['',f"相对A3分数变化{summary['delta_vs_a3']:+.8f}；{summary['sessions_improved_vs_a3']}/20个Session改善。Session重采样描述区间为{ci.tolist()}，未修正开发集反复选择偏差，不能代替独立验证。", '', '初始等价、padding不变、位移锚点约束、梯度测试见check_model.py。无标签双模型真实推理复现见experiments/anchor/stitched_runtime_audit.json；独立专家GPU/CPU差异见checkpoint_audit.json。结果仅覆盖本轮结构、训练预算及固定权重，不能推出所有位置分解模型均有效或无效。']
(ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['baseline','candidate']},indent=2))
