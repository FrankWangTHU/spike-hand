import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
baseline=next(r for r in json.loads((ROOT.parent/'branch_a/experiments/a3_all/replacement.json').read_text()) if r['alpha']==1)
selection=json.loads((ROOT/'selection.json').read_text())
rows=[]
for choice in selection['choices']:
    folder=ROOT/'experiments'/('calibrated_'+choice['kind'])
    score=next(r for r in json.loads((folder/'replacement.json').read_text()) if r['alpha']==1)
    delta=np.array([score['sessions'][s]['hrfs_local']-baseline['sessions'][s]['hrfs_local'] for s in baseline['sessions']])
    ci=np.quantile(np.random.default_rng(42).choice(delta,(10000,20)).mean(1),[.025,.975])
    rows.append(dict(**choice,mean=score['mean'],phases=score['phases'],delta_vs_a3=float(delta.mean()),sessions_improved_vs_a3=int((delta>1e-6).sum()),descriptive_session_bootstrap95=ci.tolist()))
result=dict(status='completed_no_improved_candidate',submissions_used=0,split=42,selection=selection,baseline=baseline['mean'],baseline_phases=baseline['phases'],candidates=rows,development_gate_passed=[r['kind'] for r in rows if r['mean']['hrfs_local']-52.06467483592634>=.4 and r['sessions_improved_vs_a3']>=12])
(ROOT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes={n:digest(ROOT/n) for n in ['train.py','runtime/model.py','run_oof.py','calibrator.py','fit_calibration.py','folds.json','selection.json']}
hashes['A3']=digest(ROOT.parent/'branch_a/experiments/a3_all/checkpoint.pt')
hashes['v3_zip']=digest(REPO/'baseline_v3/submission.zip')
assert hashes['A3']=='24f7758822264ede9fef9ab7f770b49d0dfe736046abc0268104fcef27a48844'
assert hashes['v3_zip']=='1c4e29a7a62b311e0b8444af5f09bb947b16a86278d39e73390c198eabbd8372'
(ROOT/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
lines=['# Ton前交叉预测偏移校正','', '本轮未提交平台，未修改v3或A3权重。外层沿用开发划分42（401训练Trial、169验证Trial）。该验证集此前已用于开发，结果不等于全新独立确认。', '', '## 数据隔离与训练','', '外层训练集按Session/condition分三折。每折A3结构从随机初始化训练150轮，余弦计划220轮、lr0.0007，参数配置与原A3一致，固定最终epoch150，不按内折或外层验证分数选权重。内折训练/预测规模为270/131、266/135、266/135。每条外层训练Trial恰好获得一次未参与该模型拟合的预测；外层验证Trial不参与内折模型训练或归一化统计。', '', '校正目标为Ton前真实腕部减去模型腕部的平均值，每条Trial仅3个数。Ton前为空的Trial不参与校正器拟合。校正器选择只使用外层训练集；外层验证坐标不参与特征生成、偏移预测或超参数选择。', '', '三组方法：group为全局/Session/Session×condition收缩均值（伪计数5）；metadata为类别、A3预测轨迹均值/起点/终点/波动/位移/时长特征的岭回归；neural额外使用Ton前log1p TX/SBP的均值、前后半均值和标准差，经训练内标准化和16维PCA降维。岭惩罚从10/100/1000选择，校正强度从0/0.25/0.5/1选择，按内层保留Trial的Session等权Ton前腕部误差选择。预测偏移限制在训练目标范数95百分位内。', '', '校正器CV再次按三折划分，但不会重新嵌套训练上游神经模型：内层选择误差不是严格嵌套CV的无偏估计。最终外层标签始终没有进入这些拟合或选择。内折神经模型只使用约2/3外层训练数据，与最终A3存在训练规模差异，这是校正迁移的限制。', '', 'Ton前全部关键点加相同偏移，保留相对手指姿态；Ton附近±0.2秒smoothstep归零，后段不变。修正后的A3替换原stacked_large成员，集成权重和后处理保持不变。', '', '## 结果','', '| 方法 | 岭惩罚 | 校正强度 | 内层腕部误差 | 外层集成分 | 相对A3 | 改善Session |','|---|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['kind']} | {r['alpha']} | {r['strength']} | {r['inner_error']:.5f} | {r['mean']['hrfs_local']:.8f} | {r['delta_vs_a3']:+.8f} | {r['sessions_improved_vs_a3']}/20 |")
lines+=['', 'A3基线52.23365739；v3本地基线52.06467484。不能把本地分差直接加到平台54.38719。完整分阶段指标和Session重采样描述区间保存在results.json；区间不修正历史开发选择偏差。', '', 'check_calibrator.py检查无标签推理、零校正恒等、后段不变和相对姿态保持；实际A3＋校正器推理复现见runtime_audit.json。所有训练命令、固定分折、预测和校正器保存在本目录。']
lines+=['', '## 本轮结论', '', '没有候选超过A3。group和neural在外层训练内选择强度0，等同保留基线；metadata选择alpha1000、强度0.25，外层总分下降0.00276233。其Ton前平均腕部距离误差26.482767→26.468326，但完整腕部评分50.098587→50.088643，说明平均早期距离与最终按Trial评分汇总的目标不完全一致。姿态、骨长与运动分数只有浮点级差异，本次下降不能归因于手指姿态被破坏。', '', '结论限于本轮分组和线性特征：位置偏移占主要误差，并不保证该偏移可由当前观测特征稳定预测。没有据此扩大非线性网络，没有按外层结果重新调校正强度，也没有进入确认或提交。保留原A3。']
(ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
