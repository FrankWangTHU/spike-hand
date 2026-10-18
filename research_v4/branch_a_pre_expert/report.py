import hashlib
import json
from pathlib import Path
import numpy as np
from runtime.data import read_trial
from train import blend_weight
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
baseline=next(r for r in json.loads((ROOT.parent/'branch_a/experiments/a3_all/replacement.json').read_text()) if r['alpha']==1)
results=json.loads((ROOT/'results.json').read_text())
rows=[]
for name,path in [('all_coordinates','pre_ton'),('inference_wrist_only','wrist_only'),('trained_wrist_only','pre_ton_wrist')]:
    folder=ROOT/'experiments'/path
    full=next(r for r in json.loads((folder/'replacement.json').read_text()) if r['alpha']==1)
    differences=np.array([full['sessions'][s]['hrfs_local']-baseline['sessions'][s]['hrfs_local'] for s in baseline['sessions']])
    rng=np.random.default_rng(42)
    ci=np.quantile(rng.choice(differences,(10000,len(differences))).mean(1),[.025,.975])
    rows.append(dict(name=name,score=full['mean']['hrfs_local'],delta_vs_a3=float(differences.mean()),sessions_improved=int((differences>0).sum()),descriptive_session_bootstrap_95pct=ci.tolist(),phases=full['phases']))
folder=ROOT/'experiments/pre_ton_wrist'
cfg=json.loads((folder/'config.json').read_text())
pred=np.load(folder/'validation_predictions.npz')
expert=np.load(folder/'expert_predictions.npz')
a3=np.load(ROOT.parent/'branch_a/experiments/a3_all/validation_predictions.npz')
max_post=0.; max_pose=0.
for session,split in cfg['split'].items():
    for file in split['validation']:
        key=session+'_'+Path(file).stem
        item=read_trial(REPO/'player_data/Data_heldin'/session/'heldin'/file,False)
        assert 'y' not in item
        w=blend_weight(item['times'],item['onset'])[:,None]
        mix=(w*expert[key]+(1-w)*a3[key]).reshape(-1,21,3)
        old=a3[key].reshape(-1,21,3)
        expected=old-old[:,:1]+mix[:,:1]
        np.testing.assert_array_equal(pred[key].reshape(-1,21,3),expected)
        t=(item['times']-item['times'][item['onset']])/1e9
        max_post=max(max_post,float(abs(expected[t>=.2]-old[t>=.2]).max()))
        max_pose=max(max_pose,float(abs((expected-expected[:,:1])-(old-old[:,:1])).max()))
results['targeted_followup']=rows
results['wrist_only_checks']=dict(trials=len(pred.files),max_post_difference=max_post,max_relative_pose_difference=max_pose,labels_read=False)
results['wrist_best_epoch']=cfg['best_epoch']
(ROOT/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
hashes=json.loads((ROOT/'provenance.json').read_text())
for name in ['train_wrist.py','run_wrist.py','report.py']:
    hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
hashes['v3_zip']=hashlib.sha256((REPO/'baseline_v3/submission.zip').read_bytes()).hexdigest()
assert hashes['v3_zip']=='1c4e29a7a62b311e0b8444af5f09bb947b16a86278d39e73390c198eabbd8372'
(ROOT/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
lines=['# 独立 Ton 前专家实验','', '仅本地实验，提交次数0。划分42：401训练Trial、169验证Trial、20个Session；该划分已用于多轮开发，结果不是独立确认。', '', '完整复制A3网络并允许全部参数更新，冻结另一份A3负责后半段；完整神经输入上下文保留。初始化来自第一轮A3，AdamW学习率1e-4，余弦60轮，batch8、hidden192、通道dropout0.25。不是只训练一个共享主干的小头。', '', '训练损失使用Ton相对时间的smoothstep权重：Ton前0.2秒及更早权重1，Ton后0.2秒及更晚权重0，中间平滑递减。位置、速度和骨长损失均应用该区间权重。初版预测全部坐标；针对性对照只优化腕部位置及腕部差分，并保持A3的腕部相对手指姿态。', '', '初版最佳轮次为'+str(results['best_epoch'])+'；腕部专用版最佳轮次为'+str(cfg['best_epoch'])+'。选择依据为固定0.2秒半宽拼接后的完整轨迹原始分，包含epoch0。最终集成替换原stacked_large成员，其他成员、权重和后处理不变。', '', '| 方案 | 集成本地分 | 相对A3 | 改善Session | Ton前腕部误差 | Ton前姿态误差 |','|---|---:|---:|---:|---:|---:|',f"| A3 | {baseline['mean']['hrfs_local']:.8f} | 0 | — | {baseline['phases']['pre_ton']['wrist_error']:.6f} | {baseline['phases']['pre_ton']['pose_error']:.6f} |"]
for row in rows:
    phase=row['phases']['pre_ton']
    lines.append(f"| {row['name']} | {row['score']:.8f} | {row['delta_vs_a3']:+.8f} | {row['sessions_improved']}/20 | {phase['wrist_error']:.6f} | {phase['pose_error']:.6f} |")
lines+=['', '初版拼接半宽敏感性检查：0.1/0.2/0.4秒分别为52.24169823/52.24171198/52.24115074，没有显示明显拼接宽度收益。腕部专用推理及训练是看到初版姿态退化后的探索，不能视为预先指定的独立实验。', '', '初版原始轨迹交接区腕部位置误差21.90658→21.95573，略变差；腕部速度误差152.26191→151.62064，略改善。平滑交接未造成这个平均速度误差上升，但不代表每条轨迹都更好。以上交接指标与集成后指标属于不同评估对象。', '', '结果JSON包含每Session分数和Session重采样区间。该区间仅描述当前开发集波动，没有修正反复选模型带来的偏差。尚未进行独立划分确认或平台提交。', '', 'GPU推理复现、CPU差异和不读取标签的审计见各实验checkpoint_audit.json。拼接重建检查覆盖169条验证Trial；腕部专用方案后段和相对姿态仅有浮点重构误差，详见results.json。']
(ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(rows,indent=2))
