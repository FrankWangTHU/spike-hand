import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parent.parent
result = json.loads((root/'results.json').read_text())
assert result['round_complete'] and not result['development_gate_passed']
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
provenance = json.loads((root/'provenance.json').read_text())
for name in ['train.py', 'runtime/model.py', 'runtime/data.py', 'run_round2.py']:
    assert digest(root/name) == provenance['source_files'][name], name
assert digest(root.parent/'branch_a/experiments/a3_all/checkpoint.pt') == provenance['initial_checkpoint_sha256']
assert digest(repo/'baseline_v3/submission.zip') == '1c4e29a7a62b311e0b8444af5f09bb947b16a86278d39e73390c198eabbd8372'
provenance['final_analysis_files'] = {n: digest(root/n) for n in ['evaluate_phase.py', 'summarize.py', 'audit_checkpoint.py', 'train_with_last.py']}
provenance['frozen_v3_and_initial_A3_verified_unchanged'] = True
(root/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
clock = root.parent/'branch_a_clock'
(clock/'provenance.json').write_text(json.dumps({'source_files': {n:digest(clock/n) for n in ['train.py','runtime/model.py','run_clock.py','evaluate_branch.py']}, 'initial_checkpoint_sha256': provenance['initial_checkpoint_sha256'], 'platform_submissions_used':0}, indent=2), encoding='utf-8')
lines = ['\n## 完成结果\n', '以下均为开发划分42，本地分数不能直接换算为平台54.38719。所有方案从同一A3权重初始化，各微调60轮。\n', '| 方案 | 最佳轮数 | 最佳融合分 | 相对冻结v3 | 改善Session |', '|---|---:|---:|---:|---:|']
for row in result['candidates']:
    lines.append(f"| {row['name']} | {row['epoch']} | {row['best_blend']:.8f} | {row['delta_vs_v3']:+.8f} | {row['sessions_improved']}/20 |")
lines += ['', '冻结v3为52.06467484，第一轮A3为52.23365739。本轮没有进一步改善。监督方案选择epoch0，表示训练后的坐标分数未超过初始化，不能把其epoch0阶段准确率当作训练失败的证据。', '', '为诊断辅助任务，原监督实验被完全重放，只额外保存第60轮权重；各已记录训练和验证指标差异为0。第60轮阶段平衡准确率0.96989060，go边界平均绝对误差0.075539秒；训练集条件时长中位数基线分别为0.92222392和0.134963秒。但该轮坐标原始分50.73319186，低于初始化50.79391849。阶段学得准，不等于坐标重建更准。', '', 'clock4使用训练Trial拟合的Session×condition go偏移中位数，不需要测试go cue，也不添加分类损失。该对照仍没有超过第一轮A3，因此当前结果不支持把问题主要归因于辅助分类损失。它不排除更大训练预算、不同路由或其他分段结构可能有效。', '', '推理检查覆盖go4_unsupervised和clock4各20个Session的一个Trial：推理不读取坐标标签，形状与有限值检查通过，GPU预测与保存结果最大差为0。CPU与GPU最大差分别0.032700和0.025116，仍有4和2个Session未通过严格CPU容差；尚不能宣称跨设备严格一致。原v3包和A3初始化权重哈希保持不变。', '', '本轮未达到相对v3约+0.4的开发筛选门槛，未做独立确认、全量重训或平台提交；本聊天消耗提交次数0。第一轮A3仍只是开发候选。建议下一项先诊断早期绝对位置偏差和运动幅度收缩，而非继续盲搜阶段数。该建议尚未实验验证。', '']
with (root/'README.md').open('a', encoding='utf-8') as f:
    f.write('\n'.join(lines))
with (clock/'README.md').open('a', encoding='utf-8') as f:
    f.write('\n完成结果见 ../branch_a_round2/README.md 和 results.json。时间门控未超过第一轮A3，未提交平台。\n')
print('Verified frozen inputs and wrote final reports.')
for row in result['candidates']:
    print(row['name'], row['best_blend'])
