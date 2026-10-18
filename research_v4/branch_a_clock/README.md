# Branch A 补充对照：训练集时间估计四阶段门控

这是第二轮go cue辅助监督实验后的定向对照，不提交平台。

阶段诊断发现：仅根据同Session、同condition训练Trial中go cue相对Ton的时长中位数，就能在开发集达到约92.2%的阶段平衡准确率。因此测试固定时间估计门控是否能改善坐标，而不引入分类辅助任务的梯度。

从第一轮A3 checkpoint初始化，复制原核心阶段专家为准备/执行两个专家，初始输出保持不变。每个Session×condition的go cue时间偏移仅在当前训练划分拟合，作为矩阵存入checkpoint config；没有同条件样本时退回该Session训练样本中位数。推理只按已有sid/cid查表，并将估计分界限制在当前公开Ton/reward之间；以0.1秒尺度sigmoid平滑切换，不读取真实测试go cue。共享主干不增加输入维度，保留原有元数据。

固定微调60轮、lr1e-4，与第二轮继续训练/无监督四阶段/辅助监督四阶段等预算。epoch0参与checkpoint选择，所有选择均为开发集结果。对照结果由evaluate_branch.py比较添加和替换两种方式，后处理只应用一次。

在本目录运行 `C:/Users/Frank/anaconda3/envs/bci_stable/python.exe run_clock.py`。实验结果见experiments/clock4。

完成结果见 ../branch_a_round2/README.md 和 results.json。时间门控未超过第一轮A3，未提交平台。
