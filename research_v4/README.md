# 独立深度学习实验 research_v4

此目录由当前聊天创建，不修改 baseline_v1、baseline_v2、baseline_v3。现有 GPU 被另一任务使用，本轮实验使用 CPU。

## 已实现

- 多尺度残差时序卷积：2 帧与 8 帧平滑的神经特征，3 倍降采样，六层膨胀卷积，最终插值回原始帧。
- 跨 Session 共享模型，Session / condition 嵌入与小幅输出适配。
- 整段通道 dropout，以及平滑神经特征重建辅助任务。
- 可选 32 维非线性姿态表示：训练时由真实姿态编码提供辅助监督，推理只接收神经信号和公开元数据。
- 可选评分函数近似目标微调，保留距离损失防止难试次梯度消失。

这是独立的轻量多任务模型，不是 LFADS、NDT2 或 DPAD 的完整复现。使用双向时间上下文及奖励时间，因此是离线解码。

## 验证边界

沿用原有 split_seed=42：401 个训练 Trial、169 个验证 Trial，20 个 Session。统计量仅由训练 Trial 计算，标准化不使用验证集。训练/验证 Trial 无交集，划分已与 v3 核对一致。

checkpoint、训练轮数及集成权重都在开发划分上选择，结果带有选择偏差，不能等同于平台分数或独立确认结果。当前目录未进行全量重训，也未生成正式 submission.zip。

`reference_ensemble.json` 固定了开始比较时 v3 的结构集成配置。`compare.py` 使用相同骨长后处理，混合权重只检查 0、0.1、0.25、0.5、1。现有基线预测只读引用。

`experiments/denoising_tcn` 是执行会话中断后保留的早期记录，不能作为完整训练结果；完整重跑在 `experiments/denoising_complete`。

## 运行命令

在本目录使用 `C:/Users/Frank/anaconda3/envs/bci_stable/python.exe` 执行：

```powershell
python train_latent.py --epochs 100 --output experiments/denoising_complete
python train_latent.py --epochs 100 --manifold --threads 2 --output experiments/manifold
python train_latent.py --epochs 50 --score --lr .0002 --init-checkpoint experiments/denoising_complete/checkpoint.pt --threads 2 --output experiments/score_finetune
python compare.py --experiment experiments/denoising_complete
python selftest.py
```

这些命令会写入指定实验目录；复跑时应换一个 output 名称保留原结果。

`predict.py` 可以加载研究 checkpoint 对 heldout 目录推理，无需 keypoint 字段。它不是已经打包的赛事提交入口。

## 方法依据

- LFADS：利用序列生成模型提取神经群体潜在动力学及去噪放电率。https://pmc.ncbi.nlm.nih.gov/articles/PMC6380887/
- NDT2：跨上下文预训练、上下文嵌入与掩码重建。这里阅读的是 2023 年预印本，不把其结果直接外推到本赛题。https://pmc.ncbi.nlm.nih.gov/articles/PMC10541112/
- 此处借鉴去噪与共享表示的思想，使用连续值神经特征的 MSE 重建；没有将 SBP 或原始 Hz 数值误当作 Poisson 整数脉冲计数。

当前开发验证增益和各实验最终结果见 `results.json`（完成比较后生成）。58 分是目标，不是已达到的成绩。

## 本轮开发结果

| 模型 | 单模型原始分 | 与固定 v3 集成混合的最佳分 |
|---|---:|---:|
| 原固定 v3 集成 | - | 51.7060 |
| 去噪时序卷积 | 49.1990 | 52.0012 |
| 非线性姿态低维约束 | 49.3631 | 52.0056 |
| 评分目标微调 | 49.2334 | 51.9915 |
| 去噪卷积与姿态模型各半 | - | **52.0699** |

最后一行的总权重为：原 v3 固定集成 75%，去噪时序卷积 12.5%，姿态约束模型 12.5%。后处理 sigma=1、骨长强度=0.5。最好结果仍距目标 58 分约 5.93 分；没有证据支持通过本轮方法达到平台 58 分。

去噪模型与姿态模型分别完成了 20 个 Session、每个 Session 一个删除标签后的验证 Trial 的推理自测，输出形状、类型、编号和有限性通过。重新加载 checkpoint 后，与保存预测最大差异分别为 0 和 0.00006104。变长序列补齐不变性检查误差约 0.00000025。

第二划分仅复核固定去噪卷积模型：80 轮、100 轮余弦学习率安排、split43，训练期间不选验证最优轮。由于当前比较所用 v3 完整集成没有对应复核预测，第二划分的参照为已有 v2 三模型集成；混合权重 0.5 在 split42 上先确定。这个复核不能证明完整 52.07 方案的增益。

## 后续研究判断

本轮结果支持“不同时间建模的模型具有少量互补性”，不支持“增加网络深度即可获得 58 分”。评分目标微调没有超过基础版本；姿态低维约束的单模型改善较小。

我更倾向下一步研究行为监督优先的潜在动力学，而不只优化神经信号重建：DPAD 将与行为相关的动力学和其他神经变化分开学习，这是一个针对解码目标的合理方向，但本目录尚未复现，效果未经本赛题验证。论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC11452342/

NDT2 的掩码预训练可作为另一条路线，但其跨来源大量预训练的结果不能直接当成只有本数据 570 个 Trial 时的收益保证。LFADS 则需要按神经观测类型设计似然和严格的训练集内预训练；SBP 不能直接作为 Poisson 脉冲计数。

## 固定配置复核结果

split43 的去噪卷积原始分为 49.9523。按事先固定的 50% 权重与 v2 三模型集成混合、使用相同后处理后，分数由 49.1406 提高到 51.4723。对应开发划分的同一组合为 48.8263 → 51.0075。这支持新卷积模型对 v2 的互补性，但不等于 v3 + 两个新模型的完整组合已经复核，也不保证正式平台得分。

本轮完成三组开发实验和一次固定配置复核，所有训练进程正常结束。未达到 58 分目标，未上传赛事平台。
