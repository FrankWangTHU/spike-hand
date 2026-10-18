# Branch B：结构化姿态输出

本聊天推进 LMLF 启发的输出结构适配。代码从 v3 单独复制，现有 baseline 及提交包不改动。

共享编码器沿用 stacked_large：两层分层双向 GRU，hidden=192，多尺度完整输入，通道 dropout=0.25。输入、损失、划分、学习率及 220 轮搜索保持原设置。开发 split_seed=42，训练 seed=42；不是独立确认结果。

| 配置 | 手腕 | 手指 |
|---|---|---|
| 原 stacked_large 对照 | 直接坐标 | 60 维直接坐标 |
| B1 | 直接 3 坐标 | 16 维因子 + 固定逆 PCA + 有界残差 |
| B2 | 直接 3 坐标 | 32 维因子 + 固定逆 PCA + 有界残差 |
| B3（待选择） | 同胜出配置 | 后续增加显式延迟上下文 |

PCA 空间是按 session 的训练均值/标准差标准化后的 60 维手腕相对坐标，跨 session 合并训练帧拟合；不含手腕、不含验证帧。逆变换参数作为 checkpoint buffer 固定保存。恢复后沿用 v3 的反标准化和手腕位置相加。不是直接在原始物理坐标上拟合 PCA。

残差为 `0.2 * tanh(raw_residual)`，逐坐标不超过对应训练标准差的 0.2 倍；包含原输出 session 适配项，不能通过适配绕过上限。因子另有 session 适配。第一轮不另加因子监督或更换损失。

## 执行与结果

在此目录使用 `C:/Users/Frank/anaconda3/envs/bci_stable/python.exe`：

```powershell
python -m unittest test_pose -v
python run_experiments.py
```

真实数据 1 轮 smoke 已通过，不能据此判断增益。正式 B1/B2 顺序运行，避免同时占用 GPU。各自的 `experiments/b1/train.log`、`experiments/b2/train.log` 记录训练过程；每完成一组，`results.json` 更新总分、Wrist、Pose、Bone、Motion 及相对原模型的差值。原对照直接读取 v3 已完成的同配置实验；运行器检查划分与训练参数一致。`source_manifest.json` 保存启动时源码哈希。

保留全部 session/trial 评分及验证预测用于诊断。模型按开发 HRFS 选 checkpoint，必须另外检查 Motion 是否下降；分数更平滑不等于动作更准确。后续优先比较原始输出，再使用统一后处理比较集成收益。B3 需要先查看 B1/B2 结果，不能提前宣称胜出或收益。尚未做 split43 固定配置复核或全量重训，不生成新提交包。
