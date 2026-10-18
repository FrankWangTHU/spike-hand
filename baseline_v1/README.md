# 精细手部运动解码初版 v1

该版本直接适配本地 `../player_data/Data_heldin`，提供真实数据训练、独立 Trial 验证、全量重训和提交打包。原始数据和官方示例保持原样，所有新结果写在本目录。

## 方法

采用轻量因果 GRU，借鉴所附 He 等论文中用 PCA 约束运动学输出的思路。这是针对竞赛 256 通道数据实现的初版，不是论文三分支 LMLF 的完整复现。

1. 拼接 TX 和 SBP，512 个特征分别做 `log1p`；使用训练 Trial 的统计量标准化并裁剪到 ±10。
2. 每帧拼接标准化特征与单向 EMA（当前帧权重 0.35），得到 1024 维输入；每个 Trial 重置 EMA。
3. Linear → LayerNorm → GELU → Dropout → 两层单向 GRU，隐藏维度 96。仅使用当前和历史神经输入，不使用动作标签或奖励事件作为模型输入。
4. 目标拆为腕部全局坐标和另外 20 点的腕部相对坐标，各维标准化后拟合 24 维 PCA。网络输出 PCA 系数及权重 0.25 的坐标残差，最后恢复为原始顺序和尺度的 63 维坐标。
5. 使用标准化腕部/局部坐标损失、PCA 系数损失和一阶差分损失。速度训练项采用逐帧差分；评分使用时间戳换算的速度，二者定义有区别。

不使用对称平滑、不读取未来神经帧，不跨 Trial 传播隐藏状态。整 Trial 小批量训练，右侧补齐区域由掩码从损失中排除；推理逐 Trial 执行。

## 数据与验证

本地数据为 20 个 Session、570 个 Trial、144,628 帧。每个 Session 按原始 `condition` 分层划分，目标验证比例 20%，种子 42。每个至少含两个 Trial 的类别至少留出一个验证 Trial，同时保留一个训练 Trial；单例只进入训练。由于许多类别很小，本次实际划分为 401 个训练 Trial、169 个验证 Trial（29.6%）。先划分再拟合输入标准化、目标标准化及 PCA。

每个 Session 最多训练 60 轮，按本地复现 HRFS 选择最佳 checkpoint，12 轮不提升早停。然后以所选轮数从随机初始化开始，使用该 Session 的全部 heldin 数据重新拟合预处理并训练正式模型。

- `validation_models/`：只用训练划分拟合的验证 checkpoint。
- `model/`：全量重训模型，用于提交。
- `reports/<session>.json`：具体训练/验证 Trial、分动作 Trial 分数、每轮曲线、最佳轮数及耗时。
- `reports/summary.json`、`reports/validation.csv`：20 个 Session 的结果与等权平均。
- `reports/validation.png`：本地分数与训练均值姿态基准的比较、训练曲线。

本次已运行结果（20 个 Session 等权平均）：

| 指标 | 数值 |
|---|---:|
| 本地 HRFS 复现 | 33.2119 |
| 同划分训练均值姿态 HRFS | 17.7630 |
| MPJPE（原始坐标单位） | 33.9163 |
| RMSE（原始坐标单位） | 23.7475 |
| Wrist / Pose / Motion / Bone | 30.0559 / 37.0323 / 14.7604 / 64.6331 |

已生成覆盖 20 个 Session 的 `submission.zip`（16,469,225 字节）。解压后以 `--device auto` 对去标签的 40 个真实 Trial、10,168 帧完成自测，耗时约 5.68 秒，HDF5 字段、类型、帧数、顺序、关键点名称和有限值均通过。此耗时只对应自测规模。

相同规模的显式 CPU 推理自测也已通过，耗时约 5.45 秒；测试记录分别位于 `selftest/report.json` 和 `selftest_cpu/report.json`。5 项单元测试通过，覆盖划分、坐标恢复、评分性质和输入到模型的完整因果路径。

验证分数也用于早停，因此不是完全独立的最终测试估计。初版只运行一个划分和训练种子；后续比较模型时应固定划分，并用多个种子或额外保留集确认收益。随包官方模型已在全体 heldin 上训练，不能直接拿来做此留出验证的公平对照。

## 本地评分的边界

`metrics.py` 按赛题、数据包说明和论文实现**有限值数据上的本地 HRFS 复现**：腕部、局部姿态、局部速度、20 条骨骼链，权重 0.25/0.35/0.20/0.10 除以有效权重之和。先 Trial 等权平均，再 Session 等权平均。速度采用 Trial 时间戳差的正值中位数，先 int64 做差再转换为秒。

骨架依据名称构造 `Wrist → Finger_1 → Finger_2 → Finger_3 → Finger_4`，支持名称顺序检查。没有坐标单位标定参数，因此所有位置误差保留原始单位，不将数值称为毫米。论文原始实验输入通道数、数据划分与本赛题不同，论文分数不能直接作为本项目的对照。

私有官方评测代码未提供，本地分数不保证逐位一致。此实现明确拒绝非有限标签/预测，不实现论文对缺失点的 coverage 分支。当前训练数据通过严格有限值检查。常量序列相关系数使用显式容差，可能与私有实现存在边界差异。

## 运行

以下命令在 `baseline_v1` 目录执行。需要 NumPy、PyTorch、h5py；绘图额外需要 Matplotlib（不参与提交推理）。

```powershell
python train.py --epochs 60 --device auto
python make_report.py
python -m unittest discover -s tests -v
python submission_tools.py --selftest-dir selftest --device auto
```

`--selftest-dir` 必须是尚不存在的目录，重复测试时换一个名称，如 `selftest_cpu`。该自测从每个 Session 取首尾两个公开 Trial，去掉标签，构造成 heldout 格式，然后解压提交包在独立目录运行正式推理入口。它只检查接口，不用于泛化评分。

只训练一个 Session 或快速试跑：

```powershell
python train.py --sessions 2026051402 --epochs 3 --output-dir ../tmp/pilot
```

不要将局部试跑输出与完整模型混用；默认训练会覆盖相同输出目录中的 checkpoint 和报告。自定义输出目录时，打包用 `--model-dir <输出目录>/model` 指定模型。

平台接口（正式 heldout 数据由平台提供）：

```powershell
python infer.py --data_dir <测试根目录> --output_dir <预测目录> --model_dir model --device auto
```

支持 `auto/cpu/cuda`，明确要求 CUDA 但不可用时会报错。每个 Session 输出一个 HDF5，包含 float32 `predicted_trajectory`、int64 `trial_lengths` 和 `trial_indices`、UTF-8 `keypoint_names`。Trial 数字编号必须从 1 连续，关键点名称必须与模型完全一致。不读取测试 `keypoint`，不联网，不在输出目录之外写入文件，并禁用 Python 字节码缓存。

`submission_tools.py` 只打包 `infer.py`、`requirements.txt`、`model/`、`runtime/`，不会打包原始数据、验证 checkpoint、日志或自测结果。脚本核对全部已知 Session 模型和关键点顺序，并检查 ZIP 小于 2 GB。不会自动上传比赛平台。

## 环境与已知限制

本机验证环境：Python 3.13.9、NumPy 2.3.5、PyTorch 2.11.0+cu128、h5py 3.15.1，GPU 为 RTX 5070 Laptop。平台声明环境为 Python 3.11.15、NumPy 2.4.4、PyTorch 2.11.0+cu130、h5py 3.16.0。代码使用兼容 Python 3.11 的语法和常规接口，但尚未在平台的完全相同环境验证。

`requirements.txt` 无额外推理依赖，不覆盖平台固定科学计算包。没有正式 heldout 数据，因此无法报告平台分数或实测完整正式测试集耗时。全量重训后的模型并不保证优于验证 checkpoint；其依据是常规的验证选轮数、全量重新拟合流程。

后续优先事项：固定划分比较直接 GRU 与 PCA 版本、增加多个种子验证、针对运动分项调整时间损失，再评估论文多分支融合。不要先按单次留出结果反复调参。

## 本地参考资料

- `../精细手部运动解码-赛题详情.pdf`：预测任务、四项评分、提交接口。
- `../player_data/COMPETITION_SUBMISSION_REQUIREMENTS.md`：实际数据格式、关键点命名、单位说明和评分汇总规则。
- `../He - Simultaneous multi-area recordings in macaque frontoparietal cortex during a manual object-interacti.pdf`，第 20–25 页：PCA 输出表示、因果时间建模及 HRFS 定义。
