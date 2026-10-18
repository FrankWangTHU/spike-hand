# 精细手部运动解码 v2

本版将首版的独立 Session GRU 改为跨 Session 共享的神经解码器，并集成三个模型。已在原始划分上达到 **48.8263 本地 HRFS**，首版为 **33.2119**；固定方案在第二划分复核得到 **49.1406**。原始划分 MPJPE 从 **33.9163** 降至 **21.4159**，第二划分为 **20.7354**（保留原始坐标单位）。平台正式测试集不在本地，不能把这些数值当作排行榜分数。

完整结果见 `reports/summary.json`；各 Session 比较见 `reports/sessions.csv`，实验对照见 `reports/experiments.csv`，图见 `reports/comparison.png`。

## 最终方案

- 输入是 TX、SBP 共 512 个特征，使用原始数值的训练集标准化、±10 裁剪，再拼接因果 EMA，得到 1024 维神经输入。底层读入接口保存 log1p 数值，`input_transform=raw` 在训练和推理两端一致恢复原始数值后再做标准化。
- 各 Session 分别保存输入/输出均值、标准差，输出仍分为腕部全局坐标和手指相对腕部坐标。推理恢复原始坐标顺序和尺度。
- 20 个 Session 共享主要输入投影和两层单向 GRU（hidden=128），保留 Session embedding、偏置及输出适配层。两成员的独立输入适配矩阵只乘 0.1，另一个成员不使用该矩阵。
- 模型辅助输入为官方公开的 `condition`、`Ton_frame_index`、`reward_frame_index`、`frame_timestamps_ns`。使用相对时间、动作阶段和傅里叶时间特征；不读测试标签。
- 三成员等权平均：适配系数 0.1 / 训练种子 42、适配系数 0.1 / 种子 7、适配系数 0 / 种子 42。最终全量重训分别为 140、140、180 轮。
- 集成之后使用标准差 1 帧的高斯平滑，再进行 0.5 强度的骨长校正。骨长目标是对应训练数据中各骨段长度的中位数。

**这是离线解码方案。** 神经 GRU 是单向的，但公开奖励时间和对称平滑使用整 Trial 离线信息，整个流程不满足严格流式因果性。双向 GRU、TCN 也做了对照，最终提交采用上述三个单向 GRU 成员。

## 验证流程

数据仍为 20 个 Session、570 个公开 heldin Trial。固定划分 42 与 v1 完全相同：按 condition 分层，401 个训练 Trial、169 个验证 Trial。每个 Session 的标准化参数、骨长及所有模型权重仅用相应训练 Trial 拟合；没有逐帧随机拆分，没有使用正式 heldout 真值。

开发划分用于比较结构、选 checkpoint 及选择后处理，因此 48.8263 不是完全独立的测试估计。选定三成员、等权权重、sigma=1、骨长系数 0.5 后，第二个划分（split seed=43）按固定 140/140/180 轮训练，**保留最后一轮，不按该划分验证分数挑选模型**。第二划分的结果也保存在 `reports/summary.json`。这是同一数据集上的另一次划分复核，不是新采集的独立测试集，两次分数不能作为无偏交叉验证平均。

本地 HRFS 沿用首版实现和评分尺度，不通过修改公式提高分数。先 Trial 等权汇总，再 Session 等权汇总。实现处理有限值数据，未实现私有官方评测器的全部缺失值边界情况，不能保证和官方逐位一致。

## 提交包

`submission.zip` 顶层为：

```text
infer.py
requirements.txt
runtime/
model/
  manifest.json
  member_0.pt
  member_1.pt
  member_2.pt
```

本次生成的提交包大小为 **25,506,630 字节**，含三个全量重训 checkpoint，覆盖全部 20 个 Session。

共享 checkpoint 内含全部 20 个 Session 的适配参数和标准化统计量，manifest 逐 Session 记录关键点名称和骨长，因此覆盖全部已知 Session。无需为共享权重复制 20 份文件。ZIP 不包含训练数据、实验预测、日志或验证 checkpoint。

推理接口保持不变：

```powershell
python infer.py --data_dir <测试根目录> --output_dir <结果目录> --model_dir <解压目录>/model --device auto
```

支持 `auto/cpu/cuda`，每 Session 输出一个包含四个规定数据集的 HDF5。推理不联网、不读取 `keypoint`、不向输出目录之外写入文件，禁用字节码缓存。平台已提供 NumPy、SciPy、h5py、PyTorch，`requirements.txt` 没有额外依赖。

## 重现最终训练和打包

在本目录运行，下列训练会覆盖同名输出目录：

```powershell
python train_shared.py --output final/seed42 --epochs 140 --input-transform raw --shared-input --adapter-scale 0.1 --refit
python train_shared.py --output final/seed7 --epochs 140 --input-transform raw --shared-input --adapter-scale 0.1 --seed 7 --refit
python train_shared.py --output final/no_adapter --epochs 180 --input-transform raw --shared-input --adapter-scale 0 --refit
python package_submission.py --members final/seed42 final/seed7 final/no_adapter --sigma 1 --bone-strength 0.5
```

训练数据默认相对路径为 `../player_data/Data_heldin`，可通过 `--data-dir` 修改。`--refit` 用全部 heldin Trial 拟合统计量、训练并保留最后一轮，不计算验证分数。每个运行目录的 `config.json` 记录完整参数，`history.json` 记录训练进度。

复现第一划分的验证模型时去掉 `--refit`，换一个输出目录；这会按验证分数选择最佳 checkpoint。第二划分复核还需增加 `--split-seed 43 --last-epoch`。后处理与集成使用 `evaluate_predictions.py`，并明确传入 `--sigma 1 --strength 0.5` 固定参数。

## 检查

```powershell
python -m unittest discover -s tests -v
python selftest.py --work-dir selftest_auto --device auto
python selftest.py --work-dir selftest_cpu --device cpu
```

自测目录必须尚不存在。自测从每个 Session 选取两个公开训练 Trial 并删除标签，仅检查 40 个 Trial / 10,168 帧的接口、独立解压运行、模型覆盖、名称、顺序、数据类型、有限值和输出目录写入约束，不计算泛化分数。

6 项单元测试已通过。GPU（auto）和 CPU 解压自测均通过，推理分别耗时约 6.15 秒、12.66 秒；记录位于 `selftest_auto/report.json` 和 `selftest_cpu/report.json`。正式全量测试集耗时不能直接等同于这个子集耗时。

本机为 Python 3.13.9 / NumPy 2.3.5 / PyTorch 2.11.0+cu128 / h5py 3.15.1，RTX 5070 Laptop。尚未在赛事 Python 3.11.15 及完全相同版本环境中运行。完整正式测试集耗时和平台得分仍需平台确认，当前没有自动上传任何提交。

## 文件用途

- `runtime/`、`infer.py`：完整提交推理实现。
- `train_shared.py`：统一训练器，包含探索中用过的 GRU、双向 GRU、TCN、不同共享强度及损失选项。最终命令只使用上述三个配置。
- `experiments/`：开发划分与第二划分的 checkpoint、逐 Trial 预测和分数；不能作为正式全量模型提交。
- `final/`：全量重训模型。
- `experiment_templates.py`、`experiment_alignment.py`：未采用的事件模板对照。
- `make_report.py`：生成实验 CSV、结果 JSON 和对比图。
