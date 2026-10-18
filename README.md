# Spike Hand：精细手部运动解码

本仓库保存从神经信号解码手部运动的基线代码、研究分支、实验说明和结果摘要。各目录的 README 记录对应方法、运行命令与验证边界。

## 目录

| 目录 | 内容 |
| --- | --- |
| [baseline_v1](baseline_v1/README.md) | 独立 Session 的因果 GRU、PCA 输出约束、训练与推理 |
| [baseline_v2](baseline_v2/README.md) | 跨 Session 共享解码器、三模型集成及离线后处理 |
| [baseline_v3](baseline_v3/README.md) | 多结构集成、固定配置复核、打包与设备一致性检查 |
| [branch_b](branch_b/README.md) | 结构化姿态输出与 PCA 因子实验 |
| [research_v4](research_v4/README.md) | 多任务时序模型及后续 Branch A 研究 |
| [research_v4/analysis](research_v4/analysis/README.md) | 误差诊断、实验清单和分析图 |
| [research_v4/PLAN_NEXT.md](research_v4/PLAN_NEXT.md) | 后续研究计划 |

## 使用说明

1. 将赛事数据放在本地 `player_data/`，保持各版本文档要求的目录结构。
2. 按所选版本的 `requirements.txt` 安装依赖；研究分支的环境与命令见其 README。
3. 按对应文档运行训练、验证、推理或打包命令。部分研究分支依赖先前版本训练生成的 checkpoint 和预测文件，需要先生成这些产物。

仓库不包含原始赛事数据、模型权重、逐帧预测、提交 ZIP、运行日志、自测临时目录及根目录参考 PDF；这些文件保留在本地。文档内的本机绝对路径需按使用环境调整。结果摘要和分析图用于记录已有实验，不表示模型产物随仓库提供。

## 结果解释

各版本文档中的本地 HRFS、开发划分和固定配置复核结果应分别解读，不能直接当作平台成绩。`score.txt` 保留已有平台分数记录。原有实验报告与自测记录描述历史执行结果，不代表每次提交都重新训练或运行全部测试。
