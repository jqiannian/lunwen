# 训练指南（Training Guide）

本指南整合了历史训练报告（`MVP_TRAINING_REPORT.md`、`TRAINING_REPORT_V2.md`），把“怎么跑、怎么看、怎么排障”收敛成一份权威入口。

## 1. 你会得到什么

- **训练产物**：`artifacts/checkpoints/*.pth`
- **训练曲线**：`reports/training_curves.png`
- **指标输出**：终端表格（Loss/AUC/F1/RuleConsistency/GradNorm 等）

## 2. 训练前准备

### 2.1 数据准备（已存在合成数据）
当前仓库已有：`data/synthetic/train`（80）与 `data/synthetic/val`（20）。

如需重新生成：

```bash
python3 scripts/prepare_synthetic_data.py --num-scenes 100 --output-dir data/synthetic
```

### 2.2 环境准备
建议使用 conda（后续以 `docs/guides/INSTALLATION_CONDA.md` 为准）。如果你先用现有 Python 环境，至少要确保：

- `torch`
- `numpy`
- `pydantic`
- `opencv-python`（用于 explain，可选）
- `pandas`（用于伪标签，可选）
- `structlog`（用于监控日志，可选）

## 3. 训练命令

### 3.1 最小回归（Smoke Test）
用于验证“数据→图→模型→loss→反传→保存”的链路能跑通：

```bash
python3 tools/train_red_light.py train --epochs 2 --max-samples 5 --device cpu
```

预期：
- 生成 `artifacts/checkpoints/checkpoint_epoch_*.pth` 与 `best.pth`
- 生成 `reports/training_curves.png`

### 3.2 标准训练（推荐）

```bash
python3 tools/train_red_light.py train --epochs 50 --device cpu
```

说明：
- CPU 训练会较慢，建议从 50 开始。
- 若你有 CUDA 环境，可把 `--device cuda`。

## 4. 如何解读训练输出

### 4.1 核心指标
- **Val Loss**：越低越好，是当前 Trainer 的主选择指标（用于 best checkpoint）。
- **AUC**：越接近 1 越好，衡量“排序能力”。
- **F1 / Precision / Recall**：依赖阈值（项目默认可能用 0.7），容易出现“F1=0”的假象。
- **Rule Consistency**：模型输出与规则输出的接近程度。
- **Grad Norm / 梯度异常**：训练稳定性的健康指标。

### 4.2 已观测到的历史表现（供对照）
- **10 epochs（CPU）**：Val Loss 可从 3.67 降到 ~0.95（历史报告）
- **79 epochs 测试**：AUC ~0.8485，Rule Consistency ~0.7628，F1 ~0.3333（阈值=0.7 时）

## 5. 常见问题与排障

### 5.1 AUC 长期在 0.5
- 常见原因：训练不足、数据/标签分布单一、模型输出塌缩。
- 处理建议：
  - 先跑 50-80 epochs 再下结论
  - 检查 `data/synthetic/` 是否真实包含 violation 场景

### 5.2 F1/Precision/Recall 变成 0
- 常见原因：**阈值过高**（例如阈值 0.7，但模型输出集中在 0.36~0.39）。
- 处理建议：
  - 测试/评估时把阈值临时下调到 0.5
  - 或基于 ROC/PR 曲线选最佳阈值（后续优化项）

### 5.3 频繁出现“梯度不平衡/梯度爆炸”警告
- 已在历史训练中观察到：约 40% batch 会出现 max/min>100 的不平衡。
- 处理建议（优化项）：
  - 使用“层级学习率”降低 `rule_focus` 与 `score_head` 的学习率
  - 将梯度裁剪从 1.0 收紧到 0.5

## 6. 输出目录约定
- `artifacts/checkpoints/`：模型权重
- `artifacts/pseudo_labels/`：伪标签
- `reports/`：训练曲线、测试报告、可解释性图片

> 禁止在根目录新增训练报告 Markdown。若需要报告，请写入 `docs/guides/` 或输出到 `reports/`。
