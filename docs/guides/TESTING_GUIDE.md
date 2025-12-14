# 测试指南（Testing Guide）

本指南把“怎么验收、怎么跑测试、怎么解释指标”收敛成一份入口。

## 1. 测试目标

- **验收三场景**（MVP核心）：
  - `parking`：红灯停车（应判定无违规）
  - `violation`：红灯闯行（应判定有违规）
  - `green_pass`：绿灯通行（应判定无违规）
- 输出：
  - 结构化测试指标
  - 场景级报告（JSON）
  - （可选）注意力/证据链可视化

## 2. 前置条件

- 已有可用 checkpoint：`artifacts/checkpoints/best.pth`
- 合成验证集存在：`data/synthetic/val`

## 3. 推荐测试命令

### 3.1 快速验收（如果已实现 test CLI）

```bash
python3 tools/test_red_light.py --scenario all --checkpoint artifacts/checkpoints/best.pth --data-root data/synthetic
```

> 如果该 CLI 参数与实际不一致，以 `tools/test_red_light.py --help` 为准。

### 3.2 指标对照（历史结果）
历史一次测试报告（`TEST_REPORT.md`）中：
- Val Loss: 0.9563
- AUC: 0.8485
- F1: 0.3333（阈值=0.7）
- Precision: 0.9000（高精度，低召回）
- Recall: 0.2045
- Rule Consistency: 0.7628

这说明：
- 排序能力（AUC）不错
- 阈值策略偏保守（Precision 高但 Recall 低）

## 4. 指标解释与阈值口径

- AUC 不依赖阈值，衡量“违规样本分数是否更高”。
- F1/Precision/Recall 强依赖阈值。
  - 若模型输出整体偏低，阈值 0.7 会导致 F1=0 的假象。
  - 建议在验收阶段先使用 0.5 作为对照阈值。

## 5. 输出目录约定

- `reports/testing/`：测试报告 JSON、可解释性图片
- `reports/`：允许放汇总图片

> 禁止在根目录新增测试报告 Markdown。若需要书面报告，请写入 `docs/guides/`。
