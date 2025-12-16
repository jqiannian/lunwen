# Traffic Rules MVP（红灯停异常检测）

本仓库实现“红灯停”场景的异常检测 MVP：**数据 → 场景图 → 多阶段注意力GAT → 规则引擎 → 约束损失 → 训练/评估 → 报告与可解释性**。

## 快速开始

- 5 分钟跑通：`QUICKSTART.md`

## 文档入口（唯一权威索引）

- 文档总索引：`docs/README.md`
- 训练指南：`docs/guides/TRAINING_GUIDE.md`
- 测试指南：`docs/guides/TESTING_GUIDE.md`
- 架构/数据流：`docs/architecture/DATA_FLOW.md`、`docs/iterations/ITER-2025-01/DESIGN.md`

## MVP常用命令

### 训练（Smoke Test）

```bash
python3 tools/train_red_light.py train --epochs 2 --max-samples 5 --device cpu
```

### 训练（标准）

```bash
python3 tools/train_red_light.py train --epochs 50 --device cpu
```

### 测试/验收

以 `tools/test_red_light.py --help` 为准（指南见 `docs/guides/TESTING_GUIDE.md`）。

## 输出目录约定（必须遵守）

- `artifacts/`：checkpoint、伪标签等训练产物
- `reports/`：训练曲线、测试报告、可解释性图片
- 禁止在根目录新增“训练/测试/会话总结”类 Markdown（统一放入 `docs/` 或输出到 `reports/`）

## 代码入口

- 训练 CLI：`tools/train_red_light.py`
- 规则引擎：`src/traffic_rules/rules/red_light.py`
- 约束损失：`src/traffic_rules/loss/constraint.py`
- 场景图构建：`src/traffic_rules/graph/builder.py`

> 依赖管理将统一切换到 conda-forge（见后续 `docs/guides/INSTALLATION_CONDA.md`）。
