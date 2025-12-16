---
name: mvp-docs-deps-roadmap
overview: 整理并治理文档与依赖管理，修复当前阻塞问题，形成可复现的训练/测试闭环，并给出MVP剩余任务的推进里程碑与验收标准。
todos:
  - id: docs-info-arch
    content: 建立 docs 五层结构（README/architecture/guides/iterations/archive）并迁移整合根目录散落文档
    status: completed
  - id: deps-conda
    content: 新增 conda-forge 优先的 environment.yml 与 environment-dev.yml，并编写 INSTALLATION_CONDA 指南，废弃/收敛 requirements.txt/poetry.lock
    status: completed
  - id: unblock-cursorignore
    content: 修复 .cursorignore 误过滤（确保 src/traffic_rules/data 与 src/traffic_rules/models 可见）
    status: completed
  - id: fix-demo-import
    content: 修复 src/traffic_rules/utils/demo_data.py 导入路径，确保演示模块可用
    status: completed
  - id: mvp-train-e2e
    content: 在新环境下完成最小训练回归与标准训练，产出 checkpoints 与 training_curves
    status: completed
  - id: mvp-test-acceptance
    content: 补齐/运行测试验收闭环，输出 reports/testing JSON 与可解释证据链
    status: completed
---

# MVP文档与依赖治理 + 剩余任务推进计划

## 目标与约束

- **目标**：让仓库达到“结构清晰、环境可复现、训练可跑通、结果可验收、报告可解释”的MVP交付状态。
- **关键约束**：
- 依赖管理以 **conda-forge** 为主（必要时可额外加入 `pytorch` channel 仅用于 PyTorch）。
- Python 版本以 **3.12** 为准（与项目环境规范一致）。
- 任何生成型产物（训练曲线、报告、checkpoint、伪标签等）必须进入 `reports/` 或 `artifacts/`，不污染根目录与 `docs/`。

## 现状结论（来自研究）

### 文档现状

- 根目录存在大量“会话/总结/报告”类 Markdown（如 `SESSION_*.md`、`TRAINING_REPORT_V2.md`、`MVP_TRAINING_REPORT.md`、`TEST_REPORT.md`、`REVIEW_FINAL_SUMMARY.md`），与 `docs/` 里的正式文档混在一起，入口不唯一，内容有重复与口径漂移风险。

### 依赖现状

- 依赖来源混用：`requirements.txt` + `pyproject.toml/poetry.lock` + conda 环境。
- 版本不一致：配置中常见 pin（如 `torch==2.4.1`）与机器实际安装（如 `torch==2.9.1`）不一致，导致不可复现。
- 缺失依赖导致模块不可用：
- `src/traffic_rules/explain/attention_viz.py` 依赖 `opencv-python`（`cv2`）
- `src/traffic_rules/self_training/pseudo_labeler.py` 依赖 `pandas`
- `src/traffic_rules/monitoring/meters.py` 依赖 `structlog`

### 代码/工程问题现状（按优先级）

- **P0 阻塞**：`.cursorignore` 目前包含 `data`、`models` 等规则，导致 `src/traffic_rules/models/` 与 `src/traffic_rules/data/` 下的文件在编辑器/工具中被过滤，影响协作与排障（示例：`src/traffic_rules/models/__init__.py`、`src/traffic_rules/data/__init__.py` 被过滤）。文件：[`.cursorignore`](./.cursorignore)
- **P1 高风险**：`src/traffic_rules/utils/demo_data.py` 使用错误导入路径 `from traffic_rules...`，而项目实际包路径是 `src.traffic_rules...`，导致演示数据模块不可用。文件：[`src/traffic_rules/utils/demo_data.py`](./src/traffic_rules/utils/demo_data.py)
- **P1 高风险**：训练链路可运行但环境不一致；小样本训练验证通过（`tools/train_red_light.py` 可跑），但长期训练与评估稳定性未验证。文件：[`tools/train_red_light.py`](./tools/train_red_light.py)
- **P2 结构债**：若干模块通过 `sys.path.insert` 方式引入项目根目录（例如 `graph/builder.py`、`tools/train_red_light.py`）。后续应统一为“可安装包 + 相对导入/绝对包导入”。文件：[`src/traffic_rules/graph/builder.py`](./src/traffic_rules/graph/builder.py)、[`tools/train_red_light.py`](./tools/train_red_light.py)

## 方案设计

## 文档治理方案（从“散乱”到“可维护”）

### 文档分层与目录规范

建立“入口-架构-迭代-指南-归档”的五层结构：

- **仓库入口**
- `README.md`：只保留项目简介、最短路径使用方式、核心链接（不堆长文）。
- `QUICKSTART.md`：保留 5 分钟验证指引。
- **docs/ 作为唯一正式文档根**
- `docs/README.md`：文档总索引（按‘我是谁/我要跑/我要改/我要验收’视角导航）。
- `docs/architecture/`：稳定不随迭代频繁改动的架构与数据流
- `SYSTEM_ARCHITECTURE.md`
- `DATA_FLOW.md`（整合根目录 `TRAINING_DATAFLOW_DIAGRAM.md` 的内容）
- `docs/guides/`：面向使用/操作的指南（可执行）
- `INSTALLATION_CONDA.md`
- `TRAINING_GUIDE.md`
- `TESTING_GUIDE.md`
- `docs/iterations/ITER-2025-01/`：该迭代的需求/设计/开发/测试定版文档
- `REQUIREMENT.md`（权威：`docs/iterations/ITER-2025-01/REQUIREMENT.md`）
- `DESIGN.md`（权威：`docs/iterations/ITER-2025-01/DESIGN.md`）
- `DEVELOPMENT.md`（权威：`docs/iterations/ITER-2025-01/DEVELOPMENT.md`）
- `TESTING.md`（权威：`docs/iterations/ITER-2025-01/TESTING.md`）
- `docs/references/`：论文/开题材料等参考资料（不与工程文档混用）
- `docs/archive/`：历史会话/旧报告/评审过程产物（明确“只读、不再维护”）

### 文档命名与状态规则

- 文件名统一：全英文大写目录 + 英文下划线或 ITER 前缀（参考资料可保留中文，但放入 `docs/references/`）。
- 每份“正式文档”首屏包含：版本号、最后更新时间、状态（Draft/Reviewed/Final）、对应迭代、对应代码入口。
- **禁止**根目录新增报告类 Markdown；所有“生成型报告”输出到 `reports/`。

### 文档迁移清单（拟）

- 根目录：
- `TRAINING_DATAFLOW_DIAGRAM.md` → `docs/architecture/DATA_FLOW.md`
- `MVP_TRAINING_REPORT.md` + `TRAINING_REPORT_V2.md` → `docs/guides/TRAINING_GUIDE.md`（合并去重，保留一份权威口径）
- `TEST_REPORT.md` → `docs/guides/TESTING_GUIDE.md`
- `SESSION_*.md`、`REVIEW_FINAL_SUMMARY.md`、`SESSION_COMPLETE_SUMMARY.md` 等 → `docs/archive/`

## 依赖治理方案（conda-forge 优先，可复现）

### 单一事实源（Single Source of Truth）

- 以 `environment.yml` / `environment-dev.yml` 为**唯一依赖事实源**。
- `requirements.txt` 与 `poetry.lock` 不再作为安装入口（迁移完成后可删除或标记 deprecated）。
- `pyproject.toml` 保留 **lint/format/test 工具配置**（`black/ruff/mypy/pytest`），不再承担依赖解析。

### 环境文件设计

- `environment.yml`（运行环境）：只包含运行训练/测试所需最小依赖。
- `environment-dev.yml`（开发环境）：在运行环境基础上增加 `pytest/ruff/mypy/black` 等。
- 建议 channel 顺序：`conda-forge` 在前，必要时增加 `pytorch` 用于 torch（macOS/CPU 友好）。
- 统一 pin 关键版本（至少：python、pytorch、numpy、opencv、pydantic）。
- 输出 `environment.lock.yml`（用 `conda env export --no-builds` 生成）作为锁定快照（可选，但推荐）。

### 版本对齐策略

- 统一到 Python 3.12，并同步更新：
- 文档中的 Python 版本描述（如 `QUICKSTART.md`、`README.md`）
- `pyproject.toml` 的 `requires-python`
- torch 版本选择：以“项目文档与现有代码测试通过”为准；若要保持与设计文档一致，可先固定到 `2.4.1`，后续再评估升级。

## 项目问题修复策略（按先后顺序）

1. **解除 `.cursorignore` 对源代码的误过滤**：将忽略规则从泛化的 `data`/`models` 改为仅忽略仓库根下的 `data/`（数据集）、`artifacts/`、`reports/`、`logs/` 等生成目录，确保 `src/traffic_rules/data/`、`src/traffic_rules/models/` 可被工具读取。文件：[`./.cursorignore`](./.cursorignore)
2. **修复演示数据模块导入路径**：将 `src/traffic_rules/utils/demo_data.py` 中 `traffic_rules...` 改为 `src.traffic_rules...`（或在包化后改为 `traffic_rules...` 的正确安装路径）。文件：[`src/traffic_rules/utils/demo_data.py`](./src/traffic_rules/utils/demo_data.py)
3. **补齐缺失依赖**（通过 conda-forge）：确保 `cv2/pandas/structlog` 可用，使 explain/self_training/monitoring 全量可导入。
4. **训练链路稳定性**：在新环境下跑最小训练（2 epochs + 小样本）与标准训练（50-100 epochs），输出 `reports/training_curves.png`、最佳 checkpoint，并记录指标曲线。
5. **减少 sys.path 黑魔法**（P2）：把项目变为“可 pip -e 安装”的包，逐步移除 `sys.path.insert`；这一步不阻塞 MVP，但能提升可维护性。

## MVP剩余任务推进计划（里程碑）

### Milestone A：环境可复现（1天）

- 产出：`environment.yml`、`environment-dev.yml`、`docs/guides/INSTALLATION_CONDA.md`
- 验收：新机器/新环境可一键创建环境并通过“全模块导入检查”。

### Milestone B：训练可跑通（1-2天）

- 产出：
- 可重复训练命令（CPU/mac）
- `artifacts/checkpoints/best.pth`
- `reports/training_curves.png`
- 验收：训练完成无报错，loss 曲线下降，验证集指标产生非退化趋势（至少 AUC 不长期固定在 0.5）。

### Milestone C：测试与验收闭环（1天）

- 产出：`tools/test_red_light.py`（若已存在则补齐验收输出），生成 `reports/testing/*.json`、必要的可视化图。
- 验收：三场景（parking/violation/green_pass）输出符合设计文档阈值口径。

### Milestone D：可解释性与监控可用（1天）

- 产出：
- attention 可视化输出（PNG/Markdown）
- 监控指标可打印/可导出（Prometheus endpoint 或结构化日志）
- 验收：给定一个违规样本能输出“证据链”（距离、速度、灯态、注意力聚焦）。

### Milestone E：文档收敛与交付（0.5-1天）

- 产出：重构后的 `docs/` 结构、索引、归档完成；根目录仅保留入口类文档。
- 验收：新人仅通过 `README.md` + `docs/README.md` 能完成环境创建、训练、测试、查看报告。

## 实施清单（必须按顺序执行）

1. 新增 `docs/README.md`，定义文档分层、命名规范、生成物归属规则。
2. 新建目录：`docs/architecture/`、`docs/guides/`、`docs/iterations/ITER-2025-01/`、`docs/archive/`、`docs/references/`。
3. 迁移并整合根目录散落文档到对应 `docs/` 目录；合并重复报告（训练/测试/评审/会话）。
4. 更新根目录 `README.md`：变成“唯一入口索引”，链接到 `docs/README.md`、`QUICKSTART.md`、关键脚本与核心模块。
5. 新增 conda 环境文件：`environment.yml`、`environment-dev.yml`，以 conda-forge 为主，必要时加入 `pytorch` channel。
6. 在 `docs/guides/INSTALLATION_CONDA.md` 写清楚：创建/更新/导出 lock/常见问题。
7. 处理依赖来源冲突：明确废弃 `requirements.txt` 与 `poetry.lock` 的安装职责（保留或删除按最终决策）。
8. 修复 `.cursorignore`：避免误过滤 `src/traffic_rules/data/` 与 `src/traffic_rules/models/`。
9. 修复 `src/traffic_rules/utils/demo_data.py` 的导入路径问题，确保演示数据模块可导入。
10. 在新 conda 环境中执行“全模块导入检查”脚本（rules/loss/graph/models/data/monitoring/self_training/explain）。
11. 执行最小训练回归：`python tools/train_red_light.py train --epochs 2 --max-samples 5 --device cpu`，确认能产出 checkpoint 与曲线图。
12. 执行标准训练：`--epochs 50~100`，记录指标并输出报告到 `reports/`。
13. 执行验收测试：运行 `tools/test_red_light.py`（或补齐），生成 `reports/testing/` 下的 JSON 与可视化。
14. 复核“可解释证据链”输出是否完整（距离/速度/灯态/注意力）。
15. 最终收敛：将所有“过程性/一次性”文档归档到 `docs/archive/`，保证根目录干净。

## 关键文件索引（将重点修改/新增）

- 文档：[`README.md`](./README.md)、[`QUICKSTART.md`](./QUICKSTART.md)、新增 `docs/README.md`、`docs/architecture/DATA_FLOW.md`、`docs/guides/INSTALLATION_CONDA.md`
- 依赖：新增 `environment.yml`、`environment-dev.yml`；调整 [`pyproject.toml`](./pyproject.toml)（保留工具配置，弱化/移除 poetry 依赖职责）
- 忽略规则：[`./.cursorignore`](./.cursorignore)
- 代码修复：[`src/traffic_rules/utils/demo_data.py`](./src/traffic_rules/utils/demo_data.py)
- 训练入口：[`tools/train_red_light.py`](./tools/train_red_light.py)