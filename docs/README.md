# 文档索引（docs/）

本仓库采用“文档驱动交付”。从现在开始：

- **根目录只保留入口型文档**：`README.md`、`QUICKSTART.md`。
- **所有正式文档统一放在 `docs/`**，并在此索引登记。
- **所有训练/测试生成物统一输出到 `reports/` 或 `artifacts/`**，禁止在根目录新增报告类 Markdown。

## 1. 快速导航（按你要做的事）

- **我想快速跑起来**：
  - `QUICKSTART.md`
  - `docs/guides/TRAINING_GUIDE.md`
- **我想看整体架构/数据流**：
  - `docs/design/Design-ITER-2025-01.md`（完整设计，权威来源）
  - `docs/architecture/DATA_FLOW.md`（训练数据流图）
- **我想看某次迭代的定版资料**：
  - `docs/iterations/ITER-2025-01/`
- **我想看历史会话/评审记录（只读）**：
  - `docs/archive/`

## 2. 目录约定

- **architecture/**：稳定的系统架构与数据流（跨迭代复用）
- **guides/**：可执行的使用指南（安装/训练/测试/排障）
- **iterations/**：按迭代归档的“需求/设计/开发/测试”定版文档
- **templates/**：模板库（新增文档必须从模板复制）
- **references/**：论文/开题等参考资料（不与工程文档混用）
- **archive/**：历史产物（会话总结、评审过程、旧报告；只读不维护）

## 3. 写文档的规则（最重要）

- **单一入口**：新增/更新文档后，必须更新本文件的索引，并在根 `README.md` 仅保留必要链接。
- **单一真相**：同一主题只能有 1 份“权威文档”（其余要么合并，要么归档）。
- **可复现**：所有“命令/指标/结论”必须可被脚本或命令复现（写清楚输入、输出目录）。

## 4. 当前文档清单

### 4.1 架构（architecture）
- `docs/architecture/DATA_FLOW.md`

### 4.2 使用指南（guides）
- `docs/guides/TRAINING_GUIDE.md`
- `docs/guides/TESTING_GUIDE.md`
- （待补）`docs/guides/INSTALLATION_CONDA.md`

### 4.3 迭代（iterations）
- `docs/iterations/ITER-2025-01/REQUIREMENT.md`
- `docs/iterations/ITER-2025-01/DESIGN.md`
- `docs/iterations/ITER-2025-01/DEVELOPMENT.md`
- `docs/iterations/ITER-2025-01/TESTING.md`

### 4.4 模板（templates）
- `docs/templates/`

### 4.5 参考资料（references）
- `docs/requirement/user-stories/`（后续逐步搬迁到 `docs/references/`）

### 4.6 归档（archive）
- `docs/archive/SESSION_SUMMARIES/`
- `docs/archive/REVIEW_RECORDS/`
