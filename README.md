# 文档管理总览

本仓库采用“文档驱动交付”模式：任何需求、设计、开发、测试、部署、用户与运维活动都必须先在对应文档中完成记录，再进入下一环节。所有文档统一采用 Markdown，并存放在 `lunwen/docs/` 目录下，模板位于 `lunwen/docs/templates/`。本文档既是入口索引，也是执行规范。

## 1. 文档索引

| 文档类型 | 目标 | 模板路径 | 生成位置 | 责任人 | 更新触发 |
| --- | --- | --- | --- | --- | --- |
| README | 规则总览、索引、迭代计划 | `docs/templates/README_TEMPLATE.md` | `README.md` | 产品负责人 | 每次新规则 / 迭代 |
| 需求文档 | 原始需求→业务→系统拆解 | `docs/templates/REQUIREMENTS_TEMPLATE.md` | `docs/requirement/iterations/迭代ID.md` | 产品经理 | 需求冻结前 |
| 设计文档 | 系统/概要/详细/数据库设计 | `docs/templates/DESIGN_TEMPLATE.md` | `docs/design/迭代ID.md` | 架构师 | 设计评审前 |
| 开发文档 | 环境、配置、开发日志、API | `docs/templates/DEVELOPMENT_TEMPLATE.md` | `docs/development/迭代ID.md` | 技术负责人 | 开发中每日更新 |
| 测试文档 | 单测/集成/验收/缺陷映射 | `docs/templates/TEST_TEMPLATE.md` | `docs/testing/迭代ID.md` | QA | 每轮测试后 |
| 部署文档 | 架构、脚本、手册、回滚 | `docs/templates/DEPLOYMENT_TEMPLATE.md` | `docs/deployment/迭代ID.md` | 运维 | 上线前 |
| 用户文档 | 操作手册、测试账号 | `docs/templates/USER_DOCUMENT_TEMPLATE.md` | `docs/user-guide/版本号.md` | 培训负责人 | 每次对外发布 |
| 运维文档 | 堡垒机、运维手册、应急 | `docs/templates/OPERATIONS_TEMPLATE.md` | `docs/operations/版本号.md` | 运维主管 | 运维策略调整时 |

> 创建实际文档前，必须先复制对应模板；同时在 README 文档索引中登记文件名、责任人、状态。

**当前已创建文档（根据 docs/ 目录扫描）**
- 需求迭代：`docs/requirement/iterations/Requirement-ITER-2025-01.md`（MVP 需求与多迭代规划）
- 需求素材：`docs/requirement/user-stories/最简模型2.md`、`论文提纲.md`、`开题报告模板（初稿）.doc`
- 设计文档：
  - `docs/design/Design-ITER-2025-01.md`（工程架构与实现细节）✅ 已细化
  - `docs/design/ALGORITHM_DESIGN_OPTIONS.md`（3种算法方案对比与数学推导）✅
  - `docs/design/DESIGN_REVIEW_SUMMARY.md`（算法评审决策记录）✅
  - `docs/design/TECHNICAL_CORRECTIONS.md`（技术勘误：6个关键问题修正）✅
  - `docs/design/REVIEW_RESPONSE_SUMMARY.md`（评审响应摘要）✅
- 开发文档：
  - `docs/development/ITER-2025-ROADMAP.md`（涵盖 ITER-2025-01~03 开发计划）
  - `docs/development/Development-ITER-2025-01.md`（本迭代任务分解）
  - `docs/development/IMPLEMENTATION_PROGRESS.md`（实施进度报告）✅
  - `docs/development/CODE_IMPLEMENTATION_SUMMARY.md`（代码实施总结）✅
- 测试占位：`docs/testing/Testing-ITER-2025-01.md`（测试策略骨架）
- 模板库：`docs/templates/`（README/需求/设计/开发/测试/部署/用户/运维 8 份模板）
- 快速开始：
  - `QUICKSTART.md`（5分钟验证指南）✅
  - `DATA_LOADING_GUIDE.md`（数据加载器使用指南，15页）✅ NEW
  - `DATA_LOADER_IMPLEMENTATION.md`（数据加载器实现报告）✅ NEW

## 2. 迭代计划（必填）

每个迭代都需要新增以下记录，建议放置在 `docs/requirement/iterations/迭代ID.md` 并在 README 中同步。表格字段定义如下：

| 字段 | 说明 |
| --- | --- |
| 迭代编号 | 形如 `ITER-2024-01`，与分支命名一致 |
| 时间范围 | 起止日期 |
| 新增业务需求 | 按编号列出，需链接到需求文档中对应章节 |
| 影响范围 | 涉及模块、系统、外部依赖 |
| 必须更新的文档 | 列出须同步的需求/设计/开发/测试/部署/用户/运维文档 |
| 关键里程碑 | 需求冻结、设计评审、开发完成、测试通过、上线 |
| 状态 | 计划中 / 进行中 / 完成 |

| 迭代编号 | 时间范围 | 新增业务需求 | 影响范围 | 必须更新的文档 | 关键里程碑 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| ITER-2025-01（MVP） | 2025-12-01 ~ 2025-12-15 | BIZ-001（红灯停最小闭环） | 数据摄取、规则推理、训练脚本 | `docs/requirement/iterations/Requirement-ITER-2025-01.md`、`docs/design/Design-ITER-2025-01.md`、`docs/testing/Testing-ITER-2025-01.md`、`docs/development/ITER-2025-ROADMAP.md`、`docs/development/Development-ITER-2025-01.md` | 需求冻结 12-02 / 开发完成 12-10 / 测试 12-13 / 交付 12-15 | 计划中 |
| ITER-2025-02（Data+Rules） | 2025-12-16 ~ 2026-01-15 | BIZ-002、BIZ-003 | 数据接入、规则 DSL、评估 | 同上 + `docs/design/ITER-2025-02.md`、`docs/testing/ITER-2025-02.md`（待创建） | 需求冻结 12-20 / 设计评审 12-24 / 测试 01-12 / 交付 01-15 | 计划中 |
| ITER-2025-03（Advanced） | 2026-01-16 ~ 2026-02-15 | BIZ-004、BIZ-005 | 自训练、记忆库、可解释性、指标 | 同上 + 对应 design/testing/deployment 文档 | 需求冻结 01-20 / 测试 02-10 / 发布 02-15 | 计划中 |

## 3. 模板使用与提交流程

1. **复制模板**：在对应目录内复制模板并重命名（例如 `cp docs/templates/REQUIREMENTS_TEMPLATE.md docs/requirement/iterations/ITER-2024-01.md`）。  
2. **填写元数据**：所有模板首屏均包含版本、状态、责任人、最后更新日期、关联迭代、相关需求等字段，必须填写。  
3. **Checklist 勾选**：模板末尾的检查清单必须逐项确认，便于评审人对照。  
4. **同步 README**：新增或更新文档后，在 README 的索引表或迭代计划章节同步状态。  
5. **评审与签字**：提交 PR 前需在文档中标注“评审记录”，包含评审人、时间、意见。  
6. **禁止跳步**：没有对应文档或 checklist 未通过的任务禁止进入下一环节（例如测试文档未准备不允许执行测试）。

## 4. 更新责任与稽核

- **触发点**：需求冻结、设计评审、每日开发、测试回合、上线/回滚都必须触发相应文档更新。  
- **稽核**：每周由产品负责人检查 README 中的文档索引与最新状态是否一致。发现缺失需在 24h 内补齐。  
- **敏感信息**：严禁在任何文档中硬编码敏感信息（密码、Token 等），统一引用配置或密钥管理系统。  
- **安全与合规**：设计、开发、运维文档必须包含安全、权限、审计、脱敏策略说明。  
- **测试约束**：前端每页面必须附 Selenium 自动化脚本路径；后端每方法需要单测；集成测试需覆盖全部业务约束；验收测试记录主线流程；性能/压力/流量测试需待业务负责人确认后执行；所有测试记录必须附可重复运行的环境还原脚本或命令。

## 5. 贡献工作流（供 AI 与成员遵循）

1. **查阅 README** → 判断需要创建/更新的文档。  
2. **更新/创建文档** → 依据模板填充，确保 checklist 通过。  
3. **开发或测试实现** → 代码变更必须引用对应需求/设计/测试章节。  
4. **PR & 评审** → 提交代码前附上本 README 的索引 diff 与文档变更链接。  
5. **发布后回填** → 部署、用户、运维文档需要记录实际上线信息、回滚演练结果与运维监控链接。

## 6. 后续规划

- `docs/templates/` 中的模板会随着项目演进补充更多示例。  
- 未来迭代将增加自动校验脚本（校验 README 与目录一致性、验证 checklist 勾选情况）。  
- 如需新增文档类型或调整模板结构，务必先更新 README 中的规则说明，并获产品经理同意。

## 7. Python 代码规范总览

- **基础风格**：全面遵守 PEP 8；使用 `pyproject.toml` 统一配置 `black`（`line-length = 100`、`target-version = ["py311"]`）与 `isort`（`profile = "black"`）。模块导入顺序为：标准库 → 第三方 → 项目内，且必须保留空行分隔。  
- **命名与结构**：变量/函数统一 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE_CASE`；包结构按业务域划分（如 `src/<domain>/controllers|services|repositories`），避免巨型 util 模块。禁止在模块顶层执行数据库/网络请求等副作用，使用 `if __name__ == "__main__":` 作为唯一入口。  
- **类型与文档**：启用 `from __future__ import annotations`，要求所有函数、方法、公共属性声明类型；docstring 采用 Google 风格，包含 `Args/Returns/Raises/Examples`，并解释幂等性或副作用。类型校验使用 `mypy --strict`，业务关键结构需定义 `TypedDict` 或 `pydantic` 模型。  
- **错误处理与日志**：禁止裸 `except`；捕获具体异常并使用 `logger.opt(exception=True)` 或 `logger.exception` 记录堆栈，日志需包含 `trace_id`、`user_id`、`biz_key` 等关键信息。所有日志语句必须通过 `logging.config.dictConfig`/`structlog` 统一配置，禁止 `print`。  
- **安全与配置**：严禁 `eval/exec`、`pickle.loads` 处理用户输入；密码、Token、密钥等从环境变量读取，使用 `pydantic.BaseSettings` 或 `dynaconf` 校验并加密落盘。入参须经过白名单校验，SQL 访问必须使用 ORM/参数化。  
- **依赖与打包**：依赖通过 `poetry add` 管理，`poetry.lock` 必须提交且禁止手工编辑；每周运行 `poetry run pip-audit` 并记录结果。内部可复用模块发布为私有 wheel，禁止直接复制源码。  
- **测试与质量**：统一使用 `pytest`，命名 `test_<module>_<scenario>`；后端每个 public 方法至少一个单测，覆盖率需 ≥90%（`pytest --cov=src --cov-report=xml`）。集成/契约测试脚本存放于 `tests/integration/`，必须附数据构造与清理脚本。  
- **静态检查**：`ruff` 负责 lint/简易格式化，启用规则集 `ruff = { select = ["E", "F", "B", "I", "UP"], ignore = ["E501"] }`；安全检查使用 `bandit -r src -ll`；CI 对 lint、类型、测试全部失败即阻断合并。  
- **监控与可观测性**：关键服务通过 `prometheus_client` 输出 QPS/延迟/错误率；核心链路集成 OpenTelemetry 追踪 ID，并在异常处理逻辑中增加降级或熔断说明；指标与日志字段需在开发文档中登记。

### Python 执行与验收 Checklist

| 步骤 | 命令 / 行动 | 说明 |
| --- | --- | --- |
| 依赖安装 | `poetry install` | 禁止跳过锁文件，安装后运行 `poetry check` |
| 代码风格 | `poetry run black --check --line-length 100 .` | 不通过时先格式化再提交 |
| 导入排序 | `poetry run isort --check-only .` | 配置与 black 一致 |
| Lint | `poetry run ruff check src tests` | 不得忽略默认规则 |
| 安全扫描 | `poetry run bandit -r src -ll` & `poetry run pip-audit` | 高风险漏洞需说明豁免理由 |
| 类型检查 | `poetry run mypy src --strict` | 禁止随意加 `# type: ignore`，需注释原因 |
| 单元测试 | `poetry run pytest --maxfail=1 --cov=src --cov-report=xml` | 覆盖率报告需附在 PR 描述 |
| 集成/回归 | `poetry run pytest tests/integration -m \"not slow\"` | 完整场景另附环境还原脚本 |
| 打包校验 | `poetry build` & `poetry run twine check dist/*`（如需发布） | 确保版本号、依赖声明正确 |

- Merge Request 必须附上述命令的截图或 CI 链接，并在 PR 模板中粘贴 “通过/失败” 勾选结果；若因客观原因跳过某项，需获得技术负责人书面确认。  

> 所有贡献者（含 AI）务必严格遵守上述规则，否则变更将不予合并。


## 8. 代码目录结构（ITER-2025-01 骨架）

> 依据 `docs/requirement/iterations/Requirement-ITER-2025-01.md` 与 `docs/design/Design-ITER-2025-01.md`，已经在仓库根目录下生成红灯停 MVP 的 Python 项目骨架，供架构与实现评审。

```
lunwen/
├── pyproject.toml            # Poetry 配置 + 质量工具
├── poetry.lock               # 暂存锁文件（需实现阶段重新生成）
├── .env.example              # DATA_ROOT、PROMETHEUS_PORT 等变量示例
├── configs/
│   └── mvp.yaml              # 运行时/模型/规则/监控占位参数
├── scripts/
│   ├── setup_mvp_env.sh      # 环境初始化指引
│   ├── prepare_synthetic_data.py
│   └── render_attention_maps.py
├── src/traffic_rules/
│   ├── config/loader.py      # Pydantic 配置模型
│   ├── data/traffic_dataset.py
│   ├── graph/builder.py
│   ├── models/gat_attention.py
│   ├── memory/memory_bank.py
│   ├── rules/red_light.py
│   ├── loss/constraint.py
│   ├── explain/attention_viz.py
│   ├── self_training/pseudo_labeler.py
│   └── monitoring/meters.py
├── tools/
│   ├── train_red_light.py    # Typer CLI 占位
│   └── test_red_light.py
├── tests/
│   ├── unit/test_placeholders.py
│   └── integration/traffic_rules/.gitkeep
├── data/.gitkeep
├── artifacts/
│   ├── checkpoints/.gitkeep
│   └── pseudo_labels/.gitkeep
└── reports/.gitkeep
```

该树形结构以业务链路（数据→图→模型→规则→监控）为主线，后续每个模块在实现时需引用对应的需求、设计与开发文档，并遵循 README 第 5~7 节的工作流与编码规范。


## 9. CLI 使用与实现进度

- `poetry run python tools/train_red_light.py run --dry-run`：构建数据→图→GAT+Memory→约束损失→监控指标的训练链路，可通过 `--epochs` / `--experiment-name` 等参数自定义运行。
- `poetry run python tools/test_red_light.py run --report-dir reports`：执行推理与规则校验，并生成注意力热力图 Markdown、伪标签 manifest。
- 训练/测试默认复用 `src/traffic_rules/utils/demo_data.py` 中的合成样本；接入真实数据集时需确保 `TrafficLightDataset` 返回的 `SceneContext.extra.entities` 具备实体坐标/速度等信息。
- Prometheus 监控在训练 CLI 中自动启动（端口由 `configs/mvp.yaml` 控制），日志通过 `traffic_rules.monitoring.meters` 统一注入 `trace_id`。
