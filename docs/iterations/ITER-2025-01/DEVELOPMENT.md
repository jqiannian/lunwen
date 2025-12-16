# 开发计划（ITER-2025-01）

> 迭代目标：在 Python 技术栈下交付“红灯停”无监督异常检测 MVP，聚焦交通规则、语义关系推理与注意力增强。

## 元数据
| 字段 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 迭代编号 | ITER-2025-01 |
| 技术负责人 | 待指派 |
| 状态 | 草稿 |
| 最后更新时间 | 2025-11-30 |
| 关联需求 | `docs/iterations/ITER-2025-01/REQUIREMENT.md` |
| 关联设计 | `docs/iterations/ITER-2025-01/DESIGN.md` |
| 关联测试 | `docs/iterations/ITER-2025-01/TESTING.md` |

## 1. 里程碑与交付
| 日期 | 里程碑 | 说明 / 交付物 |
| --- | --- | --- |
| 12-01 | 目录骨架可审核 | README 代码结构、Poetry 配置、src/traffic_rules 占位模块、CLI/Test 骨架 |
| 12-02 | 需求冻结 | README/需求/设计/测试文档同步；确认 Python 技术栈 |
| 12-03 | MVP 模块实现 | GraphBuilder/GAT+Memory、RuleEngine/Loss、可解释性、伪标签、CLI & 测试 |
| 12-05 | 数据与配置就绪 | `TrafficLightDataset`、合成数据脚本、`configs/mvp.yaml` |
| 12-10 | 模型与注意力模块完成 | GAT + 记忆注意力、规则损失、指标埋点、可解释脚本 |
| 12-13 | 测试/验收 | `pytest` + CLI 测试 + 注意力可视化输出 |
| 12-15 | 交付评审 | 提交报告、指标、Artifacts、文档更新 |

## 2. 技术环境
- **语言/运行时**：Python 3.11（Poetry 管理，`poetry.lock` 必须提交）。
- **依赖**：`torch>=2.4`, `torchvision`, `opencv-python`, `numpy`, `pydantic`, `networkx`, `prometheus-client`, `rich`.
- **硬件**：开发使用 macOS + M3，本地 CPU/GPU；测试/训练在 Ubuntu 22.04 + NVIDIA RTX 4090（CUDA 12.1）。
- **基线命令**：
  ```bash
  poetry install
  poetry run black --check --line-length 100 .
  poetry run ruff check src tests
  poetry run mypy src --strict
  poetry run pytest --maxfail=1 --cov=src --cov-report=xml
  ```

## 3. 工作分解结构
| 模块 | Owner | 任务拆分 | 产出 |
| --- | --- | --- | --- |
| 数据摄取 (WBS-01) | Data 工程 | `TrafficLightDataset`、BDD100K/Cityscapes 解析、合成数据脚本、停线距离计算 | `src/data/traffic.py`、`scripts/prepare_synthetic_data.py` |
| 语义图建模 (WBS-02) | AI 工程 | 实体特征编码、邻接矩阵构建、规则 DSL schema | `src/models/graph_builder.py` |
| 注意力增强 (WBS-03) | AI 工程 | 多头 GAT、记忆注意力检索、attention dump API | `src/models/attention.py`、`tools/visualize_attention.py` |
| 规则/损失 (WBS-04) | AI 工程 | 红灯规则 DSL、约束 Loss、命中日志 | `src/rules/red_light.py`、`src/loss/constraint.py` |
| 训练与 CLI (WBS-05) | 平台 | `tools/train_red_light.py`、`tools/test_red_light.py`、配置管理 | `configs/mvp.yaml`、`artifacts/checkpoints/` |
| 监控与指标 (WBS-06) | 平台 | Prometheus 指标、违规统计、日志结构化 | `src/monitoring/meters.py`、Grafana 面板链接 |
| 文档 & 可解释性 (WBS-07) | 全员 | README 索引、需求/设计/测试同步、注意力热力图、报告模板 | `docs/` 更新、`reports/mvp_attention.md` |
| 自训练与伪标签 (WBS-08) | AI 工程 | 置信度阈值策略、伪标签写入/回放、指标上报 | `src/self_training/pseudo_labeler.py`、`artifacts/pseudo_labels/`、监控事件 |

## 4. 风险与缓解
| 风险 | 影响 | 级别 | 缓解策略 |
| --- | --- | --- | --- |
| 数据许可或体积过大 | 无法在本地快速迭代 | 中 | MVP 使用合成 + 100 条 BDD100K 子集，真实数据延后 |
| 注意力与规则结果不一致 | 可解释性受质疑 | 高 | 在训练中引入 attention-consistency loss，并将权重与违规结果写入同一日志 |
| 资源受限（GPU） | 训练时间不可控 | 中 | 先用 synthetic 进行 sanity check，CI 仅跑小批次；提交前再跑完整训练 |
| 规则 DSL 扩展性不足 | 后续多规则迭代困难 | 中 | 采用 `pydantic` 定义规则 schema + 热更新机制；在本迭代先实现红灯规则 |

## 5. 质量策略
- **单元测试**：覆盖数据集解析、图构建、注意力模块、规则损失（`tests/unit`）。
- **集成测试**：`pytest tests/integration/traffic_rules`，模拟停车/闯红灯/绿灯通过的 CLI 场景。
- **Selenium/前端**：MVP 无前端，仅记录需求，后续如有 UI 再补。
- **静态检查**：`black + isort + ruff + mypy + bandit + pip-audit`，CI 阻断。
- **可观测性**：Prometheus 指标（loss、违规数、attention 一致率）、结构化日志（trace_id、scene_id、rule_name）。

## 6. 发布前检查
- [ ] README、需求、设计、测试文档更新且评审通过。
- [ ] 全量 lint/type/test 通过，并在 PR 附上命令输出或 CI 链接。
- [ ] `scripts/setup_mvp_env.sh`、`scripts/prepare_synthetic_data.py`、`scripts/render_attention_maps.py` 可执行。
- [ ] artifacts：模型 checkpoint、注意力权重、违规报告、Prometheus 指标截图。
- [ ] 回滚方案：保留上一版本 checkpoint + configs，提供 `tools/test_red_light.py --checkpoint <prev>` 验证指令。

## 7. 变更记录
| 日期 | 版本 | 内容 | 责任人 |
| --- | --- | --- | --- |
| 2025-11-30 | v0.1 | 新建 ITER-2025-01 开发计划，明确任务拆分与注意力增强交付物 | 技术负责人（AI） |

