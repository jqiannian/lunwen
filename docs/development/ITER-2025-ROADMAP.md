# 开发总路线图（ITER-2025 系列）

## 元数据
| 字段 | 内容 |
| --- | --- |
| 文档版本 | v0.2 |
| 覆盖迭代 | ITER-2025-01 / ITER-2025-02 / ITER-2025-03 |
| 技术负责人 | 待指派 |
| 状态 | 草稿 |
| 最后更新时间 | 2025-11-30 |
| 关联需求 | `docs/requirement/iterations/Requirement-ITER-2025-01.md` 及后续迭代需求文档 |
| 关联设计 | `docs/design/Design-ITER-2025-01.md`（后续迭代另建） |
| 关联测试 | `docs/testing/Testing-ITER-2025-01.md`（后续迭代另建） |

## 1. 路线概览
- 技术栈：Python 3.11、Poetry、PyTorch 2.4（CUDA 12.1 for RTX 4090）、torchvision、OpenCV、pydantic、prometheus-client。
- 目标：从红灯停 MVP 起步，逐步扩展到多规则语义注入、真实数据接入、自训练与可解释指标体系。

## 2. 迭代计划

| 迭代 | 时间范围 | 目标 | 核心交付物 | 依赖 | 退出标准 |
| --- | --- | --- | --- | --- | --- |
| ITER-2025-01（MVP） | 2025-12-01 ~ 2025-12-15 | 打通红灯停数据→模型→规则→可解释闭环 | `Requirement-ITER-2025-01.md`、`Design-ITER-2025-01.md`、`Testing-ITER-2025-01.md`、`Development-ITER-2025-01.md`、合成数据脚本、GAT+注意力模型、规则 DSL、训练/测试 CLI、Prometheus 指标、attention 报告 | GPU 资源（RTX 4090）、BDD100K 样本权限 | CLI 测试三场景通过；注意力可视化与违规证据链齐备；监控指标齐全 |
| ITER-2025-02（Data+Rules） | 2025-12-16 ~ 2026-01-15 | 接入真实数据集、扩展车速/车道/安全距离等规则、完善评估指标 | 数据解析器、规则 DSL 热更新、冲突检测、数据质量报告、迭代 2 设计/测试文档 | MVP 代码基线、真实数据许可、规则文案 | 数据加载成功率 >99%；新增规则在日志/指标中可追踪；评估脚本自动生成 |
| ITER-2025-03（Advanced） | 2026-01-16 ~ 2026-02-15 | 自训练 + 记忆增强 + 可解释性增强 + 指标自动化发布 | 自训练 orchestrator、伪标签 pipeline、记忆库持久化、指标/报告自动提交、部署脚本、迭代 3 文档 | 前两次迭代产物、GPU 资源、业务验收标准 | 自训练循环稳定（提供命令与指标）；指标/报告自动化；部署与回滚脚本完成 |

## 3. 环境与依赖
- **运行环境**：开发侧 macOS + M3，CI 侧 Ubuntu 22.04，训练/测试默认 1×RTX 4090（CUDA 12.1）。
- **依赖管理**：`requirements.txt` 锁定 GPU 变体（`torch==2.4.1+cu121` 等），其余依赖按 CPU 版维护。所有新依赖必须在 PR 中同步更新该文件及 README 说明。
- **基础命令**：
  ```bash
  poetry install
  poetry run black --check --line-length 100 .
  poetry run ruff check src tests
  poetry run mypy src --strict
  poetry run pytest --maxfail=1 --cov=src --cov-report=xml
  ```

## 4. 风险与缓解
| 风险 | 等级 | 描述 | 缓解措施 |
| --- | --- | --- | --- |
| 数据许可受限 | 中 | 真实数据集审批延迟 | MVP 先用合成+小规模 BDD100K 子集；审批通过后增量接入 |
| GPU 资源不足 | 中 | QA/训练排队 | 采用批量调度 + 低批量 CPU fallback；Prometheus 记录降级原因 |
| 规则冲突 | 高 | 多规则引入后逻辑不一致 | 设计规则 DSL 冲突检测模块；在开发/测试文档中列出优先级与冲突策略 |
| 自训练噪声 | 高 | 置信度阈值不当导致伪标签污染 | 设置信任阈值、attention 一致性 Loss、Prometheus 告警并允许一键暂停 |

## 5. 交付物追踪
| 迭代 | 必须更新的文档/资产 | 备注 |
| --- | --- | --- |
| ITER-2025-01 | README、Requirement-ITER-2025-01、Design-ITER-2025-01、Development-ITER-2025-01、Testing-ITER-2025-01、requirements.txt、configs/mvp.yaml、scripts/setup_mvp_env.sh | 已建成；需持续同步 |
| ITER-2025-02 | README、Requirement-ITER-2025-02、Design-ITER-2025-02、Development-ITER-2025-02、Testing-ITER-2025-02、规则 DSL 配置、数据质量报告 | 待创建 |
| ITER-2025-03 | README、Requirement-ITER-2025-03、Design-ITER-2025-03、Development-ITER-2025-03、Testing-ITER-2025-03、部署/运维文档、自训练指标报告 | 待创建 |

## 6. 工作流约束
1. 所有代码改动必须引用对应需求/设计章节，并在 PR 模板中列出受影响的文档。
2. 自训练、规则 DSL、监控配置均需在 README 迭代表与本路线图内同步说明。
3. 未完成 checklist（文档更新、lint/test、attention 报告、Prometheus 指标）的任务不得进入下一迭代。







