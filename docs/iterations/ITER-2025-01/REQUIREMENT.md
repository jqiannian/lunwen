# 需求文档（ITER-2025-01）

## 元数据
| 字段 | 内容 |
| --- | --- |
| 文档版本 | v0.4 |
| 迭代编号 | ITER-2025-01（MVP） |
| 需求负责人 | 产品经理（待指派） |
| 状态 | 草稿 |
| 最后更新时间 | 2025-11-30 |
| 关联设计 | docs/iterations/ITER-2025-01/DESIGN.md |
| 关联测试 | docs/iterations/ITER-2025-01/TESTING.md |
| 审批记录 | 待评审 |

## 1. 原始用户需求
- 来源：`docs/references/user-stories/最简模型2.md`、`docs/references/user-stories/论文提纲.md`
- 背景：需要在流程性交通场景中进行无监督异常检测，首要示例为“红灯停”规则。
- 用户痛点：
  - 缺少高质量的数据处理管道及可扩展的规则注入框架。
  - 现有模型对交通规则的理解不足，难以解释违规原因。
  - 需要减少人工标注，利用自训练与记忆模块巩固正常模式。
- 业务目标：
  - 构建可复用的交通违规检测引擎，以最小 MVP “红灯停”检测为起点，逐步扩展到多规则、多场景。
  - 融合真实数据集（BDD100K、Cityscapes）与合成数据，形成统一数据加载与验证流程。
  - 提供自训练、关系推理、语义注入等机制以提升准确率及可解释性。
- 关键约束：
  - 严格遵循交通法规语义（如红灯停、车速限制等）。
  - 模型训练需可在有限资源下完成（单机 NVIDIA RTX 4090，CUDA 12.1），默认 Python 3.12 + PyTorch 2.4+。
  - 核心模块（数据、模型、规则、注意力解释）全部以 Python 交付，依赖通过 Conda 管理。
  - 所有测试需具备可复现的数据准备脚本。

### 1.1 重点方向与技术栈约束
- **A. 交通规则合规（红灯停优先）**：MVP 必须覆盖红灯停场景，输出违规证据链（停止线距离、速度、灯色）。
- **B. 语义注入 / 关系推理**：利用图神经网络 + 规则 DSL，对车辆、交通灯、停止线等实体建模，维持语义一致性。
- **C. 注意力增强与可解释性**：要求在关系推理层中显式采用注意力机制（如多头 GAT、记忆检索注意力），并输出注意力热力图/权重供业务诊断。
- **技术栈**：Python 3.12、Conda、PyTorch 2.4+、torchvision 0.19+、OpenCV 4.9+、pydantic 2.7+、prometheus-client 0.20+；禁止混用未审查语言或脚本。所有依赖通过 `environment.yml` / `environment-dev.yml` 从 conda-forge 安装，确保版本一致性。

## 2. 业务需求说明
| 需求编号 | 标题 | 描述 | 优先级 | 业务规则 / 约束 | 计划迭代 | 实现状态 | 完成度 | 代码位置 | 验收标准 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BIZ-001 | 红灯停 MVP 场景 | 构建最小可用数据管道、关系推理模型与红灯规则约束，实现 synthetic 数据+基础测试 | P0 | 支持交通灯、车辆、停止线实体解析；CLI 训练/测试；输出违规判定 | ITER-2025-01 | 🟡 部分完成 | 85% | data/traffic_dataset.py, graph/builder.py, models/multi_stage_gat.py, rules/red_light.py, loss/constraint.py, tools/train_red_light.py, tools/test_red_light.py | CLI 训练成功、测试三种场景通过，生成违规报告与违规截图 |
| BIZ-002 | 真实数据集整合 | 接入 BDD100K、Cityscapes 数据，统一标签解析、抽样策略与评估指标 | P1 | 数据路径配置化；提供 100+ 样本验证；输出数据质量报告 | ITER-2025-02 | ❌ 未开始 | 0% | 待新增 | 数据加载成功率 >99%，评估结果可复现 |
| BIZ-003 | 多规则语义注入 | 将车速限制、车道保持、安全距离等规则形式化并注入模型 | P1 | 规则需可配置、可热更新；冲突检测 | ITER-2025-02 | 🟡 部分完成 | 15% | rules/red_light.py（仅单规则） | 每条规则均生成约束损失并在日志中可追踪 |
| BIZ-004 | 自训练与记忆增强 | 实现伪标签循环、正常模式记忆库与异常评分头 | P2 | 伪标签需包含置信度阈值；记忆库可持久化 | ITER-2025-03 | 🟡 部分完成 | 40% | self_training/pseudo_labeler.py, memory/memory_bank.py（未集成到训练） | 自训练轮次可配置，记忆检索命中率指标输出 |
| BIZ-005 | 可解释性与评估 | 生成违规证据链、关系注意力可视化、AUC/PR 等指标 | P2 | 报告模板统一；指标存档 | ITER-2025-03 | 🟡 部分完成 | 60% | explain/attention_viz.py, monitoring/metrics.py, monitoring/visualizer.py | 报告示例可供评审，指标脚本自动化执行 |
| BIZ-006 | 注意力增强 | 在 GNN / 记忆模块中引入多头注意力，暴露权重与可解释指标 | P0 | 注意力系数需与违规结论保持一致性；提供可视化接口 | ITER-2025-01 | ✅ 已完成 | 95% | models/（所有注意力模块）, loss/constraint.py | 生成 attention heatmap + 权重日志，QA 可复现 |

## 3. 系统需求拆解

### 3.1 功能需求
| 功能编号 | 关联业务需求 | 描述 | 前端影响 | 后端/服务影响 | 数据影响 | 实现状态 | 完成度 | 主要文件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYS-F-001 | BIZ-001 | 数据摄取与增强：支持 synthetic/真实数据切换、样本限制、标签归一 | CLI 配置界面（终端） | 实现 `TrafficLightDataset`、数据抽样策略、合成数据生成 | 存储实体特征、规则参数 | 🟡 部分完成 | 60% | traffic_dataset.py（synthetic完成）, prepare_synthetic_data.py |
| SYS-F-002 | BIZ-001 | 红灯规则推理：关系图构建、GAT 推理、规则损失 | 无 | 训练脚本、推理 API、规则阈值配置 | 记录违规日志、得分 | ✅ 已完成 | 95% | graph/builder.py, models/multi_stage_gat.py, rules/red_light.py, loss/constraint.py, tools/train_red_light.py |
| SYS-F-003 | BIZ-002 | 多数据集解析：BDD100K/Cityscapes 解析器、数据质量检测 | 无 | 数据转换服务、抽样任务 | 保存解析缓存、统计 | ❌ 未开始 | 0% | 待新增：data/parsers/ |
| SYS-F-004 | BIZ-003 | 规则注入框架：规则 DSL、冲突检测、热更新 | 可选可视化 | 规则管理模块、校验服务 | 规则元数据、历史版本 | 🟡 部分完成 | 15% | rules/red_light.py（仅单规则），待新增：rules/rule_manager.py |
| SYS-F-005 | BIZ-004 | 自训练管道：伪标签、课程学习调度、记忆库管理 | 无 | 训练 orchestrator、存储 | 伪标签存储、记忆 embedding | 🟡 部分完成 | 40% | self_training/pseudo_labeler.py, memory/memory_bank.py（未集成） |
| SYS-F-006 | BIZ-005 | 可解释性输出：注意力热力图、违规证据链、指标存档 | 可选报告界面 | 报告生成器、指标收集脚本 | 指标库、报告文件 | 🟡 部分完成 | 60% | explain/attention_viz.py, monitoring/metrics.py, scripts/render_attention_maps.py（占位） |
| SYS-F-007 | BIZ-006 | 注意力增强与可解释性：多头 GAT、记忆注意力、权重导出 | 无 | 注意力模块、可解释性 API、可视化脚本 | 指标/日志仓库存储 attention 数据 | ✅ 已完成 | 95% | models/local_gat.py, models/global_attention.py, models/rule_attention.py |

### 3.2 非功能需求
- 性能：单次训练轮在 4×A100 GPU 下 <= 2 小时；推理每帧延迟 <= 120ms（离线批处理场景）。
- 准确性：红灯违规检测召回率 ≥ 0.9（MVP synthetic 数据）；后续真实数据 AUC ≥ 0.85。
- 可扩展性：规则引擎需支持 ≥5 条规则；数据加载器可水平扩展到 1M 样本。
- 可维护性：配置项通过 Python 配置类管理，环境变量覆盖默认值；日志、指标具备统一命名。
- 可复现性：提供 `scripts/setup_mvp_env.sh`（待实现）用于环境还原。

## 4. 影响分析
- 系统/模块：数据摄取、训练调度、规则管理、可解释性服务均为新增模块。
- 第三方依赖：PyTorch、torchvision、OpenCV、BDD100K 官方解析库、Prometheus client。
- 风险与缓解：
  - 数据许可风险 → 仅使用公开数据并记录来源。
  - 规则冲突 → 在规则 DSL 中加入冲突检测、优先级字段。
  - 计算资源不足 → MVP 先运行 synthetic 数据，后续引入分布式训练计划。

## 5. 需求追踪矩阵
| 业务需求 | 系统需求 | 设计章节 | 开发任务 | 测试用例 | 部署检查 | 实现状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BIZ-001 | SYS-F-001/002 | DESIGN.md §3.1-3.4 | tools/train_red_light.py, tools/test_red_light.py | tests/unit/test_rule_scoring.py, 缺集成测试 | ✅ 脚本可运行 | 🟡 85%完成 | 缺三场景验收测试 |
| BIZ-002 | SYS-F-003 | 需详细设计 | 待新增BDD100K/Cityscapes解析器 | 待补 | 待补 | ❌ 未开始 | ITER-2 |
| BIZ-003 | SYS-F-004 | 需详细设计 | 待扩展规则框架 | 待补 | 待补 | 🟡 15%完成 | 仅有红灯单规则 |
| BIZ-004 | SYS-F-005 | 需详细设计 | 待集成自训练到Trainer | 待补 | 待补 | 🟡 40%完成 | 模块完成但未集成 |
| BIZ-005 | SYS-F-006 | DESIGN.md §3.8, 需补充 | 待完善报告生成 | 待补 | 待补 | 🟡 60%完成 | 缺自动化报告 |
| BIZ-006 | SYS-F-007 | DESIGN.md §3.3 | ✅ 已完成 | tests/unit/（部分） | ✅ 可验证 | ✅ 95%完成 | 仅缺权重持久化 |

## 6. 验收标准与测试入口

### 6.1 MVP验收命令（当前可执行）

#### 训练验收
```bash
# Smoke Test（快速验证）
python3 tools/train_red_light.py train --epochs 2 --max-samples 5 --device cpu

# 标准训练
python3 tools/train_red_light.py train --data-root data/synthetic --epochs 50 --device cpu
```

**预期输出**：
- `artifacts/checkpoints/best.pth` - 最佳模型
- `artifacts/checkpoints/checkpoint_epoch_*.pth` - 各epoch checkpoint
- `reports/training_curves.png` - 训练曲线（4子图：Loss/分量/指标/学习率）

#### 测试验收
```bash
python3 tools/test_red_light.py run \
  --checkpoint artifacts/checkpoints/best.pth \
  --data-root data/synthetic \
  --split val \
  --report-dir reports/testing
```

**预期输出**：
- `reports/testing/<scene_id>.json` - 每个场景的证据链（包含：距离、速度、灯态、模型分数、规则分数、注意力权重、违规判定）
- `reports/testing/summary.json` - 测试汇总

### 6.2 当前缺失的验收项（待补全）
- ❌ 三场景分类测试（parking/violation/green_pass），需要在test_red_light.py中添加`--scenario`参数支持
- ❌ 违规截图生成（需扩展test_red_light.py，调用attention_viz渲染）
- ❌ 注意力热力图批量生成（render_attention_maps.py需完善）

### 6.3 集成测试
- **当前状态**：`tests/integration/traffic_rules/test_cli.py` 仅为骨架
- **待补**：`pytest tests/integration -m traffic_rules` 覆盖所有业务约束

### 6.4 验收数据集
- ✅ 合成数据：100个场景（train 80 + val 20）已生成于 `data/synthetic/`
- ❌ BDD100K样本：计划10个样本（ITER-2025-02）
- ❌ Cityscapes样本：计划10个样本（ITER-2025-02）

### 6.5 环境还原脚本
- ✅ `scripts/prepare_synthetic_data.py` - 可生成合成数据
- 🟡 `scripts/render_attention_maps.py` - 占位脚本，待完善
- 🟡 `scripts/setup_mvp_env.sh` - 如已存在，需验证

## 7. 变更日志
| 日期 | 版本 | 变更内容 | 责任人 |
| --- | --- | --- | --- |
| 2025-11-30 | v0.1 | 首版，根据 user-stories 提炼业务与系统需求，规划 3 个迭代 | 产品经理（AI） |
| 2025-11-30 | v0.2 | 补充 Python 技术栈约束、注意力增强需求及验收指标，关联设计/测试文档 | 产品经理（AI） |
| 2025-12-16 | v0.3 | 添加实现状态列，标记各需求完成度；更新验收命令为实际CLI参数；关联CODE_IMPLEMENTATION_STATUS.md | 产品经理（AI） |
| 2025-12-16 | v0.4 | 技术栈版本统一：Python 3.12 + Conda，与 environment.yml 保持一致 | 产品经理（AI） |

## Checklist
- [x] 原始需求来源记录完备
- [x] 业务→系统需求映射清晰
- [x] 迭代计划包含 MVP 及后续扩展
- [x] 明确环境还原与验收命令
- [x] 需求实现状态标记完成（v0.3）
- [ ] 未完成需求的详细设计（进行中）
- [ ] 评审记录（待补）

