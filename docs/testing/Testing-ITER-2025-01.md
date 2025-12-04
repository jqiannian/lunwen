# 测试文档（ITER-2025-01）

> 由 `docs/templates/TEST_TEMPLATE.md` 派生，覆盖红灯停 MVP 的单测/集成/验收计划，目前为占位版本，QA 评审后补充细节。

## 元数据
| 字段 | 内容 |
| --- | --- |
| 文档版本 | v0.1 |
| 迭代编号 | ITER-2025-01 |
| QA 负责人 | 待指派 |
| 状态 | 草稿 |
| 最后更新时间 | 2025-11-30 |
| 关联需求 | `docs/requirement/iterations/Requirement-ITER-2025-01.md` |
| 关联开发 | `docs/development/Development-ITER-2025-01.md` |
| 关联部署 | 待生成 |

## 1. 测试环境（占位）
- Dev：poetry 环境 + synthetic 数据脚本。
- Test：CI runner，待补充 GPU/依赖。
- 数据准备脚本：`scripts/prepare_synthetic_data.py`（计划中）。

## 2. 单元测试计划
- 模块：`TrafficLightDataset`、`RedLightViolationDetector`、规则损失函数。
- 命令：`poetry run pytest tests/unit --cov=src`.

## 3. 前端 Selenium 测试
- MVP 无前端，仅保留此章节作为模板，待未来可视化界面时补齐。

## 4. 集成测试
- 场景：红灯停车、红灯闯行、绿灯通行。
- 命令：`poetry run python tools/test_red_light.py --scenario all`.

## 5. 验收测试
- 主线流程：准备数据 → 训练 → 测试 → 生成违规报告。
- 验收人：业务负责人（待指派）。

## 6. 缺陷管理
- 使用缺陷表格模板，链接到 `tests/report/`（待建立）。

## 7. 性能 / 压力 / 流量测试
- MVP 阶段仅记录计划：训练耗时监控、推理延迟统计，执行需在负责人确认下进行。

## 8. 测试结论（待执行）
- 将在测试完成后更新风险评估与放行意见。

## Checklist
- [ ] 单元测试覆盖所有核心方法
- [ ] 集成测试覆盖红灯规则场景
- [ ] 验收流程定义完成
- [ ] 环境还原脚本可用并记录
- [ ] QA 评审记录

