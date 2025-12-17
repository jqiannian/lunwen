# 测试文档（ITER-2025-01）

> 覆盖红灯停 MVP 的单测/集成/验收计划，基于实际代码实现更新（2025-12-16）。

## 元数据
| 字段 | 内容 |
| --- | --- |
| 文档版本 | v0.2 |
| 迭代编号 | ITER-2025-01 |
| QA 负责人 | 待指派 |
| 状态 | 🟡 测试中（部分测试可执行） |
| 最后更新时间 | 2025-12-16（更新实际测试命令与覆盖情况） |
| 关联需求 | `docs/iterations/ITER-2025-01/REQUIREMENT.md` |
| 关联开发 | `docs/iterations/ITER-2025-01/DEVELOPMENT.md` |
| 关联部署 | 待生成 |

## 1. 测试环境
- **Dev环境**：Conda环境（`environment-dev.yml`）+ synthetic数据
- **Test环境**：CI runner（待配置GPU/依赖）
- **数据准备**：✅ `scripts/prepare_synthetic_data.py` 已实现，已生成100个场景

## 2. 单元测试计划

### 2.1 已有测试文件
- ✅ `tests/unit/test_rule_scoring.py` - 红灯规则评分测试
- ✅ `tests/unit/test_placeholders.py` - 占位测试
- 🟡 `tests/integration/traffic_rules/test_cli.py` - CLI集成测试骨架

### 2.2 待补充测试
| 模块 | 测试文件 | 状态 | 备注 |
| --- | --- | --- | --- |
| TrafficLightDataset | tests/unit/test_dataset.py | ❌ 待补 | 数据加载、实体解析 |
| GraphBuilder | tests/unit/test_graph_builder.py | ❌ 待补 | 特征编码、边构建 |
| MultiStageGAT | tests/unit/test_multi_stage_gat.py | ❌ 待补 | 前向传播、注意力输出 |
| ConstraintLoss | tests/unit/test_constraint_loss.py | ✅ 已有（红灯规则内） | 损失计算、梯度流 |
| PseudoLabeler | tests/unit/test_pseudo_labeler.py | ❌ 待补 | 三策略生成 |

### 2.3 单元测试命令
```bash
# 运行所有单元测试
pytest tests/unit --cov=src/traffic_rules --cov-report=term-missing

# 运行特定模块测试
pytest tests/unit/test_rule_scoring.py -v
```

## 3. 前端 Selenium 测试
- MVP 无前端，跳过此章节。

## 4. 集成测试

### 4.1 三场景验收测试（核心）
| 场景类型 | 场景描述 | 预期结果 | 测试状态 |
| --- | --- | --- | --- |
| parking | 红灯停车（d>5m, v<0.5m/s） | 模型分数低，rule分数低，判定无违规 | ❌ 待实现 |
| violation | 红灯闯行（d<0或d<5且v>1m/s） | 模型分数高，rule分数高，判定违规 | ❌ 待实现 |
| green_pass | 绿灯通行 | 模型分数低，rule分数低，判定无违规 | ❌ 待实现 |

### 4.2 集成测试命令
```bash
# 当前可执行（输出所有场景JSON）
python3 tools/test_red_light.py run \
  --checkpoint artifacts/checkpoints/best.pth \
  --data-root data/synthetic \
  --split val \
  --report-dir reports/testing

# 待实现：三场景分类测试
python3 tools/test_red_light.py run \
  --checkpoint artifacts/checkpoints/best.pth \
  --scenario parking \  # 或 violation / green_pass
  --report-dir reports/testing
```

### 4.3 集成测试覆盖
- ✅ 端到端流程（数据→模型→规则→证据链）
- ❌ 三场景分类与验收标准对照
- ❌ 违规截图生成
- ❌ 注意力热力图批量生成

## 5. 验收测试

### 5.1 验收流程（实际可执行）
```bash
# Step 1: 准备数据（已完成）
python3 scripts/prepare_synthetic_data.py --num-scenes 100 --output-dir data/synthetic

# Step 2: 训练（Smoke Test）
python3 tools/train_red_light.py train --epochs 2 --max-samples 5 --device cpu

# Step 3: 训练（标准）
python3 tools/train_red_light.py train --data-root data/synthetic --epochs 50 --device cpu

# Step 4: 测试
python3 tools/test_red_light.py run \
  --checkpoint artifacts/checkpoints/best.pth \
  --data-root data/synthetic \
  --split val

# Step 5: 查看报告
ls reports/testing/*.json
cat reports/testing/summary.json
```

### 5.2 验收标准对照
| 验收项 | 要求 | 实际产出 | 状态 |
| --- | --- | --- | --- |
| CLI训练成功 | 运行无错误，生成checkpoint | `artifacts/checkpoints/best.pth` | ✅ 可执行 |
| 训练曲线 | Loss收敛，指标上升 | `reports/training_curves.png` | ✅ 可执行 |
| 测试三场景 | parking/violation/green_pass分类测试 | 当前仅输出统一JSON | ❌ 待实现 |
| 违规报告 | JSON格式证据链 | `reports/testing/<scene_id>.json` | ✅ 可执行 |
| 违规截图 | 带bbox和注意力标注的图片 | 待实现 | ❌ 待实现 |
| 注意力热力图 | 违规车辆的注意力可视化 | 待实现 | ❌ 待实现 |

## 6. 缺陷管理
- 使用缺陷表格模板（待建立）
- 缺陷跟踪系统：GitHub Issues（待启用）

## 7. 性能测试
- 训练耗时：待实际运行测量
- 推理延迟：待测试CLI统计
- 指标目标：见 REQUIREMENT.md §3.2（非功能需求）

## 8. 测试结论（待更新）
- **当前状态**：核心功能已可测试，缺少三场景验收与可视化输出
- **下一步**：补充三场景分类逻辑、违规截图生成、注意力热力图批量渲染

## Checklist
- [x] 单元测试计划定义（部分覆盖）
- [x] 集成测试流程可执行（基础版）
- [ ] 三场景验收测试实现
- [x] 环境还原脚本可用（`prepare_synthetic_data.py`）
- [ ] 完整验收报告生成
- [ ] QA 评审记录

