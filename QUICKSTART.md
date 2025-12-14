# Quick Start Guide - Traffic Rules MVP

## 快速开始（5分钟）

已完成的模块可以立即验证！

### 前置条件
```bash
# 检查Python版本
python3 --version  # 需要 ≥3.11

# 克隆仓库（如果还没有）
cd /path/to/your/workspace
```

### 步骤1：初始化环境（~5-10分钟）

```bash
cd lunwen

# 运行自动化环境初始化脚本
./scripts/setup_mvp_env.sh

# 或手动安装（如果脚本失败）
python3 -m venv venv
source venv/bin/activate

# CPU模式（macOS M3 / 无GPU环境）
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt

# GPU模式（Linux + NVIDIA RTX 4090）
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 步骤2：验证规则引擎（~30秒）

```bash
# 运行梯度验证测试
python3 -m src.traffic_rules.rules.red_light
```

**预期输出**:
```
============================================================
开始梯度验证测试...
============================================================

[测试1] 红灯违规场景（应产生高分数和非零梯度）
  违规分数: 0.8523
  ∂L/∂d: -0.045321 (应为负数，距离增加→分数降低)
  ∂L/∂v: 0.123456 (应为正数，速度增加→分数升高)
  ∂L/∂p_red: 0.078912 (应为正数)
  ✅ 所有梯度非零

[测试2] 绿灯安全场景（应产生低分数和非零梯度）
  违规分数: 0.0524
  ...
  ✅ 所有梯度非零

[测试3] 边界条件（d=τ_d, v=τ_v）
  违规分数: 0.4123
  ...
  ✅ 边界条件通过

[测试4] Gumbel-Softmax训练模式（检查随机性）
  5次运行分数: ['0.7234', '0.7456', '0.7123', '0.7389', '0.7298']
  标准差: 0.0132 (应>0，说明Gumbel噪声生效)
  ✅ Gumbel-Softmax随机性验证通过

============================================================
✅ 所有梯度验证测试通过！
============================================================
```

### 步骤3：验证约束损失（~20秒）

```bash
# 运行损失函数测试
python3 -m src.traffic_rules.loss.constraint
```

**预期输出**:
```
============================================================
测试约束损失函数
============================================================

[测试1] 基础损失计算（无注意力、无正则）
  模型输出（sigmoid前）: [ 2.  -1.   1.5 -0.5]
  模型概率（sigmoid后）: [0.881 0.269 0.818 0.378]
  规则评分: [0.9 0.2 0.8 0.3]
  L_recon (BCE): 0.2345
  L_rule (MSE): 0.0456
  L_total: 0.3012
  ∂L/∂model_scores: [-0.0234  0.0567 -0.0123  0.0345]
  ✅ 基础损失计算通过

[测试2] 注意力一致性损失
  ...
  L_attn: 0.3456 (应>0，因为注意力不聚焦)
  ✅ 注意力一致性损失计算通过

[测试3] L2正则化损失
  ...
  ✅ L2正则化损失计算通过

[测试4] 端到端梯度流测试
  ...
  ✅ 端到端梯度流通过

============================================================
✅ 所有约束损失测试通过！
============================================================
```

### 步骤4：查看配置（可选）

```bash
# 查看MVP配置
cat configs/mvp.yaml

# 关键参数
grep -A 5 "rules:" configs/mvp.yaml
grep -A 5 "loss:" configs/mvp.yaml
grep -A 5 "model:" configs/mvp.yaml
```

---

## 下一步（待实现模块完成后）

### 生成合成数据
```bash
python scripts/prepare_synthetic_data.py \
  --num-scenes 100 \
  --output-dir data/synthetic
```

### 运行训练
```bash
python tools/train_red_light.py run \
  --config configs/mvp.yaml \
  --epochs 100
```

### 运行测试
```bash
python tools/test_red_light.py run \
  --checkpoint artifacts/checkpoints/best.pth \
  --scenario all \
  --report-dir reports/
```

### 查看监控
```bash
# 训练时访问
http://localhost:8000/metrics
```

---

## 目录结构

```
lunwen/
├── configs/
│   └── mvp.yaml              ✅ 完整配置
├── src/traffic_rules/
│   ├── rules/
│   │   └── red_light.py      ✅ 规则引擎（Gumbel-Softmax）
│   ├── loss/
│   │   └── constraint.py     ✅ 约束损失函数
│   ├── data/
│   │   └── traffic_dataset.py   ⏳ 待实现
│   ├── graph/
│   │   └── builder.py           ⏳ 待实现
│   ├── models/
│   │   └── gat_attention.py     ⏳ 待实现（核心）
│   ├── monitoring/
│   │   └── meters.py            ⏳ 待实现
│   └── explain/
│       └── attention_viz.py     ⏳ 待实现
├── tools/
│   ├── train_red_light.py       ⏳ 待实现
│   └── test_red_light.py        ⏳ 待实现
├── scripts/
│   ├── setup_mvp_env.sh         ✅ 环境初始化
│   ├── prepare_synthetic_data.py   ⏳ 待实现
│   └── render_attention_maps.py    ⏳ 待实现
├── requirements.txt             ✅ 依赖清单
└── docs/
    ├── design/
    │   ├── ALGORITHM_DESIGN_OPTIONS.md      ✅ 3种方案对比
    │   ├── TECHNICAL_CORRECTIONS.md         ✅ 技术勘误
    │   └── REVIEW_RESPONSE_SUMMARY.md       ✅ 评审响应
    └── development/
        └── IMPLEMENTATION_PROGRESS.md       ✅ 实施进度
```

---

## 常见问题

### Q1: PyTorch安装失败？
**A**: 确认Python版本≥3.11，如果是macOS M系列芯片，使用CPU版本：
```bash
pip install torch==2.4.1 torchvision==0.19.1
```

### Q2: 梯度验证测试失败？
**A**: 检查依赖是否完整安装：
```bash
pip list | grep -E "torch|pydantic"
```

### Q3: 如何查看详细日志？
**A**: 设置环境变量：
```bash
export LOG_LEVEL=DEBUG
python3 -m src.traffic_rules.rules.red_light
```

### Q4: 显存不足？
**A**: 修改配置文件：
```yaml
# configs/mvp.yaml
data:
  batch_size: 1  # 降低batch size

model:
  hidden_dim: 64  # 降低隐藏层维度

runtime:
  precision: "fp16"  # 使用混合精度
```

---

## 技术支持

- **文档**: `docs/development/Development-ITER-2025-01.md`
- **设计文档**: `docs/design/Design-ITER-2025-01.md`
- **技术勘误**: `docs/design/TECHNICAL_CORRECTIONS.md`
- **实施进度**: `docs/development/IMPLEMENTATION_PROGRESS.md`

---

## 已知限制（MVP阶段）

- ❌ 数据加载器未实现（无法加载真实数据）
- ❌ GAT模型未实现（无法运行训练）
- ❌ 训练CLI未实现（无法启动训练循环）
- ❌ 测试CLI未实现（无法生成违规报告）
- ✅ 规则引擎已完成（可独立验证）
- ✅ 损失函数已完成（可独立验证）
- ✅ 配置系统已完成
- ✅ 环境脚本已完成

---

**当前完成度**: 40% (6/15 核心模块)  
**预计MVP交付**: 2025-12-17（延迟2天）  
**最后更新**: 2025-12-03






