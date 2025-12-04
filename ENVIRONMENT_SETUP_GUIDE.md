# 开发环境配置指南

## 当前状态
- ✅ 已实现3个核心模块（规则评分、约束损失、单元测试）
- ⏳ 需要配置Python环境才能运行测试
- 📍 暂停点：等待环境配置完成

---

## 快速配置（推荐方案）

### 方案A：使用Poetry（推荐）

```bash
# 1. 进入项目目录
cd /Users/shiyifan/Documents/CursorWorkStation/lunwen

# 2. 使用Poetry安装依赖
poetry install

# 3. 激活虚拟环境
poetry shell

# 4. 验证安装
python --version  # 应显示Python 3.11+
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 5. 运行单元测试
python tests/unit/test_rule_scoring.py
```

### 方案B：使用pip（备选）

```bash
# 1. 进入项目目录
cd /Users/shiyifan/Documents/CursorWorkStation/lunwen

# 2. 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 3. 安装PyTorch（CUDA版本）
pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 4. 安装其他依赖（基础版）
pip3 install numpy scikit-learn pydantic pyyaml

# 5. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 6. 运行单元测试
python tests/unit/test_rule_scoring.py
```

### 方案C：仅CPU版本（最快，适合测试）

```bash
# 1. 安装CPU版本PyTorch
pip3 install torch torchvision torchaudio

# 2. 安装基础依赖
pip3 install numpy scikit-learn pydantic

# 3. 运行测试
python3 tests/unit/test_rule_scoring.py
```

---

## 详细配置步骤

### 步骤1：检查当前环境

```bash
# 检查Python版本
python3 --version
# 期望：Python 3.11+ 或 3.10+

# 检查是否已安装poetry
poetry --version
# 如果未安装：curl -sSL https://install.python-poetry.org | python3 -

# 检查CUDA版本（如果需要GPU）
nvcc --version
# 期望：CUDA 12.1+
```

### 步骤2：配置项目环境

#### 选项A：Poetry环境（推荐）

```bash
cd /Users/shiyifan/Documents/CursorWorkStation/lunwen

# Poetry配置
poetry config virtualenvs.in-project true  # 虚拟环境放在项目内

# 安装依赖
poetry install

# 如果pyproject.toml中依赖不完整，手动添加：
poetry add torch==2.4.1
poetry add numpy scikit-learn pydantic pyyaml
```

#### 选项B：venv环境

```bash
cd /Users/shiyifan/Documents/CursorWorkStation/lunwen

# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装PyTorch（根据需要选择CPU或CUDA版本）
# CUDA版本：
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# CPU版本：
pip install torch torchvision

# 安装其他依赖
pip install numpy scikit-learn pydantic pyyaml
```

### 步骤3：验证安装

```bash
# 测试PyTorch
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} 安装成功')"

# 测试CUDA（如果安装了GPU版本）
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 测试其他依赖
python3 -c "import numpy, sklearn, pydantic; print('✅ 所有基础依赖安装成功')"
```

### 步骤4：运行单元测试

```bash
# 进入项目目录
cd /Users/shiyifan/Documents/CursorWorkStation/lunwen

# 确保虚拟环境已激活（如果使用venv）
# source .venv/bin/activate

# 或使用poetry
# poetry shell

# 运行规则评分测试
python3 tests/unit/test_rule_scoring.py

# 预期输出：
# ======================================================================
# 规则评分函数单元测试
# 基于: Design-ITER-2025-01.md v2.0 §3.4.1
# ======================================================================
# 
# [1/5] 边界条件测试
# ----------------------------------------------------------------------
# ✅ 测试1通过: 完全停止 | 分数=0.0000
# ✅ 测试2通过: 接近但停止 | 分数=0.0670
# ✅ 测试3通过: 闯过停止线 | 分数=0.8950
# ✅ 测试4通过: 冲向红灯 | 分数=0.8820
# ✅ 测试5通过: 绿灯通过 | 分数=0.0450
# ✅ 测试6通过: 远离停止线 | 分数=0.0000
# ...
# ✅ 所有测试通过！规则评分函数实现正确。
```

---

## 常见问题排查

### Q1：ModuleNotFoundError: No module named 'torch'

**原因**：PyTorch未安装或虚拟环境未激活

**解决**：
```bash
# 检查当前Python环境
which python3

# 如果使用venv，确保已激活
source .venv/bin/activate

# 如果使用poetry
poetry shell

# 重新安装torch
pip3 install torch
```

### Q2：CUDA不可用（torch.cuda.is_available() = False）

**原因**：
- 安装了CPU版本的PyTorch
- CUDA驱动未安装
- CUDA版本不匹配

**解决**：
```bash
# 检查CUDA版本
nvcc --version

# 卸载CPU版本
pip3 uninstall torch torchvision torchaudio

# 重新安装CUDA版本
pip3 install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

### Q3：ImportError: cannot import name 'xxx'

**原因**：模块路径问题

**解决**：
```bash
# 方案1：设置PYTHONPATH
export PYTHONPATH=/Users/shiyifan/Documents/CursorWorkStation/lunwen:$PYTHONPATH

# 方案2：安装为开发包
cd /Users/shiyifan/Documents/CursorWorkStation/lunwen
pip3 install -e .
```

---

## 配置完成检查清单

完成配置后，请依次检查：

- [ ] Python 3.11+安装成功
- [ ] 虚拟环境创建并激活
- [ ] PyTorch安装成功（`import torch`无错误）
- [ ] CUDA可用（如果需要GPU训练）
- [ ] 基础依赖安装（numpy, sklearn, pydantic）
- [ ] 项目路径在PYTHONPATH中
- [ ] 单元测试可运行（`python3 tests/unit/test_rule_scoring.py`）
- [ ] 所有18个测试通过

---

## 配置完成后的操作

### 1. 验证已实现的代码

```bash
# 运行规则评分测试
python3 tests/unit/test_rule_scoring.py

# 预期：18个测试全部通过 ✅
```

### 2. 检查实施进度

```bash
# 查看当前进度
cat lunwen/docs/development/IMPLEMENTATION_TRACKER.md | grep "进度" -A 2

# 预期输出：
# 进度：3/13 模块已实现（~23%）
```

### 3. 继续实现

```bash
# 阅读下一步行动
cat lunwen/docs/development/IMPLEMENTATION_TRACKER.md | grep "下一步行动" -A 10

# 继续执行：
# - 告诉AI "继续实现GAT模型层"
# - 或 "继续实现数据加载器"
```

---

## 快速恢复命令（环境配置完成后）

```bash
# 一键恢复环境
cd /Users/shiyifan/Documents/CursorWorkStation/lunwen
source .venv/bin/activate  # 或 poetry shell

# 验证环境
python3 -c "import torch; print('✅ 环境就绪')"

# 运行测试
python3 tests/unit/test_rule_scoring.py

# 查看进度
cat docs/development/IMPLEMENTATION_TRACKER.md | grep "当前任务" -A 5
```

---

## 联系AI继续开发

环境配置完成后，向AI发送以下消息之一：

**消息1：环境配置完成，请继续**
```
环境已配置完成，测试通过。
[ENTER EXECUTE MODE]
继续实现GAT模型层
```

**消息2：环境配置完成，请验证**
```
环境已配置完成。
[ENTER EXECUTE MODE]
运行所有已实现模块的测试
```

**消息3：遇到环境问题**
```
[ENTER RESEARCH MODE]
环境配置遇到问题：[具体错误信息]
```

---

**配置状态**：⏳ 等待用户完成环境配置  
**实施状态**：🟡 暂停（已实现3/13模块）  
**下次继续**：环境就绪后继续实现剩余10个模块

**预估剩余工作量**：
- 代码实现：~2500行
- 测试代码：~1500行
- 预估时间：8-10天（环境就绪后）



