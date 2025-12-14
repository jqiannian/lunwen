# Conda（conda-forge）安装与依赖管理

本项目从现在开始以 **conda-forge** 作为依赖管理的唯一入口（Single Source of Truth）：

- 运行环境：`environment.yml`
- 开发环境：`environment-dev.yml`

> 说明：仓库中仍保留 `requirements.txt` / `pyproject.toml` / `poetry.lock` 作为历史记录与工具配置，但**不再作为安装入口**。

## 1. 环境创建

在仓库根目录执行：

### 1.1 运行环境（推荐给只训练/测试的用户）

```bash
conda env create -f environment.yml
conda activate traffic-rules
```

### 1.2 开发环境（推荐给需要改代码/跑测试的人）

```bash
conda env create -f environment-dev.yml
conda activate traffic-rules-dev
```

## 2. 环境更新（版本变更后）

```bash
conda env update -f environment.yml --prune
# 或
conda env update -f environment-dev.yml --prune
```

`--prune` 会把不再需要的包清掉，避免环境越装越乱。

## 3. 环境锁定（可复现快照）

当你确认当前环境可用时，导出快照：

```bash
conda env export --no-builds > environment.lock.yml
```

> `environment.lock.yml` 用于“完全复现”，一般由发布/交付时生成。

## 4. 如何新增依赖（禁止随手 pip install）

- **优先**：把包加到 `environment.yml`（或 `environment-dev.yml`）的 conda 依赖里。
- **仅在 conda-forge 找不到**时，才加入 `pip:` 段。

新增后必须：

```bash
conda env update -f environment.yml --prune
```

## 5. 验证安装

```bash
python -c "import torch; print('torch', torch.__version__)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import pandas; print('pandas', pandas.__version__)"
python -c "import cv2; print('opencv', cv2.__version__)"
python -c "import structlog; print('structlog', structlog.__version__)"
```

## 6. 常见问题

### 6.1 安装很慢/失败
- 先确认你使用的是 miniforge/mambaforge（你当前环境符合）。
- 可选：用 `mamba` 替代 `conda`（更快）。

### 6.2 PyTorch 版本与代码不一致
- 以 `environment.yml` 为准。
- 如果你之前在同一个环境里用 `pip install torch` 安装过更高版本，建议删除环境重建。

### 6.3 macOS 上出现 OpenMP 冲突（libomp / OMP Error #15）
有些组合（例如 PyTorch + OpenCV + OpenMP runtime）会在 macOS 上触发类似报错：

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

临时绕过方式（仅用于本地开发/调试）：在运行命令前加环境变量：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

然后再运行训练/测试命令。
