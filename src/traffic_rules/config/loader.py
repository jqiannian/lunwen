"""配置加载器：负责解析 YAML + 环境变量，生成强类型配置模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    """运行时相关参数。"""

    seed: int = Field(default=42, description="随机种子")
    device: str = Field(default="cuda:0", description="训练/推理设备标识")
    precision: str = Field(default="fp32", description="计算精度")


class PathsConfig(BaseModel):
    data_root: str = Field(default="${DATA_ROOT}")
    artifact_root: str = Field(default="${ARTIFACT_ROOT}")
    checkpoint_dir: str = Field(default="artifacts/checkpoints")
    pseudo_label_dir: str = Field(default="artifacts/pseudo_labels")
    report_dir: str = Field(default="reports")


class RuleConfig(BaseModel):
    """单条交通规则配置。"""

    state: str = Field(default="red", description="信号灯状态")
    distance_threshold: float = Field(default=50.0, description="停止线距离阈值")
    speed_threshold: float = Field(default=0.5, description="速度阈值")
    enforce_attention_consistency: bool = Field(
        default=True, description="是否启用注意力一致性"
    )
    enabled: bool = Field(default=True, description="是否启用该规则")


class MonitoringConfig(BaseModel):
    prometheus_port: int = 8000
    log_level: str = "INFO"


class ProjectConfig(BaseModel):
    """聚合全部子配置，供 CLI 与服务使用。"""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    rules: dict[str, RuleConfig] = Field(default_factory=dict)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


def load_config(path: Path) -> ProjectConfig:
    """从 YAML 文件中加载配置，生成 ProjectConfig。"""

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    data: dict[str, Any]
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}

    return ProjectConfig(**data)
