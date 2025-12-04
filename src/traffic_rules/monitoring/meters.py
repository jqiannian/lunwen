"""监控指标与日志骨架。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import structlog
from prometheus_client import Counter, Gauge, Summary, start_http_server


@dataclass(slots=True)
class MetricHandles:
    loss: Summary
    violations: Counter
    attention_consistency: Gauge


LOGGER = structlog.get_logger("traffic_rules")


def init_metrics(namespace: str = "traffic_rules") -> MetricHandles:
    """注册 Prometheus 指标并返回句柄。"""

    loss = Summary(f"{namespace}_loss", "训练期间的总 loss")
    violations = Counter(f"{namespace}_violations", "违规数量")
    attention = Gauge(
        f"{namespace}_attention_consistency",
        "注意力与规则一致性的度量",
    )
    return MetricHandles(loss=loss, violations=violations, attention_consistency=attention)


def start_metrics_server(port: int) -> None:
    """启动 Prometheus HTTP server, 便于 CLI 快速复用。"""

    start_http_server(port)
    LOGGER.info("metrics_server_started", port=port)


def log_event(event: str, *, trace_id: str | None = None, scene_id: str | None = None, **kwargs: Any) -> None:
    """统一结构化日志入口, 默认补全 trace_id/scene_id."""

    payload = {
        "trace_id": trace_id or str(uuid4()),
        "scene_id": scene_id or "unknown",
        **kwargs,
    }
    LOGGER.info(event, **payload)


def emit_metrics(handles: MetricHandles, payload: dict[str, float], labels: dict[str, str] | None = None) -> None:
    """根据 payload 更新指标, 并可记录标签信息。"""

    if not payload:
        return
    if "loss" in payload:
        handles.loss.observe(payload["loss"])
    if "violations" in payload:
        handles.violations.inc(payload["violations"])
    if "attention_consistency" in payload:
        handles.attention_consistency.set(payload["attention_consistency"])

    if labels:
        LOGGER.info("metrics_labels", labels=labels, payload=payload)
