"""红灯停测试 CLI（MVP 验收版）。

目标：
- 使用 `data/synthetic/{train,val}` + `artifacts/checkpoints/*.pth` 进行离线验收
- 输出 `reports/testing/*.json`，包含“证据链”（距离、速度、灯态、模型分数、规则分数、注意力聚焦等）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer

# 添加项目根目录到 Python 路径（与 tools/train_red_light.py 保持一致）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.traffic_rules.data.traffic_dataset import TrafficLightDataset
from src.traffic_rules.graph.builder import GraphBuilder
from src.traffic_rules.models.multi_stage_gat import MultiStageAttentionGAT
from src.traffic_rules.rules.red_light import RedLightRuleEngine


app = typer.Typer(help="红灯停测试入口（MVP验收）", no_args_is_help=True)


def _get_light_probs(entities: list[Any]) -> torch.Tensor:
    """从实体列表推断交通灯状态概率（与 Trainer._get_light_probs 对齐）。"""
    lights = [e for e in entities if getattr(e, "type", None) == "light"]
    if not lights:
        return torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    light = lights[0]
    state = getattr(light, "light_state", None) or "green"
    conf = float(getattr(light, "confidence", 0.9))
    state_map = {"red": 0, "yellow": 1, "green": 2}
    idx = state_map.get(state, 2)

    probs = torch.zeros(1, 3, dtype=torch.float32)
    probs[0, idx] = conf
    remaining = 1.0 - probs[0, idx]
    for j in range(3):
        if j != idx:
            probs[0, j] = remaining / 2
    return probs


def _max_attention_to_rule_neighbors(
    alpha_gat: torch.Tensor,
    edge_index: torch.Tensor,
    entity_types: torch.Tensor,
    car_node_idx: int,
) -> float | None:
    """返回车辆节点对(灯/停止线)邻居的最大注意力；若不存在则返回 None。"""
    out_edges_mask = edge_index[0] == int(car_node_idx)
    if not bool(out_edges_mask.any()):
        return None

    neighbor_indices = edge_index[1, out_edges_mask]
    rule_neighbor_mask = (entity_types[neighbor_indices] == 1) | (entity_types[neighbor_indices] == 2)
    if not bool(rule_neighbor_mask.any()):
        return None

    rule_edge_indices = torch.where(out_edges_mask)[0][rule_neighbor_mask]
    return float(alpha_gat[rule_edge_indices].max().item())


@app.command()
def run(
    checkpoint: Path = typer.Option(
        Path("artifacts/checkpoints/best.pth"),
        "--checkpoint",
        help="训练产生的 checkpoint（包含 model_state_dict）",
    ),
    data_root: Path = typer.Option(
        Path("data/synthetic"),
        "--data-root",
        help="数据根目录（包含 train/val 子目录）",
    ),
    split: str = typer.Option(
        "val",
        "--split",
        help="数据集分割：train/val/test",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="推理设备：cpu/cuda",
    ),
    max_samples: int | None = typer.Option(
        None,
        "--max-samples",
        help="最大样本数（调试用）",
    ),
    threshold: float = typer.Option(
        0.7,
        "--threshold",
        help="违规判定阈值（final_score > threshold 视为违规）",
    ),
    report_dir: Path = typer.Option(
        Path("reports/testing"),
        "--report-dir",
        "-r",
        help="报告输出目录",
    ),
) -> None:
    """执行：加载数据→构建图→模型推理→规则评分→输出证据链 JSON。"""
    report_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")

    dataset = TrafficLightDataset(
        data_root=str(data_root),
        mode="synthetic",
        split=split,
        max_samples=max_samples,
    )

    # 模型
    model = MultiStageAttentionGAT(
        input_dim=10,
        hidden_dim=128,
        num_gat_layers=3,
        num_heads=8,
        dropout=0.1,
    )

    # PyTorch>=2.6 默认 weights_only=True，可能导致加载包含非权重对象的 dict 失败。
    # 该 checkpoint 由本项目训练脚本生成，属于“可信来源”，这里显式关闭 weights_only。
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容旧版本 torch（无 weights_only 参数）
        ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    builder = GraphBuilder()
    rule_engine = RedLightRuleEngine()

    all_scene_reports: list[dict[str, Any]] = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            scene = dataset[idx]
            graph = builder.build(scene)

            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            entity_types = graph.entity_types.to(device)

            if edge_index.size(1) == 0:
                continue

            output = model(x, edge_index, entity_types, return_attention=True)

            model_scores = output["scores"].detach().cpu()
            alpha_gat = output["gat_attention"].detach().cpu()
            beta_rule = output["rule_attention"].detach().cpu()

            car_entities = scene.get_entities_by_type("car")
            if not car_entities:
                continue

            light_probs = _get_light_probs(scene.entities)
            distances = torch.tensor([e.d_stop for e in car_entities], dtype=torch.float32)
            velocities = torch.tensor([e.velocity for e in car_entities], dtype=torch.float32)
            rule_scores = rule_engine.evaluate(light_probs, distances, velocities, training=False).detach().cpu()

            # 证据链：按“车辆”输出
            car_node_indices = torch.where(graph.entity_types == 0)[0].tolist()
            evidence: list[dict[str, Any]] = []

            for car_local_idx, car in enumerate(car_entities):
                m = float(model_scores[car_local_idx].item())
                r = float(rule_scores[car_local_idx].item())
                final_score = 0.6 * m + 0.4 * r

                # 车辆节点在图中的索引（用于注意力到规则邻居）
                car_node_idx = car_node_indices[car_local_idx] if car_local_idx < len(car_node_indices) else None
                max_attn = (
                    _max_attention_to_rule_neighbors(alpha_gat, graph.edge_index, graph.entity_types, car_node_idx)
                    if car_node_idx is not None
                    else None
                )

                violation = final_score > threshold
                evidence.append(
                    {
                        "entity_id": car.id,
                        "distance_to_stop": float(car.d_stop),
                        "velocity": float(car.velocity),
                        "light_state": next(
                            (e.light_state for e in scene.entities if e.type == "light" and e.light_state),
                            None,
                        ),
                        "model_score": m,
                        "rule_score": r,
                        "final_score": float(final_score),
                        "rule_focus_beta": float(beta_rule[car_local_idx].item()) if car_local_idx < len(beta_rule) else None,
                        "max_attention_to_rule_neighbors": max_attn,
                        "violation": bool(violation),
                    }
                )

            scene_report = {
                "scene_id": scene.scene_id,
                "timestamp": scene.timestamp,
                "num_entities": scene.num_entities,
                "num_cars": scene.num_cars,
                "threshold": threshold,
                "evidence": evidence,
                "summary": {
                    "violations_detected": int(sum(1 for e in evidence if e["violation"])),
                    "max_final_score": float(max((e["final_score"] for e in evidence), default=0.0)),
                },
            }
            all_scene_reports.append(scene_report)

            out_path = report_dir / f"{scene.scene_id}.json"
            out_path.write_text(json.dumps(scene_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 总结
    summary = {
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "split": split,
        "num_scenes_written": len(all_scene_reports),
        "report_dir": str(report_dir),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    typer.secho("=== 测试完成 ===", fg=typer.colors.CYAN)
    typer.echo(f"输出目录: {report_dir}")
    typer.echo(f"场景报告数: {summary['num_scenes_written']}")


if __name__ == "__main__":
    app()
