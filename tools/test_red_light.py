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

import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


class ScenarioClassifier:
    """场景分类器：从metadata中读取场景类型"""
    
    @staticmethod
    def get_scenario_type(scene_data: Any) -> str:
        """
        从场景数据中提取场景类型
        
        Args:
            scene_data: SceneContext对象
        
        Returns:
            scenario_type: 'parking' | 'violation' | 'green_pass' | 'unknown'
        """
        # 从原始数据中提取metadata（如果存在）
        metadata = getattr(scene_data, 'metadata', {})
        scenario = metadata.get('scenario', 'unknown')
        return scenario
    
    @staticmethod
    def should_include_scene(scenario_type: str, filter_scenario: str) -> bool:
        """
        判断是否应包含该场景
        
        Args:
            scenario_type: 场景实际类型
            filter_scenario: 过滤条件（'all' 或特定类型）
        
        Returns:
            bool: 是否包含
        """
        if filter_scenario == "all":
            return True
        return scenario_type == filter_scenario


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


def _generate_scene_screenshot(
    scene: Any,
    evidence: list[dict[str, Any]],
    save_path: Path,
    threshold: float = 0.7,
) -> None:
    """
    生成场景截图（虚拟可视化）使用matplotlib
    
    在空白画布上绘制实体位置和违规标注
    
    Args:
        scene: 场景数据
        evidence: 证据链列表
        save_path: 保存路径
        threshold: 违规判定阈值
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.invert_yaxis()  # y轴向下
    ax.set_aspect('equal')
    
    # 设置背景
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_title(f'Scene: {scene.scene_id}', fontsize=14, fontweight='bold')
    
    # 绘制停止线
    for entity in scene.entities:
        if entity.type == "stop":
            x1, y1 = entity.pos
            x2, y2 = getattr(entity, 'end_pos', entity.pos)
            ax.plot([x1, x2], [y1, y2], 'b--', linewidth=3, label='Stop Line')
            ax.text((x1 + x2) / 2, y1 - 5, 'STOP LINE', 
                   ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
    
    # 绘制交通灯
    for entity in scene.entities:
        if entity.type == "light":
            cx, cy = entity.pos
            light_state = getattr(entity, 'light_state', 'green')
            
            color_map = {
                'red': 'red',
                'yellow': 'yellow',
                'green': 'green',
            }
            light_color = color_map.get(light_state, 'gray')
            
            circle = patches.Circle((cx, cy), 3, color=light_color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(cx, cy + 6, light_state.upper(), ha='center', va='top', 
                   fontsize=8, color='black', fontweight='bold')
    
    # 绘制车辆
    for entity in scene.entities:
        if entity.type == "car":
            car_evidence = next((e for e in evidence if e["entity_id"] == entity.id), None)
            if car_evidence is None:
                continue
            
            is_violation = car_evidence["violation"]
            final_score = car_evidence["final_score"]
            
            cx, cy = entity.pos
            car_width, car_height = 4, 8
            
            # 车辆矩形
            color = 'red' if is_violation else 'green'
            linewidth = 3 if is_violation else 2
            
            rect = patches.Rectangle(
                (cx - car_width / 2, cy - car_height / 2),
                car_width, car_height,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # 标注信息
            info_text = f"Car {entity.id}\n"
            if is_violation:
                info_text += "VIOLATION!\n"
            info_text += f"Score: {final_score:.2f}\n"
            info_text += f"v={car_evidence['velocity']:.1f}m/s\n"
            info_text += f"d={car_evidence['distance_to_stop']:.1f}m"
            
            text_color = 'red' if is_violation else 'darkgreen'
            ax.text(cx + car_width / 2 + 2, cy, info_text,
                   ha='left', va='center', fontsize=7, color=text_color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label='Violation'),
        Line2D([0], [0], color='green', linewidth=2, label='Normal'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=3, label='Stop Line'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # 保存
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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
    scenario: str = typer.Option(
        "all",
        "--scenario",
        help="场景类型过滤：all/parking/violation/green_pass",
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
    classifier = ScenarioClassifier()

    all_scene_reports: list[dict[str, Any]] = []
    scenario_stats: dict[str, dict[str, Any]] = {
        "parking": {"total": 0, "detected_violations": 0, "total_score": 0.0},
        "violation": {"total": 0, "detected_violations": 0, "total_score": 0.0},
        "green_pass": {"total": 0, "detected_violations": 0, "total_score": 0.0},
        "unknown": {"total": 0, "detected_violations": 0, "total_score": 0.0},
    }

    with torch.no_grad():
        for idx in range(len(dataset)):
            scene = dataset[idx]
            
            # 场景分类过滤
            scenario_type = classifier.get_scenario_type(scene)
            if not classifier.should_include_scene(scenario_type, scenario):
                continue
            
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

            violations_in_scene = int(sum(1 for e in evidence if e["violation"]))
            max_score_in_scene = float(max((e["final_score"] for e in evidence), default=0.0))
            
            scene_report = {
                "scene_id": scene.scene_id,
                "scenario_type": scenario_type,
                "timestamp": scene.timestamp,
                "num_entities": scene.num_entities,
                "num_cars": scene.num_cars,
                "threshold": threshold,
                "evidence": evidence,
                "summary": {
                    "violations_detected": violations_in_scene,
                    "max_final_score": max_score_in_scene,
                },
            }
            all_scene_reports.append(scene_report)
            
            # 更新场景类型统计
            if scenario_type in scenario_stats:
                scenario_stats[scenario_type]["total"] += 1
                scenario_stats[scenario_type]["detected_violations"] += violations_in_scene
                scenario_stats[scenario_type]["total_score"] += max_score_in_scene

            # 保存JSON报告
            out_path = report_dir / f"{scene.scene_id}.json"
            out_path.write_text(json.dumps(scene_report, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # 生成违规截图（仅当有违规时）
            if violations_in_scene > 0:
                screenshot_dir = report_dir / "screenshots"
                screenshot_path = screenshot_dir / f"{scene.scene_id}_violation.png"
                _generate_scene_screenshot(scene, evidence, screenshot_path, threshold)

    # 计算每个场景类型的指标
    scenario_summary = {}
    for sc_type, stats in scenario_stats.items():
        if stats["total"] > 0:
            avg_score = stats["total_score"] / stats["total"]
            
            # 计算准确率（根据场景类型期望）
            if sc_type == "violation":
                # 违规场景：期望检测到违规
                accuracy = stats["detected_violations"] / stats["total"] if stats["total"] > 0 else 0.0
                recall = stats["detected_violations"] / stats["total"] if stats["total"] > 0 else 0.0
            elif sc_type in ["parking", "green_pass"]:
                # 正常场景：期望不检测到违规
                true_negatives = stats["total"] - stats["detected_violations"]
                accuracy = true_negatives / stats["total"] if stats["total"] > 0 else 0.0
                recall = None  # 正常场景不计算召回率
            else:
                accuracy = None
                recall = None
            
            scenario_summary[sc_type] = {
                "total_scenes": stats["total"],
                "violations_detected": stats["detected_violations"],
                "avg_max_score": round(avg_score, 4),
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
                "recall": round(recall, 4) if recall is not None else None,
            }
    
    # 总结
    summary = {
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "split": split,
        "scenario_filter": scenario,
        "num_scenes_written": len(all_scene_reports),
        "report_dir": str(report_dir),
        "scenario_summary": scenario_summary,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    
    # 生成场景分类详细报告
    (report_dir / "scenario_summary.json").write_text(
        json.dumps(scenario_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    typer.secho("=== 测试完成 ===", fg=typer.colors.CYAN)
    typer.echo(f"输出目录: {report_dir}")
    typer.echo(f"场景过滤: {scenario}")
    typer.echo(f"场景报告数: {summary['num_scenes_written']}")
    typer.echo(f"\n场景分类统计:")
    for sc_type, stats in scenario_summary.items():
        typer.echo(f"  {sc_type}: {stats['total_scenes']}个场景, "
                   f"{stats['violations_detected']}个违规检出, "
                   f"准确率={stats['accuracy']}")


if __name__ == "__main__":
    app()
