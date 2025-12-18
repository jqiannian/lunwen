#!/usr/bin/env python3
"""
批量注意力热力图渲染脚本

功能：
1. 读取test_red_light.py输出的JSON证据链
2. 为每个场景的每辆车生成注意力热力图
3. 生成HTML索引页供浏览
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.traffic_rules.data.traffic_dataset import TrafficLightDataset
from src.traffic_rules.graph.builder import GraphBuilder
from src.traffic_rules.models.multi_stage_gat import MultiStageAttentionGAT


def render_attention_heatmap(
    scene: Any,
    graph: Any,
    attention_weights: torch.Tensor,
    car_idx: int,
    save_path: Path,
    attention_type: str = "GAT",
) -> None:
    """
    为单辆车生成注意力热力图
    
    Args:
        scene: 场景数据
        graph: 场景图
        attention_weights: 注意力权重（边或节点）
        car_idx: 车辆索引
        save_path: 保存路径
        attention_type: 注意力类型（GAT/Rule/Global）
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{attention_type} Attention - Car {car_idx}', fontsize=14, fontweight='bold')
    
    # 绘制停止线
    for entity in scene.entities:
        if entity.type == "stop":
            x1, y1 = entity.pos
            x2, y2 = getattr(entity, 'end_pos', entity.pos)
            ax.plot([x1, x2], [y1, y2], 'b--', linewidth=2, alpha=0.5)
    
    # 绘制交通灯
    for entity in scene.entities:
        if entity.type == "light":
            cx, cy = entity.pos
            light_state = getattr(entity, 'light_state', 'green')
            color_map = {'red': 'red', 'yellow': 'yellow', 'green': 'green'}
            light_color = color_map.get(light_state, 'gray')
            circle = patches.Circle((cx, cy), 3, color=light_color, ec='black', linewidth=1)
            ax.add_patch(circle)
    
    # 获取车辆实体
    car_entities = scene.get_entities_by_type("car")
    if car_idx >= len(car_entities):
        print(f"Warning: car_idx {car_idx} out of range")
        plt.close(fig)
        return
    
    focal_car = car_entities[car_idx]
    
    # 绘制所有车辆（根据注意力权重着色）
    for i, entity in enumerate(car_entities):
        cx, cy = entity.pos
        car_width, car_height = 4, 8
        
        # 获取该车辆的注意力权重
        if attention_type == "GAT" and i < len(attention_weights):
            alpha = float(attention_weights[i].item())
        elif attention_type in ["Rule", "Global"] and i < len(attention_weights):
            alpha = float(attention_weights[i].item())
        else:
            alpha = 0.0
        
        # 根据注意力权重选择颜色
        if i == car_idx:
            # 焦点车辆（红色高亮）
            edgecolor = 'red'
            facecolor = (1.0, 0.0, 0.0, 0.2)  # 半透明红色
            linewidth = 4
        else:
            # 其他车辆（根据注意力权重着色）
            cmap = plt.cm.get_cmap('YlOrRd')
            edgecolor = cmap(alpha)
            facecolor = 'none'
            linewidth = 2
        
        rect = patches.Rectangle(
            (cx - car_width / 2, cy - car_height / 2),
            car_width, car_height,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor
        )
        ax.add_patch(rect)
        
        # 标注注意力权重
        if alpha > 0.05 and i != car_idx:
            ax.text(cx, cy - car_height / 2 - 1, f'{alpha:.2f}',
                   ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')
    
    # 绘制注意力连线（从焦点车辆到其他实体）
    fx, fy = focal_car.pos
    for entity in scene.entities:
        if entity.type in ["light", "stop"]:
            tx, ty = entity.pos
            # 简化：假设注意力权重与实体距离相关
            dist = np.sqrt((tx - fx)**2 + (ty - fy)**2)
            alpha = max(0, 1 - dist / 50)  # 距离越近，注意力越高
            
            if alpha > 0.1:
                ax.plot([fx, tx], [fy, ty], 'r-', alpha=alpha, linewidth=2)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Attention Weight')
    
    # 保存
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_html_index(heatmaps_dir: Path, output_path: Path) -> None:
    """
    生成HTML索引页
    
    Args:
        heatmaps_dir: 热力图目录
        output_path: HTML输出路径
    """
    # 收集所有热力图文件
    heatmap_files = sorted(heatmaps_dir.glob("*.png"))
    
    # 按场景分组
    scenes = {}
    for f in heatmap_files:
        # 文件名格式：scene_0001_car_0_GAT.png
        parts = f.stem.split('_')
        if len(parts) >= 4:
            scene_id = f"{parts[0]}_{parts[1]}"
            if scene_id not in scenes:
                scenes[scene_id] = []
            scenes[scene_id].append(f.name)
    
    # 生成HTML
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>注意力热力图索引</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        .scene-group {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .scene-group h2 {
            color: #007bff;
            margin-top: 0;
        }
        .heatmap-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        .heatmap-item {
            text-align: center;
        }
        .heatmap-item img {
            width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .heatmap-item img:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .heatmap-item p {
            margin: 10px 0 0 0;
            color: #666;
            font-size: 14px;
        }
        .stats {
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .stats p {
            margin: 5px 0;
            color: #495057;
        }
    </style>
</head>
<body>
    <h1>🔍 注意力热力图索引</h1>
    
    <div class="stats">
        <p><strong>总场景数</strong>: """ + str(len(scenes)) + """</p>
        <p><strong>总热力图数</strong>: """ + str(len(heatmap_files)) + """</p>
        <p><strong>生成时间</strong>: """ + str(output_path.stat().st_mtime if output_path.exists() else "N/A") + """</p>
    </div>
"""
    
    # 为每个场景生成一个section
    for scene_id, files in sorted(scenes.items()):
        html += f"""
    <div class="scene-group">
        <h2>{scene_id}</h2>
        <div class="heatmap-grid">
"""
        for filename in sorted(files):
            html += f"""
            <div class="heatmap-item">
                <img src="{filename}" alt="{filename}" onclick="window.open('{filename}', '_blank')">
                <p>{filename}</p>
            </div>
"""
        html += """
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    # 保存HTML
    output_path.write_text(html, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description="批量渲染注意力热力图")
    parser.add_argument(
        "--evidence-dir",
        default=Path("reports/testing"),
        type=Path,
        help="证据链JSON目录（test_red_light.py输出）",
    )
    parser.add_argument(
        "--checkpoint",
        default=Path("artifacts/checkpoints/best.pth"),
        type=Path,
        help="模型checkpoint路径",
    )
    parser.add_argument(
        "--data-root",
        default=Path("data/synthetic"),
        type=Path,
        help="数据根目录",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("reports/testing/heatmaps"),
        type=Path,
        help="热力图输出目录",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="数据集分割（train/val）",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="推理设备（cpu/cuda）",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("批量注意力热力图渲染")
    print("=" * 60)
    
    # 检查checkpoint
    if not args.checkpoint.exists():
        print(f"错误：checkpoint不存在：{args.checkpoint}")
        return
    
    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model = MultiStageAttentionGAT(
        input_dim=10,
        hidden_dim=128,
        num_gat_layers=3,
        num_heads=8,
        dropout=0.1,
    )
    
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)
    model.eval()
    
    # 加载数据集
    print(f"加载数据集: {args.data_root}")
    dataset = TrafficLightDataset(
        data_root=str(args.data_root),
        mode="synthetic",
        split=args.split,
    )
    
    builder = GraphBuilder()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 渲染每个场景
    rendered_count = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            scene = dataset[idx]
            graph = builder.build(scene)
            
            x = graph.x.to(args.device)
            edge_index = graph.edge_index.to(args.device)
            entity_types = graph.entity_types.to(args.device)
            
            if edge_index.size(1) == 0:
                continue
            
            # 前向推理获取注意力权重
            output = model(x, edge_index, entity_types, return_attention=True)
            
            alpha_gat = output["gat_attention"].detach().cpu()
            beta_rule = output["rule_attention"].detach().cpu()
            
            # 为每辆车生成热力图
            car_entities = scene.get_entities_by_type("car")
            for car_idx in range(len(car_entities)):
                # GAT注意力热力图
                gat_path = args.output_dir / f"{scene.scene_id}_car_{car_idx}_GAT.png"
                render_attention_heatmap(scene, graph, alpha_gat, car_idx, gat_path, "GAT")
                
                # 规则注意力热力图
                rule_path = args.output_dir / f"{scene.scene_id}_car_{car_idx}_Rule.png"
                render_attention_heatmap(scene, graph, beta_rule, car_idx, rule_path, "Rule")
                
                rendered_count += 2
            
            if (idx + 1) % 5 == 0:
                print(f"已处理 {idx + 1}/{len(dataset)} 场景...")
    
    print(f"✅ 共生成 {rendered_count} 个热力图")
    
    # 生成HTML索引
    html_path = args.output_dir / "index.html"
    print(f"生成HTML索引: {html_path}")
    generate_html_index(args.output_dir, html_path)
    
    print("=" * 60)
    print(f"✅ 完成！热力图保存在: {args.output_dir}")
    print(f"✅ 浏览索引页: {html_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
