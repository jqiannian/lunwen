"""
注意力可视化模块

基于Design-ITER-2025-01.md v2.0 §3.8.1设计
生成注意力热力图和违规证据报告
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Optional, Dict
from pathlib import Path


def visualize_attention(
    image: np.ndarray,
    entities: List,
    attention_weights: torch.Tensor,
    focal_entity_idx: int,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    在原始图像上叠加注意力热力图
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.8.1
    
    Args:
        image: [H, W, 3] 原始图像（numpy array）
        entities: 实体列表
        attention_weights: [N] 注意力权重向量
        focal_entity_idx: 中心实体索引（通常是待检测车辆）
        save_path: 保存路径（可选）
    
    Returns:
        annotated_image: 带注释的图像
    """
    # 复制图像避免修改原图
    annotated = image.copy()
    
    # 1. 绘制所有实体bbox
    for i, entity in enumerate(entities):
        if entity.bbox is None:
            continue
        
        x1, y1, x2, y2 = entity.bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 根据注意力权重选择颜色
        alpha = attention_weights[i].item() if i < len(attention_weights) else 0.0
        color = get_color_by_attention(alpha)
        
        # 绘制bbox
        thickness = 3 if i == focal_entity_idx else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # 标注注意力权重
        if alpha > 0.05:
            cv2.putText(
                annotated,
                f'{alpha:.2f}',
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
    
    # 2. 绘制注意力连线（focal → 其他实体）
    focal_entity = entities[focal_entity_idx]
    if focal_entity.bbox is not None:
        fx1, fy1, fx2, fy2 = focal_entity.bbox
        focal_center = (int((fx1 + fx2) / 2), int((fy1 + fy2) / 2))
        
        for i, entity in enumerate(entities):
            if i == focal_entity_idx or entity.bbox is None:
                continue
            
            alpha = attention_weights[i].item() if i < len(attention_weights) else 0.0
            
            if alpha > 0.1:  # 仅显示显著连线
                ex1, ey1, ex2, ey2 = entity.bbox
                target_center = (int((ex1 + ex2) / 2), int((ey1 + ey2) / 2))
                
                # 线条粗细与注意力成正比
                thickness = max(1, int(alpha * 5))
                color = (255, 0, 0)  # 红色连线
                
                cv2.line(annotated, focal_center, target_center, color, thickness)
    
    # 3. 绘制信息面板
    focal = entities[focal_entity_idx]
    info_text = [
        f'Entity: {focal.id}',
        f'Type: {focal.type}',
        f'Velocity: {focal.velocity:.1f}m/s' if hasattr(focal, 'velocity') else 'N/A',
        f'Distance: {focal.d_stop:.1f}m' if hasattr(focal, 'd_stop') else 'N/A',
        f'Max Attn: {attention_weights.max():.3f}',
    ]
    
    # 背景面板
    panel_height = len(info_text) * 30 + 20
    cv2.rectangle(annotated, (10, 10), (300, panel_height), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (300, panel_height), (255, 255, 255), 2)
    
    # 文字
    for i, text in enumerate(info_text):
        cv2.putText(
            annotated,
            text,
            (20, 35 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    
    # 保存
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)
    
    return annotated


def get_color_by_attention(alpha: float) -> Tuple[int, int, int]:
    """
    根据注意力权重返回颜色（BGR格式）
    
    Args:
        alpha: 注意力权重 [0, 1]
    
    Returns:
        color: (B, G, R)
    """
    # 低注意力：蓝色 → 高注意力：红色
    if alpha < 0.1:
        return (255, 100, 0)  # 蓝色
    elif alpha < 0.3:
        return (255, 255, 0)  # 青色
    elif alpha < 0.5:
        return (0, 255, 255)  # 黄色
    elif alpha < 0.7:
        return (0, 165, 255)  # 橙色
    else:
        return (0, 0, 255)  # 红色


def generate_violation_report(
    scene_id: str,
    violations: List[Dict],
    save_path: str,
):
    """
    生成违规报告（JSON格式）
    
    设计依据：Design §3.6.1
    
    Args:
        scene_id: 场景ID
        violations: 违规列表
        save_path: 保存路径
    """
    report = {
        'scene_id': scene_id,
        'timestamp': str(pd.Timestamp.now()),
        'violations': violations,
        'summary': {
            'total_violations': len(violations),
            'avg_confidence': np.mean([v['final_score'] for v in violations]) if violations else 0.0,
        }
    }
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


# ============ 导出接口 ============
__all__ = [
    'visualize_attention',
    'get_color_by_attention',
    'generate_violation_report',
]
