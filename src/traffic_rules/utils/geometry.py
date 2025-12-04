"""
几何计算工具函数

设计依据：Design-ITER-2025-01.md v2.0 §3.1.2
"""

import numpy as np
from typing import Tuple


def compute_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    计算两点之间的欧氏距离
    
    Args:
        pos1: 点1坐标 [x, y]
        pos2: 点2坐标 [x, y]
    
    Returns:
        distance: 欧氏距离（米）
    """
    return np.linalg.norm(pos1 - pos2)


def point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> Tuple[float, bool]:
    """
    计算点到线段的垂直距离（向量投影法）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.1.2
    
    公式：
        d = |  (p - s1) × (s2 - s1)  | / |s2 - s1|
    
    符号约定：
        - d > 0: 点在线段前方
        - d < 0: 点已过线段（闯过停止线）
    
    Args:
        point: 点坐标 [x, y]
        line_start: 线段起点 [x1, y1]
        line_end: 线段终点 [x2, y2]
    
    Returns:
        distance: 垂直距离（米），带符号
        crossed: 是否已过线（True表示已过）
    """
    # 向量计算
    p = point
    s1 = line_start
    s2 = line_end
    
    # 线段方向向量
    line_vec = s2 - s1
    line_length = np.linalg.norm(line_vec)
    
    if line_length < 1e-6:
        # 退化为点
        return compute_distance(point, line_start), False
    
    # 归一化方向向量
    line_dir = line_vec / line_length
    
    # 点到线段起点的向量
    point_vec = p - s1
    
    # 投影到线段方向
    projection = np.dot(point_vec, line_dir)
    
    # 垂直向量（点到线段的最近点）
    if projection < 0:
        # 最近点是起点
        closest_point = s1
    elif projection > line_length:
        # 最近点是终点
        closest_point = s2
    else:
        # 最近点在线段上
        closest_point = s1 + projection * line_dir
    
    # 计算垂直距离
    perpendicular_vec = p - closest_point
    distance = np.linalg.norm(perpendicular_vec)
    
    # 判断是否过线（使用叉积判断在线段的哪一侧）
    # 2D叉积：cross_z = (s2-s1) × (p-s1)
    cross_z = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    
    # 如果点在线段"前方"（假设停止线从左到右，车辆从下向上）
    # 需要根据实际坐标系调整符号约定
    # 这里假设：cross_z > 0 表示点在线段前方，< 0 表示已过线
    crossed = (cross_z < 0)
    
    # 返回带符号的距离
    signed_distance = distance if not crossed else -distance
    
    return signed_distance, crossed


def compute_stop_line_distance(
    vehicle_pos: np.ndarray,
    stop_line_start: np.ndarray,
    stop_line_end: np.ndarray,
) -> float:
    """
    计算车辆到停止线的距离（专用接口）
    
    Args:
        vehicle_pos: 车辆位置 [x, y]
        stop_line_start: 停止线起点 [x1, y1]
        stop_line_end: 停止线终点 [x2, y2]
    
    Returns:
        distance: 到停止线的距离（米）
            - 正数：车辆在停止线前
            - 负数：车辆已过停止线
    """
    distance, _ = point_to_line_distance(
        vehicle_pos, stop_line_start, stop_line_end
    )
    return distance


# 导出接口
__all__ = [
    'compute_distance',
    'point_to_line_distance',
    'compute_stop_line_distance',
]
