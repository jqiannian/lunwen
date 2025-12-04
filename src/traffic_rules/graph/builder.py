"""
场景图构建器

基于Design-ITER-2025-01.md v2.0 §3.2设计
将交通场景实体转换为图神经网络输入
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class GraphBatch:
    """
    图批次数据结构
    
    用于PyTorch模型输入
    """
    # 节点特征矩阵
    x: torch.Tensor  # [N, input_dim]
    
    # 边索引（稀疏格式）
    edge_index: torch.Tensor  # [2, E]
    
    # 实体类型
    entity_types: torch.Tensor  # [N] (0=car, 1=light, 2=stop)
    
    # 实体mask（有效实体）
    entity_masks: torch.Tensor  # [N]
    
    # 场景上下文信息
    scene_id: str
    num_nodes: int
    num_edges: int
    
    # 原始实体列表（用于可视化和解释）
    entities: List  # List[Entity]


class GraphBuilder:
    """
    场景图构建器
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.2
    
    功能：
        1. 提取实体特征（位置、速度、距离等）
        2. 构建邻接矩阵（稀疏边）
        3. 生成GraphBatch供模型输入
    
    节点特征维度：10
        - 位置(x, y)：2
        - 速度(vx, vy)：2
        - 尺寸(w, h)：2
        - 停止线距离：1
        - 类型one-hot：3
    """
    
    def __init__(
        self,
        r_spatial: float = 50.0,        # 空间连接半径（米）
        r_car_car: float = 30.0,        # 车-车连接半径
        r_car_light: float = 50.0,      # 车-灯连接半径
        r_car_stop: float = 100.0,      # 车-停止线连接半径
        input_dim: int = 10,            # 节点特征维度
    ):
        """
        初始化图构建器
        
        Args:
            r_spatial: 默认空间连接半径
            r_car_car: 车辆间连接半径
            r_car_light: 车辆-交通灯连接半径
            r_car_stop: 车辆-停止线连接半径
            input_dim: 节点特征维度（默认10）
        """
        self.r_spatial = r_spatial
        self.r_car_car = r_car_car
        self.r_car_light = r_car_light
        self.r_car_stop = r_car_stop
        self.input_dim = input_dim
    
    def build(
        self,
        entities: List,
        scene_id: str = "unknown",
    ) -> GraphBatch:
        """
        构建场景图
        
        Args:
            entities: 实体列表（来自数据加载器）
            scene_id: 场景ID
        
        Returns:
            graph_batch: GraphBatch对象
        """
        # 1. 提取节点特征
        x, entity_types, entity_masks = self._extract_node_features(entities)
        
        # 2. 构建边索引（稀疏邻接矩阵）
        edge_index = self._build_edges(entities, entity_types)
        
        # 3. 构造GraphBatch
        graph_batch = GraphBatch(
            x=x,
            edge_index=edge_index,
            entity_types=entity_types,
            entity_masks=entity_masks,
            scene_id=scene_id,
            num_nodes=x.size(0),
            num_edges=edge_index.size(1),
            entities=entities,
        )
        
        return graph_batch
    
    def _extract_node_features(
        self,
        entities: List,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        提取节点特征矩阵
        
        设计依据：Design §3.1.2
        
        特征组成（10维）：
            - 位置(x, y)：2维
            - 速度(vx, vy)：2维（车辆有效，其他填0）
            - 尺寸(w, h)：2维
            - 停止线距离：1维（车辆有效，其他填999）
            - 类型one-hot：3维（car, light, stop）
        
        Returns:
            x: [N, 10]
            entity_types: [N]
            entity_masks: [N]
        """
        N = len(entities)
        x = torch.zeros(N, self.input_dim)
        entity_types = torch.zeros(N, dtype=torch.long)
        entity_masks = torch.ones(N, dtype=torch.bool)
        
        type_map = {'car': 0, 'light': 1, 'stop': 2}
        
        for i, entity in enumerate(entities):
            # 位置
            x[i, 0:2] = torch.tensor(entity.position)
            
            # 速度（仅车辆）
            if entity.type == 'car':
                vx = entity.velocity * np.cos(np.radians(entity.heading))
                vy = entity.velocity * np.sin(np.radians(entity.heading))
                x[i, 2:4] = torch.tensor([vx, vy])
            else:
                x[i, 2:4] = 0.0  # 非车辆填0
            
            # 尺寸
            if entity.bbox is not None:
                x1, y1, x2, y2 = entity.bbox
                w = x2 - x1
                h = y2 - y1
                x[i, 4:6] = torch.tensor([w, h])
            else:
                x[i, 4:6] = torch.tensor([10.0, 10.0])  # 默认尺寸
            
            # 停止线距离（仅车辆）
            if entity.type == 'car' and hasattr(entity, 'd_stop'):
                x[i, 6] = entity.d_stop
            else:
                x[i, 6] = 999.0  # 非车辆填大值
            
            # 类型one-hot
            entity_type = type_map[entity.type]
            x[i, 7 + entity_type] = 1.0
            
            # 记录类型索引
            entity_types[i] = entity_type
        
        return x, entity_types, entity_masks
    
    def _build_edges(
        self,
        entities: List,
        entity_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        构建稀疏邻接矩阵（边索引）
        
        设计依据：Design §3.2.2 + §3.3.1
        
        边类型：
            1. 车辆-车辆（距离<30m）
            2. 车辆-交通灯（距离<50m）
            3. 车辆-停止线（距离<100m）
        
        Returns:
            edge_index: [2, E] - [source_nodes, target_nodes]
        """
        N = len(entities)
        edges = []
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue  # 自环后续在GAT层添加
                
                e_i = entities[i]
                e_j = entities[j]
                
                # 计算距离
                dist = np.linalg.norm(
                    np.array(e_i.position) - np.array(e_j.position)
                )
                
                # 根据实体类型和距离判断是否连接
                should_connect = False
                
                if e_i.type == 'car' and e_j.type == 'car':
                    # 车-车：<30m
                    if dist < self.r_car_car:
                        should_connect = True
                
                elif e_i.type == 'car' and e_j.type == 'light':
                    # 车-灯：<50m
                    if dist < self.r_car_light:
                        should_connect = True
                
                elif e_i.type == 'light' and e_j.type == 'car':
                    # 灯-车：<50m（双向）
                    if dist < self.r_car_light:
                        should_connect = True
                
                elif e_i.type == 'car' and e_j.type == 'stop':
                    # 车-停止线：<100m
                    if dist < self.r_car_stop:
                        should_connect = True
                
                elif e_i.type == 'stop' and e_j.type == 'car':
                    # 停止线-车：<100m（双向）
                    if dist < self.r_car_stop:
                        should_connect = True
                
                if should_connect:
                    edges.append([i, j])
        
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).T  # [2, E]
        else:
            # 空图（没有边）
            edge_index = torch.empty(2, 0, dtype=torch.long)
        
        return edge_index
    
    def build_batch(
        self,
        scenes_data: List[Dict],
    ) -> List[GraphBatch]:
        """
        批量构建场景图
        
        Args:
            scenes_data: 场景数据列表（来自DataLoader）
        
        Returns:
            graph_batches: GraphBatch列表
        """
        graph_batches = []
        
        for scene in scenes_data:
            entities = scene['entities']
            scene_id = scene.get('scene_id', 'unknown')
            
            graph_batch = self.build(entities, scene_id)
            graph_batches.append(graph_batch)
        
        return graph_batches


def compute_stopline_distance(
    car_position: Tuple[float, float],
    stopline_endpoints: Tuple[Tuple[float, float], Tuple[float, float]],
) -> float:
    """
    计算车辆到停止线的距离（点到线段的距离）
    
    设计依据：Design §3.1.2
    
    数学形式：
        d = ||(p - s1) × (s2 - s1)|| / ||s2 - s1||
    
    Args:
        car_position: 车辆中心坐标(x, y)
        stopline_endpoints: 停止线端点((x1, y1), (x2, y2))
    
    Returns:
        distance: 到停止线的有向距离（正数=未过线，负数=已过线）
    """
    p = np.array(car_position)
    s1 = np.array(stopline_endpoints[0])
    s2 = np.array(stopline_endpoints[1])
    
    # 向量投影
    line_vec = s2 - s1
    point_vec = p - s1
    
    # 叉积（2D中为标量）
    cross = point_vec[0] * line_vec[1] - point_vec[1] * line_vec[0]
    
    # 距离
    line_length = np.linalg.norm(line_vec)
    if line_length < 1e-6:
        return 0.0
    
    distance = abs(cross) / line_length
    
    # 判断是否过线（使用点积判断方向）
    dot = np.dot(point_vec, line_vec)
    if dot > line_length ** 2:
        # 车辆在停止线后方（已过线）
        distance = -distance
    
    return distance


# ============ 导出接口 ============
__all__ = [
    'GraphBuilder',
    'GraphBatch',
    'compute_stopline_distance',
]
