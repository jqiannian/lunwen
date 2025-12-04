"""
场景图构建器

将交通场景转换为图神经网络输入

设计依据：Design-ITER-2025-01.md v2.0 §3.2
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from typing import List, Tuple

from src.traffic_rules.data import Entity, SceneContext, GraphBatch
from src.traffic_rules.utils.geometry import compute_distance


class GraphBuilder:
    """
    场景图构建器
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.2
    
    功能：
        1. 实体特征编码（10维）
        2. 邻接矩阵构建（稀疏异构图）
        3. GraphBatch数据结构生成
    
    Args:
        feature_dim: 特征维度（默认10）
        r_car_car: 车辆-车辆连接半径（米）
        r_car_light: 车辆-交通灯连接半径（米）
        r_car_stop: 车辆-停止线连接半径（米）
    """
    
    def __init__(
        self,
        feature_dim: int = 10,
        r_car_car: float = 30.0,
        r_car_light: float = 50.0,
        r_car_stop: float = 100.0,
    ):
        """初始化图构建器"""
        self.feature_dim = feature_dim
        self.r_car_car = r_car_car
        self.r_car_light = r_car_light
        self.r_car_stop = r_car_stop
        
        # 实体类型映射
        self.type_map = {
            'car': 0,
            'light': 1,
            'stop': 2,
        }
    
    def encode_entity_features(self, entity: Entity) -> np.ndarray:
        """
        编码单个实体的特征向量（10维）
        
        设计依据：Design-ITER-2025-01.md v2.0 §3.1.2
        
        特征构成：
            [0-1]: 位置 (x, y)
            [2-3]: 速度 (vx, vy) - 仅车辆有效
            [4-5]: 尺寸 (w, h) - bbox宽高
            [6]:   停止线距离 d_stop - 仅车辆有效
            [7-9]: 类型one-hot [car, light, stop]
        
        Args:
            entity: 实体对象
        
        Returns:
            features: 特征向量 [10]
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # [0-1] 位置
        features[0] = entity.pos[0]
        features[1] = entity.pos[1]
        
        # [2-3] 速度（仅车辆，其他填0）
        if entity.type == 'car':
            # 简化：假设沿y轴运动
            features[2] = 0.0  # vx
            features[3] = entity.velocity  # vy
        else:
            features[2] = 0.0
            features[3] = 0.0
        
        # [4-5] 尺寸
        if entity.bbox is not None:
            w = entity.bbox[2] - entity.bbox[0]
            h = entity.bbox[3] - entity.bbox[1]
            features[4] = w
            features[5] = h
        elif entity.type == 'car':
            # 默认车辆尺寸
            features[4] = 4.0  # 宽度
            features[5] = 8.0  # 长度
        elif entity.type == 'stop' and entity.end_pos is not None:
            # 停止线长度
            length = np.linalg.norm(entity.end_pos - entity.pos)
            features[4] = length
            features[5] = 0.5  # 宽度（线）
        else:
            features[4] = 1.0
            features[5] = 1.0
        
        # [6] 停止线距离（仅车辆）
        if entity.type == 'car':
            features[6] = entity.d_stop
        else:
            features[6] = 999.0 if entity.type == 'light' else 0.0
        
        # [7-9] 类型one-hot
        type_idx = self.type_map.get(entity.type, 0)
        features[7 + type_idx] = 1.0
        
        return features
    
    def build_adjacency(
        self,
        entities: List[Entity],
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        构建邻接矩阵（稀疏COO格式）
        
        设计依据：Design-ITER-2025-01.md v2.0 §3.2.2
        
        边构建规则：
            - 车辆-车辆：距离 < r_car_car (30m)
            - 车辆-交通灯：距离 < r_car_light (50m)
            - 车辆-停止线：距离 < r_car_stop (100m)
        
        Args:
            entities: 实体列表
        
        Returns:
            edge_index: 边索引 [2, E] (COO格式)
            edge_distances: 边距离 [E]
        """
        N = len(entities)
        edges = []
        distances = []
        
        for i in range(N):
            for j in range(i + 1, N):  # 只遍历上三角
                e_i = entities[i]
                e_j = entities[j]
                
                # 计算距离
                dist = compute_distance(e_i.pos, e_j.pos)
                
                # 判断是否连接
                should_connect = False
                
                # 车辆-车辆
                if e_i.type == 'car' and e_j.type == 'car':
                    if dist < self.r_car_car:
                        should_connect = True
                
                # 车辆-交通灯（双向）
                elif (e_i.type == 'car' and e_j.type == 'light') or \
                     (e_i.type == 'light' and e_j.type == 'car'):
                    if dist < self.r_car_light:
                        should_connect = True
                
                # 车辆-停止线（双向）
                elif (e_i.type == 'car' and e_j.type == 'stop') or \
                     (e_i.type == 'stop' and e_j.type == 'car'):
                    if dist < self.r_car_stop:
                        should_connect = True
                
                # 添加边（无向图，添加两个方向）
                if should_connect:
                    edges.append([i, j])
                    edges.append([j, i])
                    distances.append(dist)
                    distances.append(dist)
        
        if len(edges) == 0:
            # 没有边，返回空边索引
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_distances = np.array([])
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).T  # [2, E]
            edge_distances = np.array(distances, dtype=np.float32)
        
        return edge_index, edge_distances
    
    def build(self, scene: SceneContext) -> GraphBatch:
        """
        构建单个场景图
        
        Args:
            scene: 场景上下文
        
        Returns:
            graph_batch: GraphBatch对象
        """
        entities = scene.entities
        N = len(entities)
        
        # 1. 编码节点特征
        features = []
        entity_types = []
        
        for entity in entities:
            feat = self.encode_entity_features(entity)
            features.append(feat)
            entity_types.append(self.type_map[entity.type])
        
        x = torch.from_numpy(np.stack(features, axis=0))  # [N, 10]
        entity_types = torch.tensor(entity_types, dtype=torch.long)  # [N]
        
        # 2. 构建邻接矩阵
        edge_index, edge_distances = self.build_adjacency(entities)
        
        # 3. 创建mask（所有实体都有效）
        entity_masks = torch.ones(N, dtype=torch.bool)
        
        # 4. 创建GraphBatch
        graph_batch = GraphBatch(
            x=x,
            edge_index=edge_index,
            entity_types=entity_types,
            entity_masks=entity_masks,
            edge_attr=torch.from_numpy(edge_distances).unsqueeze(1) if len(edge_distances) > 0 else None,
            context=scene,
        )
        
        return graph_batch
    
    def build_batch(self, scenes: List[SceneContext]) -> List[GraphBatch]:
        """
        构建批量场景图
        
        Args:
            scenes: 场景列表
        
        Returns:
            graphs: GraphBatch列表
        """
        graphs = []
        for scene in scenes:
            graph = self.build(scene)
            graphs.append(graph)
        
        return graphs


# 导出接口
__all__ = [
    'GraphBuilder',
]
