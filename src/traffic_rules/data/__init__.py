"""
数据结构定义模块

定义交通场景的核心数据结构
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch
import numpy as np


@dataclass
class Entity:
    """
    场景实体（车辆、交通灯、停止线）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.1.2
    """
    id: int
    type: str  # 'car' | 'light' | 'stop'
    pos: np.ndarray  # [x, y] 位置坐标
    
    # 车辆特有属性
    velocity: float = 0.0  # m/s
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]
    d_stop: float = 999.0  # 到停止线的距离（米）
    
    # 交通灯特有属性
    light_state: Optional[str] = None  # 'red' | 'yellow' | 'green'
    confidence: float = 1.0  # 交通灯状态置信度
    
    # 停止线特有属性
    end_pos: Optional[np.ndarray] = None  # 线段终点 [x2, y2]
    
    # 通用属性
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证数据类型"""
        if not isinstance(self.pos, np.ndarray):
            self.pos = np.array(self.pos)
        if self.end_pos is not None and not isinstance(self.end_pos, np.ndarray):
            self.end_pos = np.array(self.end_pos)


@dataclass
class SceneContext:
    """
    场景上下文（单个交通场景的完整描述）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.1.1
    """
    scene_id: str
    timestamp: float
    entities: List[Entity]
    
    # 可选属性
    image: Optional[np.ndarray] = None  # 场景图像 [H, W, 3]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """获取指定类型的所有实体"""
        return [e for e in self.entities if e.type == entity_type]
    
    @property
    def num_entities(self) -> int:
        """实体总数"""
        return len(self.entities)
    
    @property
    def num_cars(self) -> int:
        """车辆数量"""
        return len(self.get_entities_by_type('car'))
    
    @property
    def num_lights(self) -> int:
        """交通灯数量"""
        return len(self.get_entities_by_type('light'))
    
    @property
    def num_stops(self) -> int:
        """停止线数量"""
        return len(self.get_entities_by_type('stop'))


@dataclass
class GraphBatch:
    """
    图批次数据（用于GAT模型输入）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.2
    
    Attributes:
        x: 节点特征矩阵 [N, feature_dim]
        edge_index: 边索引 [2, E] (COO格式)
        entity_types: 实体类型索引 [N] (0=car, 1=light, 2=stop)
        entity_masks: 有效实体mask [N]
        batch_index: 批次索引 [N] (用于多图批处理)
        context: 原始场景上下文
    """
    x: torch.Tensor  # [N, 10] 节点特征
    edge_index: torch.Tensor  # [2, E] 边索引
    entity_types: torch.Tensor  # [N] 类型索引
    entity_masks: torch.Tensor  # [N] bool mask
    
    # 可选属性
    batch_index: Optional[torch.Tensor] = None  # [N] 批次索引
    edge_attr: Optional[torch.Tensor] = None  # [E, edge_dim] 边特征
    
    # 原始上下文（用于后处理）
    context: Optional[SceneContext] = None
    
    @property
    def num_nodes(self) -> int:
        """节点数量"""
        return self.x.size(0)
    
    @property
    def num_edges(self) -> int:
        """边数量"""
        return self.edge_index.size(1)
    
    def to(self, device: torch.device) -> 'GraphBatch':
        """转移到指定设备"""
        return GraphBatch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            entity_types=self.entity_types.to(device),
            entity_masks=self.entity_masks.to(device),
            batch_index=self.batch_index.to(device) if self.batch_index is not None else None,
            edge_attr=self.edge_attr.to(device) if self.edge_attr is not None else None,
            context=self.context,
        )


# 导出接口
__all__ = [
    'Entity',
    'SceneContext',
    'GraphBatch',
]
