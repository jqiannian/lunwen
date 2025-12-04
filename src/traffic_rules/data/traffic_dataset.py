"""
交通场景数据加载器

设计依据：Design-ITER-2025-01.md v2.0 §3.1
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from typing import List, Optional, Dict, Any
from torch.utils.data import Dataset

from src.traffic_rules.data import Entity, SceneContext
from src.traffic_rules.utils.geometry import compute_stop_line_distance


class TrafficLightDataset(Dataset):
    """
    交通灯场景数据集
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.1
    
    支持的模式：
        - synthetic: 合成数据
        - bdd100k: BDD100K数据集（待实现）
        - cityscapes: Cityscapes数据集（待实现）
    
    Args:
        data_root: 数据根目录
        mode: 数据模式 ('synthetic' | 'bdd100k' | 'cityscapes')
        split: 数据集分割 ('train' | 'val' | 'test')
        max_samples: 最大样本数（用于调试）
        augment: 是否启用数据增强
    """
    
    def __init__(
        self,
        data_root: str = "data/synthetic",
        mode: str = "synthetic",
        split: str = "train",
        max_samples: Optional[int] = None,
        augment: bool = False,
    ):
        """初始化数据集"""
        self.data_root = Path(data_root)
        self.mode = mode
        self.split = split
        self.augment = augment
        
        # 加载场景文件列表
        self.scene_files = self._load_scene_files()
        
        # 限制样本数
        if max_samples is not None:
            self.scene_files = self.scene_files[:max_samples]
        
        print(f"[TrafficLightDataset] 加载完成")
        print(f"  模式: {mode}")
        print(f"  分割: {split}")
        print(f"  场景数: {len(self.scene_files)}")
    
    def _load_scene_files(self) -> List[Path]:
        """
        加载场景文件列表
        
        Returns:
            scene_files: 场景文件路径列表
        """
        if self.mode == "synthetic":
            split_dir = self.data_root / self.split
            if not split_dir.exists():
                raise FileNotFoundError(
                    f"数据目录不存在: {split_dir}\n"
                    f"请先运行: python scripts/prepare_synthetic_data.py"
                )
            
            scene_files = sorted(split_dir.glob("*.json"))
            return scene_files
        
        elif self.mode == "bdd100k":
            # TODO: 实现BDD100K数据加载
            raise NotImplementedError("BDD100K数据集加载待实现（ITER-02）")
        
        elif self.mode == "cityscapes":
            # TODO: 实现Cityscapes数据加载
            raise NotImplementedError("Cityscapes数据集加载待实现（ITER-02）")
        
        else:
            raise ValueError(f"不支持的数据模式: {self.mode}")
    
    def _load_scene(self, scene_path: Path) -> SceneContext:
        """
        加载单个场景
        
        Args:
            scene_path: 场景JSON文件路径
        
        Returns:
            scene: SceneContext对象
        """
        with open(scene_path, 'r') as f:
            scene_dict = json.load(f)
        
        # 解析实体
        entities = []
        for entity_dict in scene_dict['entities']:
            entity = Entity(
                id=entity_dict['id'],
                type=entity_dict['type'],
                pos=np.array(entity_dict['pos']),
                velocity=entity_dict.get('velocity', 0.0),
                bbox=entity_dict.get('bbox'),
                d_stop=entity_dict.get('d_stop', 999.0),
                light_state=entity_dict.get('light_state'),
                confidence=entity_dict.get('confidence', 1.0),
                end_pos=np.array(entity_dict['end_pos']) if 'end_pos' in entity_dict else None,
                metadata=entity_dict.get('metadata', {}),
            )
            entities.append(entity)
        
        # 重新计算停止线距离（确保准确）
        entities = self._compute_stop_distances(entities)
        
        # 创建场景上下文
        scene = SceneContext(
            scene_id=scene_dict['scene_id'],
            timestamp=scene_dict['timestamp'],
            entities=entities,
            metadata=scene_dict.get('metadata', {}),
        )
        
        return scene
    
    def _compute_stop_distances(self, entities: List[Entity]) -> List[Entity]:
        """
        计算所有车辆到停止线的距离
        
        设计依据：Design-ITER-2025-01.md v2.0 §3.1.2
        
        Args:
            entities: 实体列表
        
        Returns:
            entities: 更新后的实体列表（车辆的d_stop字段）
        """
        # 找到停止线
        stop_lines = [e for e in entities if e.type == 'stop']
        if len(stop_lines) == 0:
            # 没有停止线，所有车辆距离设为999
            for entity in entities:
                if entity.type == 'car':
                    entity.d_stop = 999.0
            return entities
        
        # 使用第一个停止线
        stop_line = stop_lines[0]
        
        # 计算每辆车到停止线的距离
        for entity in entities:
            if entity.type == 'car':
                distance = compute_stop_line_distance(
                    vehicle_pos=entity.pos,
                    stop_line_start=stop_line.pos,
                    stop_line_end=stop_line.end_pos,
                )
                entity.d_stop = distance
        
        return entities
    
    def _augment(self, scene: SceneContext) -> SceneContext:
        """
        数据增强（可选）
        
        Args:
            scene: 原始场景
        
        Returns:
            scene: 增强后的场景
        """
        if not self.augment:
            return scene
        
        # TODO: 实现数据增强
        # - 位置扰动
        # - 速度扰动
        # - 光照变化（如果有图像）
        
        return scene
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.scene_files)
    
    def __getitem__(self, idx: int) -> SceneContext:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            scene: SceneContext对象
        """
        scene_path = self.scene_files[idx]
        scene = self._load_scene(scene_path)
        scene = self._augment(scene)
        
        return scene


# 导出接口
__all__ = [
    'TrafficLightDataset',
]
