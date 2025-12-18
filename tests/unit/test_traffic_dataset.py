"""
TrafficLightDataset 单元测试
"""

import sys
from pathlib import Path

# import pytest
import torch

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.traffic_rules.data.traffic_dataset import TrafficLightDataset


class TestTrafficLightDataset:
    """测试TrafficLightDataset类"""
    
    def test_synthetic_dataset_loading(self):
        """测试合成数据加载"""
        dataset = TrafficLightDataset(
            data_root="data/synthetic",
            mode="synthetic",
            split="val",
            max_samples=5,
        )
        
        assert len(dataset) == 5, f"期望5个样本，实际{len(dataset)}"
        print(f"✅ 数据集加载成功: {len(dataset)}个样本")
    
    def test_scene_context_structure(self):
        """测试SceneContext数据结构"""
        dataset = TrafficLightDataset(
            data_root="data/synthetic",
            mode="synthetic",
            split="val",
            max_samples=1,
        )
        
        scene = dataset[0]
        
        # 验证必需属性
        assert hasattr(scene, 'scene_id'), "缺少scene_id"
        assert hasattr(scene, 'entities'), "缺少entities"
        assert hasattr(scene, 'num_entities'), "缺少num_entities"
        assert hasattr(scene, 'num_cars'), "缺少num_cars"
        
        # 验证实体列表
        assert len(scene.entities) > 0, "实体列表为空"
        
        print(f"✅ SceneContext结构正确: {scene.scene_id}")
    
    def test_entity_types(self):
        """测试实体类型解析"""
        dataset = TrafficLightDataset(
            data_root="data/synthetic",
            mode="synthetic",
            split="val",
            max_samples=1,
        )
        
        scene = dataset[0]
        
        # 统计实体类型
        entity_types = {}
        for entity in scene.entities:
            etype = entity.type
            entity_types[etype] = entity_types.get(etype, 0) + 1
        
        # 验证必须有的实体类型
        assert 'car' in entity_types, "缺少车辆实体"
        assert 'light' in entity_types, "缺少交通灯实体"
        assert 'stop' in entity_types, "缺少停止线实体"
        
        print(f"✅ 实体类型正确: {entity_types}")


if __name__ == "__main__":
    # 运行测试
    test = TestTrafficLightDataset()
    
    test.test_synthetic_dataset_loading()
    test.test_scene_context_structure()
    test.test_entity_types()
    
    print("\n" + "="*60)
    print("✅ 所有TrafficLightDataset测试通过！")
    print("="*60)
