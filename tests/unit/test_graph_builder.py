"""
GraphBuilder 单元测试（修复版）
"""

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.traffic_rules.graph.builder import GraphBuilder
from src.traffic_rules.data.traffic_dataset import TrafficLightDataset


class TestGraphBuilder:
    """测试GraphBuilder类"""
    
    def test_graph_builder_initialization(self):
        """测试图构建器初始化"""
        builder = GraphBuilder(
            feature_dim=10,
            r_car_car=20.0,
            r_car_light=30.0,
            r_car_stop=15.0,
        )
        
        assert builder.feature_dim == 10
        assert builder.r_car_car == 20.0
        print("✅ GraphBuilder初始化成功")
    
    def test_build_graph_from_scene(self):
        """测试从场景构建图"""
        dataset = TrafficLightDataset(
            data_root="data/synthetic",
            mode="synthetic",
            split="val",
            max_samples=1,
        )
        
        scene = dataset[0]
        builder = GraphBuilder()
        graph = builder.build(scene)
        
        # 验证图结构
        assert hasattr(graph, 'x'), "缺少节点特征x"
        assert hasattr(graph, 'edge_index'), "缺少边索引edge_index"
        assert hasattr(graph, 'entity_types'), "缺少实体类型"
        
        # 验证维度
        assert graph.x.shape[1] == 10, f"节点特征应为10维，实际{graph.x.shape[1]}"
        assert graph.edge_index.shape[0] == 2, "边索引应为2×E"
        
        print(f"✅ 图构建成功: {graph.x.shape[0]}个节点, {graph.edge_index.shape[1]}条边")
    
    def test_edge_construction(self):
        """测试边构建逻辑"""
        dataset = TrafficLightDataset(
            data_root="data/synthetic",
            mode="synthetic",
            split="val",
            max_samples=1,
        )
        
        scene = dataset[0]
        builder = GraphBuilder()
        graph = builder.build(scene)
        
        # 验证边数大于0
        assert graph.edge_index.shape[1] > 0, "图应至少有一些边"
        
        # 验证边索引范围
        num_nodes = graph.x.shape[0]
        assert graph.edge_index.max() < num_nodes, "边索引超出节点数量"
        assert graph.edge_index.min() >= 0, "边索引不应为负"
        
        print(f"✅ 边构建正确: {graph.edge_index.shape[1]}条边")
    
    def test_entity_types_encoding(self):
        """测试实体类型编码"""
        dataset = TrafficLightDataset(
            data_root="data/synthetic",
            mode="synthetic",
            split="val",
            max_samples=1,
        )
        
        scene = dataset[0]
        builder = GraphBuilder()
        graph = builder.build(scene)
        
        # 验证实体类型
        unique_types = torch.unique(graph.entity_types)
        # 应包含：0(car), 1(light), 2(stop)
        assert 0 in unique_types, "应包含车辆类型(0)"
        assert len(unique_types) >= 2, "至少应有2种实体类型"
        
        print(f"✅ 实体类型编码正确: {unique_types.tolist()}")


if __name__ == "__main__":
    test = TestGraphBuilder()
    
    test.test_graph_builder_initialization()
    test.test_build_graph_from_scene()
    test.test_edge_construction()
    test.test_entity_types_encoding()
    
    print("\n" + "="*60)
    print("✅ 所有GraphBuilder测试通过！")
    print("="*60)
