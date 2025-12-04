"""
多阶段注意力GAT模型（完整版）

设计依据：Design-ITER-2025-01.md v2.0 §3.3
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.traffic_rules.models.local_gat import LocalGATEncoder
from src.traffic_rules.models.global_attention import GlobalSceneAttention
from src.traffic_rules.models.rule_attention import RuleFocusedAttention


class MultiStageAttentionGAT(nn.Module):
    """
    多阶段注意力增强GAT（完整方案1实现）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.3
    算法方案：ALGORITHM_DESIGN_OPTIONS.md 方案1
    
    三阶段架构：
        阶段1: 局部GAT（稀疏图，3层×8头）
        阶段2: 全局虚拟节点注意力（场景级上下文）
        阶段3: 规则聚焦注意力（规则语义注入）
    
    多路径融合：
        h_final = γ1·h_local + γ2·h_global + γ3·h_rule
        其中 γ = softmax([0.2, 0.3, 0.5]) 为可学习权重
    
    Args:
        input_dim: 输入特征维度（默认10）
        hidden_dim: 隐藏层维度（默认128）
        num_gat_layers: GAT层数（默认3）
        num_heads: GAT注意力头数（默认8）
        num_global_heads: 全局注意力头数（默认4）
        dropout: Dropout概率（默认0.1）
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        num_gat_layers: int = 3,
        num_heads: int = 8,
        num_global_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 阶段1: 局部GAT
        self.local_gat = LocalGATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # 阶段2: 全局注意力
        self.global_attn = GlobalSceneAttention(
            hidden_dim=hidden_dim,
            num_heads=num_global_heads,
            dropout=dropout,
        )
        
        # 阶段3: 规则聚焦
        self.rule_focus = RuleFocusedAttention(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        # 多路径融合权重（可学习）
        self.path_weights = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))
        
        # 异常分数头
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        entity_types: torch.Tensor,
        entity_masks: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, input_dim]
            edge_index: 边索引 [2, E]
            entity_types: 实体类型 [N] (0=car, 1=light, 2=stop)
            entity_masks: 有效实体mask [N]（可选）
            return_attention: 是否返回注意力权重
        
        Returns:
            output: 字典包含
                - scores: 异常分数 [N_car]
                - gat_attention: GAT注意力权重 [E]（可选）
                - global_attention: 全局注意力权重 [N]（可选）
                - rule_attention: 规则聚焦分数 [N_car]（可选）
        """
        N = x.size(0)
        
        # 默认mask（所有实体有效）
        if entity_masks is None:
            entity_masks = torch.ones(N, dtype=torch.bool, device=x.device)
        
        # 阶段1: 局部GAT
        h_local, alpha_gat = self.local_gat(x, edge_index)
        # h_local: [N, hidden_dim]
        # alpha_gat: [E]
        
        # 阶段2: 全局注意力
        h_global, global_attn = self.global_attn(h_local)
        # h_global: [N, hidden_dim]
        # global_attn: [N]
        
        # 残差连接
        h_global = h_global + h_local
        
        # 阶段3: 规则聚焦
        h_rule, beta_rule = self.rule_focus(h_global, entity_types, rule_id=0)
        # h_rule: [N, hidden_dim]
        # beta_rule: [N_car]
        
        # 残差连接
        h_rule = h_rule + h_global
        
        # 多路径融合（关键：梯度均衡）
        gamma = F.softmax(self.path_weights, dim=0)
        h_final = (
            gamma[0] * h_local +
            gamma[1] * h_global +
            gamma[2] * h_rule
        )
        # h_final: [N, hidden_dim]
        
        # 异常分数头（仅对车辆节点）
        car_mask = (entity_types == 0)
        if car_mask.any():
            h_cars = h_final[car_mask]
            scores = self.score_head(h_cars).squeeze()  # [N_car]
            
            # 处理单个车辆的情况（保持1D张量）
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
        else:
            scores = torch.empty(0, device=x.device)
        
        # 组装输出
        output = {
            'scores': scores,
        }
        
        if return_attention:
            output['gat_attention'] = alpha_gat
            output['global_attention'] = global_attn
            output['rule_attention'] = beta_rule
        
        return output


# 导出接口
__all__ = ['MultiStageAttentionGAT']
