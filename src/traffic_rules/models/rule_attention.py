"""
规则聚焦注意力模块

基于Design-ITER-2025-01.md v2.0 §3.3.3设计
将注意力引导到与规则相关的实体（交通灯、停止线）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RuleFocusedAttention(nn.Module):
    """
    规则聚焦注意力
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.3.3
    
    机制：
        1. 对每个车辆节点，提取最近的交通灯和停止线表征
        2. 拼接特征：[h_car || h_light || h_stop]
        3. 通过rule_scorer计算规则关注度 β_i
        4. 加权融合：h_i' = β_i·h_i + (1-β_i)·e_rule
    
    数学形式：
        β_i = sigmoid(w_rule^T [h_i || h_light || h_stop])
        h_i' = β_i·h_i + (1-β_i)·e_rule
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_rule_types: int = 5,
        dropout: float = 0.1,
    ):
        """
        初始化规则聚焦注意力
        
        Args:
            hidden_dim: 隐藏层维度
            num_rule_types: 规则类型数量（红灯停、车速、车道等）
            dropout: Dropout概率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 规则相关性评分网络
        # 输入：[h_car || h_light || h_stop] = 3*hidden_dim
        self.rule_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # 规则嵌入（可学习，区分不同规则类型）
        self.rule_embeddings = nn.Embedding(
            num_embeddings=num_rule_types,
            embedding_dim=hidden_dim,
        )
        
        # 初始化规则嵌入
        nn.init.xavier_uniform_(self.rule_embeddings.weight)
    
    def forward(
        self,
        h_fused: torch.Tensor,
        entity_types: torch.Tensor,
        entity_masks: Optional[torch.Tensor] = None,
        rule_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            h_fused: [N, hidden_dim] - 融合全局上下文后的节点表征
            entity_types: [N] - 实体类型 (0=car, 1=light, 2=stop)
            entity_masks: [N] - 有效实体mask（可选）
            rule_id: 规则ID（默认0=红灯停）
        
        Returns:
            h_rule_focused: [N, hidden_dim] - 规则聚焦后的表征
            rule_attention: [N_car] - 每个车辆的规则注意力分数β_i
        """
        N = h_fused.size(0)
        device = h_fused.device
        
        # 如果没有提供mask，默认所有实体有效
        if entity_masks is None:
            entity_masks = torch.ones(N, dtype=torch.bool, device=device)
        
        # 提取不同类型实体
        car_mask = (entity_types == 0) & entity_masks
        light_mask = (entity_types == 1) & entity_masks
        stop_mask = (entity_types == 2) & entity_masks
        
        h_cars = h_fused[car_mask]  # [N_car, D]
        h_lights = h_fused[light_mask]  # [N_light, D]
        h_stops = h_fused[stop_mask]  # [N_stop, D]
        
        N_car = h_cars.size(0)
        
        # 获取规则嵌入
        rule_emb = self.rule_embeddings(
            torch.tensor([rule_id], device=device)
        )  # [1, D]
        
        # 计算每个车辆的规则注意力分数
        rule_attention_scores = []
        h_cars_focused = []
        
        for i in range(N_car):
            h_car = h_cars[i]  # [D]
            
            # 找到最近的交通灯和停止线（简化：取平均）
            if h_lights.size(0) > 0:
                h_light_nearest = h_lights.mean(dim=0)  # [D]
            else:
                h_light_nearest = torch.zeros_like(h_car)
            
            if h_stops.size(0) > 0:
                h_stop_nearest = h_stops.mean(dim=0)  # [D]
            else:
                h_stop_nearest = torch.zeros_like(h_car)
            
            # 拼接特征：[h_car || h_light || h_stop]
            concat_feat = torch.cat([
                h_car,
                h_light_nearest,
                h_stop_nearest
            ], dim=0)  # [3D]
            
            # 计算规则相关性分数β_i
            beta_i = self.rule_scorer(concat_feat)  # [1]
            rule_attention_scores.append(beta_i)
            
            # 加权融合（规则嵌入作为软约束）
            h_car_weighted = beta_i * h_car + (1 - beta_i) * rule_emb.squeeze(0)
            h_cars_focused.append(h_car_weighted)
        
        # 重构完整的节点表征
        h_rule_focused = h_fused.clone()
        
        if N_car > 0:
            h_rule_focused[car_mask] = torch.stack(h_cars_focused)
            rule_attention = torch.stack(rule_attention_scores).squeeze(-1)  # [N_car]
        else:
            rule_attention = torch.empty(0, device=device)
        
        return h_rule_focused, rule_attention


# ============ 导出接口 ============
__all__ = [
    'RuleFocusedAttention',
]



