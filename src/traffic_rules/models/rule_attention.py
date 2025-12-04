"""
规则聚焦注意力模块

设计依据：Design-ITER-2025-01.md v2.0 §3.3.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RuleFocusedAttention(nn.Module):
    """
    规则聚焦注意力
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.3.3
    
    功能：
        - 基于规则语义的注意力重分配
        - 引导模型关注与规则相关的实体（交通灯、停止线）
        - 注入可学习的规则嵌入
    
    数学形式：
        h_light^(i) = avg({h_j : j ∈ V_light})
        h_stop^(i) = avg({h_j : j ∈ V_stop})
        β_i = sigmoid(w_rule^T [h_i || h_light || h_stop])
        h_i^rule = β_i · h_i + (1-β_i) · e_rule
    
    Args:
        hidden_dim: 隐藏维度（默认128）
        num_rules: 规则数量（默认5：红灯停、车速、车道、安全距离等）
        dropout: Dropout概率（默认0.1）
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_rules: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 规则相关性评分网络
        self.rule_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # 规则嵌入（可学习）
        self.rule_embeddings = nn.Embedding(
            num_embeddings=num_rules,
            embedding_dim=hidden_dim,
        )
        
        # 初始化规则嵌入
        nn.init.normal_(self.rule_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        h_fused: torch.Tensor,
        entity_types: torch.Tensor,
        rule_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            h_fused: 融合全局后的节点表征 [N, hidden_dim]
            entity_types: 实体类型 [N] (0=car, 1=light, 2=stop)
            rule_id: 规则ID（默认0=红灯停）
        
        Returns:
            h_rule: 规则聚焦后的表征 [N, hidden_dim]
            beta: 规则注意力分数 [N_car]（仅车辆节点）
        """
        N = h_fused.size(0)
        device = h_fused.device
        
        # 提取不同类型实体的表征
        car_mask = (entity_types == 0)
        light_mask = (entity_types == 1)
        stop_mask = (entity_types == 2)
        
        h_cars = h_fused[car_mask] if car_mask.any() else torch.empty(0, self.hidden_dim, device=device)
        h_lights = h_fused[light_mask] if light_mask.any() else torch.empty(0, self.hidden_dim, device=device)
        h_stops = h_fused[stop_mask] if stop_mask.any() else torch.empty(0, self.hidden_dim, device=device)
        
        # 获取规则嵌入
        rule_emb = self.rule_embeddings(torch.tensor([rule_id], device=device))  # [1, D]
        
        # 计算每个车辆的规则聚焦分数
        beta_list = []
        h_rule_list = []
        
        num_cars = h_cars.size(0)
        
        if num_cars > 0:
            for i in range(num_cars):
                h_car = h_cars[i]  # [D]
                
                # 获取最近的交通灯和停止线表征（简化：取平均）
                if len(h_lights) > 0:
                    h_light_nearest = h_lights.mean(dim=0)
                else:
                    h_light_nearest = torch.zeros_like(h_car)
                
                if len(h_stops) > 0:
                    h_stop_nearest = h_stops.mean(dim=0)
                else:
                    h_stop_nearest = torch.zeros_like(h_car)
                
                # 拼接特征
                concat_feat = torch.cat([
                    h_car,
                    h_light_nearest,
                    h_stop_nearest,
                ], dim=0)  # [3D]
                
                # 计算规则相关性分数
                beta = self.rule_scorer(concat_feat)  # [1]
                beta_list.append(beta)
                
                # 加权融合（规则嵌入作为软约束）
                h_weighted = beta * h_car + (1 - beta) * rule_emb.squeeze(0)
                h_rule_list.append(h_weighted)
            
            # 堆叠结果
            beta_tensor = torch.stack(beta_list).squeeze()  # [N_car]
            h_cars_rule = torch.stack(h_rule_list)  # [N_car, D]
        else:
            beta_tensor = torch.empty(0, device=device)
            h_cars_rule = torch.empty(0, self.hidden_dim, device=device)
        
        # 重构完整表征（非车辆节点保持不变）
        h_rule = h_fused.clone()
        if num_cars > 0:
            h_rule[car_mask] = h_cars_rule
        
        return h_rule, beta_tensor


# 导出接口
__all__ = ['RuleFocusedAttention']
