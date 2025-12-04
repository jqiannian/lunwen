"""
多阶段注意力增强GAT模型

基于Design-ITER-2025-01.md v2.0 §3.3集成设计
整合局部GAT、全局注意力、规则聚焦三个阶段
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .gat_layers import LocalGATEncoder
from .global_attention import GlobalSceneAttention
from .rule_attention import RuleFocusedAttention


class MultiStageAttentionGAT(nn.Module):
    """
    多阶段注意力增强GAT（方案1核心模型）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.3
    
    架构：
        输入 → 阶段1（局部GAT） → 阶段2（全局注意力） → 
               阶段3（规则聚焦） → 多路径融合 → Scoring Head → 异常分数
    
    梯度流设计（§3.3.4）：
        h_final = γ1·h_local + γ2·h_global + γ3·h_rule
        
        其中γ1, γ2, γ3为可学习权重，确保梯度流经所有阶段
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        num_gat_layers: int = 3,
        num_heads: int = 8,
        num_global_heads: int = 4,
        num_rule_types: int = 5,
        dropout: float = 0.1,
    ):
        """
        初始化多阶段注意力GAT
        
        Args:
            input_dim: 输入特征维度（默认10）
            hidden_dim: 隐藏层维度（默认128）
            num_gat_layers: GAT层数（默认3）
            num_heads: GAT注意力头数（默认8）
            num_global_heads: 全局注意力头数（默认4）
            num_rule_types: 规则类型数量（默认5）
            dropout: Dropout概率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 阶段1：局部关系编码（Multi-Head GAT）
        self.local_gat = LocalGATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # 阶段2：全局场景注意力
        self.global_attention = GlobalSceneAttention(
            hidden_dim=hidden_dim,
            num_heads=num_global_heads,
            dropout=dropout,
        )
        
        # 阶段3：规则聚焦注意力
        self.rule_focus = RuleFocusedAttention(
            hidden_dim=hidden_dim,
            num_rule_types=num_rule_types,
            dropout=dropout,
        )
        
        # 多路径融合权重（可学习，关键：梯度平衡）
        # 初始化：[0.2, 0.3, 0.5]（局部、全局、规则）
        self.path_weights = nn.Parameter(
            torch.tensor([0.2, 0.3, 0.5])
        )
        
        # Scoring Head（异常分数计算）
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # 梯度监控（用于调试）
        self.grad_norms = {}
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        entity_types: torch.Tensor,
        entity_masks: Optional[torch.Tensor] = None,
        rule_id: int = 0,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [N, input_dim] - 节点特征
            edge_index: [2, E] - 边索引
            entity_types: [N] - 实体类型
            entity_masks: [N] - 有效实体mask
            rule_id: 规则ID
            return_attention: 是否返回注意力权重
        
        Returns:
            output_dict:
                - scores: [N_car] 异常分数
                - gat_attention: [E] GAT注意力权重（最后一层）
                - global_attention: [N] 全局注意力权重
                - rule_attention: [N_car] 规则聚焦注意力分数β_i
                - h_local, h_global, h_rule: [N, hidden_dim] 各阶段表征（如果return_attention）
        """
        # 阶段1：局部关系编码（Multi-Head GAT）
        h_local, gat_attentions = self.local_gat(
            x, edge_index, return_attention_weights=True
        )
        # h_local: [N, hidden_dim]
        # gat_attentions: List[[E], [E], [E]] (3层)
        
        # 阶段2：全局场景注意力
        h_global, global_attn = self.global_attention(h_local)
        # h_global: [N, hidden_dim]
        # global_attn: [N]
        
        # 阶段3：规则聚焦注意力
        h_rule, beta_rule = self.rule_focus(
            h_global, entity_types, entity_masks, rule_id
        )
        # h_rule: [N, hidden_dim]
        # beta_rule: [N_car]
        
        # 多路径梯度融合（关键：确保梯度流经所有阶段）
        # γ权重通过softmax归一化，自动学习最优比例
        gamma = F.softmax(self.path_weights, dim=0)
        
        h_final = (
            gamma[0] * h_local +   # 短路径（直接从GAT）
            gamma[1] * h_global +  # 中路径（经过全局注意力）
            gamma[2] * h_rule      # 长路径（经过完整三阶段）
        )
        # h_final: [N, hidden_dim]
        
        # 提取车辆节点
        car_mask = (entity_types == 0)
        h_cars = h_final[car_mask]  # [N_car, hidden_dim]
        
        # 异常分数计算
        scores = self.score_head(h_cars).squeeze(-1)  # [N_car]
        
        # 构造输出字典
        output = {
            'scores': scores,
            'rule_attention': beta_rule,
        }
        
        if return_attention:
            # 返回所有注意力权重（用于可视化和损失计算）
            output.update({
                'gat_attention': gat_attentions[-1],  # 最后一层GAT注意力
                'gat_attention_all_layers': gat_attentions,  # 所有层
                'global_attention': global_attn,
                'h_local': h_local,
                'h_global': h_global,
                'h_rule': h_rule,
                'h_final': h_final,
                'path_weights': gamma,  # 当前路径权重
            })
        
        return output
    
    def get_gradient_norms(self) -> Dict[str, float]:
        """
        获取各阶段的梯度范数（用于监控梯度流）
        
        设计依据：Design §3.3.4梯度监控
        
        Returns:
            grad_norms: {
                'gat_layers': float,
                'global_attn': float,
                'rule_focus': float,
                'score_head': float,
            }
        """
        grad_norms = {}
        
        # GAT层梯度
        gat_grad = 0.0
        for param in self.local_gat.parameters():
            if param.grad is not None:
                gat_grad += param.grad.norm().item() ** 2
        grad_norms['gat_layers'] = gat_grad ** 0.5
        
        # 全局注意力梯度
        global_grad = 0.0
        for param in self.global_attention.parameters():
            if param.grad is not None:
                global_grad += param.grad.norm().item() ** 2
        grad_norms['global_attention'] = global_grad ** 0.5
        
        # 规则聚焦梯度
        rule_grad = 0.0
        for param in self.rule_focus.parameters():
            if param.grad is not None:
                rule_grad += param.grad.norm().item() ** 2
        grad_norms['rule_focus'] = rule_grad ** 0.5
        
        # Scoring head梯度
        score_grad = 0.0
        for param in self.score_head.parameters():
            if param.grad is not None:
                score_grad += param.grad.norm().item() ** 2
        grad_norms['score_head'] = score_grad ** 0.5
        
        return grad_norms
    
    def check_gradient_health(self, grad_norms: Dict[str, float]) -> bool:
        """
        检查梯度健康度
        
        理想状态：各阶段梯度在同一数量级（1e-3 ~ 1e-2）
        
        Returns:
            healthy: 梯度是否健康
        """
        import math
        
        # 检查梯度消失（<1e-4）
        for name, norm in grad_norms.items():
            if norm < 1e-4:
                print(f"⚠️  梯度消失警告: {name} grad_norm={norm:.2e}")
                return False
        
        # 检查梯度爆炸（>1e2）
        for name, norm in grad_norms.items():
            if norm > 1e2:
                print(f"⚠️  梯度爆炸警告: {name} grad_norm={norm:.2e}")
                return False
        
        # 检查梯度不平衡（最大/最小 > 100）
        norms = list(grad_norms.values())
        if max(norms) / (min(norms) + 1e-10) > 100:
            print(f"⚠️  梯度不平衡: max={max(norms):.2e}, min={min(norms):.2e}")
            return False
        
        return True


# ============ 导出接口 ============
__all__ = [
    'MultiStageAttentionGAT',
]



