"""
全局场景注意力模块

设计依据：Design-ITER-2025-01.md v2.0 §3.3.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GlobalSceneAttention(nn.Module):
    """
    全局场景注意力（虚拟节点机制）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.3.2
    
    功能：
        - 通过可学习的全局节点汇总场景信息
        - 类似Transformer的[CLS] token
        - 广播全局上下文到所有局部节点
    
    架构：
        h_local [N, D]
        → Global Query (learnable)
        → MultiheadAttention(Q=global, K=V=local)
        → Global Context [D]
        → Broadcast + Fusion
        → h_global [N, D]
    
    Args:
        hidden_dim: 隐藏维度（默认128）
        num_heads: 注意力头数（默认4）
        dropout: Dropout概率（默认0.1）
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 可学习的全局query
        self.global_query = nn.Parameter(torch.randn(1, hidden_dim))
        
        # 多头自注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # 融合MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        h_local: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            h_local: 局部GAT输出 [N, hidden_dim]
        
        Returns:
            h_global: 融合全局上下文后的表征 [N, hidden_dim]
            attn_weights: 全局注意力权重 [N]
        """
        N = h_local.size(0)
        
        # Step 1: 全局节点聚合所有局部节点信息
        # global_query: [1, D] → [1, 1, D]
        # h_local: [N, D] → [1, N, D]
        global_query = self.global_query.unsqueeze(0)  # [1, 1, D]
        h_local_batch = h_local.unsqueeze(0)  # [1, N, D]
        
        # 注意力计算
        global_context, attn_weights = self.multihead_attn(
            query=global_query,         # [1, 1, D]
            key=h_local_batch,          # [1, N, D]
            value=h_local_batch,        # [1, N, D]
            need_weights=True,
        )
        # global_context: [1, 1, D]
        # attn_weights: [1, 1, N]
        
        # Step 2: 广播全局信息到每个局部节点
        global_context = global_context.squeeze(0)  # [1, D]
        global_context = global_context.expand(N, -1)  # [N, D]
        
        # Step 3: 融合局部+全局
        h_fused = torch.cat([h_local, global_context], dim=-1)  # [N, 2D]
        h_global = self.fusion(h_fused)  # [N, D]
        
        # Dropout
        h_global = self.dropout(h_global)
        
        # 提取注意力权重
        attn_weights = attn_weights.squeeze(0).squeeze(0)  # [N]
        
        return h_global, attn_weights


# 导出接口
__all__ = ['GlobalSceneAttention']


