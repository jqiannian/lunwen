"""
全局场景注意力模块

基于Design-ITER-2025-01.md v2.0 §3.3.2设计
通过虚拟全局节点聚合场景级上下文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GlobalSceneAttention(nn.Module):
    """
    全局场景注意力（类似Transformer的[CLS] token）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.3.2
    
    机制：
        1. 虚拟全局节点通过MultiheadAttention聚合所有局部节点信息
        2. 将全局上下文广播回每个局部节点
        3. 通过MLP融合局部+全局，残差连接
    
    数学形式：
        g = softmax(Q_g K_h^T / √d_h) V_h
        h_i' = h_i + MLP_fuse([h_i || g])
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        初始化全局场景注意力
        
        Args:
            hidden_dim: 隐藏层维度
            num_heads: 多头注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 全局节点（可学习参数）
        self.global_query = nn.Parameter(torch.randn(1, hidden_dim))
        
        # Transformer式多头自注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # 融合MLP：[h_local || g] → h
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        h_local: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            h_local: [N, hidden_dim] - 局部GAT输出
        
        Returns:
            h_global: [N, hidden_dim] - 融合全局上下文后的表征
            attn_weights: [N] - 全局注意力权重（每个节点对全局的贡献）
        """
        N = h_local.size(0)
        D = self.hidden_dim
        
        # Step 1: 全局节点聚合所有局部节点信息
        # global_query: [1, D] → [1, 1, D]（batch维度）
        # h_local: [N, D] → [1, N, D]（batch维度）
        global_query = self.global_query.unsqueeze(0)  # [1, 1, D]
        h_local_batch = h_local.unsqueeze(0)  # [1, N, D]
        
        # 多头注意力：global_query作为Q，h_local作为K和V
        global_context, attn_weights = self.multihead_attn(
            query=global_query,      # [1, 1, D]
            key=h_local_batch,       # [1, N, D]
            value=h_local_batch,     # [1, N, D]
            need_weights=True,
            average_attn_weights=True,  # 平均多头权重
        )
        # global_context: [1, 1, D]
        # attn_weights: [1, 1, N]
        
        # Step 2: 广播全局信息到每个局部节点
        global_context = global_context.squeeze(0).squeeze(0)  # [D]
        global_context = global_context.unsqueeze(0).expand(N, -1)  # [N, D]
        
        # Step 3: 融合局部+全局
        h_concat = torch.cat([h_local, global_context], dim=-1)  # [N, 2D]
        h_fused = self.fusion_mlp(h_concat)  # [N, D]
        
        # 残差连接（关键：梯度短路径）
        h_global = h_fused + h_local  # [N, D]
        
        # 提取注意力权重
        attn_weights = attn_weights.squeeze(0).squeeze(0)  # [N]
        
        return h_global, attn_weights


# ============ 导出接口 ============
__all__ = [
    'GlobalSceneAttention',
]



