"""
局部GAT编码器（多层堆叠）

设计依据：Design-ITER-2025-01.md v2.0 §3.3.1
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.traffic_rules.models.gat_layers import GATLayer


class LocalGATEncoder(nn.Module):
    """
    局部关系编码器（3层GAT）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.3.1
    
    架构：
        Input [N, input_dim]
        → GAT Layer 1 [N, hidden_dim] + Residual + LayerNorm
        → GAT Layer 2 [N, hidden_dim] + Residual + LayerNorm
        → GAT Layer 3 [N, hidden_dim] + Residual + LayerNorm
        Output [N, hidden_dim]
    
    Args:
        input_dim: 输入特征维度（默认10）
        hidden_dim: 隐藏层维度（默认128）
        num_layers: GAT层数（默认3）
        num_heads: 注意力头数（默认8）
        dropout: Dropout概率（默认0.1）
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # GAT层
        self.gat_layers = nn.ModuleList([
            GATLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                concat=False,  # 使用平均而非拼接
            )
            for _ in range(num_layers)
        ])
        
        # LayerNorm（每层后）
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 存储最后一层的注意力权重
        self.last_attn_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, input_dim]
            edge_index: 边索引 [2, E]
        
        Returns:
            h: 输出特征 [N, hidden_dim]
            alpha: 最后一层注意力权重 [E]
        """
        # 输入投影
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.gelu(h)
        
        # 逐层传播
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            # 保存残差
            h_residual = h
            
            # GAT层
            h, alpha = gat_layer(h, edge_index)
            
            # 残差连接
            h = h + h_residual
            
            # LayerNorm
            h = layer_norm(h)
            
            # GELU激活（除了最后一层）
            if i < self.num_layers - 1:
                h = F.gelu(h)
                h = self.dropout(h)
            
            # 记录最后一层注意力
            if i == self.num_layers - 1:
                self.last_attn_weights = alpha
        
        return h, alpha


# 导出接口
__all__ = ['LocalGATEncoder']
