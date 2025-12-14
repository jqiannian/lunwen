"""
图注意力网络层实现

设计依据：Design-ITER-2025-01.md v2.0 §3.3.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GATLayer(nn.Module):
    """
    单层图注意力网络（Multi-Head GAT）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.3.1
    参考文献：Veličković et al. "Graph Attention Networks" ICLR 2018
    
    Args:
        in_dim: 输入特征维度
        out_dim: 输出特征维度
        num_heads: 注意力头数（默认8）
        dropout: Dropout概率（默认0.1）
        concat: 是否拼接多头输出（False则平均）
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        concat: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        
        # 每个头的维度
        if concat:
            assert out_dim % num_heads == 0
            self.head_dim = out_dim // num_heads
        else:
            self.head_dim = out_dim
        
        # 权重矩阵（每个头独立）
        self.W = nn.ModuleList([
            nn.Linear(in_dim, self.head_dim, bias=False)
            for _ in range(num_heads)
        ])
        
        # 注意力权重向量（每个头独立）
        self.a = nn.ParameterList([
            nn.Parameter(torch.randn(2 * self.head_dim, 1))
            for _ in range(num_heads)
        ])
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """Xavier初始化"""
        for w in self.W:
            nn.init.xavier_uniform_(w.weight)
        for a in self.a:
            nn.init.xavier_uniform_(a)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, in_dim]
            edge_index: 边索引 [2, E]
        
        Returns:
            h: 输出特征 [N, out_dim]
            alpha: 注意力权重 [E]（多头平均）
        """
        N = x.size(0)
        
        # 存储每个头的输出
        head_outputs = []
        head_attentions = []
        
        for k in range(self.num_heads):
            # 线性变换
            h = self.W[k](x)  # [N, head_dim]
            
            # 计算注意力系数
            src, dst = edge_index[0], edge_index[1]
            
            # 拼接源节点和目标节点特征
            h_cat = torch.cat([h[src], h[dst]], dim=1)  # [E, 2*head_dim]
            
            # 计算注意力分数
            e = self.leaky_relu(torch.matmul(h_cat, self.a[k])).squeeze()  # [E]
            
            # Softmax归一化（按目标节点分组）
            alpha = self._softmax(e, dst, N)  # [E]
            alpha = self.dropout(alpha)
            
            # 聚合邻居特征
            h_out = torch.zeros(N, self.head_dim, device=x.device)
            h_out.index_add_(0, dst, alpha.unsqueeze(1) * h[src])
            
            head_outputs.append(h_out)
            head_attentions.append(alpha)
        
        # 合并多头输出
        if self.concat:
            h_final = torch.cat(head_outputs, dim=1)  # [N, out_dim]
        else:
            h_final = torch.stack(head_outputs, dim=0).mean(dim=0)  # [N, head_dim]
        
        # 平均注意力权重
        alpha_avg = torch.stack(head_attentions, dim=0).mean(dim=0)  # [E]
        
        return h_final, alpha_avg
    
    def _softmax(self, e: torch.Tensor, index: torch.Tensor, N: int) -> torch.Tensor:
        """
        按索引分组的Softmax
        
        Args:
            e: 注意力分数 [E]
            index: 目标节点索引 [E]
            N: 节点总数
        
        Returns:
            alpha: 归一化后的注意力权重 [E]
        """
        # 计算每个节点的最大值（数值稳定性）
        e_max = torch.full((N,), float('-inf'), device=e.device)
        e_max.scatter_reduce_(0, index, e, reduce='amax', include_self=False)
        e_max = e_max[index]
        
        # Exp
        e_exp = torch.exp(e - e_max)
        
        # 求和（按目标节点）
        e_sum = torch.zeros(N, device=e.device)
        e_sum.index_add_(0, index, e_exp)
        
        # 归一化
        alpha = e_exp / (e_sum[index] + 1e-16)
        
        return alpha


# 导出接口
__all__ = ['GATLayer']


