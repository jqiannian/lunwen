"""
图注意力网络（GAT）层实现

基于Design-ITER-2025-01.md v2.0 §3.3.1设计
实现多头GAT with残差连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MultiHeadGATLayer(nn.Module):
    """
    多头图注意力层（单层）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.3.1
    参考论文：Veličković et al. "Graph Attention Networks" ICLR 2018
    
    数学形式：
        α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
        h_i' = Σ_j α_ij W h_j
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        concat: bool = True,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
    ):
        """
        初始化多头GAT层
        
        Args:
            in_channels: 输入特征维度
            out_channels: 每个头的输出维度
            num_heads: 注意力头数
            concat: 是否拼接多头输出（True）或平均（False）
            dropout: Dropout概率
            negative_slope: LeakyReLU负斜率
            add_self_loops: 是否添加自环
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        
        # 每个头的线性变换
        self.lin = nn.Linear(in_channels, num_heads * out_channels, bias=False)
        
        # 注意力参数 a^T (对每个头)
        self.att = nn.Parameter(torch.empty(1, num_heads, 2 * out_channels))
        
        # Bias（如果concat，维度是num_heads*out_channels；否则是out_channels）
        if concat:
            self.bias = nn.Parameter(torch.empty(num_heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.empty(out_channels))
        
        self.reset_parameters()
        
        # 用于存储最后一次的注意力权重（可视化用）
        self.last_attention_weights = None
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: [N, in_channels] - 节点特征
            edge_index: [2, E] - 边索引 [source, target]
            return_attention_weights: 是否返回注意力权重
        
        Returns:
            out: [N, num_heads*out_channels] or [N, out_channels]
            attention_weights: [E] (如果return_attention_weights=True)
        """
        N = x.size(0)
        H = self.num_heads
        C = self.out_channels
        
        # 线性变换：[N, in] → [N, H*C] → [N, H, C]
        x_transformed = self.lin(x).view(N, H, C)
        
        # 添加自环（可选）
        if self.add_self_loops:
            edge_index = self._add_self_loops(edge_index, N)
        
        # 提取源节点和目标节点
        source_nodes = edge_index[0]  # [E]
        target_nodes = edge_index[1]  # [E]
        
        # 获取源和目标的特征
        x_source = x_transformed[source_nodes]  # [E, H, C]
        x_target = x_transformed[target_nodes]  # [E, H, C]
        
        # 拼接特征：[x_i || x_j]
        x_concat = torch.cat([x_source, x_target], dim=-1)  # [E, H, 2C]
        
        # 计算注意力分数：a^T [W h_i || W h_j]
        # att: [1, H, 2C], x_concat: [E, H, 2C]
        # 结果: [E, H]
        alpha = (x_concat * self.att).sum(dim=-1)  # [E, H]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Softmax归一化（按目标节点分组）
        alpha = self._softmax_per_target(alpha, target_nodes, N)  # [E, H]
        
        # Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 保存注意力权重（用于可视化和损失计算）
        self.last_attention_weights = alpha.detach()
        
        # 消息传递：h_i' = Σ_j α_ij W h_j
        out = torch.zeros(N, H, C, device=x.device)
        for h in range(H):
            # 对每个头单独计算
            messages = alpha[:, h].unsqueeze(-1) * x_source[:, h, :]  # [E, C]
            # 聚合到目标节点
            out[:, h, :].index_add_(0, target_nodes, messages)
        
        # 拼接或平均多头输出
        if self.concat:
            out = out.view(N, H * C)  # [N, H*C]
        else:
            out = out.mean(dim=1)  # [N, C]
        
        # 添加bias
        out = out + self.bias
        
        if return_attention_weights:
            # 返回每条边的平均注意力权重（跨所有头）
            return out, alpha.mean(dim=1)  # [N, H*C or C], [E]
        else:
            return out, None
    
    def _add_self_loops(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """添加自环"""
        device = edge_index.device
        self_loops = torch.arange(num_nodes, device=device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        return edge_index
    
    def _softmax_per_target(
        self,
        alpha: torch.Tensor,
        target_nodes: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        按目标节点分组计算softmax
        
        Args:
            alpha: [E, H] - 原始注意力分数
            target_nodes: [E] - 目标节点索引
            num_nodes: 节点总数
        
        Returns:
            alpha_normalized: [E, H] - 归一化后的注意力权重
        """
        # 对每个头分别计算
        H = alpha.size(1)
        alpha_normalized = torch.zeros_like(alpha)
        
        for h in range(H):
            # 对每个目标节点，计算其入边的softmax
            for target in range(num_nodes):
                mask = (target_nodes == target)
                if mask.any():
                    alpha_normalized[mask, h] = F.softmax(alpha[mask, h], dim=0)
        
        return alpha_normalized


class LocalGATEncoder(nn.Module):
    """
    局部GAT编码器（多层GAT堆叠）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.3.1
    
    架构：
        Input → LayerNorm → GAT1 → GELU+Residual → 
                              GAT2 → GELU+Residual → 
                              GAT3 → GELU+Residual → Output
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        初始化局部GAT编码器
        
        Args:
            input_dim: 输入特征维度（默认10）
            hidden_dim: 隐藏层维度（默认128）
            num_layers: GAT层数（默认3）
            num_heads: 注意力头数（默认8）
            dropout: Dropout概率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # GAT层堆叠
        self.gat_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            # 每层输入输出都是hidden_dim
            # 多头输出拼接后经过线性层降维回hidden_dim
            gat_layer = MultiHeadGATLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,  # 每个头的输出维度
                num_heads=num_heads,
                concat=True,  # 拼接多头
                dropout=dropout,
                add_self_loops=True,
            )
            self.gat_layers.append(gat_layer)
        
        # 每层后的LayerNorm
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        前向传播
        
        Args:
            x: [N, input_dim] - 节点特征
            edge_index: [2, E] - 边索引
            return_attention_weights: 是否返回所有层的注意力权重
        
        Returns:
            h: [N, hidden_dim] - 编码后的节点表征
            attention_weights: List[Tensor] - 每层的注意力权重
        """
        # 输入投影 + LayerNorm
        h = self.input_proj(x)
        h = self.input_norm(h)
        
        attention_weights_list = []
        
        # 多层GAT
        for layer_idx, gat_layer in enumerate(self.gat_layers):
            # GAT层
            h_new, attn_weights = gat_layer(
                h, edge_index, return_attention_weights=True
            )
            
            # GELU激活
            h_new = F.gelu(h_new)
            
            # Dropout
            h_new = self.dropout(h_new)
            
            # 残差连接（关键：防止梯度消失）
            h = h_new + h
            
            # LayerNorm
            h = self.layer_norms[layer_idx](h)
            
            if return_attention_weights:
                attention_weights_list.append(attn_weights)
        
        if return_attention_weights:
            return h, attention_weights_list
        else:
            return h, None


# ============ 导出接口 ============
__all__ = [
    'MultiHeadGATLayer',
    'LocalGATEncoder',
]



