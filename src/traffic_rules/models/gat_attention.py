"""注意力增强 GNN 模型实现。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from traffic_rules.graph.builder import GraphBatch
    from traffic_rules.memory.memory_bank import MemoryBank


class GATAttention(nn.Module):
    """多头 GAT + 外部记忆融合, 输出节点得分与注意力权重."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        memory_bank: MemoryBank | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.linear = nn.Linear(input_dim, hidden_dim * heads, bias=False)
        self.out_proj = nn.Linear(hidden_dim * heads, hidden_dim)
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.memory_bank = memory_bank

    def forward(self, batch: GraphBatch) -> tuple[Tensor, Tensor]:
        """执行多头 GAT 并融合 MemoryBank, 返回 (scores, attention_heatmap)."""

        features = batch.features  # [N, input_dim]
        adjacency = batch.adjacency  # [N, N]

        node_states = self._multi_head_attention(features, adjacency)
        fused_states = self._inject_memory(node_states)
        scores = self.classifier(fused_states).squeeze(-1)

        attention_heatmap = self._build_attention_heatmap(node_states, adjacency)
        return scores, attention_heatmap

    def _multi_head_attention(self, features: Tensor, adjacency: Tensor) -> Tensor:
        """自定义多头 GAT 层, 利用邻接矩阵控制注意力范围。"""

        num_nodes = features.size(0)
        projected = self.linear(features)  # [N, hidden_dim*heads]
        projected = projected.view(num_nodes, self.heads, self.hidden_dim)
        projected = projected.permute(1, 0, 2)  # [heads, N, hidden_dim]

        attention_logits = torch.matmul(projected, projected.transpose(1, 2))
        attention_logits = attention_logits / math.sqrt(self.hidden_dim)
        mask = adjacency.unsqueeze(0).bool()
        attention_logits = attention_logits.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(attention_logits, dim=-1)

        attended = torch.matmul(attention_weights, projected)  # [heads, N, hidden_dim]
        attended = attended.permute(1, 0, 2).reshape(num_nodes, self.heads * self.hidden_dim)
        return self.out_proj(attended)

    def _inject_memory(self, node_states: Tensor) -> Tensor:
        """调用 MemoryBank.query 融合正常模式上下文."""

        if self.memory_bank is None:
            return torch.cat([node_states, torch.zeros_like(node_states)], dim=-1)

        memory_context = self.memory_bank.query(node_states)
        return torch.cat([node_states, memory_context], dim=-1)

    def _build_attention_heatmap(self, node_states: Tensor, adjacency: Tensor) -> Tensor:
        """构造可复用的注意力权重矩阵, 供可解释性模块使用。"""

        similarity = F.cosine_similarity(
            node_states.unsqueeze(1), node_states.unsqueeze(0), dim=-1
        )
        mask = adjacency.bool()
        similarity = similarity.masked_fill(~mask, 0.0)
        return similarity / (similarity.sum(dim=-1, keepdim=True) + 1e-8)
