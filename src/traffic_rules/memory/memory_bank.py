"""记忆库组件实现。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor


class MemoryBank:
    """保存正常模式原型，并在推理过程中提供检索能力。"""

    def __init__(self, size: int = 256, embedding_dim: int = 256) -> None:
        self.size = size
        self.embedding_dim = embedding_dim
        self.storage = torch.zeros(size, embedding_dim)

    def load(self, path: Path) -> None:
        """从 artifacts 载入记忆参数."""

        if not path.exists():
            raise FileNotFoundError(f"记忆库文件不存在: {path}")
        state = torch.load(path, map_location="cpu")
        if state.size() != (self.size, self.embedding_dim):
            raise ValueError("记忆库尺寸不匹配, 请确认 checkpoint 设置")
        self.storage = state

    def save(self, path: Path) -> None:
        """把记忆参数持久化到指定路径。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.storage, path)

    def query(self, embeddings: Tensor) -> Tensor:
        """接收节点向量并返回注意力上下文。"""

        if not torch.any(self.storage):
            return torch.zeros_like(embeddings)
        # 计算余弦相似度, 以 softmax 得到注意力权重
        normalized_storage = F.normalize(self.storage, dim=-1)
        normalized_embeddings = F.normalize(embeddings, dim=-1)
        similarity = torch.matmul(normalized_embeddings, normalized_storage.T)
        weights = torch.softmax(similarity, dim=-1)
        return torch.matmul(weights, self.storage)

    def reset(self) -> None:
        """清空存储，便于重新训练。"""

        self.storage.zero_()
