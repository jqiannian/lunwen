"""记忆库组件实现。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


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
    
    def initialize_from_data(
        self,
        model: torch.nn.Module,
        dataloader: "DataLoader",
        device: str = "cpu",
        max_samples: int | None = None,
    ) -> None:
        """
        使用K-Means从正常样本初始化记忆库
        
        Args:
            model: 模型（用于提取embedding）
            dataloader: 数据加载器
            device: 设备
            max_samples: 最大样本数（可选，用于加速）
        """
        from sklearn.cluster import KMeans
        
        print(f"[MemoryBank] 开始K-Means初始化 (size={self.size}, dim={self.embedding_dim})")
        
        # 收集所有正常样本的embedding
        embeddings_list = []
        sample_count = 0
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_samples and sample_count >= max_samples:
                    break
                
                # 这里假设batch是SceneContext对象或包含SceneContext的dict
                # 需要通过模型提取embedding
                # 简化处理：假设model有encode方法或直接前向传播获取embedding
                
                try:
                    # 尝试直接调用model的encode方法（如果有）
                    if hasattr(model, 'encode'):
                        emb = model.encode(batch)
                    else:
                        # 否则，使用模型前向传播的中间层输出
                        # 这里需要根据实际模型结构调整
                        # 暂时跳过这个batch
                        continue
                    
                    if emb is not None:
                        embeddings_list.append(emb.cpu())
                        sample_count += emb.size(0)
                
                except Exception as e:
                    print(f"[MemoryBank] 警告：处理batch {batch_idx}时出错: {e}")
                    continue
        
        if not embeddings_list:
            print("[MemoryBank] 警告：未收集到任何embedding，使用随机初始化")
            self.storage = torch.randn(self.size, self.embedding_dim) * 0.01
            return
        
        # 合并所有embedding
        embeddings = torch.cat(embeddings_list, dim=0)  # [N, D]
        print(f"[MemoryBank] 收集到 {embeddings.size(0)} 个embedding")
        
        # 如果样本数少于记忆槽数，使用重复采样
        if embeddings.size(0) < self.size:
            print(f"[MemoryBank] 警告：样本数 ({embeddings.size(0)}) < size ({self.size})，将使用重复采样")
            # 重复采样直到足够
            repeat_times = (self.size // embeddings.size(0)) + 1
            embeddings = embeddings.repeat(repeat_times, 1)[:self.size]
        
        # K-Means聚类
        print(f"[MemoryBank] 执行K-Means聚类...")
        kmeans = KMeans(
            n_clusters=self.size,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        
        embeddings_np = embeddings.numpy()
        kmeans.fit(embeddings_np)
        
        # 用聚类中心初始化记忆槽
        centers = torch.from_numpy(kmeans.cluster_centers_).float()  # [size, D]
        
        # L2归一化
        self.storage = F.normalize(centers, dim=-1)
        
        print(f"[MemoryBank] ✅ K-Means初始化完成")
        print(f"[MemoryBank]   - 聚类中心数: {self.size}")
        print(f"[MemoryBank]   - Inertia: {kmeans.inertia_:.2f}")
    
    def update(self, embeddings: Tensor, ema_decay: float = 0.9) -> None:
        """
        使用指数移动平均（EMA）更新记忆库
        
        Args:
            embeddings: 新的embedding向量 [B, D]
            ema_decay: EMA衰减系数（越大越保守）
        """
        if embeddings.size(0) == 0:
            return
        
        # 找到每个embedding最相似的记忆槽
        normalized_storage = F.normalize(self.storage, dim=-1)
        normalized_embeddings = F.normalize(embeddings, dim=-1)
        
        # 计算相似度 [B, size]
        similarities = torch.matmul(normalized_embeddings, normalized_storage.T)
        closest_indices = similarities.argmax(dim=1)  # [B]
        
        # EMA更新对应的记忆槽
        for i, idx in enumerate(closest_indices):
            idx = int(idx.item())
            self.storage[idx] = (
                ema_decay * self.storage[idx] + 
                (1 - ema_decay) * embeddings[i]
            )
            # L2归一化保持向量在单位球面上
            self.storage[idx] = F.normalize(self.storage[idx], dim=0)
