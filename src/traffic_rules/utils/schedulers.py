"""
学习率调度器

包含Warmup + CosineAnnealing组合调度器
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup + Cosine Annealing 学习率调度器
    
    阶段1（Warmup）: 线性增长 (0 → base_lr)
    阶段2（Cosine）: 余弦衰减 (base_lr → min_lr)
    
    设计目的：
        - 前期Warmup避免梯度爆炸
        - 后期平滑衰减避免振荡
    
    Args:
        optimizer: PyTorch优化器
        warmup_epochs: Warmup轮数（默认10）
        total_epochs: 总训练轮数
        min_lr: 最小学习率（默认1e-6）
        last_epoch: 上次epoch（用于恢复训练）
    
    Example:
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler = WarmupCosineScheduler(
            optimizer, 
            warmup_epochs=10, 
            total_epochs=100,
            min_lr=1e-6
        )
        
        for epoch in range(100):
            train(...)
            scheduler.step()
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        
        # 保存基础学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """计算当前epoch的学习率"""
        epoch = self.last_epoch
        
        lrs = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                # Warmup阶段：线性增长
                lr = base_lr * (epoch + 1) / self.warmup_epochs
            else:
                # Cosine Annealing阶段
                progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            
            lrs.append(lr)
        
        return lrs


# 导出接口
__all__ = ['WarmupCosineScheduler']
