"""
自训练调度器

功能：
1. 控制伪标签生成的时机
2. 动态调整置信度阈值（课程学习）
3. 管理自训练轮次
"""

from __future__ import annotations


class SelfTrainingScheduler:
    """
    自训练调度器
    
    管理自训练过程的时机和参数调整
    """
    
    def __init__(
        self,
        total_epochs: int,
        warmup_epochs: int = 10,
        pseudo_label_interval: int = 5,
        initial_threshold: float = 0.8,
        final_threshold: float = 0.6,
        strategy: str = "linear",
    ) -> None:
        """
        初始化调度器
        
        Args:
            total_epochs: 总训练轮数
            warmup_epochs: warmup轮数（在此之前不生成伪标签）
            pseudo_label_interval: 伪标签生成间隔（每N个epoch生成一次）
            initial_threshold: 初始置信度阈值（严格）
            final_threshold: 最终置信度阈值（宽松）
            strategy: 阈值衰减策略（linear/exponential/step）
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.pseudo_label_interval = pseudo_label_interval
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.strategy = strategy
        
        self.current_epoch = 0
        self._pseudo_label_count = 0
    
    def step(self) -> None:
        """前进一个epoch"""
        self.current_epoch += 1
    
    def should_generate_pseudo_labels(self) -> bool:
        """
        判断当前epoch是否应生成伪标签
        
        Returns:
            bool: 是否生成伪标签
        """
        # Warmup期间不生成
        if self.current_epoch < self.warmup_epochs:
            return False
        
        # 检查是否到达间隔
        epochs_since_warmup = self.current_epoch - self.warmup_epochs
        return epochs_since_warmup % self.pseudo_label_interval == 0
    
    def get_confidence_threshold(self) -> float:
        """
        获取当前epoch的置信度阈值（课程学习）
        
        随着训练进行，从严格（high）到宽松（low）
        
        Returns:
            threshold: 当前置信度阈值 [0, 1]
        """
        if self.current_epoch < self.warmup_epochs:
            return self.initial_threshold
        
        # 计算进度 [0, 1]
        progress = (self.current_epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )
        progress = min(1.0, max(0.0, progress))
        
        # 根据策略计算阈值
        if self.strategy == "linear":
            # 线性衰减
            threshold = self.initial_threshold - progress * (
                self.initial_threshold - self.final_threshold
            )
        
        elif self.strategy == "exponential":
            # 指数衰减
            import math
            decay_rate = math.log(self.final_threshold / self.initial_threshold)
            threshold = self.initial_threshold * math.exp(decay_rate * progress)
        
        elif self.strategy == "step":
            # 阶梯衰减（每1/3进度降低一次）
            if progress < 0.33:
                threshold = self.initial_threshold
            elif progress < 0.67:
                threshold = (self.initial_threshold + self.final_threshold) / 2
            else:
                threshold = self.final_threshold
        
        else:
            # 默认常量
            threshold = self.initial_threshold
        
        return threshold
    
    def on_pseudo_labels_generated(self, num_labels: int) -> None:
        """
        记录伪标签生成事件
        
        Args:
            num_labels: 生成的伪标签数量
        """
        self._pseudo_label_count += num_labels
    
    def get_stats(self) -> dict[str, Any]:
        """
        获取调度器统计信息
        
        Returns:
            stats: 统计字典
        """
        return {
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "in_warmup": self.current_epoch < self.warmup_epochs,
            "current_threshold": self.get_confidence_threshold(),
            "pseudo_labels_generated": self._pseudo_label_count,
            "progress": self.current_epoch / self.total_epochs,
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"SelfTrainingScheduler("
            f"epoch={stats['current_epoch']}/{stats['total_epochs']}, "
            f"threshold={stats['current_threshold']:.3f}, "
            f"warmup={'Yes' if stats['in_warmup'] else 'No'})"
        )


# 类型提示修复
from typing import Any
