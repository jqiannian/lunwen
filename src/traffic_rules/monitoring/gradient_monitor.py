"""
梯度监控器

实时监控训练过程中的梯度流，检测异常
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class GradientMonitor:
    """
    梯度监控器
    
    功能：
        - 计算各层梯度范数
        - 检测梯度爆炸/消失
        - 记录权重更新量
        - 生成诊断报告
    
    使用方法：
        monitor = GradientMonitor()
        
        # 训练循环中
        loss.backward()
        stats = monitor.monitor_step(model, optimizer, step)
        if stats['anomalies']:
            print(f"警告: {stats['anomalies']}")
        optimizer.step()
    """
    
    def __init__(
        self,
        grad_explosion_threshold: float = 10.0,
        grad_vanishing_threshold: float = 1e-5,
        imbalance_ratio_threshold: float = 100.0,
    ):
        """
        初始化监控器
        
        Args:
            grad_explosion_threshold: 梯度爆炸阈值
            grad_vanishing_threshold: 梯度消失阈值
            imbalance_ratio_threshold: 梯度不平衡比例阈值
        """
        self.grad_explosion_threshold = grad_explosion_threshold
        self.grad_vanishing_threshold = grad_vanishing_threshold
        self.imbalance_ratio_threshold = imbalance_ratio_threshold
        
        # 历史记录
        self.grad_history = []
        self.weight_history = []
    
    def compute_grad_norms(self, model: nn.Module) -> Dict[str, float]:
        """
        计算各层梯度范数
        
        Args:
            model: PyTorch模型
        
        Returns:
            grad_norms: 各参数的梯度范数字典
        """
        grad_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
            else:
                grad_norms[name] = 0.0
        
        return grad_norms
    
    def compute_layer_stats(self, grad_norms: Dict[str, float]) -> Dict[str, float]:
        """
        按层分组统计梯度
        
        Args:
            grad_norms: 各参数的梯度范数
        
        Returns:
            layer_stats: 各层的平均梯度范数
        """
        layer_groups = defaultdict(list)
        
        # 分组
        for name, norm in grad_norms.items():
            # 提取层名称（如 'local_gat.gat_layers.0.W.0.weight' → 'local_gat'）
            layer_name = name.split('.')[0]
            layer_groups[layer_name].append(norm)
        
        # 计算统计量
        layer_stats = {}
        for layer_name, norms in layer_groups.items():
            layer_stats[layer_name] = {
                'mean': np.mean(norms),
                'max': np.max(norms),
                'min': np.min(norms),
                'std': np.std(norms),
            }
        
        return layer_stats
    
    def detect_anomalies(
        self,
        total_norm: float,
        layer_stats: Dict[str, Dict[str, float]],
    ) -> List[str]:
        """
        检测梯度异常
        
        Args:
            total_norm: 总梯度范数
            layer_stats: 各层统计量
        
        Returns:
            anomalies: 异常列表
        """
        anomalies = []
        
        # 检测梯度爆炸
        if total_norm > self.grad_explosion_threshold:
            anomalies.append(f"🔴 梯度爆炸: total_norm={total_norm:.2f} > {self.grad_explosion_threshold}")
        
        # 检测梯度消失
        if total_norm < self.grad_vanishing_threshold:
            anomalies.append(f"🔴 梯度消失: total_norm={total_norm:.2e} < {self.grad_vanishing_threshold}")
        
        # 检测各层梯度不平衡
        layer_means = [stats['mean'] for stats in layer_stats.values()]
        if len(layer_means) > 1:
            max_mean = max(layer_means)
            min_mean = min(layer_means) + 1e-10
            imbalance_ratio = max_mean / min_mean
            
            if imbalance_ratio > self.imbalance_ratio_threshold:
                anomalies.append(
                    f"⚠️ 梯度不平衡: max/min={imbalance_ratio:.1f} > {self.imbalance_ratio_threshold}"
                )
        
        # 检测各层内部是否有梯度消失
        for layer_name, stats in layer_stats.items():
            if stats['mean'] < self.grad_vanishing_threshold:
                anomalies.append(f"⚠️ {layer_name}层梯度消失: mean={stats['mean']:.2e}")
        
        return anomalies
    
    def compute_weight_updates(
        self,
        model: nn.Module,
        prev_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        计算权重更新量
        
        Args:
            model: PyTorch模型
            prev_weights: 上一步的权重（可选）
        
        Returns:
            update_norms: 各参数的更新量范数
        """
        if prev_weights is None:
            return {}
        
        update_norms = {}
        
        for name, param in model.named_parameters():
            if name in prev_weights:
                delta = param.data - prev_weights[name]
                update_norms[name] = delta.norm().item()
        
        return update_norms
    
    def monitor_step(
        self,
        model: nn.Module,
        step: int,
        prev_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, any]:
        """
        单步监控（在backward()后、optimizer.step()前调用）
        
        Args:
            model: PyTorch模型
            step: 当前步数
            prev_weights: 上一步权重（可选）
        
        Returns:
            stats: 监控统计信息
        """
        # 计算梯度范数
        grad_norms = self.compute_grad_norms(model)
        
        # 总梯度范数
        total_norm = np.sqrt(sum(v**2 for v in grad_norms.values() if v > 0))
        
        # 分层统计
        layer_stats = self.compute_layer_stats(grad_norms)
        
        # 异常检测
        anomalies = self.detect_anomalies(total_norm, layer_stats)
        
        # 权重更新量（如果提供了前一步权重）
        update_norms = self.compute_weight_updates(model, prev_weights)
        
        # 组装统计信息
        stats = {
            'step': step,
            'total_norm': total_norm,
            'layer_stats': layer_stats,
            'grad_norms': grad_norms,
            'update_norms': update_norms,
            'anomalies': anomalies,
        }
        
        # 记录历史
        self.grad_history.append({
            'step': step,
            'total_norm': total_norm,
            'layer_means': {k: v['mean'] for k, v in layer_stats.items()},
        })
        
        return stats
    
    def get_summary(self) -> Dict[str, any]:
        """
        获取监控总结
        
        Returns:
            summary: 统计摘要
        """
        if len(self.grad_history) == 0:
            return {}
        
        # 提取总梯度范数历史
        total_norms = [h['total_norm'] for h in self.grad_history]
        
        summary = {
            'num_steps': len(self.grad_history),
            'grad_norm_mean': np.mean(total_norms),
            'grad_norm_std': np.std(total_norms),
            'grad_norm_max': np.max(total_norms),
            'grad_norm_min': np.min(total_norms),
            'grad_explosion_count': sum(1 for n in total_norms if n > self.grad_explosion_threshold),
            'grad_vanishing_count': sum(1 for n in total_norms if n < self.grad_vanishing_threshold),
        }
        
        return summary
    
    def save_weights_snapshot(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        保存当前权重快照（用于下一步计算更新量）
        
        Args:
            model: PyTorch模型
        
        Returns:
            weights: 权重字典
        """
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.data.clone()
        return weights


# 导出接口
__all__ = ['GradientMonitor']
