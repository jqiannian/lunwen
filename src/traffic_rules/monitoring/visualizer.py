"""
训练可视化工具
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


class TrainingVisualizer:
    """
    训练过程可视化器
    
    功能：
        - 绘制Loss曲线（训练+验证）
        - 绘制学习率曲线
        - 绘制梯度范数曲线
        - 绘制验证指标曲线
    """
    
    def __init__(self, save_dir: str = 'reports'):
        """
        初始化可视化器
        
        Args:
            save_dir: 图片保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_name: str = 'training_curves.png',
    ):
        """
        绘制训练曲线（4子图）
        
        Args:
            history: 训练历史字典
                - epochs: epoch列表
                - train_loss: 训练loss列表
                - val_loss: 验证loss列表（可选）
                - grad_norms: 梯度范数列表（可选）
                - lr: 学习率列表（可选）
                - auc, f1等验证指标（可选）
            save_name: 保存文件名
        """
        # 提取epoch列表
        epochs = list(range(len(history['train_loss'])))
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # ====== 子图1: Total Loss ======
        ax1 = axes[0, 0]
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        
        if 'val_loss' in history and len(history['val_loss']) > 0:
            # 验证loss可能不是每个epoch都有
            val_epochs = [i for i, v in enumerate(history['val_loss']) if v > 0]
            val_losses = [history['val_loss'][i] for i in val_epochs]
            ax1.plot(val_epochs, val_losses, 'r--', label='Val Loss', linewidth=2, marker='o')
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Total Loss Curve', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ====== 子图2: Loss分解 ======
        ax2 = axes[0, 1]
        
        if 'loss_recon' in history:
            ax2.plot(epochs, history.get('loss_recon', []), label='L_recon', linewidth=1.5)
        if 'loss_rule' in history:
            ax2.plot(epochs, history.get('loss_rule', []), label='L_rule', linewidth=1.5)
        if 'loss_attn' in history:
            ax2.plot(epochs, history.get('loss_attn', []), label='L_attn', linewidth=1.5)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # ====== 子图3: 学习率 + 梯度范数（双Y轴） ======
        ax3 = axes[1, 0]
        
        if 'lr' in history and len(history['lr']) > 0:
            color_lr = 'tab:green'
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Learning Rate', color=color_lr, fontsize=12)
            ax3.plot(epochs, history['lr'], color=color_lr, linewidth=2, label='LR')
            ax3.tick_params(axis='y', labelcolor=color_lr)
            ax3.set_title('Learning Rate & Gradient Norm', fontsize=14, fontweight='bold')
            
            # 创建第二个Y轴
            if 'grad_norms' in history and len(history['grad_norms']) > 0:
                ax3_twin = ax3.twinx()
                color_grad = 'tab:orange'
                ax3_twin.set_ylabel('Gradient Norm', color=color_grad, fontsize=12)
                ax3_twin.plot(epochs, history['grad_norms'], color=color_grad, 
                             linewidth=2, linestyle='--', label='Grad Norm')
                ax3_twin.tick_params(axis='y', labelcolor=color_grad)
            
            ax3.grid(True, alpha=0.3)
        
        # ====== 子图4: 验证指标 ======
        ax4 = axes[1, 1]
        
        has_metrics = False
        if 'auc' in history and len(history['auc']) > 0:
            metric_epochs = list(range(len(history['auc'])))
            ax4.plot(metric_epochs, history['auc'], 'o-', label='AUC', linewidth=2)
            has_metrics = True
        
        if 'f1' in history and len(history['f1']) > 0:
            metric_epochs = list(range(len(history['f1'])))
            ax4.plot(metric_epochs, history['f1'], 's-', label='F1 Score', linewidth=2)
            has_metrics = True
        
        if 'rule_consistency' in history and len(history['rule_consistency']) > 0:
            metric_epochs = list(range(len(history['rule_consistency'])))
            ax4.plot(metric_epochs, history['rule_consistency'], '^-', 
                    label='Rule Consistency', linewidth=2)
            has_metrics = True
        
        if 'attention_focus' in history and len(history['attention_focus']) > 0:
            metric_epochs = list(range(len(history['attention_focus'])))
            ax4.plot(metric_epochs, history['attention_focus'], 'd-',
                    label='Attention Focus', linewidth=2)
            has_metrics = True
        
        if has_metrics:
            ax4.set_xlabel('Validation Checkpoint', fontsize=12)
            ax4.set_ylabel('Score', fontsize=12)
            ax4.set_title('Validation Metrics', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.set_ylim([0, 1.05])
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No validation metrics yet', 
                    ha='center', va='center', fontsize=14)
            ax4.set_title('Validation Metrics', fontsize=14, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_gradient_flow(
        self,
        grad_stats: Dict[str, Dict[str, float]],
        save_name: str = 'gradient_flow.png',
    ):
        """
        绘制梯度流图（各层梯度范数）
        
        Args:
            grad_stats: 梯度统计信息
            save_name: 保存文件名
        """
        layers = list(grad_stats.keys())
        means = [stats['mean'] for stats in grad_stats.values()]
        maxs = [stats['max'] for stats in grad_stats.values()]
        mins = [stats['min'] for stats in grad_stats.values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(layers))
        width = 0.6
        
        # 绘制均值柱状图
        bars = ax.bar(x, means, width, label='Mean', alpha=0.8)
        
        # 添加误差线（min-max范围）
        errors = [[means[i] - mins[i] for i in range(len(layers))],
                  [maxs[i] - means[i] for i in range(len(layers))]]
        ax.errorbar(x, means, yerr=errors, fmt='none', ecolor='black', 
                   capsize=5, alpha=0.5)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Flow by Layer', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path


# 导出接口
__all__ = ['TrainingVisualizer']
