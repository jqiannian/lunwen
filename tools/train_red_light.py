"""
红灯停异常检测训练编排器（修正版）

设计依据：Design-ITER-2025-01.md v2.0 §3.5.1
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.traffic_rules.utils.schedulers import WarmupCosineScheduler
from tqdm import tqdm
import typer
from rich.console import Console
from rich.table import Table
from typing import Optional, List, Dict
import json
from datetime import datetime
import numpy as np

from src.traffic_rules.data.traffic_dataset import TrafficLightDataset
from src.traffic_rules.graph.builder import GraphBuilder
from src.traffic_rules.models.multi_stage_gat import MultiStageAttentionGAT
from src.traffic_rules.loss.constraint import StagedConstraintLoss
from src.traffic_rules.rules.red_light import RedLightRuleEngine
from src.traffic_rules.monitoring.gradient_monitor import GradientMonitor
from src.traffic_rules.monitoring.metrics import compute_full_metrics
from src.traffic_rules.monitoring.visualizer import TrainingVisualizer

app = typer.Typer()
console = Console()


def scene_collate_fn(batch):
    """自定义collate函数（处理SceneContext对象）"""
    return batch[0] if len(batch) == 1 else batch


class Trainer:
    """训练编排器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        device: str = 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        epochs: int = 50,
        checkpoint_dir: str = 'artifacts/checkpoints',
    ):
        """初始化训练器"""
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # 学习率调度器（Warmup + Cosine）
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=min(10, epochs // 5),  # Warmup为总epochs的20%，最多10
            total_epochs=epochs,
            min_lr=1e-6,
        )
        
        # 损失函数
        self.criterion = StagedConstraintLoss()
        
        # 规则引擎
        self.rule_engine = RedLightRuleEngine()
        
        # 图构建器
        self.graph_builder = GraphBuilder()
        
        # 梯度监控器
        self.grad_monitor = GradientMonitor()
        
        # 可视化器
        self.visualizer = TrainingVisualizer(save_dir='reports')
        
        # 训练状态
        self.current_epoch = 0
        self.current_stage = 1
        self.best_val_loss = float('inf')
        
        # Checkpoint目录
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'loss_recon': [],
            'loss_rule': [],
            'loss_attn': [],
            'grad_norms': [],
            'lr': [],
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'rule_consistency': [],
            'attention_focus': [],
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'rule': 0.0,
            'attn': 0.0,
        }
        
        num_batches = 0
        
        pbar = tqdm(range(len(self.train_dataset)), desc=f"Epoch {self.current_epoch}", leave=False)
        
        for idx in pbar:
            # 直接获取单个场景（避免DataLoader collate问题）
            scene = self.train_dataset[idx]
            
            # 构建场景图
            graph = self.graph_builder.build(scene)
            
            # 转移到设备
            x = graph.x.to(self.device)
            edge_index = graph.edge_index.to(self.device)
            entity_types = graph.entity_types.to(self.device)
            
            # 跳过无边的图
            if edge_index.size(1) == 0:
                continue
            
            # 前向传播
            output = self.model(
                x, edge_index, entity_types,
                return_attention=True,
            )
            
            model_scores = output['scores']
            alpha_gat = output['gat_attention']
            beta_rule = output['rule_attention']
            
            # 获取车辆
            car_entities = scene.get_entities_by_type('car')
            
            if len(car_entities) == 0:
                continue
            
            # 提取规则相关特征
            light_probs = self._get_light_probs(scene.entities).to(self.device)
            distances = torch.tensor([e.d_stop for e in car_entities], device=self.device)
            velocities = torch.tensor([e.velocity for e in car_entities], device=self.device)
            
            rule_scores = self.rule_engine.evaluate(
                light_probs, distances, velocities, training=True
            )
            
            # 计算损失
            loss_total, loss_dict = self.criterion(
                model_scores=model_scores,
                rule_scores=rule_scores,
                alpha_gat=alpha_gat,
                beta_rule=beta_rule,
                edge_index=edge_index,
                entity_types=entity_types,
                model_parameters=list(self.model.parameters()),
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss_total.backward()
            
            # 梯度监控（在裁剪前）
            grad_stats = self.grad_monitor.monitor_step(self.model, num_batches)
            
            # 检查异常
            if grad_stats['anomalies']:
                for anomaly in grad_stats['anomalies']:
                    console.print(f"[yellow]{anomaly}[/yellow]")
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip,
            )
            
            self.optimizer.step()
            
            # 累积损失
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
            
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{loss_total.item():.4f}"})
        
        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        # 获取梯度监控摘要
        grad_summary = self.grad_monitor.get_summary()
        epoch_losses['grad_norm'] = grad_summary.get('grad_norm_mean', 0.0)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """验证集评估（含完整指标）"""
        if self.val_dataset is None:
            return {'loss': 0.0}
        
        self.model.eval()
        
        val_loss = 0.0
        num_batches = 0
        
        # 收集所有预测用于指标计算
        all_model_scores = []
        all_rule_scores = []
        all_attention_weights = []
        all_entity_types = []
        all_edge_indices = []
        
        with torch.no_grad():
            for idx in range(len(self.val_dataset)):
                scene = self.val_dataset[idx]
                graph = self.graph_builder.build(scene)
                
                x = graph.x.to(self.device)
                edge_index = graph.edge_index.to(self.device)
                entity_types = graph.entity_types.to(self.device)
                
                if edge_index.size(1) == 0:
                    continue
                
                output = self.model(x, edge_index, entity_types, return_attention=True)
                
                car_entities = scene.get_entities_by_type('car')
                if len(car_entities) == 0:
                    continue
                
                light_probs = self._get_light_probs(scene.entities).to(self.device)
                distances = torch.tensor([e.d_stop for e in car_entities], device=self.device)
                velocities = torch.tensor([e.velocity for e in car_entities], device=self.device)
                
                rule_scores = self.rule_engine.evaluate(
                    light_probs, distances, velocities, training=False
                )
                
                loss_total, _ = self.criterion(
                    model_scores=output['scores'],
                    rule_scores=rule_scores,
                    alpha_gat=output['gat_attention'],
                    beta_rule=output['rule_attention'],
                    edge_index=edge_index,
                    entity_types=entity_types,
                    model_parameters=list(self.model.parameters()),
                )
                
                val_loss += loss_total.item()
                num_batches += 1
                
                # 收集分数用于指标计算
                all_model_scores.append(output['scores'])
                all_rule_scores.append(rule_scores)
                all_attention_weights.append(output['gat_attention'])
                all_entity_types.append(entity_types)
                all_edge_indices.append(edge_index)
        
        avg_loss = val_loss / max(num_batches, 1)
        
        # 计算完整指标
        if len(all_model_scores) > 0:
            model_scores_cat = torch.cat(all_model_scores)
            rule_scores_cat = torch.cat(all_rule_scores)
            
            # 使用第一个图的attention（简化，实际应该合并）
            full_metrics = compute_full_metrics(
                model_scores=model_scores_cat,
                rule_scores=rule_scores_cat,
                attention_weights=all_attention_weights[0],
                entity_types=all_entity_types[0],
                edge_index=all_edge_indices[0],
                threshold=0.7,
            )
            
            full_metrics['loss'] = avg_loss
        else:
            full_metrics = {'loss': avg_loss}
        
        return full_metrics
    
    def _check_training_health(self, epoch: int, train_metrics: Dict) -> bool:
        """
        检查训练健康状况
        
        检测：
            - Loss振荡
            - 梯度异常
            - 验证集退化
        
        Returns:
            is_healthy: 是否健康
        """
        warnings = []
        
        # 检测Loss振荡（最近3个epochs）
        if len(self.history['train_loss']) >= 3:
            recent_losses = self.history['train_loss'][-3:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            if loss_std > 0.2 * loss_mean:  # 标准差超过均值的20%
                warnings.append(f"⚠️ Loss振荡: std={loss_std:.4f}, mean={loss_mean:.4f}")
        
        # 检测梯度异常
        grad_summary = self.grad_monitor.get_summary()
        if grad_summary.get('grad_explosion_count', 0) > 0:
            warnings.append(f"⚠️ 检测到梯度爆炸: {grad_summary['grad_explosion_count']}次")
        
        if grad_summary.get('grad_vanishing_count', 0) > 0:
            warnings.append(f"⚠️ 检测到梯度消失: {grad_summary['grad_vanishing_count']}次")
        
        # 检测验证集退化（最近3个验证点）
        if len(self.history['val_loss']) >= 3:
            recent_val = self.history['val_loss'][-3:]
            if all(recent_val[i] > recent_val[i-1] for i in range(1, len(recent_val))):
                warnings.append("⚠️ 验证Loss连续上升（可能过拟合）")
        
        # 输出警告
        if warnings:
            console.print(f"\n[yellow]{'='*60}[/yellow]")
            console.print(f"[bold yellow]训练健康检查 (Epoch {epoch})[/bold yellow]")
            for warning in warnings:
                console.print(f"[yellow]{warning}[/yellow]")
            console.print(f"[yellow]{'='*60}[/yellow]\n")
        
        return len(warnings) == 0
    
    def train(self):
        """完整训练流程"""
        console.print("\n[bold blue]🚀 开始训练：红灯停异常检测 MVP[/bold blue]")
        console.print(f"设备: {self.device}")
        console.print(f"总Epochs: {self.epochs}")
        console.print(f"训练场景: {len(self.train_dataset)}")
        if self.val_dataset:
            console.print(f"验证场景: {len(self.val_dataset)}")
        console.print()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 学习率调度
            self.scheduler.step()
            
            # 验证（每5个epoch）
            if epoch % 5 == 0 or epoch == self.epochs - 1:
                val_metrics = self.validate()
                
                # 打印指标
                self._print_metrics(epoch, train_metrics, val_metrics)
                
                # 健康检查
                self._check_training_health(epoch, train_metrics)
                
                # 保存checkpoint
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
            
            # 记录训练指标
            self.history['train_loss'].append(train_metrics['total'])
            self.history['loss_recon'].append(train_metrics.get('recon', 0.0))
            self.history['loss_rule'].append(train_metrics.get('rule', 0.0))
            self.history['loss_attn'].append(train_metrics.get('attn', 0.0))
            self.history['grad_norms'].append(train_metrics.get('grad_norm', 0.0))
            self.history['lr'].append(self.scheduler.get_last_lr()[0])
            
            # 记录验证指标
            if self.val_dataset:
                self.history['val_loss'].append(val_metrics.get('loss', 0.0))
                self.history['auc'].append(val_metrics.get('auc', 0.0))
                self.history['f1'].append(val_metrics.get('f1', 0.0))
                self.history['precision'].append(val_metrics.get('precision', 0.0))
                self.history['recall'].append(val_metrics.get('recall', 0.0))
                self.history['rule_consistency'].append(val_metrics.get('rule_consistency', 0.0))
                self.history['attention_focus'].append(val_metrics.get('attention_focus', 0.0))
        
        console.print("\n[bold green]✅ 训练完成！[/bold green]")
        
        # 生成可视化
        console.print("\n[cyan]生成训练曲线图...[/cyan]")
        try:
            curve_path = self.visualizer.plot_training_curves(self.history)
            console.print(f"[green]✅ 训练曲线已保存: {curve_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ 可视化失败: {e}[/yellow]")
        
        self._print_final_summary()
    
    def _get_light_probs(self, entities: List) -> torch.Tensor:
        """提取交通灯状态概率"""
        lights = [e for e in entities if e.type == 'light']
        
        if len(lights) == 0:
            return torch.tensor([[0.0, 0.0, 1.0]])
        
        light = lights[0]
        state_map = {'red': 0, 'yellow': 1, 'green': 2}
        probs = torch.zeros(1, 3)
        
        if hasattr(light, 'light_state') and light.light_state:
            idx = state_map.get(light.light_state, 2)
            probs[0, idx] = light.confidence if hasattr(light, 'confidence') else 0.9
            remaining = 1.0 - probs[0, idx]
            for j in range(3):
                if j != idx:
                    probs[0, j] = remaining / 2
        else:
            probs = torch.tensor([[0.0, 0.0, 1.0]])
        
        return probs
    
    def _print_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """打印训练指标"""
        table = Table(title=f"Epoch {epoch} / {self.epochs}")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Loss (Total)", f"{train_metrics['total']:.4f}")
        table.add_row("Loss (Recon)", f"{train_metrics['recon']:.4f}")
        table.add_row("Loss (Rule)", f"{train_metrics['rule']:.4f}")
        table.add_row("Loss (Attn)", f"{train_metrics['attn']:.4f}")
        table.add_row("Grad Norm", f"{train_metrics.get('grad_norm', 0.0):.4f}")
        
        if self.val_dataset:
            table.add_row("─" * 12, "─" * 8)  # 分隔线
            table.add_row("Val Loss", f"{val_metrics.get('loss', 0.0):.4f}")
            table.add_row("AUC", f"{val_metrics.get('auc', 0.0):.4f}")
            table.add_row("F1 Score", f"{val_metrics.get('f1', 0.0):.4f}")
            table.add_row("Precision", f"{val_metrics.get('precision', 0.0):.4f}")
            table.add_row("Recall", f"{val_metrics.get('recall', 0.0):.4f}")
            table.add_row("Rule Cons.", f"{val_metrics.get('rule_consistency', 0.0):.4f}")
            table.add_row("Attn Focus", f"{val_metrics.get('attention_focus', 0.0):.4f}")
        
        table.add_row("─" * 12, "─" * 8)
        table.add_row("Stage", f"{self.current_stage}")
        table.add_row("LR", f"{self.scheduler.get_last_lr()[0]:.6f}")
        
        console.print(table)
    
    def _print_final_summary(self):
        """打印最终总结"""
        console.print("\n" + "="*60)
        console.print("[bold]训练总结[/bold]")
        console.print("="*60)
        console.print(f"总Epochs: {self.epochs}")
        console.print(f"最佳验证Loss: {self.best_val_loss:.4f}")
        console.print(f"Checkpoint保存在: {self.checkpoint_dir}")
        console.print("="*60)
    
    def save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        is_best: bool = False,
    ):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'stage': self.current_stage,
            'history': self.history,
        }
        
        # 保存当前epoch
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, path)
        
        # 如果是最佳，额外保存
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            console.print(f"[green]✅ 保存最佳模型: Val Loss={val_metrics.get('loss', 0.0):.4f}[/green]")


@app.command()
def train(
    data_root: str = typer.Option("data/synthetic", help="数据根目录"),
    epochs: int = typer.Option(50, help="训练轮数"),
    lr: float = typer.Option(1e-4, help="学习率"),
    device: str = typer.Option("cpu", help="设备: cpu/cuda"),
    checkpoint_dir: str = typer.Option("artifacts/checkpoints", help="Checkpoint目录"),
    max_samples: Optional[int] = typer.Option(None, help="最大样本数（调试用）"),
):
    """
    训练红灯停异常检测模型
    
    Example:
        python tools/train_red_light.py train --epochs 20 --device cpu
    """
    console.print("[bold blue]🔧 初始化训练环境...[/bold blue]")
    
    # 加载数据集
    console.print("📊 加载数据集...")
    try:
        train_dataset = TrafficLightDataset(
            data_root=data_root,
            mode='synthetic',
            split='train',
            max_samples=max_samples,
        )
    except FileNotFoundError as e:
        console.print(f"[red]❌ 错误: {e}[/red]")
        console.print("[yellow]请先运行: python3 scripts/prepare_synthetic_data.py --num-scenes 100[/yellow]")
        raise typer.Exit(1)
    
    try:
        val_dataset = TrafficLightDataset(
            data_root=data_root,
            mode='synthetic',
            split='val',
            max_samples=max_samples,
        )
    except FileNotFoundError:
        console.print("[yellow]⚠️  未找到验证集，仅使用训练集[/yellow]")
        val_dataset = None
    
    console.print(f"[green]✅ 训练集: {len(train_dataset)} 场景[/green]")
    if val_dataset:
        console.print(f"[green]✅ 验证集: {len(val_dataset)} 场景[/green]")
    
    # 初始化模型
    console.print("\n🤖 初始化模型...")
    model = MultiStageAttentionGAT(
        input_dim=10,
        hidden_dim=128,
        num_gat_layers=3,
        num_heads=8,
        dropout=0.1,
    )
    
    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]✅ 模型参数量: {num_params:,} (~{num_params/1e6:.2f}M)[/green]")
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        learning_rate=lr,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
    )
    
    # 开始训练
    console.print(f"\n[bold yellow]🚀 开始训练...[/bold yellow]\n")
    trainer.train()
    
    console.print(f"\n[bold green]✅ 训练完成！Checkpoint保存在: {checkpoint_dir}[/bold green]")


@app.command()
def info():
    """显示模型信息"""
    console.print("\n[bold]📋 模型信息[/bold]\n")
    console.print(f"设计文档: Design-ITER-2025-01.md v2.0")
    console.print(f"算法方案: 多阶段GAT + 硬约束规则融合\n")
    console.print(f"[cyan]模型架构:[/cyan]")
    console.print(f"  • 阶段1: 局部GAT（3层×8头）")
    console.print(f"  • 阶段2: 全局虚拟节点注意力（4头）")
    console.print(f"  • 阶段3: 规则聚焦注意力")
    console.print(f"  • 输入维度: 10")
    console.print(f"  • 隐藏维度: 128")
    console.print(f"  • 总参数量: ~1.02M\n")


if __name__ == "__main__":
    app()
