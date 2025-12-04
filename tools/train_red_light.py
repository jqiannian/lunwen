"""
红灯停异常检测训练编排器

基于Design-ITER-2025-01.md v2.0 §3.5.1, §3.7.3设计
支持三阶段训练流程：规则监督→混合训练→自训练
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
from tqdm import tqdm
import typer
from rich.console import Console
from rich.table import Table
from typing import Optional
import json
from datetime import datetime

from src.traffic_rules.data.traffic_dataset import TrafficLightDataset
from src.traffic_rules.graph.builder import GraphBuilder
from src.traffic_rules.models.multi_stage_gat import MultiStageAttentionGAT
from src.traffic_rules.loss.constraint import (
    StagedConstraintLoss,
    compute_model_reliability,
    should_switch_stage,
)
from src.traffic_rules.rules.red_light import RedLightRuleEngine, RuleConfig

app = typer.Typer()
console = Console()


class Trainer:
    """
    训练编排器
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.5.1, §3.7.3
    
    支持三阶段训练：
        Stage 1 (Epoch 0-20): 纯规则监督，λ_rule=0.5
        Stage 2 (Epoch 20-60): 混合训练，λ_rule=0.2
        Stage 3 (Epoch 60+): 自训练为主，λ_rule=0.1
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        epochs: int = 100,
        checkpoint_dir: str = 'artifacts/checkpoints',
    ):
        """
        初始化训练器
        
        Args:
            model: 多阶段GAT模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备（'cpu' or 'cuda'）
            learning_rate: 学习率
            weight_decay: 权重衰减
            grad_clip: 梯度裁剪阈值
            epochs: 训练轮数
            checkpoint_dir: Checkpoint保存目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
        )
        
        # 分阶段约束损失
        self.criterion = StagedConstraintLoss()
        
        # 规则引擎
        self.rule_engine = RedLightRuleEngine()
        
        # 图构建器
        self.graph_builder = GraphBuilder()
        
        # 训练状态
        self.current_epoch = 0
        self.current_stage = 1
        self.best_val_auc = 0.0
        
        # Checkpoint目录
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_auc': [],
            'model_reliability': [],
            'stage_switches': [],
        }
    
    def train_epoch(self) -> dict:
        """
        训练一个epoch
        
        Returns:
            metrics: 训练指标字典
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'rule': 0.0,
            'attn': 0.0,
            'attn_gat': 0.0,
            'attn_rule': 0.0,
            'reg': 0.0,
        }
        
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", leave=False):
            # 构建场景图
            graphs = self.graph_builder.build_batch(batch)
            
            # 合并为单个大图（简化，实际可以使用PyG的Batch）
            # 这里假设每个batch只有一个场景
            graph = graphs[0]
            
            # 转移到设备
            x = graph.x.to(self.device)
            edge_index = graph.edge_index.to(self.device)
            entity_types = graph.entity_types.to(self.device)
            
            # 前向传播
            output = self.model(
                x, edge_index, entity_types,
                return_attention=True,
            )
            
            model_scores = output['scores']
            alpha_gat = output['gat_attention']
            beta_rule = output['rule_attention']
            
            # 计算规则分数
            car_mask = (entity_types == 0)
            car_entities = [e for e in graph.entities if e.type == 'car']
            
            # 提取车辆的规则相关特征
            light_probs = self._get_light_probs(graph.entities).to(self.device)
            distances = torch.tensor([
                e.d_stop if hasattr(e, 'd_stop') else 999.0
                for e in car_entities
            ], device=self.device)
            velocities = torch.tensor([
                e.velocity for e in car_entities
            ], device=self.device)
            
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
        
        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def validate(self) -> dict:
        """
        验证集评估
        
        Returns:
            metrics: 验证指标
        """
        self.model.eval()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                graphs = self.graph_builder.build_batch(batch)
                graph = graphs[0]
                
                x = graph.x.to(self.device)
                edge_index = graph.edge_index.to(self.device)
                entity_types = graph.entity_types.to(self.device)
                
                output = self.model(x, edge_index, entity_types)
                
                # 收集预测和标签
                all_scores.append(output['scores'].cpu())
                
                # 使用规则分数作为伪标签
                car_entities = [e for e in graph.entities if e.type == 'car']
                light_probs = self._get_light_probs(graph.entities)
                distances = torch.tensor([e.d_stop if hasattr(e, 'd_stop') else 999.0 for e in car_entities])
                velocities = torch.tensor([e.velocity for e in car_entities])
                
                rule_scores = self.rule_engine.evaluate(
                    light_probs, distances, velocities, training=False
                )
                all_labels.append(rule_scores.cpu())
        
        # 合并所有batch
        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        
        # 计算指标
        auc = self._compute_auc(all_scores, all_labels > 0.5)
        f1 = self._compute_f1(all_scores > 0.7, all_labels > 0.5)
        rule_consistency = 1.0 - torch.mean(torch.abs(all_scores - all_labels)).item()
        
        return {
            'auc': auc,
            'f1': f1,
            'rule_consistency': rule_consistency,
        }
    
    def train(self):
        """
        完整训练流程（三阶段）
        """
        console.print("\n[bold blue]开始训练：红灯停异常检测[/bold blue]")
        console.print(f"设备: {self.device}")
        console.print(f"总Epochs: {self.epochs}")
        console.print(f"当前阶段: Stage {self.current_stage}\n")
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 学习率调度
            self.scheduler.step()
            
            # 验证（每5个epoch）
            if epoch % 5 == 0:
                val_metrics = self.validate()
                
                # 计算模型可靠度
                model_reliability = compute_model_reliability(
                    val_metrics['auc'],
                    val_metrics['f1'],
                    val_metrics['rule_consistency'],
                )
                
                self.history['val_auc'].append(val_metrics['auc'])
                self.history['model_reliability'].append(model_reliability)
                
                # 检查阶段切换
                should_switch, next_stage = should_switch_stage(
                    self.current_stage, epoch, model_reliability
                )
                
                if should_switch:
                    console.print(f"\n[bold yellow]阶段切换: Stage {self.current_stage} → {next_stage}[/bold yellow]")
                    console.print(f"触发条件: Epoch={epoch}, Reliability={model_reliability:.3f}\n")
                    
                    self.current_stage = next_stage
                    self.criterion.set_stage(next_stage, epoch)
                    self.history['stage_switches'].append({
                        'epoch': epoch,
                        'from_stage': self.current_stage - 1,
                        'to_stage': next_stage,
                        'reliability': model_reliability,
                    })
                
                # 打印指标
                self._print_metrics(epoch, train_metrics, val_metrics, model_reliability)
                
                # 保存checkpoint
                if val_metrics['auc'] > self.best_val_auc:
                    self.best_val_auc = val_metrics['auc']
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
            
            self.history['train_loss'].append(train_metrics['total'])
        
        console.print("\n[bold green]训练完成！[/bold green]")
        self._print_final_summary()
    
    def _get_light_probs(self, entities: List) -> torch.Tensor:
        """提取交通灯状态概率"""
        # 找到交通灯实体
        lights = [e for e in entities if e.type == 'light']
        
        if len(lights) == 0:
            # 默认绿灯
            return torch.tensor([[0.0, 0.0, 1.0]])
        
        # 使用第一个交通灯
        light = lights[0]
        state_map = {'red': 0, 'yellow': 1, 'green': 2}
        probs = torch.zeros(1, 3)
        
        if hasattr(light, 'light_state') and light.light_state:
            idx = state_map.get(light.light_state, 2)
            probs[0, idx] = light.confidence if hasattr(light, 'confidence') else 0.9
            # 其他通道分配剩余概率
            remaining = 1.0 - probs[0, idx]
            for j in range(3):
                if j != idx:
                    probs[0, j] = remaining / 2
        else:
            probs = torch.tensor([[0.0, 0.0, 1.0]])  # 默认绿灯
        
        return probs
    
    def _compute_auc(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """计算AUC（简化版）"""
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(labels.numpy(), scores.numpy()))
        except:
            return 0.5
    
    def _compute_f1(self, pred: torch.Tensor, true: torch.Tensor) -> float:
        """计算F1 Score"""
        tp = ((pred == True) & (true == True)).sum().item()
        fp = ((pred == True) & (true == False)).sum().item()
        fn = ((pred == False) & (true == True)).sum().item()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return f1
    
    def _print_metrics(self, epoch, train_metrics, val_metrics, reliability):
        """打印训练指标"""
        table = Table(title=f"Epoch {epoch} Metrics")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Train", style="magenta")
        table.add_column("Val", style="green")
        
        table.add_row("Loss (Total)", f"{train_metrics['total']:.4f}", "-")
        table.add_row("Loss (Recon)", f"{train_metrics['recon']:.4f}", "-")
        table.add_row("Loss (Rule)", f"{train_metrics['rule']:.4f}", "-")
        table.add_row("Loss (Attn)", f"{train_metrics['attn']:.4f}", "-")
        table.add_row("AUC", "-", f"{val_metrics['auc']:.4f}")
        table.add_row("F1", "-", f"{val_metrics['f1']:.4f}")
        table.add_row("Reliability", "-", f"{reliability:.4f}")
        table.add_row("Stage", f"{self.current_stage}", "-")
        
        console.print(table)
    
    def _print_final_summary(self):
        """打印最终总结"""
        console.print("\n" + "="*60)
        console.print("[bold]训练总结[/bold]")
        console.print("="*60)
        console.print(f"总Epochs: {self.epochs}")
        console.print(f"最佳AUC: {self.best_val_auc:.4f}")
        console.print(f"阶段切换次数: {len(self.history['stage_switches'])}")
        
        if len(self.history['stage_switches']) > 0:
            console.print("\n阶段切换历史:")
            for switch in self.history['stage_switches']:
                console.print(f"  Epoch {switch['epoch']}: Stage {switch['from_stage']}→{switch['to_stage']} "
                             f"(Reliability={switch['reliability']:.3f})")
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'stage': self.current_stage,
            'history': self.history,
        }
        
        # 保存最新checkpoint
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, path)
        
        # 如果是最佳，额外保存
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            console.print(f"[green]✅ 保存最佳模型: AUC={metrics['auc']:.4f}[/green]")


@app.command()
def train(
    data_root: str = typer.Option("data/traffic", help="数据根目录"),
    batch_size: int = typer.Option(8, help="Batch size"),
    epochs: int = typer.Option(100, help="训练轮数"),
    lr: float = typer.Option(1e-4, help="学习率"),
    device: str = typer.Option("cpu", help="设备: cpu/cuda"),
    checkpoint_dir: str = typer.Option("artifacts/checkpoints", help="Checkpoint目录"),
):
    """
    训练红灯停异常检测模型
    
    Example:
        poetry run python tools/train_red_light.py train --epochs 100 --device cpu
    """
    console.print("[bold blue]初始化训练环境...[/bold blue]")
    
    # 加载数据集
    console.print("加载数据集...")
    train_dataset = TrafficLightDataset(
        data_root=data_root,
        mode='synthetic',
        split='train',
    )
    
    val_dataset = TrafficLightDataset(
        data_root=data_root,
        mode='synthetic',
        split='val',
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # 每次一个场景
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    console.print(f"训练集: {len(train_dataset)}个场景")
    console.print(f"验证集: {len(val_dataset)}个场景")
    
    # 初始化模型
    console.print("初始化模型...")
    model = MultiStageAttentionGAT(
        input_dim=10,
        hidden_dim=128,
        num_gat_layers=3,
        num_heads=8,
        dropout=0.1,
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=lr,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
    )
    
    # 开始训练
    trainer.train()
    
    console.print(f"\n[bold green]训练完成！Checkpoint保存在: {checkpoint_dir}[/bold green]")


@app.command()
def info():
    """显示模型和配置信息"""
    console.print("[bold]模型信息[/bold]")
    console.print(f"设计文档: Design-ITER-2025-01.md v2.0")
    console.print(f"算法方案: 方案1（多阶段GAT + 硬约束规则融合）")
    console.print(f"\n模型参数:")
    console.print(f"  - 输入维度: 10")
    console.print(f"  - 隐藏维度: 128")
    console.print(f"  - GAT层数: 3")
    console.print(f"  - 注意力头数: 8")
    console.print(f"  - 总参数量: ~1.02M")


if __name__ == "__main__":
    app()
