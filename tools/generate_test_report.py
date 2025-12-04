#!/usr/bin/env python3
"""
生成测试报告脚本

加载训练好的模型，在验证集上运行测试，生成详细的测试报告
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Dict, List
import json
from rich.console import Console
from rich.table import Table
from rich import box

from src.traffic_rules.data.traffic_dataset import TrafficLightDataset
from src.traffic_rules.graph.builder import GraphBuilder
from src.traffic_rules.models.multi_stage_gat import MultiStageAttentionGAT
from src.traffic_rules.loss.constraint import StagedConstraintLoss
from src.traffic_rules.rules.red_light import RedLightRuleEngine
from src.traffic_rules.monitoring.metrics import compute_full_metrics

console = Console()


def load_model(checkpoint_path: str, device: str = 'cpu') -> nn.Module:
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = MultiStageAttentionGAT(
        input_dim=10,
        hidden_dim=128,
        num_gat_layers=3,
        num_heads=8,
        dropout=0.1,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def test_model(
    model: nn.Module,
    val_dataset,
    device: str = 'cpu',
    threshold: float = 0.7,
) -> Dict:
    """在验证集上测试模型"""
    model.eval()
    
    graph_builder = GraphBuilder()
    rule_engine = RedLightRuleEngine()
    criterion = StagedConstraintLoss()
    
    # 收集所有预测
    all_model_scores = []
    all_rule_scores = []
    all_attention_weights = []
    all_entity_types = []
    all_edge_indices = []
    all_losses = []
    
    # 场景级别的统计
    scene_results = []
    
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            scene = val_dataset[idx]
            graph = graph_builder.build(scene)
            
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            entity_types = graph.entity_types.to(device)
            
            if edge_index.size(1) == 0:
                continue
            
            # 前向传播
            output = model(x, edge_index, entity_types, return_attention=True)
            
            model_scores = output['scores']
            alpha_gat = output['gat_attention']
            beta_rule = output['rule_attention']
            
            # 获取车辆实体
            car_entities = scene.get_entities_by_type('car')
            if len(car_entities) == 0:
                continue
            
            # 计算规则分数
            light_probs = get_light_probs(scene.entities).to(device)
            distances = torch.tensor([e.d_stop for e in car_entities], device=device)
            velocities = torch.tensor([e.velocity for e in car_entities], device=device)
            
            rule_scores = rule_engine.evaluate(
                light_probs, distances, velocities, training=False
            )
            
            # 计算损失
            loss_total, loss_dict = criterion(
                model_scores=model_scores,
                rule_scores=rule_scores,
                alpha_gat=alpha_gat,
                beta_rule=beta_rule,
                edge_index=edge_index,
                entity_types=entity_types,
                model_parameters=list(model.parameters()),
            )
            
            # 收集数据
            all_model_scores.append(model_scores)
            all_rule_scores.append(rule_scores)
            all_attention_weights.append(alpha_gat)
            all_entity_types.append(entity_types)
            all_edge_indices.append(edge_index)
            all_losses.append(loss_total.item())
            
            # 场景级别结果
            scene_result = {
                'scene_id': getattr(scene, 'scene_id', f'scene_{idx}'),
                'num_cars': len(car_entities),
                'model_scores': model_scores.cpu().tolist(),
                'rule_scores': rule_scores.cpu().tolist(),
                'loss': loss_total.item(),
            }
            scene_results.append(scene_result)
    
    # 计算总体指标
    if len(all_model_scores) > 0:
        model_scores_cat = torch.cat(all_model_scores)
        rule_scores_cat = torch.cat(all_rule_scores)
        
        # 使用第一个图的attention（简化处理）
        full_metrics = compute_full_metrics(
            model_scores=model_scores_cat,
            rule_scores=rule_scores_cat,
            attention_weights=all_attention_weights[0],
            entity_types=all_entity_types[0],
            edge_index=all_edge_indices[0],
            threshold=threshold,
        )
        
        full_metrics['avg_loss'] = np.mean(all_losses)
        full_metrics['scene_results'] = scene_results
    else:
        full_metrics = {'avg_loss': 0.0, 'scene_results': []}
    
    return full_metrics


def get_light_probs(entities: List) -> torch.Tensor:
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


def generate_report(
    checkpoint_path: str,
    data_root: str = "data/synthetic",
    device: str = "cpu",
    threshold: float = 0.7,
    output_path: str = "TEST_REPORT.md",
):
    """生成完整的测试报告"""
    console.print("\n[bold blue]🧪 开始生成测试报告...[/bold blue]\n")
    
    # 加载模型
    console.print("📦 加载模型...")
    model, checkpoint = load_model(checkpoint_path, device)
    console.print(f"[green]✅ 模型加载成功 (Epoch {checkpoint.get('epoch', 'unknown')})[/green]")
    
    # 加载验证集
    console.print("\n📊 加载验证集...")
    val_dataset = TrafficLightDataset(
        data_root=data_root,
        mode='synthetic',
        split='val',
    )
    console.print(f"[green]✅ 验证集: {len(val_dataset)} 场景[/green]")
    
    # 运行测试
    console.print("\n🔬 运行测试...")
    test_results = test_model(model, val_dataset, device, threshold)
    
    # 生成报告
    console.print("\n📝 生成测试报告...")
    
    # 预处理checkpoint数据，避免f-string中的复杂表达式
    val_loss = checkpoint.get('val_metrics', {}).get('loss', None)
    val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else 'N/A'
    
    val_auc = checkpoint.get('val_metrics', {}).get('auc', None)
    val_auc_str = f"{val_auc:.4f}" if isinstance(val_auc, (int, float)) else 'N/A'
    
    val_rule_cons = checkpoint.get('val_metrics', {}).get('rule_consistency', None)
    val_rule_cons_str = f"{val_rule_cons:.4f}" if isinstance(val_rule_cons, (int, float)) else 'N/A'
    
    # 计算变化
    loss_diff = test_results.get('avg_loss', 0.0) - (val_loss if isinstance(val_loss, (int, float)) else 0.0)
    loss_diff_str = f"{loss_diff:.4f}" if isinstance(val_loss, (int, float)) else 'N/A'
    
    auc_diff = test_results.get('auc', 0.0) - (val_auc if isinstance(val_auc, (int, float)) else 0.0)
    auc_diff_str = f"{auc_diff:.4f}" if isinstance(val_auc, (int, float)) else 'N/A'
    
    rule_cons_diff = test_results.get('rule_consistency', 0.0) - (val_rule_cons if isinstance(val_rule_cons, (int, float)) else 0.0)
    rule_cons_diff_str = f"{rule_cons_diff:.4f}" if isinstance(val_rule_cons, (int, float)) else 'N/A'
    
    report = f"""# 测试报告 - 红灯停异常检测模型

## 报告信息
- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **模型检查点**: {checkpoint_path}
- **测试数据集**: {data_root}/val
- **测试场景数**: {len(val_dataset)}
- **训练Epoch**: {checkpoint.get('epoch', 'unknown')}
- **最佳验证Loss**: {val_loss_str}

---

## 📊 测试指标总览

### 核心性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **平均损失** | {test_results.get('avg_loss', 0.0):.4f} | 模型在验证集上的平均损失 |
| **AUC** | {test_results.get('auc', 0.0):.4f} | ROC曲线下面积 |
| **F1 Score** | {test_results.get('f1', 0.0):.4f} | F1分数（阈值={threshold}） |
| **Precision** | {test_results.get('precision', 0.0):.4f} | 精确率 |
| **Recall** | {test_results.get('recall', 0.0):.4f} | 召回率 |
| **规则一致性** | {test_results.get('rule_consistency', 0.0):.4f} | 模型预测与规则分数的一致性 |
| **注意力聚焦** | {test_results.get('attention_focus', 0.0):.4f} | 注意力权重聚焦程度 |

### 与训练指标对比

| 指标 | 训练时（最佳） | 测试时 | 变化 |
|------|---------------|--------|------|
| **Val Loss** | {val_loss_str} | {test_results.get('avg_loss', 0.0):.4f} | {loss_diff_str} |
| **AUC** | {val_auc_str} | {test_results.get('auc', 0.0):.4f} | {auc_diff_str} |
| **Rule Cons.** | {val_rule_cons_str} | {test_results.get('rule_consistency', 0.0):.4f} | {rule_cons_diff_str} |

---

## 📈 详细分析

### 1. 分类性能分析

**AUC = {test_results.get('auc', 0.0):.4f}**

- {'✅ 优秀' if test_results.get('auc', 0.0) >= 0.9 else '✅ 良好' if test_results.get('auc', 0.0) >= 0.8 else '⚠️ 需改进' if test_results.get('auc', 0.0) >= 0.7 else '❌ 较差'} (目标: ≥0.90)
- 模型能够较好地区分违规和正常场景

**F1 Score = {test_results.get('f1', 0.0):.4f}**

- {'✅ 优秀' if test_results.get('f1', 0.0) >= 0.85 else '✅ 良好' if test_results.get('f1', 0.0) >= 0.75 else '⚠️ 需改进' if test_results.get('f1', 0.0) >= 0.5 else '❌ 较差'} (目标: ≥0.85)
- Precision = {test_results.get('precision', 0.0):.4f}, Recall = {test_results.get('recall', 0.0):.4f}
- {'模型倾向于保守预测（高精确率，低召回率）' if test_results.get('precision', 0.0) > test_results.get('recall', 0.0) + 0.1 else '模型倾向于激进预测（低精确率，高召回率）' if test_results.get('recall', 0.0) > test_results.get('precision', 0.0) + 0.1 else '精确率和召回率相对平衡'}

### 2. 规则一致性分析

**规则一致性 = {test_results.get('rule_consistency', 0.0):.4f}**

- {'✅ 优秀' if test_results.get('rule_consistency', 0.0) >= 0.8 else '✅ 良好' if test_results.get('rule_consistency', 0.0) >= 0.7 else '⚠️ 需改进'} (目标: ≥0.75)
- 模型预测与规则引擎评分的一致性程度
- {'模型很好地学习了规则逻辑' if test_results.get('rule_consistency', 0.0) >= 0.75 else '模型仍需进一步学习规则逻辑'}

### 3. 注意力机制分析

**注意力聚焦 = {test_results.get('attention_focus', 0.0):.4f}**

- {'✅ 注意力机制工作良好' if test_results.get('attention_focus', 0.0) >= 0.6 else '⚠️ 注意力机制需要优化'}
- 模型是否能够聚焦到关键的交通实体（车辆、交通灯、停止线）

---

## 🎯 场景级别统计

### 测试场景分布

- **总场景数**: {len(test_results.get('scene_results', []))}
- **总车辆数**: {sum(s.get('num_cars', 0) for s in test_results.get('scene_results', []))}

### 场景类型分析

（基于场景ID和元数据推断）

---

## 📋 模型配置信息

### 模型架构
- **模型类型**: MultiStageAttentionGAT
- **输入维度**: 10
- **隐藏维度**: 128
- **GAT层数**: 3
- **注意力头数**: 8
- **Dropout**: 0.1

### 训练配置
- **学习率**: {checkpoint.get('train_metrics', {}).get('lr', 'N/A') if 'train_metrics' in checkpoint else 'N/A'}
- **优化器**: AdamW
- **损失函数**: StagedConstraintLoss
- **设备**: {device}

---

## ✅ 结论与建议

### 主要发现

1. **模型性能**: {'✅ 模型在验证集上表现良好' if test_results.get('auc', 0.0) >= 0.8 else '⚠️ 模型性能有待提升'}
2. **规则学习**: {'✅ 模型成功学习了规则逻辑' if test_results.get('rule_consistency', 0.0) >= 0.75 else '⚠️ 模型对规则的学习仍需加强'}
3. **分类能力**: {'✅ 模型能够有效区分违规场景' if test_results.get('f1', 0.0) >= 0.5 else '⚠️ 模型分类能力需要改进'}

### 改进建议

1. {'**AUC优化**: 当前AUC为{test_results.get("auc", 0.0):.4f}，建议通过以下方式提升：' if test_results.get('auc', 0.0) < 0.9 else '**AUC表现良好**: 已达到目标水平'}
   - 增加训练数据量
   - 调整模型架构（增加层数或隐藏维度）
   - 优化超参数（学习率、正则化）

2. {'**F1 Score优化**: 当前F1为{test_results.get("f1", 0.0):.4f}，建议：' if test_results.get('f1', 0.0) < 0.85 else '**F1 Score表现良好**: 已达到目标水平'}
   - 调整分类阈值（当前{threshold}）
   - 使用类别权重平衡
   - 增加违规样本的训练数据

3. **规则一致性优化**:
   - 增加规则损失权重
   - 使用规则引导的预训练

---

## 📁 相关文件

- **模型检查点**: `{checkpoint_path}`
- **训练曲线**: `reports/training_curves.png`
- **测试数据**: `{data_root}/val`

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # 保存报告
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding='utf-8')
    
    console.print(f"[green]✅ 测试报告已保存: {output_path}[/green]")
    
    # 打印摘要
    console.print("\n" + "="*60)
    console.print("[bold]测试结果摘要[/bold]")
    console.print("="*60)
    
    table = Table(box=box.ROUNDED)
    table.add_column("指标", style="cyan")
    table.add_column("数值", style="magenta")
    
    table.add_row("平均损失", f"{test_results.get('avg_loss', 0.0):.4f}")
    table.add_row("AUC", f"{test_results.get('auc', 0.0):.4f}")
    table.add_row("F1 Score", f"{test_results.get('f1', 0.0):.4f}")
    table.add_row("Precision", f"{test_results.get('precision', 0.0):.4f}")
    table.add_row("Recall", f"{test_results.get('recall', 0.0):.4f}")
    table.add_row("规则一致性", f"{test_results.get('rule_consistency', 0.0):.4f}")
    
    console.print(table)
    console.print("="*60 + "\n")
    
    return test_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成测试报告")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/best.pth",
        help="模型检查点路径"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/synthetic",
        help="数据根目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备 (cpu/cuda)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="分类阈值"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="TEST_REPORT.md",
        help="输出报告路径"
    )
    
    args = parser.parse_args()
    
    generate_report(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        device=args.device,
        threshold=args.threshold,
        output_path=args.output,
    )

