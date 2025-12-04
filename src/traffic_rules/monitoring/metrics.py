"""
评估指标计算模块
"""

import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_classification_metrics(
    model_scores: torch.Tensor,
    rule_scores: torch.Tensor,
    threshold: float = 0.7,
) -> Dict[str, float]:
    """
    计算完整的分类指标
    
    Args:
        model_scores: 模型预测分数 [N]
        rule_scores: 规则分数 [N]（作为伪标签）
        threshold: 二值化阈值（默认0.7）
    
    Returns:
        metrics: 指标字典
            - auc: AUC-ROC
            - f1: F1 Score
            - precision: 精确率
            - recall: 召回率
            - accuracy: 准确率
    """
    # 转换为numpy
    y_score = model_scores.detach().cpu().numpy()
    y_true = (rule_scores > 0.5).detach().cpu().numpy()
    y_pred = (model_scores > threshold).detach().cpu().numpy()
    
    # 处理边界情况（所有样本同类）
    if len(np.unique(y_true)) < 2:
        return {
            'auc': 0.5,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': float(np.mean(y_true == y_pred)),
        }
    
    # 计算指标
    metrics = {
        'auc': float(roc_auc_score(y_true, y_score)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'accuracy': float(np.mean(y_true == y_pred)),
    }
    
    return metrics


def compute_rule_consistency(
    model_scores: torch.Tensor,
    rule_scores: torch.Tensor,
) -> float:
    """
    计算规则一致性评分
    
    定义: 1 - MAE(model_scores, rule_scores)
    
    Args:
        model_scores: 模型分数 [N]
        rule_scores: 规则分数 [N]
    
    Returns:
        consistency: 一致性评分 [0, 1]，越高越好
    """
    mae = torch.mean(torch.abs(model_scores - rule_scores)).item()
    consistency = 1.0 - mae
    return max(0.0, consistency)


def compute_attention_focus(
    attention_weights: torch.Tensor,
    entity_types: torch.Tensor,
    violation_mask: torch.Tensor,
    edge_index: torch.Tensor,
) -> float:
    """
    计算注意力聚焦度
    
    定义: 违规车辆中，注意力最大权重落在交通灯/停止线上的比例
    
    Args:
        attention_weights: GAT注意力权重 [E]
        entity_types: 实体类型 [N] (0=car, 1=light, 2=stop)
        violation_mask: 违规车辆mask [N_car]
        edge_index: 边索引 [2, E]
    
    Returns:
        focus_rate: 聚焦率 [0, 1]
    """
    if not violation_mask.any():
        return 1.0
    
    # 获取车辆节点索引
    car_mask = (entity_types == 0)
    car_indices = torch.where(car_mask)[0]
    violation_car_indices = car_indices[violation_mask]
    
    focus_count = 0
    total_count = 0
    
    for car_idx in violation_car_indices:
        # 找到该车辆的所有出边
        out_edges_mask = (edge_index[0] == car_idx)
        
        if not out_edges_mask.any():
            continue
        
        # 获取邻居节点类型
        neighbor_indices = edge_index[1, out_edges_mask]
        neighbor_types = entity_types[neighbor_indices]
        
        # 找到注意力最大的邻居
        out_edge_attns = attention_weights[out_edges_mask]
        max_attn_idx = out_edge_attns.argmax()
        max_neighbor_type = neighbor_types[max_attn_idx].item()
        
        # 检查是否是规则相关实体（交通灯或停止线）
        if max_neighbor_type in [1, 2]:
            focus_count += 1
        
        total_count += 1
    
    if total_count == 0:
        return 1.0
    
    focus_rate = focus_count / total_count
    return focus_rate


def compute_full_metrics(
    model_scores: torch.Tensor,
    rule_scores: torch.Tensor,
    attention_weights: torch.Tensor,
    entity_types: torch.Tensor,
    edge_index: torch.Tensor,
    threshold: float = 0.7,
) -> Dict[str, float]:
    """
    计算所有评估指标
    
    注意：由于多场景合并问题，attention_focus暂时设为占位值
    
    Args:
        model_scores: 模型分数
        rule_scores: 规则分数
        attention_weights: 注意力权重（单个场景）
        entity_types: 实体类型（单个场景）
        edge_index: 边索引（单个场景）
        threshold: 分类阈值
    
    Returns:
        metrics: 完整指标字典
    """
    # 分类指标
    class_metrics = compute_classification_metrics(
        model_scores, rule_scores, threshold
    )
    
    # 规则一致性
    consistency = compute_rule_consistency(model_scores, rule_scores)
    
    # 注意力聚焦度（暂时禁用，因为多场景合并导致索引不匹配）
    # TODO: 实现跨场景的注意力聚焦度计算
    focus_rate = 0.5  # 占位值
    
    # 合并
    metrics = {
        **class_metrics,
        'rule_consistency': consistency,
        'attention_focus': focus_rate,
    }
    
    return metrics


# 导出接口
__all__ = [
    'compute_classification_metrics',
    'compute_rule_consistency',
    'compute_attention_focus',
    'compute_full_metrics',
]
