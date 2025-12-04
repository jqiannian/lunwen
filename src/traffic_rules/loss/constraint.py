"""
约束损失函数

基于Design-ITER-2025-01.md v2.0 §3.4.2-3.4.3设计
实现双层注意力监督损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class LossConfig:
    """损失函数配置"""
    lambda_recon: float = 1.0       # BCE重构损失权重
    lambda_rule: float = 0.5        # 规则一致性损失权重
    lambda_attn: float = 0.3        # 注意力一致性损失权重
    lambda_reg: float = 1e-4        # L2正则化权重
    
    # 注意力子权重（λ_attn内部分配）
    weight_attn_gat: float = 0.5    # GAT注意力权重
    weight_attn_rule: float = 0.5   # 规则聚焦注意力权重
    
    # 阈值
    violation_threshold: float = 0.5  # 违规判定阈值


def compute_gat_attention_loss(
    alpha_gat: torch.Tensor,
    edge_index: torch.Tensor,
    entity_types: torch.Tensor,
    violation_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算GAT局部注意力一致性损失
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.4.3
    
    目标：强制违规车辆的GAT注意力聚焦在交通灯/停止线上
    
    Args:
        alpha_gat: [E] - GAT边注意力权重（稀疏）
        edge_index: [2, E] - 边索引 [source_nodes, target_nodes]
        entity_types: [N] - 实体类型 (0=car, 1=light, 2=stop)
        violation_mask: [N_car] - 违规车辆mask（仅车辆节点）
    
    Returns:
        loss: 标量损失
    
    数学形式：
        L_attn^GAT = (1/|I_viol|) Σ_{i∈I_viol} (1 - max_{j∈N_rule(i)} α_ij^(L))²
    """
    device = alpha_gat.device
    loss_list = []
    
    # 获取车辆节点索引
    car_mask = (entity_types == 0)
    car_indices = torch.where(car_mask)[0]
    
    # 遍历每个违规车辆
    violation_car_indices = car_indices[violation_mask]
    
    for car_idx in violation_car_indices:
        # 找到该车辆的所有出边
        out_edges_mask = (edge_index[0] == car_idx)
        
        if not out_edges_mask.any():
            # 该车辆没有出边，跳过
            continue
        
        # 获取邻居节点索引
        neighbor_indices = edge_index[1, out_edges_mask]
        
        # 筛选规则相关邻居（交通灯type=1 或 停止线type=2）
        rule_neighbor_mask = (entity_types[neighbor_indices] == 1) | \
                             (entity_types[neighbor_indices] == 2)
        
        if not rule_neighbor_mask.any():
            # 该车辆没有规则相关邻居，跳过
            continue
        
        # 获取对规则相关邻居的注意力权重
        rule_edge_indices = torch.where(out_edges_mask)[0][rule_neighbor_mask]
        rule_neighbor_attentions = alpha_gat[rule_edge_indices]
        
        # 计算最大注意力
        max_rule_attention = rule_neighbor_attentions.max()
        
        # 损失：期望max_rule_attention → 1
        loss_list.append((1 - max_rule_attention) ** 2)
    
    if len(loss_list) > 0:
        return torch.stack(loss_list).mean()
    else:
        # 没有违规车辆或没有规则相关邻居，损失为0
        return torch.tensor(0.0, device=device)


def compute_rule_attention_loss(
    beta_rule: torch.Tensor,
    violation_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算规则聚焦注意力一致性损失
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.4.3
    
    目标：强制违规车辆的规则聚焦分数接近1
    
    Args:
        beta_rule: [N_car] - 规则聚焦注意力分数
        violation_mask: [N_car] - 违规车辆mask
    
    Returns:
        loss: 标量损失
    
    数学形式：
        L_attn^rule = (1/|I_viol|) Σ_{i∈I_viol} (1 - β_i)²
    """
    if violation_mask.any():
        return ((1 - beta_rule[violation_mask]) ** 2).mean()
    else:
        return torch.tensor(0.0, device=beta_rule.device)


class ConstraintLoss(nn.Module):
    """
    约束损失函数（集成所有损失项）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.4.2
    
    损失组成：
        L_total = L_recon + λ₁·L_rule + λ₂·L_attn + λ₃·L_reg
        
        其中 L_attn = L_attn^GAT + L_attn^rule
    """
    
    def __init__(self, config: Optional[LossConfig] = None):
        """
        初始化约束损失
        
        Args:
            config: 损失函数配置
        """
        super().__init__()
        self.config = config if config is not None else LossConfig()
    
    def forward(
        self,
        model_scores: torch.Tensor,
        rule_scores: torch.Tensor,
        alpha_gat: torch.Tensor,
        beta_rule: torch.Tensor,
        edge_index: torch.Tensor,
        entity_types: torch.Tensor,
        model_parameters: List[torch.nn.Parameter],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总损失
        
        Args:
            model_scores: [N_car] - 模型预测分数
            rule_scores: [N_car] - 规则分数
            alpha_gat: [E] - GAT边注意力权重
            beta_rule: [N_car] - 规则聚焦注意力分数
            edge_index: [2, E] - 边索引
            entity_types: [N] - 实体类型
            model_parameters: 模型参数列表（用于L2正则）
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失字典
        """
        device = model_scores.device
        
        # 1. 重构损失（BCE）
        L_recon = F.binary_cross_entropy(
            model_scores, 
            rule_scores, 
            reduction='mean'
        )
        
        # 2. 规则一致性损失（MSE）
        L_rule = F.mse_loss(model_scores, rule_scores)
        
        # 3. 注意力一致性损失（双层监督）
        # 定义违规车辆集合
        violation_mask = (rule_scores > self.config.violation_threshold)
        
        if violation_mask.any():
            # 3.1 GAT局部注意力监督
            L_attn_gat = compute_gat_attention_loss(
                alpha_gat, edge_index, entity_types, violation_mask
            )
            
            # 3.2 规则聚焦注意力监督
            L_attn_rule = compute_rule_attention_loss(
                beta_rule, violation_mask
            )
            
            # 加权组合
            L_attn = (
                self.config.weight_attn_gat * L_attn_gat +
                self.config.weight_attn_rule * L_attn_rule
            )
        else:
            # 没有违规车辆，注意力损失为0
            L_attn = torch.tensor(0.0, device=device)
            L_attn_gat = torch.tensor(0.0, device=device)
            L_attn_rule = torch.tensor(0.0, device=device)
        
        # 4. L2正则化
        L_reg = torch.tensor(0.0, device=device)
        for param in model_parameters:
            L_reg += torch.sum(param ** 2)
        
        # 总损失
        L_total = (
            self.config.lambda_recon * L_recon +
            self.config.lambda_rule * L_rule +
            self.config.lambda_attn * L_attn +
            self.config.lambda_reg * L_reg
        )
        
        # 返回详细信息
        loss_dict = {
            'total': L_total,
            'recon': L_recon,
            'rule': L_rule,
            'attn': L_attn,
            'attn_gat': L_attn_gat,
            'attn_rule': L_attn_rule,
            'reg': L_reg,
            'num_violations': violation_mask.sum().item(),
        }
        
        return L_total, loss_dict
    
    def update_config(self, **kwargs):
        """动态更新损失权重（用于三阶段训练）"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"未知配置参数: {key}")


class StagedConstraintLoss(ConstraintLoss):
    """
    分阶段约束损失（支持三阶段训练）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.7.3
    
    Stage 1（Epoch 0-20）：λ_rule = 0.5（强规则约束）
    Stage 2（Epoch 20-60）：λ_rule = 0.2（混合训练）
    Stage 3（Epoch 60+）：λ_rule = 0.1（自训练为主）
    """
    
    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__(config)
        self.current_stage = 1
        self.epoch = 0
        
        # 各阶段的λ_rule配置
        self.stage_lambda_rule = {
            1: 0.5,  # Stage 1: 强规则约束
            2: 0.2,  # Stage 2: 混合训练
            3: 0.1,  # Stage 3: 自训练为主
        }
    
    def set_stage(self, stage: int, epoch: int):
        """
        设置训练阶段
        
        Args:
            stage: 1, 2, 或 3
            epoch: 当前epoch
        """
        assert stage in [1, 2, 3], f"无效阶段: {stage}"
        
        self.current_stage = stage
        self.epoch = epoch
        
        # 更新λ_rule
        self.config.lambda_rule = self.stage_lambda_rule[stage]
        
        print(f"[Training Stage] 切换到Stage {stage} (Epoch {epoch}), λ_rule={self.config.lambda_rule}")
    
    def get_stage_info(self) -> Dict[str, any]:
        """获取当前阶段信息"""
        return {
            'stage': self.current_stage,
            'epoch': self.epoch,
            'lambda_rule': self.config.lambda_rule,
            'stage_name': {
                1: '规则监督（冷启动）',
                2: '混合训练',
                3: '自训练为主',
            }[self.current_stage]
        }


# ============ 辅助函数 ============

def compute_model_reliability(
    val_auc: float,
    val_f1: float,
    rule_consistency: float,
) -> float:
    """
    计算模型可靠度（用于阶段切换判断）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.7.3
    
    Args:
        val_auc: 验证集AUC
        val_f1: 验证集F1 Score
        rule_consistency: 模型与规则的一致性（0-1）
    
    Returns:
        reliability: 模型可靠度（0-1）
    
    阈值：
        - reliability > 0.7: Stage 1 → 2
        - reliability > 0.85: Stage 2 → 3
    """
    reliability = 0.4 * val_auc + 0.3 * val_f1 + 0.3 * rule_consistency
    return reliability


def should_switch_stage(
    current_stage: int,
    epoch: int,
    model_reliability: float,
) -> Tuple[bool, int]:
    """
    判断是否应该切换训练阶段
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.7.3
    
    Args:
        current_stage: 当前阶段（1, 2, 或 3）
        epoch: 当前epoch
        model_reliability: 模型可靠度
    
    Returns:
        should_switch: 是否切换
        next_stage: 下一阶段
    """
    if current_stage == 1:
        # Stage 1 → 2: 模型可靠度>0.7 或 达到epoch 20
        if model_reliability > 0.7 or epoch >= 20:
            return True, 2
    
    elif current_stage == 2:
        # Stage 2 → 3: 模型可靠度>0.85 或 达到epoch 60
        if model_reliability > 0.85 or epoch >= 60:
            return True, 3
    
    # 不切换
    return False, current_stage


# ============ 导出接口 ============
__all__ = [
    'compute_gat_attention_loss',
    'compute_rule_attention_loss',
    'ConstraintLoss',
    'StagedConstraintLoss',
    'LossConfig',
    'compute_model_reliability',
    'should_switch_stage',
]
