"""
红灯停规则评分引擎

基于Design-ITER-2025-01.md v2.0 §3.4.1设计
实现物理正确的分段规则评分函数
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class RuleConfig:
    """规则配置参数"""
    tau_d: float = 5.0          # 安全停车距离阈值（米）
    tau_v: float = 0.5          # 停车速度阈值（米/秒）
    alpha_d: float = 2.0        # 接近停止线敏感度
    alpha_v: float = 5.0        # 速度敏感度
    alpha_cross: float = 3.0    # 过线违规敏感度
    temperature: float = 0.5    # Gumbel-Softmax温度


def compute_rule_score_differentiable(
    light_probs: torch.Tensor,
    distances: torch.Tensor,
    velocities: torch.Tensor,
    config: Optional[RuleConfig] = None,
    training: bool = True,
) -> torch.Tensor:
    """
    完全可导的规则评分函数（物理正确版）
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.4.1
    
    Args:
        light_probs: [B, 3] - 交通灯状态概率 [red, yellow, green]
        distances: [B] - 到停止线距离（正数=未过线，负数=已过线）
        velocities: [B] - 车辆速度（米/秒）
        config: 规则配置参数（默认使用RuleConfig()）
        training: 是否训练模式（影响Gumbel-Softmax）
    
    Returns:
        rule_scores: [B] - 违规分数，0=无违规，1=严重违规
    
    物理意义：
        - 完全停止（v=0, d>0）：score ≈ 0
        - 远离停止线（d≥tau_d）：score = 0
        - 闯过停止线（d<0, v>0）：score ≈ 1
        - 接近且速度快（0<d<tau_d, v>tau_v）：score ≈ 1
    """
    if config is None:
        config = RuleConfig()
    
    B = distances.size(0)
    device = distances.device
    
    # Step 1: Gumbel-Softmax软化交通灯状态
    if training:
        # 训练时：使用Gumbel-Softmax增加探索性
        light_weights = F.gumbel_softmax(
            torch.log(light_probs + 1e-10),
            tau=config.temperature,
            hard=False
        )[:, 0]  # 提取red通道
    else:
        # 推理时：直接使用red概率
        light_weights = light_probs[:, 0]
    
    # Step 2: 计算分段距离-速度评分 f_dv(d, v)
    # 统一dtype为float32
    distances = distances.float()
    velocities = velocities.float()
    f_dv = torch.zeros(B, device=device, dtype=torch.float32)
    
    # 情况1：已过线（d < 0）
    # 物理意义：车辆闯过停止线，距离越远（负得越多）违规越严重
    crossed_mask = (distances < 0)
    if crossed_mask.any():
        f_dv[crossed_mask] = (
            torch.sigmoid(config.alpha_cross * (-distances[crossed_mask])) *
            torch.sigmoid(config.alpha_v * velocities[crossed_mask])
        )
    
    # 情况2：接近停止线（0 <= d < tau_d）
    # 物理意义：在安全距离内，距离越近且速度越高违规风险越大
    approaching_mask = (distances >= 0) & (distances < config.tau_d)
    if approaching_mask.any():
        f_dv[approaching_mask] = (
            torch.sigmoid(config.alpha_d * (config.tau_d - distances[approaching_mask])) *
            torch.sigmoid(config.alpha_v * (velocities[approaching_mask] - config.tau_v))
        )
    
    # 情况3：远离停止线（d >= tau_d）
    # f_dv保持为0（已初始化为零）
    
    # Step 3: 组合交通灯权重
    rule_scores = light_weights * f_dv
    
    return rule_scores


def compute_rule_score_batch(
    light_states: torch.Tensor,
    distances: torch.Tensor,
    velocities: torch.Tensor,
    config: Optional[RuleConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    批量计算规则分数（带详细信息）
    
    Args:
        light_states: [B, 3] - 交通灯状态概率
        distances: [B] - 到停止线距离
        velocities: [B] - 车辆速度
        config: 规则配置
    
    Returns:
        dict包含：
            - scores: [B] 总分
            - light_weights: [B] 交通灯权重
            - distance_scores: [B] 距离项分数
            - velocity_scores: [B] 速度项分数
            - violation_mask: [B] 违规mask（score>0.5）
    """
    if config is None:
        config = RuleConfig()
    
    B = distances.size(0)
    device = distances.device
    
    # 交通灯权重
    light_weights = light_states[:, 0]  # red通道
    
    # 分解计算（用于分析）
    distance_scores = torch.zeros(B, device=device)
    velocity_scores = torch.zeros(B, device=device)
    
    # 已过线
    crossed_mask = (distances < 0)
    if crossed_mask.any():
        distance_scores[crossed_mask] = torch.sigmoid(
            config.alpha_cross * (-distances[crossed_mask])
        )
        velocity_scores[crossed_mask] = torch.sigmoid(
            config.alpha_v * velocities[crossed_mask]
        )
    
    # 接近停止线
    approaching_mask = (distances >= 0) & (distances < config.tau_d)
    if approaching_mask.any():
        distance_scores[approaching_mask] = torch.sigmoid(
            config.alpha_d * (config.tau_d - distances[approaching_mask])
        )
        velocity_scores[approaching_mask] = torch.sigmoid(
            config.alpha_v * (velocities[approaching_mask] - config.tau_v)
        )
    
    # 组合
    f_dv = distance_scores * velocity_scores
    scores = light_weights * f_dv
    
    return {
        'scores': scores,
        'light_weights': light_weights,
        'distance_scores': distance_scores,
        'velocity_scores': velocity_scores,
        'f_dv': f_dv,
        'violation_mask': scores > 0.5,
    }


class RedLightRuleEngine:
    """
    红灯停规则引擎（DSL封装）
    
    提供规则评分、冲突检测、在线推理功能
    """
    
    def __init__(self, config: Optional[RuleConfig] = None):
        """
        初始化规则引擎
        
        Args:
            config: 规则配置参数
        """
        self.config = config if config is not None else RuleConfig()
    
    def evaluate(
        self,
        light_probs: torch.Tensor,
        distances: torch.Tensor,
        velocities: torch.Tensor,
        training: bool = False,
        return_details: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        评估规则违规分数
        
        Args:
            light_probs: [B, 3] 交通灯状态概率
            distances: [B] 到停止线距离
            velocities: [B] 车辆速度
            training: 是否训练模式
            return_details: 是否返回详细信息
        
        Returns:
            rule_scores: [B] 或 详细信息dict
        """
        if return_details:
            return compute_rule_score_batch(
                light_probs, distances, velocities, self.config
            )
        else:
            return compute_rule_score_differentiable(
                light_probs, distances, velocities, self.config, training
            )
    
    def hard_violation_check(
        self,
        light_state: str,
        distance: float,
        velocity: float,
    ) -> bool:
        """
        硬阈值违规检测（用于验收测试）
        
        Args:
            light_state: 'red' | 'yellow' | 'green'
            distance: 到停止线距离
            velocity: 车辆速度
        
        Returns:
            True表示违规，False表示正常
        """
        if light_state != 'red':
            return False
        
        # 已过线 或 （接近停止线且速度过快）
        if distance < 0:
            return True
        elif 0 <= distance < self.config.tau_d and velocity > self.config.tau_v:
            return True
        else:
            return False
    
    def get_violation_explanation(
        self,
        light_state: str,
        distance: float,
        velocity: float,
        score: float,
    ) -> str:
        """
        生成违规解释（自然语言）
        
        Args:
            light_state: 交通灯状态
            distance: 到停止线距离
            velocity: 车辆速度
            score: 规则分数
        
        Returns:
            violation_explanation: 违规解释文本
        """
        if score < 0.5:
            return "正常行驶，无违规"
        
        explanations = []
        
        if light_state == 'red':
            explanations.append("🔴 红灯状态")
        
        if distance < 0:
            explanations.append(f"⚠️ 已闯过停止线 {abs(distance):.1f}米")
        elif distance < self.config.tau_d:
            explanations.append(f"⚠️ 距离停止线仅 {distance:.1f}米（安全距离{self.config.tau_d}米）")
        
        if velocity > self.config.tau_v:
            explanations.append(f"⚠️ 速度 {velocity:.1f}m/s（应低于{self.config.tau_v}m/s）")
        
        explanations.append(f"违规分数: {score:.3f}")
        
        return " | ".join(explanations)
    
    def update_config(self, **kwargs):
        """动态更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"未知配置参数: {key}")


# ============ 导出接口 ============
__all__ = [
    'compute_rule_score_differentiable',
    'compute_rule_score_batch',
    'RedLightRuleEngine',
    'RuleConfig',
]
