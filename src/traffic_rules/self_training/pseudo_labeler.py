"""
自训练伪标签生成器

基于Design-ITER-2025-01.md v2.0 §3.7设计
支持三种策略：规则优先、加权融合、动态切换
"""

import torch
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PseudoLabel:
    """伪标签数据结构"""
    scene_id: str
    entity_id: str
    label: int  # 0=正常，1=违规
    confidence: float
    model_score: float
    rule_score: float
    source: str  # 'rule_priority', 'weighted_fusion', 'model_priority'
    flag: Optional[str] = None  # 'model_disagree'等


class PseudoLabeler:
    """
    伪标签生成器
    
    设计依据：Design-ITER-2025-01.md v2.0 §3.7.4-3.7.7
    
    支持三种策略：
        1. 规则优先（Stage 1-2）：模型与规则一致时生成
        2. 加权融合（Stage 2）：综合模型和规则
        3. 动态切换（Stage 2-3）：根据模型可靠度选择策略
    """
    
    def __init__(
        self,
        strategy: str = 'rule_priority',
        threshold_conf: float = 0.85,
        threshold_consistency: float = 0.2,
        weight_rule: float = 0.6,
        weight_model: float = 0.4,
    ):
        """
        初始化伪标签生成器
        
        Args:
            strategy: 'rule_priority', 'weighted_fusion', 'adaptive'
            threshold_conf: 置信度阈值
            threshold_consistency: 一致性阈值
            weight_rule: 规则权重（加权融合时）
            weight_model: 模型权重（加权融合时）
        """
        self.strategy = strategy
        self.threshold_conf = threshold_conf
        self.threshold_consistency = threshold_consistency
        self.weight_rule = weight_rule
        self.weight_model = weight_model
        
        self.pseudo_labels: List[PseudoLabel] = []
    
    def generate_rule_priority(
        self,
        model_scores: torch.Tensor,
        rule_scores: torch.Tensor,
        attention_max: torch.Tensor,
        scene_ids: List[str],
        entity_ids: List[str],
    ) -> List[PseudoLabel]:
        """
        策略1：规则优先
        
        设计依据：Design §3.7.4
        
        仅当模型与规则一致时才生成伪标签
        """
        pseudo_labels = []
        
        for i in range(len(model_scores)):
            # 计算置信度
            confidence = (
                torch.sigmoid(model_scores[i]).item() *
                rule_scores[i].item() *
                attention_max[i].item()
            )
            
            # 一致性检查
            consistency = abs(model_scores[i].item() - rule_scores[i].item())
            
            # 生成条件（AND逻辑）
            if (confidence > self.threshold_conf and
                consistency < self.threshold_consistency and
                attention_max[i] > 0.3):
                
                pseudo_labels.append(PseudoLabel(
                    scene_id=scene_ids[i],
                    entity_id=entity_ids[i],
                    label=1 if rule_scores[i] > 0.5 else 0,
                    confidence=confidence,
                    model_score=model_scores[i].item(),
                    rule_score=rule_scores[i].item(),
                    source='rule_priority',
                ))
            
            # 冲突场景处理（场景B）
            elif rule_scores[i] > 0.7 and model_scores[i] < 0.3:
                pseudo_labels.append(PseudoLabel(
                    scene_id=scene_ids[i],
                    entity_id=entity_ids[i],
                    label=1,  # 信任规则
                    confidence=0.6,  # 降低置信度
                    model_score=model_scores[i].item(),
                    rule_score=rule_scores[i].item(),
                    source='rule_override',
                    flag='model_disagree',
                ))
        
        return pseudo_labels
    
    def generate_weighted_fusion(
        self,
        model_scores: torch.Tensor,
        rule_scores: torch.Tensor,
        attention_max: torch.Tensor,
        scene_ids: List[str],
        entity_ids: List[str],
    ) -> List[PseudoLabel]:
        """
        策略2：加权融合
        
        设计依据：Design §3.7.5
        """
        pseudo_labels = []
        
        for i in range(len(model_scores)):
            # 加权评分
            fused_score = (
                self.weight_rule * rule_scores[i] +
                self.weight_model * torch.sigmoid(model_scores[i])
            )
            
            # 置信度（考虑一致性奖励）
            consistency_bonus = 1.0 - abs(model_scores[i] - rule_scores[i]) / 2.0
            confidence = (
                fused_score.item() *
                attention_max[i].item() *
                consistency_bonus.item()
            )
            
            if confidence > self.threshold_conf:
                pseudo_labels.append(PseudoLabel(
                    scene_id=scene_ids[i],
                    entity_id=entity_ids[i],
                    label=1 if fused_score > 0.5 else 0,
                    confidence=confidence,
                    model_score=model_scores[i].item(),
                    rule_score=rule_scores[i].item(),
                    source='weighted_fusion',
                ))
        
        return pseudo_labels
    
    def generate(
        self,
        model_scores: torch.Tensor,
        rule_scores: torch.Tensor,
        attention_max: torch.Tensor,
        scene_ids: List[str],
        entity_ids: List[str],
    ) -> List[PseudoLabel]:
        """
        根据策略生成伪标签
        
        Args:
            model_scores: [N] 模型预测分数
            rule_scores: [N] 规则分数
            attention_max: [N] 最大注意力权重
            scene_ids: [N] 场景ID列表
            entity_ids: [N] 实体ID列表
        
        Returns:
            pseudo_labels: 伪标签列表
        """
        if self.strategy == 'rule_priority':
            return self.generate_rule_priority(
                model_scores, rule_scores, attention_max,
                scene_ids, entity_ids,
            )
        elif self.strategy == 'weighted_fusion':
            return self.generate_weighted_fusion(
                model_scores, rule_scores, attention_max,
                scene_ids, entity_ids,
            )
        else:
            raise ValueError(f"未知策略: {self.strategy}")
    
    def save_epoch(self, epoch: int, save_dir: str = 'artifacts/pseudo_labels'):
        """
        保存当前epoch的伪标签
        
        Args:
            epoch: 当前epoch
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换为DataFrame
        data = []
        for pl in self.pseudo_labels:
            data.append({
                'scene_id': pl.scene_id,
                'entity_id': pl.entity_id,
                'label': pl.label,
                'confidence': pl.confidence,
                'model_score': pl.model_score,
                'rule_score': pl.rule_score,
                'source': pl.source,
                'flag': pl.flag,
            })
        
        if len(data) > 0:
            df = pd.DataFrame(data)
            df.to_parquet(save_dir / f'epoch_{epoch:03d}.parquet')
            
            # 保存统计信息
            stats = {
                'epoch': epoch,
                'total': len(self.pseudo_labels),
                'violations': sum(1 for pl in self.pseudo_labels if pl.label == 1),
                'avg_confidence': sum(pl.confidence for pl in self.pseudo_labels) / len(self.pseudo_labels),
                'strategy': self.strategy,
            }
            
            with open(save_dir / f'epoch_{epoch:03d}_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
        
        # 清空当前伪标签
        self.pseudo_labels.clear()


class AdaptivePseudoLabeler(PseudoLabeler):
    """
    自适应伪标签生成器
    
    设计依据：Design §3.7.6
    
    根据训练阶段和模型可靠度动态选择策略
    """
    
    def __init__(self, **kwargs):
        super().__init__(strategy='adaptive', **kwargs)
        self.epoch = 0
        self.model_reliability = 0.0
    
    def select_strategy(self) -> str:
        """
        根据训练阶段选择策略
        
        Returns:
            strategy: 'rule_priority', 'weighted_fusion', or 'model_priority'
        """
        if self.epoch < 20 or self.model_reliability < 0.7:
            return 'rule_priority'  # 早期：规则优先
        elif self.epoch < 60 or self.model_reliability < 0.85:
            return 'weighted_fusion'  # 中期：加权融合
        else:
            return 'model_priority'  # 后期：模型优先
    
    def update_reliability(self, val_auc: float, val_f1: float, rule_consistency: float):
        """更新模型可靠度"""
        self.model_reliability = (
            0.4 * val_auc +
            0.3 * val_f1 +
            0.3 * rule_consistency
        )
    
    def generate(self, *args, **kwargs) -> List[PseudoLabel]:
        """动态选择策略生成伪标签"""
        current_strategy = self.select_strategy()
        self.strategy = current_strategy
        
        return super().generate(*args, **kwargs)


# ============ 导出接口 ============
__all__ = [
    'PseudoLabeler',
    'AdaptivePseudoLabeler',
    'PseudoLabel',
]
