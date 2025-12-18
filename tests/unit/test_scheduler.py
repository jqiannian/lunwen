"""
SelfTrainingScheduler 单元测试
"""

import sys
from pathlib import Path

# import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.traffic_rules.self_training.scheduler import SelfTrainingScheduler


class TestSelfTrainingScheduler:
    """测试SelfTrainingScheduler类"""
    
    def test_initialization(self):
        """测试初始化"""
        scheduler = SelfTrainingScheduler(
            total_epochs=100,
            warmup_epochs=10,
            pseudo_label_interval=5,
        )
        
        assert scheduler.total_epochs == 100
        assert scheduler.warmup_epochs == 10
        assert scheduler.current_epoch == 0
        
        print("✅ Scheduler初始化成功")
    
    def test_warmup_period(self):
        """测试warmup期间不生成伪标签"""
        scheduler = SelfTrainingScheduler(
            total_epochs=100,
            warmup_epochs=10,
        )
        
        # Warmup期间（epoch 0-9）
        for epoch in range(10):
            scheduler.current_epoch = epoch
            should_gen = scheduler.should_generate_pseudo_labels()
            assert not should_gen, f"Warmup期间不应生成伪标签 (epoch {epoch})"
        
        print("✅ Warmup期间正确")
    
    def test_pseudo_label_generation_interval(self):
        """测试伪标签生成间隔"""
        scheduler = SelfTrainingScheduler(
            total_epochs=100,
            warmup_epochs=10,
            pseudo_label_interval=5,
        )
        
        # Warmup后（epoch 10, 15, 20...应生成）
        should_generate = []
        for epoch in range(10, 30):
            scheduler.current_epoch = epoch
            should_generate.append(scheduler.should_generate_pseudo_labels())
        
        # epoch 10, 15, 20, 25应生成
        expected = [True, False, False, False, False,  # 10-14
                   True, False, False, False, False,   # 15-19
                   True, False, False, False, False,   # 20-24
                   True, False, False, False, False]   # 25-29
        
        assert should_generate == expected, "生成间隔不正确"
        
        print("✅ 伪标签生成间隔正确")
    
    def test_confidence_threshold_linear_decay(self):
        """测试线性衰减置信度阈值"""
        scheduler = SelfTrainingScheduler(
            total_epochs=100,
            warmup_epochs=10,
            initial_threshold=0.8,
            final_threshold=0.6,
            strategy='linear',
        )
        
        # Warmup时应为初始阈值
        scheduler.current_epoch = 5
        assert scheduler.get_confidence_threshold() == 0.8
        
        # Warmup后应开始衰减
        scheduler.current_epoch = 10
        threshold_start = scheduler.get_confidence_threshold()
        assert threshold_start == 0.8
        
        # 中期应介于initial和final之间
        scheduler.current_epoch = 55  # 中点
        threshold_mid = scheduler.get_confidence_threshold()
        assert 0.6 < threshold_mid < 0.8, f"中期阈值应在0.6-0.8之间: {threshold_mid}"
        
        # 末期应接近final
        scheduler.current_epoch = 100
        threshold_end = scheduler.get_confidence_threshold()
        assert abs(threshold_end - 0.6) < 0.01, f"末期阈值应接近0.6: {threshold_end}"
        
        print("✅ 线性衰减策略正确")
    
    def test_step_function(self):
        """测试step函数"""
        scheduler = SelfTrainingScheduler(total_epochs=100)
        
        assert scheduler.current_epoch == 0
        
        scheduler.step()
        assert scheduler.current_epoch == 1
        
        scheduler.step()
        assert scheduler.current_epoch == 2
        
        print("✅ step函数正常")
    
    def test_get_stats(self):
        """测试统计信息获取"""
        scheduler = SelfTrainingScheduler(
            total_epochs=100,
            warmup_epochs=10,
        )
        
        scheduler.current_epoch = 15
        stats = scheduler.get_stats()
        
        assert 'current_epoch' in stats
        assert 'total_epochs' in stats
        assert 'in_warmup' in stats
        assert 'progress' in stats
        
        assert stats['current_epoch'] == 15
        assert stats['in_warmup'] == False  # 已过warmup
        
        print(f"✅ 统计信息正确: {stats}")


if __name__ == "__main__":
    test = TestSelfTrainingScheduler()
    
    test.test_initialization()
    test.test_warmup_period()
    test.test_pseudo_label_generation_interval()
    test.test_confidence_threshold_linear_decay()
    test.test_step_function()
    test.test_get_stats()
    
    print("\n" + "="*60)
    print("✅ 所有Scheduler测试通过！")
    print("="*60)
