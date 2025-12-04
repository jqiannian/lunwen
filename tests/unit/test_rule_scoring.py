"""
规则评分函数单元测试

测试规则：Design-ITER-2025-01.md v2.0 §3.4.1
验证物理正确性和梯度可导性
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.traffic_rules.rules.red_light import (
    compute_rule_score_differentiable,
    compute_rule_score_batch,
    RedLightRuleEngine,
    RuleConfig,
)


class TestRuleScoreBoundaryConditions:
    """测试规则分数的边界条件（物理正确性）"""
    
    def test_complete_stop_far_from_stopline(self):
        """测试用例1：完全停止，远离停止线"""
        light_probs = torch.tensor([[0.9, 0.05, 0.05]])  # 红灯
        distances = torch.tensor([10.0])  # 10米（>tau_d=5）
        velocities = torch.tensor([0.0])  # 完全停止
        
        score = compute_rule_score_differentiable(
            light_probs, distances, velocities, training=False
        )
        
        # 期望：score ≈ 0（远离停止线，分数为0）
        assert score.item() < 0.1, f"完全停止应无违规，实际分数: {score.item():.4f}"
        print(f"✅ 测试1通过: 完全停止 | 分数={score.item():.4f}")
    
    def test_complete_stop_near_stopline(self):
        """测试用例2：完全停止，接近停止线"""
        light_probs = torch.tensor([[0.9, 0.05, 0.05]])
        distances = torch.tensor([3.0])  # 3米（<tau_d=5）
        velocities = torch.tensor([0.0])  # 完全停止
        
        score = compute_rule_score_differentiable(
            light_probs, distances, velocities, training=False
        )
        
        # 期望：score ≈ 0（虽然接近但完全停止）
        # sigmoid(5*(0-0.5)) = sigmoid(-2.5) ≈ 0.076
        # score = 0.9 * sigmoid(2*(5-3)) * 0.076 ≈ 0.9 * 0.982 * 0.076 ≈ 0.067
        assert score.item() < 0.15, f"完全停止应低分，实际分数: {score.item():.4f}"
        print(f"✅ 测试2通过: 接近但停止 | 分数={score.item():.4f}")
    
    def test_crossed_stopline_moving(self):
        """测试用例3：已闯过停止线，移动中"""
        light_probs = torch.tensor([[0.9, 0.05, 0.05]])
        distances = torch.tensor([-2.0])  # -2米（已过线）
        velocities = torch.tensor([2.0])  # 2m/s
        
        score = compute_rule_score_differentiable(
            light_probs, distances, velocities, training=False
        )
        
        # 期望：score ≈ 1（严重违规）
        # sigmoid(3*2) * sigmoid(5*2) = sigmoid(6) * sigmoid(10) ≈ 0.998 * 1.0
        assert score.item() > 0.8, f"闯红灯应高分，实际分数: {score.item():.4f}"
        print(f"✅ 测试3通过: 闯过停止线 | 分数={score.item():.4f}")
    
    def test_approaching_fast(self):
        """测试用例4：接近停止线且速度快"""
        light_probs = torch.tensor([[0.9, 0.05, 0.05]])
        distances = torch.tensor([3.0])  # 3米
        velocities = torch.tensor([2.0])  # 2m/s（>tau_v=0.5）
        
        score = compute_rule_score_differentiable(
            light_probs, distances, velocities, training=False
        )
        
        # 期望：score ≈ 1（冲向红灯）
        # sigmoid(2*(5-3)) * sigmoid(5*(2-0.5)) = sigmoid(4) * sigmoid(7.5) ≈ 0.982 * 0.999
        assert score.item() > 0.8, f"冲向红灯应高分，实际分数: {score.item():.4f}"
        print(f"✅ 测试4通过: 冲向红灯 | 分数={score.item():.4f}")
    
    def test_green_light_pass(self):
        """测试用例5：绿灯通过停止线（正常）"""
        light_probs = torch.tensor([[0.05, 0.05, 0.9]])  # 绿灯
        distances = torch.tensor([-2.0])  # 已过线
        velocities = torch.tensor([2.0])  # 2m/s
        
        score = compute_rule_score_differentiable(
            light_probs, distances, velocities, training=False
        )
        
        # 期望：score ≈ 0（绿灯权重接近0）
        assert score.item() < 0.1, f"绿灯通过应无违规，实际分数: {score.item():.4f}"
        print(f"✅ 测试5通过: 绿灯通过 | 分数={score.item():.4f}")
    
    def test_far_from_stopline_any_speed(self):
        """测试用例6：远离停止线，任意速度"""
        light_probs = torch.tensor([[0.9, 0.05, 0.05]])
        distances = torch.tensor([20.0])  # 20米（远离）
        velocities = torch.tensor([10.0])  # 高速
        
        score = compute_rule_score_differentiable(
            light_probs, distances, velocities, training=False
        )
        
        # 期望：score = 0（分段函数定义，d>=tau_d时为0）
        assert score.item() == 0.0, f"远离停止线应为0，实际分数: {score.item():.4f}"
        print(f"✅ 测试6通过: 远离停止线 | 分数={score.item():.4f}")


class TestRuleScoreGradient:
    """测试梯度可导性"""
    
    def test_gradient_nonzero_all_inputs(self):
        """测试所有输入的梯度非零"""
        light_probs = torch.tensor([[0.9, 0.05, 0.05]], requires_grad=True)
        distances = torch.tensor([3.0], requires_grad=True)
        velocities = torch.tensor([2.0], requires_grad=True)
        
        score = compute_rule_score_differentiable(
            light_probs, distances, velocities, training=False
        )
        
        # 反向传播
        score.backward()
        
        # 验证梯度非零
        assert distances.grad is not None, "距离梯度为None"
        assert velocities.grad is not None, "速度梯度为None"
        assert light_probs.grad is not None, "交通灯梯度为None"
        
        assert distances.grad.abs().item() > 1e-6, f"距离梯度为0: {distances.grad.item()}"
        assert velocities.grad.abs().item() > 1e-6, f"速度梯度为0: {velocities.grad.item()}"
        assert light_probs.grad.abs().sum().item() > 1e-6, f"交通灯梯度为0"
        
        print(f"✅ 梯度测试通过:")
        print(f"   ∂L/∂d = {distances.grad.item():8.4f}")
        print(f"   ∂L/∂v = {velocities.grad.item():8.4f}")
        print(f"   ∂L/∂p_red = {light_probs.grad[0, 0].item():8.4f}")
    
    def test_gradient_flow_edge_cases(self):
        """测试边界情况的梯度"""
        test_cases = [
            (torch.tensor([10.0]), torch.tensor([0.0])),   # d远，v=0
            (torch.tensor([0.5]), torch.tensor([1.0])),    # d很近，v中等
            (torch.tensor([-5.0]), torch.tensor([5.0])),   # d过线远，v高
        ]
        
        light_probs = torch.tensor([[0.9, 0.05, 0.05]])
        
        for distances, velocities in test_cases:
            distances.requires_grad = True
            velocities.requires_grad = True
            
            score = compute_rule_score_differentiable(
                light_probs, distances, velocities, training=False
            )
            
            if score.item() > 0.01:  # 只有非零分数才检查梯度
                score.backward()
                # 梯度应该存在且合理
                assert distances.grad is not None
                assert velocities.grad is not None
            
            # 清空梯度
            distances.grad = None
            velocities.grad = None
        
        print(f"✅ 边界情况梯度测试通过")


class TestRuleScoreBatchProcessing:
    """测试批处理功能"""
    
    def test_batch_processing(self):
        """测试批量计算"""
        batch_size = 10
        light_probs = torch.rand(batch_size, 3)
        light_probs = F.softmax(light_probs, dim=1)
        
        distances = torch.randn(batch_size) * 10  # -10到10米
        velocities = torch.rand(batch_size) * 5   # 0到5m/s
        
        scores = compute_rule_score_differentiable(
            light_probs, distances, velocities, training=False
        )
        
        # 验证输出维度
        assert scores.shape == (batch_size,), f"输出维度错误: {scores.shape}"
        
        # 验证分数范围
        assert torch.all(scores >= 0) and torch.all(scores <= 1), \
            f"分数超出[0,1]范围: min={scores.min():.4f}, max={scores.max():.4f}"
        
        print(f"✅ 批处理测试通过: batch_size={batch_size}")
    
    def test_batch_with_details(self):
        """测试带详细信息的批处理"""
        light_probs = torch.tensor([
            [0.9, 0.05, 0.05],  # 红灯
            [0.05, 0.05, 0.9],  # 绿灯
        ])
        distances = torch.tensor([3.0, -2.0])
        velocities = torch.tensor([2.0, 2.0])
        
        result = compute_rule_score_batch(light_probs, distances, velocities)
        
        # 验证返回字段
        assert 'scores' in result
        assert 'light_weights' in result
        assert 'distance_scores' in result
        assert 'velocity_scores' in result
        assert 'f_dv' in result
        assert 'violation_mask' in result
        
        # 验证violation_mask
        # 第1个：红灯+接近+高速 → 违规
        # 第2个：绿灯 → 不违规
        assert result['violation_mask'][0] == True, "红灯冲向停止线应判违规"
        assert result['violation_mask'][1] == False, "绿灯通过应判正常"
        
        print(f"✅ 详细信息测试通过")
        print(f"   分数: {result['scores']}")
        print(f"   违规mask: {result['violation_mask']}")


class TestRedLightRuleEngine:
    """测试规则引擎类"""
    
    def test_engine_initialization(self):
        """测试引擎初始化"""
        # 默认配置
        engine1 = RedLightRuleEngine()
        assert engine1.config.tau_d == 5.0
        
        # 自定义配置
        custom_config = RuleConfig(tau_d=10.0, alpha_d=3.0)
        engine2 = RedLightRuleEngine(custom_config)
        assert engine2.config.tau_d == 10.0
        assert engine2.config.alpha_d == 3.0
        
        print(f"✅ 引擎初始化测试通过")
    
    def test_engine_evaluate(self):
        """测试引擎评估功能"""
        engine = RedLightRuleEngine()
        
        light_probs = torch.tensor([[0.9, 0.05, 0.05]])
        distances = torch.tensor([3.0])
        velocities = torch.tensor([2.0])
        
        # 简单评估
        score = engine.evaluate(light_probs, distances, velocities)
        assert score.shape == (1,)
        assert 0 <= score.item() <= 1
        
        # 详细评估
        details = engine.evaluate(
            light_probs, distances, velocities, return_details=True
        )
        assert 'scores' in details
        assert 'light_weights' in details
        
        print(f"✅ 引擎评估测试通过")
    
    def test_hard_violation_check(self):
        """测试硬阈值违规检测"""
        engine = RedLightRuleEngine()
        
        # 场景1：红灯停车（正常）
        assert engine.hard_violation_check('red', 10.0, 0.0) == False
        
        # 场景2：红灯闯过停止线（违规）
        assert engine.hard_violation_check('red', -2.0, 2.0) == True
        
        # 场景3：红灯接近且速度快（违规）
        assert engine.hard_violation_check('red', 3.0, 2.0) == True
        
        # 场景4：绿灯通过（正常）
        assert engine.hard_violation_check('green', -2.0, 2.0) == False
        
        # 场景5：红灯接近但速度低（正常）
        assert engine.hard_violation_check('red', 3.0, 0.3) == False
        
        print(f"✅ 硬阈值检测测试通过")
    
    def test_violation_explanation(self):
        """测试违规解释生成"""
        engine = RedLightRuleEngine()
        
        # 场景1：闯红灯
        explanation = engine.get_violation_explanation(
            'red', -2.0, 2.0, 0.95
        )
        assert '红灯' in explanation
        assert '闯过停止线' in explanation
        assert '速度' in explanation
        
        # 场景2：正常
        explanation = engine.get_violation_explanation(
            'green', 10.0, 0.0, 0.05
        )
        assert '正常行驶' in explanation
        
        print(f"✅ 违规解释测试通过")
        print(f"   示例: {explanation}")
    
    def test_config_update(self):
        """测试配置动态更新"""
        engine = RedLightRuleEngine()
        
        # 更新配置
        engine.update_config(tau_d=8.0, alpha_v=10.0)
        assert engine.config.tau_d == 8.0
        assert engine.config.alpha_v == 10.0
        
        # 测试无效参数
        try:
            engine.update_config(invalid_param=100)
            assert False, "应该抛出ValueError"
        except ValueError:
            pass  # 预期的异常
        
        print(f"✅ 配置更新测试通过")


class TestNumericalStability:
    """测试数值稳定性"""
    
    def test_no_nan_or_inf(self):
        """测试无NaN或Inf"""
        # 极端输入
        light_probs = torch.tensor([
            [1.0, 0.0, 0.0],   # 完全红灯
            [0.0, 0.0, 1.0],   # 完全绿灯
        ])
        distances = torch.tensor([-100.0, 100.0])  # 极端距离
        velocities = torch.tensor([0.0, 100.0])    # 极端速度
        
        scores = compute_rule_score_differentiable(
            light_probs, distances, velocities, training=False
        )
        
        # 验证无NaN或Inf
        assert not torch.any(torch.isnan(scores)), "出现NaN"
        assert not torch.any(torch.isinf(scores)), "出现Inf"
        assert torch.all(scores >= 0) and torch.all(scores <= 1), "分数超出范围"
        
        print(f"✅ 数值稳定性测试通过")
    
    def test_gradient_stability(self):
        """测试梯度数值稳定性"""
        light_probs = torch.tensor([[0.5, 0.3, 0.2]], requires_grad=True)
        distances = torch.tensor([2.5], requires_grad=True)
        velocities = torch.tensor([1.5], requires_grad=True)
        
        score = compute_rule_score_differentiable(
            light_probs, distances, velocities, training=False
        )
        score.backward()
        
        # 梯度不应过大或过小
        assert distances.grad.abs() < 100, f"距离梯度过大: {distances.grad.item()}"
        assert velocities.grad.abs() < 100, f"速度梯度过大: {velocities.grad.item()}"
        
        print(f"✅ 梯度稳定性测试通过")


# ============ 性能测试 ============
class TestPerformance:
    """测试性能"""
    
    def test_batch_performance(self):
        """测试批处理性能"""
        import time
        
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            light_probs = torch.rand(batch_size, 3)
            light_probs = F.softmax(light_probs, dim=1)
            distances = torch.randn(batch_size) * 10
            velocities = torch.rand(batch_size) * 5
            
            # GPU测试（如果可用）
            if torch.cuda.is_available():
                light_probs = light_probs.cuda()
                distances = distances.cuda()
                velocities = velocities.cuda()
                
                start = time.time()
                for _ in range(100):
                    score = compute_rule_score_differentiable(
                        light_probs, distances, velocities, training=False
                    )
                elapsed = time.time() - start
                
                print(f"✅ GPU性能测试: batch_size={batch_size}, "
                      f"100次耗时={elapsed*1000:.2f}ms, "
                      f"单次={elapsed*10:.2f}ms")
            else:
                print(f"⚠️  GPU不可用，跳过性能测试")


# ============ 主测试运行器 ============
if __name__ == "__main__":
    print("="*70)
    print("规则评分函数单元测试")
    print("基于: Design-ITER-2025-01.md v2.0 §3.4.1")
    print("="*70)
    
    # 运行所有测试
    import torch.nn.functional as F
    
    print("\n[1/5] 边界条件测试")
    print("-"*70)
    boundary_tests = TestRuleScoreBoundaryConditions()
    boundary_tests.test_complete_stop_far_from_stopline()
    boundary_tests.test_complete_stop_near_stopline()
    boundary_tests.test_crossed_stopline_moving()
    boundary_tests.test_approaching_fast()
    boundary_tests.test_green_light_pass()
    boundary_tests.test_far_from_stopline_any_speed()
    
    print("\n[2/5] 梯度测试")
    print("-"*70)
    gradient_tests = TestRuleScoreGradient()
    gradient_tests.test_gradient_nonzero_all_inputs()
    gradient_tests.test_gradient_flow_edge_cases()
    
    print("\n[3/5] 批处理测试")
    print("-"*70)
    batch_tests = TestRuleScoreBatchProcessing()
    batch_tests.test_batch_processing()
    batch_tests.test_batch_with_details()
    
    print("\n[4/5] 规则引擎测试")
    print("-"*70)
    engine_tests = TestRedLightRuleEngine()
    engine_tests.test_engine_initialization()
    engine_tests.test_engine_evaluate()
    engine_tests.test_hard_violation_check()
    engine_tests.test_violation_explanation()
    engine_tests.test_config_update()
    
    print("\n[5/5] 数值稳定性测试")
    print("-"*70)
    stability_tests = TestNumericalStability()
    stability_tests.test_no_nan_or_inf()
    stability_tests.test_gradient_stability()
    
    print("\n[性能测试]")
    print("-"*70)
    perf_tests = TestPerformance()
    perf_tests.test_batch_performance()
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！规则评分函数实现正确。")
    print("="*70)

