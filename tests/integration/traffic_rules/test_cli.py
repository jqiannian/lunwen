"""
CLI 端到端集成测试
"""

import sys
import subprocess
from pathlib import Path
import json

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCLIIntegration:
    """测试CLI端到端流程"""
    
    def test_train_smoke_test(self):
        """测试训练smoke test（快速验证）"""
        print("\n[测试] 训练smoke test（2 epochs, 5 samples）")
        
        result = subprocess.run(
            [
                'python3', 'tools/train_red_light.py', 'train',
                '--epochs', '2',
                '--max-samples', '5',
                '--device', 'cpu',
            ],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0, f"训练失败: {result.stderr}"
        
        # 验证checkpoint生成
        checkpoint_path = project_root / 'artifacts/checkpoints/best.pth'
        assert checkpoint_path.exists(), "未生成best.pth"
        
        print(f"  ✅ 训练成功，checkpoint已生成")
    
    def test_test_all_scenarios(self):
        """测试三场景验收测试"""
        print("\n[测试] 三场景验收测试")
        
        # 确保有checkpoint
        checkpoint_path = project_root / 'artifacts/checkpoints/best.pth'
        if not checkpoint_path.exists():
            print("  ⚠️ 跳过：需要先运行训练")
            return
        
        result = subprocess.run(
            [
                'python3', 'tools/test_red_light.py',
                '--scenario', 'all',
                '--split', 'val',
            ],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0, f"测试失败: {result.stderr}"
        
        # 验证报告生成
        summary_path = project_root / 'reports/testing/scenario_summary.json'
        assert summary_path.exists(), "未生成scenario_summary.json"
        
        # 读取并验证报告内容
        summary = json.loads(summary_path.read_text())
        assert 'parking' in summary, "缺少parking统计"
        assert 'violation' in summary, "缺少violation统计"
        assert 'green_pass' in summary, "缺少green_pass统计"
        
        print(f"  ✅ 三场景测试成功")
        print(f"     - parking: {summary['parking']['total_scenes']}个场景")
        print(f"     - violation: {summary['violation']['total_scenes']}个场景")
        print(f"     - green_pass: {summary['green_pass']['total_scenes']}个场景")
    
    def test_scenario_filtering(self):
        """测试场景过滤功能"""
        print("\n[测试] 场景过滤功能")
        
        checkpoint_path = project_root / 'artifacts/checkpoints/best.pth'
        if not checkpoint_path.exists():
            print("  ⚠️ 跳过：需要先运行训练")
            return
        
        for scenario in ['parking', 'violation', 'green_pass']:
            result = subprocess.run(
                [
                    'python3', 'tools/test_red_light.py',
                    '--scenario', scenario,
                    '--split', 'val',
                ],
                cwd=str(project_root),
                capture_output=True,
                text=True,
            )
            
            assert result.returncode == 0, f"{scenario}场景测试失败"
            assert f"场景过滤: {scenario}" in result.stdout, "输出应显示过滤类型"
        
        print(f"  ✅ 场景过滤功能正常（parking/violation/green_pass）")
    
    def test_heatmap_generation(self):
        """测试热力图生成"""
        print("\n[测试] 批量热力图生成")
        
        checkpoint_path = project_root / 'artifacts/checkpoints/best.pth'
        if not checkpoint_path.exists():
            print("  ⚠️ 跳过：需要先运行训练")
            return
        
        result = subprocess.run(
            [
                'python3', 'scripts/render_attention_maps.py',
                '--output-dir', 'reports/testing/heatmaps_test',
            ],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0, f"热力图生成失败: {result.stderr}"
        
        # 验证HTML索引生成
        index_path = project_root / 'reports/testing/heatmaps_test/index.html'
        assert index_path.exists(), "未生成index.html"
        
        print(f"  ✅ 热力图生成成功")
    
    def test_acceptance_report_generation(self):
        """测试验收报告生成"""
        print("\n[测试] 验收报告生成")
        
        # 确保有测试结果
        summary_path = project_root / 'reports/testing/summary.json'
        if not summary_path.exists():
            print("  ⚠️ 跳过：需要先运行测试")
            return
        
        result = subprocess.run(
            [
                'python3', 'tools/generate_acceptance_report.py',
                '--test-results', 'reports/testing',
                '--output', 'reports/ACCEPTANCE_REPORT_TEST.md',
            ],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0, f"报告生成失败: {result.stderr}"
        
        # 验证报告文件
        report_path = project_root / 'reports/ACCEPTANCE_REPORT_TEST.md'
        assert report_path.exists(), "未生成验收报告"
        
        content = report_path.read_text()
        assert '场景分类统计' in content, "报告应包含场景统计"
        assert 'parking' in content, "报告应包含parking"
        assert 'violation' in content, "报告应包含violation"
        
        print(f"  ✅ 验收报告生成成功")


if __name__ == "__main__":
    test = TestCLIIntegration()
    
    print("="*60)
    print("端到端集成测试")
    print("="*60)
    
    test.test_train_smoke_test()
    test.test_test_all_scenarios()
    test.test_scenario_filtering()
    test.test_heatmap_generation()
    test.test_acceptance_report_generation()
    
    print("\n" + "="*60)
    print("✅ 所有集成测试通过！")
    print("="*60)
