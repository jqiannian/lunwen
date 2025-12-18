#!/usr/bin/env python3
"""
验收报告自动化生成脚本

功能：
1. 读取test_red_light.py输出的JSON证据链
2. 按scenario分类统计
3. 计算各场景准确率、召回率等指标
4. 生成Markdown格式的验收报告
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_scene_reports(test_results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """
    加载所有场景报告并按scenario分类
    
    Args:
        test_results_dir: 测试结果目录（包含scene_*.json文件）
    
    Returns:
        scenarios: {scenario_type: [scene_report, ...]}
    """
    scenarios = defaultdict(list)
    
    # 读取所有scene_*.json文件
    scene_files = sorted(test_results_dir.glob("scene_*.json"))
    
    for scene_file in scene_files:
        try:
            data = json.loads(scene_file.read_text(encoding='utf-8'))
            scenario_type = data.get('scenario_type', 'unknown')
            scenarios[scenario_type].append(data)
        except Exception as e:
            print(f"警告：无法读取 {scene_file}: {e}")
    
    return dict(scenarios)


def compute_scenario_metrics(scenes: list[dict[str, Any]], scenario_type: str) -> dict[str, Any]:
    """
    计算单个场景类型的指标
    
    Args:
        scenes: 场景报告列表
        scenario_type: 场景类型
    
    Returns:
        metrics: 指标字典
    """
    total_scenes = len(scenes)
    total_violations_detected = sum(s['summary']['violations_detected'] for s in scenes)
    
    # 计算平均分数
    all_scores = []
    for scene in scenes:
        for evidence in scene['evidence']:
            all_scores.append(evidence['final_score'])
    
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    max_score = max(all_scores) if all_scores else 0.0
    min_score = min(all_scores) if all_scores else 0.0
    
    # 计算准确率（根据场景类型）
    if scenario_type == "violation":
        # 违规场景：期望检测到违规
        # 准确率 = 检测到至少1个违规的场景数 / 总场景数
        scenes_with_violations = sum(1 for s in scenes if s['summary']['violations_detected'] > 0)
        accuracy = scenes_with_violations / total_scenes if total_scenes > 0 else 0.0
        
        # 召回率 = 总检测到的违规数 / 总场景数（假设每个场景至少有1个违规）
        recall = total_violations_detected / total_scenes if total_scenes > 0 else 0.0
        
        # 精确率（这里简化为1.0，因为我们只统计真正的violation场景）
        precision = 1.0
        
    elif scenario_type in ["parking", "green_pass"]:
        # 正常场景：期望不检测到违规
        # 准确率 = 未检测到违规的场景数 / 总场景数
        scenes_without_violations = sum(1 for s in scenes if s['summary']['violations_detected'] == 0)
        accuracy = scenes_without_violations / total_scenes if total_scenes > 0 else 0.0
        
        # 对于正常场景，召回率和精确率不适用
        recall = None
        precision = None
        
    else:
        accuracy = None
        recall = None
        precision = None
    
    return {
        'total_scenes': total_scenes,
        'violations_detected': total_violations_detected,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'avg_score': avg_score,
        'max_score': max_score,
        'min_score': min_score,
    }


def generate_markdown_report(
    scenarios: dict[str, list[dict[str, Any]]],
    output_path: Path,
    screenshots_dir: Path | None = None,
    heatmaps_index: Path | None = None,
) -> None:
    """
    生成Markdown格式的验收报告
    
    Args:
        scenarios: 按scenario分类的场景报告
        output_path: 报告输出路径
        screenshots_dir: 截图目录（可选）
        heatmaps_index: 热力图索引页路径（可选）
    """
    lines = []
    
    # 标题
    lines.append("# 红灯停MVP验收报告")
    lines.append("")
    lines.append(f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 总览
    total_scenes = sum(len(scenes) for scenes in scenarios.values())
    total_violations = sum(
        sum(s['summary']['violations_detected'] for s in scenes)
        for scenes in scenarios.values()
    )
    
    lines.append("## 📊 测试总览")
    lines.append("")
    lines.append(f"- **总场景数**：{total_scenes}")
    lines.append(f"- **总违规检出数**：{total_violations}")
    lines.append(f"- **场景类型数**：{len(scenarios)}")
    lines.append("")
    
    # 场景分类统计表
    lines.append("## 📈 场景分类统计")
    lines.append("")
    lines.append("| 场景类型 | 场景数 | 违规检出 | 准确率 | 召回率 | 平均分数 |")
    lines.append("|---------|-------|---------|--------|--------|----------|")
    
    for scenario_type in sorted(scenarios.keys()):
        scenes = scenarios[scenario_type]
        metrics = compute_scenario_metrics(scenes, scenario_type)
        
        accuracy_str = f"{metrics['accuracy']:.1%}" if metrics['accuracy'] is not None else "N/A"
        recall_str = f"{metrics['recall']:.1%}" if metrics['recall'] is not None else "N/A"
        
        lines.append(
            f"| {scenario_type} | {metrics['total_scenes']} | "
            f"{metrics['violations_detected']} | {accuracy_str} | {recall_str} | "
            f"{metrics['avg_score']:.3f} |"
        )
    
    lines.append("")
    
    # 各场景类型详情
    lines.append("## 📝 场景详情")
    lines.append("")
    
    for scenario_type in sorted(scenarios.keys()):
        scenes = scenarios[scenario_type]
        metrics = compute_scenario_metrics(scenes, scenario_type)
        
        lines.append(f"### {scenario_type.upper()} 场景")
        lines.append("")
        lines.append(f"- **场景数**：{metrics['total_scenes']}")
        lines.append(f"- **违规检出**：{metrics['violations_detected']}")
        
        if metrics['accuracy'] is not None:
            lines.append(f"- **准确率**：{metrics['accuracy']:.1%}")
        if metrics['recall'] is not None:
            lines.append(f"- **召回率**：{metrics['recall']:.1%}")
        
        lines.append(f"- **平均分数**：{metrics['avg_score']:.3f}")
        lines.append(f"- **分数范围**：[{metrics['min_score']:.3f}, {metrics['max_score']:.3f}]")
        lines.append("")
        
        # 示例场景（前3个）
        if len(scenes) > 0:
            lines.append("**示例场景**：")
            lines.append("")
            
            for i, scene in enumerate(scenes[:3]):
                scene_id = scene['scene_id']
                violations = scene['summary']['violations_detected']
                max_score = scene['summary']['max_final_score']
                
                lines.append(f"{i+1}. `{scene_id}` - 违规检出: {violations}, 最高分数: {max_score:.3f}")
                
                # 添加截图链接（如果存在）
                if screenshots_dir:
                    screenshot_path = screenshots_dir / f"{scene_id}_violation.png"
                    if screenshot_path.exists():
                        rel_path = screenshot_path.relative_to(output_path.parent)
                        lines.append(f"   - 📷 [查看截图]({rel_path})")
            
            lines.append("")
    
    # 可视化资源
    lines.append("## 🎨 可视化资源")
    lines.append("")
    
    if screenshots_dir and screenshots_dir.exists():
        screenshot_count = len(list(screenshots_dir.glob("*.png")))
        if screenshot_count > 0:
            rel_screenshots_dir = screenshots_dir.relative_to(output_path.parent)
            lines.append(f"- **违规截图**：{screenshot_count} 张")
            lines.append(f"  - 目录：`{rel_screenshots_dir}/`")
            lines.append("")
    
    if heatmaps_index and heatmaps_index.exists():
        rel_heatmaps_index = heatmaps_index.relative_to(output_path.parent)
        lines.append(f"- **注意力热力图索引**：[打开浏览]({rel_heatmaps_index})")
        lines.append("")
    
    # 验收结论
    lines.append("## ✅ 验收结论")
    lines.append("")
    
    # 检查验收标准
    violation_metrics = compute_scenario_metrics(scenarios.get('violation', []), 'violation')
    parking_metrics = compute_scenario_metrics(scenarios.get('parking', []), 'parking')
    green_metrics = compute_scenario_metrics(scenarios.get('green_pass', []), 'green_pass')
    
    lines.append("### 验收标准检查")
    lines.append("")
    
    checks = []
    
    # 标准1：violation场景召回率 >= 0.9
    if violation_metrics['recall'] is not None:
        if violation_metrics['recall'] >= 0.9:
            checks.append(("✅", f"violation场景召回率 ≥ 0.9: {violation_metrics['recall']:.1%}"))
        else:
            checks.append(("⚠️", f"violation场景召回率 < 0.9: {violation_metrics['recall']:.1%}"))
    
    # 标准2：parking/green_pass场景准确率 >= 0.95
    if parking_metrics['accuracy'] is not None:
        if parking_metrics['accuracy'] >= 0.95:
            checks.append(("✅", f"parking场景准确率 ≥ 0.95: {parking_metrics['accuracy']:.1%}"))
        else:
            checks.append(("⚠️", f"parking场景准确率 < 0.95: {parking_metrics['accuracy']:.1%}"))
    
    if green_metrics['accuracy'] is not None:
        if green_metrics['accuracy'] >= 0.95:
            checks.append(("✅", f"green_pass场景准确率 ≥ 0.95: {green_metrics['accuracy']:.1%}"))
        else:
            checks.append(("⚠️", f"green_pass场景准确率 < 0.95: {green_metrics['accuracy']:.1%}"))
    
    # 标准3：是否生成了截图和热力图
    if screenshots_dir and screenshots_dir.exists():
        screenshot_count = len(list(screenshots_dir.glob("*.png")))
        if screenshot_count > 0:
            checks.append(("✅", f"生成了 {screenshot_count} 张违规截图"))
        else:
            checks.append(("⚠️", "未生成违规截图"))
    
    if heatmaps_index and heatmaps_index.exists():
        checks.append(("✅", "生成了注意力热力图索引"))
    else:
        checks.append(("⚠️", "未生成注意力热力图"))
    
    for icon, check in checks:
        lines.append(f"- {icon} {check}")
    
    lines.append("")
    
    # 总结
    passed_count = sum(1 for icon, _ in checks if icon == "✅")
    total_count = len(checks)
    
    if passed_count == total_count:
        lines.append(f"**总体结论**：✅ **通过验收** ({passed_count}/{total_count})")
    else:
        lines.append(f"**总体结论**：⚠️ **部分通过** ({passed_count}/{total_count})")
    
    lines.append("")
    
    # 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description="生成验收报告")
    parser.add_argument(
        "--test-results",
        default=Path("reports/testing"),
        type=Path,
        help="测试结果目录（包含scene_*.json文件）",
    )
    parser.add_argument(
        "--output",
        default=Path("reports/ACCEPTANCE_REPORT.md"),
        type=Path,
        help="报告输出路径",
    )
    parser.add_argument(
        "--screenshots-dir",
        default=Path("reports/testing/screenshots"),
        type=Path,
        help="截图目录（可选）",
    )
    parser.add_argument(
        "--heatmaps-index",
        default=Path("reports/testing/heatmaps/index.html"),
        type=Path,
        help="热力图索引页路径（可选）",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("生成验收报告")
    print("=" * 60)
    
    # 加载场景报告
    print(f"加载测试结果: {args.test_results}")
    scenarios = load_scene_reports(args.test_results)
    
    if not scenarios:
        print("错误：未找到任何场景报告")
        return
    
    print(f"找到 {len(scenarios)} 种场景类型：")
    for scenario_type, scenes in scenarios.items():
        print(f"  - {scenario_type}: {len(scenes)} 个场景")
    
    # 生成报告
    print(f"\n生成报告: {args.output}")
    generate_markdown_report(
        scenarios,
        args.output,
        screenshots_dir=args.screenshots_dir if args.screenshots_dir.exists() else None,
        heatmaps_index=args.heatmaps_index if args.heatmaps_index.exists() else None,
    )
    
    print("=" * 60)
    print(f"✅ 报告已生成: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
