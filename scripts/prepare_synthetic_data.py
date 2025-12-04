#!/usr/bin/env python3
"""
合成数据生成脚本

生成交通场景合成数据（红灯停/闯/绿灯通过）

设计依据：Design-ITER-2025-01.md v2.0 §3.1
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import argparse
from typing import List, Dict, Any
from datetime import datetime
import random


def generate_parking_scenario(scene_id: str) -> Dict[str, Any]:
    """
    生成红灯停车场景（无违规）
    
    场景描述：
        - 红灯状态
        - 车辆停在停止线前 5-10米
        - 速度 < 0.5 m/s（接近静止）
    
    Args:
        scene_id: 场景ID
    
    Returns:
        scene_dict: 场景数据字典
    """
    # 停止线位置（水平线，y=100）
    stop_line_y = 100.0
    stop_line = {
        "id": 0,
        "type": "stop",
        "pos": [50.0, stop_line_y],
        "end_pos": [150.0, stop_line_y],
    }
    
    # 交通灯（红灯）
    traffic_light = {
        "id": 1,
        "type": "light",
        "pos": [100.0, 110.0],  # 在停止线后方10米
        "light_state": "red",
        "confidence": 0.95,
    }
    
    # 生成3-5辆车，都停在停止线前
    num_cars = random.randint(3, 5)
    vehicles = []
    
    for i in range(num_cars):
        # 车辆停在停止线前5-15米
        distance_before_line = random.uniform(5.0, 15.0)
        car_y = stop_line_y - distance_before_line
        car_x = random.uniform(40.0, 160.0)
        
        # 速度接近0
        velocity = random.uniform(0.0, 0.3)
        
        vehicle = {
            "id": 2 + i,
            "type": "car",
            "pos": [car_x, car_y],
            "velocity": velocity,
            "bbox": [car_x - 2, car_y - 4, car_x + 2, car_y + 4],
            "d_stop": distance_before_line,  # 将在数据加载时重新计算
        }
        vehicles.append(vehicle)
    
    # 组装场景
    scene = {
        "scene_id": scene_id,
        "timestamp": datetime.now().timestamp(),
        "entities": [stop_line, traffic_light] + vehicles,
        "metadata": {
            "scenario": "parking",
            "description": "Red light - vehicles parked before stop line",
            "expected_violations": 0,
        }
    }
    
    return scene


def generate_violation_scenario(scene_id: str) -> Dict[str, Any]:
    """
    生成红灯违规场景（闯红灯）
    
    场景描述：
        - 红灯状态
        - 至少1辆车已过停止线或接近停止线且速度快
        - 速度 > 1.0 m/s
    
    Args:
        scene_id: 场景ID
    
    Returns:
        scene_dict: 场景数据字典
    """
    # 停止线
    stop_line_y = 100.0
    stop_line = {
        "id": 0,
        "type": "stop",
        "pos": [50.0, stop_line_y],
        "end_pos": [150.0, stop_line_y],
    }
    
    # 交通灯（红灯）
    traffic_light = {
        "id": 1,
        "type": "light",
        "pos": [100.0, 110.0],
        "light_state": "red",
        "confidence": 0.98,
    }
    
    # 生成2-4辆车
    num_cars = random.randint(2, 4)
    vehicles = []
    
    for i in range(num_cars):
        if i == 0:
            # 第1辆车：严重违规（已过停止线）
            distance_after_line = random.uniform(2.0, 10.0)
            car_y = stop_line_y + distance_after_line
            velocity = random.uniform(1.5, 3.0)
            d_stop = -distance_after_line  # 负数表示已过线
        elif i == 1 and random.random() < 0.7:
            # 第2辆车：接近停止线且速度快（70%概率）
            distance_before_line = random.uniform(1.0, 4.0)
            car_y = stop_line_y - distance_before_line
            velocity = random.uniform(1.2, 2.5)
            d_stop = distance_before_line
        else:
            # 其他车：正常停车
            distance_before_line = random.uniform(8.0, 20.0)
            car_y = stop_line_y - distance_before_line
            velocity = random.uniform(0.0, 0.5)
            d_stop = distance_before_line
        
        car_x = random.uniform(40.0, 160.0)
        
        vehicle = {
            "id": 2 + i,
            "type": "car",
            "pos": [car_x, car_y],
            "velocity": velocity,
            "bbox": [car_x - 2, car_y - 4, car_x + 2, car_y + 4],
            "d_stop": d_stop,
        }
        vehicles.append(vehicle)
    
    # 组装场景
    scene = {
        "scene_id": scene_id,
        "timestamp": datetime.now().timestamp(),
        "entities": [stop_line, traffic_light] + vehicles,
        "metadata": {
            "scenario": "violation",
            "description": "Red light - at least one vehicle violating",
            "expected_violations": 1,  # 至少1辆违规
        }
    }
    
    return scene


def generate_green_pass_scenario(scene_id: str) -> Dict[str, Any]:
    """
    生成绿灯通过场景（无违规）
    
    场景描述：
        - 绿灯状态
        - 车辆正常速度通过停止线
        - 速度 1.0-3.0 m/s（正常行驶）
    
    Args:
        scene_id: 场景ID
    
    Returns:
        scene_dict: 场景数据字典
    """
    # 停止线
    stop_line_y = 100.0
    stop_line = {
        "id": 0,
        "type": "stop",
        "pos": [50.0, stop_line_y],
        "end_pos": [150.0, stop_line_y],
    }
    
    # 交通灯（绿灯）
    traffic_light = {
        "id": 1,
        "type": "light",
        "pos": [100.0, 110.0],
        "light_state": "green",
        "confidence": 0.97,
    }
    
    # 生成2-4辆车，正常通过
    num_cars = random.randint(2, 4)
    vehicles = []
    
    for i in range(num_cars):
        # 车辆可能在停止线前后
        if random.random() < 0.5:
            # 已过停止线
            distance = random.uniform(0.0, 20.0)
            car_y = stop_line_y + distance
            d_stop = -distance
        else:
            # 接近停止线
            distance = random.uniform(0.0, 10.0)
            car_y = stop_line_y - distance
            d_stop = distance
        
        # 正常速度
        velocity = random.uniform(1.5, 3.5)
        car_x = random.uniform(40.0, 160.0)
        
        vehicle = {
            "id": 2 + i,
            "type": "car",
            "pos": [car_x, car_y],
            "velocity": velocity,
            "bbox": [car_x - 2, car_y - 4, car_x + 2, car_y + 4],
            "d_stop": d_stop,
        }
        vehicles.append(vehicle)
    
    # 组装场景
    scene = {
        "scene_id": scene_id,
        "timestamp": datetime.now().timestamp(),
        "entities": [stop_line, traffic_light] + vehicles,
        "metadata": {
            "scenario": "green_pass",
            "description": "Green light - vehicles passing normally",
            "expected_violations": 0,
        }
    }
    
    return scene


def generate_dataset(
    num_scenes: int = 100,
    output_dir: str = "data/synthetic",
    train_ratio: float = 0.8,
):
    """
    批量生成合成数据集
    
    Args:
        num_scenes: 总场景数
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    output_path = Path(output_dir)
    train_path = output_path / "train"
    val_path = output_path / "val"
    
    # 创建目录
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # 场景类型分布
    num_parking = int(num_scenes * 0.35)
    num_violation = int(num_scenes * 0.35)
    num_green_pass = num_scenes - num_parking - num_violation
    
    print(f"\n生成合成数据集...")
    print(f"  总场景数: {num_scenes}")
    print(f"  - 红灯停车: {num_parking}")
    print(f"  - 红灯违规: {num_violation}")
    print(f"  - 绿灯通过: {num_green_pass}")
    print(f"  训练集比例: {train_ratio:.1%}\n")
    
    # 生成场景
    scenes = []
    scene_id = 0
    
    # 生成各类场景
    for _ in range(num_parking):
        scene = generate_parking_scenario(f"scene_{scene_id:04d}")
        scenes.append(scene)
        scene_id += 1
    
    for _ in range(num_violation):
        scene = generate_violation_scenario(f"scene_{scene_id:04d}")
        scenes.append(scene)
        scene_id += 1
    
    for _ in range(num_green_pass):
        scene = generate_green_pass_scenario(f"scene_{scene_id:04d}")
        scenes.append(scene)
        scene_id += 1
    
    # 打乱顺序
    random.shuffle(scenes)
    
    # 分割训练/验证集
    split_idx = int(len(scenes) * train_ratio)
    train_scenes = scenes[:split_idx]
    val_scenes = scenes[split_idx:]
    
    # 保存
    for scene in train_scenes:
        scene_file = train_path / f"{scene['scene_id']}.json"
        with open(scene_file, 'w') as f:
            json.dump(scene, f, indent=2)
    
    for scene in val_scenes:
        scene_file = val_path / f"{scene['scene_id']}.json"
        with open(scene_file, 'w') as f:
            json.dump(scene, f, indent=2)
    
    print(f"✅ 数据生成完成！")
    print(f"  训练集: {len(train_scenes)} 场景 → {train_path}")
    print(f"  验证集: {len(val_scenes)} 场景 → {val_path}")
    
    # 生成元数据
    metadata = {
        "total_scenes": num_scenes,
        "train_scenes": len(train_scenes),
        "val_scenes": len(val_scenes),
        "scenario_distribution": {
            "parking": num_parking,
            "violation": num_violation,
            "green_pass": num_green_pass,
        },
        "generated_at": datetime.now().isoformat(),
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n元数据保存至: {metadata_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成交通场景合成数据"
    )
    parser.add_argument(
        '--num-scenes',
        type=int,
        default=100,
        help='总场景数（默认100）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/synthetic',
        help='输出目录（默认data/synthetic）'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='训练集比例（默认0.8）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认42）'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 生成数据
    generate_dataset(
        num_scenes=args.num_scenes,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
