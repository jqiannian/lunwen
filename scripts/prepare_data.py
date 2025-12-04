#!/usr/bin/env python3
"""
数据准备脚本

功能：
1. 解压BDD100K数据集（自动检测zip文件并解压）
2. 生成合成数据（用于MVP快速验证）
3. 数据统计与验证

Usage:
    # 解压BDD100K
    python scripts/prepare_data.py --task extract_bdd100k
    
    # 生成合成数据
    python scripts/prepare_data.py --task generate_synthetic --num-scenes 100
    
    # 完整流程
    python scripts/prepare_data.py --task all
"""

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# 添加项目路径以导入logger
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.traffic_rules.utils.logger import get_logger

logger = get_logger(__name__)


def extract_bdd100k(data_root: Path, keep_zip: bool = True):
    """
    解压BDD100K数据集并规整目录结构
    
    功能：
    1. 自动检测zip文件（支持多种命名）
    2. 解压到标准目录结构
    3. 验证解压结果
    4. 规整目录（确保路径一致）
    
    Args:
        data_root: 数据根目录
        keep_zip: 是否保留原始zip文件（解压后删除）
    """
    logger.info("=" * 60)
    logger.info("解压 BDD100K 数据集")
    logger.info("=" * 60)
    
    bdd_dir = data_root / "Obeject Detect" / "BDD100K"
    
    if not bdd_dir.exists():
        logger.error("BDD100K目录不存在", path=str(bdd_dir))
        logger.info("请将BDD100K zip文件放置在以下目录:", path=str(bdd_dir))
        return
    
    # 需要解压的文件（支持多种命名）
    zip_patterns = {
        "images": ["bdd100k_images.zip", "bdd100k_images_*.zip", "images.zip"],
        "labels": ["bdd100k_labels.zip", "bdd100k_labels_*.zip", "labels.zip"],
    }
    
    # 查找zip文件
    found_zips = {}
    for target_dir, patterns in zip_patterns.items():
        for pattern in patterns:
            # 支持通配符
            if "*" in pattern:
                matches = list(bdd_dir.glob(pattern))
                if matches:
                    found_zips[target_dir] = matches[0]
                    break
            else:
                zip_path = bdd_dir / pattern
                if zip_path.exists():
                    found_zips[target_dir] = zip_path
                    break
    
    # 解压文件
    for target_dir, zip_path in found_zips.items():
        extract_to = bdd_dir / target_dir
        
        # 检查是否已解压
        if extract_to.exists() and any(extract_to.rglob("*.jpg")) or any(extract_to.rglob("*.json")):
            logger.info("目录已存在，跳过解压", target_dir=target_dir, path=str(extract_to))
            continue
        
        logger.info("开始解压", zip_file=zip_path.name, target_dir=target_dir)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 显示进度
                members = zip_ref.namelist()
                with tqdm(total=len(members), desc=f"解压 {zip_path.name}") as pbar:
                    for member in members:
                        zip_ref.extract(member, bdd_dir)
                        pbar.update(1)
            
            logger.info("解压完成", target_dir=target_dir, path=str(extract_to))
            
            # 如果不需要保留zip文件
            if not keep_zip:
                zip_path.unlink()
                logger.info("已删除zip文件", zip_file=zip_path.name)
        
        except zipfile.BadZipFile:
            logger.error("zip文件损坏", zip_file=zip_path.name)
            continue
        except Exception as e:
            logger.error("解压失败", zip_file=zip_path.name, error=str(e))
            continue
    
    # 规整目录结构
    logger.info("规整目录结构...")
    _organize_bdd100k_structure(bdd_dir)
    
    # 验证解压结果
    logger.info("验证解压结果...")
    images_dir = bdd_dir / "images" / "100k" / "train"
    labels_file = bdd_dir / "labels" / "bdd100k_labels_images_train.json"
    
    if images_dir.exists():
        num_images = len(list(images_dir.glob("*.jpg")))
        logger.info("训练图像", num_images=num_images, path=str(images_dir))
    else:
        logger.warning("训练图像目录不存在", path=str(images_dir))
    
    if labels_file.exists():
        with open(labels_file, "r") as f:
            labels = json.load(f)
        logger.info("训练标注", num_labels=len(labels), path=str(labels_file))
    else:
        logger.warning("训练标注文件不存在", path=str(labels_file))
    
    logger.info("=" * 60)


def _organize_bdd100k_structure(bdd_dir: Path):
    """
    规整BDD100K目录结构
    
    确保目录结构为：
    BDD100K/
    ├── images/
    │   └── 100k/
    │       ├── train/
    │       └── val/
    └── labels/
        ├── bdd100k_labels_images_train.json
        └── bdd100k_labels_images_val.json
    """
    # 查找可能的图像目录
    possible_image_dirs = [
        bdd_dir / "images" / "100k",
        bdd_dir / "bdd100k" / "images" / "100k",
        bdd_dir / "100k" / "images",
    ]
    
    for possible_dir in possible_image_dirs:
        if possible_dir.exists():
            # 移动到标准位置
            target_dir = bdd_dir / "images" / "100k"
            if possible_dir != target_dir:
                logger.info("移动图像目录", from_path=str(possible_dir), to_path=str(target_dir))
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(possible_dir), str(target_dir))
            break
    
    # 查找可能的标注目录
    possible_label_dirs = [
        bdd_dir / "labels",
        bdd_dir / "bdd100k" / "labels",
    ]
    
    for possible_dir in possible_label_dirs:
        if possible_dir.exists() and any(possible_dir.glob("*.json")):
            # 移动到标准位置
            target_dir = bdd_dir / "labels"
            if possible_dir != target_dir:
                logger.info("移动标注目录", from_path=str(possible_dir), to_path=str(target_dir))
                target_dir.mkdir(parents=True, exist_ok=True)
                # 移动所有json文件
                for json_file in possible_dir.glob("*.json"):
                    shutil.move(str(json_file), str(target_dir / json_file.name))
            break


def generate_synthetic_scene(
    scene_id: str,
    image_size: tuple[int, int] = (720, 1280),
) -> tuple[np.ndarray, dict]:
    """
    生成单个合成场景
    
    Args:
        scene_id: 场景ID
        image_size: 图像尺寸(H, W)
    
    Returns:
        image: 合成图像
        scene_data: 场景元数据
    """
    h, w = image_size
    
    # 创建空白图像（灰色道路背景）
    image = np.ones((h, w, 3), dtype=np.uint8) * 128
    
    # 绘制道路
    cv2.rectangle(image, (0, int(h*0.3)), (w, h), (80, 80, 80), -1)
    
    # 绘制车道线
    for i in range(3):
        y = int(h * (0.5 + i * 0.15))
        cv2.line(image, (0, y), (w, y), (255, 255, 255), 2, cv2.LINE_AA)
    
    # 随机生成场景类型
    scene_type = random.choice(["parking", "violation", "green_pass"])
    
    entities = []
    
    # 1. 生成交通灯
    light_x, light_y = w // 4, int(h * 0.25)
    light_state = "red" if scene_type in ["parking", "violation"] else "green"
    light_color = (0, 0, 255) if light_state == "red" else (0, 255, 0)
    
    cv2.circle(image, (light_x, light_y), 15, light_color, -1)
    cv2.circle(image, (light_x, light_y), 17, (255, 255, 255), 2)
    
    entities.append({
        "id": "light_1",
        "type": "light",
        "position": [light_x, light_y],
        "bbox": [light_x-20, light_y-20, light_x+20, light_y+20],
        "light_state": light_state,
        "light_confidence": 1.0,
    })
    
    # 2. 生成停止线
    stopline_y = int(h * 0.7)
    cv2.line(image, (0, stopline_y), (w, stopline_y), (255, 255, 0), 3)
    
    entities.append({
        "id": "stopline_1",
        "type": "stop",
        "position": [w//2, stopline_y],
        "line_endpoints": [[0, stopline_y], [w, stopline_y]],
    })
    
    # 3. 生成车辆
    if scene_type == "parking":
        # 红灯停车：车辆停在停止线前
        car_x, car_y = w // 2, stopline_y - 100
        car_velocity = 0.0
    elif scene_type == "violation":
        # 红灯闯行：车辆越过停止线
        car_x, car_y = w // 2, stopline_y + 50
        car_velocity = 5.0
    else:  # green_pass
        # 绿灯通过：车辆正常通过
        car_x, car_y = w // 2, stopline_y - 50
        car_velocity = 3.0
    
    # 绘制车辆（简化为矩形）
    car_w, car_h = 80, 120
    cv2.rectangle(
        image, 
        (car_x - car_w//2, car_y - car_h//2),
        (car_x + car_w//2, car_y + car_h//2),
        (0, 0, 255) if scene_type == "violation" else (200, 200, 200),
        -1
    )
    cv2.rectangle(
        image, 
        (car_x - car_w//2, car_y - car_h//2),
        (car_x + car_w//2, car_y + car_h//2),
        (0, 0, 0),
        2
    )
    
    # 计算距离停止线的距离（像素）
    distance_pixels = abs(car_y - stopline_y)
    distance_meters = distance_pixels * 0.05  # 假设1像素=0.05米
    
    entities.append({
        "id": "car_1",
        "type": "car",
        "position": [car_x, car_y],
        "bbox": [car_x-car_w//2, car_y-car_h//2, car_x+car_w//2, car_y+car_h//2],
        "velocity": car_velocity,
        "heading": 0.0,
        "distance_to_stopline": distance_meters,
    })
    
    # 添加文字标注（场景类型）
    cv2.putText(
        image, 
        f"Scene: {scene_type}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.0, 
        (255, 255, 255), 
        2
    )
    cv2.putText(
        image, 
        f"Light: {light_state}", 
        (10, 60), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        light_color, 
        2
    )
    cv2.putText(
        image, 
        f"Speed: {car_velocity:.1f} m/s", 
        (10, 90), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (255, 255, 255), 
        2
    )
    cv2.putText(
        image, 
        f"Distance: {distance_meters:.1f} m", 
        (10, 120), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (255, 255, 255), 
        2
    )
    
    # 场景元数据
    scene_data = {
        "scene_id": scene_id,
        "scene_type": scene_type,
        "timestamp": 0.0,
        "entities": entities,
        "image_size": [h, w],
    }
    
    return image, scene_data


def generate_synthetic_data(
    data_root: Path,
    num_scenes: int = 100,
    train_ratio: float = 0.8,
):
    """
    生成合成数据集
    
    Args:
        data_root: 数据根目录
        num_scenes: 生成场景数量
        train_ratio: 训练集比例
    """
    print("\n" + "=" * 60)
    print(f"生成合成数据集（{num_scenes}个场景）")
    print("=" * 60)
    
    synthetic_dir = data_root / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建train/val目录
    train_dir = synthetic_dir / "train"
    val_dir = synthetic_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # 场景类型分布
    scene_types = ["parking", "violation", "green_pass"]
    scene_type_counts = {t: 0 for t in scene_types}
    
    # 生成场景
    for i in tqdm(range(num_scenes), desc="生成场景"):
        scene_id = f"scene_{i:04d}"
        
        # 生成场景
        image, scene_data = generate_synthetic_scene(scene_id)
        
        # 决定train/val
        split_dir = train_dir if random.random() < train_ratio else val_dir
        
        # 保存图像
        image_path = split_dir / f"{scene_id}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # 保存元数据
        json_path = split_dir / f"{scene_id}.json"
        with open(json_path, "w") as f:
            json.dump(scene_data, f, indent=2)
        
        # 统计
        scene_type_counts[scene_data["scene_type"]] += 1
    
    # 统计信息
    train_count = len(list(train_dir.glob("*.png")))
    val_count = len(list(val_dir.glob("*.png")))
    
    logger.info("生成完成", train_count=train_count, val_count=val_count)
    logger.info("场景类型分布", **scene_type_counts)
    
    logger.info("=" * 60)


def data_statistics(data_root: Path):
    """数据统计"""
    logger.info("=" * 60)
    logger.info("数据统计")
    logger.info("=" * 60)
    
    # Synthetic数据
    synthetic_dir = data_root / "synthetic"
    if synthetic_dir.exists():
        logger.info("Synthetic数据")
        for split in ["train", "val"]:
            split_dir = synthetic_dir / split
            if split_dir.exists():
                num_scenes = len(list(split_dir.glob("*.png")))
                logger.info(f"{split}集", num_scenes=num_scenes)
    
    # BDD100K数据
    bdd_dir = data_root / "Obeject Detect" / "BDD100K"
    if bdd_dir.exists():
        logger.info("BDD100K数据")
        
        # 图像
        images_dir = bdd_dir / "images" / "100k"
        if images_dir.exists():
            for split in ["train", "val"]:
                split_dir = images_dir / split
                if split_dir.exists():
                    num_images = len(list(split_dir.glob("*.jpg")))
                    logger.info(f"{split}集图像", num_images=num_images)
        
        # 标注
        labels_dir = bdd_dir / "labels"
        if labels_dir.exists():
            for split in ["train", "val"]:
                labels_file = labels_dir / f"bdd100k_labels_images_{split}.json"
                if labels_file.exists():
                    with open(labels_file, "r") as f:
                        labels = json.load(f)
                    logger.info(f"{split}集标注", num_labels=len(labels))
    
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="数据准备脚本")
    parser.add_argument(
        "--task",
        type=str,
        choices=["extract_bdd100k", "generate_synthetic", "statistics", "all"],
        default="generate_synthetic",
        help="任务类型"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="数据根目录"
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=100,
        help="生成合成数据场景数"
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="保留原始zip文件"
    )
    
    args = parser.parse_args()
    data_root = Path(args.data_root)
    
    if args.task == "extract_bdd100k":
        extract_bdd100k(data_root, args.keep_zip)
    
    elif args.task == "generate_synthetic":
        generate_synthetic_data(data_root, args.num_scenes)
    
    elif args.task == "statistics":
        data_statistics(data_root)
    
    elif args.task == "all":
        extract_bdd100k(data_root, args.keep_zip)
        generate_synthetic_data(data_root, args.num_scenes)
        data_statistics(data_root)
    
    logger.info("数据准备完成！")
    logger.info("下一步：测试数据加载器", command="python -m src.traffic_rules.data.traffic_dataset")


if __name__ == "__main__":
    main()

