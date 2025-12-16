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
import os
import random
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# 添加项目路径以导入logger
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.traffic_rules.utils.logger import get_logger
from src.traffic_rules.utils.logger import setup_logger

# 仅控制台输出，避免在受限环境创建 logs/ 目录导致失败
setup_logger(__name__, enable_file=False, enable_console=True)
logger = get_logger(__name__)

def summarize_bdd100k(
    external_root: Path,
    output_dir: Path,
    max_label_files: int = 0,
    sample_images: int = 50,
):
    """
    生成 BDD100K 数据集特征 summary（从 zip 直接读取，不要求预先解压）。

    Args:
        external_root: 外部数据集根目录（应包含 BDD100K/）
        output_dir: 输出目录（例如 data/BDD100K）
        max_label_files: 最多处理多少个 label JSON。0 表示全部（80k 可能耗时较长）
        sample_images: 从 images zip 中抽样解码多少张图片用于分辨率统计
    """
    t0 = time.time()

    bdd_root = external_root / "BDD100K"
    labels_zip = bdd_root / "bdd100k_labels.zip"
    images_zip = bdd_root / "bdd100k_images.zip"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not labels_zip.exists():
        raise FileNotFoundError(f"找不到 bdd100k_labels.zip: {labels_zip}")
    if not images_zip.exists():
        raise FileNotFoundError(f"找不到 bdd100k_images.zip: {images_zip}")

    logger.info("开始生成BDD100K summary", labels_zip=str(labels_zip), images_zip=str(images_zip))

    # --------- 1) 读取 zip 成员列表（不解压）---------
    with zipfile.ZipFile(labels_zip) as zf_labels:
        label_members = [n for n in zf_labels.namelist() if n.lower().endswith(".json")]
    with zipfile.ZipFile(images_zip) as zf_images:
        image_members = [n for n in zf_images.namelist() if n.lower().endswith(".jpg")]

    # 建立 images 的快速查找集合（用于抽样一致性核对）
    image_member_set = set(image_members)

    def _split_of_label_member(name: str) -> str:
        # 典型路径：bdd100k/labels/100k/train/<id>.json
        if "/train/" in name:
            return "train"
        if "/val/" in name:
            return "val"
        if "/test/" in name:
            return "test"
        return "unknown"

    def _corresponding_image_member(label_member: str) -> str | None:
        # label: .../train/xxxx.json  -> image: bdd100k/images/100k/train/xxxx.jpg
        # 若 split 不明，返回 None
        split = _split_of_label_member(label_member)
        if split == "unknown":
            return None
        stem = Path(label_member).stem
        return f"bdd100k/images/100k/{split}/{stem}.jpg"

    label_count_by_split = Counter(_split_of_label_member(n) for n in label_members)
    image_count_by_split = Counter(
        ("train" if "/train/" in n else "val" if "/val/" in n else "test" if "/test/" in n else "unknown")
        for n in image_members
    )

    # --------- 2) 统计 labels 内容（逐 JSON 解析）---------
    scene_counter: Counter[str] = Counter()
    timeofday_counter: Counter[str] = Counter()
    weather_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    traffic_light_color_counter: Counter[str] = Counter()
    occluded_counter: Counter[str] = Counter()
    truncated_counter: Counter[str] = Counter()

    num_files_processed = 0
    num_objects_total = 0
    num_frames_total = 0
    per_image_object_counts: list[int] = []

    # bbox stats（简单统计）
    bbox_w_min = float("inf")
    bbox_h_min = float("inf")
    bbox_area_min = float("inf")
    bbox_w_max = 0.0
    bbox_h_max = 0.0
    bbox_area_max = 0.0
    bbox_w_sum = 0.0
    bbox_h_sum = 0.0
    bbox_area_sum = 0.0
    bbox_count = 0

    # 抽样一致性：labels->images 是否存在对应 jpg
    match_check_total = 0
    match_check_hit = 0

    # 决定处理多少个 label 文件
    target_members = label_members
    if max_label_files and max_label_files > 0:
        target_members = label_members[:max_label_files]

    with zipfile.ZipFile(labels_zip) as zf_labels:
        for member in tqdm(target_members, desc="解析BDD100K labels(JSON)", unit="file"):
            try:
                with zf_labels.open(member) as fp:
                    data = json.load(fp)
            except Exception as e:
                logger.warning("解析label失败，已跳过", member=member, error=str(e))
                continue

            num_files_processed += 1

            # 顶层 attributes: scene/timeofday/weather
            attrs = data.get("attributes", {}) if isinstance(data, dict) else {}
            if isinstance(attrs, dict):
                scene = attrs.get("scene")
                tod = attrs.get("timeofday")
                weather = attrs.get("weather")
                if scene:
                    scene_counter[str(scene)] += 1
                if tod:
                    timeofday_counter[str(tod)] += 1
                if weather:
                    weather_counter[str(weather)] += 1

            frames = data.get("frames", []) if isinstance(data, dict) else []
            if isinstance(frames, list):
                num_frames_total += len(frames)
                # BDD100K images label 通常只有 1 帧
                if frames:
                    f0 = frames[0]
                    objects = f0.get("objects", []) if isinstance(f0, dict) else []
                    if isinstance(objects, list):
                        per_image_object_counts.append(len(objects))
                        for obj in objects:
                            if not isinstance(obj, dict):
                                continue
                            cat = obj.get("category")
                            if cat:
                                category_counter[str(cat)] += 1
                            obj_attrs = obj.get("attributes", {})
                            if isinstance(obj_attrs, dict):
                                if "occluded" in obj_attrs:
                                    occluded_counter[str(obj_attrs.get("occluded"))] += 1
                                if "truncated" in obj_attrs:
                                    truncated_counter[str(obj_attrs.get("truncated"))] += 1
                                # trafficLightColor 只对交通灯更有意义
                                if "trafficLightColor" in obj_attrs:
                                    traffic_light_color_counter[str(obj_attrs.get("trafficLightColor"))] += 1
                            box2d = obj.get("box2d")
                            if isinstance(box2d, dict):
                                try:
                                    w = float(box2d["x2"]) - float(box2d["x1"])
                                    h = float(box2d["y2"]) - float(box2d["y1"])
                                    if w < 0 or h < 0:
                                        continue
                                    area = w * h
                                    bbox_count += 1
                                    bbox_w_sum += w
                                    bbox_h_sum += h
                                    bbox_area_sum += area
                                    bbox_w_min = min(bbox_w_min, w)
                                    bbox_h_min = min(bbox_h_min, h)
                                    bbox_area_min = min(bbox_area_min, area)
                                    bbox_w_max = max(bbox_w_max, w)
                                    bbox_h_max = max(bbox_h_max, h)
                                    bbox_area_max = max(bbox_area_max, area)
                                except Exception:
                                    pass

                            num_objects_total += 1

            # labels->images 对应性（抽样检查：每 200 个检查一次，避免太慢）
            if num_files_processed % 200 == 0:
                img_member = _corresponding_image_member(member)
                if img_member:
                    match_check_total += 1
                    if img_member in image_member_set:
                        match_check_hit += 1

    # bbox 均值
    bbox_stats = {
        "count": bbox_count,
        "width_min": (None if bbox_count == 0 else bbox_w_min),
        "width_max": (None if bbox_count == 0 else bbox_w_max),
        "width_mean": (None if bbox_count == 0 else bbox_w_sum / bbox_count),
        "height_min": (None if bbox_count == 0 else bbox_h_min),
        "height_max": (None if bbox_count == 0 else bbox_h_max),
        "height_mean": (None if bbox_count == 0 else bbox_h_sum / bbox_count),
        "area_min": (None if bbox_count == 0 else bbox_area_min),
        "area_max": (None if bbox_count == 0 else bbox_area_max),
        "area_mean": (None if bbox_count == 0 else bbox_area_sum / bbox_count),
    }

    # 每图对象数统计（均值/最小/最大）
    if per_image_object_counts:
        o_min = min(per_image_object_counts)
        o_max = max(per_image_object_counts)
        o_mean = sum(per_image_object_counts) / len(per_image_object_counts)
    else:
        o_min = o_max = o_mean = None

    # --------- 3) 抽样解码图片，统计分辨率 ---------
    resolutions: Counter[str] = Counter()
    sampled_images = 0
    decode_fail = 0

    if sample_images and sample_images > 0:
        # 从 zip 成员头部开始抽样（确定性，便于复现）
        sample_list = image_members[:sample_images]
        with zipfile.ZipFile(images_zip) as zf_images:
            for member in tqdm(sample_list, desc="抽样解码images(JPG)", unit="img"):
                try:
                    with zf_images.open(member) as fp:
                        buf = fp.read()
                    arr = np.frombuffer(buf, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        decode_fail += 1
                        continue
                    h, w = img.shape[:2]
                    resolutions[f"{w}x{h}"] += 1
                    sampled_images += 1
                except Exception:
                    decode_fail += 1
                    continue

    # --------- 4) 汇总输出 ---------
    summary = {
        "external_root": str(external_root),
        "bdd_root": str(bdd_root),
        "inputs": {
            "labels_zip": str(labels_zip),
            "images_zip": str(images_zip),
        },
        "zip_members": {
            "labels_json_total": len(label_members),
            "labels_json_by_split": dict(label_count_by_split),
            "images_jpg_total": len(image_members),
            "images_jpg_by_split": dict(image_count_by_split),
        },
        "labels_stats": {
            "processed_label_files": num_files_processed,
            "max_label_files_arg": max_label_files,
            "frames_total": num_frames_total,
            "objects_total": num_objects_total,
            "objects_per_image": {
                "count_images": len(per_image_object_counts),
                "min": o_min,
                "max": o_max,
                "mean": o_mean,
            },
            "attributes_distribution": {
                "scene": dict(scene_counter),
                "timeofday": dict(timeofday_counter),
                "weather": dict(weather_counter),
            },
            "category_distribution": dict(category_counter),
            "traffic_light_color_distribution": dict(traffic_light_color_counter),
            "occluded_distribution": dict(occluded_counter),
            "truncated_distribution": dict(truncated_counter),
            "bbox_stats": bbox_stats,
            "label_to_image_match_check": {
                "checked": match_check_total,
                "matched": match_check_hit,
                "match_rate": (None if match_check_total == 0 else match_check_hit / match_check_total),
                "note": "每解析200个label抽样检查一次对应jpg是否存在",
            },
        },
        "images_stats": {
            "sample_images_arg": sample_images,
            "sampled_images_decoded": sampled_images,
            "decode_fail": decode_fail,
            "resolution_distribution": dict(resolutions),
        },
        "runtime": {
            "seconds": time.time() - t0,
        },
        "generated_at_unix": time.time(),
    }

    json_path = output_dir / "bdd100k_summary.json"
    md_path = output_dir / "bdd100k_summary.md"

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    # Markdown（面向论文/报告更好读）
    def _topn(counter: Counter, n: int = 20):
        return counter.most_common(n)

    lines = []
    lines.append("# BDD100K 数据集特征摘要（自动生成）")
    lines.append("")
    lines.append("## 输入位置")
    lines.append(f"- 外部数据集根目录：`{external_root}`")
    lines.append(f"- labels zip：`{labels_zip}`")
    lines.append(f"- images zip：`{images_zip}`")
    lines.append("")
    lines.append("## Zip 成员统计")
    lines.append(f"- labels JSON 总数：**{len(label_members)}**（按 split：{dict(label_count_by_split)}）")
    lines.append(f"- images JPG 总数：**{len(image_members)}**（按 split：{dict(image_count_by_split)}）")
    lines.append("")
    lines.append("## labels 统计（基于解析的 label JSON）")
    lines.append(f"- 实际解析文件数：**{num_files_processed}**（max_label_files={max_label_files}，0=全部）")
    lines.append(f"- frames 总数：**{num_frames_total}**")
    lines.append(f"- objects 总数：**{num_objects_total}**")
    lines.append(f"- 每图 objects：min={o_min} mean={None if o_mean is None else round(o_mean, 3)} max={o_max}")
    lines.append("")
    lines.append("### 场景/时间/天气分布（Top20）")
    lines.append("- scene：")
    for k, v in _topn(scene_counter, 20):
        lines.append(f"  - {k}: {v}")
    lines.append("- timeofday：")
    for k, v in _topn(timeofday_counter, 20):
        lines.append(f"  - {k}: {v}")
    lines.append("- weather：")
    for k, v in _topn(weather_counter, 20):
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("### 类别分布（Top30）")
    for k, v in _topn(category_counter, 30):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### trafficLightColor 分布（Top20）")
    for k, v in _topn(traffic_light_color_counter, 20):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### bbox 基础统计（px）")
    lines.append(f"- bbox_count：{bbox_stats['count']}")
    lines.append(f"- width(min/mean/max)：{bbox_stats['width_min']} / {bbox_stats['width_mean']} / {bbox_stats['width_max']}")
    lines.append(f"- height(min/mean/max)：{bbox_stats['height_min']} / {bbox_stats['height_mean']} / {bbox_stats['height_max']}")
    lines.append(f"- area(min/mean/max)：{bbox_stats['area_min']} / {bbox_stats['area_mean']} / {bbox_stats['area_max']}")
    lines.append("")
    lines.append("### labels -> images 抽样一致性检查")
    lines.append(f"- 检查次数：{match_check_total}，命中：{match_check_hit}，命中率：{summary['labels_stats']['label_to_image_match_check']['match_rate']}")
    lines.append("")
    lines.append("## images 统计（抽样解码）")
    lines.append(f"- 抽样参数：sample_images={sample_images}")
    lines.append(f"- 成功解码：{sampled_images}，失败：{decode_fail}")
    lines.append("- 分辨率分布：")
    for k, v in _topn(resolutions, 50):
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("## 运行信息")
    lines.append(f"- 耗时（秒）：{round(summary['runtime']['seconds'], 3)}")
    lines.append("")

    with md_path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

    logger.info("BDD100K summary已生成", json=str(json_path), md=str(md_path))


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
        choices=["extract_bdd100k", "generate_synthetic", "statistics", "bdd100k_summary", "all"],
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
    parser.add_argument(
        "--external-dataset-root",
        type=str,
        default=os.getenv("DATASET_PATH", "/Users/shiyifan/Documents/dataset"),
        help="外部数据集根目录（应包含 BDD100K/）",
    )
    parser.add_argument(
        "--max-label-files",
        type=int,
        default=0,
        help="最多解析多少个 label JSON（0=全部，80k可能耗时较长）",
    )
    parser.add_argument(
        "--sample-images",
        type=int,
        default=50,
        help="从 images zip 中抽样解码多少张图用于分辨率统计",
    )
    
    args = parser.parse_args()
    data_root = Path(args.data_root)
    
    if args.task == "extract_bdd100k":
        extract_bdd100k(data_root, args.keep_zip)
    
    elif args.task == "generate_synthetic":
        generate_synthetic_data(data_root, args.num_scenes)
    
    elif args.task == "statistics":
        data_statistics(data_root)

    elif args.task == "bdd100k_summary":
        summarize_bdd100k(
            external_root=Path(args.external_dataset_root),
            output_dir=data_root / "BDD100K",
            max_label_files=args.max_label_files,
            sample_images=args.sample_images,
        )
    
    elif args.task == "all":
        extract_bdd100k(data_root, args.keep_zip)
        generate_synthetic_data(data_root, args.num_scenes)
        data_statistics(data_root)
    
    logger.info("数据准备完成！")
    logger.info("下一步：测试数据加载器", command="python -m src.traffic_rules.data.traffic_dataset")


if __name__ == "__main__":
    main()

