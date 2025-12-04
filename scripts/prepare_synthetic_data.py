"""生成 MVP 所需的合成交通场景样本。

本脚本是 prepare_data.py 的便捷入口，统一调用 prepare_data.py 中的生成函数。

脚本职责：
1. 读取 configs/mvp.yaml 中的数据配置（可选）
2. 调用 prepare_data.py 中的 generate_synthetic_data() 函数
3. 输出到 data/synthetic 目录

Usage:
    # 使用默认配置（100个场景）
    python scripts/prepare_synthetic_data.py
    
    # 指定场景数量
    python scripts/prepare_synthetic_data.py --num-scenes 200
    
    # 指定数据根目录
    python scripts/prepare_synthetic_data.py --data-root data --num-scenes 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 添加项目路径以导入prepare_data模块
sys.path.insert(0, str(Path(__file__).parent))

# 导入prepare_data中的函数
from prepare_data import generate_synthetic_data

from src.traffic_rules.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 MVP 合成数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成100个场景（默认）
  python scripts/prepare_synthetic_data.py
  
  # 生成200个场景
  python scripts/prepare_synthetic_data.py --num-scenes 200
  
  # 指定数据根目录
  python scripts/prepare_synthetic_data.py --data-root /path/to/data --num-scenes 100
        """,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="数据根目录（默认: data）",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=100,
        help="生成场景数量（默认: 100）",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例（默认: 0.8）",
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    logger.info("开始生成合成数据", num_scenes=args.num_scenes, data_root=str(data_root))
    
    # 调用prepare_data.py中的函数
    generate_synthetic_data(
        data_root=data_root,
        num_scenes=args.num_scenes,
        train_ratio=args.train_ratio,
    )
    
    logger.info("合成数据生成完成", output_dir=str(data_root / "synthetic"))


if __name__ == "__main__":
    main()
