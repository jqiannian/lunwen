"""根据注意力权重生成可解释性报告的骨架脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def render_placeholder_report(attention_dir: Path, report_dir: Path) -> None:
    """创建占位 markdown，提示未来在此输出热力图链接。"""

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "mvp_attention_report.md"
    report_path.write_text(
        "# 注意力可视化报告\n\n此处将展示注意力热力图、违规证据链等内容。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染注意力热力图占位文件")
    parser.add_argument(
        "--config",
        default=Path("configs/mvp.yaml"),
        type=Path,
        help="配置路径，用于解析 report_dir",
    )
    parser.add_argument(
        "--attention-dir",
        default=Path("artifacts/attention"),
        type=Path,
        help="注意力数据输入目录",
    )
    parser.add_argument(
        "--report-dir",
        default=Path("reports"),
        type=Path,
        help="输出报告目录",
    )
    args = parser.parse_args()

    if args.config.exists():
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        report_dir = Path(config.get("paths", {}).get("report_dir", args.report_dir))
    else:
        report_dir = args.report_dir

    render_placeholder_report(args.attention_dir, report_dir)
    print(f"报告骨架生成于 {report_dir}")


if __name__ == "__main__":
    main()
