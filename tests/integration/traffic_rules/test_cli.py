# ruff: noqa: E402, I001
"""CLI 集成测试。"""

from __future__ import annotations

from pathlib import Path
import sys

import yaml
from typer.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools import test_red_light, train_red_light


runner = CliRunner()


def test_train_cli_dry_run() -> None:
    result = runner.invoke(
        train_red_light.app,
        ["configs/mvp.yaml", "--dry-run"],
    )
    assert result.exit_code == 0


def test_test_cli_generates_reports(tmp_path: Path) -> None:
    base_cfg = PROJECT_ROOT / "configs/mvp.yaml"
    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
    cfg.setdefault("paths", {})["report_dir"] = str(tmp_path)
    override = tmp_path / "config.override.yaml"
    override.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    result = runner.invoke(
        test_red_light.app,
        [
            str(override),
        ],
    )
    assert result.exit_code == 0
    assert any(tmp_path.iterdir())
