"""红灯停测试 CLI。"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from typing_extensions import Annotated

from traffic_rules.config.loader import load_config
from traffic_rules.explain.attention_viz import AttentionVisualizer
from traffic_rules.graph.builder import GraphBuilder
from traffic_rules.memory.memory_bank import MemoryBank
from traffic_rules.models.gat_attention import GATAttention
from traffic_rules.monitoring.meters import log_event
from traffic_rules.rules.red_light import RedLightRuleConfig, RuleEngine
from traffic_rules.self_training.pseudo_labeler import (
    PseudoLabelOrchestrator,
    PseudoLabelRecord,
)
from traffic_rules.utils.demo_data import build_demo_samples

app = typer.Typer(help="红灯停测试入口", no_args_is_help=True)

DEFAULT_CONFIG = Path("configs/mvp.yaml")
DEFAULT_SCENARIOS = ("red_stop", "red_violation", "green_pass")

ConfigArg = Annotated[Path, typer.Argument(help="配置文件")]
CheckpointOpt = Annotated[
    Path | None,
    typer.Option("--checkpoint", help="记忆库 checkpoint"),
]
ScenariosOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--scenarios",
        help="需要回放的场景列表",
    ),
]
ReportDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--report-dir",
        "-r",
        dir_okay=True,
        file_okay=False,
        help="报告输出目录, 默认读取配置中的 report_dir",
    ),
]


@app.command()
def run(
    config: ConfigArg = DEFAULT_CONFIG,
    checkpoint: CheckpointOpt = None,
    scenarios: ScenariosOpt = None,
    report_dir: ReportDirOpt = None,
) -> None:
    """执行测试流程: 构建图→推理→规则校验→可解释输出。"""

    project_cfg = load_config(config)
    scenario_list = list(scenarios) if scenarios else list(DEFAULT_SCENARIOS)
    samples = build_demo_samples(scenario_list)
    builder = GraphBuilder()
    batch = builder.build_batch(samples)

    hidden_dim = 32
    memory_bank = MemoryBank(size=128, embedding_dim=hidden_dim)
    if checkpoint and checkpoint.exists():
        memory_bank.load(checkpoint)

    model = GATAttention(
        input_dim=builder.feature_dim,
        hidden_dim=hidden_dim,
        heads=4,
        memory_bank=memory_bank,
    )
    scores, attention = model(batch)

    rule_cfg = project_cfg.rules.get("red_light_stop", RedLightRuleConfig())
    engine = RuleEngine(rule_cfg)

    destination = report_dir or Path(project_cfg.paths.report_dir)
    visualizer = AttentionVisualizer(destination)
    orchestrator = PseudoLabelOrchestrator(Path("artifacts/pseudo_labels"))

    records: list[PseudoLabelRecord] = []
    markdown_paths: list[Path] = []
    for idx, scene_id in enumerate(batch.context_ids):
        scene = batch.scenes[scene_id]
        rule_eval = engine.evaluate(scene)
        attention_vector = attention[idx].tolist()
        markdown_paths.append(visualizer.render_heatmaps(attention_vector, scene_id))

        label_path = destination / f"{scene_id}_label.json"
        label_path.write_text(
            json.dumps({"rule_score": rule_eval.score, "reason": rule_eval.reason}),
            encoding="utf-8",
        )
        if scores[idx].item() > 0.5:
            records.append(
                PseudoLabelRecord(
                    scene_id=scene_id,
                    confidence=float(scores[idx].item()),
                    label_path=label_path,
                    attention_snapshot=destination / f"{scene_id}_attention.png",
                )
            )

    manifest_path = None
    if records:
        parquet_path = orchestrator.persist(records)
        orchestrator.extend_dataset(Path("data/pseudo"))
        manifest_path = parquet_path

    typer.secho("=== 测试完成 ===", fg=typer.colors.CYAN)
    typer.echo(f"生成报告: {[p.name for p in markdown_paths]}")
    if manifest_path:
        typer.echo(f"伪标签已保存: {manifest_path}")

    log_event(
        "test_run",
        scene_id=",".join(batch.context_ids),
        report_dir=str(destination),
        pseudo_labels_written=len(records),
    )


if __name__ == "__main__":
    app()
