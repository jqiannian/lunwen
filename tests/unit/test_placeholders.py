"""核心模块单元测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from traffic_rules.explain.attention_viz import AttentionVisualizer
from traffic_rules.graph.builder import GraphBuilder
from traffic_rules.loss.constraint import ConstraintLoss
from traffic_rules.rules.red_light import RedLightRuleConfig, RuleEngine
from traffic_rules.self_training.pseudo_labeler import (
    PseudoLabelOrchestrator,
    PseudoLabelRecord,
)
from traffic_rules.utils.demo_data import build_demo_samples


def test_graph_builder_outputs_shapes() -> None:
    samples = build_demo_samples(["red_stop"])
    batch = GraphBuilder().build_batch(samples)
    assert batch.features.ndim == 2
    assert batch.adjacency.shape[0] == batch.adjacency.shape[1]
    assert set(batch.context_ids) == {samples[0].context.scene_id}


def test_rule_engine_and_constraint_loss() -> None:
    samples = build_demo_samples(["red_stop"])
    rule_engine = RuleEngine(RedLightRuleConfig())
    score = rule_engine.evaluate(samples[0].context).score
    assert 0.0 <= score <= 1.0

    loss_fn = ConstraintLoss()
    model_scores = torch.randn(4)
    rule_scores = torch.rand(4)
    attention = torch.rand(4, 4)
    losses = loss_fn(model_scores, rule_scores, attention)
    assert {"main", "rule", "attention", "total"} <= losses.keys()


def test_attention_visualizer_and_pseudo_label_orchestrator(tmp_path: Path) -> None:
    visualizer = AttentionVisualizer(tmp_path)
    markdown = visualizer.render_heatmaps([0.1, 0.9, 0.5], "demo")
    assert markdown.exists()
    assert (tmp_path / "demo_attention.png").exists()

    orchestrator = PseudoLabelOrchestrator(tmp_path / "pseudo")
    label_path = tmp_path / "scene-0.json"
    label_path.write_text("{}", encoding="utf-8")
    records = [
        PseudoLabelRecord(
            scene_id="scene-0",
            confidence=0.95,
            label_path=label_path,
            attention_snapshot=tmp_path / "demo_attention.png",
        )
    ]
    parquet_path = orchestrator.persist(records)
    orchestrator.extend_dataset(tmp_path / "manifest")
    assert parquet_path.exists()
    assert (tmp_path / "manifest/pseudo_labels_manifest.json").exists()
