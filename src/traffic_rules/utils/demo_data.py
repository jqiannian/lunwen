"""生成用于离线演示的合成交通样本。"""

from __future__ import annotations

from pathlib import Path

from traffic_rules.data.traffic_dataset import SceneContext, TrafficSample


def build_demo_samples(scenarios: list[str]) -> list[TrafficSample]:
    """根据场景关键字构造合成样本, 便于 CLI/测试复现。"""

    samples: list[TrafficSample] = []
    for idx, scenario in enumerate(scenarios):
        is_red = "red" in scenario
        traffic_light = "red" if is_red else "green"
        base_speed = 5.0 + idx
        distance = 30.0 - idx * 3.0
        extra = {
            "timestamp": float(idx),
            "entities": [
                {
                    "id": f"{scenario}-vehicle",
                    "type": "vehicle",
                    "position": [idx * 5.0 + 1.0, 0.5],
                    "speed": base_speed,
                    "stop_line_distance": distance,
                    "timestamp": float(idx),
                },
                {
                    "id": f"{scenario}-light",
                    "type": "traffic_light",
                    "position": [idx * 5.0 + 1.0, 10.0],
                    "timestamp": float(idx),
                },
                {
                    "id": f"{scenario}-stopline",
                    "type": "stop_line",
                    "position": [idx * 5.0, -1.0],
                    "timestamp": float(idx),
                },
            ],
        }
        context = SceneContext(
            scene_id=f"scene-{idx}",
            traffic_light_state=traffic_light,
            vehicle_speed=base_speed,
            stop_line_distance=distance,
            extra=extra,
        )
        samples.append(TrafficSample(Path(f"synthetic/{context.scene_id}.png"), context))
    return samples
