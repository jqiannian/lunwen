"""生成用于离线演示的合成交通样本。

注意：本模块用于“无需读文件、直接构造 SceneContext”的演示/调试。
它必须与当前代码中的数据结构保持一致（见：src/traffic_rules/data/__init__.py）。
"""

from __future__ import annotations

import numpy as np

from src.traffic_rules.data import Entity, SceneContext


def build_demo_samples(scenarios: list[str]) -> list[SceneContext]:
    """根据场景关键字构造合成样本, 便于 CLI/测试复现。

    约定：
    - scenario 包含 'red' 表示红灯，否则绿灯
    - scenario 包含 'violation' 表示闯行（d_stop < 0 且 v 较大）
    - scenario 包含 'stop' 表示停车（v=0 且 d_stop > 0）
    """

    samples: list[SceneContext] = []
    for idx, scenario in enumerate(scenarios):
        is_red = "red" in scenario
        traffic_light = "red" if is_red else "green"

        # 让每个场景位置稍微不同，避免完全一致
        base_x = float(idx) * 5.0

        # 停止线：一条水平线段
        stop_start = np.array([base_x, 0.0], dtype=np.float32)
        stop_end = np.array([base_x + 10.0, 0.0], dtype=np.float32)

        # 车辆：根据场景决定速度与“到停止线距离”符号
        if "violation" in scenario and is_red:
            velocity = 3.0
            d_stop = -2.0  # 已过线
            car_pos = np.array([base_x + 2.0, -2.0], dtype=np.float32)
        elif "stop" in scenario and is_red:
            velocity = 0.0
            d_stop = 3.0  # 停止线前
            car_pos = np.array([base_x + 2.0, 3.0], dtype=np.float32)
        else:
            # 绿灯通行或其他：不违规（这里用“距离较远”表达）
            velocity = 2.0
            d_stop = 8.0
            car_pos = np.array([base_x + 2.0, 8.0], dtype=np.float32)

        entities = [
            Entity(
                id=idx * 10 + 1,
                type="car",
                pos=car_pos,
                velocity=float(velocity),
                d_stop=float(d_stop),
            ),
            Entity(
                id=idx * 10 + 2,
                type="light",
                pos=np.array([base_x + 5.0, 10.0], dtype=np.float32),
                light_state=traffic_light,
                confidence=0.9,
            ),
            Entity(
                id=idx * 10 + 3,
                type="stop",
                pos=stop_start,
                end_pos=stop_end,
            ),
        ]

        samples.append(
            SceneContext(
                scene_id=f"scene-{idx}",
                timestamp=float(idx),
                entities=entities,
                metadata={"scenario": scenario},
            )
        )
    return samples
