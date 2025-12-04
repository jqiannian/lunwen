# 数据加载器使用指南

## ✅ 已完成模块

您的数据加载系统已经完整实现！包括：

1. ✅ **BDD100K数据加载器**（真实数据）
2. ✅ **合成数据生成器**（MVP快速验证）
3. ✅ **数据增强pipeline**（亮度、对比度、翻转、裁剪）
4. ✅ **停止线距离计算**（向量投影算法）
5. ✅ **场景上下文构建**（用于规则引擎）

---

## 🚀 快速开始

### 步骤1：生成合成数据（5分钟）

```bash
# 生成100个合成场景（训练集80，验证集20）
python scripts/prepare_data.py --task generate_synthetic --num-scenes 100

# 查看生成结果
ls -lh data/synthetic/train/
ls -lh data/synthetic/val/
```

**生成的场景类型**：
- `parking`: 红灯停车（车辆停在停止线前）
- `violation`: 红灯闯行（车辆越过停止线）
- `green_pass`: 绿灯通过（正常行驶）

每个场景包含：
- `scene_XXXX.png` - 场景图像
- `scene_XXXX.json` - 场景元数据（实体、速度、距离等）

---

### 步骤2：解压BDD100K数据（可选）

如果您想使用真实数据训练：

```bash
# 解压BDD100K数据集（需要时间）
python scripts/prepare_data.py --task extract_bdd100k

# 验证解压结果
python scripts/prepare_data.py --task statistics
```

**预期输出**：
```
📊 BDD100K数据:
   train images: 70000 张
   train labels: 70000 条
   val images: 10000 张
   val labels: 10000 条
```

---

### 步骤3：测试数据加载器

```bash
# 测试Synthetic数据加载
python -m src.traffic_rules.data.traffic_dataset
```

**预期输出**：
```
============================================================
测试 TrafficLightDataset
============================================================

[测试1] 加载Synthetic数据
[TrafficLightDataset] Loaded 80 samples (mode=synthetic, split=train)
样本数: 80
图像形状: torch.Size([3, 720, 1280])
实体数量: 3
场景ID: scene_0000
交通灯状态: red
车辆速度: 0.00 m/s
停止线距离: 5.00 m

[测试2] 加载BDD100K数据
[TrafficLightDataset] Loaded 70000 samples (mode=bdd100k, split=train)
样本数: 70000
图像形状: torch.Size([3, 720, 1280])
实体数量: 15
实体统计: {'car': 10, 'light': 4, 'stop': 1}

[测试3] 测试数据增强
增强1: 实体数=3, 图像范围=[0.02, 0.98]
增强2: 实体数=3, 图像范围=[0.01, 0.99]
增强3: 实体数=2, 图像范围=[0.03, 0.97]

============================================================
✅ 数据加载器测试完成
============================================================
```

---

## 📖 API使用示例

### 示例1：加载合成数据

```python
from src.traffic_rules.data.traffic_dataset import TrafficLightDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = TrafficLightDataset(
    data_root="data",
    mode="synthetic",
    split="train",
    max_samples=None,  # 加载所有样本
    augmentation=True,  # 启用数据增强
)

# 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
)

# 迭代数据
for batch_idx, batch in enumerate(dataloader):
    images = batch['image']  # [B, 3, H, W]
    entities = batch['entities']  # List of List[Entity]
    contexts = batch['context']  # List of SceneContext
    
    print(f"Batch {batch_idx}:")
    print(f"  Images: {images.shape}")
    print(f"  Num entities: {[len(e) for e in entities]}")
    
    # 提取规则评分所需信息
    for context in contexts:
        print(f"  Scene {context.scene_id}:")
        print(f"    Light: {context.traffic_light_state}")
        print(f"    Speed: {context.vehicle_speed:.2f} m/s")
        print(f"    Distance: {context.stop_line_distance:.2f} m")
```

---

### 示例2：加载BDD100K数据

```python
# BDD100K数据集（真实数据）
dataset_bdd = TrafficLightDataset(
    data_root="data",
    mode="bdd100k",
    split="train",
    max_samples=1000,  # 仅加载1000个样本（调试）
    augmentation=True,
)

# 获取单个样本
sample = dataset_bdd[0]

# 访问实体信息
for entity in sample['entities']:
    print(f"Entity {entity.id}:")
    print(f"  Type: {entity.type}")
    print(f"  Position: {entity.position}")
    if entity.type == 'light':
        print(f"  State: {entity.light_state}")
    elif entity.type == 'car':
        print(f"  Speed: {entity.velocity} m/s")
        print(f"  Distance to stop: {entity.distance_to_stopline} m")
```

---

### 示例3：自定义数据增强

```python
# 自定义增强配置
augmentation_config = {
    "brightness_jitter": 0.3,    # 亮度扰动±0.3
    "contrast_jitter": 0.3,      # 对比度扰动±0.3
    "crop_probability": 0.5,     # 50%概率裁剪
    "horizontal_flip": 0.5,      # 50%概率翻转
}

dataset = TrafficLightDataset(
    data_root="data",
    mode="synthetic",
    split="train",
    augmentation=True,
    augmentation_config=augmentation_config,
)
```

---

## 🔍 数据格式说明

### 1. Entity（实体）

```python
Entity(
    id="car_1",                    # 实体ID
    type="car",                    # 类型: car|light|stop
    position=(640.0, 500.0),       # 中心坐标(x, y)
    bbox=(600.0, 440.0, 680.0, 560.0),  # 边界框(x1,y1,x2,y2)
    velocity=3.5,                  # 速度(m/s)
    heading=0.0,                   # 朝向(度)
    light_state="red",             # 交通灯状态(仅light类型)
    distance_to_stopline=5.2,      # 到停止线距离(m)
)
```

### 2. SceneContext（场景上下文）

```python
SceneContext(
    scene_id="scene_0042",
    timestamp=0.0,
    vehicle_speed=2.5,             # 主车辆速度
    stop_line_distance=3.8,        # 主车辆到停止线距离
    traffic_light_state="red",     # 交通灯状态
    entities=[...],                # 所有实体列表
    image=np.ndarray,              # 图像数据
)
```

### 3. DataLoader返回格式

```python
batch = {
    'image': torch.Tensor,         # [B, 3, H, W]
    'entities': List[List[Entity]], # [B][N]
    'context': List[SceneContext], # [B]
}
```

---

## 📊 数据统计

### Synthetic数据分布

| 场景类型 | 数量 | 占比 | 说明 |
|---------|------|------|------|
| parking | ~33 | 33% | 红灯停车（正常） |
| violation | ~33 | 33% | 红灯闯行（违规） |
| green_pass | ~34 | 34% | 绿灯通过（正常） |

### BDD100K数据规模

| 分割 | 图像数 | 标注数 | 车辆数（平均） | 交通灯数（平均） |
|------|--------|--------|--------------|----------------|
| train | 70,000 | 70,000 | ~8 | ~2 |
| val | 10,000 | 10,000 | ~8 | ~2 |

---

## ⚙️ 配置选项

### 数据集参数

```python
TrafficLightDataset(
    data_root="data",              # 数据根目录
    mode="synthetic",              # synthetic | bdd100k
    split="train",                 # train | val | test
    max_samples=None,              # 最大样本数（None=全部）
    augmentation=True,             # 是否启用数据增强
    augmentation_config={...},     # 数据增强配置
)
```

### 增强参数

```python
{
    "brightness_jitter": 0.2,      # 亮度扰动范围
    "contrast_jitter": 0.2,        # 对比度扰动范围
    "crop_probability": 0.3,       # 随机裁剪概率
    "horizontal_flip": 0.5,        # 水平翻转概率
}
```

---

## 🐛 常见问题

### Q1: Synthetic数据未找到？

**A**: 先运行数据生成脚本：
```bash
python scripts/prepare_data.py --task generate_synthetic --num-scenes 100
```

### Q2: BDD100K标注解析失败？

**A**: 确保数据已正确解压：
```bash
# 检查目录结构
ls -lh data/Obeject\ Detect/BDD100K/images/100k/train/
ls -lh data/Obeject\ Detect/BDD100K/labels/

# 重新解压
python scripts/prepare_data.py --task extract_bdd100k
```

### Q3: 数据增强后实体数量变化？

**A**: 这是正常的。随机裁剪可能导致部分实体移出图像边界，会被自动过滤。

### Q4: 如何可视化加载的数据？

**A**: 使用可视化脚本（待实现）或手动绘制：
```python
import cv2
sample = dataset[0]
image = sample['image'].permute(1, 2, 0).numpy() * 255
image = image.astype('uint8')

# 绘制实体
for entity in sample['entities']:
    if entity.bbox:
        x1, y1, x2, y2 = map(int, entity.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Scene', image)
cv2.waitKey(0)
```

---

## 🔧 高级用法

### 1. 过滤特定场景

```python
# 仅加载红灯违规场景
class ViolationDataset(TrafficLightDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # 跳过非违规场景
        while sample['context'].traffic_light_state != 'red' or \
              sample['context'].stop_line_distance > 5.0:
            idx = (idx + 1) % len(self)
            sample = super().__getitem__(idx)
        return sample
```

### 2. 自定义实体提取

```python
# 仅提取车辆和交通灯，忽略停止线
def filter_entities(entities):
    return [e for e in entities if e.type in ['car', 'light']]

sample = dataset[0]
filtered_entities = filter_entities(sample['entities'])
```

### 3. 计算数据集统计

```python
from collections import Counter

# 统计交通灯状态分布
light_states = []
for i in range(len(dataset)):
    sample = dataset[i]
    light_states.append(sample['context'].traffic_light_state)

print(Counter(light_states))
# 输出: Counter({'red': 45, 'green': 35, 'yellow': 5})
```

---

## 📝 下一步

数据加载器已就绪，可以开始：

1. ✅ **场景图构建**（`src/graph/builder.py`）
2. ✅ **GAT模型实现**（`src/models/gat_attention.py`）
3. ✅ **训练CLI**（`tools/train_red_light.py`）

---

**最后更新**: 2025-12-03  
**作者**: 算法架构师（AI）




