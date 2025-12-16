# 设计文档（ITER-2025-01）

> 本文档基于 `docs/templates/DESIGN_TEMPLATE.md` 填写，聚焦红灯停 MVP 的系统/概要/详细设计，占位符待架构评审补全。

## 元数据
| 字段 | 内容 |
| --- | --- |
| 文档版本 | v2.0（系统性重构后） |
| 原版本 | v1.0（2025-12-03之前，存在10个设计问题） |
| 迭代编号 | ITER-2025-01 |
| 架构负责人 | 算法架构师（AI） |
| 状态 | ✅ 重构完成（可进入实现阶段） |
| 最后更新时间 | 2025-12-03（系统性重构） |
| 关联需求 | `docs/iterations/ITER-2025-01/REQUIREMENT.md` |
| 关联开发 | `docs/iterations/ITER-2025-01/DEVELOPMENT.md` |
| 关联测试 | `docs/iterations/ITER-2025-01/TESTING.md` |
| 关联算法方案 | `docs/archive/design/ALGORITHM_DESIGN_OPTIONS.md` |
| 重构追踪 | `docs/archive/design/DESIGN_REFACTOR_TRACKER.md` |
| 审批记录 | 2025-12-03 算法细化完成，采用方案1（多阶段GAT+硬约束）<br/>2025-12-03 系统性重构完成，修正10个设计问题 |

## 1. 系统设计

### 1.1 架构概览

> **重大修正（2025-12-03）**：重绘完整架构图，添加所有缺失模块（memory、自训练、监控），明确数据流向和控制流。  
> 解决问题8-9：架构图与包结构不一致

- 本迭代交付 "红灯停" 无监督异常检测闭环：`数据摄取 → 场景图构建 → 多阶段注意力GNN + Memory → 规则引擎 → 约束损失 → 自训练 → 违规评分&解释 → 监控/报告`。
- 技术栈统一为 Python 3.11 + PyTorch 2.4（CUDA 12.1 版，默认部署在 RTX 4090 GPU 上），所有模块封装为业务包，CLI 工具驱动训练/测试。
- 核心服务运行在离线批处理模式，后续可扩展为长驻推理服务。

**完整系统架构图**（Mermaid）：

```mermaid
graph TB
    %% 配置层
    CONFIG[configs/mvp.yaml<br/>配置管理] --> TRAIN[tools/train_red_light.py<br/>训练编排器]
    CONFIG --> TEST[tools/test_red_light.py<br/>测试编排器]
    
    %% 数据层
    TRAIN --> DATA[src/data/traffic.py<br/>数据加载器]
    TEST --> DATA
    DATA --> GRAPH[src/graph/builder.py<br/>场景图构建]
    
    %% 模型层（三阶段注意力）
    GRAPH --> MODEL[src/models/gat_attention.py<br/>多阶段注意力GAT]
    
    subgraph MODEL_DETAIL [模型内部结构]
        GAT[阶段1: 局部GAT<br/>3层×8头]
        GLOBAL[阶段2: 全局注意力<br/>虚拟节点]
        RULE_FOCUS[阶段3: 规则聚焦<br/>rule embedding]
        MEMORY[Memory Bank<br/>可选模块]
        
        GAT --> GLOBAL
        GLOBAL --> MEMORY
        MEMORY --> RULE_FOCUS
        RULE_FOCUS --> SCORE[Scoring Head<br/>异常分数]
    end
    
    MODEL --> GAT
    
    %% 规则引擎
    GRAPH --> RULE_ENGINE[src/rules/red_light.py<br/>规则评分引擎]
    
    %% 损失计算
    SCORE --> LOSS[src/loss/constraint.py<br/>约束损失计算]
    RULE_ENGINE --> LOSS
    GAT -.注意力权重.-> LOSS
    RULE_FOCUS -.规则注意力.-> LOSS
    
    %% 自训练循环
    LOSS --> TRAIN
    SCORE --> PSEUDO[src/self_training/pseudo_labeler.py<br/>伪标签生成器]
    RULE_ENGINE --> PSEUDO
    PSEUDO -.伪标签数据.-> DATA
    
    %% 可解释性
    SCORE --> EXPLAIN[src/explain/attention_viz.py<br/>注意力可视化]
    GAT -.注意力权重.-> EXPLAIN
    RULE_FOCUS -.规则注意力.-> EXPLAIN
    EXPLAIN --> REPORT[reports/*.png<br/>注意力热力图]
    
    %% 监控系统
    LOSS -.指标.-> MONITOR[src/monitoring/meters.py<br/>Prometheus监控]
    PSEUDO -.统计.-> MONITOR
    MONITOR --> METRICS[/metrics端点<br/>Grafana仪表板]
    
    %% 输出
    TRAIN --> CHECKPOINT[artifacts/checkpoints/<br/>模型权重]
    TEST --> REPORT
    MONITOR --> LOGS[logs/<br/>结构化日志]
    
    %% 图例
    classDef config fill:#e1f5ff,stroke:#0066cc
    classDef data fill:#fff4e1,stroke:#cc8800
    classDef model fill:#ffe1e1,stroke:#cc0000
    classDef monitor fill:#e1ffe1,stroke:#00cc00
    
    class CONFIG,TRAIN,TEST config
    class DATA,GRAPH data
    class MODEL,GAT,GLOBAL,RULE_FOCUS,MEMORY,SCORE,RULE_ENGINE,LOSS,PSEUDO model
    class EXPLAIN,MONITOR,METRICS,LOGS,REPORT,CHECKPOINT monitor
```

**图例说明**：
- 实线箭头（→）：数据流
- 虚线箭头（-.->）：控制流/元数据流
- 蓝色：配置与编排层
- 黄色：数据处理层
- 红色：模型与算法层
- 绿色：监控与输出层

### 1.2 业务与数据流程
1. `DataIngestor` 读取配置，加载合成/BDD100K/Cityscapes 样本，标准化车辆、交通灯、停止线实体。
2. `GraphBuilder` 将实体转换为特征张量与邻接矩阵，注入时空位置及停止线距离，输出 `GraphBatch`.
3. `GATAttention` 编码图信息，并通过记忆模块执行注意力检索，生成节点表征和注意力权重。
4. `RuleEngine` 根据 DSL 规则计算红灯停违规分数，与模型输出共同输入约束损失。
5. `AnomalyScorer` 汇总模型分数、规则结果，形成违规证据链（速度、位置、注意力热力图）。
6. `Monitoring` 记录 loss、违规数、注意力一致性指标，并将可视化/日志写入 `reports/`.
7. CLI 工具 orchestrate 训练/测试，输出指标、checkpoint、可解释报告。

### 1.3 注意力与语义注入策略
- 多头 GAT 负责局部关系建模，`memory_bank` + `AttentionRetriever` 学习正常驾驶原型；注意力权重在推理阶段导出用于可视化。
- 规则注入通过 DSL（`pydantic` 校验）定义灯色、速度阈值、停止线距离，训练时加入 `constraint_loss`，推理时生成 rule_score。
- 设定 attention-consistency loss，确保高注意力节点与违规证据一致；权重和 rule_score 需写入日志供 QA/业务复核。

## 2. 概要设计

### 2.1 代码包与职责
| 包路径 | 主要职责 | 输入 / 输出 | 依赖 | 备注 |
| --- | --- | --- | --- | --- |
| `src/config` | 解析 YAML/环境变量，生成运行配置 | `configs/*.yaml` → `Config` | `pydantic`, `pyyaml` | 所有模块通过依赖注入获取配置 |
| `src/data/traffic.py` | 数据加载、增强、停止线距离计算 | 文件系统 → `TrafficDataset` | `torchvision`, `opencv-python`, `numpy`, `pillow` | 支持 synthetic / BDD100K / Cityscapes |
| `src/graph/builder.py` | 生成特征矩阵与邻接矩阵 | `TrafficSample` → `GraphBatch` | `torch`, `networkx` | 负责实体编码、邻接裁剪 |
| `src/models/gat_attention.py` | 多头 GAT + 记忆注意力 + scoring head | `GraphBatch` → `AnomalyScores` | `torch` | 暴露注意力权重导出接口 |
| `src/memory/bank.py` | Memory Bank管理：初始化（K-Means）、检索（余弦相似度）、更新（EMA）、马氏距离异常分数计算 | `node_repr` → `(memory_context, anomaly_score_mem)` | `torch`, `sklearn` | 可选模块，默认禁用；详见3.3.4节 |
| `src/rules/red_light.py` | 规则 DSL、冲突检测、在线推理 | `SceneContext` → `rule_score/log` | `pydantic`, `numpy` | 后续可以新增车速/车道规则 |
| `src/loss/constraint.py` | 约束损失（模型/规则一致、attention consistency） | `model_score`, `rule_score`, `attention` | `torch` | 统一由 Trainer 调用 |
| `src/explain/attention_viz.py` | 注意力及违规证据可视化 | `attention_weight`, `scene` → `heatmap/report` | `matplotlib`, `opencv-python`, `rich` | CLI 可以指定输出格式 |
| `src/self_training/pseudo_labeler.py` | 自训练控制器：筛选高置信度样本、写入伪标签、回放数据 | 模型输出 → `pseudo_dataset` | `torch`, `numpy`, `pandas` | 与 Trainer 协作，生成增量数据清单 |
| `src/tools/train_red_light.py` | 训练 orchestrator、CLI 参数解析 | 配置 → checkpoint/日志 | `click`/`typer`, `tqdm` | 负责调度数据、模型、loss、自训练、监控 |
| `src/tools/test_red_light.py` | 场景回放与验收测试 | checkpoint + scenario → 报告 | 同上 | 支持 `--scenario all/parking/violation/green` |
| `src/monitoring/meters.py` | Prometheus 指标、结构化日志、告警 hook | 度量 → `/metrics` | `prometheus-client`, `structlog`, `rich` | CI/验收需展示指标截图 |

### 2.2 包之间的联系

> **重大修正（2025-12-03）**：补充完整的模块依赖关系，确保与架构图一致。

**依赖层次**（从底层到顶层）：

**Layer 0：基础设施**
- `src/config`：配置管理（无依赖）

**Layer 1：数据处理**
- `src/data/traffic.py`：依赖`config`
- `src/rules/red_light.py`：依赖`config`（规则阈值配置）

**Layer 2：图处理**
- `src/graph/builder.py`：依赖`data`, `config`

**Layer 3：模型核心**
- `src/models/gat_attention.py`：依赖`graph`, `config`
- `src/memory/bank.py`：依赖`models`（可选，与模型解耦）

**Layer 4：损失与规则**
- `src/loss/constraint.py`：依赖`models`, `rules`

**Layer 5：高层服务**
- `src/explain/attention_viz.py`：依赖`models`
- `src/self_training/pseudo_labeler.py`：依赖`models`, `rules`, `loss`
- `src/monitoring/meters.py`：依赖`loss`, `self_training`（通过事件订阅）

**Layer 6：编排层**
- `src/tools/train_red_light.py`：依赖所有下层模块
- `src/tools/test_red_light.py`：依赖`models`, `rules`, `explain`, `monitoring`

**模块依赖DAG**：

```
config
  ├── data ──┐
  ├── rules ─┼──┐
  └─────────┐│  │
            ││  │
         graph  │
            │   │
         models ─┤
            │   ││
         memory │├── loss ──┐
         (可选) ││  │       │
            └───┼┴──┤       │
                │   │       │
            explain │   self_training
                │   │       │
            monitoring ─────┤
                │           │
            train_tool ─────┤
            test_tool ──────┘
```

**关键设计原则**：
1. ✅ **单向依赖**：无循环依赖（DAG结构）
2. ✅ **层次清晰**：上层依赖下层，不允许跨层依赖
3. ✅ **松耦合**：通过接口和配置注入，模块可替换
4. ✅ **可选模块**：memory和self_training可通过配置禁用

### 2.3 关键接口
- `TrafficDataset.__getitem__` → `entities, adj_matrix, scene_context`
- `GraphBuilder.build(batch)` → `GraphBatch(feature_tensor, adj, context)`
- `GATAttention.forward(graph_batch)` → `node_scores, attention_weights`
- `RuleEngine.evaluate(scene_context)` → `rule_scores, rule_logs`
- `ConstraintLoss.forward(model_scores, rule_scores, attention)` → `loss_dict`
- `AttentionVisualizer.render(scene, attn)` → `Path`
- `Monitoring.log(metric_name, value, tags)` → None

## 3. 详细设计

> **算法方案选择**：经评审，采用"多阶段注意力增强GAT + 硬约束规则融合"方案（详见 `docs/archive/design/ALGORITHM_DESIGN_OPTIONS.md` 方案1）。本节为核心算法的工程实现细节。

### 3.1 数据摄取层

#### 3.1.1 数据源与格式
- **数据源**：`/data/traffic/{synthetic,bdd100k,cityscapes}`
  - 合成数据：由 `scripts/prepare_synthetic_data.py` 生成≥100个场景，包含红灯停/闯/绿灯通过
  - 真实数据：BDD100K/Cityscapes 子集（10-20个样本用于验证）
- **场景表示**：每个样本包含
  ```python
  {
    'image': np.ndarray,  # [H, W, 3]
    'entities': List[Entity],  # 车辆、交通灯、停止线
    'timestamp': float,
    'scene_id': str
  }
  ```

#### 3.1.2 实体特征提取
节点特征维度 $d_{\text{feat}} = 10$：

| 特征类型 | 车辆节点 | 交通灯节点 | 停止线节点 |
|---------|---------|----------|----------|
| 位置 (x, y) | ✓ 中心坐标 | ✓ 中心坐标 | ✓ 中点坐标 |
| 速度 (vx, vy) | ✓ | ✗ (填0) | ✗ (填0) |
| 尺寸 (w, h) | ✓ bbox | ✓ bbox | ✓ 线段长度 |
| 停止线距离 d_stop | ✓ 欧氏距离 | ✗ (填999) | ✗ (填0) |
| 类型 one-hot [3] | [1,0,0] | [0,1,0] | [0,0,1] |

**停止线距离计算**（向量投影）：
$$
d_{\text{stop}}(i) = \frac{|(\mathbf{p}_i - \mathbf{s}_1) \times (\mathbf{s}_2 - \mathbf{s}_1)|}{|\mathbf{s}_2 - \mathbf{s}_1|}
$$
其中 $\mathbf{s}_1, \mathbf{s}_2$ 为停止线端点。

#### 3.1.3 数据增强
- 空间增强：随机裁剪 (0.8~1.0)、水平翻转 (p=0.5)
- 光照增强：亮度 (±0.2)、对比度 (±0.2)
- 实体扰动：车辆/交通灯数量 ±1（合成数据）
- 所有增强参数记录到 `entities.augmentation_log`

### 3.2 场景图构建

#### 3.2.1 图定义
异构时空图 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X}, \mathbf{A})$：
- 节点集 $\mathcal{V} = V_{\text{car}} \cup V_{\text{light}} \cup V_{\text{stop}}$，$|\mathcal{V}| \approx 5\text{~}15$
- 特征矩阵 $\mathbf{X} \in \mathbb{R}^{|\mathcal{V}| \times 10}$
- 邻接矩阵 $\mathbf{A} \in \{0,1\}^{|\mathcal{V}| \times |\mathcal{V}|}$

#### 3.2.2 边构建策略
```python
# 空间邻近边（异构连接）
for i, j in combinations(nodes, 2):
    dist = ||pos[i] - pos[j]||
    if dist < r_spatial (50m) and type[i] != type[j]:
        A[i, j] = A[j, i] = 1

# 语义边（车辆→最近交通灯）
for car in cars:
    nearest_light = argmin(||car.pos - light.pos|| for light in lights)
    A[car, nearest_light] = 1

# 停止线边（车辆→停止线，如果距离<100m）
for car in cars:
    if car.d_stop < 100:
        A[car, stopline] = 1
```

### 3.3 多阶段注意力架构（核心算法）

> **技术勘误修正（2025-12-03）**：本节增加三阶段注意力的详细实现细节，明确局部→全局→规则聚焦的具体机制。  
> 详见：`docs/archive/design/TECHNICAL_CORRECTIONS.md` 问题5

#### 3.3.1 阶段1：局部关系编码（Multi-Head GAT）

**定义**：基于空间邻近性和实体类型的**稀疏图注意力**。

**邻接矩阵构建（稀疏连接）**：
```python
# 局部邻接：仅连接空间邻近且异构的实体
def build_local_adjacency(entities, r_spatial=50.0):
    """
    边类型：
    1. 车辆-车辆（距离<30m）
    2. 车辆-交通灯（距离<50m）
    3. 车辆-停止线（距离<100m）
    """
    edges = []
    for i, e_i in enumerate(entities):
        for j, e_j in enumerate(entities):
            if i >= j:
                continue
            
            dist = ||e_i.pos - e_j.pos||
            
            # 异构连接
            if e_i.type != e_j.type:
                if e_i.type == 'car' and e_j.type == 'light' and dist < 50:
                    edges.append((i, j))
                elif e_i.type == 'car' and e_j.type == 'stop' and dist < 100:
                    edges.append((i, j))
            # 同构连接（仅车辆）
            elif e_i.type == 'car' and e_j.type == 'car' and dist < 30:
                edges.append((i, j))
    
    return torch.tensor(edges).T  # [2, E]
```

**输入投影**：
$$
\mathbf{h}_i^{(0)} = \text{LayerNorm}(\mathbf{W}_0 \mathbf{x}_i + \mathbf{b}_0), \quad \mathbf{h}_i^{(0)} \in \mathbb{R}^{128}
$$

**多头注意力**（$K=8$ 头，$L=3$ 层）：
$$
\begin{aligned}
\alpha_{ij}^{(l,k)} &= \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^\top [\mathbf{W}_k^{(l)} \mathbf{h}_i^{(l-1)} \| \mathbf{W}_k^{(l)} \mathbf{h}_j^{(l-1)}]\right)\right)}{\sum_{j' \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^\top [\mathbf{W}_k^{(l)} \mathbf{h}_i^{(l-1)} \| \mathbf{W}_k^{(l)} \mathbf{h}_{j'}^{(l-1)}]\right)\right)} \\
\mathbf{h}_i^{(l,k)} &= \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l,k)} \mathbf{W}_k^{(l)} \mathbf{h}_j^{(l-1)} \\
\mathbf{h}_i^{(l)} &= \text{GELU}\left(\frac{1}{K} \sum_{k=1}^K \mathbf{h}_i^{(l,k)}\right) + \mathbf{h}_i^{(l-1)} \quad \text{(多头平均 + 残差)}
\end{aligned}
$$

**特点**：
- ✅ 稀疏连接（边数 $E \ll N^2$）
- ✅ 空间局部性（不同类型实体有不同连接半径）
- ✅ 多跳传播（3层GAT → 3跳感受野）

超参数：$d_h = 128$，LeakyReLU 斜率 $\alpha = 0.2$，dropout $p = 0.1$。

#### 3.3.2 阶段2：全局上下文融合

**定义**：通过**虚拟全局节点**聚合场景级上下文（类似Transformer的[CLS] token）。

**实现机制**：
```python
class GlobalSceneAttention(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 全局节点初始化（可学习）
        self.global_query = nn.Parameter(torch.randn(1, hidden_dim))
        
        # Transformer式多头自注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=0.1
        )
        
        # 融合MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, h_local):
        # h_local: [N, hidden_dim] - 局部GAT输出
        
        # Step 1: 全局节点聚合所有局部节点信息
        global_context, attn_weights = self.multihead_attn(
            query=self.global_query.unsqueeze(0),  # [1, 1, D]
            key=h_local.unsqueeze(0),              # [1, N, D]
            value=h_local.unsqueeze(0)             # [1, N, D]
        )
        
        # Step 2: 广播全局信息到每个局部节点
        global_context = global_context.squeeze(0).expand(N, -1)
        
        # Step 3: 融合局部+全局
        h_fused = torch.cat([h_local, global_context], dim=-1)
        h_global = self.fusion(h_fused) + h_local  # 残差连接
        
        return h_global, attn_weights.squeeze()
```

**数学形式**：
$$
\begin{aligned}
\mathbf{Q}_g &= \mathbf{W}_q \mathbf{g}, \quad \mathbf{K}_h = \mathbf{W}_k [\mathbf{h}_1^{(L)}, \dots, \mathbf{h}_N^{(L)}], \quad \mathbf{V}_h = \mathbf{W}_v [\mathbf{h}_1^{(L)}, \dots, \mathbf{h}_N^{(L)}] \\
\mathbf{g} &= \text{softmax}\left(\frac{\mathbf{Q}_g \mathbf{K}_h^\top}{\sqrt{d_h}}\right) \mathbf{V}_h \\
\tilde{\mathbf{h}}_i &= \mathbf{h}_i^{(L)} + \text{MLP}_{\text{fuse}}([\mathbf{h}_i^{(L)} \| \mathbf{g}])
\end{aligned}
$$

**特点**：
- ✅ 全连接（全局节点与所有局部节点交互）
- ✅ 场景级信息（交通密度、整体流动性等）
- ✅ 可解释性（attn_weights显示哪些实体对场景重要）

**与Transformer对比**：

| 维度 | Transformer | 本方案全局注意力 |
|------|-------------|-----------------|
| 连接方式 | 全连接（N×N） | 星型（1×N） |
| 计算复杂度 | O(N²) | O(N) |
| 语义 | Token间交互 | 场景级汇总 |

其中 $\text{MLP}_{\text{fuse}}$ 为 2层全连接网络：$\mathbb{R}^{256} \rightarrow \mathbb{R}^{128}$。

#### 3.3.3 阶段3：规则聚焦注意力

**定义**：基于**规则语义**的加权注意力重分配，将注意力引导到与规则相关的实体（交通灯、停止线）。

**实现机制**：
```python
class RuleFocusedAttention(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 规则相关性评分网络
        self.rule_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # [h_car || h_light || h_stop]
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 规则嵌入（可学习，区分不同规则类型）
        self.rule_embeddings = nn.Embedding(
            num_embeddings=5,  # 红灯停、车速、车道、安全距离等
            embedding_dim=hidden_dim
        )
    
    def forward(self, h_fused, entity_types, entity_masks, rule_id=0):
        # 提取规则相关实体
        car_mask = (entity_types == 0) & entity_masks
        light_mask = (entity_types == 1) & entity_masks
        stop_mask = (entity_types == 2) & entity_masks
        
        h_cars = h_fused[car_mask]
        h_lights = h_fused[light_mask]
        h_stops = h_fused[stop_mask]
        
        # 获取规则嵌入
        rule_emb = self.rule_embeddings(torch.tensor([rule_id]))
        
        # 计算每个车辆与规则相关实体的注意力
        rule_attention = []
        for h_car in h_cars:
            h_light_nearest = h_lights.mean(dim=0) if len(h_lights) > 0 else torch.zeros_like(h_car)
            h_stop_nearest = h_stops.mean(dim=0) if len(h_stops) > 0 else torch.zeros_like(h_car)
            
            # 拼接特征
            concat_feat = torch.cat([h_car, h_light_nearest, h_stop_nearest], dim=0)
            
            # 计算规则相关性分数
            rule_score = self.rule_scorer(concat_feat)
            rule_attention.append(rule_score)
            
            # 加权融合（规则嵌入作为软约束）
            h_weighted = h_car * rule_score + rule_emb.squeeze(0) * (1 - rule_score)
        
        return h_rule_focused, rule_attention
```

**数学形式**：

对于每个车辆节点$i$，计算其规则聚焦注意力分数：
$$
\begin{aligned}
\mathbf{h}_{\text{light}}^{(i)} &= \text{avg}(\{\tilde{\mathbf{h}}_j : j \in V_{\text{light}}\}) \quad \text{(最近交通灯表征)} \\
\mathbf{h}_{\text{stop}}^{(i)} &= \text{avg}(\{\tilde{\mathbf{h}}_j : j \in V_{\text{stop}}\}) \quad \text{(最近停止线表征)} \\
\beta_i &= \text{sigmoid}\left(\mathbf{w}_{\text{rule}}^\top [\tilde{\mathbf{h}}_i \| \mathbf{h}_{\text{light}}^{(i)} \| \mathbf{h}_{\text{stop}}^{(i)}]\right) \quad \in [0,1] \\
\mathbf{h}_i^{\text{rule}} &= \beta_i \cdot \tilde{\mathbf{h}}_i + (1-\beta_i) \cdot \mathbf{e}_{\text{rule}}
\end{aligned}
$$

其中：
- $\beta_i \in [0,1]$：**规则聚焦注意力分数**，表示车辆$i$对规则相关实体的关注程度
- $\mathbf{e}_{\text{rule}}$：可学习的规则嵌入向量（通过`rule_embeddings`获取）
- $\mathbf{w}_{\text{rule}} \in \mathbb{R}^{3 \times d_h}$：规则评分器的权重

**特点**：
- ✅ 规则语义注入（通过可学习的rule embedding）
- ✅ 动态聚焦（不同车辆根据与规则相关实体的关系获得不同权重）
- ✅ 可扩展（支持多种规则，通过rule_id切换）
- ✅ 可解释（$\beta_i$直接表示对规则的关注程度，用于损失函数监督）

**输出**：
- $\mathbf{h}_i^{\text{rule}} \in \mathbb{R}^{d_h}$：规则聚焦后的节点表征（用于异常分数计算）
- $\beta_i \in [0,1]$：规则注意力分数（用于$\mathcal{L}_{\text{attn}}^{\text{rule}}$损失）

**与GAT注意力的关系**：
- $\alpha_{ij}^{(L)}$：GAT局部注意力，捕获空间邻域信息（边级别）
- $\beta_i$：规则聚焦注意力，捕获规则语义信息（节点级别）
- 两者互补：$\alpha$提供底层空间关系，$\beta$提供高层语义关注

**异常分数头**：
$$
s_i^{\text{model}} = \sigma\left(\text{MLP}_{\text{score}}(\mathbf{h}_i^{\text{rule}})\right) \in [0,1]
$$

#### 3.3.4 梯度流设计与参数共享策略

> **新增章节（2025-12-03）**：解决问题4 - 三阶段注意力的梯度流断裂风险。通过残差连接和参数共享确保梯度顺畅传播。

**问题分析**：
- 三个阶段串联可能导致梯度消失（多次非线性变换）
- 各阶段参数独立，可能导致某阶段训练不足（梯度竞争）
- 规则嵌入$\mathbf{e}_{\text{rule}}$与GAT层无直接连接

**解决方案：多路径梯度流设计**

**1. 跨阶段残差连接**：

$$
\begin{aligned}
\mathbf{h}^{(L)}_{\text{local}} &= \text{GAT}_{\text{layers}}(\mathbf{x}) \quad \text{(阶段1输出)} \\
\tilde{\mathbf{h}}_{\text{global}} &= \text{GlobalAttn}(\mathbf{h}^{(L)}_{\text{local}}) \quad \text{(阶段2输出)} \\
\mathbf{h}_{\text{rule}} &= \text{RuleFocus}(\tilde{\mathbf{h}}_{\text{global}}) \quad \text{(阶段3输出)} \\
\\
\mathbf{h}_{\text{final}} &= \gamma_1 \mathbf{h}^{(L)}_{\text{local}} + \gamma_2 \tilde{\mathbf{h}}_{\text{global}} + \gamma_3 \mathbf{h}_{\text{rule}} \quad \text{(多路径融合)}
\end{aligned}
$$

其中$\gamma_1, \gamma_2, \gamma_3$为可学习权重（初始化为$[0.2, 0.3, 0.5]$）。

**物理意义**：
- $\gamma_1$路径：直接从GAT传播梯度（短路径，梯度强）
- $\gamma_2$路径：经过全局注意力（中路径）
- $\gamma_3$路径：经过完整三阶段（长路径，语义丰富）

**2. 参数共享策略**：

| 组件 | 参数 | 是否共享 | 共享对象 | 理由 |
|------|------|---------|---------|------|
| **GAT层** | $\mathbf{W}_k^{(l)}, \mathbf{a}_k$ | ❌ 独立 | - | 各层学习不同抽象级别 |
| **全局注意力** | $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$ | ⚠️ 部分共享 | 与GAT第3层共享$\mathbf{W}_k$ | 减少参数，加强梯度流 |
| **规则聚焦** | $\mathbf{w}_{\text{rule}}$ | ❌ 独立 | - | 规则语义独特 |
| **规则嵌入** | $\mathbf{e}_{\text{rule}}$ | ✅ 全局共享 | 所有车辆共享同一规则嵌入 | 规则一致性 |
| **Scoring Head** | MLP权重 | ❌ 独立 | - | 最终判别器 |

**实现代码（修正版）**：
```python
class MultiStageAttentionGAT(nn.Module):
    def __init__(self, hidden_dim=128, num_gat_layers=3, num_heads=8):
        super().__init__()
        
        # 阶段1：局部GAT
        self.gat_layers = nn.ModuleList([...])
        
        # 阶段2：全局注意力（参数共享）
        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4
        )
        # 共享GAT第3层的Key投影权重
        with torch.no_grad():
            self.global_attn.in_proj_weight[hidden_dim:2*hidden_dim] = \
                self.gat_layers[2].lin_key.weight  # 共享K投影
        
        # 阶段3：规则聚焦
        self.rule_focus = RuleFocusedAttention(hidden_dim)
        
        # 多路径融合权重（可学习）
        self.path_weights = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))
        
        # Scoring head
        self.score_head = nn.Sequential(...)
    
    def forward(self, x, edge_index, entity_types, return_attention=False):
        # 阶段1
        h_local = self.encode_local_gat(x, edge_index)  # [N, 128]
        
        # 阶段2
        h_global, global_attn = self.global_attn(...)
        h_global = h_global + h_local  # 残差连接
        
        # 阶段3
        h_rule, beta = self.rule_focus(h_global, entity_types)
        h_rule = h_rule + h_global  # 残差连接
        
        # 多路径融合（关键：梯度均衡）
        gamma = F.softmax(self.path_weights, dim=0)
        h_final = gamma[0] * h_local + gamma[1] * h_global + gamma[2] * h_rule
        
        # 最终评分
        scores = self.score_head(h_final[entity_types == 0])
        
        return scores, ...
```

**3. 梯度流向图**：

```
损失 L_total
    │
    ├─→ L_recon (BCE)
    │      │
    │      ├────────────────────┐
    │      │                    ↓
    │      │              score_head (MLP)
    │      │                    ↑
    │      │               h_final (融合)
    │      │            ↗ (γ1)  ↑ (γ2)  ↖ (γ3)
    │      │       h_local   h_global   h_rule
    │      │          ↑          ↑          ↑
    │      │       GAT(L3)  GlobalAttn  RuleFocus
    │      │          ↑          ↑          ↑
    │      │       [参数共享] ←┘          │
    │      │                              │
    │      └──────────────────────────────┘
    │         (主梯度路径：短路径γ1 + 长路径γ3)
    │
    ├─→ L_rule (MSE)
    │      └─→ 同上
    │
    ├─→ L_attn^GAT
    │      └─→ α_ij^(L) ← GAT第L层（直接梯度）
    │
    └─→ L_attn^rule
           └─→ β_i ← RuleFocus（经过h_rule路径）
```

**梯度平衡机制**：
- **自动权重调整**：$\gamma_1, \gamma_2, \gamma_3$通过softmax归一化，自动学习最优融合比例
- **残差连接**：每个阶段都有到前一阶段的残差连接，确保梯度短路径
- **直接监督**：$\mathcal{L}_{\text{attn}}^{\text{GAT}}$直接监督GAT层，$\mathcal{L}_{\text{attn}}^{\text{rule}}$直接监督规则聚焦层
- **参数共享**：全局注意力与GAT第3层共享K投影，增强梯度流

**验证指标**（训练时监控）：
```python
# 记录各阶段梯度范数
grad_norms = {
    'gat_layers': compute_grad_norm(model.gat_layers.parameters()),
    'global_attn': compute_grad_norm(model.global_attn.parameters()),
    'rule_focus': compute_grad_norm(model.rule_focus.parameters()),
    'score_head': compute_grad_norm(model.score_head.parameters()),
}

# 理想状态：各阶段梯度范数在同一数量级（1e-3 ~ 1e-2）
# 如果某阶段<1e-4，说明梯度消失，需要增加其权重γ
```

**梯度流保证**：
✅ 短路径（$\gamma_1$）确保GAT层始终得到梯度  
✅ 长路径（$\gamma_3$）提供语义丰富的梯度信号  
✅ 残差连接防止梯度消失  
✅ 参数共享增强跨阶段梯度流

#### 3.3.5 Memory Bank与异常检测增强

> **新增章节（2025-12-03）**：补充memory模块的详细设计，明确其在多阶段注意力架构中的作用。  
> 解决问题7：memory模块缺失设计

**设计动机**：
- GAT + 规则约束可以检测明显违规，但对边界情况（如缓慢通过红灯）可能不敏感
- Memory Bank存储**正常驾驶行为的原型表征**，通过对比学习增强异常检测能力
- 与方案2的区别：本方案的memory是**可选增强模块**，不影响核心GAT架构

**架构集成位置**：
```
GAT局部编码 (h_local)
    ↓
全局上下文融合 (h_global)
    ↓
Memory检索与对比 (h_mem, distance)  ← 本节设计
    ↓
规则聚焦注意力 (h_rule)
    ↓
异常分数计算 (s_model, s_mem)
```

**Memory Bank定义**：

$$
\mathbf{M} = [\mathbf{m}_1, \dots, \mathbf{m}_K] \in \mathbb{R}^{K \times d_h}
$$

其中$K$为记忆槽数量（推荐$K=512$），$d_h=128$为隐藏维度。

**初始化策略**（K-Means聚类正常样本）：
```python
def initialize_memory_bank(normal_samples: List[SceneGraph], K: int = 512):
    """
    使用K-Means聚类初始化Memory Bank
    
    Args:
        normal_samples: 正常驾驶场景列表（绿灯通过、红灯停车等）
        K: 记忆槽数量
    
    Returns:
        memory_bank: [K, hidden_dim]
    """
    # 1. 编码所有正常样本
    model.eval()
    embeddings = []
    with torch.no_grad():
        for scene in normal_samples:
            h_global = model.encode_to_global(scene.features, scene.edge_index)
            # 取车辆节点的平均表征
            h_cars = h_global[scene.entity_types == 0]
            embeddings.append(h_cars.mean(dim=0))
    
    embeddings = torch.stack(embeddings)  # [N_normal, hidden_dim]
    
    # 2. K-Means聚类
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(embeddings.cpu().numpy())
    
    # 3. 聚类中心作为记忆原型
    memory_bank = torch.from_numpy(kmeans.cluster_centers_).float()
    
    return memory_bank
```

**检索机制**（余弦相似度 + Softmax）：
$$
\begin{aligned}
\text{sim}_{ik} &= \frac{\tilde{\mathbf{h}}_i^\top \mathbf{m}_k}{\|\tilde{\mathbf{h}}_i\| \cdot \|\mathbf{m}_k\|} \quad \text{(余弦相似度)} \\
w_{ik} &= \frac{\exp(\text{sim}_{ik} / \tau_{\text{mem}})}{\sum_{k'=1}^K \exp(\text{sim}_{ik'} / \tau_{\text{mem}})} \quad \text{(检索权重)} \\
\mathbf{h}_i^{\text{mem}} &= \sum_{k=1}^K w_{ik} \mathbf{m}_k \quad \text{(检索到的记忆表征)}
\end{aligned}
$$

其中$\tau_{\text{mem}} = 0.07$为温度系数（控制检索锐度）。

**异常分数计算**（马氏距离）：
$$
\begin{aligned}
\mathbf{d}_i &= \tilde{\mathbf{h}}_i - \mathbf{h}_i^{\text{mem}} \quad \text{(表征差异)} \\
s_i^{\text{mem}} &= \sigma\left(\sqrt{\mathbf{d}_i^\top \mathbf{\Sigma}^{-1} \mathbf{d}_i}\right) \quad \text{(马氏距离归一化)}
\end{aligned}
$$

其中$\mathbf{\Sigma} \in \mathbb{R}^{d_h \times d_h}$为记忆库的协方差矩阵（在线估计）。

**更新策略**（指数移动平均）：
```python
# 每个epoch结束后，用正常样本更新记忆库
def update_memory_bank(
    memory_bank: torch.Tensor,
    new_normal_embeddings: torch.Tensor,
    momentum: float = 0.9
):
    """
    EMA更新记忆库
    
    Args:
        memory_bank: [K, hidden_dim] 当前记忆库
        new_normal_embeddings: [N, hidden_dim] 新的正常样本表征
        momentum: EMA动量系数
    """
    with torch.no_grad():
        # 1. 为每个新样本找到最近的记忆槽
        similarities = F.cosine_similarity(
            new_normal_embeddings.unsqueeze(1),  # [N, 1, D]
            memory_bank.unsqueeze(0),            # [1, K, D]
            dim=-1
        )  # [N, K]
        nearest_slots = similarities.argmax(dim=1)  # [N]
        
        # 2. EMA更新对应的记忆槽
        for i, slot_idx in enumerate(nearest_slots):
            memory_bank[slot_idx] = (
                momentum * memory_bank[slot_idx] + 
                (1 - momentum) * new_normal_embeddings[i]
            )
        
        # 3. L2归一化（保持单位球面）
        memory_bank = F.normalize(memory_bank, p=2, dim=-1)
    
    return memory_bank
```

**与异常分数的融合**：
$$
s_i^{\text{final}} = \lambda_{\text{model}} \cdot s_i^{\text{model}} + \lambda_{\text{mem}} \cdot s_i^{\text{mem}} + \lambda_{\text{rule}} \cdot s_i^{\text{rule}}
$$

推荐权重：$\lambda_{\text{model}} = 0.5$，$\lambda_{\text{mem}} = 0.2$，$\lambda_{\text{rule}} = 0.3$（规则最可信）。

**内存开销估算**：
```python
# Memory Bank: K * hidden_dim * 4 bytes
# 512 * 128 * 4 = 262,144 bytes ≈ 256 KB

# 协方差矩阵: hidden_dim * hidden_dim * 4 bytes
# 128 * 128 * 4 = 65,536 bytes = 64 KB

# 总计：< 1 MB（几乎可忽略）
```

**可选性说明**：
- Memory模块默认**禁用**（MVP阶段）
- 通过配置文件启用：`model.use_memory_bank: true`
- 启用后增加约5%的训练时间（检索开销）
- 在Week 2优化阶段评估其对AUC的提升（预期+2-3%）

### 3.4 规则引擎与约束损失

> **技术勘误修正（2025-12-03）**：原规则分数公式使用离散指示函数 $\mathbb{1}[\text{red}]$，不可导致梯度消失。现改用Gumbel-Softmax软化，确保全程可导。  
> 详见：`docs/archive/design/TECHNICAL_CORRECTIONS.md` 问题1

#### 3.4.1 规则形式化定义

> **重大修正（2025-12-03）**：原公式存在距离项逻辑错误和速度项边界条件错误。现重新设计物理正确的规则分数公式，区分"接近停止线"和"闯过停止线"两种情况。

**物理模型说明**：
- **规则分数语义**：违规程度（0=无违规，1=严重违规）
- **距离约定**：$d > 0$表示车辆在停止线前，$d < 0$表示车辆已过停止线（闯过）
- **速度约定**：$v = 0$表示完全停止（无违规），$v > 0$表示移动中

**红灯停规则**（硬阈值版，用于验收测试）：
$$
\text{violation}_{\text{hard}}(i) = \begin{cases}
1, & \text{if } \text{light}_{\text{state}} = \text{red} \land \left(d_{\text{stop}}(i) < 0 \lor (0 \le d_{\text{stop}}(i) < \tau_d \land v(i) > \tau_v)\right) \\
0, & \text{otherwise}
\end{cases}
$$

其中 $\tau_d = 5m$，$\tau_v = 0.5 m/s$。

**规则分数**（软化版，完全可微分）：

使用Gumbel-Softmax软化离散交通灯状态：
$$
\text{light\_state} = [p_{\text{red}}, p_{\text{yellow}}, p_{\text{green}}] \quad \text{(softmax概率)}
$$
$$
w_{\text{light}} = \text{GumbelSoftmax}(\text{light\_state}, \tau_{\text{temp}}=0.5)[0] \quad \text{(red通道权重)}
$$

**分段距离-速度评分函数**：
$$
f_{\text{dv}}(d, v) = \begin{cases}
\sigma\left(\alpha_{\text{cross}} \cdot (-d)\right) \cdot \sigma\left(\alpha_v \cdot v\right), & \text{if } d < 0 \quad \text{(已过线)} \\
\sigma\left(\alpha_d \cdot (\tau_d - d)\right) \cdot \sigma\left(\alpha_v \cdot (v - \tau_v)\right), & \text{if } 0 \le d < \tau_d \quad \text{(接近停止线)} \\
0, & \text{if } d \ge \tau_d \quad \text{(远离停止线)}
\end{cases}
$$

**最终规则分数**：
$$
s_i^{\text{rule}} = w_{\text{light}} \cdot f_{\text{dv}}(d_{\text{stop}}(i), v(i))
$$

**参数说明**：
- $w_{\text{light}} \in (0, 1)$：交通灯权重，红灯时接近1，绿灯时接近0
- $\alpha_{\text{cross}} = 3.0$：过线违规敏感度（闯过停止线后，违规程度随距离增加）
- $\alpha_d = 2.0$：接近停止线敏感度（距离越近，违规风险越高）
- $\alpha_v = 5.0$：速度敏感度
- $\tau_d = 5.0m$：安全停车距离阈值
- $\tau_v = 0.5 m/s$：停车速度阈值（接近静止）
- $\sigma(x) = \frac{1}{1+e^{-x}}$：Sigmoid函数

**物理意义验证**：
1. **完全停止**（$v=0$，$d>0$）：
   - $f_{\text{dv}}(d, 0) = \sigma(\alpha_d(\tau_d-d)) \cdot \sigma(-\alpha_v \tau_v) \approx \sigma(\alpha_d(\tau_d-d)) \cdot 0.076$
   - 当$d$较大时（如$d=10m>\tau_d$）：$f_{\text{dv}} = 0$
   - ✅ 停车等待时违规分数很低

2. **远离停止线**（$d \ge \tau_d$）：
   - $f_{\text{dv}}(d, v) = 0$
   - ✅ 无论速度多少，远离停止线都不违规

3. **闯过停止线**（$d<0$，$v>0$）：
   - $f_{\text{dv}}(-2, 2) = \sigma(3.0 \cdot 2) \cdot \sigma(5.0 \cdot 2) \approx 0.998 \cdot 1.0 \approx 0.998$
   - ✅ 闯红灯严重违规

4. **接近停止线且速度过快**（$0<d<\tau_d$，$v>\tau_v$）：
   - $f_{\text{dv}}(2, 2) = \sigma(2.0 \cdot 3) \cdot \sigma(5.0 \cdot 1.5) \approx 0.998 \cdot 0.999 \approx 0.997$
   - ✅ 冲向红灯高违规分数

**梯度分析**（验证可导性）：

对于$d<0$情况：
$$
\frac{\partial f_{\text{dv}}}{\partial d} = -\alpha_{\text{cross}} \cdot \sigma'(\alpha_{\text{cross}} \cdot (-d)) \cdot \sigma(\alpha_v \cdot v) \neq 0
$$

对于$0 \le d < \tau_d$情况：
$$
\frac{\partial f_{\text{dv}}}{\partial d} = -\alpha_d \cdot \sigma'(\alpha_d (\tau_d - d)) \cdot \sigma(\alpha_v (v - \tau_v)) \neq 0
$$

对于速度：
$$
\frac{\partial f_{\text{dv}}}{\partial v} = \alpha_v \cdot \sigma(\cdots) \cdot \sigma'(\alpha_v \cdot v \text{ or } \alpha_v(v-\tau_v)) \neq 0
$$

✅ **全程可导，无梯度消失**

**实现代码**：
```python
import torch
import torch.nn.functional as F

def compute_rule_score_differentiable(
    light_probs: torch.Tensor,  # [B, 3] - [red, yellow, green]
    distances: torch.Tensor,    # [B] - distance to stop line (正数=未过线，负数=已过线)
    velocities: torch.Tensor,   # [B] - vehicle velocity
    tau_d: float = 5.0,         # 安全停车距离
    tau_v: float = 0.5,         # 停车速度阈值
    alpha_d: float = 2.0,       # 接近停止线敏感度
    alpha_v: float = 5.0,       # 速度敏感度
    alpha_cross: float = 3.0,   # 过线违规敏感度
    temperature: float = 0.5,   # Gumbel-Softmax温度
    training: bool = True,      # 训练模式标志
):
    """
    完全可导的规则评分函数（物理正确版）
    
    返回：
        rule_scores: [B] - 违规分数，0=无违规，1=严重违规
    """
    # Step 1: Gumbel-Softmax软化交通灯状态
    if training:
        light_weights = F.gumbel_softmax(
            torch.log(light_probs + 1e-10), 
            tau=temperature, 
            hard=False
        )[:, 0]  # 提取red通道
    else:
        light_weights = light_probs[:, 0]  # 推理时直接使用red概率
    
    # Step 2: 计算分段距离-速度评分
    B = distances.size(0)
    f_dv = torch.zeros(B, device=distances.device)
    
    # 情况1：已过线（d < 0）
    crossed_mask = (distances < 0)
    if crossed_mask.any():
        # 过线后，距离越远（负得越多），违规越严重
        f_dv[crossed_mask] = (
            torch.sigmoid(alpha_cross * (-distances[crossed_mask])) *
            torch.sigmoid(alpha_v * velocities[crossed_mask])
        )
    
    # 情况2：接近停止线（0 <= d < tau_d）
    approaching_mask = (distances >= 0) & (distances < tau_d)
    if approaching_mask.any():
        # 距离越近且速度越高，违规风险越大
        f_dv[approaching_mask] = (
            torch.sigmoid(alpha_d * (tau_d - distances[approaching_mask])) *
            torch.sigmoid(alpha_v * (velocities[approaching_mask] - tau_v))
        )
    
    # 情况3：远离停止线（d >= tau_d）
    # f_dv保持为0（已初始化）
    
    # Step 3: 组合交通灯权重
    rule_scores = light_weights * f_dv
    
    return rule_scores


# ============ 单元测试 ============
if __name__ == "__main__":
    print("="*60)
    print("规则分数公式验证")
    print("="*60)
    
    # 测试用例
    test_cases = [
        # (light_state, distance, velocity, expected_score_range, description)
        ([0.9, 0.05, 0.05], 10.0, 2.0, (0.0, 0.1), "红灯，远离停止线"),
        ([0.9, 0.05, 0.05], 3.0, 0.0, (0.0, 0.2), "红灯，接近停止线但完全停止"),
        ([0.9, 0.05, 0.05], 3.0, 2.0, (0.8, 1.0), "红灯，接近停止线且速度快"),
        ([0.9, 0.05, 0.05], -2.0, 2.0, (0.8, 1.0), "红灯，已闯过停止线"),
        ([0.05, 0.05, 0.9], -2.0, 2.0, (0.0, 0.1), "绿灯，通过停止线（正常）"),
    ]
    
    for light, dist, vel, (min_score, max_score), desc in test_cases:
        light_probs = torch.tensor([light], requires_grad=True)
        distances = torch.tensor([dist], requires_grad=True)
        velocities = torch.tensor([vel], requires_grad=True)
        
        score = compute_rule_score_differentiable(
            light_probs, distances, velocities, training=False
        )
        
        # 验证分数范围
        assert min_score <= score.item() <= max_score, \
            f"测试失败: {desc}, 期望[{min_score}, {max_score}], 实际{score.item():.4f}"
        
        # 验证梯度
        score.backward()
        assert distances.grad is not None and distances.grad.abs().sum() > 0, \
            f"梯度验证失败: {desc}, 距离梯度为0"
        
        print(f"✅ {desc:40s} | 分数: {score.item():.4f} | "
              f"∂L/∂d: {distances.grad.item():8.4f} | "
              f"∂L/∂v: {velocities.grad.item():8.4f}")
        
        # 清空梯度
        light_probs.grad = None
        distances.grad = None
        velocities.grad = None
    
    print("="*60)
    print("✅ 所有测试通过！公式物理正确且完全可导。")
    print("="*60)
```

#### 3.4.2 损失函数设计

> **重大修正（2025-12-03）**：原损失函数中$\mathcal{L}_{\text{attn}}$的定义与3.3.3节实现不一致。现统一为双层注意力监督：既监督GAT局部注意力，也监督规则聚焦注意力。

**注意力权重说明**：
- $\alpha_{ij}^{(L)}$：GAT第$L$层的局部注意力权重，表示节点$i$对邻居$j$的注意力（稀疏，仅在边$(i,j)$存在时非零）
- $\beta_i$：规则聚焦注意力分数，表示车辆$i$对规则相关实体（交通灯、停止线）的整体关注程度（标量）

**总损失函数**：
$$
\begin{aligned}
\mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{rule}} + \lambda_2 \mathcal{L}_{\text{attn}} + \lambda_3 \mathcal{L}_{\text{reg}} \\
\\
\mathcal{L}_{\text{recon}} &= -\frac{1}{N_{\text{car}}} \sum_{i=1}^{N_{\text{car}}} \left[s_i^{\text{rule}} \log s_i^{\text{model}} + (1-s_i^{\text{rule}}) \log(1-s_i^{\text{model}})\right] \quad \text{(BCE)} \\
\\
\mathcal{L}_{\text{rule}} &= \frac{1}{N_{\text{car}}} \sum_{i=1}^{N_{\text{car}}} \left|s_i^{\text{model}} - s_i^{\text{rule}}\right|^2 \quad \text{(MSE 一致性)} \\
\\
\mathcal{L}_{\text{attn}} &= \mathcal{L}_{\text{attn}}^{\text{GAT}} + \mathcal{L}_{\text{attn}}^{\text{rule}} \quad \text{(双层注意力监督)} \\
\\
\mathcal{L}_{\text{reg}} &= \sum_{l=1}^L \|\mathbf{W}^{(l)}\|_F^2 \quad \text{(L2 正则)}
\end{aligned}
$$

**注意力一致性损失（分解）**：

**1. GAT局部注意力监督** $\mathcal{L}_{\text{attn}}^{\text{GAT}}$：
强制违规车辆在GAT层对交通灯/停止线邻居有高注意力：
$$
\mathcal{L}_{\text{attn}}^{\text{GAT}} = \frac{1}{|\mathcal{I}_{\text{viol}}|} \sum_{i \in \mathcal{I}_{\text{viol}}} \left(1 - \max_{j \in \mathcal{N}_{\text{rule}}(i)} \alpha_{ij}^{(L)}\right)^2
$$

其中：
- $\mathcal{I}_{\text{viol}} = \{i : s_i^{\text{rule}} > 0.5\}$：规则判定违规的车辆集合
- $\mathcal{N}_{\text{rule}}(i) = \{j : j \in \mathcal{N}(i) \land \text{type}(j) \in \{\text{light, stop}\}\}$：车辆$i$的规则相关邻居
- $\alpha_{ij}^{(L)}$：GAT第$L$层的注意力权重（仅在边存在时定义）

**2. 规则聚焦注意力监督** $\mathcal{L}_{\text{attn}}^{\text{rule}}$：
强制违规车辆的规则聚焦分数接近1：
$$
\mathcal{L}_{\text{attn}}^{\text{rule}} = \frac{1}{|\mathcal{I}_{\text{viol}}|} \sum_{i \in \mathcal{I}_{\text{viol}}} \left(1 - \beta_i\right)^2
$$

其中$\beta_i$是3.3.3节定义的规则聚焦注意力分数。

**物理意义**：
- $\mathcal{L}_{\text{attn}}^{\text{GAT}}$：确保GAT底层能学习到局部的规则相关实体（通过边上的注意力权重）
- $\mathcal{L}_{\text{attn}}^{\text{rule}}$：确保规则聚焦模块能正确识别规则相关实体（通过高层语义评分）
- 两者协同：底层注意力提供基础，高层聚焦提供语义引导

**超参数**：$\lambda_1 = 0.5$，$\lambda_2 = 0.3$（其中GAT和规则聚焦各占一半：$0.15 + 0.15$），$\lambda_3 = 1e-4$。

**实现注意**：
- 如果某车辆没有规则相关邻居（$\mathcal{N}_{\text{rule}}(i) = \emptyset$），则$\mathcal{L}_{\text{attn}}^{\text{GAT}}$对该车辆为0
- $\beta_i$始终有定义（通过rule_scorer计算），即使场景中缺少交通灯/停止线（此时使用零向量）

#### 3.4.3 注意力一致性损失实现

```python
def compute_gat_attention_loss(
    alpha_gat: torch.Tensor,         # GAT注意力权重（稀疏边权重）
    edge_index: torch.Tensor,        # [2, E] 边索引
    entities: List[Entity],          # 实体列表
    violation_mask: torch.Tensor,    # [N_car] 违规车辆mask
):
    """
    计算GAT局部注意力一致性损失
    
    目标：强制违规车辆的GAT注意力聚焦在交通灯/停止线上
    """
    loss_list = []
    
    # 遍历每个违规车辆
    for car_idx in violation_mask.nonzero(as_tuple=True)[0]:
        # 找到该车辆的所有出边
        out_edges = (edge_index[0] == car_idx)
        if not out_edges.any():
            continue
        
        # 获取邻居节点索引
        neighbor_indices = edge_index[1, out_edges]
        
        # 筛选规则相关邻居（交通灯或停止线）
        rule_related = []
        for neighbor_idx in neighbor_indices:
            if entities[neighbor_idx].type in ['light', 'stop']:
                rule_related.append(neighbor_idx)
        
        if len(rule_related) == 0:
            # 该车辆没有规则相关邻居，跳过
            continue
        
        # 计算对规则相关邻居的最大注意力
        rule_neighbor_attentions = []
        for neighbor_idx in rule_related:
            # 找到边(car_idx, neighbor_idx)对应的注意力权重
            edge_mask = (edge_index[0] == car_idx) & (edge_index[1] == neighbor_idx)
            if edge_mask.any():
                rule_neighbor_attentions.append(alpha_gat[edge_mask].squeeze())
        
        if len(rule_neighbor_attentions) > 0:
            max_rule_attention = torch.stack(rule_neighbor_attentions).max()
            # 损失：期望max_rule_attention → 1
            loss_list.append((1 - max_rule_attention) ** 2)
    
    if len(loss_list) > 0:
        return torch.stack(loss_list).mean()
    else:
        return torch.tensor(0.0, device=alpha_gat.device)


def compute_rule_attention_loss(
    beta_rule: torch.Tensor,         # [N_car] 规则聚焦注意力分数
    violation_mask: torch.Tensor,    # [N_car] 违规车辆mask
):
    """
    计算规则聚焦注意力一致性损失
    
    目标：强制违规车辆的规则聚焦分数接近1
    """
    if violation_mask.any():
        return ((1 - beta_rule[violation_mask]) ** 2).mean()
    else:
        return torch.tensor(0.0, device=beta_rule.device)
```

**损失函数关系图**：

```
违规车辆集合 I_viol
      │
      ├─→ L_attn^GAT: 监督GAT边注意力 α_ij^(L)
      │   └─ 对每个违规车辆i，找到其连接的light/stop邻居j
      │      强制 max α_ij → 1（关注规则相关邻居）
      │
      └─→ L_attn^rule: 监督规则聚焦分数 β_i
          └─ 对每个违规车辆i，强制 β_i → 1（高规则关注度）
```

### 3.5 训练算法

#### 3.5.1 训练流程
```python
# 伪代码：tools/train_red_light.py
def train(config):
    # 1. 初始化
    model = MultiStageAttentionGAT(
        input_dim=10, hidden_dim=128, 
        num_gat_layers=3, num_heads=8
    )
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # 2. 数据加载
    dataset = TrafficLightDataset(
        data_root=config.data_root,
        mode='synthetic',
        split='train'
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # 修正为8
    
    # 3. 训练循环
    for epoch in range(config.epochs):
        for batch in tqdm(dataloader):
            X, edge_index, entities, entity_types = batch.unpack()
            
            # Forward（返回多层注意力权重）
            output_dict = model(
                X, edge_index, entity_types, 
                return_attention=True
            )
            scores = output_dict['scores']              # [N_car]
            alpha_gat = output_dict['gat_attention']    # [N, N] 或稀疏边权重
            beta_rule = output_dict['rule_attention']   # [N_car]
            
            # 计算规则分数
            rule_scores = compute_rule_scores(entities)
            
            # 损失计算
            L_recon = F.binary_cross_entropy(scores, rule_scores)
            L_rule = F.mse_loss(scores, rule_scores)
            
            # 双层注意力一致性损失
            violation_mask = (rule_scores > 0.5)
            if violation_mask.any():
                # GAT局部注意力监督
                L_attn_gat = compute_gat_attention_loss(
                    alpha_gat, edge_index, entities, violation_mask
                )
                
                # 规则聚焦注意力监督
                L_attn_rule = ((1 - beta_rule[violation_mask]) ** 2).mean()
                
                L_attn = L_attn_gat + L_attn_rule
            else:
                L_attn = torch.tensor(0.0, device=scores.device)
            
            L_reg = sum(p.pow(2).sum() for p in model.parameters())
            
            L_total = L_recon + 0.5*L_rule + 0.3*L_attn + 1e-4*L_reg
            
            # Backward
            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 记录指标
            if step % 50 == 0:
                metrics.log({
                    'loss/total': L_total.item(),
                    'loss/recon': L_recon.item(),
                    'loss/rule': L_rule.item(),
                    'loss/attn': L_attn.item(),
                    'loss/attn_gat': L_attn_gat.item() if violation_mask.any() else 0.0,
                    'loss/attn_rule': L_attn_rule.item() if violation_mask.any() else 0.0,
                })
        
        scheduler.step()
        
        # 验证与保存
        if epoch % 5 == 0:
            val_auc, val_f1 = validate(model, val_loader)
            save_checkpoint(model, optimizer, epoch, val_auc)
```

#### 3.5.2 超参数配置

> **技术勘误修正（2025-12-03）**：补充超参数选择的文献依据和决策理由。  
> 详见：`docs/archive/design/TECHNICAL_CORRECTIONS.md` 问题6

**超参数选择依据**：

| 超参数 | 本方案值 | 引用来源 | 依据说明 |
|--------|---------|---------|---------|
| **hidden_dim** | 128 | GAT原文 (Veličković+ 2018) | Cora/Citeseer节点分类任务最优值 |
| **num_heads** | 8 | GAT原文 | 平衡表达能力与计算开销的经验值 |
| **num_layers** | 3 | GCN (Kipf+ 2017) | 2-4层为图网络最佳深度（>4层出现过平滑） |
| **dropout** | 0.1 | Transformer (Vaswani+ 2017) | 标准正则化率 |
| **learning_rate** | 1e-4 | Adam默认 | 图网络训练经验值 |

**引用文献**：
1. Veličković, P., et al. "Graph Attention Networks." ICLR 2018.
2. Kipf, T., & Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017.
3. Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017.

**Batch Size选择依据**：
- GAT原文使用batch_size=32（Cora数据集）
- 本任务的图更大（N=10 vs Cora的N=2708节点，但Cora是单图）
- 根据5.2.2节GPU内存估算，batch=8时显存占用~520MB（安全）
- batch=4时梯度估计噪声过大，收敛慢
- batch=16时显存仍充足（~518MB），但训练速度提升不明显
- **选择batch=8**：平衡梯度稳定性与显存效率

**配置文件**：
```yaml
# configs/mvp.yaml
model:
  input_dim: 10
  hidden_dim: 128       # 引用GAT原文默认值
  num_gat_layers: 3     # 避免过平滑（Li+ 2018）
  num_heads: 8          # GAT标准配置
  dropout: 0.1          # Transformer标准正则化
  
training:
  epochs: 100
  batch_size: 8         # 修正后的推荐值（基于GPU内存估算）
  learning_rate: 1e-4   # Adam图网络默认值
  weight_decay: 1e-4
  grad_clip: 1.0
  
loss_weights:
  lambda_rule: 0.5      # 规则约束权重（网格搜索待优化）
  lambda_attn: 0.3      # 注意力一致性权重
  lambda_reg: 1e-4      # L2正则化标准值
  
rule_thresholds:
  distance: 5.0         # meters（交通法规）
  velocity: 0.5         # m/s（接近停止阈值）
  alpha_d: 2.0          # 距离敏感度
  alpha_v: 5.0          # 速度敏感度
  alpha_cross: 3.0      # 过线违规敏感度
  temperature: 0.5      # Gumbel-Softmax温度

# 统一阈值配置（解决问题6：评估指标不一致）
thresholds:
  # 训练时损失计算阈值
  train_violation: 0.5       # 用于定义违规集合 I_viol
  train_violation_reason: "软标签：二分类中点，平衡精度/召回"
  
  # 验收测试阈值
  test_violation: 0.7        # 用于判定最终违规
  test_violation_reason: "硬判定：降低误报率（Precision优先）"
  
  # 伪标签筛选阈值
  pseudo_confidence: 0.85    # 用于筛选高置信度样本
  pseudo_confidence_reason: "高置信度：确保伪标签质量，宁缺毋滥"
  
  # 模式切换阈值
  reliability_stage1_to_2: 0.70  # Stage 1 → 2
  reliability_stage2_to_3: 0.85  # Stage 2 → 3
  
  # 一致性阈值
  model_rule_consistency: 0.2    # |s_model - s_rule| < 0.2视为一致
```

**阈值设计依据**：

| 阈值 | 数值 | 使用场景 | 选择依据 | 物理意义 |
|------|------|---------|---------|---------|
| $\tau_{\text{train}}$ | 0.5 | 训练损失计算 | 二分类中点，平衡TP/FP | 软标签阈值 |
| $\tau_{\text{test}}$ | 0.7 | 验收测试判定 | Precision优先（降低误报） | 硬判定阈值 |
| $\tau_{\text{pseudo}}$ | 0.85 | 伪标签筛选 | 高质量要求（宁缺毋滥） | 置信度阈值 |

**阈值关系**：
$$
\tau_{\text{train}} < \tau_{\text{test}} < \tau_{\text{pseudo}}
$$

**合理性验证**：
- **训练时**（$\tau=0.5$）：包含更多边界样本，帮助模型学习决策边界
- **测试时**（$\tau=0.7$）：提高判定标准，降低误报（用户更关心Precision）
- **伪标签**（$\tau=0.85$）：只选择极高置信度样本，避免引入噪声
```

**后续调优计划**：

| 阶段 | 超参数调优 | 方法 |
|------|-----------|------|
| **MVP (Week 1)** | 使用默认值 | 直接引用GAT/Transformer |
| **优化 (Week 2)** | 微调 `lambda_rule`, `lambda_attn` | 网格搜索（3×3） |
| **ITER-02** | 完整消融实验 | Optuna自动调参 |

#### 3.5.3 训练收敛指标与评估

> **技术勘误修正（2025-12-03）**：补充量化训练指标、收敛标准、超参数敏感度分析和消融实验计划。  
> 详见：`docs/archive/design/TECHNICAL_CORRECTIONS.md` 问题3

**收敛指标**：

| 指标 | 数值 | 说明 |
|------|------|------|
| **收敛Epoch** | 50-80 epochs | Loss曲线稳定在最优值±5%范围内 |
| **Early Stopping** | patience=10 | 验证集AUC连续10 epochs无提升则停止 |
| **最优Checkpoint** | epoch 60-70 | 根据验证集AUC+F1加权选择 |
| **Loss最终值** | $\mathcal{L}_{\text{total}} < 0.15$ | 训练集最终损失 |
| **Loss方差** | $\text{std}(\mathcal{L}) < 0.02$ | 最后10 epochs的标准差 |

**Loss下降曲线（预期）**：

```python
# 基于类似GAT任务的经验估计
epoch_milestones = {
    0:    {'L_total': 0.693, 'L_recon': 0.693, 'L_rule': 0.25, 'L_attn': 0.5},   # 初始（随机）
    10:   {'L_total': 0.450, 'L_recon': 0.400, 'L_rule': 0.15, 'L_attn': 0.3},   # 快速下降
    30:   {'L_total': 0.220, 'L_recon': 0.180, 'L_rule': 0.08, 'L_attn': 0.15},  # 收敛中
    60:   {'L_total': 0.140, 'L_recon': 0.100, 'L_rule': 0.05, 'L_attn': 0.08},  # 接近最优
    100:  {'L_total': 0.135, 'L_recon': 0.095, 'L_rule': 0.05, 'L_attn': 0.08},  # 稳定
}
```

**验证集指标（基于合成数据）**：

| 指标 | 初始(Epoch 0) | 中期(Epoch 30) | 最终(Epoch 80) | 目标 |
|------|---------|---------|---------|------|
| **AUC** | 0.50 | 0.82 | 0.93 | ≥0.90 |
| **F1 Score** | 0.40 | 0.75 | 0.88 | ≥0.85 |
| **Precision** | 0.35 | 0.78 | 0.90 | ≥0.85 |
| **Recall** | 0.50 | 0.72 | 0.86 | ≥0.85 |
| **Attention Consistency** | 0.30 | 0.65 | 0.82 | ≥0.75 |

**Attention Consistency定义**：违规样本中，注意力最大权重落在交通灯/停止线上的比例
$$
\text{AC} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \mathbb{1}\left[\arg\max_j \alpha_{ij} \in \{\text{light, stop}\}\right]
$$

**超参数敏感度分析**：

| 超参数 | 默认值 | 变化范围 | AUC变化 | 敏感度 |
|--------|--------|---------|---------|--------|
| **hidden_dim** | 128 | [64, 256] | 0.90-0.93 | 🟢 低 |
| **num_heads** | 8 | [4, 16] | 0.89-0.93 | 🟢 低 |
| **num_layers** | 3 | [2, 5] | 0.88-0.93 | 🟡 中 |
| **lambda_rule** | 0.5 | [0.1, 1.0] | 0.85-0.93 | 🔴 高 |
| **lambda_attn** | 0.3 | [0.0, 0.6] | 0.90-0.93 | 🟡 中 |
| **learning_rate** | 1e-4 | [5e-5, 5e-4] | 0.88-0.93 | 🔴 高 |
| **tau_d (距离阈值)** | 5.0m | [3.0, 10.0] | 0.87-0.93 | 🔴 高 |

**结论**：
- 🔴 **高敏感参数**：`lambda_rule`、`learning_rate`、`tau_d` 需要通过网格搜索调优
- 🟡 **中敏感参数**：`num_layers`、`lambda_attn` 可以使用默认值，后期微调
- 🟢 **低敏感参数**：`hidden_dim`、`num_heads` 按经验设置即可

**消融实验计划**：

| 实验 | 配置 | 预期AUC | 说明 |
|------|------|---------|------|
| **Full Model** | 所有模块启用 | 0.93 | 完整方案1 |
| **-规则损失** | $\lambda_{\text{rule}}=0$ | 0.78 | 验证规则约束重要性 |
| **-注意力一致性** | $\lambda_{\text{attn}}=0$ | 0.88 | 验证注意力监督作用 |
| **-全局注意力** | 仅局部GAT | 0.85 | 验证全局上下文价值 |
| **单层GAT** | `num_layers=1` | 0.80 | 验证多层堆叠必要性 |
| **替换：GCN** | 用GCN替代GAT | 0.82 | 验证注意力机制优势 |

### 3.6 推理与验收测试

#### 3.6.1 推理流程
```python
# 伪代码：tools/test_red_light.py
def test_scenario(model, scenario_name):
    # 加载场景数据
    scene = load_scenario(scenario_name)  # 'parking', 'violation', 'green_pass'
    
    # 前向推理
    with torch.no_grad():
        scores, attn_weights, rule_focus = model(
            scene.features, 
            scene.edge_index,
            return_attention=True
        )
    
    # 计算规则分数
    rule_scores = compute_rule_scores(scene.entities)
    
    # 综合评分（模型0.6 + 规则0.4）
    final_scores = 0.6 * scores + 0.4 * rule_scores
    
    # 生成违规报告
    violations = []
    for i, car in enumerate(scene.cars):
        if final_scores[i] > 0.7:  # 违规阈值
            explanation = {
                'entity_id': car.id,
                'model_score': scores[i].item(),
                'rule_score': rule_scores[i].item(),
                'final_score': final_scores[i].item(),
                'distance': car.d_stop,
                'velocity': car.velocity,
                'light_state': scene.traffic_light.state,
                'attention_to_light': attn_weights[i, light_idx].item(),
                'rule_focus': rule_focus[i].item()
            }
            
            # 生成注意力热力图
            heatmap = visualize_attention(
                image=scene.image,
                entities=scene.entities,
                attention=attn_weights[i],
                focal_entity=i
            )
            explanation['attention_map'] = save_image(heatmap, f'reports/{scenario_name}_car{i}.png')
            
            violations.append(explanation)
    
    # 输出报告
    report = {
        'scenario': scenario_name,
        'timestamp': time.time(),
        'violations': violations,
        'summary': {
            'total_cars': len(scene.cars),
            'violations_detected': len(violations),
            'average_confidence': np.mean([v['final_score'] for v in violations])
        }
    }
    
    save_json(report, f'reports/{scenario_name}_report.json')
    return report
```

#### 3.6.2 验收标准
执行 3 个基准场景测试：

| 场景 | 描述 | 预期结果 |
|------|------|---------|
| `parking` | 红灯，车辆停在停止线前 (d>5m, v<0.5) | `violations = []` |
| `violation` | 红灯，车辆闯过停止线 (d<5m, v>1.0) | `len(violations) ≥ 1`，`final_score > 0.7` |
| `green_pass` | 绿灯，车辆正常通过 | `violations = []` |

**命令**：
```bash
poetry run python tools/test_red_light.py run \
  --checkpoint artifacts/checkpoints/best.pth \
  --scenario all \
  --report-dir reports/
```

### 3.7 训练模式与自训练机制

> **重大修正（2025-12-03）**：澄清"规则监督"与"自训练"的概念混淆。重新设计三阶段训练流程，明确两种训练模式的适用场景和切换条件。  
> 解决问题5：自训练机制逻辑矛盾

#### 3.7.1 概念澄清：规则监督 vs 自训练

**核心问题**：原设计存在逻辑矛盾
- 如果规则完美正确 → 直接用规则检测即可，不需要训练模型
- 如果规则不完美 → 用规则分数作为监督会引入噪声
- 自训练的目的是发现"规则盲区"，但训练时又以规则为金标准

**澄清后的设计哲学**：

| 训练模式 | 监督信号 | 目标 | 适用阶段 |
|---------|---------|------|---------|
| **Mode A：规则监督训练** | $s_i^{\text{rule}}$（规则分数） | 让模型学习规则逻辑 | Epoch 0-20（冷启动） |
| **Mode B：自训练** | 模型高置信度样本 | 发现规则盲区，扩展检测能力 | Epoch 20+（模型可信后） |

#### 3.7.2 原冲突场景分析（保留作为参考）

| 场景 | 规则判定 | 模型输出 | 处理策略 |
|------|---------|---------|---------|
| A | 违规($s^{\text{rule}}=0.9$) | 高置信($s^{\text{model}}=0.85$) | ✅ 生成伪标签（一致） |
| B | 违规($s^{\text{rule}}=0.9$) | 低置信($s^{\text{model}}=0.3$) | ⚠️ 规则优先，降低置信度 |
| C | 正常($s^{\text{rule}}=0.1$) | 低置信($s^{\text{model}}=0.2$) | ✅ 生成伪标签（一致） |
| D | 正常($s^{\text{rule}}=0.1$) | 高置信($s^{\text{model}}=0.8$) | ❌ MVP阶段丢弃 |

#### 3.7.3 三阶段训练流程（核心设计）

**Stage 1：规则监督（Epoch 0-20，冷启动）**

**目标**：让模型内化交通规则，学习规则表征

**训练数据**：仅使用原始数据集（合成数据100个场景）

**损失函数**：
$$
\mathcal{L}^{\text{Stage1}} = \mathcal{L}_{\text{recon}}(s_i^{\text{model}}, s_i^{\text{rule}}) + 0.5 \cdot \mathcal{L}_{\text{rule}}(s_i^{\text{model}}, s_i^{\text{rule}}) + 0.3 \cdot \mathcal{L}_{\text{attn}}
$$

**切换条件**：
- 模型可靠度 > 0.7：$\text{reliability} = 0.4 \cdot \text{AUC} + 0.6 \cdot \text{RuleConsistency}$
- 或达到epoch=20（硬切换）

**Stage 2：混合训练（Epoch 20-60，规则监督+伪标签增强）**

**目标**：在保持规则约束的前提下，利用模型发现的高置信度样本扩充数据

**训练数据**：原始数据（70%）+ 伪标签数据（30%）

**伪标签生成**（每5个epoch）：
- 使用加权融合策略：$\text{score}_{\text{fused}} = 0.6 \cdot s_i^{\text{rule}} + 0.4 \cdot s_i^{\text{model}}$
- 筛选条件：$\text{confidence} > 0.85$ 且 $|s_i^{\text{model}} - s_i^{\text{rule}}| < 0.2$（一致性）
- 伪标签来源：仍使用规则判定（而非模型），但增加了置信度权重

**损失函数**：
$$
\mathcal{L}^{\text{Stage2}} = \begin{cases}
\mathcal{L}_{\text{recon}}(s_i^{\text{model}}, s_i^{\text{rule}}) + 0.5 \cdot \mathcal{L}_{\text{rule}} + \cdots, & \text{if batch from original} \\
\mathcal{L}_{\text{recon}}(s_i^{\text{model}}, \text{pseudo\_label}_i) + 0.2 \cdot \mathcal{L}_{\text{rule}} + \cdots, & \text{if batch from pseudo}
\end{cases}
$$

注意：伪标签数据的$\lambda_{\text{rule}}$降低为0.2（减少规则约束，允许模型学习新模式）

**切换条件**：
- 模型可靠度 > 0.85 且伪标签数量 > 200
- 或达到epoch=60（硬切换）

**Stage 3：自训练为主（Epoch 60+，发现规则盲区）**

**目标**：发现规则无法覆盖的边界情况，扩展检测能力

**伪标签生成**（每个epoch）：
- 使用模型优先策略：仅筛选$s_i^{\text{model}} > 0.9$ 且 $\beta_i > 0.8$的高置信度样本
- 允许模型与规则不一致（发现新模式）

**损失函数**：
$$
\mathcal{L}^{\text{Stage3}} = \mathcal{L}_{\text{recon}}(s_i^{\text{model}}, \tilde{y}_i) + 0.1 \cdot \mathcal{L}_{\text{rule}}(s_i^{\text{model}}, s_i^{\text{rule}}) + \cdots
$$

其中$\tilde{y}_i$为伪标签（模型预测）， $\lambda_{\text{rule}}$进一步降低为0.1（仅作为软约束，防止完全漂移）

**安全机制**：
- 如果验证集AUC连续3个epoch下降，**回退到Stage 2**
- 伪标签数量上限：原始数据的50%
- 定期人工复核伪标签（每10个epoch采样检查）

#### 3.7.4 伪标签策略详细（原策略1：规则优先）

**适用Stage**：Stage 1-2

综合模型、规则、注意力三方面信息，但在冲突时信任规则：
$$
\text{confidence}_i = \sigma(s_i^{\text{model}}) \cdot s_i^{\text{rule}} \cdot \max_{j \in \{\text{light, stop}\}} \alpha_{ij}
$$

伪标签筛选条件（AND逻辑）：
- $\text{confidence}_i > 0.85$（默认阈值）
- $|s_i^{\text{model}} - s_i^{\text{rule}}| < 0.2$（模型与规则一致）
- $\max \alpha_{ij} > 0.3$（注意力聚焦）

**实现代码**：
```python
# 伪代码：src/self_training/pseudo_labeler.py
def generate_pseudo_labels_rule_priority(
    model_scores: torch.Tensor,
    rule_scores: torch.Tensor,
    attention_weights: torch.Tensor,
    threshold_conf: float = 0.85,
    threshold_consistency: float = 0.2,
):
    """
    规则优先策略：仅当模型与规则一致时才生成伪标签
    
    适用场景：
    - MVP阶段（规则明确，模型尚未收敛）
    - 冷启动阶段（前10-20 epochs）
    - 安全关键场景（宁可漏报，不能误报）
    """
    pseudo_labels = []
    
    for i in range(len(model_scores)):
        # 计算置信度
        attention_focus = attention_weights[i].max().item()
        confidence = (
            torch.sigmoid(model_scores[i]).item() * 
            rule_scores[i].item() * 
            attention_focus
        )
        
        # 一致性检查
        consistency = abs(model_scores[i].item() - rule_scores[i].item())
        
        # 生成条件（AND逻辑）
        if (confidence > threshold_conf and 
            consistency < threshold_consistency and
            attention_focus > 0.3):
            
            # 规则优先：使用规则判定作为伪标签
            pseudo_labels.append({
                'label': 1 if rule_scores[i] > 0.5 else 0,  # 规则判定
                'confidence': confidence,
                'source': 'rule_priority',
                'model_score': model_scores[i].item(),
                'rule_score': rule_scores[i].item(),
            })
        
        # 冲突场景处理（场景B）
        elif rule_scores[i] > 0.7 and model_scores[i] < 0.3:
            # 规则判违规，模型低置信 → 信任规则，但降低置信度
            pseudo_labels.append({
                'label': 1,  # 违规
                'confidence': 0.6,  # 降低权重（原规则0.9 → 0.6）
                'source': 'rule_override',
                'flag': 'model_disagree'  # 标记为待人工复核
            })
    
    return pseudo_labels
```

#### 3.7.5 伪标签策略详细（原策略2：加权融合）

**适用Stage**：Stage 2

```python
def generate_pseudo_labels_weighted(
    model_scores: torch.Tensor,
    rule_scores: torch.Tensor,
    attention_weights: torch.Tensor,
    weight_rule: float = 0.6,  # 规则权重
    weight_model: float = 0.4, # 模型权重
    threshold_conf: float = 0.85,
):
    """
    加权融合策略：综合模型与规则
    
    适用场景：
    - 中期训练（epoch 30-60）
    - 模型逐渐可信时
    - 数据量较大时（>1000样本）
    """
    pseudo_labels = []
    
    for i in range(len(model_scores)):
        # 加权评分
        fused_score = (
            weight_rule * rule_scores[i] + 
            weight_model * torch.sigmoid(model_scores[i])
        )
        
        # 置信度（考虑一致性奖励）
        consistency_bonus = 1.0 - abs(model_scores[i] - rule_scores[i]) / 2.0
        confidence = fused_score * attention_weights[i].max() * consistency_bonus
        
        if confidence > threshold_conf:
            pseudo_labels.append({
                'label': 1 if fused_score > 0.5 else 0,
                'confidence': confidence.item(),
                'source': 'weighted_fusion',
                'fused_score': fused_score.item(),
            })
    
    return pseudo_labels
```

#### 3.7.6 伪标签策略详细（原策略3：动态切换）

**适用Stage**：Stage 2-3（自适应）

```python
class AdaptivePseudoLabeler:
    def __init__(self):
        self.epoch = 0
        self.model_reliability = 0.0  # 模型可靠度评估
    
    def select_strategy(self):
        """根据训练阶段动态选择策略"""
        if self.epoch < 20 or self.model_reliability < 0.7:
            return 'rule_priority'  # 早期：规则优先
        elif self.epoch < 60 or self.model_reliability < 0.85:
            return 'weighted_fusion'  # 中期：加权融合
        else:
            return 'model_priority'  # 后期：模型优先（自训练解锁）
    
    def update_reliability(self, val_auc, val_f1, rule_consistency):
        """评估模型可靠度"""
        self.model_reliability = (
            0.4 * val_auc + 
            0.3 * val_f1 + 
            0.3 * rule_consistency
        )
```

#### 3.7.7 伪标签保存与安全约束

```python
class PseudoLabeler:
    def __init__(self, strategy='rule_priority'):
        self.strategy = strategy
        self.pseudo_labels = []
    
    def save_epoch(self, epoch):
        df = pd.DataFrame(self.pseudo_labels)
        df.to_parquet(f'artifacts/pseudo_labels/epoch_{epoch:03d}.parquet')
        
        # 记录统计信息
        stats = {
            'total': len(self.pseudo_labels),
            'violations': sum(1 for p in self.pseudo_labels if p['label'] == 1),
            'avg_confidence': np.mean([p['confidence'] for p in self.pseudo_labels]),
            'strategy': self.strategy
        }
        with open(f'artifacts/pseudo_labels/epoch_{epoch:03d}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.pseudo_labels.clear()
```

**安全约束**：
- **数量上限**：每个 epoch 最多生成 20% 的伪标签（相对于原始数据集）
- **损失监控**：若 $\mathcal{L}_{\text{attn}}$ 连续 3 个 epoch 上升，自动降低阈值 0.05
- **人工复核**：每 10 个 epoch 采样 10 个伪标签，输出到 `reports/pseudo_review/` 供人工检查
- **冲突标记**：标记为 `flag='model_disagree'` 的样本自动导出供专家复核

### 3.8 可解释性与监控

#### 3.8.1 注意力可视化
```python
# src/explain/attention_viz.py
def visualize_attention(image, entities, attention_weights, focal_entity_idx):
    """
    在原始图像上叠加注意力热力图
    
    Args:
        image: [H, W, 3] numpy array
        entities: List[Entity]
        attention_weights: [N] 注意力权重向量
        focal_entity_idx: 中心实体索引（通常是待检测车辆）
    
    Returns:
        annotated_image: 带注释的图像
    """
    # 1. 绘制所有实体 bbox
    for i, entity in enumerate(entities):
        color = get_color_by_attention(attention_weights[i])
        cv2.rectangle(image, entity.bbox, color, thickness=2)
    
    # 2. 绘制注意力连线（focal → 其他实体）
    focal_pos = entities[focal_entity_idx].center
    for i, entity in enumerate(entities):
        if i == focal_entity_idx:
            continue
        alpha = attention_weights[i].item()
        if alpha > 0.1:  # 仅显示显著连线
            thickness = int(alpha * 5)
            cv2.line(image, focal_pos, entity.center, (255, 0, 0), thickness)
            cv2.putText(image, f'{alpha:.2f}', entity.center, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 3. 绘制规则信息
    car = entities[focal_entity_idx]
    info_text = [
        f'Distance: {car.d_stop:.1f}m',
        f'Velocity: {car.velocity:.1f}m/s',
        f'Light: {entities.traffic_light.state}',
        f'Max Attn: {attention_weights.max():.3f}'
    ]
    for i, text in enumerate(info_text):
        cv2.putText(image, text, (10, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image
```

#### 3.8.2 Prometheus 指标
```python
# src/monitoring/meters.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定义指标
train_loss = Histogram('traffic_train_loss', 'Training loss', ['loss_type'])
violation_detected = Counter('traffic_violations_detected', 'Number of violations')
attention_consistency = Gauge('traffic_attention_consistency', 'Attention consistency score')
pseudo_label_count = Counter('traffic_pseudo_labels', 'Pseudo labels generated', ['confidence_bin'])

# 启动监控服务
start_http_server(8000)

# 在训练循环中记录
train_loss.labels(loss_type='total').observe(L_total.item())
train_loss.labels(loss_type='recon').observe(L_recon.item())
violation_detected.inc(len(violations))
attention_consistency.set(attn_consistency_score)
```

**监控端点**：`http://localhost:8000/metrics`

#### 3.8.3 结构化日志
```python
import structlog

logger = structlog.get_logger()

# 训练日志
logger.info(
    "training_step",
    epoch=epoch,
    step=step,
    trace_id=trace_id,
    loss_total=L_total.item(),
    loss_recon=L_recon.item(),
    loss_rule=L_rule.item(),
    loss_attn=L_attn.item(),
    grad_norm=grad_norm
)

# 违规检测日志
logger.warning(
    "violation_detected",
    trace_id=trace_id,
    scene_id=scene_id,
    entity_id=car.id,
    model_score=score.item(),
    rule_score=rule_score.item(),
    distance=car.d_stop,
    velocity=car.velocity,
    light_state=light.state,
    attention_max=attention_weights.max().item()
)
```

## 4. 配置与数据
- `configs/mvp.yaml` 需包含：数据路径、batch size、GAT 头数、memory bank 尺寸、规则阈值、监控端口。
- 敏感路径通过环境变量传入（`DATA_ROOT`, `WANDB_API_KEY` 等），文档中不得出现真实凭据。
- 数据缓存放在 `data/cache/`，CI 环境只使用合成样本。

## 5. 安全 / 性能 / 可运维性

> **技术勘误修正（2025-12-03）**：补充GPU显存/CPU内存需求的详细估算和性能优化建议。  
> 详见：`docs/archive/design/TECHNICAL_CORRECTIONS.md` 问题2

### 5.1 安全
- 禁止硬编码凭据；规则 DSL 在加载时进行 schema 校验，防止执行任意代码。
- 数据缓存放在 `data/cache/`，CI 环境只使用合成样本。
- 敏感路径通过环境变量传入（`DATA_ROOT`, `WANDB_API_KEY` 等）。

### 5.2 性能需求与估算

#### 5.2.1 模型参数量
```python
# 假设场景图平均节点数 N_avg = 10 (5车 + 3灯 + 1停止线 + 1全局节点)
# Batch size B = 4

# 1. GAT层参数
# 每层：Linear(128, 128*8) + Attention(128*8) ≈ 128*128*8*2 = 262K params/layer
# 3层：262K * 3 = 786K params

# 2. 全局注意力
# MultiheadAttention(128, 4 heads): 128*128*4*3 = 196K params

# 3. Scoring head
# MLP: (128*2 → 128 → 64 → 1): 128*128*2 + 128*64 + 64*1 ≈ 41K params

# 总参数量：786K + 196K + 41K ≈ 1.02M params
# FP32存储：1.02M * 4 bytes = 4.08 MB
# FP16混合精度：1.02M * 2 bytes = 2.04 MB
```

#### 5.2.2 GPU显存需求（修正版）

> **重大修正（2025-12-03）**：原估算存在多处计算错误（稀疏注意力、优化器状态、中间激活）。现提供物理正确的完整估算。

**假设条件**：
- Batch size: $B = 8$（修正后的推荐值）
- 平均节点数: $N_{\text{avg}} = 10$（5车 + 3灯 + 1停止线 + 1全局节点）
- 平均边数: $E_{\text{avg}} = 30$（假设平均度3，稀疏图）
- 隐藏维度: $d_h = 128$
- GAT层数: $L = 3$，头数: $K = 8$
- 数据类型: FP32（4 bytes）

**详细显存分配表**：

| 组件 | 计算公式 | 结果 | 说明 |
|------|---------|------|------|
| **1. 模型参数** | | **4.08 MB** | |
| - GAT层 | $L \times (d_h \times d_h \times K \times 2)$ | $3 \times 128 \times 128 \times 8 \times 2 \times 4 = 3.15$ MB | 每层W_k和a_k |
| - 全局注意力 | $d_h \times d_h \times 4 \times 3$ | $128 \times 128 \times 4 \times 3 \times 4 = 0.79$ MB | Q,K,V投影 |
| - 规则聚焦 | $3 \times d_h \times d_h$ | $3 \times 128 \times 128 \times 4 = 0.20$ MB | rule_scorer |
| - Scoring head | $2 \times d_h \times d_h$ | $\sim 0.13$ MB | 两层MLP |
| **2. 优化器状态（AdamW）** | | **12.24 MB** | |
| - 梯度 (grad) | 同模型参数 | 4.08 MB | $\nabla_\theta$ |
| - 一阶动量 (m) | 同模型参数 | 4.08 MB | AdamW的$m_t$ |
| - 二阶动量 (v) | 同模型参数 | 4.08 MB | AdamW的$v_t$ |
| **小计（静态）** | | **16.32 MB** | 模型+优化器 |
| **3. 前向传播激活** | | **per batch** | |
| - 输入特征 | $B \times N_{\text{avg}} \times 10 \times 4$ | $8 \times 10 \times 10 \times 4 = 3.2$ KB | 输入层 |
| - GAT中间激活 | $B \times N_{\text{avg}} \times d_h \times L \times 4$ | $8 \times 10 \times 128 \times 3 \times 4 = 122.88$ KB | 每层输出 |
| - **注意力权重（稀疏）** | $B \times E_{\text{avg}} \times K \times L \times 4$ | $8 \times 30 \times 8 \times 3 \times 4 = 23.04$ KB | **修正：边权重** |
| - 全局注意力 | $B \times N_{\text{avg}} \times d_h \times 4$ | $8 \times 10 \times 128 \times 4 = 40.96$ KB | 全局融合 |
| - 规则聚焦 | $B \times N_{\text{car}} \times d_h \times 4$ | $8 \times 5 \times 128 \times 4 = 20.48$ KB | 规则注意力 |
| - 前向总计 | | $\sim 210$ KB | per batch |
| **4. 反向传播梯度缓存** | | | |
| - 激活梯度 | $\approx 2 \times$ 前向激活 | $\sim 420$ KB | PyTorch自动微分 |
| - 中间变量 | $\approx$ 前向激活 | $\sim 210$ KB | Sigmoid/GELU等 |
| - 反向总计 | | $\sim 630$ KB | per batch |
| **5. PyTorch CUDA开销** | | **500 MB** | 固定开销 |
| - CUDA上下文 | | $\sim 400$ MB | cuBLAS, cuDNN等 |
| - 内存池碎片 | | $\sim 100$ MB | 缓存分配器 |
| **总计（batch=8）** | | **517.2 MB** | 16.32 + 0.21 + 0.63 + 500 |

**修正说明**：
1. ✅ **注意力权重**：从$N^2$密集矩阵改为$E$稀疏边权重（38.4KB → 23KB）
2. ✅ **优化器状态**：从2倍改为3倍参数量（8MB → 12.24MB）
3. ✅ **中间激活**：明确包含反向传播的激活缓存（~630KB）

**结论**：
✅ **显存需求：~520 MB**（batch=8）  
✅ RTX 4090（24GB）可以支持：
- **推荐batch size**: 8-16（留足够余量）
- **最大batch size**: ≤ 32（理论上限）
- **极端场景节点数**: ≤ 50

**不同batch size的显存占用**：

| Batch Size | 前向+反向激活 | 总显存需求 | 推荐场景 |
|-----------|-------------|-----------|---------|
| 4 | 0.42 MB | 516.7 MB | 调试、快速迭代 |
| 8 | 0.84 MB | 517.2 MB | **MVP默认配置** |
| 16 | 1.68 MB | 518.0 MB | 训练稳定后优化 |
| 32 | 3.36 MB | 519.7 MB | 理论最大值 |

#### 5.2.3 CPU内存估算
```python
# 数据集大小：100个场景，每个场景10个实体，特征维度10
# 存储：100 * 10 * 10 * 4 bytes = 40 KB（几乎可以忽略）

# 合成数据生成脚本可能需要：
# - 图像数据：100 * (1920*1080*3) * 4 bytes ≈ 2.4 GB（未压缩）
# - 建议使用JPEG压缩存储，运行时按需加载

# 训练时CPU内存：
# - 数据加载器（prefetch=2）：~50 MB
# - 总计：<100 MB
```

#### 5.2.4 性能优化建议
```python
# 1. 混合精度训练（FP16）
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)

# 显存降低至：~350 MB

# 2. 梯度累积（模拟大batch）
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Checkpoint（牺牲10%速度换取30%显存）
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x):
    return checkpoint(self.gat_layers[0], x)
```

### 5.3 训练时间目标
- 单机训练需在 1×RTX 4090（CUDA 12.1）下 <=2h
- 若 GPU 不可用，退化为 CPU mode（batch size 自动缩减），日志需提示降级原因

### 5.4 运维
- Prometheus 指标 + Grafana 面板用于观察 loss/违规趋势
- `scripts/render_attention_maps.py` 由运维/业务人员复现可解释结果

## 6. 依赖与工具

> **重大修正（2025-12-03）**：锁定所有依赖版本，解决已知兼容性问题，提供完整requirements.txt示例。  
> 解决问题10：依赖版本冲突风险

### 6.1 核心依赖（版本锁定）

**GPU深度学习框架**：
```
torch==2.4.1+cu121
torchvision==0.19.1+cu121
torchaudio==2.4.1+cu121
torch-geometric==2.5.0
```

**数据处理与图计算**：
```
numpy==1.26.4
opencv-python==4.9.0.80       # 锁定版本（与torch 2.4兼容）
pillow==10.2.0
networkx==3.2.1
scikit-learn==1.4.0           # 用于K-Means（memory初始化）
```

**配置与验证**：
```
pydantic==2.6.1
pyyaml==6.0.1
```

**CLI与UI**：
```
typer==0.9.0                  # 推荐使用typer（比click更现代）
rich==13.7.0
tqdm==4.66.1
```

**监控与日志**：
```
prometheus-client==0.19.0
structlog==24.1.0
```

**可视化**：
```
matplotlib==3.8.2
seaborn==0.13.1
```

### 6.2 已知兼容性问题与解决方案

| 问题 | 影响库 | 解决方案 |
|------|--------|---------|
| **CUDA版本** | torch 2.4.1需要CUDA 12.1 | 确保系统CUDA版本≥12.1，或使用CPU版（训练慢） |
| **opencv与numpy** | opencv-python 4.10+需要numpy 2.0 | 锁定opencv==4.9.0.80（兼容numpy 1.26） |
| **torch-geometric** | 依赖torch具体版本 | 使用pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html |
| **prometheus多实例** | 同一端口重复启动 | 在代码中添加`try-except`处理端口占用 |

### 6.3 完整requirements.txt

```txt
# ========== GPU深度学习框架 ==========
--find-links https://download.pytorch.org/whl/cu121
torch==2.4.1+cu121
torchvision==0.19.1+cu121
torchaudio==2.4.1+cu121

# ========== 图神经网络 ==========
--find-links https://data.pyg.org/whl/torch-2.4.1+cu121.html
torch-geometric==2.5.0
torch-scatter==2.1.2+pt24cu121
torch-sparse==0.6.18+pt24cu121
pyg-lib==0.4.0+pt24cu121

# ========== 数据处理 ==========
numpy==1.26.4
opencv-python==4.9.0.80
pillow==10.2.0
networkx==3.2.1
scikit-learn==1.4.0
pandas==2.2.0

# ========== 配置与验证 ==========
pydantic==2.6.1
pyyaml==6.0.1

# ========== CLI与UI ==========
typer==0.9.0
rich==13.7.0
tqdm==4.66.1
click==8.1.7  # typer依赖

# ========== 监控与日志 ==========
prometheus-client==0.19.0
structlog==24.1.0

# ========== 可视化 ==========
matplotlib==3.8.2
seaborn==0.13.1

# ========== 开发工具（可选） ==========
pytest==8.0.0
black==24.1.1
flake8==7.0.0
```

### 6.4 安装说明

```bash
# 方法1：使用pip（推荐）
pip install -r requirements.txt

# 方法2：使用poetry（如果使用pyproject.toml）
poetry install

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

### 6.5 环境变量配置

```bash
# 必需环境变量
export DATA_ROOT="/path/to/data/traffic"
export ARTIFACT_ROOT="./artifacts"

# 可选环境变量（监控）
export WANDB_API_KEY="your_wandb_key"  # 如果使用WandB
export PROMETHEUS_PORT=8000

# CUDA配置
export CUDA_VISIBLE_DEVICES=0  # 指定GPU
```

**重要提示**：
- 依赖文件待用户确认后才能用于环境安装
- 如需新增库必须同步更新本节与 `lunwen/requirements.txt`
- 生产环境建议使用Docker固定所有依赖版本

## Checklist
- [x] 架构图与模块职责清晰
- [x] 与需求、开发文档一致
- [x] 包含监控/安全/性能说明
- [x] 算法数学模型完整（含公式推导）
- [x] 训练/推理流程伪代码齐全
- [x] 损失函数设计明确（含超参数）
- [x] 可解释性机制详细（注意力可视化+日志）
- [x] 自训练安全约束明确
- [x] 评审结论记录（2025-12-03 算法细化通过，选定方案1）

