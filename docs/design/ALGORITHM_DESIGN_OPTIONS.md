# 算法设计方案（ITER-2025-01 细化版）

## 元数据
| 字段 | 内容 |
| --- | --- |
| 文档版本 | v2.0（方案1系统性重构后） |
| 原版本 | v1.0（存在设计问题） |
| 迭代编号 | ITER-2025-01 |
| 创建时间 | 2025-12-03 |
| 责任人 | 算法架构师 |
| 状态 | ✅ 重构完成（方案1已完善） |
| 关联文档 | `Design-ITER-2025-01.md` v2.0, `Requirement-ITER-2025-01.md` |
| 重构追踪 | `DESIGN_REFACTOR_TRACKER.md` |

## 文档目的
针对"红灯停无监督异常检测"场景，提供**3种技术路线**的完整算法设计，包括：
- 数学模型与公式推导
- 网络架构与超参数
- 训练/推理算法伪代码
- 优劣势对比与选型建议

---

# 方案对比总览

| 维度 | 方案1：多阶段注意力GAT + 硬约束 | 方案2：记忆增强对比学习 + 软规则 | 方案3：因果图网络 + 反事实推理 |
|------|------|------|------|
| **核心思想** | 显式规则约束 + 多尺度注意力 | 正常模式记忆库 + 对比度量 | 因果推理 + 反事实解释 |
| **监督需求** | 无标签（规则代替） | 无标签（自监督） | 无标签（因果先验） |
| **可解释性** | ★★★☆☆ 注意力权重 | ★★★★☆ 记忆检索路径 | ★★★★★ 因果链推理 |
| **工程复杂度** | ★★☆☆☆ 中等 | ★★★☆☆ 较高 | ★★★★☆ 高 |
| **训练稳定性** | ★★★★☆ 规则提供强监督 | ★★★☆☆ 对比学习需调参 | ★★☆☆☆ 因果发现易过拟合 |
| **扩展性** | ★★★☆☆ 新规则需手工编写 | ★★★★☆ 记忆库自适应 | ★★★★★ 因果图可迁移 |
| **计算成本** | ★★★☆☆ 中等（多头注意力） | ★★★★☆ 高（检索+对比） | ★★★★★ 很高（因果推理） |
| **MVP 适配度** | ★★★★★ 最适合快速交付 | ★★★☆☆ 需更多数据 | ★★☆☆☆ 研究性强 |

**推荐**: 
- **MVP 首选：方案1**（工程风险低，可解释性达标）
- **ITER-02 演进：方案2**（数据量增加后性能更优）
- **论文创新：方案3**（学术价值高，可作为长期方向）

---

# 方案1：多阶段注意力增强 GAT + 硬约束规则融合

## 1.1 核心思想
将交通场景建模为**异构时空图**，通过多头图注意力网络（GAT）学习实体间关系，同时引入**显式规则约束损失**强制模型符合交通法规。采用**三阶段注意力机制**（局部→全局→规则聚焦）提升违规实体的识别准确性。

## 1.2 数学模型

### 1.2.1 场景图定义
给定时刻 $t$ 的交通场景，构造有向图 $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t, \mathbf{X}_t, \mathbf{A}_t)$：

$$
\begin{aligned}
\mathcal{V}_t &= \{v_1, \dots, v_{N_{\text{car}}}, v_{N_{\text{car}}+1}, \dots, v_{N_{\text{car}}+N_{\text{light}}}, v_{\text{stop}}\} \\
\mathbf{X}_t &\in \mathbb{R}^{|\mathcal{V}_t| \times d_{\text{feat}}} \quad \text{(节点特征矩阵)} \\
\mathbf{A}_t &\in \{0,1\}^{|\mathcal{V}_t| \times |\mathcal{V}_t|} \quad \text{(邻接矩阵)}
\end{aligned}
$$

**节点特征** $\mathbf{x}_i$ 包含：
- 车辆节点：位置 $(x, y)$、速度 $(v_x, v_y)$、朝向 $\theta$、bounding box $(w, h)$、停止线距离 $d_{\text{stop}}$
- 交通灯节点：位置、状态 one-hot `[red, yellow, green]`、置信度
- 停止线节点：线段端点 $(x_1, y_1, x_2, y_2)$

**边构建策略**：
$$
\mathbf{A}_{ij} = \begin{cases}
1, & \text{if } \|\mathbf{p}_i - \mathbf{p}_j\|_2 < r_{\text{spatial}} \text{ and } \text{type}(v_i) \neq \text{type}(v_j) \\
1, & \text{if } v_i \text{ is car and } v_j \text{ is nearest traffic light} \\
0, & \text{otherwise}
\end{cases}
$$
其中 $r_{\text{spatial}} = 50m$（可配置）。

### 1.2.2 多阶段注意力架构

> **技术勘误修正（2025-12-03）**：补充三阶段注意力的详细实现细节，明确局部→全局→规则聚焦的具体机制。  
> 详见：`docs/design/TECHNICAL_CORRECTIONS.md` 问题5

#### 阶段1：局部关系编码（Local GAT）

**定义**：基于空间邻近性和实体类型的**稀疏图注意力**。

**邻接矩阵构建策略**：
```python
def build_local_adjacency(entities, r_spatial=50.0):
    """
    局部邻接：仅连接空间邻近且异构的实体
    
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
            
            dist = np.linalg.norm(e_i.pos - e_j.pos)
            
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

对每个车辆节点，学习其与邻近实体的关系：

$$
\begin{aligned}
\mathbf{h}_i^{(0)} &= \text{LayerNorm}(\mathbf{W}_0 \mathbf{x}_i + \mathbf{b}_0) \\
\alpha_{ij}^{(l,k)} &= \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^\top [\mathbf{W}_k^{(l)} \mathbf{h}_i^{(l-1)} \| \mathbf{W}_k^{(l)} \mathbf{h}_j^{(l-1)}]\right)\right)}{\sum_{j' \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^\top [\mathbf{W}_k^{(l)} \mathbf{h}_i^{(l-1)} \| \mathbf{W}_k^{(l)} \mathbf{h}_{j'}^{(l-1)}]\right)\right)} \\
\mathbf{h}_i^{(l,k)} &= \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l,k)} \mathbf{W}_k^{(l)} \mathbf{h}_j^{(l-1)} \\
\mathbf{h}_i^{(l)} &= \text{GELU}\left(\frac{1}{K} \sum_{k=1}^K \mathbf{h}_i^{(l,k)}\right) + \mathbf{h}_i^{(l-1)} \quad \text{(多头平均 + 残差)}
\end{aligned}
$$

**特点**：
- ✅ 稀疏连接（边数 $E \ll N^2$）
- ✅ 空间局部性（不同类型实体有不同连接半径）
- ✅ 多跳传播（3层GAT → 3跳感受野）

超参数：$L=3$ 层，$K=8$ 头，$d_h = 128$ 隐藏维度。

#### 阶段2：全局上下文融合（Global Attention）

**定义**：通过**虚拟全局节点**聚合场景级上下文（类似Transformer的[CLS] token）。

**实现代码**：
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
        # Step 1: 全局节点聚合所有局部节点信息
        global_context, attn_weights = self.multihead_attn(
            query=self.global_query.unsqueeze(0),
            key=h_local.unsqueeze(0),
            value=h_local.unsqueeze(0)
        )
        
        # Step 2: 广播全局信息到每个局部节点
        global_context = global_context.squeeze(0).expand(N, -1)
        
        # Step 3: 融合局部+全局（残差连接）
        h_fused = torch.cat([h_local, global_context], dim=-1)
        h_global = self.fusion(h_fused) + h_local
        
        return h_global, attn_weights
```

**数学形式**：
$$
\begin{aligned}
\mathbf{g} &= \text{softmax}\left(\frac{\mathbf{Q}_g \mathbf{K}_h^\top}{\sqrt{d_h}}\right) \mathbf{V}_h \quad \text{where } \mathbf{K}_h = [\mathbf{h}_1^{(L)}, \dots, \mathbf{h}_N^{(L)}] \\
\tilde{\mathbf{h}}_i &= \mathbf{h}_i^{(L)} + \text{MLP}_{\text{fuse}}([\mathbf{h}_i^{(L)} \| \mathbf{g}])
\end{aligned}
$$

**特点**：
- ✅ 全连接（全局节点与所有局部节点交互）
- ✅ 场景级信息（交通密度、整体流动性等）
- ✅ 计算高效（O(N) vs Transformer的O(N²)）

#### 阶段3：规则聚焦注意力（Rule-Focused Attention）

**定义**：基于**规则语义**的加权注意力重分配。

**实现代码**：
```python
class RuleFocusedAttention(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 规则相关性评分网络
        self.rule_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 规则嵌入（可学习）
        self.rule_embeddings = nn.Embedding(5, hidden_dim)
    
    def forward(self, h_fused, entity_types, rule_id=0):
        # 提取规则相关实体
        car_mask = (entity_types == 0)
        h_cars = h_fused[car_mask]
        
        # 计算每个车辆与规则相关实体的注意力
        rule_emb = self.rule_embeddings(torch.tensor([rule_id]))
        
        for h_car in h_cars:
            concat_feat = torch.cat([h_car, h_light, h_stop], dim=0)
            rule_score = self.rule_scorer(concat_feat)
            h_weighted = h_car * rule_score + rule_emb * (1 - rule_score)
        
        return h_rule_focused, rule_attention
```

**数学形式**：
$$
\begin{aligned}
\beta_{i,\text{light}} &= \text{sigmoid}\left(\mathbf{w}_{\text{rule}}^\top [\tilde{\mathbf{h}}_i \| \mathbf{h}_{\text{light}} \| \mathbf{h}_{\text{stop}}]\right) \\
\mathbf{h}_i^{\text{rule}} &= \beta_{i,\text{light}} \odot \tilde{\mathbf{h}}_i + (1-\beta_{i,\text{light}}) \odot \mathbf{e}_{\text{rule}}
\end{aligned}
$$

**特点**：
- ✅ 规则语义注入（可学习的rule embedding）
- ✅ 动态聚焦（不同车辆获得不同权重）
- ✅ 可扩展（支持多种规则）

最终异常分数：
$$
s_i^{\text{model}} = \sigma\left(\text{MLP}_{\text{score}}(\mathbf{h}_i^{\text{rule}})\right) \in [0,1]
$$

### 1.2.3 规则约束损失

> **重大修正（2025-12-03）**：原公式存在距离项逻辑错误和速度项边界条件错误。现重新设计物理正确的规则分数公式，区分"接近停止线"和"闯过停止线"两种情况。  
> 详见：`docs/design/TECHNICAL_CORRECTIONS.md` 问题1 + 系统性重构

#### 红灯停规则形式化

**物理模型说明**：
- **规则分数语义**：违规程度（0=无违规，1=严重违规）
- **距离约定**：$d > 0$表示车辆在停止线前，$d < 0$表示车辆已过停止线
- **速度约定**：$v = 0$表示完全停止（无违规），$v > 0$表示移动中

**硬阈值版（用于验收测试）**：
$$
\text{violation}(i) = \begin{cases}
1, & \text{if } \text{light}_{\text{state}} = \text{red} \land \left(d_{\text{stop}}(i) < 0 \lor (0 \le d_{\text{stop}}(i) < \tau_d \land v(i) > \tau_v)\right) \\
0, & \text{otherwise}
\end{cases}
$$
其中 $\tau_d = 5m$，$\tau_v = 0.5 m/s$。

**规则分数（软化版，完全可微分）**：

使用Gumbel-Softmax软化交通灯状态：
$$
w_{\text{light}} = \text{GumbelSoftmax}([p_{\text{red}}, p_{\text{yellow}}, p_{\text{green}}], \tau_{\text{temp}}=0.5)[0]
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
- $\alpha_{\text{cross}} = 3.0$：过线违规敏感度
- $\alpha_d = 2.0$：接近停止线敏感度
- $\alpha_v = 5.0$：速度敏感度
- $\tau_d = 5.0m$，$\tau_v = 0.5 m/s$：阈值

**物理意义验证**：
- 完全停止（$v=0$，$d=10m>\tau_d$）：$s^{\text{rule}} = 0$ ✅
- 闯过停止线（$d=-2m$，$v=2m/s$）：$s^{\text{rule}} \approx 0.998$ ✅
- 远离停止线（$d=10m$）：$s^{\text{rule}} = 0$ ✅

**实现代码**：
```python
import torch
import torch.nn.functional as F

def compute_rule_score_differentiable(
    light_probs: torch.Tensor,  # [B, 3] - [red, yellow, green]
    distances: torch.Tensor,    # [B] - distance (正=未过线，负=已过线)
    velocities: torch.Tensor,   # [B] - vehicle velocity
    tau_d: float = 5.0,
    tau_v: float = 0.5,
    alpha_d: float = 2.0,
    alpha_v: float = 5.0,
    alpha_cross: float = 3.0,
    temperature: float = 0.5,
    training: bool = True,
):
    """物理正确的完全可导规则评分函数"""
    # Gumbel-Softmax软化
    if training:
        light_weights = F.gumbel_softmax(
            torch.log(light_probs + 1e-10), 
            tau=temperature, 
            hard=False
        )[:, 0]
    else:
        light_weights = light_probs[:, 0]
    
    # 分段距离-速度评分
    B = distances.size(0)
    f_dv = torch.zeros(B, device=distances.device)
    
    # 情况1：已过线（d < 0）
    crossed_mask = (distances < 0)
    if crossed_mask.any():
        f_dv[crossed_mask] = (
            torch.sigmoid(alpha_cross * (-distances[crossed_mask])) *
            torch.sigmoid(alpha_v * velocities[crossed_mask])
        )
    
    # 情况2：接近停止线（0 <= d < tau_d）
    approaching_mask = (distances >= 0) & (distances < tau_d)
    if approaching_mask.any():
        f_dv[approaching_mask] = (
            torch.sigmoid(alpha_d * (tau_d - distances[approaching_mask])) *
            torch.sigmoid(alpha_v * (velocities[approaching_mask] - tau_v))
        )
    
    # 情况3：远离停止线（d >= tau_d）：f_dv保持为0
    
    # 组合
    rule_scores = light_weights * f_dv
    
    return rule_scores
```

#### 总损失函数

> **重大修正（2025-12-03）**：统一注意力一致性损失定义，明确GAT注意力与规则聚焦注意力的关系。

$$
\begin{aligned}
\mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{rule}} + \lambda_2 \mathcal{L}_{\text{attn}} + \lambda_3 \mathcal{L}_{\text{reg}} \\
\\
\mathcal{L}_{\text{recon}} &= -\frac{1}{N_{\text{car}}} \sum_{i=1}^{N_{\text{car}}} \left[s_i^{\text{rule}} \log s_i^{\text{model}} + (1-s_i^{\text{rule}}) \log(1-s_i^{\text{model}})\right] \quad \text{(BCE)} \\
\\
\mathcal{L}_{\text{rule}} &= \frac{1}{N_{\text{car}}} \sum_{i=1}^{N_{\text{car}}} \left|s_i^{\text{model}} - s_i^{\text{rule}}\right|^2 \quad \text{(MSE)} \\
\\
\mathcal{L}_{\text{attn}} &= \mathcal{L}_{\text{attn}}^{\text{GAT}} + \mathcal{L}_{\text{attn}}^{\text{rule}} \quad \text{(双层监督)} \\
\\
\mathcal{L}_{\text{attn}}^{\text{GAT}} &= \frac{1}{|\mathcal{I}_{\text{viol}}|} \sum_{i \in \mathcal{I}_{\text{viol}}} \left(1 - \max_{j \in \mathcal{N}_{\text{rule}}(i)} \alpha_{ij}^{(L)}\right)^2 \quad \text{(局部注意力)} \\
\\
\mathcal{L}_{\text{attn}}^{\text{rule}} &= \frac{1}{|\mathcal{I}_{\text{viol}}|} \sum_{i \in \mathcal{I}_{\text{viol}}} \left(1 - \beta_i\right)^2 \quad \text{(规则聚焦)} \\
\\
\mathcal{L}_{\text{reg}} &= \sum_{l=1}^L \|\mathbf{W}^{(l)}\|_F^2
\end{aligned}
$$

其中：
- $\mathcal{I}_{\text{viol}} = \{i : s_i^{\text{rule}} > 0.5\}$：违规车辆集合
- $\mathcal{N}_{\text{rule}}(i) = \{j : j \in \mathcal{N}(i) \land \text{type}(j) \in \{\text{light, stop}\}\}$：车辆$i$的规则相关邻居
- $\alpha_{ij}^{(L)}$：GAT第$L$层的边注意力权重
- $\beta_i$：规则聚焦注意力分数

超参数：$\lambda_1 = 0.5$，$\lambda_2 = 0.3$（其中GAT和规则各占一半），$\lambda_3 = 1e-4$。

## 1.3 训练算法

```python
Algorithm: Multi-Stage Attention GAT Training

Input: 
  - Dataset D = {G_1, ..., G_M} (scene graphs)
  - Rule thresholds τ_d, τ_v
  - Hyperparameters: epochs E, batch_size B, lr η
  
Output: 
  - Trained model θ*
  
1: Initialize model parameters θ ~ N(0, 0.02)
2: optimizer ← AdamW(θ, lr=η, weight_decay=1e-4)
3: scheduler ← CosineAnnealingLR(optimizer, T_max=E)
4: 
5: for epoch = 1 to E do
6:     for batch G_b in DataLoader(D, batch_size=B, shuffle=True) do
7:         # Forward pass
8:         X, A, entities ← G_b.unpack()
9:         
10:        # Stage 1: Local GAT
11:        H^(0) ← LayerNorm(W_0 X + b_0)
12:        for layer l = 1 to L do
13:            for head k = 1 to K do
14:                α^(l,k) ← MultiHeadAttention(H^(l-1), A)
15:                H^(l,k) ← MessagePassing(H^(l-1), α^(l,k))
16:            H^(l) ← GELU(Mean(H^(l,1:K))) + H^(l-1)
17:        
18:        # Stage 2: Global context
19:        g ← GlobalAttentionPooling(H^(L))
20:        H_tilde ← H^(L) + MLP_fuse([H^(L) || g])
21:        
22:        # Stage 3: Rule-focused attention
23:        β ← RuleFocusedAttention(H_tilde, entities)
24:        H_rule ← β ⊙ H_tilde
25:        s_model ← Sigmoid(MLP_score(H_rule))
26:        
27:        # Compute rule scores
28:        s_rule ← ComputeRuleScores(entities, τ_d, τ_v)
29:        
30:        # Loss computation
31:        L_recon ← BinaryCrossEntropy(s_model, s_rule)
32:        L_rule ← MSE(s_model, s_rule)
33:        L_attn ← AttentionConsistencyLoss(α, β, s_rule)
34:        L_reg ← sum(W^2 for W in θ)
35:        
36:        L_total ← L_recon + λ_1*L_rule + λ_2*L_attn + λ_3*L_reg
37:        
38:        # Backward pass
39:        optimizer.zero_grad()
40:        L_total.backward()
41:        clip_grad_norm_(θ, max_norm=1.0)
42:        optimizer.step()
43:        
44:        # Logging
45:        if step % 50 == 0:
46:            log_metrics(L_total, L_recon, L_rule, L_attn)
47:            visualize_attention(α, β, entities)
48:    
49:    scheduler.step()
50:    
51:    # Validation
52:    if epoch % 5 == 0:
53:        val_metrics ← evaluate(model, D_val)
54:        save_checkpoint(θ, epoch, val_metrics)
55:
56: return θ
```

## 1.4 推理算法

```python
Algorithm: Violation Detection & Explanation

Input:
  - Scene graph G_t
  - Trained model θ*
  - Rule thresholds τ_d, τ_v
  
Output:
  - Violation report {entity_id, score, explanation, attention_map}

1: # Load model and preprocess
2: model ← load_checkpoint(θ*)
3: X, A, entities ← preprocess(G_t)
4:
5: # Forward inference
6: with torch.no_grad():
7:     H, α, β ← model.forward(X, A, return_attention=True)
8:     s_model ← model.score_head(H)
9:     s_rule ← compute_rule_scores(entities, τ_d, τ_v)
10:
11: # Aggregate scores
12: s_final ← 0.6 * s_model + 0.4 * s_rule
13:
14: # Generate explanations
15: violations ← []
16: for i in range(len(entities)):
17:     if s_final[i] > threshold_violation (e.g., 0.7):
18:         explanation ← {
19:             'entity_id': entities[i].id,
20:             'type': entities[i].type,
21:             'model_score': s_model[i],
22:             'rule_score': s_rule[i],
23:             'distance_to_stopline': entities[i].d_stop,
24:             'velocity': entities[i].velocity,
25:             'traffic_light_state': get_nearest_light(entities[i]).state,
26:             'attention_to_light': α[i, light_idx],
27:             'attention_to_stopline': α[i, stop_idx],
28:             'rule_focus': β[i]
29:         }
30:         
31:         # Generate attention heatmap
32:         attention_map ← visualize_attention(
33:             image=G_t.image,
34:             entities=entities,
35:             attention_weights=α[i],
36:             focal_entity=i
37:         )
38:         explanation['attention_map_path'] ← save(attention_map)
39:         
40:         violations.append(explanation)
41:
42: # Generate report
43: report ← format_report(violations, timestamp=G_t.timestamp)
44: return report
```

## 1.5 网络架构细节

```python
class MultiStageAttentionGAT(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,          # 实体特征维度
        hidden_dim: int = 128,        # GAT隐藏层维度
        num_gat_layers: int = 3,      # GAT层数
        num_heads: int = 8,           # 多头注意力头数
        dropout: float = 0.1,         # Dropout概率
        alpha: float = 0.2,           # LeakyReLU负斜率
    ):
        super().__init__()
        
        # Stage 1: Local GAT layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                negative_slope=alpha,
                add_self_loops=True,
                concat=True
            )
            for _ in range(num_gat_layers)
        ])
        
        # Stage 2: Global attention
        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Stage 3: Rule-focused attention
        self.rule_query = nn.Parameter(torch.randn(1, hidden_dim))
        self.rule_attention = nn.Linear(hidden_dim * 3, 1)
        
        # Scoring head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: Tensor,                    # [N, input_dim]
        edge_index: Tensor,           # [2, E]
        entity_types: Tensor,         # [N] (0=car, 1=light, 2=stop)
        return_attention: bool = False
    ):
        # Stage 1: Local GAT
        h = self.layer_norm(self.input_proj(x))
        attention_weights_local = []
        
        for gat_layer in self.gat_layers:
            h_new, attn = gat_layer(h, edge_index, return_attention_weights=True)
            h = F.gelu(h_new) + h  # Residual connection
            attention_weights_local.append(attn)
        
        # Stage 2: Global context
        h_global, attn_global = self.global_attn(
            query=h.unsqueeze(0),
            key=h.unsqueeze(0),
            value=h.unsqueeze(0)
        )
        h_global = h_global.squeeze(0)
        
        h_fused = h + self.fusion_mlp(torch.cat([h, h_global], dim=-1))
        
        # Stage 3: Rule-focused attention
        # Extract rule-relevant entities (traffic lights and stop lines)
        rule_mask = (entity_types == 1) | (entity_types == 2)
        h_rule_entities = h_fused[rule_mask]
        
        # Compute attention between cars and rule entities
        car_mask = (entity_types == 0)
        h_cars = h_fused[car_mask]
        
        # Broadcasting attention computation
        rule_focus = torch.zeros(h_cars.size(0), device=x.device)
        for i, h_car in enumerate(h_cars):
            attn_scores = []
            for h_rule in h_rule_entities:
                concat_feat = torch.cat([h_car, h_rule, h_car * h_rule], dim=-1)
                score = torch.sigmoid(self.rule_attention(concat_feat))
                attn_scores.append(score)
            if len(attn_scores) > 0:
                rule_focus[i] = torch.stack(attn_scores).max()
        
        h_cars_focused = h_cars * rule_focus.unsqueeze(-1)
        
        # Reconstruct full node embeddings
        h_final = torch.zeros_like(h_fused)
        h_final[car_mask] = h_cars_focused
        h_final[~car_mask] = h_fused[~car_mask]
        
        # Scoring
        scores = self.score_head(h_final[car_mask]).squeeze(-1)
        
        if return_attention:
            return scores, attention_weights_local, attn_global, rule_focus
        return scores
```

## 1.6 自训练策略详细

> **技术勘误修正（2025-12-03）**：补充自训练机制的双路径伪标签策略，解决模型与规则冲突问题。  
> 详见：`docs/design/TECHNICAL_CORRECTIONS.md` 问题4

### 1.6.1 问题分析：模型与规则冲突

| 场景 | 规则判定 | 模型输出 | 当前处理 | 问题 |
|------|---------|---------|---------|------|
| A | 违规($s^{\text{rule}}=0.9$) | 高置信($s^{\text{model}}=0.85$) | ✅ 生成伪标签 | 无冲突 |
| B | 违规($s^{\text{rule}}=0.9$) | 低置信($s^{\text{model}}=0.3$) | ❓ 未定义 | **冲突** |
| C | 正常($s^{\text{rule}}=0.1$) | 低置信($s^{\text{model}}=0.2$) | ✅ 生成伪标签 | 无冲突 |
| D | 正常($s^{\text{rule}}=0.1$) | 高置信($s^{\text{model}}=0.8$) | ❓ 未定义 | **冲突** |

### 1.6.2 策略1：规则优先（保守策略，推荐MVP）

```python
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
                'label': 1 if rule_scores[i] > 0.5 else 0,
                'confidence': confidence,
                'source': 'rule_priority',
            })
        
        # 冲突场景B处理：规则判违规，模型低置信
        elif rule_scores[i] > 0.7 and model_scores[i] < 0.3:
            pseudo_labels.append({
                'label': 1,
                'confidence': 0.6,  # 降低权重
                'source': 'rule_override',
                'flag': 'model_disagree'
            })
    
    return pseudo_labels
```

### 1.6.3 策略2：加权融合（均衡策略）

```python
def generate_pseudo_labels_weighted(
    model_scores: torch.Tensor,
    rule_scores: torch.Tensor,
    attention_weights: torch.Tensor,
    weight_rule: float = 0.6,
    weight_model: float = 0.4,
    threshold_conf: float = 0.85,
):
    """
    加权融合策略：综合模型与规则
    
    适用场景：
    - 中期训练（epoch 30-60）
    - 模型逐渐可信时
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
            })
    
    return pseudo_labels
```

### 1.6.4 策略3：动态切换（自适应策略）

```python
class AdaptivePseudoLabeler:
    def __init__(self):
        self.epoch = 0
        self.model_reliability = 0.0
    
    def select_strategy(self):
        """根据训练阶段动态选择策略"""
        if self.epoch < 20 or self.model_reliability < 0.7:
            return 'rule_priority'
        elif self.epoch < 60 or self.model_reliability < 0.85:
            return 'weighted_fusion'
        else:
            return 'model_priority'
    
    def update_reliability(self, val_auc, val_f1, rule_consistency):
        """评估模型可靠度"""
        self.model_reliability = (
            0.4 * val_auc + 
            0.3 * val_f1 + 
            0.3 * rule_consistency
        )
```

## 1.7 优势与局限（基于系统性重构后）

> **更新（2025-12-03）**：基于系统性重构后的最新设计

### 优势
1. ✅ **物理正确的规则公式**：分段函数设计，区分"过线"、"接近"、"远离"三种情况
2. ✅ **多尺度注意力**：从局部→全局→规则聚焦，层次清晰
3. ✅ **梯度流稳定**：多路径融合+残差连接+参数共享，确保各阶段均被训练
4. ✅ **工程友好**：基于成熟的GAT架构，GPU显存需求仅~520MB
5. ✅ **可解释性**：双层注意力监督（$\alpha_{ij}$+$\beta_i$），注意力权重可直接可视化
6. ✅ **训练流程清晰**：三阶段训练（冷启动→混合→自训练），逻辑自洽
7. ✅ **自训练安全**：双路径策略+阶段切换条件，防止模型漂移
8. ✅ **Memory增强（可选）**：可在Week 2启用，预期AUC提升+2-3%

### 局限
1. ❌ **规则硬编码**：新规则需要修改损失函数（但已提供扩展接口rule_id）
2. ⚠️ **阈值敏感**：$\tau_d$、$\tau_v$需要针对不同场景调整（但已提供网格搜索计划）
3. ⚠️ **长尾场景泛化弱**：依赖规则定义的完备性（但自训练Stage 3可部分缓解）
4. ❌ **分段函数的不连续性**：$f_{\text{dv}}(d,v)$在$d=0$和$d=\tau_d$处一阶导数不连续（工程上可接受，理论上可用smooth函数改进）

### 改进建议（ITER-02）
1. 🔧 引入可学习的规则参数（$\tau_d$, $\tau_v$）：从固定阈值改为可微分参数
2. 🔧 使用平滑分段函数（如soft-plus代替ReLU）：消除导数不连续
3. 🔧 规则库扩展：支持多规则联合检测（红灯停+车速+车道）
4. 🔧 Memory Bank默认启用：在数据量增加后开启

---

# 方案2：记忆增强对比学习 + 软规则引导

## 2.1 核心思想
构建**正常驾驶行为记忆库**（Memory Bank），通过对比学习使模型学习正常模式的原型表征。异常检测通过计算样本与记忆库的**马氏距离**实现。规则作为**软引导信号**而非硬约束，增强模型的自适应能力。

## 2.2 数学模型

### 2.2.1 记忆库设计
维护可学习的记忆矩阵 $\mathbf{M} \in \mathbb{R}^{K \times d_m}$，其中 $K$ 为记忆槽数量，$d_m$ 为记忆维度。

**记忆初始化**：通过K-Means聚类正常样本的编码：
$$
\mathbf{M}^{(0)} = \text{KMeans}\left(\{\mathbf{h}_i^{\text{normal}}\}_{i=1}^{N_{\text{init}}}, K\right)
$$

**记忆检索**：给定场景编码 $\mathbf{h}_i$，计算注意力权重：
$$
\begin{aligned}
w_{ik} &= \frac{\exp(\mathbf{h}_i^\top \mathbf{m}_k / \tau)}{\sum_{k'=1}^K \exp(\mathbf{h}_i^\top \mathbf{m}_{k'} / \tau)} \\
\tilde{\mathbf{h}}_i &= \sum_{k=1}^K w_{ik} \mathbf{m}_k \quad \text{(检索到的记忆表征)}
\end{aligned}
$$
其中 $\tau = 0.07$ 为温度系数。

### 2.2.2 对比学习框架

#### 正负样本构造
- **正样本**：同场景的不同增强视图（裁剪、遮挡、噪声）
- **负样本**：batch内其他场景 + 历史队列样本

对比损失（InfoNCE）：
$$
\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+) / \tau)}{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+) / \tau) + \sum_{j \in \text{neg}} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j^-) / \tau)}
$$

#### 记忆对比损失
强制正常样本与记忆库接近，异常样本远离：
$$
\mathcal{L}_{\text{mem}} = \begin{cases}
\|\mathbf{h}_i - \tilde{\mathbf{h}}_i\|_2^2, & \text{if } y_i = 0 \text{ (normal)} \\
\max(0, m - \|\mathbf{h}_i - \tilde{\mathbf{h}}_i\|_2)^2, & \text{if } y_i = 1 \text{ (anomaly)}
\end{cases}
$$
其中 $m = 2.0$ 为margin。

### 2.2.3 软规则引导

规则不直接约束损失，而是作为**伪标签生成器**：
$$
\tilde{y}_i = \begin{cases}
\text{normal}, & \text{if } s_i^{\text{rule}} < 0.3 \\
\text{uncertain}, & \text{if } 0.3 \le s_i^{\text{rule}} \le 0.7 \\
\text{anomaly}, & \text{if } s_i^{\text{rule}} > 0.7
\end{cases}
$$

对于不确定样本，使用半监督损失：
$$
\mathcal{L}_{\text{semi}} = -\sum_{i: \tilde{y}_i \neq \text{uncertain}} \left[\tilde{y}_i \log p_i + (1-\tilde{y}_i) \log(1-p_i)\right]
$$

### 2.2.4 异常评分

马氏距离异常分数：
$$
s_i^{\text{anomaly}} = \sqrt{(\mathbf{h}_i - \tilde{\mathbf{h}}_i)^\top \mathbf{\Sigma}^{-1} (\mathbf{h}_i - \tilde{\mathbf{h}}_i)}
$$
其中 $\mathbf{\Sigma}$ 为记忆库的协方差矩阵（在线估计）。

## 2.3 训练算法

```python
Algorithm: Memory-Augmented Contrastive Learning

Input:
  - Dataset D (unlabeled scenes)
  - Memory size K, embedding dim d_m
  - Contrastive temperature τ
  
Output:
  - Encoder f_θ, Memory bank M

1: # Initialize
2: encoder ← GATEncoder(hidden_dim=d_m)
3: memory_bank ← initialize_memory(K, d_m)  # K-Means on normal samples
4: queue ← FIFO(max_size=4096)  # Negative sample queue
5: optimizer ← AdamW([encoder.params, memory_bank], lr=1e-4)
6:
7: for epoch = 1 to E do
8:     for batch (X_b, A_b, entities_b) in DataLoader(D):
9:         # Data augmentation: create two views
10:        (X1, A1), (X2, A2) ← augment(X_b, A_b)
11:        
12:        # Encode both views
13:        H1 ← encoder(X1, A1)  # [B, d_m]
14:        H2 ← encoder(X2, A2)
15:        
16:        # Memory retrieval
17:        W ← softmax(H1 @ memory_bank.T / τ)  # [B, K]
18:        H_mem ← W @ memory_bank  # [B, d_m]
19:        
20:        # Compute rule pseudo-labels
21:        rule_scores ← compute_rule_scores(entities_b)
22:        pseudo_labels ← discretize_rule_scores(rule_scores)
23:        
24:        # Contrastive loss (InfoNCE)
25:        logits_pos ← similarity(H1, H2) / τ
26:        logits_neg ← similarity(H1, queue) / τ
27:        L_contrast ← -log(exp(logits_pos) / (exp(logits_pos) + sum(exp(logits_neg))))
28:        
29:        # Memory contrastive loss
30:        L_mem ← 0
31:        for i in range(B):
32:            if pseudo_labels[i] == 'normal':
33:                L_mem += ||H1[i] - H_mem[i]||^2
34:            elif pseudo_labels[i] == 'anomaly':
35:                L_mem += max(0, margin - ||H1[i] - H_mem[i]||)^2
36:        L_mem ← L_mem / B
37:        
38:        # Semi-supervised loss (only for confident pseudo-labels)
39:        confident_mask ← (pseudo_labels != 'uncertain')
40:        if sum(confident_mask) > 0:
41:            distances ← compute_mahalanobis(H1[confident_mask], H_mem[confident_mask])
42:            probs ← sigmoid(distances)
43:            L_semi ← binary_cross_entropy(probs, pseudo_labels[confident_mask])
44:        else:
45:            L_semi ← 0
46:        
47:        # Total loss
48:        L_total ← L_contrast + λ_mem * L_mem + λ_semi * L_semi
49:        
50:        # Backward & update
51:        optimizer.zero_grad()
52:        L_total.backward()
53:        clip_grad_norm_([encoder.params, memory_bank], max_norm=1.0)
54:        optimizer.step()
55:        
56:        # Update negative queue
57:        queue.enqueue(H2.detach())
58:        
59:        # EMA update for memory bank (optional)
60:        if epoch > warmup_epochs:
61:            with torch.no_grad():
62:                memory_bank ← 0.9 * memory_bank + 0.1 * update_memory(H1, pseudo_labels)
63:
64: return encoder, memory_bank
```

## 2.4 网络架构

```python
class MemoryAugmentedDetector(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 256,
        memory_size: int = 512,
        temperature: float = 0.07,
    ):
        super().__init__()
        
        # Encoder (GAT backbone)
        self.encoder = GATEncoder(input_dim, hidden_dim, num_layers=4)
        
        # Memory bank (learnable)
        self.memory_bank = nn.Parameter(torch.randn(memory_size, hidden_dim))
        nn.init.xavier_uniform_(self.memory_bank)
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128)
        )
        
        # Anomaly scoring head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        self.temperature = temperature
        self.register_buffer('cov_matrix', torch.eye(hidden_dim))
    
    def retrieve_memory(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """Memory retrieval with attention"""
        # h: [B, hidden_dim]
        sim = F.cosine_similarity(
            h.unsqueeze(1),              # [B, 1, hidden_dim]
            self.memory_bank.unsqueeze(0),  # [1, K, hidden_dim]
            dim=-1
        )  # [B, K]
        
        weights = F.softmax(sim / self.temperature, dim=-1)
        h_mem = torch.matmul(weights, self.memory_bank)  # [B, hidden_dim]
        
        return h_mem, weights
    
    def compute_anomaly_score(self, h: Tensor, h_mem: Tensor) -> Tensor:
        """Mahalanobis distance-based scoring"""
        diff = h - h_mem  # [B, hidden_dim]
        
        # Mahalanobis distance
        inv_cov = torch.inverse(self.cov_matrix + 1e-6 * torch.eye(h.size(1), device=h.device))
        mahal_dist = torch.sqrt(torch.sum(diff @ inv_cov * diff, dim=-1))
        
        return torch.sigmoid(mahal_dist)
    
    def forward(self, x: Tensor, edge_index: Tensor, return_embeddings: bool = False):
        # Encode scene
        h = self.encoder(x, edge_index)  # [N, hidden_dim]
        
        # Pool to scene-level (mean pooling for simplicity)
        h_scene = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        
        # Retrieve memory
        h_mem, mem_weights = self.retrieve_memory(h_scene)
        
        # Anomaly scoring
        score = self.compute_anomaly_score(h_scene, h_mem)
        
        if return_embeddings:
            return score, h_scene, h_mem, mem_weights
        return score
    
    def update_covariance(self, embeddings: Tensor):
        """Online covariance estimation"""
        with torch.no_grad():
            cov = torch.cov(embeddings.T)
            self.cov_matrix = 0.9 * self.cov_matrix + 0.1 * cov
```

## 2.5 优势与局限

### 优势
1. ✅ **自适应性强**：记忆库可随数据分布演化
2. ✅ **规则解耦**：不依赖显式规则，适合复杂场景
3. ✅ **少样本学习**：对比学习在小数据集上表现优于监督学习
4. ✅ **可解释性**：记忆检索权重提供决策依据

### 局限
1. ❌ **训练复杂**：对比学习需要精心设计数据增强
2. ❌ **计算开销大**：记忆检索 + 协方差估计增加推理时间
3. ❌ **冷启动问题**：记忆库初始化需要足够的正常样本

---

# 方案3：因果图网络 + 反事实推理（Causal GNN + Counterfactual Explanation）

## 3.1 核心思想
将交通场景建模为**因果图**，显式建模实体间的因果关系（如"红灯 → 车辆应停止"）。通过**结构因果模型（SCM）**学习因果机制，异常检测等价于因果违背检测。可解释性通过**反事实推理**实现："如果灯是绿色，车辆会通过吗？"

## 3.2 数学模型

### 3.2.1 结构因果模型（SCM）

定义因果变量：
- $L \in \{\text{red, yellow, green}\}$：交通灯状态
- $D \in \mathbb{R}_+$：车辆到停止线距离
- $V \in \mathbb{R}_+$：车辆速度
- $A \in \{0,1\}$：车辆行为（0=停止，1=通过）

因果图结构：
$$
L \rightarrow A \leftarrow D \leftarrow V
$$

结构方程：
$$
\begin{aligned}
L &\sim \text{Categorical}([0.5, 0.1, 0.4]) \quad \text{(外生变量)} \\
V &\sim \mathcal{N}(\mu_v, \sigma_v^2) \\
D &= f_D(V, U_D), \quad U_D \sim \mathcal{N}(0, \sigma_D^2) \\
A &= f_A(L, D, U_A), \quad U_A \sim \mathcal{N}(0, \sigma_A^2)
\end{aligned}
$$

其中 $f_A$ 是可学习的因果机制（神经网络）：
$$
f_A(L, D) = \sigma\left(\mathbf{w}_L^\top \text{onehot}(L) + \mathbf{w}_D \cdot D + b\right)
$$

### 3.2.2 因果图神经网络

#### 因果邻接矩阵
不同于传统GNN的对称邻接矩阵，因果图使用**有向邻接矩阵** $\mathbf{A}_{\text{causal}}$，元素 $A_{ij}=1$ 当且仅当 $v_i$ 是 $v_j$ 的因果父节点。

#### 因果消息传递
$$
\begin{aligned}
\mathbf{m}_{i \rightarrow j} &= \phi_{\text{cause}}\left(\mathbf{h}_i, \mathbf{h}_j, \mathbf{e}_{ij}\right) \quad \text{if } A_{\text{causal}, ij} = 1 \\
\mathbf{h}_j^{(l+1)} &= \psi_{\text{effect}}\left(\mathbf{h}_j^{(l)}, \sum_{i \in \text{Parents}(j)} \mathbf{m}_{i \rightarrow j}\right)
\end{aligned}
$$

其中 $\phi_{\text{cause}}$ 和 $\psi_{\text{effect}}$ 是可学习的神经网络。

### 3.2.3 反事实推理

给定观测 $(L=\text{red}, D=3m, V=5m/s, A=1)$（闯红灯），计算反事实：

**干预（Intervention）**：强制 $\text{do}(L=\text{green})$，重新计算：
$$
A_{\text{cf}} = f_A(\text{green}, D, U_A) = \sigma(\mathbf{w}_L^\top [0,0,1] + \mathbf{w}_D \cdot 3 + b)
$$

**反事实解释**：
$$
\text{Explanation} = \begin{cases}
\text{"因果违背：红灯应停止"}, & \text{if } A_{\text{cf}} = 0 \land A_{\text{obs}} = 1 \\
\text{"正常行为"}, & \text{otherwise}
\end{cases}
$$

### 3.2.4 异常检测损失

$$
\begin{aligned}
\mathcal{L}_{\text{causal}} &= \mathcal{L}_{\text{NLL}} + \lambda_1 \mathcal{L}_{\text{DAG}} + \lambda_2 \mathcal{L}_{\text{CF}} \\
\\
\mathcal{L}_{\text{NLL}} &= -\sum_{i=1}^N \log p(A_i | \text{Parents}(A_i)) \\
\\
\mathcal{L}_{\text{DAG}} &= \text{trace}(\mathbf{e}^{\mathbf{A} \circ \mathbf{A}}) - d \quad \text{(无环约束)} \\
\\
\mathcal{L}_{\text{CF}} &= \sum_{i=1}^N \left\|A_i^{\text{obs}} - A_i^{\text{cf}}(\text{do}(L_i=\text{normal}))\right\|^2
\end{aligned}
$$

其中 $\mathcal{L}_{\text{DAG}}$ 保证因果图无环（Zheng et al., 2018）。

## 3.3 训练算法

```python
Algorithm: Causal GNN with Counterfactual Learning

Input:
  - Dataset D = {(X_i, A_i, entities_i)}
  - Causal graph structure prior G_prior
  
Output:
  - Causal model θ_causal
  
1: # Initialize causal adjacency matrix
2: A_causal ← initialize_from_prior(G_prior)  # e.g., L→A, D→A
3: A_causal ← make_learnable(A_causal)
4:
5: # Initialize causal mechanisms
6: f_A ← CausalMLP(input_dim=num_parents)
7: optimizer ← AdamW([A_causal, f_A.params], lr=1e-3)
8:
9: for epoch = 1 to E do
10:    for batch (X, entities, actions) in DataLoader(D):
11:        # Extract causal variables
12:        L ← extract_light_state(entities)  # [B]
13:        D ← extract_distance(entities)     # [B]
14:        V ← extract_velocity(entities)     # [B]
15:        A_obs ← actions                    # [B]
16:        
17:        # Forward: predict action from causal parents
18:        parents_feat ← concat([onehot(L), D, V])  # [B, d_parents]
19:        A_pred ← f_A(parents_feat)  # [B]
20:        
21:        # Loss 1: Negative log-likelihood
22:        L_NLL ← binary_cross_entropy(A_pred, A_obs)
23:        
24:        # Loss 2: DAG constraint (acyclicity)
25:        A_squared ← A_causal @ A_causal
26:        L_DAG ← trace(exp(A_squared)) - d
27:        
28:        # Loss 3: Counterfactual consistency
29:        L_CF ← 0
30:        for i in range(B):
31:            if L[i] == 'red' and A_obs[i] == 1:  # Violation observed
32:                # Intervene: do(L='green')
33:                L_cf ← 'green'
34:                parents_cf ← concat([onehot(L_cf), D[i], V[i]])
35:                A_cf ← f_A(parents_cf)
36:                
37:                # Counterfactual should predict "pass"
38:                L_CF += (A_cf - 1)^2
39:        L_CF ← L_CF / B
40:        
41:        # Total loss
42:        L_total ← L_NLL + λ_1 * L_DAG + λ_2 * L_CF
43:        
44:        # Backward
45:        optimizer.zero_grad()
46:        L_total.backward()
47:        
48:        # Project A_causal to valid DAG space
49:        with torch.no_grad():
50:            A_causal ← threshold(A_causal, min=0, max=1)
51:            A_causal ← A_causal * (1 - eye(d))  # Remove self-loops
52:        
53:        optimizer.step()
54:    
55:    # Validate causal graph
56:    if epoch % 10 == 0:
57:        is_dag ← check_acyclic(A_causal)
58:        if not is_dag:
59:            warn("Causal graph not DAG, applying projection")
60:            A_causal ← project_to_dag(A_causal)
61:
62: return f_A, A_causal
```

## 3.4 反事实解释生成

```python
Algorithm: Counterfactual Explanation Generation

Input:
  - Observed scene: (L_obs, D_obs, V_obs, A_obs)
  - Causal model: f_A, A_causal
  
Output:
  - Explanation with counterfactual scenarios

1: # Check if violation occurred
2: if not is_violation(L_obs, D_obs, V_obs, A_obs):
3:     return "Normal behavior, no explanation needed"
4:
5: explanations ← []
6:
7: # Counterfactual 1: What if light was green?
8: if L_obs == 'red':
9:     A_cf1 ← f_A(onehot('green'), D_obs, V_obs)
10:    explanation_1 ← {
11:        'type': 'intervention_light',
12:        'intervention': 'do(Light=green)',
13:        'predicted_action': 'pass' if A_cf1 > 0.5 else 'stop',
14:        'consistency': 'violated' if A_cf1 > 0.5 else 'maintained',
15:        'message': f"If light was green, vehicle would {'pass' if A_cf1 > 0.5 else 'stop'}"
16:    }
17:    explanations.append(explanation_1)
18:
19: # Counterfactual 2: What if distance was larger?
20: D_cf2 ← D_obs + 20  # Add 20m
21: A_cf2 ← f_A(onehot(L_obs), D_cf2, V_obs)
22: explanation_2 ← {
23:     'type': 'intervention_distance',
24:     'intervention': f'do(Distance={D_cf2}m)',
25:     'predicted_action': 'pass' if A_cf2 > 0.5 else 'stop',
26:     'message': f"If vehicle was {D_cf2}m away, it would {'pass' if A_cf2 > 0.5 else 'stop'}"
27: }
28: explanations.append(explanation_2)
29:
30: # Counterfactual 3: What if velocity was zero?
31: A_cf3 ← f_A(onehot(L_obs), D_obs, V_cf=0)
32: explanation_3 ← {
33:     'type': 'intervention_velocity',
34:     'intervention': 'do(Velocity=0)',
35:     'predicted_action': 'stop',
36:     'message': "Stopping would comply with red light rule"
37: }
38: explanations.append(explanation_3)
39:
40: # Generate causal attribution
41: attribution ← compute_shapley_values(f_A, [L_obs, D_obs, V_obs])
42: explanations.append({
43:     'type': 'attribution',
44:     'light_contribution': attribution[0],
45:     'distance_contribution': attribution[1],
46:     'velocity_contribution': attribution[2],
47:     'message': f"Primary cause: {argmax(attribution)}"
48: })
49:
50: return format_explanation(explanations)
```

## 3.5 网络架构

```python
class CausalGNN(nn.Module):
    def __init__(
        self,
        num_vars: int = 4,  # L, D, V, A
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        
        # Learnable causal adjacency matrix
        self.causal_adj = nn.Parameter(torch.zeros(num_vars, num_vars))
        
        # Initialize with prior knowledge
        # L(0) → A(3), D(1) → A(3), V(2) → D(1), V(2) → A(3)
        with torch.no_grad():
            self.causal_adj[0, 3] = 1.0  # L → A
            self.causal_adj[1, 3] = 1.0  # D → A
            self.causal_adj[2, 1] = 1.0  # V → D
            self.causal_adj[2, 3] = 1.0  # V → A
        
        # Causal mechanisms (one per variable)
        self.mechanisms = nn.ModuleDict({
            'distance': nn.Sequential(  # D = f(V)
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ),
            'action': nn.Sequential(  # A = f(L, D, V)
                nn.Linear(3 + 1 + 1, hidden_dim),  # 3 for L (one-hot), 1 for D, 1 for V
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        })
    
    def forward(self, light, distance, velocity):
        """
        Args:
            light: [B, 3] one-hot encoded
            distance: [B, 1]
            velocity: [B, 1]
        Returns:
            action_prob: [B, 1]
        """
        # Predict action from causal parents
        parents = torch.cat([light, distance, velocity], dim=-1)
        action_prob = self.mechanisms['action'](parents)
        return action_prob
    
    def intervene(self, light, distance, velocity, intervention):
        """
        Perform do-calculus intervention
        
        Args:
            intervention: dict, e.g., {'light': tensor([0, 0, 1])} for do(L=green)
        """
        if 'light' in intervention:
            light = intervention['light']
        if 'distance' in intervention:
            distance = intervention['distance']
        if 'velocity' in intervention:
            velocity = intervention['velocity']
        
        return self.forward(light, distance, velocity)
    
    def dag_penalty(self):
        """Compute DAG constraint: h(A) = tr(e^(A◦A)) - d"""
        adj_squared = torch.matmul(self.causal_adj, self.causal_adj)
        return torch.trace(torch.matrix_exp(adj_squared)) - self.causal_adj.size(0)
    
    def get_causal_graph(self):
        """Extract binary causal graph"""
        with torch.no_grad():
            return (torch.sigmoid(self.causal_adj) > 0.5).float()
```

## 3.6 优势与局限

### 优势
1. ✅ **最强可解释性**：反事实推理提供"why"和"what-if"答案
2. ✅ **因果泛化**：学到的因果机制可迁移到新场景
3. ✅ **规则自动发现**：无需手工编写规则，从数据中学习因果关系
4. ✅ **学术价值高**：因果推理是AI可解释性的前沿方向

### 局限
1. ❌ **极高复杂度**：因果发现、DAG约束、反事实计算均为NP难问题
2. ❌ **数据需求大**：需要丰富的干预数据（或强先验）才能学到正确因果图
3. ❌ **训练不稳定**：DAG约束难以优化，易陷入局部最优
4. ❌ **工程化困难**：现有因果推理库（如DoWhy）与深度学习框架集成不佳

---

# 综合评估与建议

## 对比分析

| 评估维度 | 方案1（GAT+硬约束） | 方案2（记忆对比） | 方案3（因果推理） |
|---------|---------|---------|---------|
| **MVP交付速度** | ⭐⭐⭐⭐⭐ 1-2周 | ⭐⭐⭐ 3-4周 | ⭐⭐ 6-8周 |
| **代码复杂度** | ~1200 LOC | ~2000 LOC | ~3500 LOC |
| **论文价值** | ⭐⭐ 工程实现 | ⭐⭐⭐⭐ 创新方法 | ⭐⭐⭐⭐⭐ 顶会水平 |
| **可维护性** | ⭐⭐⭐⭐ 模块清晰 | ⭐⭐⭐ 依赖对比学习框架 | ⭐⭐ 因果图维护成本高 |
| **扩展到多规则** | ⭐⭐⭐ 需逐个编写损失 | ⭐⭐⭐⭐ 记忆库自适应 | ⭐⭐⭐⭐⭐ 自动因果发现 |
| **实际部署难度** | ⭐⭐ 易于部署 | ⭐⭐⭐ 需维护记忆库 | ⭐⭐⭐⭐ 推理开销大 |

## 推荐方案

### ✅ **立即执行：方案1（多阶段注意力GAT + 硬约束）**
**理由**：
1. 满足MVP时间要求（12-15交付）
2. 工程风险最低，基于成熟的GAT架构
3. 可解释性达标（注意力权重可视化）
4. 规则约束保证符合交通法规

**实施建议**：
- 优先实现3层GAT + 规则损失
- 第2阶段（全局注意力）和第3阶段（规则聚焦）可渐进式开发
- 使用PyTorch Geometric加速实现

### 📋 **ITER-02规划：方案2（记忆增强对比学习）**
**理由**：
1. 数据量增加后性能更优
2. 减少对手工规则的依赖
3. 可作为方案1的升级路径（保留GAT backbone，增加记忆模块）

**实施建议**：
- 在方案1基础上增加记忆库模块
- 使用MoCo v3框架简化对比学习实现
- 记忆库与规则约束并行，形成混合方案

### 🔬 **论文方向：方案3（因果图网络）**
**理由**：
1. 学术创新性最高，适合发表
2. 可作为长期研究方向
3. 与方案1/2不冲突，可并行探索

**实施建议**：
- 作为研究型任务，不纳入MVP交付
- 可在ITER-03或后续迭代中实验
- 建议先阅读因果发现文献（Zheng et al., 2018; Ke et al., 2019）

---

## 下一步行动

1. **立即决策**：选定方案1作为MVP实现路径
2. **更新设计文档**：将方案1的详细设计合并到 `Design-ITER-2025-01.md`
3. **启动开发**：按照算法伪代码实现 `MultiStageAttentionGAT` 类
4. **准备数据**：执行 `scripts/prepare_synthetic_data.py` 生成训练数据
5. **建立基线**：先实现简化版（单层GAT + 规则损失），验证流程

---

## 参考文献

1. Veličković et al. (2018). "Graph Attention Networks." ICLR.
2. Chen et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML.
3. Zheng et al. (2018). "DAGs with NO TEARS: Continuous Optimization for Structure Learning." NeurIPS.
4. Schölkopf et al. (2021). "Toward Causal Representation Learning." Proceedings of the IEEE.
5. Gong et al. (2022). "Memory-augmented Graph Neural Networks." AAAI.

---

## Checklist

- [x] 提供3种完整算法方案
- [x] 包含数学公式与推导
- [x] 提供伪代码与网络架构
- [x] 给出优劣势对比
- [x] 明确推荐方案与实施路径
- [ ] 评审人签字（待评审）


