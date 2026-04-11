# OmniRetarget Interaction Mesh 算法细节记录

## 1. 核心计算流程

### 1.1 Delaunay → L 矩阵 → Laplacian 坐标 → 优化

```
人手坐标 (米)
    │
    × scale (缩放到 robot 尺度, 例如 α = h_robot / h_human)
    │
    ▼
缩放后的人手关键点 + 物体/地面采样点
    │
    Delaunay 四面体化 → 邻接表 adj_list (谁是谁的邻居)
    │
    构建 L 矩阵 (N×N, 纯拓扑, 无物理单位)
    │
    ├── L @ scaled_human_pts → L_source (优化目标, robot 尺度的偏移向量)
    │
    └── L @ robot_pts → L_robot (当前值, 随关节角 q 变化)
                │
                ▼
        SQP 优化: 调 q 使 L_robot → L_source
                │
                ▼
            输出: 每帧的 robot 关节角
```

### 1.2 L 矩阵详解

L 矩阵只编码拓扑关系，和实际位置/距离无关 (uniform weight 模式下):

```
L[i, i] = 1.0
L[i, j] = -1 / degree(i)    如果 j 是 i 的邻居
L[i, j] = 0                 不是邻居
```

例如点 0 有 2 个邻居 (点 1 和点 2):
```
L[0] = [1.0, -0.5, -0.5, 0, ...]
```

-0.5 代表"2 个邻居各占 50% 权重"，和距离无关。

**距离信息出现在 L @ pts 之后**:
```
result[0] = 1.0 * pts[0] + (-0.5) * pts[1] + (-0.5) * pts[2]
          = pts[0] - mean(pts[1], pts[2])
          = 点 0 相对于邻居质心的偏移向量 (有物理单位, 米)
```

### 1.3 缩放的必要性

`L_source = L @ human_pts` 的结果包含人的绝对尺度。如果人手 15cm, robot 手 10cm, 直接要求 `L_robot ≈ L_source` 会强制 robot 的偏移量等于人的，物理上不合理。

OmniRetarget 的处理: **在计算 L_source 之前，先把人缩放到机器人尺度**:
```python
scale = robot_height / human_height  # 例如 0.68
human_joint_motions *= scale_factor  # 整个轨迹等比缩放
# 之后才传入 retarget_motion()
```

对于 WujiHand: 人手和机器手尺寸接近, `global_scale ≈ 1.0`。

---

## 2. 各 Mode 的配置差异

### 2.1 所有 mode 共享的硬编码参数

| 参数 | 值 | 位置 | 说明 |
|------|---|------|------|
| `laplacian_weights` | 10 | retargeter L105 | 所有顶点统一权重 |
| `smooth_weight` | 0.2 | retargeter L106 | 时间平滑正则化 |
| `uniform_weight` | True | 所有调用 | 均匀邻居权重 (从不使用距离权重) |
| `step_size` | 0.2 | RetargeterConfig | 信赖域半径 |
| `penetration_tolerance` | 0.001 | RetargeterConfig | 碰撞容差 |

**这些参数不可通过配置文件修改，不因 task type 而变。**

### 2.2 Mode 特定的差异

| | robot_only | object_interaction (box) | climbing |
|---|---|---|---|
| 人体关键点 | 15 (SMPLH/LAFAN) | 15 (SMPLH) | 15 (MOCAP) |
| 物体采样点 | 0 | 100 (均匀表面) | 100 (z>0.9 加权 20x) |
| 地面网格点 | 225 (15×15, [-1,1]) | 0 | 64 (8×8, [-2,2], 仅 augmentation) |
| 总 mesh 点数 | 240 | 115 | 179 (含地面) / 115 (不含) |
| 碰撞约束 | ground only | ground + object | ground + multi_boxes |
| 脚约束 | foot sticking | foot sticking | foot sticking |
| nominal tracking | w=0 (原始) | w=0 (原始) | w=5, tau=10 (衰减) |
| Q_diag | task-specific | task-specific | task-specific |
| 坐标系 | world frame | object local frame | object local frame |

### 2.3 Laplacian 权重方案

**论文明确声明**: "For all our experiments, we use uniform weights, setting w_ij = 1/|N(i)|" (论文公式 1 下方)。

代码中存在距离权重的实现 (`uniform_weight=False`):
```python
# utils.py L451-454 — calculate_laplacian_coordinates
weights = 1.0 / (1.5 * distances + epsilon)   # 未归一化

# utils.py L488-491 — calculate_laplacian_matrix  
weights = 1.0 / (distances + epsilon)          # 归一化
weights = weights / np.sum(weights)
```

两个函数的距离衰减公式不同 (1.5d vs d)，暗示不同阶段的尝试。
但 **所有调用始终传入 `uniform_weight=True` (默认值)**，距离权重代码是开发遗留的死代码，
论文未讨论距离权重作为备选方案。

### 2.4 可配置参数 (RetargeterConfig)

```python
@dataclass(frozen=True)
class RetargeterConfig:
    q_a_init_idx: int = -7           # 优化变量起始索引 (-7=含浮动基座, 0=仅关节)
    activate_joint_limits: bool = True
    activate_obj_non_penetration: bool = True
    activate_foot_sticking: bool = True
    penetration_tolerance: float = 0.001
    foot_sticking_tolerance: float = 1e-3
    step_size: float = 0.2           # 信赖域
    w_nominal_tracking_init: float = 5.0
    nominal_tracking_tau: float = 1e6  # climbing 覆盖为 10
```

**注意: `laplacian_weights`, `smooth_weight`, 权重方案均不在此配置中。**

---

## 3. Delaunay 三角化/四面体化

### 3.1 库来源

`scipy.spatial.Delaunay` → 底层是 **Qhull** 库 (C 语言, BSD 开源):
- 源码: http://www.qhull.org/
- scipy 封装: https://github.com/scipy/scipy/blob/main/scipy/spatial/_qhull.pyx

### 3.2 算法原理

**核心规则**: 任何四面体的外接球内部不能包含其他点。

**3D 增量算法 (Qhull 实际使用的)**:

```
不是从大球缩小，也不是从小球扩大。
是通过 "凸包提升" (convex hull lifting) 实现的:

1. 把 3D 点 (x, y, z) 提升到 4D 抛物面: (x, y, z, x²+y²+z²)
2. 在 4D 空间中计算凸包 (Qhull 的核心能力)
3. 凸包的下表面投影回 3D → 就是 Delaunay 四面体化

数学证明: 4D 凸包下表面的每个面对应一个 3D 四面体,
且该四面体自动满足外接球空条件。
```

**等价的直觉理解 (逐点插入法)**:
```
1. 初始化: 构建一个包含所有点的超大四面体
2. 逐个插入点 p:
   a. 找到所有外接球包含 p 的四面体 → 标记为"坏的"
   b. 删除这些坏四面体 → 形成一个多面体空洞
   c. 从 p 向空洞的每个面连线 → 形成新四面体
   d. 这些新四面体自动满足 Delaunay 条件
3. 最后删除和超大四面体相关的部分
```

**不是从某个点出发找最近三个点**——那是 K 近邻，不是 Delaunay。Delaunay 是全局最优的三角化，不是贪心的。

### 3.3 可以替换

Delaunay 只是建拓扑的一种方式。整个优化引擎只需要 `adj_list`，不关心它从哪来:

```python
# 方案 1: K 近邻
from scipy.spatial import KDTree
tree = KDTree(points)
adj = [tree.query(p, k=6)[1][1:].tolist() for p in points]

# 方案 2: 手动骨架拓扑
adj = {
    "palm": ["thumb_mcp", "index_mcp", ...],
    "index_mcp": ["palm", "index_pip"],
    ...
}

# 方案 3: 距离阈值
from scipy.spatial.distance import cdist
D = cdist(points, points)
adj = [np.where((D[i] > 0) & (D[i] < threshold))[0].tolist() for i in range(N)]

# 方案 4: Delaunay + 剪枝 (删除跨结构的不合理连接)
```

Delaunay 的优势: 自动、几何合理 (近者为邻)、和物体/地面点混合时自然建立接触拓扑。
Delaunay 的劣势: 稀疏点集会产生跨结构连接 (如手指间)。

---

## 4. 优化目标的完整形式

每次 SQP 迭代求解的 SOCP:

```
minimize:
    laplacian_weight * ||lap_var - L_source||²     # Laplacian 变形能量 (主)
  + smooth_weight * ||dq - dq_smooth||²            # 时间平滑
  + ||sqrt(Q_diag) ⊙ (dq + q_current)||²          # 关节正则化
  + w_nominal * ||dq[indices] - dq_nominal||²      # 名义轨迹跟踪 (仅 augmentation)

subject to:
    J_L @ dq - lap_var == -L_current               # Laplacian 线性化
    J_collision @ dq >= -phi - tol                  # 碰撞回避
    q_lb - q_current <= dq <= q_ub - q_current      # 关节极限
    J_foot_xy @ dq ∈ [p_prev - tol, p_prev + tol]  # 脚不滑 (XY)
    ||dq||₂ <= step_size                            # 信赖域 (SOC)
```

所有权重 (laplacian=10, smooth=0.2, step_size=0.2) 跨 mode 不变。
差异仅在于: 哪些约束被激活、mesh 包含哪些点、Q_diag 的值。

---

## 5. 采样密度即隐式权重 — Interaction Mesh 的核心机制

### 5.1 Robot-only 模式下的退化

robot_only 模式 (无物体) 下, interaction mesh 退化为关键点直接对齐:

```
21 个 source 关键点 ←→ 21 个 robot 关键点 (一一对应)
L_source = L @ source_pts
L_robot  = L @ robot_pts
目标: min ||L_robot - L_source||²
当 robot_pts = source_pts 时, loss = 0 → 最优解就是完美对齐
```

分析: 20 DOF 被 63 个 Laplacian 分量 (21×3) 约束, 过约束但所有约束指向同一个解。
**robot_only 下 interaction mesh 没有优势**, 和直接 IK 效果相同。

### 5.2 物体点如何打破对齐退化

加入物体点后, L 矩阵本身发生变化 — 手指关键点的邻居不再只是其他手指:

```
无物体: L[手指A] = 手指A - mean(手指B, 手指C, 手指D)  ← 纯手指间关系

有物体: L[手指A] = 手指A - mean(手指B, 杯子1, 杯子2, ..., 杯子8)
                              1个手指    8个杯子点 (Jacobian=0, 固定锚点)
```

物体点不动 (Jacobian=0), 但参与 Laplacian 计算:
- L_source[手指A] 编码了"人手指相对于杯子的偏移"
- 优化要求 L_robot[手指A] 也保持同样的偏移
- 杯子点像钉子钉在空间中, robot 手指必须围绕这些钉子调整姿态

**不是一一对齐了** — 当骨骼比例不同时, "保持手指间关系" 和 "保持手指-物体关系" 会矛盾,
优化在两者之间折中。这个折中解才是 interaction mesh 的价值。

### 5.3 采样密度 = 优先级

论文用 100 个物体表面采样点, 不是为了精确描述物体形状, 而是为了**在 loss 中提高接触保持的隐式权重**:

```
手指A 的邻居: {手指B, 杯子点1, 杯子点2, ..., 杯子点8}
                1/9 权重     8/9 权重

L[手指A] = 手指A - (1/9 * 手指B + 8/9 * mean(杯子点们))
→ 主要编码"手指A 相对于杯子表面的偏移", 手指B 的影响被稀释

||L_robot[手指A] - L_source[手指A]||² 
→ 误差项主要惩罚"手指偏离杯子", 而非"手指偏离其他手指"
```

采样密度直接控制了优化的隐式优先级:
- 物体 100 点 vs 关节 15 点 → 接触保持 >> 骨架形状
- 如果减少到物体 10 点 → 接触保持 ≈ 骨架形状
- 论文不需要显式设置"接触权重", 采样密度已经隐式完成了

### 5.4 自由度 vs 约束分析

```
robot_only (21 点, 20 DOF):
  约束: 21×3 = 63 个 Laplacian 分量
  自由度: 20 个关节角
  过约束但约束一致 → 解几乎唯一 (对齐) → 无松动空间

有物体 (21+50 点, 20 DOF):
  约束: 71×3 = 213 个 Laplacian 分量
  自由度: 还是 20 个关节角
  过约束且约束可能矛盾 → 优化在矛盾中找折中 → 解不同于纯对齐
  物体点的矛盾正是接触保持的来源
```

**核心结论**: interaction mesh 的价值不在于 Delaunay 或 Laplacian 的数学形式,
而在于**通过物体采样密度隐式编码优先级, 用固定锚点打破纯对齐退化**。
robot_only 只是管线验证, 加入物体后才是真正的测试。
