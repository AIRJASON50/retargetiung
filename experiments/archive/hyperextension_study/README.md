# 实验记录: Per-Segment 骨段比缩放 + 自适应调参

日期: 2026-04-14
数据: `data/manus_for_pinch/manus1_5k.pkl` (5000 帧, 左手, Manus 手套)
Retargeter: InteractionMeshHandRetargeter (Pinocchio fixed-base, 20 DOF)

---

## 1. 动机

Interaction mesh retargeting 存在严重反弓 (hyperextension): 99.8% 的帧至少有一个弯曲关节 q < 0。假设根因: source (Manus) 骨段长度与 robot 骨段长度不同, 优化器用负关节角"拉伸"有效臂长来更好匹配 source 位置。

前置发现:
- MediaPipe 使用固定模板骨架 (所有用户骨段长度一致, CV=0%)
- Manus 手套骨段长度帧内完美恒定 (CV=0%), 但反映真实手部比例
- Wuji baseline 用手动标定的 per-user segment_scaling (5x3=15 参数) 处理比例差异

## 2. 方案

### Phase 1: 固定 per-segment 骨段比缩放

从 warmup 帧 (前 10 帧) 自动计算每段比例:
```
alpha[finger, segment] = robot 骨段长度 / source 骨段长度
```

每根手指 4 段: Wrist-MCP, MCP-PIP, PIP-DIP, DIP-TIP。
缩放在 source keypoints 上逐段累积应用, 在 Laplacian 计算之前完成。

### Phase 2: 自适应在线微调 (worktree 沙盒)

每 50 帧检查 PIP (J3) 和 DIP (J4) 关节的反弓情况。检测到反弓则将对应段的 alpha 拉向 1.0:
```
alpha_new = alpha_old + lambda * severity * (1.0 - alpha_old)
lambda=0.3, severity = mean(max(0, -q)), clamp 到 [0.85, 1.15]
```

---

## 3. 计算得到的骨段比 (Manus 数据)

| 手指 | Wrist-MCP | MCP-PIP | PIP-DIP | DIP-TIP |
|------|-----------|---------|---------|---------|
| 拇指 | 0.988 | 0.983 | **0.812** | **0.770** |
| 食指 | 0.947 | 1.066 | **1.178** | 1.093 |
| 中指 | 0.914 | 1.011 | 1.084 | 1.086 |
| 无名指 | 0.897 | 1.064 | 1.054 | 1.066 |
| 小指 | **0.831** | **1.499** | **1.771** | **1.333** |

关键观察: 小指比例极端 (1.3-1.8x)。Manus 手套报告的小指骨段远短于 robot 统一的 29.5mm PIP-DIP link。

MediaPipe 骨段长度稳定性验证:
- Manus 数据: 所有骨段 CV=0.000% (完美恒定, 10/50/100/500/5000 帧窗口结果完全一致)
- HO-Cap 数据: 非 TIP 段 CV=0.000%, DIP-TIP 段 CV=0.5-2.4%
- **结论: warmup=1 帧即可, 不需要多帧取中位数**

---

## 4. 结果

### 4.1 反弓帧率 (PIP + DIP, q < 0 占比)

| 手指 | 关节 | IM baseline | 固定缩放 | 自适应缩放 | Wuji BL |
|------|------|-------------|---------|-----------|---------|
| 拇指 | DIP | 3.6% | 15.0% | 6.3% | 12.1% |
| 食指 | PIP | 70.6% | 78.8% | **60.5%** | 57.9% |
| 食指 | DIP | 17.0% | 4.9% | **12.9%** | 6.9% |
| 中指 | PIP | 35.1% | 41.7% | **28.1%** | 41.6% |
| 无名指 | PIP | 29.6% | 33.7% | **27.1%** | 18.2% |
| 小指 | PIP | 48.4% | 30.6% | 44.4% | 21.3% |
| 小指 | DIP | 0.0% | 0.0% | 5.9% | 0.0% |

### 4.2 反弓严重度 (q<0 帧的平均反弓角度, 度)

| 手指 | 关节 | IM baseline | 自适应缩放 | Wuji BL |
|------|------|-------------|-----------|---------|
| 食指 | PIP | 15.1 | **11.8** | 4.4 |
| 食指 | DIP | 14.5 | **10.8** | 0.7 |
| 无名指 | PIP | 5.3 | **3.4** | 3.9 |
| 小指 | PIP | 21.4 | **19.8** | 3.8 |

### 4.3 指尖位置误差 (mm)

| 手指 | IM baseline | 自适应缩放 | Wuji BL |
|------|-------------|-----------|---------|
| 拇指 | 13.5+-1.3 | 16.0+-2.0 | 11.3+-1.5 |
| 食指 | 10.3+-5.2 | 10.9+-6.1 | 3.2+-3.1 |
| 中指 | 12.9+-4.9 | 12.5+-4.9 | 7.1+-3.0 |
| 无名指 | 9.5+-5.3 | 11.1+-5.2 | 4.5+-3.6 |
| 小指 | 11.8+-5.1 | 18.4+-5.5 | 18.6+-6.3 |
| **均值** | **11.6** | **13.8** | **8.9** |

---

## 5. 自适应 Alpha 收敛曲线

每 50 帧更新一次, 大部分 ratio 在约 2000 帧内收敛到 ~1.0:

| 骨段 | 初始值 | Frame 500 | Frame 1000 | Frame 2000 | 最终值 | 变化次数 |
|------|--------|-----------|-----------|-----------|--------|---------|
| 食指 s1 (MCP-PIP) | 1.066 | 1.016 | 1.016 | 1.016 | 1.000 | 28 |
| 食指 s2 (PIP-DIP) | 1.150 | 1.150 | 1.012 | 1.001 | 1.001 | 13 |
| 小指 s1 (MCP-PIP) | 1.150 | 1.105 | 1.003 | 1.000 | 1.000 | 31 |
| 无名指 s1 (MCP-PIP) | 1.064 | 1.031 | 1.008 | 1.000 | 1.000 | 24 |

**关键发现: 几乎所有 ratio 收敛到 1.0, 等于 bone scaling 被自适应算法自行取消。**

---

## 6. 根因分析

### 6.1 为什么固定骨段缩放效果不好

Per-segment 缩放是**局部操作** (每根手指独立缩放), 但 Laplacian cost 是**全局耦合的** (所有关键点通过 Delaunay mesh 邻接关系耦合)。缩放一根手指的骨段不仅改变了该点的 L_source, 还影响了所有 Delaunay 邻居 (可能来自其他手指) 的 cost, 产生跨手指冲突。优化器通过在其他关节反弓来妥协解决这些冲突。

### 6.2 为什么自适应收敛到 1.0

自适应算法正确检测到骨段缩放**正在引起反弓** (通过跨手指 Laplacian 耦合), 因此主动回退。这不是调参问题 -- 是优化 formulation 在告诉我们: 非均匀缩放与 Laplacian cost 结构不兼容。

### 6.3 与 Wuji baseline 的对比

Wuji 的 `FullHandVec` loss 是 per-finger 独立的向量匹配:
```
Wuji: cost = sum_finger ||robot_vec_f - alpha_f * source_vec_f||^2
  -> 缩放 finger_f 不影响 finger_g 的 cost
```

Interaction mesh:
```
IM: cost = ||L_robot - L_source||^2
  -> L 矩阵通过 Delaunay 邻接耦合所有点
  -> 缩放任何一个点都影响所有邻居的 cost 项
```

这解释了为什么 Wuji 的 per-user segment_scaling (5x3=15 个手动参数) 效果好, 而相同概念在 Laplacian 框架中失效。

---

## 7. 同期实验: Orientation Probe Points

在 bone scaling 之前还尝试了 fingertip orientation probe points (21pt -> 26pt):

### 做法
每个 fingertip 额外添加一个沿 DIP->TIP 方向偏移 5mm 的虚拟探针点, 通过两点位置差隐式编码指尖朝向。

### 结果 (cosine similarity 指标)
- Pinky DIP-TIP 反弓 (cos < 0): 56 帧 -> 1 帧 (-98%)
- 严重偏离 (cos < 0.5): Pinky 579 -> 146 帧 (-75%)

### 结果 (关节空间 q < 0 指标)
改善有限, 全局反弓率 99.8% -> 98.8%。Probe 点解决的是 position ambiguity (位置相似但朝向不同的歧义), 但大多数反弓不是歧义造成的, 而是优化器主动选择负关节角来更好匹配 source 位置。

---

## 8. 结论

1. **Per-segment bone scaling 温和改善**部分关节 (食指 PIP -10%, 中指 PIP -7%), 但引入其他关节的新反弓
2. **Laplacian formulation 从根本上抗拒非均匀缩放**, 因为 Delaunay mesh 拓扑的跨手指耦合
3. **自适应调参收敛到 alpha ~1.0**, 证实了 formulation 层面的不兼容性
4. **Probe points 对 cosine 指标改善明显**, 但对关节空间反弓改善有限
5. **指尖误差**从 11.6mm 升到 13.8mm (bone scaling), Wuji BL 为 8.9mm

## 9. EXP-4: ARAP 逐顶点旋转补偿 (2026-04-14)

### 动机

骨段缩放方案受限于 Laplacian 跨手指耦合, 换一个正交方向: 不改 source 点位置, 而是改 Laplacian target 的方向. 用 ARAP (Sorkine 2007) 的 per-vertex SVD 估计邻域旋转, 旋转 source Laplacian target.

### 结果

| 指标 | Baseline IK | IM 原始 | IM+ARAP | 变化 |
|------|------------|---------|---------|------|
| 反弓帧比率 | 84.1% | 94.8% | **69.2%** | -25.6pp |
| 食指 PIP 反弓 | 57.9% | 70.6% | **19.4%** | -51.2pp |
| 中指 PIP 最差角 | -27.8° | -15.1° | **-9.1°** | 改善 |
| 速度 | 401 fps | 67 fps | 26 fps | 2.5x 变慢 |

退化: 拇指 DIP 3.6%->9.2%, 食指 DIP 17.0%->18.5%

### 机制

PIP 自身 |L|≈0, R_i 旋转效果弱. 但 MCP/TIP (|L| 大) 获得更好方向目标, 通过 Laplacian 矩阵耦合间接约束 PIP. 详细分析见 `with_arap_rotation_comp/results.md`.

### 状态

worktree 分支 feat/arap-rotation-comp, 未合入主分支.

---

## 10. EXP-7: ARAP 逐边能量 + 骨架拓扑 (2026-04-14)

### 动机

EXP-4 (ARAP 旋转补偿) 通过间接路径改善反弓, 但本质上仍使用 Laplacian 能量 (平均化丢方向). 尝试更根本的 formulation 变革: 用 ARAP 逐边能量替代 Laplacian 能量.

### 方法

**ARAP 逐边能量**: 对每条边 (i,j) 独立比较, 不做邻居平均:
```
Laplacian: min ||L @ robot - L @ source||²       (L 平均了所有邻居)
ARAP edge: min Σ_{(i,j)} ||(r_i-r_j) - R_i(s_i-s_j)||²  (逐边保方向)
```
R_i 由 SVD per-vertex rotation estimation 提供 (复用 EXP-4 的 `estimate_per_vertex_rotations`).
无辅助变量 (去掉 lap_var 和等式约束), SOCP 更简洁.

**骨架拓扑**: 用手部骨骼结构替代 Delaunay, 消除跨手指连接:
```
Delaunay: 60-80 条边, 含跨手指连接 (PIP 邻居来自其他手指)
骨架:     20 条边, 每指独立链 (Wrist→MCP→PIP→DIP→TIP)
```

### 四方对比结果 (1000 帧)

| 指标 | Laplacian | ARAP rot-comp | Edge+Delaunay | **Edge+Skeleton** |
|------|-----------|---------------|---------------|-------------------|
| **Overall 反弓 %** | 74.9% | 65.3% | 89.6% | **63.6%** |
| 速度 (fps) | 68 | 27 | 20 | **44** |

逐关节反弓 (% 帧 < 0):

| 关节 | Laplacian | ARAP rot | Edge+Dela | **Edge+Skel** |
|------|-----------|----------|-----------|---------------|
| Thumb DIP | 18.2% | 43.5% | 58.9% | **0.0%** |
| Index PIP | 12.1% | 1.2% | 8.3% | 18.8% |
| Index DIP | 39.6% | 26.1% | 37.6% | **0.0%** |
| Middle PIP | 33.2% | 32.9% | 53.1% | 49.1% |
| Ring PIP | 51.8% | 37.9% | 52.1% | 56.8% |
| Pinky PIP | 58.6% | 29.8% | 65.8% | **0.0%** |
| Pinky DIP | 0.0% | 0.0% | 0.0% | **0.0%** |

### 核心发现

1. **Delaunay + ARAP edge 比 Laplacian 更差 (89.6% vs 74.9%)**: Delaunay 跨手指邻接让 per-vertex rotation R_i 是不兼容方向的折中, 方向引导不准. 和 bone scaling 失败的根因相同 -- **Delaunay 跨手指耦合**.

2. **骨架拓扑彻底修复 DIP 反弓 (全部 0%)**: 拇指/食指/小指的 DIP 从大量反弓变为零, 因为骨架拓扑中 R_i 的邻居全在同一指链上, 运动方向一致.

3. **骨架拓扑的 PIP 仍有反弓 (Middle 49%, Ring 57%)**: PIP 在骨架中只有 2 个邻居 (MCP + DIP), R_i 约束不足. 可能需要给 PIP 加相邻手指 MCP 作为弱连接.

4. **速度提升**: Edge+Skeleton 44fps, 比 Edge+Delaunay 20fps 快 2.2x (20 边 vs 60-80 边), 比 ARAP rot-comp 27fps 也快.

### 结论

骨架拓扑是解决 Delaunay 跨手指耦合的正确方向. DIP 反弓从根本消除. 但 PIP 需要更多邻居约束, 下一步探索 PIP-to-adjacent-MCP 弱连接.

worktree 分支: feat/arap-edge-energy

---

## 11. 后续方向

1. **PIP 弱跨指连接**: 骨架拓扑 + 给 PIP 加相邻手指 MCP 作为低权重邻居
2. **ReConForM 风格自适应权重**: 替换 uniform weight, 按 source 距离/接触状态动态调权
3. **骨架拓扑 + Laplacian 能量**: 只换拓扑不换 cost, 看是否也能改善
4. **Direction C (GeoRT 路线)**: 几何 loss 训 MLP, 彻底跳出 per-frame 优化

---

## 11. 代码改动清单

### 主分支 (已应用)

| 文件 | 改动 | 说明 |
|------|------|------|
| `src/hand_retarget/config.py` | 添加 `use_orientation_probes`, `probe_offset`, `_PROBE_MAPPING_LEFT/RIGHT` | Probe 点配置 |
| | 添加 `use_bone_scaling`, `bone_scaling_warmup` | 骨段缩放配置 |
| | 修改 `joints_mapping` 属性, 添加 YAML 解析 | 支持 probe 映射合并 |
| `src/hand_retarget/mujoco_hand.py` | `MuJoCoHandModel.__init__` 添加 `probe_offset` 参数 | 兼容 probe 模式 |
| | 添加 `_add_probe_frames` 方法 | Pinocchio 模型动态添加 5 个 probe frame |
| | 添加 `get_default_qpos` 方法 | bone scaling 计算 robot 骨段长度需要 |
| `src/hand_retarget/retargeter.py` | 添加 `_augment_with_probes` 方法 | 21->26 点增强 |
| | 添加 `_finger_segments`, `_bone_ratios`, `_warmup_seg_lengths` | 骨段缩放数据结构 |
| | 添加 `_compute_robot_segment_lengths`, `_compute_source_segment_lengths` | 骨段长度计算 |
| | 添加 `_apply_bone_scaling` | warmup + per-segment 累积缩放 |
| | `retarget_frame` 中集成 probe 增强和骨段缩放 | 在 Laplacian 计算前应用 |
| `demos/legacy/play_interaction_mesh.py` | 添加 `--probes`, `--bone-scale` 参数 | CLI 开关 |
| | 添加 probe 点可视化 (橙色/绿色) | source probe + robot probe 渲染 |
| | 添加 cache suffix 区分模式 | `_probes`, `_bscale` |
| | 默认 pkl 路径改为 `data/manus_for_pinch/manus1_5k.pkl` | |
| `demos/legacy/play_mesh_only.py` | 默认 pkl 路径同步更新 | |

### Worktree 沙盒 (未合入, 路径 `.claude/worktrees/agent-a72cf773/`)

| 文件 | 改动 |
|------|------|
| `config.py` | 添加 `bone_scaling_adaptive`, `bone_scaling_clamp` 字段 |
| `retargeter.py` | 添加 `_update_bone_ratios` 自适应方法 (每 50 帧检查 J3/J4 反弓, 拉 alpha 向 1.0) |

### 默认行为

所有新功能默认关闭:
- `use_orientation_probes = False`
- `use_bone_scaling = False`

不影响现有 pipeline 的任何行为。

---

## 12. 实验数据索引

所有数据存放在 `experiments/hyperextension_study/` 下:

```
hyperextension_study/
├── README.md                          # 本文件
├── baseline_im/
│   ├── q_sequence.npy                 # IM baseline (21pt, 无缩放), shape (5000, 20)
│   ├── proc_landmarks.npy             # 预处理后的 source landmarks, shape (5000, 21, 3)
│   └── cos_errors.npy                 # DIP-TIP 骨段方向 cosine similarity, shape (5000, 5)
├── baseline_wuji/
│   └── q_sequence.npy                 # Wuji baseline (NLopt IK)
├── with_probes/
│   ├── q_sequence.npy                 # IM + orientation probe points (26pt)
│   └── cos_errors.npy                 # Probe 模式的 cosine similarity
├── with_bone_scaling_fixed/
│   └── q_sequence.npy                 # IM + 固定 per-segment 骨段缩放
├── with_bone_scaling_adaptive/
│   └── q_sequence.npy                 # IM + 自适应骨段缩放
├── with_arap_rotation_comp/           # EXP-4: ARAP 逐顶点旋转补偿
│   ├── results.md                     #   详细结果文档
│   └── exp_hyperextension.py          #   三方对比实验脚本
├── with_arap_edge_energy/             # EXP-7: ARAP 逐边能量 + 骨架拓扑
│   └── exp_arap_edge.py               #   四方对比实验脚本
├── exp_hyperextension.py              # EXP-4 实验入口脚本 (根级)
└── cache/                             # play_interaction_mesh.py 播放缓存
    ├── manus1_5k_im_cache.npz         # baseline
    ├── manus1_5k_im_probes_cache.npz  # probes
    ├── manus1_5k_im_bscale_cache.npz  # bone scaling
    ├── manus1_5k_im_rotcomp_cache.npz # ARAP rotation compensation
    ├── manus1_5k_bl_cache.npz         # Wuji baseline (exp_hyperextension 格式)
    └── manus1_5k_wuji_cache.npz       # Wuji baseline (play_manus 格式)
```

数据源: `data/manus_for_pinch/manus1_5k.pkl` (5000 帧, 左手, Manus 手套)
