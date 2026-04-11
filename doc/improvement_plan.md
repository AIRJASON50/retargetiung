# Interaction Mesh 手部 Retargeting -- 改进计划

> 状态: robot_only 核心算法 + 三组实验完成
> 最佳配置: 固定拓扑 + 语义权重 (对指间距保持首次优于 baseline)
> 核心发现: Laplacian 天然适合间距保持, 语义权重强化接触区域

---

## 已完成

### 基础移植
- [x] OmniRetarget robot_only 核心算法移植到 WujiHand
- [x] 关键点 21 个 (palm + 5×4), 映射匹配 baseline (跳过 link2)
- [x] FK: Pinocchio (有 tip_link)
- [x] 预处理: baseline 的 SVD + OPERATOR2MANO + 15° Z 旋转
- [x] 全局缩放替代逐段缩放
- [x] Laplacian weight=10, smooth_weight=0.2
- [x] 可视化: mesh/source/robot 点, 暂停回放, 快捷键切换
- [x] 缓存: 首次自动缓存, 支持多模式独立缓存

### 实验
- [x] **EXP-1 固定拓扑**: 方向误差 -11%, 抖动 -43%, Ho 2010 建议验证通过
- [x] **EXP-2 距离权重**: 位置精度恶化 (14.8→20.1mm), 平滑性改善, **不采用**
- [x] **EXP-3 语义权重**: 对指间距保持 -32% (固定拓扑下), 首次优于 baseline, **采用**

### 确定的最佳配置
```
固定拓扑 (首帧 Delaunay, 后续复用) + 语义权重 (对指时指尖 5x boost)
  → --fixed-topology --semantic-weight
```

---

## 实验结果汇总

### EXP-1: 固定拓扑 vs 逐帧重建

| 指标 | Baseline | IM 逐帧 | IM 固定 |
|------|----------|--------|--------|
| 指尖位置 (mm) | **12.0** | 14.5 | 14.8 |
| 指尖方向 (deg) | **10.8** | 23.6 | **20.9** |
| 抖动 Jerk | **4568** | 8492 | **4873** |

**结论**: 固定拓扑全面优于逐帧重建, 抖动降 43%

### EXP-2: 距离权重 (Ho 2010 原设计)

| 指标 | IM 均匀 | IM 距离权重 |
|------|--------|-----------|
| 指尖位置 (mm) | **14.8** | 20.1 |
| 抖动 Jerk | 4873 | **3536** |

**结论**: 骨架结构约束被破坏, 不采用. 改 L 矩阵内部权重是错误方向.

### EXP-3: 语义权重 (对指感知)

| 指标 | Baseline | IM 均匀 | IM 语义 |
|------|----------|--------|--------|
| 拇-食间距 (对指帧) | 5.46mm | 1.30mm | **0.88mm** |
| 拇-食间距 (非对指帧) | 5.97mm | 7.36mm | 7.36mm |
| 指尖位置 (mm) | **12.0** | 14.8 | 14.8 |

**结论**: 对指帧改善 32%, 非对指帧零影响. 间距保持首次优于 baseline.

---

## 管线中的人类先验清单

| # | 先验 | 可否自动化 |
|---|------|---------|
| 1 | JOINTS_MAPPING (21 对手动对应) | 换手需重写 |
| 2 | OPERATOR2MANO 旋转矩阵 | 硬编码 |
| 3 | 15° Z 旋转 (manus 设备) | 换设备需重调 |
| 4 | global_scale | 可自动算 (palm→tip 比) |
| 5 | 跳过 link2 的决策 | 基于距离检查 |
| 6 | uniform 权重选择 | 设计选择 |
| 7 | Laplacian weight=10 | OmniRetarget 经验值 |
| 8 | smooth_weight=0.2 | 经验值 |
| 9 | step_size=0.1 | 经验值 |
| 10 | 对指阈值 10/30mm, boost 5x | 经验值 |
| 11 | 首帧 50 迭代, 后续 10 | 经验值 |
| 12 | 初始 q = mid-range | 假设 |
| 13 | SVD 帧估计用哪三个点 | 硬编码 |

---

## 待做

### 近期 (优化框架内)

- [ ] **帧间速度硬约束**: OmniRetarget 公式 3d, |q_new - q_prev| ≤ v_max*dt
  - 当前 smooth_weight=0.2 是软惩罚, 已经是 dex-retargeting 的 50 倍
  - 逐帧 Delaunay 的抖动靠固定拓扑已解决, 硬约束作为额外保障
  
- [ ] **碰撞约束**: 开启 self-collision, 防手指穿透
  
- [ ] **指尖方向描述符**: 叠加 ||cosine(dir_robot, dir_source)||² 到 loss
  - 当前方向误差 20.9° vs baseline 10.8°, 有改善空间
  - 来自 ReConForM 的 L_dir 项, 比例无关
  
- [ ] **语义邻接**: 手动骨架拓扑替代 Delaunay, 消除跨指连接

### 中期 (扩展)

- [ ] **Phase 2: 加物体交互**: interaction mesh 的真正价值
  - 物体采样点作为固定锚点
  - 采样密度 = 隐式优先级
  
- [ ] **ReConForM 描述符融合**: L_dist + L_dir + L_pen 叠加到 Laplacian
  - 方案 A: Laplacian 粗匹配 + 描述符细保持
  
- [ ] **归一化 Laplacian**: 除以局部特征长度, 解决比例敏感问题

### 长期 (路线选择)

- [ ] 精读 GeoRT — 几何 loss 训练 retarget 网络, 可能是优化和学习的桥梁
- [ ] 评估是否需要完全替换 Laplacian (ReConForM 方案 C)

---

## 文件结构

```
retargeting/
├── src/hand_retarget/           # 核心库
│   ├── retargeter.py            # InteractionMeshHandRetargeter (含语义权重)
│   ├── mesh_utils.py            # Delaunay, Laplacian
│   ├── mujoco_hand.py           # Pinocchio FK/Jacobian
│   ├── mediapipe_io.py          # 预处理 (SVD + MANO + 缩放)
│   └── config.py                # 配置 + JOINTS_MAPPING
├── config/                      # YAML 配置
├── data/                        # 数据 + 缓存
│   ├── manus1.pkl               # 完整轨迹 (13341帧)
│   ├── manus1_5k.pkl            # 默认 (5000帧)
│   └── cache/                   # 自动缓存
├── demos/
│   ├── hand/baseline/           # baseline 播放
│   ├── hand/interaction_mesh/   # IM 播放 (--fixed-topology --semantic-weight)
│   ├── hand/compare/            # 双窗口对比
│   └── humanoid/                # OmniRetarget G1 demo
├── experiments/                 # 实验脚本 + 结果
│   ├── benchmark.py             # 通用 benchmark (含 pinch 指标)
│   ├── exp1_fixed_topology.py
│   ├── exp2_distance_weight.py
│   ├── exp3_semantic_weight.py
│   └── results_exp{1,2,3}.md
├── doc/
│   ├── omni.md                  # OmniRetarget 算法细节
│   └── improvement_plan.md      # 本文件
└── lib/                         # 参考库 (gitignore)
```
