# Interaction Mesh 手部 Retargeting -- 改进计划

> 状态: robot_only 核心算法 + 反弓研究 + 代码重构完成
> 最佳配置: 固定拓扑 + 语义权重 (对指间距保持首次优于 baseline)
> 核心发现: Laplacian 天然适合间距保持, 语义权重强化接触区域, 非均匀骨段缩放与 Laplacian 跨手指耦合不兼容

---

## 已完成

### 基础移植
- [x] OmniRetarget robot_only 核心算法移植到 WujiHand
- [x] 关键点 21 个 (palm + 5x4), 映射匹配 baseline (跳过 link2)
- [x] FK: Pinocchio (有 tip_link)
- [x] 预处理: baseline 的 SVD + OPERATOR2MANO + 15 Z 旋转
- [x] 全局缩放替代逐段缩放
- [x] Laplacian weight=10, smooth_weight=0.2
- [x] 可视化: mesh/source/robot 点, 暂停回放, 快捷键切换
- [x] 缓存: 首次自动缓存, 支持多模式独立缓存

### 实验
- [x] **EXP-1 固定拓扑**: 方向误差 -11%, 抖动 -43%, **采用**
- [x] **EXP-2 距离权重**: 位置精度恶化, **不采用**
- [x] **EXP-3 语义权重**: 对指间距保持 -32%, 首次优于 baseline, **采用**
- [x] **EXP-4 ARAP 旋转补偿**: 反弓帧 94.8%->69.2%, 食指 PIP 70.6%->19.4%, 速度 67->26fps, **待评估** (worktree, 未合入)
- [x] **EXP-5 Orientation Probes**: cosine 指标改善 (Pinky 反弓 56->1 帧), 关节空间改善有限
- [x] **EXP-6 Per-Segment Bone Scaling + Adaptive**: Laplacian 跨手指耦合导致缩放失效, adaptive 收敛到 alpha~1.0
  - 详细结果: `experiments/hyperextension_study/README.md`

### 代码重构 (2026-04-14)
- [x] 提取 `demos/shared/` (overlay, playback, cache), 5 个 demo 脚本精简 17%
- [x] `mesh_utils.py` 消除 laplacian 计算重复
- [x] `mujoco_hand.py` 提取 `_rebuild_caches()`
- [x] 删除死代码 (重复 exp_hyperextension, report_combined, input_devices)
- [x] 补齐 `from __future__ import annotations` + 返回类型注解

### 确定的最佳配置
```
固定拓扑 (首帧 Delaunay, 后续复用) + 语义权重 (对指时指尖 5x boost)
  -> --fixed-topology --semantic-weight
```

---

## 迁移检查: OmniRetarget 未迁移功能

### 优先级 1: 直接解决当前问题

#### P1-1. Q_diag 正则化 (反弓惩罚)
- **OmniRetarget**: `||sqrt(Q_diag) * (dq + q_current)||^2` 惩罚特定关节的绝对角度
- **手部适用**: 高 -- 对 PIP/DIP 的负角度设正权重, 直接惩罚反弓
- **改动量**: ~5 行, 在 `solve_single_iteration` 的 `obj_terms` 中加一项
- **与现有功能关系**: 与 Laplacian cost 正交, 不引入跨手指耦合
- **优先做**: 这是反弓的最直接解法, 比 bone scaling/probes/ARAP 都简单且无副作用
- **测试目标**: PIP/DIP 反弓帧率下降, 指尖误差不显著恶化

#### P1-2. ~~浮动基座四元数归一化~~ -- 不适用
- **OmniRetarget**: floating base 用四元数表示旋转, 求解后需 `q[3:7] /= norm(q[3:7])`
- **手部现状**: wrist 用 3 hinge joint (rx, ry, rz) 而非四元数, 不存在归一化问题
- **结论**: 不适用, 跳过

### 优先级 2: 物体交互质量

#### P2-1. 物体坐标系 Laplacian -- 已验证, 仅 augmentation 需要
- **OmniRetarget**: robot 关节先变换到物体局部坐标再算 Delaunay + Laplacian
- **论文动机 (Sec VI-C2, Fig 8)**: 物体位姿变化时, world-frame Laplacian 会变 (L_W: (0,1)->(0,-1)), 但 object-frame Laplacian 不变 (L_O 恒为 (0,1)). 目的是让 augmentation (物体位姿扰动) 后优化目标不变, robot 自动适配新位姿
- **手部验证**: worktree feat/object-frame-laplacian, clip 100 (g20_1, 247 帧, 物体平移 637mm + 旋转 10 度)
- **结果**: 原始 demo retarget 下 |q_diff| = 0 (1e-14 级别). 每帧 source/robot 看到相同物体位姿, 两种坐标系的 cost 数学上等价
- **结论**: 对原始 demo retarget 无效果; 对 augmentation (物体位姿扰动生成训练数据) 是必要功能. 当前不需要, 留待 augmentation pipeline 时再实现

#### P2-2. 完整碰撞流水线
- **OmniRetarget**: mj_collision 宽阶段 + mj_geomDistance 窄阶段, 任意 geom 对
- **手部现状**: 只查指尖-物体穿透, 缺手指间自碰撞
- **改动量**: ~30 行, 扩展 `query_tip_penetration` 为通用碰撞查询
- **测试目标**: 自碰撞帧率下降

### 优先级 3: 改善框架内

#### P3-1. 帧间速度硬约束 (improvement_plan.md 原有)
- `|q_new - q_prev| <= v_max * dt`, 当前只有 smooth_weight 软惩罚
- 固定拓扑已解决大部分抖动, 作为额外保障

#### P3-2. 指尖方向描述符 (improvement_plan.md 原有)
- `||cosine(dir_robot, dir_source)||^2` 叠加到 loss
- 当前方向误差 20.9 vs baseline 10.8, 有改善空间
- 与 orientation probes 是同一问题的不同解法, probes 已验证效果有限

#### P3-3. 语义邻接 (improvement_plan.md 原有)
- 手动骨架拓扑替代 Delaunay, 消除跨指连接
- 可能解决 bone scaling 的跨手指耦合问题

#### P3-4. Nominal Tracking Cost
- **OmniRetarget**: 对选定关节追踪参考轨迹, 权重按 exp(-t/tau) 衰减
- **手部适用**: 中 -- 可用于引导 PIP/DIP 在前几帧不反弓, 之后衰减让 Laplacian 接管

### 优先级 4: 低优先级 / 不适用

#### P4-1. 向量/矩阵 smooth weight
- 标量已够用, per-DOF 版本不是瓶颈

#### P4-2. Ground interaction
- 手部不接触地面, 不适用

#### P4-3. 数据增广流水线
- 物体位姿扰动 + nominal tracking 二次优化, 用于 RL 训练数据生成
- 中期方向, 非近期需求

---

## 实验结果汇总

### EXP-1~3: Laplacian 参数消融

| 指标 | Baseline | IM 逐帧 | IM 固定 | IM 语义 |
|------|----------|--------|--------|--------|
| 指尖位置 (mm) | **12.0** | 14.5 | 14.8 | 14.8 |
| 指尖方向 (deg) | **10.8** | 23.6 | **20.9** | 20.9 |
| 抖动 Jerk | **4568** | 8492 | **4873** | ~4873 |
| 拇-食间距 (对指帧) | 5.46mm | 1.30mm | -- | **0.88mm** |

### EXP-4: ARAP 旋转补偿

| 指标 | Baseline | IM 原始 | IM+ARAP |
|------|---------|---------|---------|
| 反弓帧比率 | 84.1% | 94.8% | **69.2%** |
| 食指 PIP 反弓 | 57.9% | 70.6% | **19.4%** |
| 速度 | 401 fps | 67 fps | 26 fps |

### EXP-5~6: 反弓专题

| 指标 (PIP+DIP q<0 %) | IM baseline | IM+Probes | IM+BScale adapt | Wuji BL |
|---|---|---|---|---|
| 食指 PIP | 70.6% | 67.2% | **60.5%** | 57.9% |
| 中指 PIP | 35.1% | 28.0% | **28.1%** | 41.6% |
| 无名指 PIP | 29.6% | 25.6% | **27.1%** | 18.2% |
| 小指 PIP | 48.4% | 59.3% | 44.4% | **21.3%** |
| 指尖误差均值 | 11.6mm | -- | 13.8mm | **8.9mm** |

核心发现: Laplacian formulation 跨手指耦合导致非均匀骨段缩放不兼容, adaptive 收敛到 alpha~1.0.

---

## 管线中的人类先验清单

| # | 先验 | 可否自动化 | 状态 |
|---|------|---------|------|
| 1 | JOINTS_MAPPING (21 对手动对应) | 换手需重写 | 固定 |
| 2 | OPERATOR2MANO 旋转矩阵 | 硬编码 | 固定 |
| 3 | 15 Z 旋转 (manus 设备) | 换设备需重调 | 固定 |
| 4 | global_scale | 可自动算 (palm->tip 比) | 未实现 |
| 5 | 跳过 link2 的决策 | 基于距离检查 | 固定 |
| 6 | uniform 权重选择 | 设计选择 (EXP-2 验证) | 固定 |
| 7 | Laplacian weight=10 | OmniRetarget 经验值 | 固定 |
| 8 | smooth_weight=0.2 | 经验值 | 固定 |
| 9 | step_size=0.1 | 经验值 | 固定 |
| 10 | 对指阈值 10/30mm, boost 5x | 经验值 | 固定 |
| 11 | 首帧 50 迭代, 后续 10 | 经验值 | 固定 |
| 12 | 初始 q = mid-range | 假设 | 固定 |
| 13 | SVD 帧估计用哪三个点 | 硬编码 | 固定 |
| 14 | MediaPipe 固定模板骨架 | 不可改 (上游) | 已确认 |
| 15 | Per-segment bone ratio | 可自动算 (warmup) | 已实现, 效果有限 |

---

## 下一步行动建议

### 立即可做 (P1, 改动量极小)
1. **Q_diag 正则化** -- 在 SOCP cost 中加 PIP/DIP 负角度惩罚 (~5 行)
2. **浮动基座四元数归一化** -- retarget_hocap_sequence 中加 norm (~3 行)

### 近期 (P2~P3, 需要验证)
3. **物体坐标系 Laplacian** -- HO-Cap 物体交互准确性
4. **ARAP 旋转补偿评估** -- 决定是否合入主分支 (速度 vs 精度权衡)
5. **语义邻接** -- 手动骨架拓扑, 可能解决跨手指耦合

### 中期
6. **完整碰撞流水线** -- 手指间自碰撞
7. **ReConForM 描述符融合** -- L_dist + L_dir + L_pen
8. **归一化 Laplacian** -- 除以局部特征长度

### 长期
9. **GeoRT 路线评估** -- 已完成论文分析, 几何 loss + MLP 是最 scalable 方案
10. **混合 cost** -- Laplacian + per-finger 独立项, 兼得拓扑保持和独立缩放

---

## 文件结构 (重构后, 2026-04-14)

```
retargeting/
├── src/hand_retarget/           # 核心库 (1551 行)
│   ├── retargeter.py            # InteractionMeshHandRetargeter
│   ├── mesh_utils.py            # Delaunay, Laplacian, ARAP rotation
│   ├── mujoco_hand.py           # Pinocchio (fixed) + MuJoCo (floating) FK
│   ├── mediapipe_io.py          # 预处理 + HO-Cap 加载
│   └── config.py                # HandRetargetConfig + JOINTS_MAPPING
├── src/scene_builder/           # MuJoCo 场景构建 (546 行)
│   └── hand_builder.py          # MjSpec 运行时注入
├── demos/
│   ├── shared/                  # 共享工具 (329 行)
│   │   ├── overlay.py           # MuJoCo 可视化 (add_sphere, add_line, set_geom_alpha)
│   │   ├── playback.py          # PlaybackController
│   │   └── cache.py             # load_or_compute
│   ├── legacy/                  # Manus 数据回放
│   │   ├── play_interaction_mesh.py  # IM 主 demo (--fixed-topology --semantic-weight --probes --bone-scale)
│   │   ├── play_manus.py        # Wuji baseline 对比
│   │   └── play_mesh_only.py    # 纯 MediaPipe + Delaunay 可视化
│   ├── hocap/                   # HO-Cap 物体交互
│   │   └── play_hocap.py        # 浮动基座 + 物理碰撞
│   └── humanoid/                # OmniRetarget 参考
│       └── play_omniretarget_demo.py
├── experiments/
│   ├── benchmark.py             # 通用指标
│   ├── hyperextension_study/    # 反弓研究 (EXP-4~6)
│   │   ├── README.md            # 完整实验记录
│   │   ├── baseline_im/         # IM baseline
│   │   ├── baseline_wuji/       # Wuji baseline
│   │   ├── with_probes/         # Orientation probes
│   │   ├── with_bone_scaling_*/  # Bone scaling (fixed + adaptive)
│   │   ├── with_arap_rotation_comp/  # ARAP 旋转补偿
│   │   └── cache/               # 播放缓存
│   ├── hocap_pipeline/          # HO-Cap 管线实验
│   └── laplacian_ablation/      # EXP-1~3 参数消融
├── config/                      # YAML 配置
├── data/manus_for_pinch/        # Manus 手套数据
└── doc/
    ├── improvement_plan.md      # 本文件
    └── omni.md                  # OmniRetarget 算法细节
```
