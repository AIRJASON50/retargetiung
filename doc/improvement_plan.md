# Interaction Mesh 手部 Retargeting -- 改进计划

> 状态: 算法核心已从 OmniRetarget robot_only mode 正确移植
> 问题: retargeting 质量不够好
> 目标: 先忠实复现 robot_only mode, 再探索改进

---

## Phase 0: 忠实复现 Robot-Only Mode (当前)

**原则**: 灵巧手和人手尺度/比例接近 (不像人形全身差距大).
OmniRetarget 只用一个全局缩放 (robot_height / human_height), 没有逐段缩放.
Laplacian 变形 + FK 约束应该自动处理局部比例差异.

### 已完成

- [x] 关键点 16 -> 21: 补回 5 个 link1 (MCP) 点, 和 baseline 的完整 link 集合一致
- [x] smooth_weight 0.1 -> 0.2: 匹配 OmniRetarget 默认值

### 待完成 (匹配 OmniRetarget robot_only)

- [ ] **全局缩放替代逐段缩放**: 去掉 segment_scaling, 换成单一全局缩放
  - OmniRetarget: `smpl_scale = robot_height / human_height` (一个标量作用于所有点)
  - 没有逐指逐段的手动标定
  - 手部场景: 自动计算 palm → middle_fingertip 距离比
  - config 中 segment_scaling 全部设为 `[1.0, 1.0, 1.0]`
  - 在 `preprocess_landmarks` 的坐标变换后加全局缩放

- [ ] **Q_diag 关节正则化**: OmniRetarget 用于防止腰部过度扭转
  - 手部: 对每根手指的 joint1 (侧摆/外展) 加惩罚
  - 防止不自然的手指张角
  - 默认权重: 0.2 (匹配 OmniRetarget 的腰部正则化)

- [ ] **nominal_tracking 代价**: OmniRetarget 用衰减权重追踪默认姿态
  - w_init = 5.0, tau = 10 帧
  - 第一帧: 强拉向默认姿态
  - 后续帧: 指数衰减到零
  - 防止初始姿态跳飞, 提供 warm-start 稳定性

### 不在 OmniRetarget robot_only 中 (暂不添加)

- 逐段缩放 (baseline 特有, OmniRetarget 不用)
- 碰撞约束 (OmniRetarget 有地面碰撞, 固定基座手部不需要)
- 速度限制 (论文 Eq.3d 声称有, 但代码未实现)
- 脚贴地约束 (手部不适用)

### 评估

Phase 0 完成后, 用 manus1.pkl 跑一次, 和 baseline 视觉对比.
质量相当 -> robot_only mode 忠实复现成功.
质量更差 -> 诊断哪个缺失项影响最大.

---

## Phase 1: Ho 2010 原论文改进 (Phase 0 通过后)

这些改进来自 Ho 2010 原论文, 被 OmniRetarget 简化掉了.
只在 Phase 0 证明算法正确但质量仍可提升时尝试.

### EXP-1: 固定拓扑 (来自 Ho 2010)

**内容**: Ho 2010 明确保持 mesh 拓扑在所有帧固定.
OmniRetarget 每帧重建 Delaunay.
Ho 2010 原文: "re-computing the tetrahedralization at each morph-step would result
in gradual drifting of the motion away from the original sequence"

**对手部的意义**: 手部关键点间距 ~cm 级, 微小运动就能翻转 Delaunay 边.

**改法**: 用第一帧的 Delaunay 构建邻接, 后续帧复用.

---

### EXP-2: 距离权重 Laplacian (来自 Ho 2010)

**内容**: Ho 2010 用 w_ij = 1/distance(i,j). OmniRetarget 用等权.
论文: "normalized weights which are set as inversely proportional to the distance"

**对手部的意义**: 等权让不同手指的关节点互相拉扯力度一样.
距离权重自然衰减跨手指耦合.

**改法**: laplacian 函数中 `uniform_weight=False`.

---

### EXP-3: 语义邻接 (新方案, 两篇论文都没有)

**内容**: 用手动定义的手部拓扑替代 Delaunay.
每根手指是独立链, 手指之间只通过 palm 连接.

**原因**: 即使用了距离权重, Delaunay 仍可能创建错误的跨手指连接.
手部骨架拓扑是已知先验, 不需要自动生成.

**改法**: mesh_utils.py 新增 `create_hand_adjacency()`.

---

### EXP-4: 二阶时间平滑 (来自 Ho 2010)

**内容**: Ho 2010 用加速度约束 (Eq. 4). OmniRetarget 只有一阶平滑.

**改法**: 增加 || q_{t-2} - 2*q_{t-1} + q_t ||^2 代价项.

---

## Phase 1 实验顺序

```
EXP-1 (固定拓扑)     -- 5 min, 手部场景最可能有效
  |
  v
EXP-2 (距离权重)     -- 2 min, 减少跨手指耦合
  |
  v
EXP-3 (语义邻接)     -- 30 min, 彻底消除跨手指耦合
  |
  v
EXP-4 (二阶平滑)     -- 10 min, 减少帧间抖动
```

---

## 关键洞察: 预处理的区别

```
Ho 2010 (原作):
  不对 source 点做预处理
  用 morph step 处理尺寸差 (逐步骨段长度插值)
  Laplacian + 骨段约束全自动完成

OmniRetarget:
  一个全局缩放: smpl_scale = robot_height / human_height
  没有逐段缩放, 没有逐指标定
  先均匀缩放所有 source 关键点, 再让 Laplacian 处理局部差异
  比 Ho 2010 简单 (不需要 morph step), 但需要全局缩放作为提示

Baseline (wuji_retargeting):
  逐指逐段手动缩放 (15+ 参数需要手工调)
  必须这样做因为 baseline 是直接 IK 位置匹配, 不是 Laplacian

手部场景:
  人手和机器手尺度接近 (~1:1 到 ~2:1)
  全局缩放应该足够 (自动计算 palm-to-fingertip 比例)
  逐段缩放在 interaction mesh 下不应该需要
```

---

## 诊断工具

实验前, 在 `scripts/` 下添加:

1. **拓扑稳定性检查**: 打印 frame 0, 50, 100 的邻接表. 如果不同 -> EXP-1 有意义
2. **Laplacian 矩阵热力图**: 画 |L| 热力图. 如果跨手指项大 -> EXP-2/3 有意义
3. **关节角直方图**: 每关节在全序列的范围. 如果 joint1 过大 -> 需要 Q_diag
4. **加速度图**: || q_{t-1} - 2q_t + q_{t+1} || 随时间变化. 如果尖刺多 -> EXP-4
5. **全局缩放检查**: 对比 source 和 robot FK 的 palm-to-tip 距离. 如果差 >20% -> 需要全局缩放
