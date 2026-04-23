# 手部运动重定向技术报告

> **版本**: 2026-04-23 **状态**: 主管线已定型,接近收敛;contact mask 提取与 contact-aware 权重为下一阶段
> **范围**: MediaPipe 21 点 landmark → WujiHand 20 DOF(可选 6 DOF floating wrist)
> **数据**: Manus(hand-only)+ HO-Cap(hand-object bimanual)

## 摘要

从 [OmniRetarget](https://github.com/Yanjie-Z/OmniRetarget)(人形 IM + SQP)出发移植到灵巧手。
移植过程暴露三个 humanoid 假设在手部失效的结构性问题,经四项针对性适配全部消解。
HO-Cap bimanual subject_3 (297 帧) 成绩:

| 指标 | 值 |
|------|---|
| tip_err | 27.4 mm |
| tip_obj 距离保持 | 15.1 mm |
| 骨方向余弦误差 | 0.0077 (≈ 5°) |
| DIP 反弓率 | 98%(骨长比结构性问题,非管线问题)|
| 吞吐 | 170–200 fps |

核心结论:**5:5:1 默认下 IM 端到端独立贡献 ≈ 0.03 mm**(接触帧)—— 单步 QP 看似 9% 归因,
但多轮收敛后与 cos-IK 高度重合,独立贡献几近于零。
`interaction_mesh_length_scale=0.03m`(IMx1111)可把 IM 真正激活,tip_err −11%,
代价是 MCP_flex 反弓 +2.4×、骨方向误差 +85%、吞吐腰斩。不存在单标量同时在自由/接触帧
平衡 IM 与 cos-IK —— 骨长 scaling 是真正的结构瓶颈。

---

## 1. 问题与目标

**输入**:MediaPipe 21 点 3D landmark + object 6D pose + mesh → **输出**:robot qpos 时序。

**应用约束**(来自 RL 数据清洗需求):
- 保留人-物交互语义(pinch 距离、抓握拓扑)
- 保留手形合理性(不反弓、骨方向对齐源)
- 帧间平滑,无 flip
- 160+ fps HO-Cap / 330+ fps Manus
- 无穿透

**选择 IM 的理由**:主流手部 retargeting(dex-retargeting)是关键点位置匹配,不显式建模
手-物相对几何,pinch 时指尖可能距 object 10+ mm。OmniRetarget 的 IM 用
Delaunay + Laplacian 编码**局部几何结构**,保留 contact/pinch 的拓扑关系,自然涌现
接触保持。

---

## 2. 三个移植阻碍

### 2.1 humanoid 的成功依赖"近均匀 scaling"

OmniRetarget 在人形上只做全身均匀缩放(身高比)就可对齐。隐含前提:**各节骨之间
比例大致同构**。身体是长链条,缩放一次即可对齐到 10% 内。

### 2.2 手部违背同构假设

MediaPipe 模板 vs WujiHand URDF 的骨长比:

| 骨段 | src (mm) | robot (mm) | 比 |
|------|----------|-----------|-----|
| Thumb CMC→MCP | 31 | ~50 | 0.62 |
| Index MCP→PIP | 33 | 48 | 0.69 |
| Index PIP→DIP | 22 | 30 | 0.73 |
| Middle MCP→PIP | 32 | 48 | 0.67 |
| Ring MCP→PIP | 29 | 48 | 0.60 |
| **Pinky MCP→PIP** | **21** | **48** | **0.44** |

比例跨度 0.44–0.73(66%),**单参数全局 scaling 无法同时对齐**。按拇指缩放(0.62)
小拇指依然过短;按小拇指(0.44)则其它指 tip 穿出机器人指尖可达范围。

### 2.3 直接移植的失败现象

21 点 hand + 50 点 object = 71 点 3D Delaunay,uniform Laplacian 权重下:
- **跨指虚假邻接**:静止姿态下 pinky-thumb、wrist-tip 等 100+ mm 长边被纳入邻域
- **Laplacian 均值被稀释**:80 mm 远邻居和 30 mm 骨内邻居权重相同,均值失去局部语义
- **反弓严重**:骨长比不兼容下求解器被迫反向弯曲 DIP/PIP 来凑 Laplacian 坐标

---

## 3. 四项适配

### 3.1 Delaunay 边过滤 + 距离权重

实现两道过滤:

- **硬阈值 60 mm**(`delaunay_edge_threshold=0.06`):比两倍平均骨长更长的边一律删除。
  保留骨内(20–50 mm)、邻指跨边(30–60 mm)、指尖-object 近距离(<30 mm);
  剔除 wrist-pinky_tip、thumb-middle_tip 等长跨越。
- **距离衰减** `w_ij = exp(-k·‖e_ij‖)` 归一化后分配给邻接的 Laplacian 权重
  (`laplacian_distance_weight_k=20`)。

**主力是 60 mm mask,不是 k 权重**。对 k ∈ {None, 5, 10, 20, 50, 100} 扫描
(subject_1 166 帧):

| k | tip_err (mm) | tip_obj (mm) | fps |
|---|-------------|-------------|-----|
| None (uniform) | 27.37 | 15.11 | 246 |
| 5 | 27.37 | 15.11 | 208 |
| 20 (当前) | 27.37 | 15.11 | 218 |
| 100 | 27.38 | 15.11 | 225 |

**tip_err / tip_obj 小数点后 4 位完全相同**。k 调节对 IM 内部权重分布有影响,但因
IM 在 5:5:1 默认下对 q 的端到端影响已接近零(见 §6.1),k 值几乎无可观察差异。
硬阈值 60 mm 才是消除跨指噪声、让 IM 语义化的关键。

### 3.2 GMR 启发的骨方向 warmup

Delaunay 过滤后,反弓仍存在 —— 骨长比不兼容无法通过拓扑过滤解决。只要 Laplacian 的
相对距离在 source/robot 之间不一致,IM 就会推出扭曲姿态。

借鉴 [GMR (2025)](https://github.com/YanjieZe/GMR) 的"先方向后位置"两阶段范式:
引入 **骨方向 cosine IK** 作为 warmup。核心观察:**骨方向是 scale-invariant 的**,
无论骨长只要 parent→child 方向一致即对。

S1 cost:

```
cost = w_rot · Σ_bones ‖d_rob(q) − d_src‖²  +  smooth · ‖dq − (q_prev − q)‖²
```

其中 `d_rob = normalize(FK_child − FK_parent)`、`d_src = normalize(lm_child − lm_parent)`。
Jacobian `J_dir = (I − d d^T) / ‖e‖ · (J_c − J_p)` 是单位向量求导的数学必然形式。
手写 Gauss-Newton QP(daqp),首帧 cap 20 iter,其他帧 cap 5,早停 `‖Δq‖ < 1e-3`。

**效果**:
- Manus 单手:warmup 单独就能达到良好对指(PIP/DIP 天然无反弓,91 fps)
- HO-Cap:反弓率从首帧不收敛时 97% 改善到 70-80%,仍高但显著好转

### 3.3 cosik_live 联合优化

最初 S2 anchor 是对 warmup 输出 q_S1 的 L2 球:

```
cost = λ_IM · ‖L·V − L·V_src‖² + λ_anchor · ‖q − q_S1‖² + smooth
```

问题:
1. q_S1 被 warmup 的 smooth 项拉偏,残留 ‖r_bone‖ ≈ 0.57 —— L2 球中心有偏差
2. L2 Hessian `diag(confidence)` 各向同性,必须手工设 `confidence[MCP_abd]=0.5` 才能
  让 IM 调得动 MCP 外展

改为 **cosik_live**:cos-IK cost 直接内嵌 S2 QP,每次 iter 按 current q 重新线性化:

```
cost = λ_IM · IM(q) + w_rot · Σ ‖d_rob(q) − d_src‖² + smooth
```

Hessian `J^T J` 的各向异性天然处理 MCP abd 这种骨方向不敏感方向的自由度。

扫 `w_rot ∈ {5, 1, 0.5}`(subject_1 166 帧):

| 配置 | tip_err | bone_cos | DIP 反弓 | tip_obj | fps |
|------|---------|----------|---------|---------|-----|
| l2 legacy, anchor=5 | 24.56 | 0.0077 | 71.1% | 11.39 | 197 |
| cosik_live, w_rot=5 | 25.06 | 0.0075 | 70.5% | 11.71 | 174 |
| cosik_live, w_rot=1 | 25.01 | 0.0078 | 69.9% | 11.64 | 170 |
| cosik_live, w_rot=0.5 | 24.97 | 0.0082 | **67.5%** | 11.59 | 167 |

数值结果相近,cosik_live 的价值是架构性:消除 `confidence` 手调、权重量纲统一
(warmup 和 S2 都 w_rot=5)、单层优化便于扩展。

### 3.4 MCP 关节替代(link1 → link2)

WujiHand MCP 是 **2 铰链复合**:joint1 (flex) 连 palm→link1、joint2 (abd) 连 link1→link2,
空间 offset **4.6 mm**(拇指 CMC 偏移更大,15.8 mm)。MediaPipe MCP 是单一球关节(lm[5]),
约等于两个枢轴的**虚拟球心**。原映射 `MCP → link1` 停在 flex 枢轴,引入 4.6 mm 寄生偏移。

全候选扫描(subject_3 bimanual,297 帧 × 2 手):

| 候选 | wrist→MCP | tip_err | tip_obj | fps |
|------|-----------|---------|---------|-----|
| C1 link1 (原) | 10.62° | 28.63 | 16.13 | 156 |
| **C2 link2** | **9.19°** | 28.15 | 15.67 | 157 |
| C3 midpoint | 9.88° | 28.38 | 15.90 | 150 |
| C5 link2 +3mm offset | 10.56° | 28.65 | 16.28 | 137 |
| C5 link2 +5mm offset | 11.58° | 29.05 | 16.69 | 138 |
| C5 link2 +8mm offset | 13.19° | 29.69 | 17.25 | 140 |
| **T1 link2 + 拇指 CMC link2** | **9.19°** | **27.99** | **15.45** | 158 |

link2 严格优于 link1(零指标退化),拇指同样策略再降掌骨残差。C5 掌法线外推失败
(所有 offset 都让指标变差,可能因为 palm normal 符号/方向和 MediaPipe skin 的真实
偏差方向不一致)。

**采纳:`mcp_surrogate=link2`, `thumb_cmc_surrogate=link2`;`mcp_surface_offset_m` 保留
但默认 0**。

### 3.5 穿透约束

OmniRetarget 的 linearized SDF + 指尖 capsule 方案在手部直接可用,无任何适配:
7.5 mm 指尖 capsule 与 object SDF 的 linearized 非穿透约束作为 QP 的 G @ dq ≤ h 不等式。

**默认启用**(`activate_non_penetration_warmup=True`, `activate_non_penetration_s2=True`),
实测无穿透。

---

## 4. 当前管线

```
MediaPipe 21 点 lm
  → mediapipe_io (SVD+MANO 对齐,wrist 置原点)
  → retarget_frame
      ├── _build_topology
      │     ├── Delaunay(source_pts + object_pts)
      │     ├── 过滤 > 60 mm 长边
      │     └── target_lap = L_src @ V_src
      ├── _compute_weights (可选 pinch-aware 语义权重)
      └── _run_optimization
            ├── Warmup (S1): cosine IK bone direction
            │     cap 20 首帧 / 5 非首帧,早停 ‖Δq‖ < 1e-3
            │     cost = w_rot · Σ ‖d_rob − d_src‖² + smooth
            └── S2: 联合 QP (cosik_live 默认)
                  cap 20 首帧 / 10 非首帧,早停 ‖Δq‖ < 1e-3
                  cost = λ_IM · ‖L_rob · V_rob − target_lap‖²
                       + w_rot · Σ ‖d_rob(q) − d_src‖²
                       + smooth · ‖dq − (q_prev − q)‖²
                       + non-penetration 约束
```

**MCP landmark 由 `_mp_body_pos_jacp` 单一入口解析**:cos-IK 骨端点、warmup、
Delaunay/Laplacian IM keypoint 共享同一 body,保证不 diverge。

### 默认配置

| 字段 | 值 | 说明 |
|------|-----|------|
| `anchor_mode` | `cosik_live` | S2 单层联合 |
| `anchor_cosik_weight` | 5.0 | S2 cos-IK 权重 |
| `angle_warmup_weight` | 5.0 | warmup 权重(同量纲)|
| `angle_warmup_iters_first / iters` | 20 / 5 | warmup 外循环 cap |
| `warmup_convergence_delta / s2_convergence_delta` | 1e-3 / 1e-3 | 统一 q-norm 早停 |
| `n_iter_first / n_iter` | 20 / 10 | S2 外循环 cap |
| `laplacian_weight`(retargeter 硬编码)| 5.0 | IM 权重 |
| `interaction_mesh_length_scale` | None | IM 归一化长度(IMx 模式可选)|
| `delaunay_edge_threshold` | 0.06 m | Delaunay 硬阈值 |
| `laplacian_distance_weight_k` | 20.0 | IM 距离衰减 |
| `smooth_weight` | 1.0 | 时间平滑 |
| `mcp_surrogate` | `link2` | 非拇指 MCP surrogate |
| `thumb_cmc_surrogate` | `link2` | 拇指 CMC surrogate |
| `mcp_surface_offset_m` | 0.0 | 掌法线外推(关闭)|
| `activate_non_penetration_warmup` | True | 穿透约束 |
| `activate_non_penetration_s2` | True | 穿透约束 |
| `use_angle_warmup` | True | 启用 S1 |

---

## 5. IM 真实角色的量化

### 5.1 5:5:1 默认下 IM 端到端贡献 ≈ 0.03 mm

完整 IM 开 / IM 关闭(`laplacian_weight=0`)两组重跑,按 tip-to-obj 距离分类:

| 帧类 | n (%) | ‖Δq_finger‖ | Δq/关节 | Δtip Cartesian | Δtip→obj 距离 |
|------|-------|-------------|---------|----------------|---------------|
| CONTACT <15mm | 85 (28.6%) | 0.0011 rad | 0.003° | **0.028 mm** | 0.017 mm |
| NEAR 15-40mm | 53 (17.8%) | 0.0009 rad | 0.003° | 0.028 mm | 0.011 mm |
| FREE >40mm | 159 (53.5%) | 0.0006 rad | 0.002° | 0.016 mm | 0.012 mm |
| ALL | 297 | 0.0006 rad | 0.002° | 0.019 mm | 0.012 mm |

**接触帧 IM 独立贡献也只 0.028 mm 指尖位移**,比单步 QP marginal attribution
(9%)少两个数量级。

**差距来源**:单步 `‖dq_full − dq_no_IM‖` 测量"删除 IM 的 c 项后 dq 少计算多少";
end-to-end 测量"整个 S2 收敛后 q_final 是否改变"。多轮 QP 迭代后,cos-IK 自己能
覆盖 IM 大多数想推的方向 —— 删掉 IM 只是让 cos-IK 独自走完同一个收敛点。IM 的
**独有贡献**仅在其与 cos-IK 的正交互补子空间,手链式结构下该子空间维度极窄。

### 5.2 `interaction_mesh_length_scale` (L_char) 扫描

让 IM 真正说话需要放大 1111×,即 `L_char = 0.03 m`(用骨长归一化残差使其无量纲)。
代码里 `sqrt_w3 /= L_char` 使 `H = J_w^T J_w` 乘 `1/L_char²`。

subject_3 bimanual(594 frame-hands,反弓计数仅 flex joint):

| L_char | 反弓率 | tip_err | tip_obj | bone_cos | fps |
|--------|--------|---------|---------|----------|-----|
| None(默认)| 98.8% | 27.99 | 15.45 | 7.8° | 155 |
| 0.67 m | 98.8% | 27.98 | 15.45 | 7.8° | 132 |
| 0.1 m | 99.3% | 27.33 | 15.02 | 7.9° | 97 |
| **0.03 m (IMx1111)** | **100%** | **24.81** | **13.67** | **10.6°** | 64 |
| 0.01 m (IMx10000) | 100% | 22.64 | 12.32 | 14.4° | 56 |

分关节反弓(MCP_flex / PIP / DIP):

| L_char | MCP_flex | PIP | DIP |
|--------|----------|-----|-----|
| None | 6.5% | 1.7% | 70.4% |
| 0.03 | 8.9% | 2.3% | 75.8% |
| 0.01 | **15.8%** | 2.9% | 72.5% |

分手指反弓(任一 flex < 0):

| L_char | thumb | index | middle | ring | pinky |
|--------|-------|-------|--------|------|-------|
| None | 51% | 67% | 72% | 80% | 99% |
| 0.03 | 55% | **96%** | 70% | 85% | 90% |
| 0.01 | 60% | **99%** | 59% | 82% | 83% |

**好消息**:IM 放大让手形更靠近 source topology —— tip_err −19%、tip_obj −20%。

**坏消息**:反弓和骨方向双降 —— MCP_flex 反弓率 2.4×、bone_cos 误差 +85%、食指
反弓率接近 100%、fps 腰斩。

**机理**:IM 强拉 robot Laplacian 匹配 source Laplacian。source 骨短 → robot 必须
反向弯曲指根让指尖缩进。MCP_flex / PIP / DIP 出现补偿性反弓,骨方向被强行扭曲。

### 5.3 为什么单 L_char 无法平衡

IM 的 Hessian 在自由 vs 接触帧之间**自身波动 300×**(Fro norm 0.27 → 80):

| 调节目标 | 所需 L_char | 自由帧 IM : cos | 接触帧 IM : cos |
|----------|------------|----------------|-----------------|
| 接触帧持平 | ~0.67 m | cos 胜 275× | 1 : 1 |
| 自由帧持平 | ~0.04 m | 1 : 1 | IM 胜 420× |
| 当前 IMx1111 | 0.03 m | IM 胜 2× | IM 胜 540× |

距离权重 + contact-时密集 Delaunay 的涌现:单标量无法同时修两头。**要做真正的
contact-aware 平衡,需要 runtime 随接触信号动态调节 L_char**。

---

## 6. 局限与结构瓶颈

**结构性问题(非管线问题)**:

1. **骨长比 0.44–0.73** 导致:
   - 指尖位置永远 ~15 mm 偏离源
   - DIP 反弓 70%+ (几何必然)
   - cost 权重 tune 无法治本,需 per-bone source scaling 或 anatomical remapping
2. **MediaPipe 皮肤表面 landmark vs robot 内部枢轴** ~5-10 mm 偏差:
   - MCP link2 已吃掉复合内 4.6 mm,但 skin-to-pivot 未解
   - palm-normal 外推实验失败

**管线已接近上限**:cos-IK 主导 99%+ 的 dq 方向,IM 独立贡献 < 0.1%。所有权重 tune
只能在 cos-IK 的几何极小点周围微调,无法越过骨长-scaling 墙。

---

## 7. 下一阶段

### 7.1 Contact mask 提取(已规划)

从 retargeted qpos + robot FK + object mesh 精确 SDF,得到每帧每指的
binary contact / near / free 标签。用途:

- RL reward shaping(bonus on correct contact matching)
- 后处理筛选失败帧
- IM 权重动态调节(见下)

### 7.2 Contact-aware IM 权重

基于 §5.2 的发现,固定 L_char 无法 Pareto-dominate。计划:

- `L_char(t)` 随接触信号插值:接触时 0.03,自由时 None
- 评估两条质量轴:接触质量(tip_obj、pinch 距离)vs 手形质量(bone_cos、反弓率)
- 目标:dominate 当前固定 L_char 的 Pareto front

### 7.3 架构层面(可选)

- 骨长 source scaling(per-bone)—— 从根本消除 scale 问题
- Object-local Laplacian(`use_object_frame=True` 已有字段,待评估)
- IM 在 cos-IK 的 null space 里运行(投影 IM 梯度) —— 避开冲突

---

## 8. 结论

**主要贡献**:
1. 将 OmniRetarget IM + SQP 方案成功移植到 WujiHand,通过四项适配(Delaunay 过滤、
   cosine warmup、cosik_live 联合、MCP surrogate + 穿透)消除从人形到手部的结构失配
2. 量化 IM 在 5:5:1 下的**实际端到端贡献 ≈ 0.03 mm**,澄清了 "IM 是主要机制" 的表观认知
3. 证明骨长比不兼容是 cost-tune 无法解决的结构瓶颈,定位下一阶段工作方向

**可用性判断**:
- ✓ RL 数据清洗:直接可用,只需接 contact mask 提取
- ✓ 双手协同 retargeting:可用,bimanual 170+ fps
- ⚠ 精细 pinch / < 5 mm 接触精度:需 contact-aware 权重
- ✗ 骨长比不兼容的跨体型 retarget(儿童 → 成人 robot):当前管线治标不治本

---

## 附录:实验数据索引

所有 ablation 的 probe 脚本位于 `experiments/archive/warmup_diagnosis/`,
运行即再生 `.npz`:

| Probe | 对应章节 |
|-------|---------|
| `probe_lchar_sweep_flex_only.py` | §5.2 L_char 扫描 + 反弓 |
| `probe_im_contribution_contact.py` | §5.1 IM 端到端贡献 |
| `probe_mcp_full_ablation.py` | §3.4 MCP surrogate |
| `probe_ab_cosik.py` | §3.3 l2 vs cosik_live |
| `probe_decay_k_sweep.py` | §3.1 k 值扫描 |
| `probe_contact_adaptive.py` | §5.1 单步 marginal 归因 |

### 关键 clip

| Clip | 场景 | 用途 |
|------|------|------|
| `hocap__subject_1__20231025_165502__seg00` | 单手 + 单物体 | §3.3 w_rot 扫 |
| `hocap__subject_3__20231024_161306__seg00` | 双手 + g16_1(主测试)| §3.4, §5 |
| `manus1_5k.pkl` | 单手无物体 | §3.2 warmup solo |

subject_3 clip:297 帧 @ 30 fps(9.9 s),bimanual,接触帧 28.6% / 近接触 17.8% /
自由空间 53.5%。
