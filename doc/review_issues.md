# Code Review Issues

5 位 expert(CS 架构 / Python / 数据结构 / 机器人学 / 优化理论)并行审查结果,2026-04-23。

## 进度总表(2026-04-23 更新)

**20 个 bug:7 fully merged / 3 partial(完成了 safe subset)/ 1 rejected / 9 open**:

| 状态 | 数量 | Bugs |
|---|---|---|
| ✅ 完全 merged | 7 | BUG-01 · 11 · 12 · 17 · 18 · 19 · 20 |
| 🟡 Partial(subset done) | 3 | BUG-09 · 14 · 15 |
| ❌ Rejected(负结果)| 1 | BUG-16 |
| ❌ Open | 9 | BUG-02 · 03 · 04 · 05 · 06 · 07 · 08 · 10 · 13 |

本会话从 28 commit 扎进 main,Gate 从 6 → 28 pytest cases,全套 1.7s。详情见下面 "归档" 段。

## Legend

| Mark | 含义 |
|---|---|
| ⚠️  | 涉及旋转/坐标系约定,改动必须谨慎校对 |
| 🔴 P0 | 正确性 bug |
| 🟠 P1 | 优化公式/数值行为 |
| 🟡 P2 | 架构工艺 |
| 🟢 P3 | 性能 |
| 🔵 P4 | Python 工艺 |

---

# 活跃 bugs(13 项)

剩下的都是"大活"— 需要数值验证、API 设计或长期重构。

## 🔵 被 blocked 的小活(1 项)

### BUG-09 剩余 · 拆 mujoco_hand.py → pinocchio_hand.py + mujoco_hand.py
**优先级**: 🟡 P2 · 规模:**小**

**现状**:`MuJoCoHandModel` alias 已删(8b0ff7e),但文件名 `mujoco_hand.py` 里仍然装着 `PinocchioHandModel`(Pinocchio 后端)和 `MuJoCoFloatingHandModel`(MuJoCo 后端)两个类,命名误导。

**预演和考虑**:
- 做法:新建 `src/hand_retarget/pinocchio_hand.py`,把 `PinocchioHandModel`(+ 相关 import)搬过去;`mujoco_hand.py` 只留 `MuJoCoFloatingHandModel` + `HandModelProtocol`;改 `retargeter.py` 的 import 路径。
- 风险:**零数值影响**(纯代码移动)
- 阻塞:working tree 有 background task 的 mujoco_hand.py uncommitted 改动(`query_tip_penetration` 删除 -57 行)。如果我拆文件,那些改动需要手动 merge 到新文件结构。
- 前置:等 background task 先 commit,然后做拆分,或主动把 background 清理并入拆分 commit。

---

## 🔴 正确性 bugs(2 项)

### BUG-02 ⚠️ · _align_frame 物体对齐不一致(rotation bug)
**优先级**: 🔴 P0 · 规模:**中**

**文件**:
- `src/hand_retarget/retargeter.py:_align_frame` SVD 分支
- `src/hand_retarget/retargeter.py:1093-1099` `retarget_hocap_sequence` 对象变换

**问题**:`_align_frame` SVD 分支用 `np.linalg.lstsq(lm_centered[1:6], trans[1:6])` 从 5 点对反推 `R_t` 变换 object 点云 — 返回的**不是正交矩阵**。而 `retarget_hocap_sequence` 对同一物体用严格的 `R_align = R_svd @ OPERATOR2MANO`。两处不一致时,IM 约束看到的 obj 与 `set_object_pose` 看到的几何漂移数度。

**预演**:
1. 改 `_align_frame` SVD 分支 → 用 `estimate_frame_from_hand_points(lm_centered) @ OPERATOR2MANO_{LEFT/RIGHT}`,与主循环共享同一 R_align
2. 删除 lstsq 路径(冗余代码)
3. 需要校对:OPERATOR2MANO 按 hand_side 分发、变换语义 `(obj_t - wrist) @ R_align`

**风险**:
- 会改变 HO-Cap qpos(**正确方向** — 修 bug,不是引入 regression)
- 需要重建 HO-Cap baseline(`tests/hocap_gate_baseline.npz`)
- 需要跑一次 EXP-13 compare 确认 pen_max / tip_err / 反弓率没回退

**前置决策**:
- **baseline 重建流程**:重建完说 "数值变化符合预期"(非 byte-identical 这次是故意的)
- **视觉校对**:跑 subject_3/161306 bimanual 肉眼看对齐,应比之前更正确
- ⚠️ 这是**旋转相关 + 修 bug**,校对要严格:改前 vs 改后对比 `R_obj_world`、`obj_center_aligned` 几组数值

**价值**:高 — 修一个真实的 bug,可能间接改善 BUG-06(穿透硬约束不可行)的根因(对齐错 → 约束集错)。

---

### BUG-03 · DIP joint limit 允许反弓到 -27°
**优先级**: 🔴 P0 · 规模:**小中**

**文件**:
- `assets/wujihand/xmls/wuji_lh_xml/left_mjx.xml:71,97,123,149,175`(DIP joint range `-0.47 ~ 1.57`)
- `src/hand_retarget/retargeter.py` `__init__` 构造 `q_lb` 后

**问题**:MuJoCo XML 允许 DIP 反弓 -27°,`activate_joint_limits=True` 也卡不住。CLAUDE.md 记载 "HO-Cap 启用 S1 反弓率 97%" 的硬件层根因。

**预演**:
1. 新增 `config.dip_nonnegative: bool = True`
2. 在 `InteractionMeshHandRetargeter.__init__` 末尾:
   ```python
   if config.dip_nonnegative:
       offset = 6 if config.floating_base else 0
       for f in range(5):  # 5 fingers
           self.q_lb[offset + 4*f + 3] = max(0.0, self.q_lb[offset + 4*f + 3])
   ```
3. manus.yaml + hocap.yaml 默认启用(或留默认 True,YAML 不用动)

**风险**:
- 改 `q_lb` → QP 可行域紧了 → qpos 变(**方向好**,消除反弓)
- Manus baseline + HO-Cap baseline 都要重建
- 可能有帧 cos-IK 想把 DIP 推到 -5° → 现被卡在 0,bone direction cost 变大,但不会 crash

**前置决策**:
- 重建 2 个 baseline
- 跑 Manus + HO-Cap 对比,确认反弓率 0(或接近)
- 确认不引入新问题(tip 误差不应大幅增加)

**价值**:高 — CLAUDE.md 已知根因,修掉后反弓问题基本解决

---

## 🟠 优化公式 bugs(3 项)

### BUG-04 · IM 权重量纲不一致 → λ_IM 默认被 cos 项 30× 压制
**优先级**: 🟠 P1 · 规模:**中**

**文件**:`src/hand_retarget/retargeter.py:290-292`, `src/hand_retarget/config.py:255`

**问题**:Laplacian 残差以米为单位(~1e-3 m²),cos-IK bone dir 残差无量纲 O(1),smooth 是 rad²。`laplacian=5 : anchor_cosik=5 : smooth=1` 纯经验值,量纲不可比。EXP 验证 IM 被 cos 项 ~30× 主导。

**预演**:
- 做法:`interaction_mesh_length_scale` 默认从 `None` 改为 `0.03`(平均骨长),IM 项 `(1/L_char)² · ‖Δ‖² ≈ O(1)`
- 验证:改后 IM 权重有效增大 ~1000 倍(= 30²),可能太过,需要同步调低 `laplacian_weight`。真正要做 A/B 寻找新 5:5:1 比例

**风险**:
- 改权重平衡,**所有 qpos 变**
- IM 主导上升 → 可能改善反弓(Laplacian 保留相对形态),也可能让 tip err 变大(cos-IK 不再压倒性驱动)
- **需 EXP-13 全 compare**(多个 clip)+ A/B scan 新的权重比例

**前置决策**:
- 写 `experiments/im_weight_sweep/`(新 subdir),扫 L_char ∈ {None, 0.01, 0.03, 0.05, 0.1},每个 L_char 下扫 (λ_IM, w_rot) 比例
- 至少 2-3 个 HO-Cap clip + Manus,看 tip_err / 反弓率 / pen_max
- 确定新默认参数后才改 config.py
- baseline 重建

**价值**:中 — 可能让反弓率、tip 误差同时下降,但参数搜索成本高

---

### BUG-05 · Gauss-Newton Hessian 无 Levenberg-Marquardt damping
**优先级**: 🟠 P1 · 规模:**中**

**文件**:`src/hand_retarget/retargeter.py:306, 435, 440`

**问题**:`H = JᵀJ + w_s·I + 1e-12·I`。1e-12 只防奇异。cos-IK 项 `(I-ddᵀ)/‖e‖` 的 null-space 方向沿 `d_rob` 奇异,完全靠 `w_s` 偶然填充。

**预演**:
1. 引入 μ 状态(retargeter 内部属性,per-frame reset 或全程维持)
2. `H ← JᵀJ + w_s·I + μ·diag(JᵀJ)` (Marquardt)
3. cost 成功降 → μ /= 2;失败 → μ *= 10
4. 合并到 `_solve_qp_trust_shrink`,和 penetration trust shrink 共存

**风险**:
- 改 SQP 数值行为,**qpos 变**
- μ 初值选错可能让收敛慢(过大 μ = 梯度下降步长变小)
- 好的场景:hard clips 原来不收敛的现在能收敛

**前置决策**:
- μ 初值建议:1e-3(温和)或 `trace(JᵀJ) * 1e-3`(scale-aware)
- 先在 1 个 clip 上验证 μ 调度的稳定性(μ 不爆炸、不过度保守)
- 可能需要 new config 字段 `lm_mu_init`, `lm_mu_up`, `lm_mu_down`
- baseline 重建 + hard clips 测试

**价值**:中低 — 一般情况下现有代码已经能收敛(daqp + trust region),LM 主要是理论严谨性和 hard case 的备用

---

### BUG-06 · 源 landmark 穿透时非穿透硬约束不可行
**优先级**: 🟠 P1 · 规模:**中**

**文件**:`src/hand_retarget/retargeter.py:757-803` `_build_penetration_constraints`

**问题**:source landmark 本身穿进 object mesh 时,`h = φ + tol` 永远 < 0 → QP 不可行 → 3 次 trust shrink 耗尽 → stall。 新 gate 里 `test_hocap_penetration_bound` 测早期帧 pen_max 36mm 就是这个症状。

**预演**:
- 做法:混合 penalty
  - `φ < -5mm`:硬约束(source 穿进太深,强制)
  - `-5mm ≤ φ < 0`:软 penalty `w_pen · max(0, -φ - tol)²` 加 cost
  - `φ ≥ 0`:忽略
- 另加接触法向 EMA 平滑(前一 iter 混合 0.5)避免 capsule-mesh 切换处 C⁰ 不连续

**风险**:
- 改约束 handling,**qpos 变**
- 反而可能让接触更好(避免无效硬约束导致的 stall)
- 需要 EXP-13 全 compare 看 pen_max、tip_err、反弓

**前置决策**:
- 软/硬阈值 `-5mm` 是否合理?需要扫 `[-2, -5, -10] mm` 看最佳
- `w_pen` 取值范围
- baseline 重建

**价值**:高 — 直接解决 gate 里 `test_hocap_penetration_bound` 的症状,让非穿透约束真正生效

**推荐**:BUG-04 + BUG-05 + BUG-06 可以**联合做**(都是 cost formulation,都需 baseline 重建 + A/B scan,一次性做完摊薄工作量)。

---

## 🟡 架构重构(3 项)

### BUG-07 · InteractionMeshHandRetargeter 是 God Class
**优先级**: 🟡 P2 · 规模:**大(1 周)**

**文件**:`src/hand_retarget/retargeter.py`(1282 行,20+ 方法)

**问题**:同一个类里 QP 构造 / 拓扑构造 / 预处理对齐 / HO-Cap 时序驱动 / 体名映射 / 非穿透约束 / telemetry 全塞。`retarget_hocap_sequence` 140 行控制流应剥离。

**预演**:
1. 抽 abstract base class `Cost` + `Constraint`
2. 拆 4 策略类:
   - `BoneDirectionCost`(cos-IK)
   - `LaplacianCost`(S2 IM)
   - `PenetrationConstraint`(非穿透)
   - `QPTrustShrinkSolver`(求解 + 信赖域)
3. `InteractionMeshHandRetargeter` 变为 orchestrator + config
4. `retarget_hocap_sequence` 移到 `scripts/retarget_hocap.py` 或新 `src/hand_retarget/drivers/hocap.py`

**风险**:
- 零数值变化(纯重构)
- 改动面巨大,caller 接口变化
- **中途破坏概率高**(1000+ 行改动,各种隐式耦合)

**前置决策**:
- 独立长期分支(`refactor/god-class-split`)
- 分 5-7 个原子 commit(每个引入 1 个抽象 / 迁移 1 个 cost)
- 每步 gate 全过 + byte-identical
- 可能需要 mock FK 在单测里隔离 cost 类

**推荐执行方式**:独立 PR,不走 session 快速 commit 流。和 BUG-08 可联合(都是重构)。

**价值**:高(代码可维护性) — 但不紧急(现在能跑)

---

### BUG-08 · HandRetargetConfig 字段膨胀 + 互斥字段共存
**优先级**: 🟡 P2 · 规模:**大**

**文件**:`src/hand_retarget/config.py:194-303`

**问题**:30 个平铺字段(CLAUDE.md 过时写 "17")。`anchor_mode` / `angle_anchor_weight` / `anchor_cosik_weight` 三者互斥又共存(l2 vs cosik_live)。

**预演**:
- 拆嵌套:
  - `OptimizerConfig`(step_size、n_iter 等)
  - `PenetrationConfig`(3 个 np 字段)
  - `IMTopologyConfig`(Delaunay、Laplacian、midpoint)
  - `AnchorConfig`(sealed union `L2Anchor | CosikLiveAnchor`)
  - `PreprocessConfig`(mediapipe_rotation、global_scale)
- 改 `from_yaml` 按 nested section 解析
- 全仓库 `cfg.xyz` → `cfg.optimizer.xyz` 改 ~50-100 处

**风险**:
- **破坏 YAML schema**,需迁移 manus.yaml + hocap.yaml(但会变更清晰)
- 所有 caller 需改 field access(~50-100 处)
- `from_yaml` 的 unknown key silent drop 行为要同步迁移

**前置决策**:
- 是否保留 flat 访问作 backward-compat(`cfg.smooth_weight` 同 `cfg.optimizer.smooth_weight`)?
- YAML schema 版本化还是 hard break?

**推荐**:和 BUG-07 合做。若 BUG-07 做了,config 拆分自然跟进。

**价值**:中(使用体验、文档性)

---

### BUG-10 · 核心库对 scene_builder 私有符号依赖
**优先级**: 🟡 P2 · 规模:**小中**

**文件**:`src/hand_retarget/mujoco_hand.py:191,305`

**问题**:`MuJoCoFloatingHandModel.inject_object_mesh` 直接 import `_inject_fingertip_sites`, `_inject_wrist_frame`, `_inject_wrist6dof_mode`(下划线私有符号),破坏 "核心库无 viewer 依赖" 承诺(`hand_retarget_viz/__init__.py:2`)。

**预演**:
- 方案 A:`scene_builder` 暴露 public API `build_scene_with_object(scene_xml, mesh_path, hand_side, add_wrist6dof=True)`,核心库只调这个
- 方案 B:把 `MuJoCoFloatingHandModel` 下沉到 `scene_builder/`(因为它天然依赖 scene 构造)
- 方案 A 更符合现有分层(hand model 在核心库,scene building 在独立模块)

**风险**:
- 零数值(纯 API 重命名)
- 改 `_inject_*` → `inject_*`(去下划线前缀)
- 改一处 import

**前置决策**:
- 确认 API 签名 `build_scene_with_object(...)` 参数
- 是否保留 `_inject_*` 旧名作 deprecation shim?建议不保留(干净砍)
- 和 BUG-09 拆文件有协同:都是 scene_builder/mujoco_hand 边界整理

**推荐**:可以当"中等活",一次 PR 完成;和 BUG-09 剩余一起做可能更好(scene_builder API 重新设计时顺带处理 PinocchioHandModel 拆分)。

**价值**:中(架构清晰) — 不紧急

---

## 🟢 性能剩余(3 项)

### BUG-13 · Laplacian 矩阵每帧每 iter 从头 np.zeros((N,N))
**优先级**: 🟢 P3 · 规模:**中**

**文件**:`src/hand_retarget/mesh_utils.py:139-158` `calculate_laplacian_matrix`

**问题**:每 S2 iter 调 `calculate_laplacian_matrix` — Python for 循环填稠密 N×N,每帧数千次。

**预演**:
1. 骨架拓扑模式下帧间邻接固定,预建 COO 索引模板(retargeter 层面 cache)
2. 每 iter 只更新 `data` 数组(浮点权重)
3. 返回 `csr_matrix` 替代当前 dense(节约一次 `sp.csr_matrix(L)` 转换)

**风险**:
- 返回类型变 → caller 适配(retargeter.py 多处用 L dense 索引)
- 如果 caller 用 `L[i, j]` 索引,CSR 上慢;改成 `L @ v` OK
- Delaunay 模式下拓扑变化,模板不能缓存(只有骨架模式受益)
- 可能 **非 byte-identical**(稀疏组装顺序不同)

**前置决策**:
- 实测当前 `calculate_laplacian_matrix` 占每帧多少%?可能不是瓶颈(BUG-12 改 Hessian 装配只 +3.6%)
- 先 profile,如果 <5% 不做

**价值**:低中 — benchmark 驱动,可能 reject(像 BUG-16)

---

### BUG-14 part 3 · _get_robot_keypoints + _get_robot_jacobians 合并 FK-pass
**优先级**: 🟢 P3 · 规模:**中**

**文件**:`src/hand_retarget/retargeter.py:1258-1282`

**问题**:两个方法各自循环 21 body 各调一次 `hand.forward` + `mj_jacBody`,双循环重复 FK。

**预演**:
1. 合并为单个 `_get_robot_keypoints_and_jacobians(q)` 方法
2. 内部一次 FK pass,循环 21 body 各查 pos + jacp
3. 返回 `(positions, jacobians)` 元组
4. 改所有 caller(估计 5-10 处)

**风险**:
- 改调用顺序 → 可能 **ULP 漂移**(像 BUG-12、BUG-20)
- 有多少 caller 同时用 pos+jacp?若多数只用一个,合并反而浪费

**前置决策**:
- 先 grep 找所有 caller,看 pos-only、jacp-only、both 各占比
- 如果 both > 70%,做合并
- 可能接受 ULP 漂移(有前例)

**价值**:中 — BUG-14 原 review 声称 "20-40% fps" 的大头,但前提是双循环真的是瓶颈

---

### BUG-15 剩余 · query_hand_penetration active-set + broadphase
**优先级**: 🟢 P3 · 规模:**中大**

**文件**:`src/hand_retarget/mujoco_hand.py:431-499`

**问题**:HO-Cap 每帧 ~200 次 `mj_geomDistance` + 200 次 `mj_jac`,实测 ≤3 pair 真正 `φ<threshold`。Python 开销是瓶颈。

**预演**:
1. **active-set 缓存**:上一 iter 命中的 geom idx 优先查(改变迭代顺序,不改约束集 membership,理论上 byte-identical)
2. **AABB 粗筛**:`model.geom_aabb` + obj aabb 手动 SAT 检查,淘汰 ≥80%(改变约束集 membership — 边界情况可能漏 pair)
3. **mj_collision broadphase**:MuJoCo C-level broadphase(彻底换查询路径)

**风险**:
- #1 byte-identical 应该可做
- #2、#3 **改约束集 membership**,需 EXP-13 全量 A/B 验证
- 先 profile:到底是 mj_geomDistance 慢还是 mj_jac 慢?

**前置决策**:
- 先 profile HO-Cap 一帧,确认 `query_hand_penetration` 占总时间 %
- 如果 < 10%,做 #1(byte-identical)即可,#2 #3 跳过
- 如果 > 20%,优化全做但需 A/B

**价值**:可能中高(HO-Cap 专属 optimization)或者低(如果不是瓶颈)

---

## 待决策全表

剩下 13 项按 "能否 byte-identical + 是否需 baseline 重建 + 规模":

| Bug | byte-id 可行? | baseline 重建? | 规模 | 前置 |
|---|---|---|---|---|
| BUG-09 剩余 | ✅ | 否 | 小 | 等 background commit |
| BUG-02 ⚠️ | ❌(修 bug)| 是 | 中 | 旋转严格校对 |
| BUG-03 | ❌ | 是 | 小中 | 2 baseline |
| BUG-04 | ❌ | 是 | 中大 | 参数 sweep |
| BUG-05 | ❌ | 是 | 中 | μ 调度策略 |
| BUG-06 | ❌ | 是 | 中 | 软硬阈值选值 |
| BUG-07 | ✅ | 否 | 大 | 独立分支 |
| BUG-08 | ✅ | 否 | 大 | YAML 迁移 |
| BUG-10 | ✅ | 否 | 小中 | API 设计 |
| BUG-13 | 可能 | 可能 | 中 | profile 先 |
| BUG-14 pt 3 | ULP-drift | 看 tol | 中 | caller 统计 |
| BUG-15 剩余 | 部分 | 部分是 | 中大 | profile 先 |

## 推荐下一批

**Stream A · 正确性 → 公式联合**(推荐先做)
- BUG-02 → BUG-03 → BUG-04 + 05 + 06 联合 A/B
- 共 1 次 baseline 重建 + 一次系统 compare
- 价值最高,解决 gate 里的 `test_hocap_penetration_bound` 痛点

**Stream B · 架构大重构**(长期)
- BUG-07 → BUG-08 → BUG-10 + BUG-09 剩余
- 独立分支,零数值,但改动 ~2000 行

**Stream C · 性能 profile 驱动**(可选)
- 先跑一次 profile → 挑 BUG-13 / 14p3 / 15 剩余里真是瓶颈的做

选一个 stream 或具体 bug 告诉我。

---

# 归档

## ✅ 完全 merged(7 bugs · 10 commits)

| Bug | 描述 | Commit | 收益 |
|---|---|---|---|
| BUG-01 | retargeter.py NameError risk(条件 import 提升)| 215fa18 | 消除潜在 NameError |
| BUG-11 | 依赖散落 + 无 pyproject(part 1+2)| ac082f3 + 94e63a1 | pyproject.toml + tests/conftest.py 集中 WUJI_SDK_PATH |
| BUG-12 | QP Hessian 稠密 + np.diag row-scaling | 28ff8da | +3.6% Manus fps,ULP 级噪声 |
| BUG-17 | config.py 裸容器类型注解 | f59ebec | 类型正确 |
| BUG-18 | mj_name2id 吞异常 | 4be173d | 不再掩盖真实错误 |
| BUG-19 | 测试覆盖缺口(test_config + test_mesh_utils + pytest-ify)| 5e7d210 + 3d1df17 + 8aa2d8e | gate 从 6 → 28 cases |
| BUG-20 ⚠️ | Jacobian pin.LOCAL → LOCAL_WORLD_ALIGNED | f74c069 | 代码干净 + ULP 级噪声 |

## 🟡 Partial merged(3 bugs,完成了 safe subset)

### BUG-09 · PinocchioHandModel 伪装 — alias 已删
- **Done**:`MuJoCoHandModel = PinocchioHandModel` alias 删除 + 两处 docstring 清理(commit 8b0ff7e)
- **TODO**:拆文件 `pinocchio_hand.py` / `mujoco_hand.py`(见上方活跃 bugs 段)

### BUG-14 · 每 iter mj_jacBody 分配 jacp — 2/3 已完成
- **Part 1**:`_jacp_buf` 预分配 + `.copy()` 语义(commit 3271db4)
- **Part 2**:`mp_idx → body_id` 预解析 + `get_body_*_by_id` 方法(commit cbc7684),**+3.3% Manus fps**
- **TODO**:Part 3 FK-pass 合并(见上方活跃 bugs 段)

### BUG-15 · query_hand_penetration 25 geom × each iter — buffer done
- **Done**:`_pen_fromto_buf` 预分配 + 复用 `_jacp_buf`(commit e31531d),HO-Cap byte-identical
- **TODO**:active-set cache + AABB 粗筛 + mj_collision broadphase(见上方活跃 bugs 段)

## ❌ Rejected(1 bug · 负结果)

### BUG-16 · Delaunay adj_list 帧间 hash cache
- **实测结果**:Manus cache hit rate 45% / HO-Cap 36%(原 review 假设 "95% reuse" 不成立 — object 模式下 obj surface 样本浮动使 Delaunay 频繁换拓扑)
- **Benchmark**:Manus 100f 180→181ms,无改善(`get_adjacency_list` 本身 < 1ms,不是热点)
- **决定**:不 merge,避免无收益的维护面
- **Lesson**:性能 review 的收益预估需要实测 cache hit rate,尤其是缓存对象依赖输入分布时

---

## 状态字段(历史说明)

每项 bug 从 `open` → `in-progress` → `merged` / `rejected`。Partial-merged 用单独段写 done/todo。
