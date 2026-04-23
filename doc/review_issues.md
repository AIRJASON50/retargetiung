# Code Review Issues

5 位 expert(CS 架构 / Python / 数据结构 / 机器人学 / 优化理论)并行审查结果,2026-04-23。

| Legend | |
|---|---|
| ⚠️  | 涉及旋转/坐标系约定,改动必须谨慎校对 |
| 🔴 P0 | 正确性 bug,必修 |
| 🟠 P1 | 优化公式/数值行为,有风险 |
| 🟡 P2 | 架构工艺,中等改动 |
| 🟢 P3 | 性能,可测量 |
| 🔵 P4 | Python 工艺,小改动 |

## 状态字段

每项 bug 从 `open` → `in-progress` → `merged` / `rejected`。

---

## 🔴 P0 Correctness bugs

### BUG-01 · retargeter.py:1067 — 条件 import 造成 NameError 风险
**状态**: merged (commit 215fa18, 2026-04-23)
**文件**: `src/hand_retarget/retargeter.py:1067-1069`

`from wuji_retargeting.mediapipe import estimate_frame_from_hand_points` 在 `if np_any and _has_object:` 内部导入,后续 1093 行 `R_svd = estimate_frame_from_hand_points(...)` 再次使用时依赖同样的条件才能保证符号已定义。一旦两处 gate 逻辑不一致,`NameError`。

**建议**: 把 import 提到模块顶部(已在 mediapipe_io 或 retargeter 其他位置 import 过则复用)。

---

### BUG-02 · ⚠️ _align_frame vs retarget_hocap_sequence 物体对齐不一致
**状态**: open
**文件**:
- `src/hand_retarget/retargeter.py:_align_frame` SVD 分支(约 line 1182)
- `src/hand_retarget/retargeter.py:1093-1099` `retarget_hocap_sequence` 对象变换

`_align_frame` SVD 分支用 `np.linalg.lstsq(lm_centered[1:6], trans[1:6])` 无约束最小二乘反推 `R_t` 变换 object 点云 — 返回的不是正交矩阵。而 `retarget_hocap_sequence` 对同一物体用严格的 `R_align = R_svd @ OPERATOR2MANO`。两处变换**不一致时** IM 约束看到的 obj 与 `set_object_pose` 看到的几何漂移数度 → `query_hand_penetration` 几何 ≠ Laplacian 目标几何。

**建议**: `_align_frame` SVD 分支也改为 `R_align = estimate_frame_from_hand_points(lm_centered) @ OPERATOR2MANO`,与主循环共享同一 `R_align`,避免重复拟合。

**⚠️ 校对要点**:
- `OPERATOR2MANO_LEFT` vs `OPERATOR2MANO_RIGHT` 按 hand_side 分发
- 变换语义:点云是 `(obj_t - wrist) @ R_align` (右乘 passive) — 新旧两处要对齐
- 校验方式:跑 HO-Cap subject_3/161306 (bimanual),penetration metric 不应有 frame-level 跳变
- baseline 回归:`pytest tests/test_gate.py` + 重生成 `tests/refactor_gate_baseline.npz` 对比

---

### BUG-03 · DIP joint limit 允许反弓到 -27°
**状态**: open
**文件**:
- `assets/wujihand/xmls/wuji_lh_xml/left_mjx.xml:71,97,123,149,175` (DIP joint range 约 `-0.47 ~ 1.57`)
- `src/hand_retarget/retargeter.py` joint_limits 应用点

MuJoCo XML 允许 DIP 反弓 -27°,`activate_joint_limits=True` 也卡不住,CLAUDE.md 所述"HO-Cap 启用 S1 反弓率 97%"的硬件层根因。

**建议**: 新增 config 字段 `dip_nonnegative: bool = True`,在构造 `q_lb` 时原地改写 `q_lb[6+4f+3] = max(0.0, q_lb[6+4f+3])`(floating base offset +6)。注意 `4f+3` 是 DIP 索引(MCP_flex/MCP_abd/PIP/DIP 四联关节)。

---

## 🟠 P1 Optimization formulation

### BUG-04 · IM 权重量纲不一致 → λ_IM 默认被 cos 项 30× 压制
**状态**: open
**文件**: `src/hand_retarget/retargeter.py:290-292`, `src/hand_retarget/config.py:255`

Laplacian 残差以米为单位(~1e-3 m²),cos-IK bone dir 残差无量纲 O(1),smooth 是 rad²。`laplacian=5 : anchor_cosik=5 : smooth=1` 纯经验值,量纲不可比。EXP 验证 IM 被 cos 项 ~30× 主导。

**建议**: `interaction_mesh_length_scale` 默认值从 `None` 改为 `0.03`(平均骨长),使 IM 项变 `(1/L_char)² · ‖Δ‖² ≈ O(1)`。这个改动会使所有 baseline qpos 漂移 — 需要重建 baseline npz + EXP-13 完整 compare 验证不退化。

---

### BUG-05 · Gauss-Newton Hessian 无 Levenberg-Marquardt damping
**状态**: open
**文件**: `src/hand_retarget/retargeter.py:306, 435, 440`

`H = JᵀJ + w_s·I + 1e-12·I`:1e-12 只防奇异,不是信赖域。cos-IK 项 `(I-ddᵀ)/‖e‖` 的 null-space 方向(沿 d_rob)会使 `JᵀJ` 奇异,完全依赖 `w_s` 偶然填充。

**建议**: 加自适应 LM:`H ← JᵀJ + w_s·I + μ·diag(JᵀJ)`,按 Marquardt 规则随成功/失败步调 μ(cost 降 → `/=2`,升 → `*=10`)。可合并入当前 `_solve_qp_trust_shrink`。

---

### BUG-06 · 源 landmark 穿透时非穿透硬约束不可行
**状态**: open
**文件**: `src/hand_retarget/retargeter.py:757-803` `_build_penetration_constraints`

source landmark 本身穿进 object mesh 时 `h = φ + tol` 永远 < 0 → QP 不可行 → 3 次 trust shrink 耗尽 → stall。此外 `mj_geomDistance` 的 `∂φ/∂q` 在 capsule-mesh 三角面切换处 C⁰ 不连续,线性化假设会抖。

**建议**:
1. 改混合 penalty:`φ < -5mm` 触发硬约束,`-5mm ≤ φ < 0` 用 `max(0, -φ-tol)²` 软惩罚,缓冲 source 本身的穿透
2. 对接触法向做 EMA 平滑(前一 iter 混合 0.5)减缓 switching 抖动

---

## 🟡 P2 Architecture

### BUG-07 · InteractionMeshHandRetargeter 是 God Class(1282 行,20+ 方法)
**状态**: open
**文件**: `src/hand_retarget/retargeter.py:88-1282`

同一个类里 QP 构造 / 拓扑构造 / 预处理对齐 / HO-Cap 时序驱动 / 体名映射 / 非穿透约束 / telemetry 全塞一起。`retarget_hocap_sequence` 140 行控制流应从类剥离。

**建议**: 拆 `BoneDirectionCost` / `LaplacianCost` / `PenetrationConstraint` / `QPTrustShrinkSolver` 四个策略类,retargeter 只做 orchestrator;`retarget_hocap_sequence` 移到 `scripts/retarget_hocap.py`。

---

### BUG-08 · HandRetargetConfig 字段膨胀 + 互斥字段共存
**状态**: open
**文件**: `src/hand_retarget/config.py:194-303`

30 个平铺字段(CLAUDE.md 已过时写 "17 个")。`anchor_mode` / `angle_anchor_weight` / `anchor_cosik_weight` 三者互斥又共存(l2 vs cosik_live)。

**建议**: 拆成嵌套 `OptimizerConfig` / `PenetrationConfig` / `IMTopologyConfig` / `AnchorConfig` / `PreprocessConfig`;用 sealed union(`Anchor = L2Anchor | CosikLiveAnchor`)替换互斥字段;同步 CLAUDE.md 字段描述。

---

### BUG-09 · ⚠️ PinocchioHandModel 伪装在 mujoco_hand.py
**状态**: partially merged (commit 8b0ff7e, 2026-04-23) — alias + 两处 docstring 已清理;拆文件 pinocchio_hand.py / mujoco_hand.py 留大重构
**文件**: `src/hand_retarget/mujoco_hand.py`

文件名是 `mujoco_hand.py`,`PinocchioHandModel`(595 行)是 Pinocchio 实现。`MuJoCoHandModel = PinocchioHandModel` alias 无实际 MuJoCo 内核,只是别名。`MuJoCoFloatingHandModel` 才是真 MuJoCo 后端。文件级 docstring 也把 `MuJoCoHandModel` 写入 Protocol 说明。

**建议**: 拆 `pinocchio_hand.py` + `mujoco_hand.py`,删除 `MuJoCoHandModel` alias,修正 docstring。

**⚠️ 校对要点**: Pinocchio 的 Jacobian frame 约定(`pin.LOCAL` / `pin.LOCAL_WORLD_ALIGNED`)别混。拆文件时确认所有调用方 import 能更新。

---

### BUG-10 · 核心库对 scene_builder 私有符号依赖
**状态**: open
**文件**: `src/hand_retarget/mujoco_hand.py:191,305`

`MuJoCoFloatingHandModel.inject_object_mesh` 直接 `from scene_builder.hand_builder import _inject_fingertip_sites, _inject_wrist_frame, _inject_wrist6dof_mode` — 跨模块引用下划线私有符号,破坏"核心库无 viewer 依赖"承诺(`hand_retarget_viz/__init__.py:2`)。

**建议**: `scene_builder` 暴露 public API `build_scene_with_object(scene_xml, mesh_path, hand_side, add_wrist6dof=True)`,或把 `MuJoCoFloatingHandModel` 下沉到 `scene_builder/` 下。

---

### BUG-11 · 无 pyproject.toml / requirements.txt,依赖散落
**状态**: merged — part 1: pyproject.toml (ac082f3); part 2: tests/conftest.py + sys.path cleanup (94e63a1). script-mode entrypoints(scripts/experiments/demos)仍保留 `WUJI_SDK_PATH` 默认回退一行,conda env 已 resolvable,暂留
**文件**: 仓库根(缺失)

所有脚本通过 `sys.path.insert` 加 `/home/l/ws/doc/WujiRepo/...` 硬编码 WUJI_SDK_PATH。依赖(`qpsolvers[daqp]`, `pinocchio`, `mujoco`, `trimesh`, `tqdm`, `pyyaml`, `scipy`) 只在 import 里。

**建议**: 新建 `pyproject.toml` 声明 `[project] dependencies` + `[project.optional-dependencies.dev]`;`tests/conftest.py` 统一注入 `WUJI_SDK_PATH` env;把硬编码路径改为"env 必需 + 明确报错"。

---

## 🟢 P3 Performance (预计合并可再提 20-40% fps)

### BUG-12 · QP Hessian 走稠密,丢失 sparse kron 结构
**状态**: open
**文件**: `src/hand_retarget/retargeter.py:298`

`J_L = L_sp ⊗ I3` 是 CSR(nnz~5%),但立刻 `.toarray()`,`np.diag(sqrt_w3)` 3V×3V 稠密,路径 O(V²·nq)。

**建议**: `J_w = sqrt_w3[:, None] * J_L` 保持 sparse(O(nnz)),再 `(J_w.T @ J_w).toarray()`(daqp 要稠密 H 无解,但 J^T J 稀疏乘再 densify 比稠密快一个数量级)。

---

### BUG-13 · Laplacian 矩阵每帧每 iter 从头 np.zeros((N,N))
**状态**: open
**文件**: `src/hand_retarget/mesh_utils.py:139-158`

每 S2 iter 调 `calculate_laplacian_matrix` — Python for 循环填稠密 N×N,每帧数千次。

**建议**:
1. 预建稀疏 COO 索引模板(骨架拓扑模式帧间不变)
2. 每 iter 只更新 `data` 数组
3. 直接返回 `csr_matrix` 跳过 `sp.csr_matrix(L)` 转换
4. `calculate_laplacian_coordinates` 复用同一矩阵

预计从 ~0.5ms 降到 <50μs。

---

### BUG-14 · 每 iter 对 21 body 调 mj_jacBody + 分配 jacp buffer
**状态**: partially merged (commit 3271db4, 2026-04-23) — jacp buffer 预分配 + .copy() 语义保留 byte-identical;预解析 mp_idx→body_id、FK-pass 合并待后续
**文件**: `src/hand_retarget/retargeter.py:1258-1282` (`_get_robot_keypoints`/`_get_robot_jacobians`), `_mp_body_pos_jacp`

每次都 `self._body_ids[name]` 查字典 + `np.zeros((3, nv))` 新分配,每 bone × 每 iter × 每帧。

**建议**:
1. `__init__` 里预解析 `mp_idx → body_id` 数组
2. 预分配 `self._jacp_buf = np.zeros((3, nv))`,MuJoCo API 支持 out-buffer
3. `_get_robot_keypoints` + `_get_robot_jacobians` 合并为一次 FK-pass

---

### BUG-15 · query_hand_penetration 25 geom × 每 iter Python for loop
**状态**: open
**文件**: `src/hand_retarget/mujoco_hand.py:431-499`

HO-Cap 每帧 ~200 次 `mj_geomDistance` + 200 次 `mj_jac`,实测 ≤3 pair 真正 `φ<threshold`。Python 开销是瓶颈,不是 MuJoCo 调用。

**建议**:
1. active-set 缓存:上一 iter 命中的 geom idx 优先查
2. AABB 粗筛(`model.geom_aabb` + obj aabb)淘汰 ≥80%
3. 预分配 `fromto`、`jacp`、`results` buffer
4. 用 `mj_collision` 原生 broadphase(C-level)

---

### BUG-16 · Delaunay 每帧重算,但 HO-Cap 骨长模板固定
**状态**: open
**文件**: `src/hand_retarget/retargeter.py:727`(`create_interaction_mesh` 调用)

MediaPipe 骨长在 HO-Cap 9 个被试间完全相同(模板),相邻帧拓扑 <5% 变化,却每帧调 scipy.spatial.Delaunay(~1ms)。

**建议**: `hash(simplices.tobytes())` 对比相邻帧,相同则复用 adj_list;若启用 cache,`edge_list` 同步缓存,`filter_adjacency_by_distance` 只对变化边重算。

---

## 🔵 P4 Python hygiene

### BUG-17 · config.py 类型注解错误
**状态**: merged (commit f59ebec, 2026-04-23)
**文件**: `src/hand_retarget/config.py:303` 和 (可能) `mediapipe_rotation: dict`

`exclude_fingers_from_laplacian: list = None` — 裸 `list`(丢泛型参数) + 非 Optional 字段赋 None。

**建议**:
```python
exclude_fingers_from_laplacian: list[int] | None = None
mediapipe_rotation: dict[str, float] = field(default_factory=lambda: {...})
```

---

### BUG-18 · mj_name2id 吞异常
**状态**: merged (commit 4be173d, 2026-04-23)
**文件**: `src/hand_retarget/mujoco_hand.py:530-536`

```python
try:
    bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name)
except Exception:
    bid = -1
```
MuJoCo `mj_name2id` 正常路径是**返回 -1 不抛异常**,这里 `except Exception` 掩盖真实错误。

**建议**: 删 try/except,直接用返回值 `< 0` 判空。

---

### BUG-19 · 测试覆盖缺口
**状态**: partially merged — test_config.py (5e7d210, 7 tests), test_mesh_utils.py (3d1df17, 8 tests), pytest-ify test_refactor_baseline (8aa2d8e, 1 test);test_hocap_smoke.py 基本冗余(strengthened gate 7a75c1b 已覆盖 HO-Cap),可关闭
**文件**: `tests/`

仅 `test_gate.py`(5 个 assert)+ 脚本式 `test_refactor_baseline.py`(非 pytest)。

**建议新增**:
- `test_mesh_utils.py`:Delaunay 邻接 + Laplacian 数值对小拓扑
- `test_config.py`:`from_yaml` 所有字段映射、override、unknown key
- `test_hocap_smoke.py`:10 帧端到端,assert qpos 非 NaN
- `test_refactor_baseline.py` 改为 pytest(`pytest.skip` 判 baseline 缺失)

---

### BUG-20 · ⚠️ Jacobian 用 pin.LOCAL 手动旋回 world
**状态**: merged (commit f74c069, 2026-04-23) — 改用 pin.LOCAL_WORLD_ALIGNED;ULP 级(~7.77e-16 rad)数值噪声,所有 gate 远过
**文件**: `src/hand_retarget/mujoco_hand.py:141-143`(`get_body_jacp`)

`pin.getFrameJacobian(model, data, fid, pin.LOCAL)` 然后 `R @ J_local[:3]` — 等价于 `pin.LOCAL_WORLD_ALIGNED` 前 3 行。手动乘 `R` 有漏 `updateFramePlacements` 的风险。

**建议**: 换 `pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)[:3]`,单行替代。

**⚠️ 校对要点**: 等价替换,但要确认调用方没有假设返回的是 LOCAL 坐标系。跑 test_gate 对比 qpos 逐 iter 数值。

---

## 执行顺序(建议)

**第一波(低风险,先修)**: BUG-01, BUG-17, BUG-18 → 基础清理,无数值影响
**第二波(测试底座)**: BUG-19 → 给后续重构打底
**第三波(⚠️ 旋转相关,单独校对)**: BUG-02, BUG-20, BUG-09 → 每个单独 worktree + 独立 review
**第四波(硬件/约束层)**: BUG-03, BUG-06
**第五波(架构重构,大手术)**: BUG-08 → BUG-07 → BUG-10 → BUG-11
**第六波(性能)**: BUG-12, BUG-13, BUG-14, BUG-15, BUG-16(每个独立 benchmark)
**第七波(数值公式)**: BUG-04, BUG-05 — 需 baseline 重建 + EXP-13 完整 compare
