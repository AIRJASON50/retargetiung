# EXP-13: 非穿透硬约束 Ablation(warmup × S2)

> 日期:2026-04-22
> 分支:`exp/penetration-constraint`
> 状态:**坐标系 gap 修好**,internal 和 viz model 完全一致,ablation 信号跨 3 clip 稳定

## 设计

非穿透硬约束按 OmniRetarget 思路以"**管线不变量**"形式加入(和 joint limits / smooth 同等地位):每 iter 重查 `mj_geomDistance`,把 `phi + J_contact·dq ≥ -tol` 线性化成 QP 不等式。

- **范围**:全手 × object(25 collision geom:5 palm boxes + 20 finger capsules),不做 self-collision
- **Tolerance**:1mm 纯 SQP slack
- **Jacobian**:在接触最近点取
- **Infeasibility**:trust-region 缩小 3 次重试 → stall 记 struct_infeas

## 关键修复:object pose 坐标系对齐

之前 `set_object_pose(obj_t - wrist_world, obj_q)` 用**世界帧差分**,retargeter 内部 model 的 hand 是在**MANO 对齐帧**,两者差一个 `R_align` 旋转。导致:
- Internal 测到的 pen 反映"错位 object 下的几何" — 看起来 C 模式降到 1mm
- 但 viz model 里 object 在真实世界位置,hand 和它的相对几何**几乎没变** → 用户视觉几乎看不到改善

**修复**(和 `play_hocap.py:qpos_to_world` 反向一致):
```python
R_svd = estimate_frame_from_hand_points(lm_centered)
R_align = R_svd @ OPERATOR2MANO              # R_inv = R_align.T
obj_center_aligned = (obj_t - wrist) @ R_align
R_obj_aligned = R_align.T @ R_obj_world
```

验证:internal pen 和 viz pen **逐 mm 一致**(e.g., subject_3/161306 C:internal 1.25mm,viz 1.25mm)。

## 4 组对比

| 组 | Warmup | S2 |
|---|---|---|
| D | 无 | 无(baseline,object 不 inject,行为 = main,max diff 0.000000°)|
| A | 硬 | 无 |
| B | 无 | 硬 |
| C | 硬 | 硬(推荐) |

## 结果(viz model,用户看到的真实 pen)

| Clip | Hand | Mode | T | pen_max (mm) | pen_mean | 帧≥0.5mm |
|---|---|---|--:|--:|--:|--:|
| subject_1/seg00        | left     | **D** | 166 | **50.21** | 31.04 | 118 |
| subject_1/seg00        | left     | A     | 166 | 50.13 | 31.21 | 120 |
| subject_1/seg00        | left     | B     | 166 | 50.14 | 30.55 | 120 |
| subject_1/seg00        | left     | **C** | 166 | **34.25** | **15.40** | 120 |
| subject_3/161306/seg00 | bimanual | **D** | 297×2 | **31.64** | 14.08 | 201 |
| subject_3/161306/seg00 | bimanual | A     | 297×2 | 26.59 | 13.89 | 207 |
| subject_3/161306/seg00 | bimanual | B     | 297×2 | 26.59 | 6.41 | 207 |
| subject_3/161306/seg00 | bimanual | **C** | 297×2 | **1.25** | **0.70** | 207 |
| subject_3/162409/seg00 | bimanual | **D** | 247×2 | **48.19** | 24.93 | 170 |
| subject_3/162409/seg00 | bimanual | A     | 247×2 | 46.62 | 27.09 | 171 |
| subject_3/162409/seg00 | bimanual | B     | 247×2 | 46.75 | 26.52 | 171 |
| subject_3/162409/seg00 | bimanual | **C** | 247×2 | **14.64** | **3.10** | 171 |

## 三个关键信号

1. **C 模式普遍最好**,跨不同 clip 质量:
   - subject_3/161306:pen_max 31.6 → **1.25mm**(-96%)
   - subject_3/162409:pen_max 48.2 → **14.6mm**(-70%)
   - subject_1/seg00:pen_max 50.2 → **34.3mm**(-32%)
2. **A 单 warmup** = D 水平:max 几乎不动,warmup 的努力被 S2 Laplacian 抹掉
3. **B 单 S2** 改善 mean,但 **max 不动**:SQP 一阶段 5mm/iter × 10 iter 不够从 30-50mm 深度退出
4. **C 协同是非线性**:warmup 先推出深穿透,S2 收敛到 tolerance

## subject_1 / subject_3/162409 C 残留的原因

subject_1 C 仍 34mm,subject_3/162409 C 仍 15mm。原因:
- MediaPipe landmark 本身嵌进 mesh 深度很大(subject_1 D 阶段 50mm)
- SQP 单 iter ~5mm Cartesian,10-20 iter 累计 ~100mm 但 cost 同时往 source 拉
- 这是 pipeline 极限

**Layer 0 预处理**(投影 source landmark 出 mesh + tip_radius)能补齐这 30-50mm 剩余。

## 代码变更(只改动了穿透约束 + 必要的坐标系对齐)

| 文件 | 改动 |
|---|---|
| `src/hand_retarget/config.py` | `activate_non_penetration_warmup/_s2`(替换单一 bool)、`penetration_tolerance=1e-3`、`penetration_max_trust_shrinks=3`;make_stamp 加 np-* 标记 |
| `src/hand_retarget/mujoco_hand.py` | 新 `query_hand_penetration()` + `_collect_hand_col_geoms()`(filter: hand body tree ∩ contype!=0 ∩ type!=MESH → 25 geom) |
| `src/hand_retarget/retargeter.py` | 新 `_build_penetration_constraints()` / `_solve_qp_trust_shrink()`;warmup + S2 的 QP 走 helpers;per-frame `_frame_np_metrics` telemetry;**set_object_pose 应用 R_align(匹配 play_hocap 的 R_inv 公式)对齐 internal 和 viz 坐标系** |
| `config/hocap.yaml` | 加 `activate_non_penetration_*` + `penetration_tolerance` |
| `demos/hocap/play_hocap.py` | `--np-mode {off,warmup,s2,both}` shortcut |
| `experiments/exp_penetration_ablation.py` | 4 模式 × N clip 驱动;cache 写 `R_inv = (R_svd @ OPERATOR2MANO).T`(和 play_hocap `retarget_hand` 一致);bimanual clip 两手都 retarget |

**D 模式 qpos diff vs main: 0.000000°**(无 regression)。

## 可视化

```bash
cd /home/l/ws/RL/retargeting/.worktrees/exp-penetration-constraint
conda activate mjplgd

# subject_3/161306 最强信号 pen 31mm → 1.25mm
PYTHONPATH=src python demos/hocap/play_hocap.py \
    --clip hocap__subject_3__20231024_161306__seg00 --np-mode off
PYTHONPATH=src python demos/hocap/play_hocap.py \
    --clip hocap__subject_3__20231024_161306__seg00 --np-mode both

# subject_3/162409 之前坐标 gap 的那个 clip,48 → 14.6mm
PYTHONPATH=src python demos/hocap/play_hocap.py \
    --clip hocap__subject_3__20231024_162409__seg00 --np-mode off
PYTHONPATH=src python demos/hocap/play_hocap.py \
    --clip hocap__subject_3__20231024_162409__seg00 --np-mode both
```

窗口 MuJoCo 默认开 `mjVIS_CONTACTPOINT` + `mjVIS_CONTACTFORCE`,对比 D 和 C 的接触点深度一目了然。

## 结论

**采用 C 模式**(warmup + S2 都开硬约束)。在 raw 对齐本来就撞对的 clip 上 pen 降到 1mm,在其他 clip 上降 32-70%。剩余穿透源于 source landmark 深度嵌入 mesh,需要 Layer 0 预处理才能补齐。

## 复现

```bash
cd /home/l/ws/RL/retargeting/.worktrees/exp-penetration-constraint
conda activate mjplgd
PYTHONPATH=src python experiments/exp_penetration_ablation.py --quick   # 100f × subject_1
PYTHONPATH=src python experiments/exp_penetration_ablation.py            # 全 3 clip
```
