# EXP-12: 删除 warmup 的 TIP 位置 anchor

> 日期: 2026-04-22
> 状态: 采用(已合入主管线)
> 分支: `remove-warmup-tip-anchor`
> A/B 数据: `.worktrees/exp-tip-anchor-removal/experiments/tip_anchor_removal_exp/`

## 动机

`solve_angle_warmup` 在每指 4 条骨的方向 cosine IK 之外,额外有一项 **TIP 位置 anchor**:

```python
# 原代码(已删除)
w_tip = 100.0
res_tip = rob_tip - src_tip
residuals.append(np.sqrt(w_tip) * res_tip)
J_rows.append(np.sqrt(w_tip) * J_tip)
```

引入这一项的初衷是模仿 GMR 的 ankle 接地锚定(防止长链 IK 全局漂移)。但:

1. **几何上的冲突**:tip anchor 是**直接位置** cost,把 robot tip body 拉到 MediaPipe
   landmark(关节中心)的绝对位置。而 Laplacian 是**相对几何**,shift-invariant。
   两种 cost 对"landmark 相对物体的位置"敏感度不同。
2. **与 OmniRetarget 不一致**:OmniRetarget 只用 Laplacian-based cost(全管线
   shift-invariant),不需要 bias 就能跟硬约束共存。我们的 tip anchor 与非穿透硬
   约束会在"source landmark 穿进 mesh"时形成 7.5mm 量级的持续矛盾,之前代码里
   `CAPSULE_RADIUS=7.5mm` 的 tolerance 就是为了和解这一项。
3. **分析上的近 no-op**:`retargeter.py` docstring + Manus 上的梯度分解测得 tip 项
   对 q 的影响 ~1°,骨方向项 ~10°。J_dir 的 `1/bone_length ≈ 33×` 在 Hessian 里
   完全主导。HO-Cap 上未验证过。

本次实验:**在 Manus 和 HO-Cap 两端验证 tip anchor 是否真的是 no-op,再决定是否删除**。

## 方法

Worktree `.worktrees/exp-tip-anchor-removal` 加 config flag `use_warmup_tip_anchor`
(default True = 保留,False = 删除),gate 住那 8 行代码。

**A 组**:`use_warmup_tip_anchor=True`(baseline,当时行为)
**B 组**:`use_warmup_tip_anchor=False`(删除版本)

数据:
- Manus `manus1_5k.pkl` 全 5000 帧
- HO-Cap 3 个 clip(总 710 帧):`subject_1__20231025_165502__seg00`、
  `subject_3__20231024_161306__seg00`、`subject_3__20231024_162409__seg00`

**指标**:
- qpos 逐帧差(A vs B,degrees)—— max / RMS / mean
- 五指 tip Cartesian 误差 `||rob_tip - src_tip||`(mm)
- 反弓率(PIP/DIP < 0 的帧占比)
- 吞吐(fps)

## 结果

### Manus(固定基座,无物体,5000 帧)

| 指标 | A(有 tip anchor) | B(删除) | Δ |
|---|---:|---:|---:|
| fps | 368.3 | 385.4 | **+17.2**(删掉更快)|
| Tip 误差 mean (mm) | 15.38 | 15.96 | +0.58 |
| Tip 误差 p95 (mm) | 27.66 | 28.17 | +0.51 |
| 反弓率 | 98.7% | 98.7% | 0.00pp |

**qpos 差异**: max **1.89°**,RMS **0.34°**,mean 0.26°

### HO-Cap(floating base,手+物,3 clip)

| Clip | 帧数 | fps A | fps B | qpos RMS (°) | Tip err A | Tip err B | 反弓 A | 反弓 B |
|------|---|------:|------:|-------------:|----------:|----------:|-------:|-------:|
| subject_1/seg00 | 166 | 180.3 | 185.9 | 0.241 | 24.56mm | 25.01mm | 71.1% | **69.9%** |
| subject_3/seg00 | 297 | 176.8 | 174.8 | 0.238 | 26.85mm | 27.27mm | 97.3% | 97.3% |
| subject_3/seg00-2 | 247 | 178.2 | 170.1 | 0.240 | 26.99mm | 27.43mm | 99.6% | 99.6% |

- **qpos RMS 跨所有 HO-Cap clip 稳定在 0.24°**,比 Manus 更窄
- **Tip 误差**:增加 0.4-0.5mm,在 MediaPipe 模板误差 (cm 量级) 下可忽略
- **反弓率**:持平或微降(subject_1/seg00 **-1.2pp**)
- **fps**:持平

## 结论

✅ **安全删除**。核心数据:
- Manus qpos max 1.89° / RMS 0.34°,HO-Cap max < 0.8° / RMS 0.24°
- 所有指标无回归(> 2mm 或 > 5pp)
- 一个微小改善方向(反弓)

原 docstring 的 near-no-op 分析方向正确:**bone-direction 的 Hessian (1/bone_length² ~ 1000×) 主导,tip anchor 的 w_tip=100 在联合矩阵里被完全吃掉**。HO-Cap 的 floating base 把结果收得更紧(wrist 锁死减少了 tip anchor 的可达范围)。

## 删除 tip anchor 的后续意义

这清理为后续**硬约束 ablation** 铺了平:

1. **Warmup 变成纯骨方向对齐**,和 OmniRetarget 的 Laplacian cost 同属**shift-invariant**
   家族。整个管线 cost 对"landmark 相对物体的绝对位置"都不敏感。
2. **非穿透硬约束可以用严格 1mm slack**,不再需要 `CAPSULE_RADIUS=7.5mm` 去和解 tip
   anchor 带来的 cost-constraint 冲突。
3. **Ablation 诚实性**:下一步做"warmup-only / S2-only / both / off" 四组对比时,
   差异全部归因于硬约束,不再混入 tip anchor 与约束的交互。

## 改动

- `src/hand_retarget/retargeter.py`:
  - 删除 `w_tip = 100.0` 和整个 TIP position anchor 代码块(~12 行)
  - 更新 `solve_angle_warmup` docstring,指向本 doc
- `CLAUDE.md`:HO-Cap 注意事项段 + 实验历史表新增 EXP-12
- `doc/exp8_link_midpoint_joint_optimization.md`:标注历史上 tip anchor 的移除
- `tests/refactor_baseline.npz`、`tests/refactor_gate_baseline.npz`:重建
- `tests/test_refactor_baseline.py`:顺带修了一个 pre-existing 语法错误
  (`Path(_WUJI_SDK "x")` → `Path(_WUJI_SDK) / "x"`)

## 相关文件(worktree 内)

- `.worktrees/exp-tip-anchor-removal/experiments/exp_tip_anchor_removal.py` — A/B 驱动
- `.worktrees/exp-tip-anchor-removal/experiments/tip_anchor_removal_exp/REPORT.md` — 完整报告
- `.worktrees/exp-tip-anchor-removal/experiments/tip_anchor_removal_exp/summary.json` — 聚合指标

## 复现

```bash
cd /home/l/ws/RL/retargeting/.worktrees/exp-tip-anchor-removal
conda activate mjplgd
PYTHONPATH=src python experiments/exp_tip_anchor_removal.py --quick   # 200+100f
PYTHONPATH=src python experiments/exp_tip_anchor_removal.py            # 全量
```
