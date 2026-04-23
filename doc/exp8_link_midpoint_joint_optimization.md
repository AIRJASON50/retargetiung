# EXP-8: Link Midpoint + 联合角度/IM 优化

> 日期: 2026-04-16 ~ 2026-04-17
> 状态: 已合入主分支

## 动机

1. 手部接触发生在 link 表面而非关节中心 — joint origin 不代表接触位置
2. OmniRetarget 全身用 15 点, 手部 21 点太密导致 Delaunay 跨指噪声边
3. 骨段比例差异 (拇指 0.83x, 小拇指 1.50x) 导致 Laplacian 反弓
4. 角度空间天然无尺度, 可消除骨段比例影响

## 方法

### Keypoint 改进: 20 link 中点

每指 4 点: 3 个相邻关节中点 + 1 个 TIP (保留)
- Source: `mid = (landmarks[parent] + landmarks[child]) / 2`
- Robot: `mid = (FK(parent_body) + FK(child_body)) / 2`, Jacobian = `(J_p + J_c) / 2`
- Wrist 排除 (已对齐, loss=0)

### GMR-inspired cosine IK (S1)

两阶段设计理念借鉴 GMR, 但实现差异明显:
- GMR 使用 `mink.FrameTask`(SO(3) SE3 任务),对每个 body 的 pos/rot 分别设权重,
  Stage 1 里绝大多数 body `pos_weight=0`,只对需要接地的 anchor(如 `ankle_roll_link`)
  给非零 pos_weight。
- 本实现手写 Gauss-Newton QP: 每条骨匹配 **单位向量差** `||d_rob - d_src||²`(尺度不变)。
- **历史**: 原实现还附带指尖位置项 `||p_rob - p_src||²`(w_tip=100),
  模仿 GMR 的 ankle 接地锚定。但 HO-Cap 验证为 near-no-op(Manus max 1.89°,
  HO-Cap RMS 0.24°),已于 **EXP-12 移除**。详见 `doc/exp_warmup_tip_anchor_removal.md`。
- 收敛检测: `||q_new - q_old|| < warmup_convergence_delta` (默认 1e-3)
- 外循环上限: 首帧 `angle_warmup_iters_first=20`(远离默认姿态,需要更大预算),
  非首帧 `angle_warmup_iters=5`(warm-start,2-3 iter 就收敛)
- 91fps, PIP/DIP 0% 反弓 (Manus 上天然; HO-Cap 上因骨长比差 0.44-0.69× 仍会反弓)

**历史重构**: 原实现在 warmup 外循环后又调用 `_extract_cosik_targets` 多跑 3 iter cosine IK
产出 S2 anchor 的 `q_target`。实测 99.4% 帧里这 3 iter 第 1 次就 break 且 drift <
1e-4 rad(数值噪声),纯冗余。已删除,`q_target` 直接使用 warmup 收敛结果;唯一真正需要
多迭代的首帧通过 `iters_first=20` 覆盖。

### 联合 cost (非两阶段)

GMR 的顺序两阶段在手部失败 — S2 (Laplacian) 会完全覆盖 S1 (角度) 的结果 (11228 flips / 100000)。原因: 骨段比例差导致 Laplacian 最优解和角度最优解差距大。

改为联合优化:
```
cost = laplacian_weight * ||L_robot - L_source||²
     + angle_anchor_weight * ||q - q_angle_target||²  (角度锚定)
     + smooth_weight * ||dq - dq_smooth||²
```
角度锚定持续存在于每次 SQP 迭代, 不会被 Laplacian 覆盖。

### 小拇指排除

骨段比例差 1.5x → Laplacian 在小拇指上产生最严重的反弓。
解决: 小拇指 4 个 midpoint 的 Jacobian 置零 → Laplacian 对小拇指 q 无梯度。
小拇指角度完全由 angle_anchor 控制, 但位置仍参与 Delaunay 拓扑影响其他手指。

## Tuning 过程

### S1:S2 比例扫描 (manus 500 frames, anchor=5 fixed, 小拇指排除)

| lap_w | tip_all | Thumb | Index | Ring | Pinky | pk_pip | pk_dip | fps |
|-------|---------|-------|-------|------|-------|--------|--------|-----|
| 1 | 16.7 | 17.3 | 12.8 | -- | 24.9 | 0.0% | 4.8% | 24 |
| 5 | 16.6 | 17.2 | 12.7 | -- | 24.9 | 0.0% | 4.8% | 24 |
| 50 | 16.2 | 16.9 | 12.0 | 14.1 | 24.9 | 0.0% | 4.8% | 23 |
| 200 | 15.6 | 16.7 | 10.9 | 12.6 | 24.9 | 0.0% | 4.8% | 20 |
| 500 | 15.0 | 16.8 | 9.8 | 11.5 | 24.9 | 0.0% | 4.8% | 15 |
| S1 solo | 16.8 | 17.4 | 12.9 | 15.3 | 25.0 | 0.0% | 4.8% | 91 |

Pinky 完全不受影响 (排除有效), 其他手指 tip error 随 lap_w 增大而下降。

### 关键发现

1. **角度锚定 vs 两阶段**: 联合 cost 的 flips 仅 20-21 (vs 顺序两阶段 6880-11228)
2. **S2 drift < 0.3°**: 联合优化下 Laplacian 只做微调, 不破坏角度对齐
3. **收敛检测有效**: S1 平均 2-3 iter, S2 平均 2-3 iter; cap 几乎永远不触发
4. **首帧需要更大 warmup 预算**: 起点是 default pose,距离 source 远,默认 5 iter 不足
   (实测 ||Δq|| 在 iter 5 仍 ≈ 0.32 rad)。已通过 `angle_warmup_iters_first=20` 吸收。
   S2 的 `n_iter_first` 同时降到 20(原 50/200 皆为 dead code)。

## 配置

```python
config.use_link_midpoints = True
config.use_angle_warmup = True
config.angle_anchor_weight = 5.0         # 角度锚定权重
config.exclude_fingers_from_laplacian = [4]  # 小拇指排除
# laplacian_weight 通过 retargeter.laplacian_weight 设置 (默认 10, 推荐 200 for midpoint)
# smooth_weight = 1.0 (联合优化模式)
```

## 复现

```bash
# Manus 可视化
PYTHONPATH=src python demos/manus.py \
    --link-midpoint --angle-warmup --semantic-weight

# HO-Cap 可视化
PYTHONPATH=src python demos/hocap.py \
    --link-midpoint --angle-warmup --semantic-weight \
    --clip hocap__subject_3__20231024_162409__seg00
```

## 局限和后续

1. **HO-Cap floating base**: 联合优化在 26 DOF floating base 下效果待优化 (参数需重新 tune)
2. **小拇指**: 排除 Laplacian 是 workaround, 根因是骨段比例差; 长期需要 per-bone scaling 或 angle-space-only 方案
3. **速度**: 联合优化 20fps vs baseline 30fps, 主要开销在 S1 的 FK + Jacobian
4. **MCP abduction**: S1 的 abd 提取精度有限 (palm plane 分解), 拇指对掌靠 S2 锚点补偿

## EXP-11: cosik_live 切为 S2 anchor 默认

> 日期: 2026-04-22
> 状态: 已采用为默认

### 动机

旧默认 `anchor_mode="l2"` 在 S2 cost 中用 `w_a · ‖q − q_S1‖²` 锚定 warmup
输出,本质是 **bilevel 近似**:先跑 warmup 得 q_S1,再用 L2 球钉住。这种设
计有两个暗坑:
1. `q_S1` 未必是 cos-IK 的真实极小点(warmup 会被 smooth 项往 q_prev 拉,
   平衡位置有残余 `‖r_bones‖ ≈ 0.57`)。L2 球的中心点有偏差。
2. L2 Hessian `diag(confidence)` 各向同性,凡 q 方向都同等惩罚。为了让 IM
   在 MCP 外展这种"骨方向不敏感"的方向能动,必须手工设 `confidence=0.5`。

`anchor_mode="cosik_live"` 是**单层联合优化**:
```
cost = λ_IM · IM(q) + w_rot · Σ_bones ‖d_rob(q) − d_src‖² + smooth
```
cos-IK cost 在每次 S2 iter 基于 current q 重算,Jacobian `J^T J` 的各向
异性让骨方向约束**自适应** —— 敏感方向墙壁陡峭、不敏感方向墙壁平缓。

### 数据(HO-Cap `subject_1_165502_seg00`, 166 帧, 扫 `w_rot ∈ {5, 1, 0.5}`)


| 指标 | A (l2, w=5) | B (cosik_live, w=5) | B (w=1) | B (w=0.5) |
|------|------------|---------------------|---------|-----------|
| tip_err (mm) | 24.56 | 25.06 | 25.01 | 24.97 |
| bone_cos | 0.0077 | 0.0075 | 0.0078 | 0.0082 |
| DIP 反弓率 | 71.1% | 70.5% | 69.9% | **67.5%** |
| obj tip-距离误差 (mm) | 11.39 | 11.71 | 11.64 | 11.59 |
| pinch 距离误差 (mm) | 6.63 | 6.68 | 6.60 | **6.58** |
| ‖q − q_A‖ (rad) | 0 | 0.031 | 0.053 | 0.101 |
| fps | 197 | 174 | 170 | 167 |

### 结论

- **手形保持 OK**:bone_cos、tip_err、obj-distance 误差都在噪声范围内
- **轻微改善**:DIP 反弓率随 w_rot 下降最多降 3.6pp,pinch 误差降 0.05mm
- **吞吐 -15%**:每 S2 iter 多一次 FK + 20 骨 Jacobian
- **架构收益(主因)**:
  1. 消除 `confidence[MCP_abd] = 0.5` 的手调(由 `J^T J` 结构化处理)
  2. 权重语义统一:warmup 和 S2 都用 w_rot,含义相同
  3. 单层优化,消除 bilevel 蛋白质

### 配置

```python
cfg.anchor_mode = "cosik_live"   # 默认
cfg.anchor_cosik_weight = 5.0    # 与 warmup 同量纲
# 老配置(l2)仍保留,需要时显式切:
# cfg.anchor_mode = "l2"
# cfg.angle_anchor_weight = 5.0
```
