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

### GMR 风格 cosine IK (S1)

参考 GMR (General Motion Retargeting) 的两阶段设计:
- 对每条骨, 匹配 robot vs source 的方向 (单位向量差)
- 指尖锚点 (w=100) 防止全局漂移
- 收敛检测: `||q_new - q_old|| < 1e-3` 或 max 5 iter
- 91fps, PIP/DIP 0% 反弓 (天然, 无正则)

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
3. **收敛检测有效**: S1 平均 2-5 iter, S2 平均 2-6 iter, 无需固定迭代次数
4. **首帧 warmup 不需要**: S1 已做 warmup, n_iter_first=50 可统一为 n_iter

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
PYTHONPATH=src python demos/legacy/play_interaction_mesh.py \
    --link-midpoint --angle-warmup --semantic-weight

# HO-Cap 可视化
PYTHONPATH=src python demos/hocap/play_hocap.py \
    --link-midpoint --angle-warmup --semantic-weight \
    --clip hocap__subject_3__20231024_162409__seg00
```

## 局限和后续

1. **HO-Cap floating base**: 联合优化在 26 DOF floating base 下效果待优化 (参数需重新 tune)
2. **小拇指**: 排除 Laplacian 是 workaround, 根因是骨段比例差; 长期需要 per-bone scaling 或 angle-space-only 方案
3. **速度**: 联合优化 20fps vs baseline 30fps, 主要开销在 S1 的 FK + Jacobian
4. **MCP abduction**: S1 的 abd 提取精度有限 (palm plane 分解), 拇指对掌靠 S2 锚点补偿
