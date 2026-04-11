# EXP-2 Results: Distance Weight vs Uniform Weight

**Date**: 2026-04-11
**Data**: manus1_5k.pkl, 500 frames, 156Hz, left hand
**条件**: 固定拓扑 (首帧 Delaunay), 三种方案对比

## Summary

| Metric | Baseline | IM Uniform | IM Distance |
|--------|----------|------------|-------------|
| Tip Pos Error (mm) | **12.0** | 14.8 | 20.1 |
| Tip Dir Error (deg) | **10.8** | 20.9 | 20.3 |
| Inter-Tip Dist Error (mm) | **6.3** | 6.8 | 11.5 |
| Jerk (rad/s^3) | 4568 | 4873 | **3536** |
| Temporal Consistency | 0.032 | 0.032 | **0.029** |
| C-Space Coverage | 0.509 | 0.570 | 0.534 |
| Speed (fps) | 292 | 64 | 33 |

## Per-Finger Tip Position Error (mm)

| Finger | Baseline | IM Uniform | IM Distance |
|--------|----------|------------|-------------|
| Thumb | 14.0 | 15.8 | **15.6** |
| Index | **7.0** | 15.1 | 21.1 |
| Middle | **9.5** | 16.2 | 21.7 |
| Ring | **8.6** | 13.0 | 19.0 |
| Pinky | **20.9** | **14.1** | 23.1 |

## Key Findings

1. **距离权重让位置精度变差了**: 20.1mm vs uniform 的 14.8mm. 原因分析见下.

2. **距离权重让平滑性变好了**: jerk 从 4873 降到 3536 (减 27%), 比 baseline 还好.
   temporal consistency 也更好 (0.029 vs 0.032).

3. **方向误差几乎没变**: 20.3° vs 20.9°. 距离权重对方向的影响不大.

4. **拇指改善, 其他变差**: 拇指位置误差从 15.8→15.6 (微改善),
   但 index/middle/ring 都恶化 5-8mm.

5. **速度降了一半**: 33fps vs 64fps. 距离权重的 L 矩阵非稀疏, 求解更慢.

## 分析: 为什么位置精度变差

距离权重让近邻权重极端化:
- 对指时食指尖→拇指尖权重 ~94%, 其他邻居 ~6%
- L 矩阵接近退化 (几乎只看最近邻)
- **全局形状约束丢失**: 手指链上 link1→link3→link4→tip 的相邻约束被压制
- 优化过度聚焦于最近邻关系, 忽略了骨架结构

这和之前预测的风险一致: "L 矩阵退化成只看最近邻 → 全局形状约束丢失"

## Conclusion

距离权重在 robot_only 下不适用:
- 位置精度显著恶化
- 平滑性改善 (但 uniform 已经足够好)
- 速度减半

Ho 2010 的距离权重设计是为有物体的场景 — 物体表面点作为密集锚点,
距离权重让接触面的贡献更大. 在纯手部 (无物体锚点) 下, 距离权重
反而让骨架内部的结构约束退化.

**下一步**: 不采用距离权重. 尝试 ReConForM 的 W 矩阵方式 — 不改 L 矩阵内部权重,
而是在 loss 外部对不同点加权. 这样保持 L 矩阵的全局形状约束,
同时在接触区域叠加额外的描述符约束.
