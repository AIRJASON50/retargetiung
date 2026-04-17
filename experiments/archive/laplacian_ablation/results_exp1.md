# EXP-1 Results: Fixed Topology vs Per-Frame Delaunay

**Date**: 2026-04-11
**Data**: manus1_5k.pkl, 500 frames, 156Hz, left hand

## Summary

| Metric | Baseline | IM PerFrame | IM Fixed |
|--------|----------|-------------|----------|
| Tip Pos Error (mm) | **11.98** | 14.50 | 14.82 |
| Tip Dir Error (deg) | **10.83** | 23.61 | **20.94** |
| Inter-Tip Dist Error (mm) | **6.34** | 8.01 | **6.84** |
| Jerk (rad/s^3) | **4568** | 8492 | **4873** |
| Temporal Consistency | **0.032** | 0.033 | **0.032** |
| C-Space Coverage | 0.509 | 0.568 | 0.570 |
| Joint Limit Violation | 0% | 0% | 0% |

## Key Findings

1. **Baseline wins on position accuracy**: 12mm vs 14-15mm. Interaction mesh 在 robot_only 下确实退化为劣化版 IK。

2. **Fixed topology (C) 优于 per-frame (B)**:
   - 方向误差: 20.9° vs 23.6° (减少 11%)
   - 指间距离误差: 6.84 vs 8.01 mm (减少 15%)
   - Jerk: 4873 vs 8492 (减少 43% — 大幅改善)
   - 时间一致性: 0.032 vs 0.033 (接近 baseline)
   
   **Ho 2010 的建议在手部场景下被验证**: 固定拓扑确实减少漂移和抖动。

3. **Per-frame Delaunay 的抖动问题严重**: jerk 是 baseline 的 1.86 倍，固定拓扑后降到 1.07 倍。
   拓扑帧间跳变是抖动的主要来源。

4. **C-Space Coverage 两种 IM 都优于 baseline** (0.57 vs 0.51):
   Interaction mesh 使用了更多的关节范围。但这不一定是优势——可能意味着更多不自然的关节角。

## Conclusion

- 固定拓扑明显优于逐帧重建，与 Ho 2010 的预测一致
- 但两种 IM 方案都不如 baseline 的位置精度
- 固定拓扑的平滑性接近 baseline，值得作为默认设置
- robot_only 下 interaction mesh 没有优势，需要加物体才能体现价值
