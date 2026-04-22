# EXP-10: Delaunay 后端 Ablation

> 日期: 2026-04-21
> 状态: 不采用(主管线保持 scipy.Delaunay)
> 分支: `exp/delaunay-backend`(worktree `.worktrees/exp-delaunay-backend`)
> 详细数据: `experiments/delaunay_backend_exp/REPORT.md`(worktree 内)

## 动机

`retargeter._build_topology` 每帧调 `scipy.spatial.Delaunay`(qhull 2020.2)。历史分析怀疑 qhull 在近平面手掌 + 物体点配置下可能:
- 抛 `QH6154: Initial simplex is flat`(硬崩帧)
- 把共面点放进 `tri.coplanar` 静默丢点 → 下游 `adj=[]` → Laplacian 信号归零

预期换 CGAL / VTK 等后端,凭借 exact predicates 或不同算法可提升鲁棒性。本实验量化验证。

## 方法

Worktree 内 `src/hand_retarget/mesh_utils.py` 加 `DELAUNAY_BACKEND` 环境变量,支持 `scipy` / `pyvista` 切换。每帧 telemetry:tets 数、coplanar 数、失败标志、耗时。

测试分两层:
1. **真实数据**:Manus `manus1_5k.pkl` 5000 帧 + 3 个 HO-Cap clip(4710 帧)
2. **压力测试**:手工构造 7 个退化输入(近平面、严格平面、共线、重复点等)

## 结果

### 真实数据 — 两后端等价

| 数据 | qpos max Δ | qpos RMS Δ | coplanar 帧 | scipy fps | vtk fps |
|------|-----------:|-----------:|------------:|----------:|--------:|
| Manus 5000f | 0.022° | 0.002° | 0 | 372 | 291 |
| HO-Cap 166-297f × 3 | ≤0.012° | ≤0.001° | 0 | 183-193 | 155-162 |

- **全部 9710 帧中 coplanar 事件为 0** — qhull 静默丢点从未触发
- qpos 差异全是数值噪声(< 0.03° max)
- 同帧 edge Jaccard 0.919:两后端**有 8% 边不同**,但被 Laplacian + 距离衰减 + 60mm 阈值**完全吸收**
- VTK 在 HO-Cap 上**慢 17-22%**

### open3d 补测(上一轮分析勘误)

此前认为 "open3d 没有独立 3D Delaunay" — **错误**,它有 `o3d.geometry.TetraMesh.create_from_point_cloud`。但实测:
- open3d 二进制内嵌 `qhull_r 8.0.2 (2020.2.r 2020/08/31)` — **和 scipy 同一份 qhull**
- 非退化输入下 tet 集合与 scipy **字节级相同**(351/351 重叠,Jaccard 1.000)
- 唯一差别:open3d 重排输入点序,需用返回的 `pt_map` 反向映射索引

换 open3d 既无收益也无代价,**没必要换**。

### 压力测试 — 退化输入

| 情况 | scipy | open3d | VTK | scipy + `QJ` |
|------|-------|--------|-----|--------------|
| 正常 71 点 | ✓ 369 | ✓ 同 scipy | ✓ 364 | ✓ 369 |
| 近平面 z=5e-5 | ✓ 332 | ✓ 340 | ⚠ **18**(退化) | ✓ 332 |
| z 噪声 1e-9 | ✓ 338 | ✓ 336 | ⚠ **0** | ✓ 338 |
| 严格平面 z=0 | ✗ QH6154 | ✗ QH6239 | ⚠ 0 静默 | ✓ 340 |
| 全部共线 | ✗ QH6154 | ✗ QH6239 | ⚠ 0 静默 | ✓ 54 |

**关键修正**:上一轮分析推测 "VTK 在共面鲁棒性上更好" — **实测反了**。VTK 在近平面 / 严格平面下**静默返回 0 tets**(= 空邻接表 = 下游失效),不是比 qhull 好,只是从硬崩换成了软崩。唯一能干净处理严格退化的是 **scipy + `qhull_options="QJ"`**(joggle,qhull 2020.2 内置)。

### Timing(5000 帧 71 点)

| 后端 | 每帧 us | 相对 |
|------|--------:|-----:|
| scipy | 283 | 1.00× |
| scipy + QJ | 279 | 0.99×(略快) |
| open3d | 290 | 1.02× |
| VTK/pyvista | 1166 | **4.12×** |

## 结论

1. 本项目真实数据上,**Delaunay 后端选择对 retargeting 结果无可测影响**。Laplacian cost + 距离衰减(k=20)+ 60mm 阈值把 8% 的边集差异完全吸收。
2. 不换后端。VTK 慢 4 倍、且在近平面场景下更差;open3d 等价于 scipy;CGAL 投入产出比太低(非一行替换,需要 handle→index 映射 + 安装路径)。
3. **唯一值得做的防护是在 `create_interaction_mesh` 的 try/except 兜底里加 `qhull_options="QJ"`**,2 行代码,0 perf 影响。真实数据 0/4710 帧触发,纯防御性。
4. CGAL 的 exact predicates 在这条管线里没有下游落脚点 — 同帧 edge Jaccard 0.919 的歧义被 Laplacian 数值鲁棒性吸收了,predicate-level 的确定性没有下游价值。

## 上轮分析的勘误

| 预测 | 实测 |
|------|------|
| VTK 在共面鲁棒性上更好 | ✗ 反了,VTK 更差(静默 0 tets) |
| VTK 慢 2-5× | 部分对,实测 4.1× |
| open3d 没有独立 Delaunay | ✗ 有,但内部就是 qhull,等价 scipy |
| 真实数据有 coplanar 事件 | ✗ 9710 帧中 0 次 |
| 非退化帧两后端等价 | ✓ 确认 |

## 相关文件(worktree 内)

- `experiments/exp_delaunay_backend.py` — 驱动脚本(`--quick` / 全量)
- `experiments/delaunay_backend_exp/REPORT.md` — 完整数据报告
- `experiments/delaunay_backend_exp/summary.json` — 聚合指标
- `src/hand_retarget/mesh_utils.py` — 加了 `DELAUNAY_BACKEND` env var + `TELEMETRY` sink
