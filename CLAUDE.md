# CLAUDE.md -- 手部运动重定向

## 概述

手部运动重定向系统：MediaPipe 21 点手部关键点 → WujiHand 20 DOF 机器人手。
管线：SVD+MANO 腕部对齐 → S1 骨方向余弦 IK → S2 Laplacian 交互网格位置优化。
求解器：daqp QP（Manus 330+ fps，HO-Cap 160+ fps）。

## 环境

```bash
conda activate mjplgd  # Python 3.11, pinocchio, mujoco, qpsolvers[daqp]
export WUJI_SDK_PATH="/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public"  # 可选，有默认回退
```

## 目录结构

```
retargeting/
├── src/hand_retarget/           # 核心库
│   ├── retargeter.py            # InteractionMeshHandRetargeter（S1+S2 管线）
│   ├── mesh_utils.py            # Delaunay 三角化、Laplacian 坐标、骨架邻接
│   ├── mujoco_hand.py           # PinocchioHandModel（固定基座）+ MuJoCoFloatingHandModel
│   ├── mediapipe_io.py          # SVD+MANO 预处理、HO-Cap clip 加载
│   └── config.py                # HandRetargetConfig（17 个字段）+ JOINTS_MAPPING
├── src/scene_builder/           # MuJoCo 场景注入（指尖 site、腕部 6DOF、碰撞）
├── assets/                      # 机器人模型（MuJoCo XML、STL 网格）
├── config/                      # YAML 配置文件
├── demos/
│   ├── legacy/                  # Manus 可视化（play_interaction_mesh.py、play_manus.py）
│   ├── hocap/                   # HO-Cap 可视化（play_hocap.py）
│   └── shared/                  # 叠加层、回放、缓存工具
├── scripts/                     # 批量重定向（retarget_hocap.py）
├── tests/                       # pytest 门控测试（test_gate.py）
├── experiments/archive/         # 已完成实验结果（EXP-1~9）
├── doc/                         # 算法笔记、实验文档
└── lib/                         # 参考库（gitignored）
```

## 快速开始

```bash
# Manus 可视化
PYTHONPATH=src python demos/legacy/play_interaction_mesh.py --semantic-weight

# HO-Cap 可视化
PYTHONPATH=src python demos/hocap/play_hocap.py --clip hocap__subject_3__20231024_161306__seg00

# 运行测试
PYTHONPATH=src pytest tests/test_gate.py -v

# 代码检查
ruff check src/ && ruff format src/
```

## 核心管线

```
MediaPipe 21点 → mediapipe_io（SVD+MANO 对齐）→ retarget_frame()
                                                  ├── _build_topology（Delaunay → 邻接表 → Laplacian）
                                                  ├── _compute_weights（pinch 感知语义权重）
                                                  └── _run_optimization
                                                        ├── S1: solve_angle_warmup（余弦 IK，daqp）
                                                        └── S2: solve_single_iteration（Laplacian，daqp）
```

默认配置：`anchor=5, laplacian=5, smooth=1`（5:5:1 比例）。S1+S2 默认开启。

## 关键配置字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `use_angle_warmup` | True | S1 骨方向余弦 IK 对齐 |
| `angle_anchor_weight` | 5.0 | S2 联合 cost 中的 S1 角度锚定权重 |
| `smooth_weight` | 1.0 | 时间平滑 |
| `use_link_midpoints` | False | 使用 20 个连杆中点替代 21 个关节原点 |
| `exclude_fingers_from_laplacian` | None | 排除的手指索引（0-4），该手指不受 S2 梯度影响 |
| `delaunay_edge_threshold` | 0.06 | 过滤 Delaunay 边 > 60mm |
| `laplacian_distance_weight_k` | 20.0 | Laplacian 指数距离衰减权重 |

## HO-Cap 注意事项

- 强制 SVD 对齐（wrist_q 不可靠，SVD 胜率 92%）
- 腕部 6 DOF 全部锁死（SVD 已将 landmarks 置于腕部坐标系）
- MediaPipe 骨段长度为固定模板（9 个被试完全相同）
- S1 在 HO-Cap 上导致 DIP 反弓（robot 默认 DIP ≈ 31° vs source ≈ 5-16°）
- HO-Cap 最佳配置：不启用 S1（反弓 33.7% vs 启用 S1 时 97%）

## 关节索引速查

```
20 DOF，5 根手指 × 4 关节：q[4f], q[4f+1], q[4f+2], q[4f+3] = MCP屈伸, MCP外展, PIP, DIP
拇指：q[0-3]  食指：q[4-7]  中指：q[8-11]  无名指：q[12-15]  小指：q[16-19]
```

## 实验历史

| # | 名称 | 结果 | 状态 |
|---|------|------|------|
| 1 | 固定拓扑 | 抖动 -43%，方向 -11% | 采用 |
| 2 | 距离权重 | 位置恶化 | 不采用 |
| 3 | 语义权重 | Pinch 间距 -32% | 采用 |
| 4 | ARAP 旋转补偿 | 反弓 -25pp，速度 -2.5x | 不采用 |
| 5 | 方向探针 | 改善有限 | 不采用 |
| 6 | 骨段缩放 | Laplacian 抗拒非均匀缩放 | 不采用 |
| 7 | ARAP 边能量 + 骨架拓扑 | DIP 反弓 0%，整体 63.6% | 未合入 |
| 8 | 连杆中点 + 联合优化 | 食指 tip -24%，小指保护有效 | 采用 |
| 9 | daqp 求解器 | 10 倍加速，输出一致 | 采用 |
