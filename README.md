# 灵巧手运动重定向

基于交互网格（Interaction Mesh）的手部运动重定向系统，移植自 [OmniRetarget](https://arxiv.org/abs/2501.00000) 并针对灵巧手场景优化。

## 算法

```
MediaPipe 21 点手部关键点（人手）
  → 坐标变换（SVD 手掌平面估计 + OPERATOR2MANO 旋转对齐）
  → S1 骨方向余弦 IK（无尺度的方向匹配，daqp QP 求解）
  → Delaunay 四面体化 → 邻接表 → Laplacian 坐标（目标）
  → S2 Laplacian 优化：寻找使 ||Laplacian_source - Laplacian_robot||² 最小的关节角
       + 角度锚定（保护 S1 的方向对齐）
       + 语义权重（pinch 接近时 boost）
       + 关节限位 + 信赖域约束
  → 输出关节角（20 DOF）
```

默认 cost 比例：`anchor : laplacian : smooth = 5 : 5 : 1`

## 性能

| 数据集 | 速度 | 说明 |
|--------|------|------|
| Manus（手套数据） | 330+ fps | S1+S2 联合优化 |
| HO-Cap（手物交互） | 160+ fps | S2 Laplacian + 物体点 |

## 目录结构

```
retargeting/
├── src/hand_retarget/           # 核心库
│   ├── retargeter.py            # InteractionMeshHandRetargeter（S1+S2 管线）
│   ├── mesh_utils.py            # Delaunay、Laplacian、骨架邻接
│   ├── mujoco_hand.py           # PinocchioHandModel（固定基座 FK/Jacobian）
│   │                            # MuJoCoFloatingHandModel（浮动基座 + 碰撞）
│   ├── mediapipe_io.py          # SVD+MANO 预处理、HO-Cap clip 加载
│   └── config.py                # HandRetargetConfig + 关键点映射
├── src/scene_builder/           # MuJoCo 场景构建（指尖 site 注入、腕部 6DOF）
├── src/hand_retarget_viz/       # 可视化 helpers（overlay/playback/cache/world_frame）
├── assets/                      # 机器人手模型（MuJoCo XML、STL 网格）
├── config/                      # YAML 配置文件
├── demos/                       # 端到端演示（manus.py、hocap.py、omniretarget.py）
├── scripts/                     # 批量 + 对比工具（retarget_hocap、compare_hocap）
├── tests/                       # pytest 门控测试
├── experiments/                 # 当前活跃实验
└── doc/                         # 算法笔记、改进计划
```

## 快速开始

### 环境

```bash
conda activate mjplgd
# 可选：设置 wuji_retargeting SDK 路径（不设置时使用默认路径）
export WUJI_SDK_PATH="/path/to/wuji_retargeting_private/public"
```

### 运行

```bash
# Manus 手套数据可视化
PYTHONPATH=src python demos/manus.py --semantic-weight

# HO-Cap 手物交互可视化
PYTHONPATH=src python demos/hocap.py --clip hocap__subject_3__20231024_161306__seg00

# 运行测试
PYTHONPATH=src pytest tests/test_gate.py -v

# 代码检查和格式化
ruff check src/ && ruff format src/
```

## 依赖

| 包 | 用途 |
|----|------|
| `pinocchio` | 前向运动学 + Jacobian（固定基座模式） |
| `mujoco` | 可视化 + 浮动基座 FK + 碰撞检测 |
| `qpsolvers[daqp]` | QP 求解器（替代 CVXPY+Clarabel，10 倍加速） |
| `scipy` | Delaunay 三角化 + 稀疏矩阵 |
| `wuji_retargeting` | SVD+MANO 对齐预处理 |

## 模型资源

| 用途 | 路径 |
|------|------|
| 重定向 FK | `wuji_hand_description/urdf/left.urdf`（Pinocchio，含 tip_link） |
| 可视化 | `assets/scenes/single_hand_obj_left.xml`（MuJoCo 场景） |
| 网格 | `assets/wujihand/meshes/left/`（26 个 STL 文件） |

## 文档

- `doc/improvement_plan.md` — 改进计划与已知问题
- `doc/exp8_link_midpoint_joint_optimization.md` — EXP-8 连杆中点联合优化实验详情
- `doc/refactoring_progress.md` — 代码重构进度记录
- `doc/omni.md` — OmniRetarget 算法移植笔记
