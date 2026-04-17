# CLAUDE.md -- 手部 Retargeting 项目

## 项目概述

Interaction mesh 手部运动重定向系统, 将 MediaPipe 21 点手部关键点映射到 WujiHand (20 DOF) 机器人手.
核心算法: Delaunay 四面体化 -> Laplacian 坐标 -> SQP+SOCP 优化, 移植自 OmniRetarget.

## 环境

- conda 环境: `mjplgd` (默认环境, pinocchio, cvxpy, clarabel, mujoco)
- Python 3.11, ruff 格式化 (配置见 `ruff.toml`, line-width=120)
- wuji_retargeting 依赖: editable install 在 `/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public/`
- URDF: `/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public/wuji_retargeting/wuji_hand_description/urdf/left.urdf`
- MuJoCo XML (可视化): `/home/l/ws/doc/WujiRepo/urdf_cali/reference/result/xml/left.xml`

## 目录结构

```
retargeting/
├── src/hand_retarget/           # 核心库
│   ├── __init__.py              # 公开 API: InteractionMeshHandRetargeter, HandRetargetConfig
│   ├── retargeter.py            # InteractionMeshHandRetargeter (Laplacian / ARAP edge / HO-Cap)
│   ├── mesh_utils.py            # Delaunay, Laplacian, ARAP rotation estimation, skeleton adjacency
│   ├── mujoco_hand.py           # HandModelProtocol + Pinocchio FK (固定基座) + MuJoCo (浮动基座)
│   ├── mediapipe_io.py          # 预处理 (SVD + MANO + 缩放) + HO-Cap clip 加载
│   └── config.py                # HandRetargetConfig + JOINTS_MAPPING + _YAML_FIELD_MAP
├── src/scene_builder/           # MuJoCo 场景构建工具
│   ├── __init__.py              # 公开 API: load_scene_model
│   └── hand_builder.py          # MjSpec 注入 (fingertip sites, wrist6dof, physics, collision)
├── assets/                      # 机器人手模型资源
│   ├── scenes/                  # MuJoCo XML 场景 (单手/双手/物体)
│   ├── wujihand/meshes/         # STL 网格 (left/ + right/ 各 26 个)
│   └── wujihand/xmls/           # MuJoCo XML (wuji_lh_xml/left_mjx.xml, wuji_rh_xml/right_mjx.xml)
├── config/                      # YAML 配置 (interaction_mesh_left.yaml, baseline_left.yaml)
├── data/
│   ├── manus_for_pinch/         # Manus 手套录制数据 (.pkl)
│   ├── hocap/                   # HO-Cap 原始数据集 (关键点 + 物体资源)
│   └── cache/                   # 自动缓存 (.npz)
│       └── hocap/               # HO-Cap clip 缓存 (both/ left/ right/ 子目录)
├── demos/
│   ├── legacy/                  # 可视化回放脚本
│   │   ├── play_interaction_mesh.py  # IM 回放 (含 Delaunay 边绿/红着色)
│   │   ├── play_manus.py             # Baseline IK 回放
│   │   └── play_mesh_only.py         # 仅网格可视化
│   ├── hocap/                   # HO-Cap 数据集回放
│   │   └── play_hocap.py             # 含物体 + mesh 边着色
│   ├── humanoid/                # OmniRetarget demo
│   │   └── play_omniretarget_demo.py
│   └── shared/                  # 回放共用工具
│       ├── overlay.py           #   add_sphere / add_line (MuJoCo viewer geom)
│       ├── playback.py          #   PlaybackController (pause/step/reverse)
│       └── cache.py             #   缓存读写工具
├── scripts/
│   ├── batch_retarget_hocap.py  # 批量生成 HO-Cap qpos 缓存
│   └── retarget_hocap.py        # 单 clip retarget
├── tests/
│   ├── test_refactor_baseline.py  # 回归测试: 100 帧 retarget 结果 vs baseline
│   └── refactor_baseline.npz     # 基准数据 (manus1_5k 前 100 帧)
├── experiments/                 # 共用工具 + 已完成实验归档
│   ├── benchmark.py             # RetargetBenchmark (指尖位置/方向/间距/抖动/pinch)
│   ├── object_interaction_benchmark.py  # 物体交互指标 (contact recall/pen/grasp-phase)
│   └── archive/                 # 已完成实验 (只读归档)
│       ├── laplacian_ablation/  #   EXP-1/2/3 Laplacian 参数消融
│       ├── hyperextension_study/ #  EXP-4~7 反弓专题
│       ├── hocap_pipeline/      #   HO-Cap 流水线分析
│       ├── baseline_before_probes/
│       ├── exp_palm_spread.py + results_palm_spread.md
│       ├── exp_object_frame.py
│       └── exp_edge_ratio.py    #   旧版 (最新版在 worktree)
├── .worktrees/                  # 进行中实验 (git worktree, 各自独立分支)
│   ├── exp-edge-ratio/          #   exp/edge-ratio-cost: Zhang23 边比例 cost
│   ├── exp-dual-space/          #   exp/dual-space-energy: bone方向+cross距离 dual cost
│   ├── exp-contact-graph/       #   exp/contact-graph: 接触图方法
│   └── exp-link-midpoint/       #   exp/link-midpoint: 连杆中点方法
├── ruff.toml                    # ruff 配置 (line-width=120, isort, UP rules)
├── doc/
│   ├── improvement_plan.md      # 改进计划 (全局状态追踪)
│   └── omni.md                  # OmniRetarget 算法笔记
└── lib/                         # 参考库 (gitignore)
```

## 核心架构

```
MediaPipe 21点 → mediapipe_io (SVD+MANO对齐) → retargeter.retarget_frame()
                                                   ├── mesh_utils (Delaunay → adj_list → Laplacian)
                                                   ├── SQP loop: solve_single_iteration() × n_iter
                                                   │     └── CVXPY + Clarabel SOCP solver
                                                   └── mujoco_hand (FK + Jacobian)
                                                         ├── MuJoCoHandModel (Pinocchio, 20 DOF)
                                                         └── MuJoCoFloatingHandModel (MuJoCo, 26 DOF + 碰撞)
```

关键接口: `HandModelProtocol` (mujoco_hand.py) 定义 FK/Jacobian 契约, 支持扩展到新手型.

## 运行方式

```bash
# 可视化 (MuJoCo 窗口)
python demos/legacy/play_interaction_mesh.py
python demos/legacy/play_manus.py
python demos/hocap/play_hocap.py

# 回归测试
conda run -n mjplgd PYTHONPATH=src python tests/test_refactor_baseline.py --verify

# 物体交互 Benchmark (需先生成 HO-Cap cache)
python scripts/batch_retarget_hocap.py --skip-existing
PYTHONPATH=src python experiments/object_interaction_benchmark.py --verbose

# 代码格式检查
ruff check src/ experiments/benchmark.py
ruff format --check src/ experiments/benchmark.py
```

## 代码规范

遵循 `/home/l/ws/doc/代码规范/CODING_STYLE_IS.md` (Robot Learning / RL 项目规范):

- 文件结构: `# ===` 分节符 (Imports / Constants / Classes / Private)
- 类结构: `# ===` 分区 (Dunder / Properties / Public / Private)
- 类型注解: 全函数签名, PEP 585 现代语法 (`list[int]` 非 `List[int]`, `X | None` 非 `Optional[X]`)
- Docstrings: Google style (Args / Returns / Raises)
- 常量: `UPPER_SNAKE_CASE` + ClassVar 注解 + inline docstring
- Magic numbers: 提取为模块级命名常量
- ruff: 自动格式化 + isort, 配置见 `ruff.toml`

## 实验范式

新实验使用 git worktree 隔离, 已完成实验归档在 `experiments/archive/`.

### 新实验工作流 (worktree 模式)

```bash
# 创建新实验
git worktree add .worktrees/exp-<name> -b exp/<name>
cd .worktrees/exp-<name>
ln -s ../../data data          # 共享数据目录 (可选)
```

每个 worktree 是自包含的实验包:
```
.worktrees/exp-<name>/
├── src/                        # 可自由修改核心算法 (与主分支隔离)
├── experiments/
│   ├── benchmark.py            # 共用工具 (从主分支同步)
│   ├── exp_<name>.py           # 实验脚本
│   └── <name>_exp/             # 结果包
│       ├── results.md          # 结论文档
│       └── cache/              # 缓存 (.npz)
└── data -> ../../data          # 共享数据 symlink
```

实验完成后:
- 胜出方案: cherry-pick/rebase 到 main
- 落选方案: 结论留在分支历史, `git worktree remove` 清理

### 实验脚本模板

每个实验脚本:
1. 定义对比条件 (通常 2-3 个: baseline + 变体)
2. 对每个条件运行 retarget, 结果缓存到 .npz (含 `--no-cache` 强制重算)
3. 计算指标并打印对比表
4. 支持 `--frames N` 限制帧数快速测试

### 指标定义

- **benchmark.py 通用指标**: 指尖位置/方向误差, 指间距误差, 抖动 (jerk), 时序一致性, C-space 覆盖, pinch 间距
- **反弓 (hyperextension)**: PIP/DIP 关节角 < 0 的帧比率
  - PIP indices: [2, 6, 10, 14, 18], DIP indices: [3, 7, 11, 15, 19]
  - URDF 允许约 -27 度, 但物理上不自然

### 缓存命名规范

```
{数据名}_{方法缩写}_cache.npz
  manus1_5k_im_cache.npz               # IM (当前默认)
  manus1_5k_bl_cache.npz               # baseline IK
  manus1_5k_wuji_cache.npz             # Wuji baseline
  manus1_5k_im_arapedge_skel_cache.npz # IM + ARAP edge + skeleton topology
```

### 结果文档规范

中文写作, 包含:
1. 动机/假设
2. 方法描述 (含代码改动清单)
3. 完整数据表 (标注改善/退化/无变化)
4. 机制分析
5. 局限和后续方向
6. 复现命令

## 已完成实验

| # | 名称 | 结论 | 状态 |
|---|------|------|------|
| EXP-1 | 固定拓扑 (`archive/laplacian_ablation/`) | 抖动 -43%, 方向 -11%, **采用** | 合入 |
| EXP-2 | 距离权重 (`archive/laplacian_ablation/`) | 位置恶化, **不采用** | 合入 |
| EXP-3 | 语义权重 (`archive/laplacian_ablation/`) | 对指间距 -32%, **采用** | 合入 |
| EXP-4 | ARAP 旋转补偿 (`archive/hyperextension_study/`) | 反弓 94.8%->69.2%, 速度 -2.5x, **待评估** | worktree 未合入 |
| EXP-5 | Orientation Probes (`archive/hyperextension_study/`) | 关节空间改善有限 (99.8%→98.8%), **改善有限** | 代码合入 (默认关闭) |
| EXP-6 | Bone Scaling (`archive/hyperextension_study/`) | Laplacian 抗拒非均匀缩放, **不采用** | 代码合入 (默认关闭) |
| EXP-7 | ARAP 逐边能量 + 骨架拓扑 (`archive/hyperextension_study/`) | DIP 反弓 0%, Overall 63.6%, **待评估** | worktree 未合入 |
| EXP-8 | Link Midpoint + 联合优化 | **已合入主分支**, 见下方详细说明 | 合入 |

### EXP-8: Link Midpoint + 联合角度/IM 优化 (已合入)

```
方案: 20 个 link 中点替代 21 个 joint origin + cosine IK 方向对齐 + 联合 cost

核心设计:
  1. Keypoint 改进: 用相邻关节中点代表 link 接触面 (而非关节中心)
     → wrist 排除 (已对齐, loss=0), 每指 4 中点 (3 midpoint + 1 TIP)
  2. GMR 风格 cosine IK (S1): 骨方向对齐, 91fps, PIP/DIP 0% 反弓
     → 用 FK + Jacobian 计算骨方向残差和指尖锚点, 收敛检测 (delta < 1e-3)
  3. 联合 cost (而非两阶段): angle_anchor + Laplacian 同时优化
     → 角度锚定保护 S1 的方向对齐不被 Laplacian 覆盖 (flips 仅 20/100000)
  4. 小拇指排除 Laplacian 梯度: 骨段比例差 1.5x, Jacobian 置零
     → 小拇指只受角度控制, 但位置仍参与 Delaunay 拓扑影响其他手指

Tuning 结果 (manus 500 frames, anchor=5 fixed):
  lap=1:  tip=16.7mm  pk_pip=0%  pk_dip=4.8%  24fps
  lap=5:  tip=16.6mm  pk_pip=0%  pk_dip=4.8%  24fps
  lap=200: tip=15.6mm pk_pip=0%  pk_dip=4.8%  20fps  ← Index -24%, Ring -25%
  S1solo: tip=16.8mm  pk_pip=0%  pk_dip=4.8%  91fps

启用方式:
  config.use_link_midpoints = True
  config.use_angle_warmup = True
  config.angle_anchor_weight = 5.0
  config.exclude_fingers_from_laplacian = [4]

Demo:
  play_interaction_mesh.py --link-midpoint --angle-warmup --semantic-weight
  play_hocap.py --link-midpoint --angle-warmup --semantic-weight
```

## 进行中实验 (worktree)

| 分支 | 路径 | 方向 |
|------|------|------|
| `exp/edge-ratio-cost` | `.worktrees/exp-edge-ratio/` | Zhang23 风格边比例归一化 |
| `exp/dual-space-energy` | `.worktrees/exp-dual-space/` | Angle-IM / edge descriptor 范式探索 |
| `exp/contact-graph` | `.worktrees/exp-contact-graph/` | 接触图方法 |
| `exp/link-midpoint` | `.worktrees/exp-link-midpoint/` | 连杆中点方法 (实验完成, 代码已合入) |

## 骨段比例差异 (核心挑战)

```
人手 vs WujiHand 骨段长度比 (默认姿态):
  Thumb IP→TIP:  0.77x   Index PIP→DIP: 1.18x   Pinky MCP→PIP: 1.50x
  Thumb MCP→IP:  0.81x   Index DIP→TIP: 1.09x   Pinky PIP→DIP: 1.77x  ← 最大差异
  Wrist→MCP:     0.83-0.99x                      Pinky DIP→TIP: 1.33x
```

Laplacian cost 同时约束方向和距离, 骨段比例差导致两个目标不可能同时满足, 优化器在折中中产生反弓等 artifact.

## 关节索引速查

```
20 DOF, 5 根手指 x 4 关节:
  finger N (0-indexed): q[4N], q[4N+1], q[4N+2], q[4N+3]
                      = MCP屈伸, MCP外展, PIP, DIP

  拇指: q[0-3]   (CMC/MCP/IP/DIP, 运动学与四指不同)
  食指: q[4-7]
  中指: q[8-11]
  无名指: q[12-15]
  小指: q[16-19]
```
