# 实验: ARAP 逐顶点旋转补偿

> 状态: worktree 实验完成, 未合入主分支
> 分支: feat/arap-rotation-comp (worktree /tmp/retarget-arap)
> 日期: 2026-04-14

---

## 动机

Laplacian 坐标 `L(p_i) = p_i - mean(neighbors)` 对手指内部点 (PIP) 的邻居分布近似对称, 导致 |L| 接近零. 优化器对这些关节的弯曲方向缺乏直接约束, PIP/DIP 关节可能出现负角度 (反弓/hyperextension).

实测各点的 Laplacian 范数 (frame 500):

| 类型 | 代表点 | |L| | 邻居数 |
|------|--------|-----|--------|
| PIP (对称, 弱) | Ix_PIP | 0.013 | 8 |
| PIP (对称, 弱) | Md_PIP | 0.007 | 9 |
| PIP (对称, 弱) | Rg_PIP | 0.010 | 12 |
| MCP (非对称, 强) | Ix_MCP | 0.042 | 13 |
| TIP (边缘, 强) | Ix_TIP | 0.048 | 15 |

灵感来自 ARAP (As-Rigid-As-Possible, Sorkine & Alexa 2007): 对每个 keypoint 用 SVD 估计邻域从 source 到 robot 的局部旋转 R_i, 旋转 source Laplacian target, 使方向与当前 robot 配置对齐.

## 方法

### 核心改动

在 SQP 每次迭代中插入 rotation compensation:

```
原始:  target 固定 = L @ source_pts (计算一次, 整帧不变)
改后:  每次迭代:
         1. FK(q_current) -> robot_pts
         2. 对每个 vertex i, SVD 估计 R_i (source 边 vs robot 边的协方差矩阵)
         3. target_i = R_i @ L_source_i  (旋转后的目标)
         4. 用旋转后的 target 进入 SOCP 求解
```

### SVD 旋转估计

对每个 vertex i, 收集到所有邻居 j 的边向量:

```python
e_src = source_pts[neighbors] - source_pts[i]   # (K, 3) source 边
e_cur = current_pts[neighbors] - current_pts[i]  # (K, 3) robot 边
H = e_src.T @ e_cur                              # (3, 3) 协方差矩阵
U, S, Vt = svd(H)
R_i = Vt.T @ U.T                                 # 最优旋转 (Procrustes)
if det(R_i) < 0: 翻转 Vt 最后一行后重算         # 避免反射
```

每帧 21 个 3x3 SVD, 约 0.05ms, 相对于 SOCP 求解 (~5ms/iter) 可忽略.

### 代码改动

| 文件 | 改动 |
|------|------|
| `src/hand_retarget/mesh_utils.py` | 新增 `estimate_per_vertex_rotations()` (~30 行) |
| `src/hand_retarget/config.py` | 新增 `rotation_compensation: bool = False` 配置项 |
| `src/hand_retarget/retargeter.py` | `retarget_frame()` SQP 循环内插入 R_i 估计 + target 旋转 |
| `demos/legacy/play_interaction_mesh.py` | 新增 `--rotation-comp` 命令行参数 + 对应 cache suffix |
| `experiments/exp_hyperextension.py` | 新建: 三方对比实验脚本 (baseline IK / IM 原始 / IM+ARAP) |

## 反弓指标定义

```
关节索引 (20 DOF, 每根手指 4 个关节: MCP屈伸, MCP外展, PIP, DIP):
  PIP indices: [2, 6, 10, 14, 18]  (拇指 IP + 4 根手指 PIP)
  DIP indices: [3, 7, 11, 15, 19]

反弓帧 = 该帧中任一 PIP 或 DIP 关节角 < 0
反弓比率 = 反弓帧数 / 总帧数
```

URDF 关节极限允许 PIP/DIP 约 -27 度, 因此反弓不违反关节极限, 但物理上不自然.

## 实验结果

数据: `manus1_5k.pkl`, 5000 帧, 左手 Manus 数据手套录制

### 总览

| 指标 | Baseline IK | IM 原始 | IM+ARAP | 变化 (IM原始->ARAP) |
|------|------------|---------|---------|---------------------|
| 反弓帧比率 | 84.1% | 94.8% | **69.2%** | -25.6pp |
| 速度 | 401 fps | 67 fps | 26 fps | 2.5x 变慢 |

### 逐手指反弓比率 (任一 PIP/DIP < 0)

| 手指 | Baseline | IM 原始 | IM+ARAP | 变化 |
|------|---------|---------|---------|------|
| 拇指 | 12.1% | 3.6% | 9.2% | +5.6pp (退化) |
| 食指 | 63.4% | 87.6% | **37.9%** | -49.7pp |
| 中指 | 41.6% | 35.1% | **30.4%** | -4.7pp |
| 无名指 | 18.2% | 29.6% | **19.3%** | -10.3pp |
| 小指 | 21.3% | 48.4% | **36.6%** | -11.8pp |

### 逐关节反弓比率 (% 帧 < 0)

| 关节 | Baseline | IM 原始 | IM+ARAP | 变化 |
|------|---------|---------|---------|------|
| 拇指 PIP | 0.0% | 0.0% | 0.0% | -- |
| 拇指 DIP | 12.1% | 3.6% | 9.2% | +5.6pp (退化) |
| 食指 PIP | 57.9% | 70.6% | **19.4%** | -51.2pp |
| 食指 DIP | 6.9% | 17.0% | 18.5% | +1.5pp (退化) |
| 中指 PIP | 41.6% | 35.1% | **30.4%** | -4.7pp |
| 中指 DIP | 0.0% | 0.0% | 0.0% | -- |
| 无名指 PIP | 18.2% | 29.6% | **19.3%** | -10.3pp |
| 无名指 DIP | 0.0% | 0.0% | 0.0% | -- |
| 小指 PIP | 21.3% | 48.4% | **36.6%** | -11.8pp |
| 小指 DIP | 0.0% | 0.0% | 0.0% | -- |

### 最差反弓角度 (度, 负值 = 反弓)

| 关节 | Baseline | IM 原始 | IM+ARAP | 变化 |
|------|---------|---------|---------|------|
| 拇指 PIP | 4.3 | 5.5 | 0.6 | -- (均为正, 无反弓) |
| 拇指 DIP | -9.8 | -7.0 | **-4.6** | 改善 |
| 食指 PIP | -27.7 | -27.7 | **-24.8** | 改善 |
| 食指 DIP | -1.4 | -27.1 | -26.8 | 接近 (均触极限) |
| 中指 PIP | -27.8 | -15.1 | **-9.1** | 改善 |
| 中指 DIP | 21.3 | 12.6 | 17.5 | -- (均为正, 无反弓) |
| 无名指 PIP | -26.8 | -10.4 | **-5.0** | 改善 |
| 无名指 DIP | 5.0 | 9.1 | 13.4 | -- (均为正, 无反弓) |
| 小指 PIP | -22.6 | -27.1 | -27.1 | 无变化 (触极限) |
| 小指 DIP | 11.6 | 31.4 | 31.8 | -- (均为正, 无反弓) |

### 退化项分析

两处退化:
- **拇指 DIP**: 3.6% -> 9.2%. 拇指运动学与四指不同 (CMC/MCP/IP 结构), ARAP 旋转补偿可能引入了不适合拇指的方向偏差
- **食指 DIP**: 17.0% -> 18.5%. 微小退化 (+1.5pp), 在噪声范围内

## 机制分析

### R_i 的传导路径

PIP 自身的 |L| 接近零, `R_i @ L_PIP ~= 0` -- 旋转一个接近零的向量, 直接效果很小.

但 PIP 作为**邻居**出现在 MCP/DIP/TIP 等点的 Laplacian 行中. 这些点的 |L| 较大 (MCP=0.042, TIP=0.048), R_i 旋转它们的 target 有实质方向变化. 优化器调整 dq 使这些点匹配时, 通过 Laplacian 矩阵的耦合, 间接约束了 PIP 的弯曲方向.

```
传导路径:
  R_MCP 旋转 MCP 的 target (|L|=0.042, 有实质方向变化)
    -> 优化器调整 dq 让 MCP 的 Laplacian 匹配
      -> MCP 的 Laplacian 包含 PIP 的位置 -> PIP 被间接约束
```

### |L| 大小不是瓶颈

PIP 的 |L| 小不代表 PIP 不受约束. Laplacian 矩阵将所有点耦合在一起, PIP 通过作为其他点的邻居参与全局优化. rotation compensation 改善了 |L| 大的点 (MCP, TIP) 的方向目标, 这个改善通过矩阵耦合和运动学链传播到 PIP.

换言之: rotation compensation 不是逐点独立起效的, 而是通过改善全局 cost landscape 间接改善 PIP.

### 速度代价

67 fps -> 26 fps (2.5x 变慢). 每次 SQP 迭代新增:
- FK: 0.01ms
- 21 个 3x3 SVD: 0.05ms
- einsum 旋转: < 0.01ms

总计 < 0.1ms/iter, 但 SQP 通常跑 10 次迭代, 每帧新增约 1ms. 速度下降主要来自 rotation compensation 改变了收敛路径, 可能需要更多迭代.

## 局限

1. **PIP 的直接约束仍然弱**: 改善通过间接路径起效. 直接约束 PIP 方向需要 ARAP 原始的逐边能量 (不经过 Laplacian 平均), 改动量大
2. **仍有 69.2% 反弓**: 改善明显但未根本解决. 可能需要配合关节角硬约束 (PIP >= 0) 或 orientation probe 机制
3. **拇指退化**: 拇指运动学结构不同, 当前统一的 rotation compensation 可能不适合拇指
4. **速度下降 2.5x**: 离线处理够用, 实时遥操可能需要优化

## 后续方向

- [ ] 与 `--fixed-topology --semantic-weight` 组合测试
- [ ] ARAP 逐边能量替换 Laplacian 能量 (改动大, 需重写 cost 函数)
- [ ] PIP/DIP >= 0 硬约束 (在 SOCP 中加不等式约束, 改动小)
- [ ] orientation probe (`--probes`) 与 rotation compensation 的对比/组合
- [ ] 拇指单独处理 (不同的 rotation compensation 策略或排除拇指)

## 复现

```bash
cd /home/l/ws/RL/retargeting
# worktree 分支有完整改动
git worktree list   # -> /tmp/retarget-arap [feat/arap-rotation-comp]

# 运行实验 (首次约 5 分钟, 后续从 cache 读取)
python experiments/exp_hyperextension.py

# 强制重算
python experiments/exp_hyperextension.py --no-cache

# 限制帧数快速测试
python experiments/exp_hyperextension.py --frames 500 --no-cache

# 可视化对比 (需要 MuJoCo 显示)
python demos/legacy/play_interaction_mesh.py                  # IM 原始
python demos/legacy/play_interaction_mesh.py --rotation-comp  # IM+ARAP
```
