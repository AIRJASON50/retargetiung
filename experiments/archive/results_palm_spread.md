# EXP: Palm Spread Correction in Bone Scaling

## 动机与假设

当前 bone scaling 只做 per-finger 径向链缩放（wrist→MCP→PIP→DIP→TIP），
不缩放跨手指的掌内宽度（MCP-to-MCP 横向距离）。
人手掌宽 81.4mm，WujiHand 掌宽 65.9mm（比值 0.810）。
假设：加入掌宽修正后，手掌拓扑更接近机器人，优化器工作点更合理，减少反弓。

## 方法描述

### 三个对比条件

| 条件 | 说明 |
|------|------|
| A: No scaling | IM 基线，无骨骼缩放 |
| B: Radial only | `use_bone_scaling=True, use_palm_spread_scaling=False` |
| C: Radial+Palm | `use_bone_scaling=True, use_palm_spread_scaling=True` |

### 代码改动

- `src/hand_retarget/config.py`：新增 `use_palm_spread_scaling: bool = True`
- `src/hand_retarget/retargeter.py`：`_apply_bone_scaling` Phase 2 增加 `self.config.use_palm_spread_scaling` 门控
- `experiments/exp_palm_spread.py`：三条件对比脚本

### Palm spread correction（Phase 2）实现

```python
spread_dir  = normalize(index_mcp - pinky_mcp)
spread_center = (index_mcp + pinky_mcp) / 2
for each non-thumb MCP (mp_idx in [5,9,13,17]):
    lateral = dot(scaled[mcp] - spread_center, spread_dir)
    scaled[mcp] += (palm_spread_ratio - 1.0) * lateral * spread_dir
```

掌宽比值 0.810，所以非拇指 MCP 向掌心方向收缩 19%。

## 结果（500 帧，manus1_5k）

### 观测到的骨骼比值

```
Bone ratio Thumb:  s0=0.988, s1=0.983, s2=0.812, s3=0.770
Bone ratio Index:  s0=0.947, s1=1.066, s2=1.178, s3=1.093
Bone ratio Middle: s0=0.914, s1=1.011, s2=1.084, s3=1.086
Bone ratio Ring:   s0=0.897, s1=1.064, s2=1.054, s3=1.066
Bone ratio Pinky:  s0=0.831, s1=1.499, s2=1.771, s3=1.333
Palm spread ratio: 0.810 (robot 65.9mm / source 81.4mm)
```

机器人和人手最大的结构差异：
- **Pinky 特别短**（人 98.4mm → 机器人 81.7mm），但 Pinky MCP 之后反而比人**更长**（PIP 段 1.499x）
- **拇指远端短**（IP 0.812x，DIP 0.770x）
- **掌宽窄** 0.810x

### 数值表

```
                       A: No scaling  B: Radial only  C: Radial+Palm
----------------------------------------------------------------------
FPS                              64              70              71
Overall hyper %                49.8            80.8            79.2  ← B,C 显著更差

Per-finger:
  Thumb                         36.4            62.2            64.6
  Index                         24.2            40.6            41.2
  Middle                        23.8            22.4            21.8
  Ring                          24.2            26.8            25.6
  Pinky                         17.2             0.0             0.0  ← 骨骼缩放修好了 Pinky

Per-joint:
  Thumb DIP                     36.4            62.2            64.6  ← 缩放后变差
  Index PIP                     24.2            40.6            41.0
  Pinky PIP                     17.2             0.0             0.0  ← 缩放后改善
```

### Deltas vs A（总体反弓）

- B: +31.0 pp（骨骼缩放让整体反弓显著变差）
- C: +29.4 pp（掌宽修正小幅改善 1.6pp，但仍比无缩放差很多）

## 机制分析

### Pinky 改善的原因

Pinky s1-s3 比值 1.499-1.771：机器人 Pinky 指节比人手长很多。
不缩放时，优化器将 Pinky 目标视为人手长度，机器人 Pinky 需要收缩（负角度）才能达到目标。
骨骼缩放后，目标被拉长，对应机器人实际骨段长度，Pinky PIP 不再需要收缩 → 反弓归零。

### 拇指 / 食指变差的原因

拇指远端比值 0.812/0.770：机器人拇指远端比人手短。
骨骼缩放后，目标点向掌心收缩，Laplacian 优化器为了"跟上"被压缩的拇指目标，
可能推动拇指 DIP 进入超伸展区间。

### Laplacian 耦合放大了问题

单独修正每根手指的比值，通过 Delaunay 网格的 Laplacian 耦合，
会将修正效果传播到相邻关节（理论空缺节记录的机制）。
Pinky 的大比值修正（1.771x）会在 Laplacian 中对中指和无名指产生拉力，
可能间接影响 Index/Thumb 的优化结果。

### 掌宽修正（B → C）效果有限的原因

掌宽缩放只挪动 4 个 MCP 的横向投影，Laplacian 计算用的顶点位置改变量很小（约 20% × 横向投影）。
相比骨段长度方向的拉力（特别是 Pinky 1.771x），掌宽修正对优化目标的改变量级更小。

## 局限

1. **整体反弓变差**：bone scaling 对总体质量有负效果；需考虑是否废弃骨骼缩放，或改变其应用位置
2. **缩放 vs 不缩放的 trade-off**：Pinky 改善了，Thumb/Index 变差了；不同手指受益方向相反
3. **Laplacian 耦合难以隔离**：在 SQP 框架内，单独修正一根手指的目标点坐标无法避免传播效应
4. **掌宽修正效果不显著**：1.6pp 改善在测量误差范围内

## 结论

palm spread correction 有方向性的改善（-1.6pp），但效果弱。
骨骼缩放本身（radial only）导致整体反弓增加 31pp，原因是：
1. Pinky 段比值大，拉伸效果强，通过 Laplacian 耦合影响邻近手指
2. 拇指远端压缩，推动拇指 DIP 超伸展

**后续方向**：
- 考虑 per-finger 选择性缩放（只对 Pinky 做 MCP→TIP 缩放，Thumb 不缩放远端）
- 或放弃骨骼缩放，转向关节角空间正则化来处理比例差异
- 在 edge-ratio 实验中加关节空间正则后测试效果（edge-ratio 本身的局部保真度好，缺的是关节 prior）

## 复现命令

```bash
conda activate mjplgd
cd /home/l/ws/RL/retargeting

# 完整实验（500 帧）
PYTHONPATH=src python experiments/exp_palm_spread.py --frames 500 --no-cache

# 强制重算（清除缓存）
PYTHONPATH=src python experiments/exp_palm_spread.py --frames 500 --no-cache

# 1000 帧
PYTHONPATH=src python experiments/exp_palm_spread.py --frames 1000 --no-cache
```
