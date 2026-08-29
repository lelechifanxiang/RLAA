# 4自由度交替对准系统

## 概述

在原有2自由度（dx, dy）对准系统基础上，扩展为4自由度系统，新增倾斜调整（rx, ry），并实现交替运动模式。

### 主要改进

1. **增加倾斜自由度**
   - 新增 rx（绕x轴倾斜）和 ry（绕y轴倾斜）两个自由度
   - 倾斜范围：±1.0°
   - 倾斜步长：0.05°/步

2. **交替运动模式**
   - **奇数步**：仅调整偏心（dx, dy），倾斜动作被屏蔽为0
   - **偶数步**：仅调整倾斜（rx, ry），偏心动作被屏蔽为0
   - 模拟实际工艺：先调位置，再调角度，交替收敛

3. **运动模式对比**
   - **交替模式（alternating）**：降低动作空间复杂度，更接近实际工艺
   - **同时模式（simultaneous）**：4自由度同时调整，作为对比基线

## 文件结构

```
alignment_rl-dev/
├── config_4dof.py                      # 4自由度配置文件
├── env/
│   └── alternating_lens_env.py         # 交替运动环境包装器
├── train_4dof_alternating.py           # 训练脚本
├── analyze_4dof_alternating.py         # 分析和可视化脚本
└── README_4DOF_ALTERNATING.md          # 本文档
```

## 配置参数

### 偏心自由度（保持原有配置）
- 初始范围：±0.8 mm
- 单步上限：0.05 mm
- 行程限制：±0.8 mm

### 倾斜自由度（新增）
- 初始范围：±1.0°
- 单步上限：0.05°
- 行程限制：±1.0°

### 关键配置
```python
from config_4dof import make_4dof_alternating_config, Alternating4DOFConfig

# 创建4自由度配置
lens_cfg = make_4dof_alternating_config(fast_mode=True)

# 交替模式配置
alt_cfg = Alternating4DOFConfig(
    motion_mode='alternating',           # 'alternating' 或 'simultaneous'
    decenter_indices=[0, 1],             # dx, dy的索引
    tilt_indices=[2, 3],                 # rx, ry的索引
    decenter_on_odd_steps=True,          # True: 奇数步调偏心
)
```

## 使用方法

### 1. 训练交替模式

```bash
# 默认配置（SAC算法，交替模式，100万步）
python train_4dof_alternating.py

# 自定义配置
python train_4dof_alternating.py \
    --algo sac \
    --mode alternating \
    --timesteps 2000000 \
    --seed 42

# 训练同时模式（对比实验）
python train_4dof_alternating.py --mode simultaneous
```

### 2. 分析和可视化

```bash
# 分析训练好的模型
python analyze_4dof_alternating.py \
    --model models/sac_4dof_alternating_TIMESTAMP_final.zip \
    --mode alternating \
    --algo sac \
    --n-episodes 5 \
    --output-dir analysis_results
```

### 3. 断点续训

```bash
python train_4dof_alternating.py \
    --resume_from models/sac_4dof_alternating_TIMESTAMP_ckpt/rl_model_500000_steps.zip \
    --timesteps 2000000
```

## 运动模式分析

### 交替模式的优势
1. **降低学习复杂度**：每步只需学习2维动作，而非4维
2. **符合实际工艺**：模拟人工对准流程（先调位置，后调角度）
3. **避免耦合干扰**：偏心和倾斜分步调整，减少相互干扰

### 交替模式的挑战
1. **收敛步数增加**：每个自由度只在50%的步数中被调整
2. **策略学习难度**：需要学会在不同步识别当前可调的自由度

### 预期性能对比
- **交替模式**：收敛更稳定，但可能需要更多步数
- **同时模式**：收敛更快，但对策略网络要求更高

## 可视化输出

分析脚本会生成以下可视化图表：

1. **偏心轨迹图**：(dx, dy) 平面的运动轨迹
2. **倾斜轨迹图**：(rx, ry) 平面的运动轨迹
3. **时序调整历史**：各自由度随步数的变化，标注激活区域
4. **MTF质量曲线**：质量指标随步数的改进情况

每个子图都会标注：
- 绿色星形：起点
- 红色叉形：终点
- 红色背景：偏心调整步
- 蓝色背景：倾斜调整步

## 实验建议

### 基础对比实验
1. 训练交替模式：`python train_4dof_alternating.py --mode alternating`
2. 训练同时模式：`python train_4dof_alternating.py --mode simultaneous`
3. 对比成功率、收敛速度、最终质量

### 消融实验
1. 调整交替顺序：修改 `decenter_on_odd_steps` 参数
2. 调整倾斜范围：修改 `init_rx/ry_deg` 和 `limit_rx/ry_deg`
3. 调整步长比例：修改 `step_rx/ry_deg` 与 `step_dx/dy_mm` 的比例

### 评估指标
- **成功率**：达到质量阈值（0.05）的episode比例
- **平均步数**：成功episode的平均收敛步数
- **最终质量**：episode结束时的平均质量指标
- **总奖励**：累计奖励（反映收敛速度和质量）

## 理论分析

### 动作空间维度对比
- **2DOF（原始）**：动作空间 R²，学习相对简单
- **4DOF同时**：动作空间 R⁴，复杂度显著增加
- **4DOF交替**：等效于两个 R² 空间，复杂度介于中间

### 收敛步数估计
假设偏心和倾斜各需 N 步收敛：
- **同时模式**：约 N 步（理想情况，假设无干扰）
- **交替模式**：约 2N 步（每个自由度轮流调整）

但考虑到学习难度，实际可能：
- **同时模式**：学习困难，可能需要 >N 步才能掌握策略
- **交替模式**：学习容易，2N 步内稳定收敛

## 后续扩展方向

1. **自适应交替**：根据当前误差动态决定调整偏心还是倾斜
2. **多阶段策略**：初期快速粗调，后期精细微调
3. **层次化强化学习**：上层决策调整哪个自由度，下层执行具体动作
4. **迁移学习**：先训练2DOF，再扩展到4DOF

## 注意事项

1. **显存需求**：4自由度需要更大的观测空间，显存占用约增加20%
2. **训练时间**：交替模式可能需要更多训练步数（建议200万步）
3. **评估模式**：评估时建议使用高精度模式（128光线）以准确衡量性能
4. **初始状态**：4自由度的初始错位空间更大，难度提升

## 技术细节

### 动作屏蔽机制
```python
# 在 AlternatingLensEnv.step() 中
if step_count % 2 == 1:  # 奇数步
    masked_action[2:4] = 0.0  # 屏蔽 rx, ry
else:  # 偶数步
    masked_action[0:2] = 0.0  # 屏蔽 dx, dy
```

### 观测空间
```
观测 = [MTF历史 (10步 × n_freq × n_field × 2) +
        动作历史 (10步 × 4)]
```

### 奖励函数（保持不变）
```
reward = (quality_t - quality_{t-1}) + bonus × success
```

## 问题排查

### 常见问题

1. **训练不收敛**
   - 检查倾斜步长是否过大（建议0.05°）
   - 尝试增加训练步数（200万+）
   - 检查初始范围是否合理

2. **显存不足**
   - 减少并行环境数（从12降到8或6）
   - 使用快速模式（32光线）

3. **评估性能差**
   - 确认使用正确的运动模式配置
   - 检查模型是否充分训练
   - 尝试增加评估episodes获得更稳定统计

## 参考

- 原始2DOF系统：[lens_env.py](env/lens_env.py)
- 配置系统：[config.py](config.py)
- 批量训练环境：[batch_lens_env.py](env/batch_lens_env.py)
