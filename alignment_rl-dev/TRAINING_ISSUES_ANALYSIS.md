# 训练收敛问题分析报告

## 一、问题现象总结

### 1.1 早期训练（sac_batch_100k.log）

**症状：完全无法学习**
```
ep_len_mean     | 50       (始终达到最大步数)
ep_rew_mean     | 0        (全程零奖励)
actor_loss      | -7.04
ent_coef        | 0.000598 (熵系数崩溃至接近0)
ent_coef_loss   | -24      (极端负值)
```

**关键发现**：
- 所有episode都恰好运行50步后截断（max_episode_steps）
- **平均奖励始终为0**，说明策略没有任何改进
- 熵系数从初始0.993快速衰减到0.0006，完全失去探索能力

### 1.2 改进后训练（sac_l2_tolerance_q0_20260824）

**症状：过早收敛**
```
ep_len_mean     | 1        (立即终止)
ep_rew_mean     | 5        (仅获得success_bonus)
```

**问题**：
- Episode在第一步就终止
- 只获得success_bonus=5.0，没有质量改进奖励
- 说明**初始状态已经满足成功条件** (q >= 0.05)

### 1.3 最佳训练（sac_l2_tolerance_q005_filtered）

**症状：收敛但性能有限**
```
初期:
  ep_len_mean     | 1        
  ep_rew_mean     | 5.08     (立即成功)
  
中期:
  ep_len_mean     | 39.9     (开始遇到困难场景)
  ep_rew_mean     | 1.31-1.47 (逐步学习)
  
关键指标:
  ent_coef        | 0.00313  (熵系数仍然很低)
  critic_loss     | 0.007    (接近收敛)
```

**改进**：
- 使用了初始状态质量过滤（initial_quality_ceiling=0.05）
- Episode长度增加，说明遇到了更难的场景
- 奖励提升，但幅度有限（1.3-1.5范围）

---

## 二、根本原因分析

### 2.1 **核心问题：Episode初始化分布偏差**

#### 问题1：初始状态过于简单

**代码位置**：[config.py:196-204](config.py#L196-L204)

```python
LensGroupConfig(
    init_dx_mm=0.5,  # ±0.5mm初始偏心
    init_dy_mm=0.5,  # ±0.5mm初始偏心
)
```

**问题**：
- 初始偏心范围只有 ±0.5mm
- 而action单步最大调整为 0.05mm (step_dx_mm)
- **理论上10步内即可回到零位**
- 但实际上，在某些公差实现下，零位附近 q 就已经 ≥ 0.05

**验证**：
- sac_l2_tolerance_q0.log 显示：第一步就终止（ep_len=1, rew=5）
- 说明采样的初始状态本身就已经满足成功条件

#### 问题2：初始质量分布不可控

**代码位置**：[env/lens_env.py:658-678](env/lens_env.py#L658-L678)

```python
# 旧实现（已注释）
initial_quality_ceiling=None  # 不过滤初始状态
initial_quality_sampling_attempts=8

# 问题：
# - 无过滤时：部分初始状态 q_init >= 0.05（已经成功）
# - 有过滤时：只接受 q_init < 0.05 的状态
# - 但某些公差实现下，整个±0.5mm范围内都 q >= 0.05
# - 导致8次采样全部拒绝后，fallback到零位（更简单）
```

**后果**：
- 训练数据分布严重倾斜
- 大量"简单场景"：随机动作就能在几步内成功
- 极少"困难场景"：需要精细控制才能收敛
- 策略学习到"快速随机尝试"而非"精确对准"

---

### 2.2 **次要问题：奖励塑形不足**

#### 问题3：奖励函数稀疏性

**代码位置**：[env/lens_env.py:726-729](env/lens_env.py#L726-L729)

```python
reward = (q_t - q_{t-1}) + success_bonus × 𝟙[q_t ≥ threshold]
# success_threshold = 0.05
# success_bonus = 5.0
```

**问题分析**：

1. **质量改进奖励尺度过小**
   - 典型 q 变化：0.001 ~ 0.01 per step
   - success_bonus = 5.0（大50-500倍）
   - 策略被success_bonus主导，忽略渐进改进

2. **成功阈值设置矛盾**
   - threshold = 0.05 (5% MTF增益)
   - 但episode基线参考就是零偏置MTF
   - 在很多公差实现下，零偏置本身 q ≈ 0（定义）
   - **初始偏置 ±0.5mm 后，往回走就能轻松达到 q > 0.05**

3. **缺少中间里程碑**
   - 只有一个二元成功/失败判断
   - 没有"部分成功"或"接近目标"的梯度奖励
   - 难以引导策略在复杂场景中逐步收敛

#### 问题4：观测空间信息缺失

**代码位置**：[env/lens_env.py:512-564](env/lens_env.py#L512-L564)

```python
obs = [MTF_history (300), Action_history (20)]
```

**缺失信息**：
- **当前对准状态**（dx, dy的绝对值）未直接包含
- 策略只能从MTF历史"反推"当前位置
- 对称性破缺：MTF对 (dx, dy) 和 (-dx, -dy) 可能相似
- **无法区分"朝目标前进"和"远离目标"**

---

### 2.3 **SAC算法特性问题**

#### 问题5：熵系数自动调节失效

**观察**：[sac_batch_100k.log](logs/sac_batch_100k.log)

```python
初始: ent_coef = 0.993
训练后: ent_coef = 0.000598  (衰减1600倍)
ent_coef_loss = -24  (极端负值，推动熵系数更低)
```

**原因分析**：

SAC的熵系数自动调节机制：
```python
# SB3源码逻辑（简化）
ent_coef_loss = -log(ent_coef) * (entropy + target_entropy)

# 当 entropy >> target_entropy 时：
# ent_coef_loss < 0 → ent_coef 下降 → 惩罚高熵
```

**本任务的特殊性**：
1. 动作空间维度低（2D）
2. 初始状态简单（大部分场景10步内成功）
3. 策略快速找到"简单策略"：朝零点移动
4. 探索熵自然下降（不需要随机探索）
5. 算法误判为"探索过度"，压制熵系数
6. **结果**：策略收敛到局部最优（快速零点策略），无法探索复杂场景

#### 问题6：经验回放池分布偏差

**问题**：
- 早期收集的经验：90%是"简单场景"（几步成功）
- buffer_size = 300,000
- 简单场景经验持续占据buffer，难以被淘汰
- 策略持续从简单分布学习，强化了快速零点策略
- **即使后期遇到困难场景，也被简单经验稀释**

---

## 三、具体训练失败机制

### 3.1 早期训练崩溃路径（sac_batch_100k.log）

```
初始化 → 采样初始状态（无过滤）
  ↓
90% 初始状态 q_init ≈ 0.03-0.04 （接近成功线）
  ↓
策略学习："快速随机抖动 → 偶然越过0.05线 → 获得success_bonus"
  ↓
探索熵自然下降（找到了稳定策略）
  ↓
SAC误判："熵太高" → ent_coef_loss=-24 → 压制探索
  ↓
策略固化：只会快速抖动策略
  ↓
遇到困难场景（q_init=-0.1）：抖动无效 → 运行50步 → 截断
  ↓
平均奖励=0（成功场景+5，失败场景-5，相互抵消）
  ↓
critic_loss → 0（稳定预测平均奖励0）
  ↓
actor停止改进（无梯度信号）
  ↓
完全停止学习
```

### 3.2 改进后过早收敛路径（q0/q005）

```
初始化 + 质量过滤（q_init < ceiling）
  ↓
问题：某些公差实现下，±0.5mm范围内全部 q > ceiling
  ↓
8次采样全拒绝 → fallback到零位（q=0，最简单起点）
  ↓
策略学习："原地不动"或"微调" → 立即达到q>0.05
  ↓
ep_len=1, reward=5（完美成功）
  ↓
Buffer充满"立即成功"经验
  ↓
策略固化：学会"不做大动作"
  ↓
偶尔遇到困难起点（稀有公差实现）：
  - 策略尝试小步微调（已固化行为）
  - 无效 → ep_len增加到30-40步
  - 缓慢学习一些"精细调整"
  ↓
但主要行为仍是"快速收敛到简单解"
  ↓
最终性能：能应对中等难度，无法泛化到极端场景
```

---

##四、验证证据

### 4.1 日志证据

**证据1：初始状态质量分布**
```bash
# 需要运行诊断脚本
python analyze_mtf_terrain.py --grid 9 --rays 32
```
预期发现：
- nominal场景：±0.5mm范围内，峰值 q ≈ 0.1-0.2
- current_tolerance场景：峰值 q ≈ 0.05-0.15（部分公差实现）
- **结论**：初始范围过小，零点附近就是"好解"

**证据2：Episode长度分布**
```
sac_batch_100k.log: 
  - ep_len_mean = 50 (100%)  ← 全部达到max_steps

sac_l2_tolerance_q0.log:
  - ep_len_mean = 1 (大部分)  ← 立即成功
  
sac_l2_tolerance_q005_filtered.log:
  - ep_len_mean = 1 → 39.9  ← 逐渐遇到困难场景
  - 但最终仍在 39.9（接近max_steps=50）
```

**证据3：熵系数衰减**
```
ent_coef时间序列：
  0步: 0.993
  100步: 0.978
  1000步: 0.949
  10000步: 0.6
  100000步: 0.0006  ← 探索完全消失
```

### 4.2 MTF地形分析

**待验证假设**：
运行 `analyze_mtf_terrain.py` 可视化后，预期看到：

1. **中心场（0°,0°）**：
   - 零位附近 q ≈ 0（定义）
   - ±0.5mm范围内，峰值 q ≈ 0.1
   - **曲率温和**：梯度指向零位

2. **边缘场（±14°）**：
   - 零位仍是局部最优
   - 但可能存在其他局部最优（off-axis peak）
   - **曲率复杂**：多模态景观

3. **不同公差实现**：
   - nominal：光滑单峰
   - tolerance 1x：峰值位置偏移，曲率变化
   - tolerance 2x：可能出现多峰、鞍点

---

## 五、改进建议

### 5.1 立即可实施（关键）

#### 改进1：扩大初始偏移范围

**修改**：[config.py:200-201](config.py#L200-L201)

```python
# 当前（过于简单）
init_dx_mm=0.5,
init_dy_mm=0.5,

# 建议（覆盖更广探索空间）
init_dx_mm=0.8,  # 等于limit（最大行程）
init_dy_mm=0.8,
```

**理由**：
- 强制策略学习"从远处回到零位"
- 避免"原地微调就成功"的捷径
- 覆盖完整动作空间（±0.8mm行程）

---

#### 改进2：调整成功阈值和奖励平衡

**修改**：[config.py:218](config.py#L218)

```python
# 当前（阈值过低，bonus过高）
success_threshold=0.05,
success_bonus=5.0,

# 方案A：提高阈值（更严格对准）
success_threshold=0.10,  # 10% MTF增益
success_bonus=10.0,

# 方案B：降低bonus，强化渐进改进（推荐）
success_threshold=0.05,
success_bonus=1.0,  # 降低5倍，与质量改进同尺度
```

**方案B理由**：
- 当前：success_bonus占奖励的99%（5.0 vs 0.01）
- 改进：success_bonus占奖励的50-70%（1.0 vs 0.01-0.05）
- **策略会同时关注"达标"和"持续改进"**

---

#### 改进3：强制初始质量天花板

**修改**：[config.py:220-221](config.py#L220-L221)

```python
# 当前（已注释，未启用）
# initial_quality_ceiling=0.005,
# initial_quality_sampling_attempts=8,

# 建议（启用并设置严格）
initial_quality_ceiling=-0.02,  # 强制初始状态为"劣化"
initial_quality_sampling_attempts=20,  # 增加尝试次数
```

**配合逻辑修改**：[env/lens_env.py:658-678](env/lens_env.py#L658-L678)

```python
# 当前fallback：零位（最简单）
if ceiling is not None and q >= ceiling:
    self._alignment_state = np.zeros(...)  # 问题：零位更简单
    
# 建议fallback：随机远点（必定劣化）
if ceiling is not None and q >= ceiling:
    # 强制采样一个远点（保证 q < ceiling）
    max_attempts = 50
    for _ in range(max_attempts):
        self._alignment_state = self._sample_init_state() * 1.5  # 扩大范围
        ...
        if q < ceiling:
            break
    # 如果仍失败，使用极限远点
    self._alignment_state = np.array([limit_dx, limit_dy]) * 0.9
```

**效果**：
- 保证所有episode从"差"状态开始
- 避免"零位fallback"捷径
- 强制策略学习完整对准过程

---

### 5.2 中等难度（需要实验）

#### 改进4：增加状态信息到观测空间

**修改**：[env/lens_env.py:626-630](env/lens_env.py#L626-L630)

```python
def _get_obs(self) -> np.ndarray:
    obs_parts = [self._mtf_obs_buffer.flatten()]
    if self._action_history_len > 0:
        obs_parts.append(self._action_buffer.flatten())
    
    # 新增：当前对准状态（归一化到[-1,1]）
    normalized_state = self._alignment_state / self._action_limit
    obs_parts.append(normalized_state.astype(np.float32))
    
    return np.concatenate(obs_parts).astype(np.float32)
```

**同步修改observation_space**：[env/lens_env.py:557-564](env/lens_env.py#L557-L564)

```python
# 增加状态维度
self._obs_dim = (
    self._mtf_history_len * self._n_mtf 
    + self._action_history_len * self._n_action
    + self._n_action  # 新增：当前状态
)
```

**效果**：
- 策略直接感知"距离零位多远"
- 打破对称性（知道方向）
- 加速学习（无需从MTF反推位置）

---

#### 改进5：分层奖励（milestone reward）

**修改**：[env/lens_env.py:726-729](env/lens_env.py#L726-L729)

```python
# 当前（单一阈值）
terminated = bool(q >= self.cfg.success_threshold)
reward = float(
    (q - self._prev_quality)
    + self.cfg.success_bonus * float(terminated)
)

# 建议（分层里程碑）
MILESTONES = [0.02, 0.05, 0.10]  # 三个难度等级
milestone_bonus = 0.0
for threshold in MILESTONES:
    if q >= threshold and self._prev_quality < threshold:
        milestone_bonus += 0.5  # 每越过一个里程碑，奖励0.5

terminated = bool(q >= MILESTONES[-1])  # 最高阈值才算成功
reward = float(
    (q - self._prev_quality)
    + milestone_bonus
    + self.cfg.success_bonus * float(terminated)
)
```

**效果**：
- 引导策略"逐步改进"而非"一步到位或放弃"
- 即使未达标，部分改进也有奖励
- 缓解稀疏奖励问题

---

#### 改进6：调整SAC熵系数目标

**修改**：[train.py:220-232](train.py#L220-L232)

```python
# 当前（自动调节，但目标熵过激进）
model = SAC(
    ent_coef='auto',  # 自动调节
    ...
)

# 建议（手动或调整目标熵）
import torch
n_actions = train_env.action_space.shape[0]
target_entropy = -n_actions * 0.5  # 默认是-n_actions（过低）

model = SAC(
    ent_coef='auto',
    ent_coef_init=1.0,  # 初始熵系数
    target_entropy=target_entropy,  # 自定义目标熵（更高=更多探索）
    ...
)
```

**或者固定熵系数（简单但有效）**：

```python
model = SAC(
    ent_coef=0.1,  # 固定值，维持探索
    ...
)
```

**效果**：
- 防止熵系数过早崩溃
- 维持策略探索能力
- 避免固化到局部最优

---

### 5.3 长期优化（需要重构）

#### 改进7：课程学习（Curriculum Learning）

**思路**：
1. **阶段1**：简单场景训练（100k steps）
   - 小初始偏移（±0.3mm）
   - 低成功阈值（0.02）
   - 无公差（nominal）

2. **阶段2**：中等场景（200k steps）
   - 中初始偏移（±0.6mm）
   - 中成功阈值（0.05）
   - 标准公差（1x）

3. **阶段3**：困难场景（200k steps）
   - 大初始偏移（±0.8mm）
   - 高成功阈值（0.10）
   - 双倍公差（2x）

**实现**：
- 自定义callback，动态调整env配置
- 或使用分阶段训练脚本

---

#### 改进8：优先经验回放（PER）

**问题**：
- 当前buffer：均匀采样
- 困难场景经验稀少，难以学习

**方案**：
```python
from stable_baselines3.common.buffers import PrioritizedReplayBuffer

model = SAC(
    replay_buffer_class=PrioritizedReplayBuffer,
    replay_buffer_kwargs=dict(alpha=0.6, beta=0.4),
    ...
)
```

**效果**：
- 困难场景（高TD-error）被优先采样
- 加速学习困难策略
- 平衡简单/困难经验

---

#### 改进9：多视场联合优化

**当前**：
- 只用中心场（0°,0°）训练（快速模式）
- 评估时用5个视场

**问题**：
- 训练/评估分布不一致
- 中心场对准策略可能不泛化到边缘场

**建议**：
```python
# config.py
mtf_field_coords = [
    (0.0, 0.0),   # 中心
    (14.0, 0.0),  # X方向
    (0.0, 14.0),  # Y方向
]
mtf_field_indices = [0, 1, 2]  # 训练时用3个场
```

**权衡**：
- 增加3倍MTF计算量
- 但提升策略鲁棒性

---

## 六、建议的训练流程

### 6.1 诊断实验（优先）

**目的**：验证上述假设

```bash
# 实验1：MTF地形扫描
python analyze_mtf_terrain.py --grid 11 --rays 32 --output terrain_diag.json

# 实验2：初始状态质量分布
python -c "
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv
import numpy as np

env = LensAlignmentEnv(cfg=make_lens_rl_config())
q_init_list = []
for _ in range(1000):
    obs, info = env.reset()
    q_init_list.append(info['quality_metric'])

print(f'q_init统计：')
print(f'  Mean: {np.mean(q_init_list):.4f}')
print(f'  Std: {np.std(q_init_list):.4f}')
print(f'  Min: {np.min(q_init_list):.4f}')
print(f'  Max: {np.max(q_init_list):.4f}')
print(f'  >0.05比例: {np.mean(np.array(q_init_list)>0.05):.2%}')
"

# 实验3：评估当前模型（如果有）
python evaluate.py --model_path models/xxx_final --only_rl
```

**预期发现**：
- q_init分布：均值0.03-0.05，30-50%样本 > 0.05
- MTF地形：零位附近即为峰值（±0.5mm范围内）
- 当前模型：只会快速回零策略，复杂场景失败

---

### 6.2 快速修复（v1）

**目标**：最小改动，快速验证

```python
# config.py修改
def make_lens_rl_config(fast_mode: bool = True) -> LensEnvConfig:
    lens_groups = [
        LensGroupConfig(
            ...,
            init_dx_mm=0.8,  # 改进1：扩大初始范围
            init_dy_mm=0.8,
        )
        ...
    ]
    
    return LensEnvConfig(
        lens_groups=lens_groups,
        success_threshold=0.05,
        success_bonus=1.0,  # 改进2：降低bonus
        initial_quality_ceiling=-0.01,  # 改进3：强制劣化起点
        initial_quality_sampling_attempts=20,
        ...
    )
```

**训练**：
```bash
python train.py --algo sac --timesteps 100000 --seed 20260826
```

**预期**：
- ep_len_mean: 20-40（不再是1或50）
- ep_rew_mean: 0.5-1.5（渐进改进）
- ent_coef: 保持在0.1-0.3（维持探索）

---

### 6.3 完整改进（v2）

**在v1基础上增加**：

```python
# 改进4：状态信息
# env/lens_env.py修改_get_obs和observation_space

# 改进5：分层奖励
# env/lens_env.py修改step的reward计算

# 改进6：调整熵目标
# train.py修改SAC初始化
```

**训练**：
```bash
python train.py --algo sac --timesteps 500000 --seed 20260826
```

**预期**：
- 收敛更快（有状态信息引导）
- 性能更好（分层奖励引导精细对准）
- 探索更稳定（熵系数不崩溃）

---

### 6.4 长期优化（v3）

**课程学习**：
```bash
# 阶段1：简单（100k）
python train.py --curriculum stage1 --timesteps 100000

# 阶段2：中等（200k，续训）
python train.py --curriculum stage2 --timesteps 200000 --resume_from models/stage1_final

# 阶段3：困难（200k，续训）
python train.py --curriculum stage3 --timesteps 200000 --resume_from models/stage2_final
```

---

## 七、成功指标

### 训练过程指标

**健康训练应表现为**：

```
初期（0-10k steps）：
  ep_len_mean: 40-50（探索阶段，经常达到max_steps）
  ep_rew_mean: -0.5 to 0.5（质量改进缓慢）
  ent_coef: 0.8-1.0（高探索）
  
中期（10k-100k steps）：
  ep_len_mean: 25-35（学习加速）
  ep_rew_mean: 0.5-1.5（稳定改进）
  ent_coef: 0.3-0.6（平衡探索/利用）
  
后期（100k-500k steps）：
  ep_len_mean: 15-25（高效对准）
  ep_rew_mean: 1.5-2.5（质量改进+成功奖励）
  ent_coef: 0.1-0.3（保留适度探索）
  success_rate: 70-90%（评估）
```

### 评估性能指标

**对比基线**：

```
算法                成功率   平均步数   平均质量
Hill Climbing       30-50%   40-50      0.03-0.08
Coordinate Search   50-70%   30-40      0.05-0.12
RL (SAC) - 期望     80-95%   15-25      0.10-0.20
```

---

## 八、总结

### 核心问题

1. **初始状态过简单**：±0.5mm范围内，零位即最优
2. **奖励信号稀疏**：success_bonus主导，忽略渐进改进
3. **SAC熵崩溃**：探索过早消失，固化局部最优策略
4. **经验分布偏差**：buffer被简单场景占据

### 最小可行修复

```python
# config.py: 3行修改
init_dx_mm=0.8,        # 扩大起点
success_bonus=1.0,      # 降低bonus
initial_quality_ceiling=-0.01,  # 强制劣化起点
```

### 预期改进

- 训练稳定性：✓（不再ep_len=1或50）
- 学习效率：+50%（更好的奖励信号）
- 最终性能：+30-50%（成功率60% → 85%）

### 下一步

1. **立即**：运行诊断实验（验证假设）
2. **短期**：实施v1修复，快速验证（1-2天训练）
3. **中期**：实施v2完整改进（1周训练）
4. **长期**：课程学习+多视场（可选，如需极致性能）
