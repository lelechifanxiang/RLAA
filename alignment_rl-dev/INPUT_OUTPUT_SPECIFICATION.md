# RLAA系统：输入/输出参数与空间定义

## 📥 输入参数

### 1. 光学系统输入（固定输入）
```python
# 光学处方（ZMX文件）
zmx_file: str = "double_gauss_6element.zmx"
  • 22个光学面
  • 6个透镜元件
  • 4个Coordinate Break组（镜组分隔）

# 视场配置
mtf_field_coords: List[Tuple[float, float]] = [
    (0.0, 0.0),      # 中心场
    (14.0, 0.0),     # X正向边缘场
    (-14.0, 0.0),    # X负向边缘场
    (0.0, 14.0),     # Y正向边缘场
    (0.0, -14.0),    # Y负向边缘场
]

# MTF频点配置
mtf_frequencies: List[float] = [20.0, 30.0, 50.0]  # lp/mm
```

### 2. Episode级输入（每次reset变化）

#### 2.1 制造公差（Domain Randomization）
```python
# 面级公差（每个光学面独立随机）
tol_radius_rel: float = 0.001        # 曲率半径相对误差 (0.1%)
tol_thickness_mm: float = 0.010      # 厚度误差 (10μm)
tol_decenter_mm: float = 0.015       # 偏心误差 (15μm)
tol_tilt_deg: float = 0.03           # 倾斜误差 (0.03°)

# 组级公差（每个CB组的刚体误差）
tol_lens_decenter_mm: float = 0.015  # 镜组装配偏心 (15μm)
tol_lens_tilt_deg: float = 0.03      # 镜组装配倾斜 (0.03°)

# 采样方式：高斯分布 N(0, tol)
# 每个episode随机采样一次，episode内保持不变
```

#### 2.2 初始对准状态
```python
# 初始偏移范围（均匀分布）
init_dx_mm: float = 0.5   # X方向 ±0.5mm
init_dy_mm: float = 0.5   # Y方向 ±0.5mm

# 采样方式：uniform(-init_dx_mm, init_dx_mm)
# 只对激活的镜组（group 2）采样
```

#### 2.3 可选：质量过滤
```python
initial_quality_ceiling: float | None = 0.05  # 质量上限
initial_quality_sampling_attempts: int = 20   # 最大采样次数

# 逻辑：拒绝 q_init >= ceiling 的初始状态
# 重复采样直到满足条件或达到最大尝试次数
```

### 3. Step级输入（每步变化）

#### 3.1 动作（从策略网络输出）
```python
action: np.ndarray  # shape=(2,)
  • 范围：[-1, 1]（归一化）
  • 物理含义：
    - action[0] → Δdx ∈ [-0.05, 0.05] mm
    - action[1] → Δdy ∈ [-0.05, 0.05] mm
  
# 转换公式
physical_action = action * step_dx_mm
new_state = old_state + physical_action
new_state = clip(new_state, -limit, +limit)
```

---

## 📤 输出参数

### 1. 观测空间（返回给RL算法）
```python
observation: np.ndarray  # shape=(320,)

组成：
  • MTF历史：10步 × 30个MTF值 = 300维
  • 动作历史：10步 × 2个动作 = 20维
  
总维度：320

# 详细分解
MTF历史 (300维)：
  └─ 10步历史
      └─ 每步：5视场 × 3频点 × 2方向(sag/tang) = 30个MTF值
      └─ 值范围：相对log MTF ∈ [-2.0, 2.0]
                q = log((MTF_current + ε) / (MTF_ref + ε))

动作历史 (20维)：
  └─ 10步历史
      └─ 每步：2个归一化动作 ∈ [-1, 1]
```

### 2. 奖励（标量）
```python
reward: float

# 计算公式
reward = (q_t - q_{t-1}) + success_bonus × 𝟙[q_t >= threshold]

其中：
  • q_t = mean(log((MTF_t + ε) / (MTF_ref + ε)))
  • q_{t-1} = 上一步的质量指标
  • success_bonus = 5.0（当前配置）
  • threshold = 0.05（成功阈值）
  • 𝟙[·] = 指示函数（条件为真时=1，否则=0）

奖励组成：
  • 稠密部分：(q_t - q_{t-1})  # 质量改进奖励，典型值 ±0.01
  • 稀疏部分：5.0 × 𝟙[成功]   # 成功奖励，一次性获得
```

### 3. 终止信号（布尔）
```python
terminated: bool

# 终止条件1：成功
q_t >= success_threshold  # q >= 0.05

# 终止条件2：超时（在Gymnasium的TimeLimit wrapper中）
step_count >= max_episode_steps  # step >= 50
```

### 4. Info字典（调试信息）
```python
info: dict = {
    'quality_metric': float,      # 当前q值
    'state': np.ndarray,           # 当前对准状态 [dx, dy] mm
    'step': int,                   # 当前步数
    'success': bool,               # 是否成功
    'mtf_obs': np.ndarray,         # 相对log MTF (30,)
    'raw_mtf_obs': np.ndarray,     # 原始MTF (30,)
    'episode_ref_mtf_obs': np.ndarray,  # episode基线MTF (30,)
    'nominal_mtf_obs': np.ndarray,      # 标称MTF (30,)
    'compensator_z': float,        # 补偿器找到的像面位置 mm
}
```

---

## 🎮 空间定义（Gymnasium格式）

### 动作空间
```python
action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(2,),      # 2D偏心对准
    dtype=np.float32
)

# 说明
• 类型：连续动作空间
• 维度：2（dx, dy）
• 范围：[-1, 1]（归一化）
• 映射：action → action * 0.05mm → 物理增量
```

### 观测空间
```python
observation_space = spaces.Box(
    low=np.concatenate([
        np.full(300, -2.0),   # MTF历史范围 [-2.0, 2.0]
        np.full(20, -1.0),    # 动作历史范围 [-1.0, 1.0]
    ]),
    high=np.concatenate([
        np.full(300, 2.0),
        np.full(20, 1.0),
    ]),
    shape=(320,),
    dtype=np.float32
)

# 说明
• 类型：连续观测空间
• 维度：320（300 MTF + 20 动作）
• 范围：
  - MTF部分：[-2.0, 2.0]（相对log MTF裁剪）
  - 动作部分：[-1.0, 1.0]（归一化）
```

---

## 📊 数据流示意

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Episode开始（reset）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入（Episode级）：
  └─ 随机采样制造公差 → 施加到光学系统
  └─ 计算零位参考MTF → MTF_ref (30,)
  └─ 随机采样初始状态 → (dx, dy) ∈ [-0.5, 0.5]²
  └─ 【可选】质量过滤 → 拒绝过简单的初始状态

输出：
  └─ observation_0 (320,)
      ├─ MTF历史 (300,): 全部填充初始MTF（相对log）
      └─ 动作历史 (20,): 全部填充0
  └─ info_0: {'quality_metric': q_0, 'state': [dx, dy], ...}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Episode进行（step）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入（Step级）：
  └─ action_t (2,) ∈ [-1, 1]
      ├─ 来源：策略网络 π(observation_{t-1})
      └─ 物理含义：Δdx, Δdy 各 ±0.05mm

处理：
  1. 状态更新
     └─ state_t = clip(state_{t-1} + action_t * 0.05mm, ±0.8mm)
  
  2. 光学计算
     └─ 应用 state_t 到光学系统
     └─ 光线追迹 → PSF → MTF → MTF_t (30,)
  
  3. 质量计算
     └─ q_t = mean(log((MTF_t + ε) / (MTF_ref + ε)))
  
  4. 奖励计算
     └─ reward_t = (q_t - q_{t-1}) + 5.0 × 𝟙[q_t >= 0.05]
  
  5. 终止判断
     └─ terminated = (q_t >= 0.05)
     └─ truncated = (step >= 50)  # 由TimeLimit wrapper处理
  
  6. 观测更新
     └─ 滚动MTF历史：移除最旧，添加最新
     └─ 滚动动作历史：移除最旧，添加当前action

输出：
  └─ observation_t (320,): 更新后的历史
  └─ reward_t (float): 本步奖励
  └─ terminated (bool): 是否成功终止
  └─ truncated (bool): 是否超时截断
  └─ info_t (dict): 调试信息

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔢 维度汇总表

| 参数类型 | 名称 | 维度/范围 | 说明 |
|---------|------|----------|------|
| **输入** | | | |
| 光学处方 | ZMX | 22面×属性 | 固定 |
| 视场 | field_coords | 5×2 | (x,y)角度 [deg] |
| 频点 | frequencies | 3 | [20,30,50] lp/mm |
| 制造公差 | tolerance | 6种×22面 | 每episode随机 |
| 初始状态 | init_state | 2 | (dx,dy) ∈ [-0.5,0.5]² mm |
| 动作 | action | 2 | (Δdx,Δdy) ∈ [-1,1] → ±0.05mm |
| **输出** | | | |
| 观测（MTF） | mtf_obs | 300 | 10步×30值，∈[-2,2] |
| 观测（动作） | action_obs | 20 | 10步×2值，∈[-1,1] |
| 观测（总） | observation | 320 | 300+20 |
| 奖励 | reward | 1 | ∈ [-∞, 5.xx] |
| 质量指标 | q | 1 | ∈ [-∞, +∞] |
| 终止 | terminated | bool | q>=0.05 |
| **中间** | | | |
| 原始MTF | raw_mtf | 30 | 5场×3频×2方向 |
| 参考MTF | ref_mtf | 30 | episode零位基线 |
| 对准状态 | state | 2 | (dx,dy) ∈ [-0.8,0.8]² mm |

---

## 💡 关键设计说明

### 1. 观测空间不包含状态信息
```python
# 当前设计（已实现）
observation = [MTF历史, 动作历史]  # 320维

# 缺失信息
• 当前对准状态 (dx, dy)：策略需要从MTF反推位置
• 距离零位的距离：策略不知道"还有多远"
• 公差特征：策略无法区分不同公差实现

# 这是第4/10次训练失败的原因之一！
```

### 2. 奖励信号失衡
```python
# 当前配置
质量改进奖励：典型 ±0.01 per step
成功奖励：5.0（一次性）

# 比例：500:1
# 问题：策略被成功奖励主导，忽略质量改进

# 建议配置
success_bonus: 1.0  # 降低到100:1比例
```

### 3. 相对质量指标 q
```python
# 定义
q = mean(log((MTF_current + ε) / (MTF_ref + ε)))

# 优点
• Episode间可比（消除公差差异）
• q=0 表示零位（直观）
• q>0 表示改进（目标）

# 物理意义
q=0.05 表示：MTF平均提升 exp(0.05)≈1.051 = 5.1%
```

### 4. 历史窗口设计
```python
# 10步历史 = 最近5秒（假设2步/秒训练速率）

# 作用
• 捕获动态趋势（MTF在上升还是下降）
• 打破马尔可夫性（策略感知"动量"）
• 帮助策略判断"当前策略是否有效"
```

---

这份文档完整定义了系统的输入/输出接口和空间定义。需要我补充更多细节吗？
