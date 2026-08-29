# 初始质量过滤（Initial Quality Filtering）详解

## 一、什么是质量过滤？

**初始质量过滤**是在每个训练episode开始时，对随机采样的初始状态进行筛选，**拒绝过于简单的起点**，确保episode从"有挑战性"的状态开始。

---

## 二、为什么需要质量过滤？

### 问题场景

在光学对准任务中：

```
Episode开始流程（无过滤）：
1. 随机施加制造公差（每个episode不同）
2. 随机采样初始偏移：dx, dy ∈ [-0.5, 0.5] mm
3. 计算初始MTF质量 q_init
4. 开始训练...

问题：
某些情况下，随机采样的初始状态本身就已经"很好"了：
- 例如：采样到 dx=0.1mm, dy=0.05mm（接近零位）
- 此时 q_init = 0.08（已经超过成功阈值0.05）
- 策略只需"原地不动"或"随便动一步"就能获得success_bonus=5.0
```

**后果**：
- 策略学会了"碰运气"而非"精确对准"
- 训练数据中充斥大量"简单场景"
- 无法学到应对困难场景的能力

---

## 三、质量过滤的工作原理

### 配置参数

```python
# config.py
class LensEnvConfig:
    initial_quality_ceiling: float | None = None  # 质量上限
    initial_quality_sampling_attempts: int = 8     # 最大采样尝试次数
    success_threshold: float = 0.05                # 成功阈值
```

### 核心逻辑

```python
# env/lens_env.py: reset()方法

# 步骤1: 施加制造公差
apply_mfg_tolerances()

# 步骤2: 记录零偏置参考MTF
set_episode_reference()  # MTF_ref at (dx=0, dy=0)

# 步骤3: 重复采样，直到找到"够难"的初始状态
ceiling = 0.05  # 示例值
attempts = 8

for _ in range(attempts):
    # 随机采样初始状态
    dx, dy = uniform(-0.5, 0.5)
    
    # 计算初始质量
    MTF_init = compute_mtf(dx, dy)
    q_init = log(MTF_init / MTF_ref)  # 相对对数MTF
    
    # 判断是否满足条件
    if q_init < ceiling:  # 够差，接受
        break
else:
    # 8次都没找到 → fallback策略
    dx, dy = 0, 0  # 退回零位（代码当前实现）
```

### 流程图

```
开始Episode
    ↓
施加制造公差（随机）
    ↓
记录零位参考MTF_ref
    ↓
采样初始状态 (dx, dy)
    ↓
计算 q_init = log(MTF_init / MTF_ref)
    ↓
q_init < ceiling? ──Yes──> 接受，开始训练
    ↓ No
尝试次数 < 8? ──Yes──> 重新采样
    ↓ No
Fallback到零位（问题！）
    ↓
开始训练
```

---

## 四、质量指标 q 的含义

### 定义

```python
q = mean(log((MTF_current + ε) / (MTF_ref + ε)))
```

### 物理意义

- **q = 0**：当前MTF = 参考MTF（零偏置）
- **q > 0**：当前MTF > 参考MTF（改进）
- **q < 0**：当前MTF < 参考MTF（劣化）

### 示例

```
Episode初始化：
  1. 施加公差 → MTF_ref = 0.60（零偏置处）
  2. 采样初始状态 dx=0.4mm, dy=0.3mm
  3. MTF_init = 0.55（偏离零位，质量下降）
  4. q_init = log(0.55 / 0.60) = -0.087

判断：
  - ceiling = 0.05
  - q_init = -0.087 < 0.05 ✓ 接受
```

---

## 五、10次训练中质量过滤的演变

### 第1-5次：无过滤（initial_quality_ceiling=None）

```python
# 配置
initial_quality_ceiling = None  # 不过滤

# 后果
- 接受任何初始状态
- 训练数据混杂简单/困难场景
- 第1-4次：完全失败或极慢
- 第5次（nominal）：成功（因为nominal场景整体简单）
```

---

### 第6-7次：ceiling=0.0（过于宽松）

```python
# 配置
initial_quality_ceiling = 0.0  # 拒绝 q_init ≥ 0

# 期望
只接受"劣化"起点（q < 0）

# 实际结果
Episode立即终止（ep_len=1, rew=5）

# 原因
即使强制 q_init < 0，从初始状态到零位的路径上：
  - 步骤1：dx=0.4 → dx=0.35（靠近零位）
  - MTF提升，q可能从-0.05提升到0.06
  - 越过success_threshold=0.05 → 立即终止

问题根源：初始范围±0.5mm太小
```

---

### 第8次：ceiling=0.01（仍然宽松）

```python
# 配置
initial_quality_ceiling = 0.01  # 拒绝 q_init ≥ 0.01

# 实际结果
仍然 ep_len=1, rew=5（立即成功）

# 分析
ceiling=0.01仍然不够严格，因为：
  - 初始范围±0.5mm内，很多点 q ∈ [0, 0.04]
  - 这些点虽然 < 0.01，但移动几步就能达到0.05
```

---

### 第9次：ceiling=0.01 + filtered sampling（优化逻辑）

```python
# 配置
initial_quality_ceiling = 0.01
# 可能增加了更严格的采样逻辑

# 结果
训练失败或快速停止

# 推测原因
过滤太严格 + fallback逻辑问题：
  - 8次采样都被拒绝（±0.5mm范围内找不到q<0.01的点）
  - Fallback到零位（q=0，反而更简单）
  - 或者环境初始化出错
```

---

### 第10次：ceiling=0.05（放宽，当前最佳）

```python
# 配置
initial_quality_ceiling = 0.05  # 拒绝 q_init ≥ 0.05
initial_quality_sampling_attempts = 20

# 结果
✓ 出现学习曲线（ep_len: 1→39, rew: 5.08→1.42）

# 分析
ceiling=0.05 = success_threshold：
  - 拒绝"已经成功"的初始状态
  - 但仍接受"接近成功"的状态（q=0.04）
  
学习曲线解读：
  - 初期：ep_len=1, rew=5（简单场景多）
  - 中期：ep_len=24, rew=3（开始遇到挑战）
  - 后期：ep_len=39, rew=1.42（困难场景增多）
  
说明过滤开始起作用，但仍不够完美
```

---

## 六、当前实现的问题

### 问题1：Fallback逻辑不合理

**当前代码**（第669-678行）：

```python
if ceiling is not None and q >= ceiling:
    # 8次采样都失败 → fallback到零位
    self._alignment_state = np.zeros(self._n_action, dtype=np.float64)
    # 问题：零位是最简单的起点（q=0）！
```

**后果**：
- 某些公差实现下，±0.5mm范围内全部 q ≥ ceiling
- Fallback到零位反而更简单
- 策略学会"原地不动"

**应改为**：

```python
if ceiling is not None and q >= ceiling:
    # Fallback到远点（最大行程）
    self._alignment_state = self._action_limit * 0.9  # [0.72, 0.72] mm
    # 或继续扩大采样范围
    for _ in range(50):
        self._alignment_state = self._sample_init_state() * 1.5
        q = compute_quality(self._alignment_state)
        if q < ceiling:
            break
```

---

### 问题2：初始范围±0.5mm与过滤目标冲突

**矛盾**：
```
初始范围：±0.5mm（固定，从未改变）
零位距离：最大 √(0.5² + 0.5²) = 0.707mm
零位质量：q = 0（定义）

在很多公差实现下：
  - ±0.5mm范围内，q的分布：[-0.2, 0.2]
  - 要找到 q < -0.05 的点：稀少
  - 要找到 q < 0.0 的点：约50%
  - 要找到 q < 0.05 的点：约70-80%

结论：
  - ceiling < 0：频繁fallback
  - ceiling = 0.05：勉强work
  - 但无论如何，范围太小，学不到"长距离对准"
```

**解决方案**：

```python
# 同时调整两个参数
init_dx_mm = 0.8  # 扩大到行程上限
initial_quality_ceiling = -0.02  # 强制劣化起点

# 此时：
# - 范围内可选点更多（±0.8mm）
# - 更容易找到q<-0.02的点
# - 策略必须学会"从远处对准"
```

---

## 七、不同ceiling值的效果对比

| ceiling | 接受条件 | 典型初始状态 | Episode表现 | 评价 |
|---------|---------|------------|-----------|-----|
| **None** | 接受全部 | q ∈ [-0.2, 0.2] 随机 | ep_len=50, rew=0 | ❌ 过于混乱 |
| **0.0** | q < 0（劣化） | q ∈ [-0.2, -0.05] | ep_len=1, rew=5 | ❌ 仍然过简单 |
| **0.01** | q < 0.01 | q ∈ [-0.2, 0] | ep_len=1, rew=5 | ❌ 仍然过简单 |
| **0.05** | q < 0.05（未成功） | q ∈ [-0.2, 0.04] | ep_len=1→39, rew=5→1.4 | ⚠️ 部分改进 |
| **-0.02** | q < -0.02（明显劣化） | q ∈ [-0.2, -0.03] | ？（未测试） | ？ 理论更好 |

**最佳实践**：

```python
# 推荐配置
initial_quality_ceiling = -0.02  # 强制明显劣化
initial_quality_sampling_attempts = 30  # 增加尝试次数
init_dx_mm = 0.8  # 扩大采样范围（关键！）
```

---

## 八、质量过滤的正确使用场景

### ✓ 适合使用质量过滤的情况

1. **初始范围覆盖整个动作空间**
   - 例如：init_range = action_limit（±0.8mm）
   - 此时过滤确实能区分简单/困难场景

2. **任务存在明显的"简单区域"**
   - 例如：零位附近是单峰最优
   - 过滤避免从峰顶开始

3. **需要均衡训练数据分布**
   - 确保策略在各种难度下都有经验

### ❌ 不适合使用质量过滤的情况

1. **初始范围本身很小**
   - 例如：±0.5mm（当前问题）
   - 过滤后可选点更少，反而限制多样性

2. **Fallback逻辑不合理**
   - 当前实现：fallback到零位（最简单）
   - 应该：fallback到远点（最难）

3. **任务本身不存在"简单捷径"**
   - 如果所有起点难度相当，过滤无意义

---

## 九、实验建议

### 对比实验：验证质量过滤的价值

**实验组A：无过滤 + 大范围**
```python
initial_quality_ceiling = None
init_dx_mm = 0.8
```

**实验组B：严格过滤 + 大范围**
```python
initial_quality_ceiling = -0.02
init_dx_mm = 0.8
```

**实验组C：严格过滤 + 小范围（当前配置）**
```python
initial_quality_ceiling = 0.05
init_dx_mm = 0.5
```

**预期结果**：
- A > C（大范围更重要）
- B > A（过滤锦上添花）
- B >> C（当前配置）

**结论**：
- **扩大初始范围是第一优先级**
- 质量过滤是第二优先级（配合大范围使用）

---

## 十、总结

### 质量过滤的本质

**一句话**：通过拒绝"过于简单"的初始状态，强制策略学习应对困难场景。

### 当前问题

1. **初始范围太小**（±0.5mm）→ 无论如何过滤都过简单
2. **Fallback逻辑错误**（退回零位）→ 过滤失败后反而更简单
3. **Ceiling设置矛盾**（0.05 = success_threshold）→ 临界点不稳定

### 改进方案

```python
# 三位一体改进
init_dx_mm = 0.8           # 扩大范围（必须）
initial_quality_ceiling = -0.02  # 严格过滤（推荐）
# 修复fallback逻辑到远点（必须）
```

### 类比理解

**质量过滤 = 考试难度筛选**

```
无过滤：
  - 考试题目随机抽取（简单+困难混杂）
  - 学生碰到简单题就"蒙混过关"
  - 遇到难题就"完全不会"
  
有过滤：
  - 拒绝"过于简单"的题目
  - 确保每次考试都有一定难度
  - 学生被迫学会解决困难问题
  
但如果题库本身都是简单题（范围±0.5mm）：
  - 无论如何筛选，都是简单题
  - 必须先扩大题库（范围±0.8mm）
```

---

**关键洞察**：10次训练都专注于调整质量过滤参数（ceiling），但忽略了更根本的问题——**初始范围本身太小**。这就像试图通过筛选来提高题目难度，但题库本身都是简单题。正确做法是：**先扩大题库（init_range），再筛选难题（ceiling）**。
