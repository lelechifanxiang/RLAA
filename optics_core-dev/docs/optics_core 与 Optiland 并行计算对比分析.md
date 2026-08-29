# optics_core 与 Optiland 并行计算对比分析

## 1. 对比范围

本文对比以下本地源码快照：

- optics_core：提交 `1b157c3`。
- Optiland：`C:\Users\Huwei\Project\OpenSource\optiland`，提交 `fb1bc07`。

重点考察前向仿真的并行数据模型，以及 spot、PSF、MTF 的理论计算效率。结论来自源码结构、默认参数和计算复杂度，不代表同一台机器上的实测加速比。

为避免错误结论，文中始终区分两种口径：

1. **同算法、同精度、同采样量**：用于判断实现效率。
2. **各自默认接口**：用于判断开箱即用的绝对耗时，但结果精度和物理语义可能不同。

## 2. 核心结论

两套框架的并行目标不同：

- optics_core 是 **design-first / batch-first**：同一表面拓扑下，不同曲率、厚度、偏心和倾斜的设计共享一个 `MultiOpticalSystem`，光线主要形状为 `[design, field, wavelength, ray]`。
- Optiland 是 **backend-first**：一个 `Optic` 表示一个光学系统，NumPy 或 Torch 后端负责并行同一系统内的大量光线、像素和矩阵运算；多个不同设计仍主要通过对象列表或 Python 循环执行。

因此总体判断如下：

1. 数百至数千个同拓扑设计的装配公差 Monte Carlo、批量 spot 或惠更斯 MTF，optics_core 具有明确的架构优势。
2. 单系统、小光线数的 spot，Optiland 的 NumPy 后端或 Torch FP32 路径可能更快；其默认 spot 只有 127 条光线，而 optics_core 默认约 2791 条，默认耗时不能直接比较。
3. 对双方的直接 Huygens PSF，optics_core 默认计算量显著更小，并能批量 design 和波长；但 Optiland 已有像点分块，单设计超高采样时更不容易耗尽显存。
4. Optiland 还提供 FFT PSF、MMDFT PSF 和对应 MTF。允许改用这些算法时，单系统绝对耗时可能远低于 optics_core 的直接惠更斯积分，但这不是同算法性能对比。
5. 对等价的 Huygens MTF，optics_core 当前预计明显领先：它不计算理想 PSF，支持多设计和全波长混合，并用两次一维 FFT 代替完整二维 FFT。
6. CPU 运行和算法覆盖面是 Optiland 的优势；大规模 FP64 多设计 GPU 吞吐是 optics_core 的优势。

## 3. 并行数据模型

| 维度 | optics_core | Optiland |
| --- | --- | --- |
| 不同光学设计 `D` | 原生 design 维度，参数矩阵、frame 和一阶量均按 design 组织 | 一个 `Optic` 保存一套表面参数；`MultiConfiguration` 是 `list[Optic]` |
| 多视场 `F` | 光线张量包含 field 维度 | 底层 `trace()` 可把多个视场展平成一维光线；高层 spot/PSF/MTF 多处仍逐视场循环 |
| 多波长 `W` | 光线张量包含 wavelength 维度，可一次追迹并支持全波长 PSF 混合 | 高层 spot 逐波长追迹；公开 PSF/MTF 类按单波长工作 |
| 光瞳光线 `P` | Tensor 化，保留逻辑维度 | `RealRays` 使用多个一维数组，字段和光瞳点通常展平成一维 |
| 像面像素 `I²` | Huygens 内核 Tensor 化 | Torch Huygens 按像点分块；NumPy Huygens 使用 Numba 并行循环 |
| 设备 | Torch CPU/CUDA | 全局切换 NumPy/Torch，Torch 可选 CPU/CUDA |
| 默认精度 | FP64，惠更斯使用 complex128 | NumPy 默认 FP64；Torch 默认 FP32，可切换 FP64 |
| 批量显存控制 | PSF/MTF 自动探测 design minibatch，OOM 后减半重试 | 无 design minibatch；Torch Huygens 固定按 1024 个像点分块 |

### 3.1 optics_core

`MultiOpticalSystem` 将共享表面拓扑与 `ParameterVectorBatch` 分开，追迹张量的主要逻辑形状为：

```text
[design, field, wavelength, pupil_ray]
```

表面仍按顺序在 Python 中循环，但每个表面内核同时处理所有 design、视场、波长和光线。`design_batch_view()` 对已准备的 `FrameData`、`FirstOrderData` 和净口径 Tensor 做连续切片；这些切片是 Tensor 视图，不复制数值，也不触发额外 CPU-GPU 搬运。

PSF/MTF 会先测量单设计峰值显存，再选择 design minibatch。运行中若仍发生 CUDA OOM，则减半重试。该机制解决的是“设计数量过多”的显存问题，同时保持每个 minibatch 的 GPU 并行度。

当前实现也存在两项开销：

- `surface_value()` 仍从 Python 参数向量收集数值，再创建目标设备上的 FP64 Tensor；批量 spot 中这一主机端组装和传输可能成为可见开销。
- 所有系统都执行通用局部 frame 的 `einsum` 变换。对完全轴对称、无偏心倾斜的单设计，这比专用轴对称路径更重。

### 3.2 Optiland

Optiland 的后端抽象支持：

```python
be.set_backend("numpy")
be.set_backend("torch")
```

NumPy 后端默认 FP64；Torch 后端默认 CPU、FP32，可切换 CUDA 和 FP64，并提供自动微分能力。后端是全局状态，切换操作本身不是线程安全的。

`RealRays` 用 `x/y/z/L/M/N/intensity/wavelength/opd` 等一维数组保存光线。`RealRayTracer.trace()` 可以接收多个 `Hx/Hy`，将 `field × pupil` 展平后一次追迹，因此底层具备同一系统内的多视场向量化能力。

但不同设计没有等价的一级 Tensor 维度：

- 表面半径、位置和坐标系主要是单系统标量 Tensor。
- `MultiConfiguration.add_configuration()` 对 `Optic` 执行 `deepcopy()`，最终保存 `list[Optic]`，不是 stacked design batch。
- `MonteCarlo.run()` 逐次重置系统、施加扰动、补偿并评价，外层是 Python iteration 循环。

因此切换 Torch/CUDA 只会加速“当前 `Optic` 内部”的数值运算，不能自动把 1000 个 `Optic` 合并成一个 GPU batch。

### 3.3 Optiland Torch 路径中的额外同步和存储

Optiland 的通用性带来几项对 GPU 吞吐不利的实现细节：

- 每个表面追迹后都会复制并保存 8 组光线数据，包括位置、方向、强度和 OPD，内存与写带宽约为 `O(S·R)`。optics_core 在 spot、波前和 PSF 前置追迹中使用 `record_intersections=False`，只保留最终状态。
- 坐标系存在 `if self.rx`、`if self.ry` 等标量 Tensor 分支；标准面和 Newton 求交也存在 `be.all()`、`be.max()` 后进入 Python 条件的代码。Torch/CUDA 下这些位置可能引入设备同步。
- Torch Huygens 在判断振幅是否为复数时，无条件执行一次 `be.to_numpy(pupil_amp)`，会把 pupil 振幅同步到 CPU；对每个 PSF 都会产生不必要的边界传输。

这些问题不会改变算法复杂度，但会削弱小批量 GPU 计算和高频调用的效率。

## 4. Spot 计算效率

### 4.1 当前执行方式

optics_core：

- 一次追迹全部 `D × F × W × P`。
- 默认 `ray_density=30` 的 hexapolar 采样为 `1 + 3×30×31 = 2791` 个评价点，另附一条参考主光线。
- 统一使用 FP64，并在 GPU 上直接统计多波长 RMS/GEO 半径。

Optiland：

- `SpotDiagram._generate_data()` 先循环视场，再循环波长；每个组合单独调用一次 `optic.trace()`。
- 默认 `num_rings=6`，即 `1 + 3×6×7 = 127` 条光线。
- 可用 NumPy FP64、Torch FP32 或 Torch FP64。
- 底层 tracer 虽能一次展平多个视场，但当前高层 SpotDiagram 没有利用该能力。

### 4.2 复杂度与理论性能

设表面数为 `S`，两边的主要算术量均为：

```text
O(D · F · W · P · S)
```

差异在于任务组织：

```text
optics_core: 1 次大追迹，表面循环外没有 D/F/W Python 循环
Optiland:    每个设计由外部循环，每个 F×W 组合再次启动一条追迹链路
```

理论判断：

- **各自默认参数、单设计**：Optiland 通常更快，因为默认光线数约为 optics_core 的 `1/22`；该结果不能说明同精度实现更快。
- **同光线数、单设计、单视场、单波长**：胜负不确定。Optiland 的单系统表面参数更轻，NumPy 对小数组尤其有利；但 Torch 路径的标量同步和逐面记录会抵消部分优势。
- **同光线数、多视场、多波长**：optics_core 更容易保持 GPU 饱和，并减少重复的 Python 调用和 kernel launch，预计逐渐占优。
- **数百至数千设计**：optics_core 预计明显领先。Optiland 需要逐个 `Optic` 执行完整追迹，无法在同一表面内核中并行不同设计。
- **CPU、小规模分析**：Optiland 的 NumPy 路径更自然，通常优于 optics_core 的 Torch CPU 路径。

## 5. PSF 计算效率

### 5.1 PSF 类型并不完全等价

optics_core 当前公开的是 Zemax 对标方向的标量 Huygens PSF：

1. 规则入瞳采样并追迹到像面。
2. 由像面交点、OPL 和方向余弦构造局部平面波相位。
3. 在共同像面网格直接叠加复振幅。
4. 支持单波长或全部波长的强度混合。
5. PSF 默认不归一化；公开 PSF 同时计算理想 PSF 和 Strehl。

Optiland 提供多种方法：

| 方法 | 主计算 | 公开接口粒度 | 与 optics_core 是否同口径 |
| --- | --- | --- | --- |
| `ScalarHuygensPSF` | 出瞳球面次波，含距离、`1/R` 和倾斜因子 | 单系统、单视场、单波长 | 最接近，但传播公式和归一化仍不同 |
| `ScalarFFTPSF` | 复瞳函数二维 FFT | 单系统、单视场、单波长 | 不同算法 |
| `MMDFTPSF` | `L @ pupil @ R` 矩阵三乘积 | 单系统、单视场、单波长 | 不同算法 |
| Vectorial 版本 | 矢量场传播 | 单系统、单视场、单波长 | optics_core 暂无对应功能 |

所以“Optiland PSF 更快”必须附带方法名称。FFT/MMDFT 的更低耗时不能视为直接 Huygens 内核领先。

### 5.2 默认直接 Huygens 计算量

直接积分的主项为：

```text
O(D · F · W · P · I²)
```

双方默认设置差异很大：

- optics_core：pupil `32×32`，image `32×32`。实际 PSF 约有 `32²×32² = 1,048,576` 个 pupil-image 相互作用；公开 PSF 还计算理想核，合计约 `2.10×10⁶`。
- Optiland：pupil 网格 `128×128`，裁圆后实际为 12644 点；image `128×128`。实际 PSF 约有 `12644×128² = 207,159,296` 个相互作用，理想峰值的直接积分只额外计算一个像点；非轴上 PSF 还会额外生成一次轴上波前数据。

因此默认公开 PSF 的主相互作用数，Optiland 约为 optics_core 的 **99 倍**。这只是工作量比例，不是预计速度倍率；Optiland 的默认 pupil 和 image 分辨率也都更高。

若双方都设为 `32×32`，Optiland 的裁圆 pupil 有 740 点，而 optics_core 当前仍处理完整 1024 点并用权重/有效掩码排除无效贡献。此时默认工作量差异大幅缩小，必须实测才能判断单设计速度。

### 5.3 单次相互作用和显存路径

optics_core：

- 相位主要是方向余弦与像面偏移的点积，再执行 `cos/sin` 和求和。
- 对一个 design minibatch 完整物化近似 `[B, W, P, I, I]` 的相位和 complex128 kernel。
- 自动缩小的是 design batch；尚未按 pupil 或 image 分块。因此极高采样下，`B=1` 仍可能 OOM。
- MTF 消费的 PSF 路径跳过理想 PSF，直接把积分量减半。

Optiland Torch Huygens：

- 每个相互作用还计算三维距离 `sqrt`、球面传播 `exp(ikR)/R` 和倾斜因子，单位样本算术更重。
- 固定按最多 1024 个像点分块，临时张量约为 `[image_chunk, valid_pupil]`。
- 不支持 design、视场或波长联合 batch，但单设计高采样的峰值显存有明确上界。

Optiland NumPy Huygens 使用 `Numba @njit(parallel=True, fastmath=True)`，直接循环像点与 pupil 点，不创建同样大小的二维临时 Tensor。这是其 CPU 和低显存运行的重要优势。

### 5.4 PSF 理论判断

- **默认直接 Huygens**：optics_core 预计显著更快，首要原因是默认相互作用数约少 99 倍，而非单纯内核优化。
- **同有效 pupil 点数、同像面网格、FP64、只算实际 PSF、单设计**：optics_core 的局部平面波公式算术更轻，Optiland 则已有像点分块。总体倾向 optics_core，但不应在没有实测时给出倍数。
- **同 pupil/image 网格边长的公开 PSF**：Optiland 会先裁掉圆外 pupil 点，而 optics_core 当前仍处理完整方形网格；optics_core 还计算完整理想 PSF，因此优势会进一步缩小。
- **多设计或全波长**：optics_core 明显占优。Optiland 需要为每个设计、视场和波长重新构造分析对象并执行追迹与积分。
- **多个视场的公开 PSF 导出**：双方当前高层接口都存在视场循环；optics_core 仍能在每个视场内批量 design 和全部波长。其 MTF 内部路径则会一次追迹多个视场，再按视场分块积分。
- **单设计超高采样**：Optiland Torch 的像点分块和 NumPy Numba 路径更稳健；optics_core 当前的 design minibatch 无法解决单设计 kernel 本身过大的问题。
- **允许 FFT/MMDFT**：Optiland 通常会远快于直接 Huygens，尤其图像网格增大时。`ScalarFFTPSF(num_rays=128)` 的默认规则实际使用 64×64 pupil 和 256×256 FFT 网格，复杂度接近 `O(N² log N)`；MMDFT 还能直接利用高效 GEMM。这是算法选择优势，不代表 Zemax 风格 Huygens 对标更快。

## 6. MTF 计算效率

### 6.1 optics_core

当前链路为：

```text
多设计、多视场、单波长/全波长 Huygens PSF
→ 横纵积分得到两条 LSF
→ 补零到不小于 8I 的 2 次幂
→ 两次 torch.fft.rfft
→ 复 OTF 插值到目标频率
```

MTF 路径明确关闭理想 PSF 和 Strehl。PSF、LSF、FFT 和插值均在当前设备上按 design/field 执行，最后统一回收 CPU。

### 6.2 Optiland

`ScalarHuygensMTF` 的流程为：

```text
先构造一个轴上 ScalarHuygensPSF 取得 normalization
→ 逐视场重新构造 ScalarHuygensPSF
→ 每个 PSF 做完整二维 FFT
→ 读取中心行、中心列并归一化
```

有三个直接影响性能的细节：

1. 归一化用的 `norm_psf` 会先计算完整轴上 PSF，而不是只生成理想中心值；若视场列表包含轴上视场，该实际 PSF 随后还会再计算一次。
2. 视场通过 Python 循环执行，`BaseMTF` 只解析一个波长，不支持 optics_core 的全波长共同网格混合。
3. 已有二维 PSF 后仍执行完整二维 FFT，而目标只使用中心行和中心列。

Optiland 另有 `ScalarFFTMTF`，其 PSF 来自 FFT 瞳函数。该路径比直接 Huygens MTF 快得多，但仍是不同计算方法。

### 6.3 MTF 理论判断

- **同采样 Huygens MTF、单设计**：optics_core 预计领先。PSF 阶段省去额外归一化 PSF，后处理又从二维 FFT 降为两次一维 FFT。
- **多视场**：Optiland 需要约 `F+1` 次完整 PSF；optics_core 需要 `F` 次实际 PSF，并共享一次多视场/多波长前置追迹。
- **多设计、全波长**：optics_core 具有结构性优势，预计差距进一步扩大；Optiland 当前没有等价的 design batch 和全波长混合 MTF。
- **只比较已有 PSF 到 MTF**：小图像、单样本时二维 FFT 本身很快，差异可能不明显；批量增大后，optics_core 的一维 batched FFT 和无中途 CPU 同步更有优势。
- **允许 FFTMTF**：Optiland 的绝对耗时可能明显更低，因为避免了 `O(P·I²)` 的直接积分。该结果不能与 optics_core 的 Huygens MTF 作为同物理算法比较。

## 7. CPU、GPU 与精度影响

### CPU

- Optiland 更占优势：NumPy 是一等后端，直接 Huygens 还有 Numba 多核实现。
- optics_core 虽能在 Torch CPU 上运行，但设计重点是大批量 Tensor/GPU；单系统、小数组容易被 Torch 调度和通用 frame 开销主导。

### V100 等 FP64 GPU

- 公平比较应把 Optiland Torch 也设置为 FP64/complex128。
- optics_core 能通过 design batch 提供足够大的工作集，更容易持续占满 GPU。
- Optiland 单系统 Torch 路径可能受视场/波长循环、标量同步和逐面记录限制；其 MMDFT 的矩阵乘法则能较好利用 GPU，但属于另一种 PSF 算法。

### 消费级 GPU

- Optiland 默认 Torch FP32 在显存和峰值吞吐上更有利，但数值精度与 optics_core FP64 不同。
- 若强制双方 FP64，消费级 GPU 的 FP64 吞吐会同时限制两者；optics_core 的大 design batch 只能提高利用率，不能改变硬件峰值差距。

## 8. 场景化结论

| 场景 | 预计占优方 | 判断依据 |
| --- | --- | --- |
| 单系统、默认参数 spot | Optiland | 默认只有 127 条光线，且可用 NumPy/FP32；采样量不等价 |
| 单系统、同 FP64/同光线数 spot | 不确定 | Optiland 单系统路径更轻，但有同步和逐面记录；需实测 |
| 数百至数千设计 spot | optics_core | 原生 design batch，避免逐设计和逐 F/W 的完整调用链 |
| 默认参数直接 Huygens PSF | optics_core | 公开 PSF 的主相互作用数仍约少 99 倍 |
| 同有效采样直接 Huygens PSF、单设计 | optics_core 倾向占优 | 相位公式更轻；完整理想 PSF 会缩小公开接口的优势 |
| 单设计超高采样 Huygens PSF | Optiland 更稳健 | Torch 像点分块、NumPy Numba 低临时内存 |
| 多设计、全波长 Huygens PSF | optics_core | design/wavelength batch 和自动 design minibatch |
| 单系统 FFT/MMDFT PSF | Optiland | 已有渐近复杂度更低的实现；optics_core 暂无同类接口 |
| 同算法 Huygens MTF | optics_core | 无冗余归一化 PSF，两次一维 FFT，支持多设计/全波长 |
| 单系统 FFT MTF | Optiland | FFT PSF 路径避免直接 Huygens 积分 |
| CPU 单系统分析 | Optiland | NumPy/Numba 后端成熟 |
| 大规模装配公差前向评价 | optics_core | 参数矩阵直接形成多系统 batch；Optiland Monte Carlo 逐次变更 `Optic` |

本项目更准确的定位是：

> optics_core 面向共享拓扑的大规模 FP64 多设计仿真；Optiland 面向单系统、可切换数值后端和多种光学分析算法。

在当前重点案例——1000 个以上装配公差设计、3 个视场、全部波长、32×32 Huygens PSF/MTF——optics_core 具有明显的并行架构优势。Optiland 值得借鉴的是 NumPy/Numba CPU 路径、FFT/MMDFT 快速 PSF、Huygens 像点分块，以及更完整的标量/矢量分析方法；其当前多配置和 Monte Carlo 组织方式不适合作为本项目的 design batch 替代方案。

## 9. 公平实测建议

若后续需要报告可信的速度倍率，建议拆成四组基准：

1. **纯追迹**：同一纯球面 Double Gauss、FP64、相同 pupil 坐标，分别测试 `D=1/32/1024` 和固定 `F/W/P`。
2. **Spot**：统一参考点和 RMS 定义，禁止使用双方默认的不同 ring 数。
3. **直接 Huygens PSF**：统一有效 pupil 点、物理像面网格、仅计算实际 PSF，并单独记录归一化/理想 PSF 的耗时。
4. **MTF 后处理**：给两边输入同一批 PSF，分别测二维 FFT 与 LSF+一维 FFT；完整 Huygens MTF 另行计时。

测试时还应：

- 关闭绘图和文件导出。
- Torch 双方统一 FP64，CUDA warm-up 后同步计时。
- 同时报告有效光线数、PSF 相互作用数、峰值显存和中位耗时。
- Optiland 的 `D>1` 使用当前公开接口逐设计循环，不能把多个独立进程的总吞吐误认为单 GPU design batch。
- FFT/MMDFT 单独作为“快速算法组”，不与直接 Huygens 合并计算加速比。

## 10. 主要源码依据

optics_core：

- `optics_core/system.py`：`MultiOpticalSystem`、参数矩阵和 design view。
- `optics_core/system_state.py`：准备态 Tensor 的连续 design 切片。
- `optics_core/first_order.py`：当前参数收集和设备 Tensor 构造。
- `optics_core/tracing/_core.py`、`_dispatch.py`：多维顺序追迹和 frame 变换。
- `optics_core/spot_diagram.py`：批量 spot 追迹与指标统计。
- `optics_core/wavefront.py`：PSF 前置追迹与像面波数据。
- `optics_core/huygens_psf.py`：Huygens 积分和自动 design minibatch。
- `optics_core/huygens_mtf.py`：LSF、一维 FFT 和频率插值。
- `examples/batch_spot.py`、`batch_psf.py`、`batch_mtf.py`：当前批量案例。

Optiland：

- `optiland/backend/__init__.py`、`backend/torch_backend/config.py`：NumPy/Torch 后端和精度配置。
- `optiland/rays/real_rays.py`、`raytrace/real_ray_tracer.py`：一维光线数据和 field×pupil 展平。
- `optiland/surfaces/surface_group.py`、`surfaces/standard_surface.py`：逐面追迹和逐面光线记录。
- `optiland/coordinate_system.py`、`geometries/standard.py`、`geometries/newton_raphson.py`：坐标变换和 Torch 标量条件。
- `optiland/analysis/spot_diagram/core.py`、`distribution.py`：spot 的 F/W 循环和默认采样。
- `optiland/psf/huygens_fresnel.py`、`huygens_fresnel_strategies.py`：直接 Huygens PSF 与后端策略。
- `optiland/psf/fft.py`、`psf/mmdft.py`：FFT 和矩阵 DFT PSF。
- `optiland/mtf/huygens_fresnel.py`、`mtf/fft.py`、`mtf/base.py`：Huygens/FFT MTF。
- `optiland/multiconfig/multi_configuration.py`、`tolerancing/monte_carlo.py`：多配置对象列表和逐次 Monte Carlo。
