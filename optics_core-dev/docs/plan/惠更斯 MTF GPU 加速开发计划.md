# 惠更斯 MTF GPU 加速开发计划

## 1. 目标

本文针对以下固定案例分析和规划加速工作：

```text
设计数量：6561
视场数量：3
波长数量：3，使用全部波长混合
光瞳采样：32×32
像面采样：32×32
输出频率数量：301
计算精度：FP64
```

目标分为两层：

1. 删除 MTF 链路中不需要的 PSF、Strehl 和中间张量计算，提高所有 GPU 的吞吐率。
2. 重构惠更斯积分的计算形式，使核心负载从逐点三角函数和大型复数张量读写转向 FP64/complex128 矩阵乘加，进一步发挥 V100 的 FP64 优势。

本轮不降低几何、材料、追迹和惠更斯积分的 FP64 精度，也不改变现有 PSF/MTF 数值定义。

## 2. 当前实测基线

### 2.1 V100 SXM2 16 GB

```text
design_batch_size=32
minibatch_count=206
elapsed_seconds=55.339
designs_per_second=118.560
peak_allocated_gib=8.332
peak_reserved_gib=12.154
```

重复测试吞吐率约为：

```text
110～140 designs/s
```

### 2.2 RTX 4090 D

```text
design_batch_size=48
minibatch_count=137
elapsed_seconds=43.051
designs_per_second=152.401
peak_allocated_gib=12.485
peak_reserved_gib=17.035
```

4090 D 相对当前 V100 基线快约：

```text
152.401 / 118.560 = 1.29 倍
```

两次运行的峰值 allocated 显存除以 batch size 后几乎一致：

```text
V100：  8.332 / 32 ≈ 0.260 GiB/design
4090D：12.485 / 48 ≈ 0.260 GiB/design
```

这说明当前自动探测结果主要受显存容量约束。4090 D 可以同时容纳约 1.5 倍设计，并减少约三分之一 minibatch，但这不足以单独解释全部性能差异。

为了准确区分架构速度与 batch size 影响，后续性能测试必须补充 V100 和 4090 D 在相同 `design_batch_size=32` 下的内部基准。该固定 batch size 只用于 benchmark，不增加公开设置参数。

## 3. 为什么 V100 的 FP64 优势尚未体现

NVIDIA 官方资料给出的 V100 SXM2 峰值能力为：

```text
FP64：最高 7.8 TFLOPS
显存带宽：最高 900 GB/s
```

V100 的确拥有面向 HPC 的高吞吐 FP64 运算单元。然而，理论 FP64 峰值主要描述规则的加法、乘法和 FMA 吞吐率，不能直接代表任意 FP64 程序的速度。

当前 `_huygens_integral()` 为每个设计、视场、波长、光瞳点和像面点构造：

```text
phase_tilt
phase
ideal_phase
kernel
ideal_kernel
```

核心形状近似为：

```text
(design, field_block=1, wavelength=3, pupil=1024, image_y=32, image_x=32)
```

每个设计每个视场约有：

```text
3 × 1024 × 32 × 32 ≈ 315 万
```

个光瞳到像面贡献。当前每个贡献至少执行实际相位和理想相位两组 `cos`、`sin`，并生成 complex128 中间结果。

因此当前瓶颈更接近：

1. FP64 三角函数吞吐。
2. 大型 complex128 临时张量的生成和显存读写。
3. 多个 PyTorch elementwise kernel 的启动和中间结果物化。
4. 最后沿 pupil 维执行规约。

它并不是以 FP64 FMA 为主的高算术强度内核。4090 D 虽然通用 FP64 乘加能力弱于 V100，但拥有更多并行执行资源、更高时钟、更大的缓存和显存容量；对于三角函数、elementwise、规约和大规模并行任务，仍可能取得更高的端到端吞吐率。

结论是：不能通过单纯增大 V100 batch size 挖掘 FP64 优势，必须改变惠更斯积分的计算形式。

## 4. 第一阶段：删除 MTF 链路冗余

这一阶段风险最低，应先实施并建立新的性能基线。

### 4.1 MTF 不计算理想 PSF 和 Strehl

当前 MTF 最终只消费混合后的二维 PSF，但共享的 PSF 计算路径仍然生成：

```text
ideal_phase
ideal_kernel
ideal_psf_by_wavelength
strehl_by_wavelength
strehl_ratio
```

增加一个简洁的内部计算选项，使 MTF batch 路径只返回：

```text
mixed_psf
pixel_pitch_um
wavelength_indices
```

公开 PSF 分析继续计算理想 PSF 和 Strehl，行为保持不变。

不要为此复制一套惠更斯算法。建议让底层积分支持：

```python
compute_ideal_psf: bool
```

并让 PSF 与 MTF 使用同一套光线追迹、相位和积分实现。

预期收益：

1. 删除约一半三角函数和 complex128 kernel 构造。
2. 降低峰值显存，使自动 design batch size 可能增大。
3. 全 GPU 均可受益，预计是当前最确定的优化项。

### 4.2 删除重复 mask 和大型无效张量

当前无效光线已经通过零积分权重表达，之后再次对 `kernel` 和 `ideal_kernel` 执行 `torch.where()` 没有必要。

应保留一次权重构造：

```python
integration_weight = where(valid_points, pupil_weights, 0)
integration_weight /= integration_weight.sum(...)
```

后续直接使用该零权重参与积分。

同时检查以下张量是否可以通过广播表达，避免实际展开或重复构造：

```text
image_grid
grid_offset
valid mask
wave_number
integration_weight
```

### 4.3 避免每批同步回收造成流水线停顿

当前每个 minibatch 计算 MTF 后立即执行 `.cpu()`，会形成 GPU 到 CPU 的同步点。

最终 MTF 结果约为：

```text
6561 × 3 × 301 × 2 × 8 bytes ≈ 90 MiB
```

可以选择：

1. 在显存允许时预分配完整 GPU MTF 输出，按 design 区间写入，最后一次性回收。
2. 或预分配 pinned CPU 输出，使用独立 CUDA stream 异步复制上一批结果，并与下一批计算重叠。

优先实现第一种简单方案。若 V100 显存压力影响 batch size，再使用 pinned CPU 双缓冲。

### 4.4 batch size 按吞吐率选择

当前 batch size 由单设计峰值显存线性估算，只回答“能放多少”，没有回答“哪个最快”。

建议保留显存估算作为上界，在首次 benchmark 中测试少量候选值：

```text
estimated
estimated × 3/4
estimated × 1/2
```

每个候选只运行相同数量的完整 minibatch，使用 CUDA event 计时，选择 designs/s 最高者。本轮仍不增加用户公开参数，也不做跨进程缓存。

该优化应在删除冗余张量后进行，否则探测的是即将被替换的显存模型。

## 5. 第二阶段：将惠更斯积分改写为可分离相位矩阵乘

这是发挥 V100 FP64 优势的核心优化。

### 5.1 相位可分离形式

规则像面网格可以写为：

```text
P(x, y) = Pchief + x·ex + y·ey
```

对第 `r` 根光线：

```text
phase(r, x, y)
  = k·relative_path(r)
  + k·dot(direction(r), ex)·x
  + k·dot(direction(r), ey)·y
```

定义：

```text
A(r)    = weight(r) · exp(i·k·relative_path(r))
Ex(r,x) = exp(i·k·dot(direction(r), ex)·x)
Ey(r,y) = exp(i·k·dot(direction(r), ey)·y)
```

则复振幅为：

```text
amplitude(y,x) = Σr Ey(r,y) · A(r) · Ex(r,x)
```

对每个 design、field、wavelength，可整理为：

```text
amplitude = transpose(Ey) @ (A[:, None] * Ex)
```

矩阵形状为：

```text
(32, 1024) @ (1024, 32) -> (32, 32)
```

实际 PSF 与当前直接积分在数学上等价，没有引入 FFT 近似，也不改变像面采样网格。

### 5.2 为什么该形式更适合 V100

与当前实现相比：

1. 三角函数数量从每根光线的 `32×32` 个像面点，降低为 `32+32` 个轴向点。
2. 不再物化 `(pupil, image_y, image_x)` complex128 kernel。
3. 光瞳求和改为 batched complex128 GEMM。
4. 核心计算从特殊函数和显存流量转向规则的 FP64 乘加。
5. cuBLAS 可以对规则矩阵乘进行成熟的 tiling、共享内存复用和指令调度。

对于 32×32 网格，单个矩阵较小，但 batch 维包含：

```text
design_batch × field × wavelength
```

V100 当前 batch 32 时即有：

```text
32 × 3 × 3 = 288
```

个独立矩阵乘，可以使用 `torch.bmm()` 或等价的 strided batched GEMM 一次提交，具备足够批量并行度。

这条路线才可能使 V100 的 FP64 单元成为主要性能因素，并有机会缩小乃至反转当前与 4090 D 的差距。

### 5.3 理想 PSF 复用

公开 PSF 分析仍需要理想 PSF。理想振幅使用相同的 `Ex` 和 `Ey`，只需令：

```text
Aideal(r) = weight(r)
```

因此可复用轴向相位矩阵，不再重复计算 `ideal_phase`、`cos` 和 `sin`。

MTF 路径则完全跳过理想振幅矩阵乘。

### 5.4 数值注意事项

实现时必须：

1. `relative_path`、轴向相位和权重保持 FP64。
2. `Ex`、`Ey`、`A` 和振幅保持 complex128。
3. 保留当前有效光瞳权重归一化。
4. 保持当前主波长主光线共同网格及多波长混合顺序。
5. 不使用 TF32、FP32 或 Tensor Core 近似。

矩阵乘会改变求和顺序，结果不要求 bitwise 一致。应使用与 Zemax 误差相比足够严格、但允许 FP64 规约顺序差异的容差进行验证。

## 6. 第三阶段：必要时使用融合 CUDA 内核

如果 batched complex128 GEMM 在 32×1024×32 的小矩阵上未达到预期，可进一步实现专用 CUDA extension：

1. 每个 block 负责一个或多个像面 tile。
2. 光线方向、相对光程和权重按块载入 shared memory。
3. 在寄存器中累计 complex128 振幅。
4. 使用规则网格相位递推，减少 `sincos` 调用。
5. 直接输出强度，不写回完整 complex kernel。

这一方案可能获得更高绝对性能，但开发、编译和维护成本明显高于 PyTorch + cuBLAS。只有在 profiler 证明小矩阵 GEMM 效率不足后才实施。

不建议优先使用 Triton：当前核心数据类型是 complex128，且目标包含 V100；应先确认所用 Triton 版本对 complex128 和目标架构具有可靠支持。

## 7. 不应优先投入的方向

### 7.1 单纯继续增大 design batch size

batch size 32 已经提供数百个 design-field-wavelength 工作项。继续增大 batch 可以减少 Python 循环，但不能改变每个贡献的三角函数和显存流量。

4090 D 的 batch size 48 主要来自更大显存，并不说明 batch 48 一定比 batch 32 有更高单位吞吐。

### 7.2 只优化 MTF FFT

当前 MTF 对 32 点 LSF 补零到 256 点并执行一维 FFT。其计算量远小于：

```text
1024 pupil rays × 1024 image pixels × 3 wavelengths
```

优化 FFT、插值或 301 个频率输出不会显著改变端到端吞吐率。

### 7.3 CUDA Graph 作为首要方案

CUDA Graph 可以减少 kernel launch 和 Python 调度开销，但当前单个 minibatch 耗时约数百毫秒，主要成本仍在积分内核。它适合作为积分重构完成后的末端优化，而不是当前首要矛盾。

### 7.4 降低精度

FP32 或混合精度可能显著偏向 4090 D，且会引入相位精度风险，不符合项目统一 FP64 以及与 Zemax 对标的要求。

## 8. Profiling 与实施顺序

### 阶段 A：建立可解释基线

1. 在两台 GPU 上分别固定 batch size 32。
2. 使用 CUDA event 分段统计：
   - 光线构造。
   - 光线追迹。
   - 实际惠更斯积分。
   - 理想 PSF/Strehl。
   - 多波长混合。
   - MTF 投影、FFT 和插值。
   - GPU 到 CPU 回收。
3. 使用 `torch.profiler` 或 Nsight Systems 记录 kernel 时间。
4. 使用 Nsight Compute 抽查积分内核的：
   - FP64 指令吞吐率。
   - DRAM 吞吐率。
   - special-function 指令占比。
   - occupancy。
   - kernel launch 数量。

### 阶段 B：低风险通用优化

1. MTF 跳过理想 PSF 和全部 Strehl。
2. 删除重复 mask。
3. 减少中间张量物化。
4. 优化最终结果回收。
5. 重新探测吞吐最优 batch size。

完成后重新记录 V100 与 4090 D 基线。

### 阶段 C：可分离相位 GEMM

1. 新增内部可分离积分实现。
2. 保留当前直接积分作为测试期间的参考实现。
3. 验证单色、多色、轴上和离轴 PSF。
4. 验证 MTF 曲线。
5. 对比两台 GPU 的积分阶段和端到端吞吐率。
6. 验证通过后删除旧积分实现，避免长期保留两套算法。

### 阶段 D：按 profiler 决定后续

只有在阶段 C 后仍存在明确瓶颈时，再选择：

```text
专用融合 CUDA 内核
CUDA Graph
pinned CPU 双缓冲
跨调用 batch-size 缓存
```

## 9. 最小测试集

### 9.1 数值测试

1. 可分离积分与原直接积分对比：
   - 单设计、单视场、主波长。
   - 单设计、离轴视场、全部波长。
   - 多设计、多视场、全部波长。
2. 新旧 PSF 峰值、总能量和二维分布误差。
3. 新旧 S/T MTF 全频率曲线误差。
4. 现有 Zemax PSF/MTF 回归测试无新增失败。

### 9.2 批处理测试

1. minibatch 与单批结果一致。
2. 最后一批不足 batch size 时顺序和 shape 正确。
3. MTF 路径不计算 ideal PSF/Strehl。
4. OOM 减半逻辑保持有效。

### 9.3 性能测试

统一使用本文开头的 6561-design 案例，分别记录：

```text
GPU 型号
固定 batch=32 吞吐率
自动 batch size
自动 batch 吞吐率
积分阶段耗时
追迹阶段耗时
MTF 阶段耗时
峰值 allocated/reserved 显存
```

性能测试需要至少预热一次，再连续执行三次，报告中位数和范围。

## 10. 完成条件

1. MTF 不再计算未使用的 ideal PSF 和 Strehl。
2. 惠更斯积分不再物化完整 `(pupil, image_y, image_x)` complex128 kernel。
3. 核心光瞳积分由 batched complex128 GEMM 或性能更高的等价融合内核完成。
4. FP64/complex128 精度约定保持不变。
5. 与 Zemax 的 PSF/MTF 精度无显著退化。
6. V100 和 4090 D 均获得明确吞吐提升。
7. V100 相对 4090 D 的差距明显缩小；若未缩小，profiler 能明确证明剩余瓶颈位于非 FP64 FMA 部分。

## 11. 预期判断

第一阶段删除冗余理想 PSF 后，两台 GPU 都应明显提速，但 4090 D 仍可能保持领先，因为算法形态尚未根本改变。

第二阶段的可分离相位 GEMM 同时减少三角函数、显存流量和临时张量，并把主要工作转化为 FP64/complex128 矩阵乘加，是最有希望发挥 V100 HPC 架构优势的方案。

在完成固定 batch size、分阶段 profiler 和 GEMM 原型测试前，不应直接承诺 V100 一定超过 4090 D。端到端结果还会受到光线追迹、特殊函数、矩阵尺寸、时钟和框架调度影响。但相比继续调整 batch size，这条路线具有更明确的硬件依据和更高的潜在收益。

## 12. 参考资料

1. NVIDIA Tesla V100 Performance Guide：V100 最高 7.8 TFLOPS FP64、900 GB/s 显存带宽。  
   https://images.nvidia.com/content/pdf/volta-marketing-v100-performance-guide-us-r6-web.pdf
2. NVIDIA Volta Architecture Whitepaper：V100 HBM2 与 Volta 计算架构。  
   https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
3. NVIDIA GeForce RTX 4090 官方规格：Ada 架构、CUDA core、时钟和显存配置。4090 D 的实测数据以本项目日志为准。  
   https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
4. NVIDIA CUDA C++ Best Practices Guide：occupancy、指令延迟、显存访问和性能分析原则。  
   https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
