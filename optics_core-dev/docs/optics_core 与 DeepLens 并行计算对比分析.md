# optics_core 与 DeepLens 并行计算对比分析

## 1. 对比范围

本文对比以下源码快照：

- optics_core：提交 `1b157c3`。
- AutoLens DeepLens：`C:\Users\Huwei\Project\OpenSource\AutoLens\deeplens`，提交 `e58db32`。

重点分析前向仿真的并行数据模型，以及 spot、PSF、MTF 的计算效率。文中的性能结论主要来自源码结构和计算复杂度；除特别标注的本项目既有 V100 记录外，不代表同机实测结果。

## 2. 核心结论

两套框架的“batch”含义不同：

- optics_core 把 `design` 作为一级批量维度，同一套表面拓扑下，不同曲率、厚度、偏心和倾斜可以同时计算。
- DeepLens 的 Ray 支持多个视场点和多条光线并行，但一个 `GeoLens` 仍表示一个镜头，表面曲率和位置主要是标量 Tensor。多个不同镜头通常需要在 Python 中逐个修改和计算。

因此不能笼统地判断谁更快：

1. 对大量同拓扑设计执行装配公差 Monte Carlo、批量 spot 或惠更斯 MTF，本项目具有结构性优势。
2. 对单个镜头执行 FP32 几何追迹、几何 PSF 或几何 MTF，DeepLens 很可能更快。
3. 对等价的直接惠更斯积分，在当前默认采样下，本项目的计算量远小于 DeepLens，并且支持 design、视场和波长批量，理论上明显领先。
4. DeepLens 的默认 MTF 来自几何 PSF，本项目的 MTF 来自惠更斯 PSF。前者绝对耗时更低，但不包含相同的衍射计算，不能直接作为性能胜负依据。
5. DeepLens 已具备单镜头自动微分、Adam、L-BFGS 和 LM；本项目当前前向批量能力更强，但参数读取仍会切断自动微分链路。

## 3. 并行数据模型

| 维度 | optics_core | DeepLens |
| --- | --- | --- |
| 不同光学设计 | 原生 `design` 维度，参数矩阵和准备态按 design 批量 | 一个 `GeoLens` 保存一组标量表面参数，无原生 lens/design batch |
| 多视场 | 追迹张量包含 field 维度 | Ray 允许任意前导维度，spot 和几何 PSF 可批量视场 |
| 多波长 | 追迹张量包含 wavelength 维度，可一次追迹 | `Ray.wvln` 是单波长标量，分析通常在 Python 中循环波长 |
| 多光线 | pupil/ray 维度 Tensor 化 | ray/spp 维度 Tensor 化 |
| 显存控制 | PSF/MTF 自动探测 design batch size，OOM 后减半重试 | 无 design minibatch；惠更斯积分固定按最多 10000 条光线分块 |
| 精度 | 几何和波动计算统一 FP64，惠更斯使用 complex128 | 几何路径默认 FP32；相干和惠更斯路径要求 FP64/complex128 |
| 输出回收 | MTF 在 GPU 完成后统一回收 CPU；PSF 按 minibatch 回收 | MTF 将 PSF 同步到 CPU，用 NumPy FFT |

### 3.1 optics_core

本项目光线张量的主要逻辑形状为：

```text
[design, field, wavelength, pupil_ray]
```

追迹时仍按表面顺序执行 Python 循环，但每个表面内核同时处理所有 design、视场、波长和光线。`design_batch_view()` 对准备态 Tensor 使用连续切片，FrameData 和 FirstOrderData 不发生实体复制。

PSF/MTF 通过 `HuygensPSFDesignBatchIterator` 自动测量单设计峰值显存，选择 design minibatch；运行中若仍 OOM，则将 batch size 减半重试。这使大量设计可以持续利用 GPU，同时避免一次性展开全部设计。

当前仍有两个明显开销：

1. `surface_value()` 会遍历每个 design 的 Python 参数列表，再通过 `torch.tensor(..., device=...)` 构造设备 Tensor，存在主机端组装和 CPU→GPU 传输。
2. 普通轴对称系统也走通用局部 frame 变换，每个表面包含 `einsum` 坐标变换；目前没有“无偏心、无倾斜”的轴对称快速路径。

这些开销在单设计、少光线分析中较明显，但在大量 design 或大量光线时能够被大张量计算摊薄。

### 3.2 DeepLens

DeepLens 的 Ray 使用：

```text
[*batch_shape, ray, xyz]
```

因此多个视场、物点或图像像素可以并行追迹。`sample_grid_rays()`、`psf_geometric()` 和 `psf_map()` 都利用了这种能力，几何 PSF 的 `forward_integral()` 也通过带 batch 索引的 `index_put_(accumulate=True)` 一次累积多个视场。

但 `GeoLens.surfaces` 是表面对象列表，表面曲率 `c`、位置 `d` 等主要是标量 Tensor。部分分支还直接对标量曲率执行 Python 判断，因此不能只给这些参数增加一个 design 维度，就得到多镜头并行。

这一点在 `tolerancing_monte_carlo()` 中表现得很直接：每个 trial 都依次修改同一个镜头、重新聚焦、计算多个视场，再进入下一次 trial。它是“单镜头内部光线并行”，不是“多个公差镜头并行”。

DeepLens 对单个中心系统有更轻的执行路径：表面参数已经驻留在 GPU，轴对称面不需要每次执行批量 frame 矩阵变换，几何分析默认 FP32。这些因素有利于单镜头吞吐。

## 4. Spot 对比

### 4.1 算法差异

optics_core：

- 一次追迹全部 design、视场和波长。
- 默认 hexapolar `ray_density=30` 时，普通光线约为 `1 + 6 × (1+...+30) = 2791` 条，另加参考主光线。
- 多波长统一相对主波长主光线像点统计 RMS。
- 统一使用 FP64。

DeepLens：

- 一次追迹多个视场，但 `analysis_spot()` 在 Python 中逐波长追迹。
- 默认每个视场、每个波长随机采样 16384 条光线。
- 将所有波长有效光线合并后，以共同质心统计 RMS。
- 几何追迹默认 FP32。

两者默认的 pupil 采样数、采样分布和 RMS 参考点不同，默认结果和耗时都不能直接比较。

### 4.2 复杂度

设 design 数为 `D`、视场数为 `F`、波长数为 `W`、光线数为 `P`、表面数为 `S`：

```text
optics_core: O(D · F · W · P · S)，一次批量链路
DeepLens:    O(D · F · W · P · S)，但 D 和 W 通常位于 Python 循环外层
```

### 4.3 理论性能判断

- 单设计、相同光线数：DeepLens 很可能略快。其默认 FP32、标量表面参数和轴对称快速执行路径占优；本项目存在参数组装和通用 frame 变换开销。
- 使用各自默认采样：本项目光线数约为 DeepLens 的 1/5.9，可能抵消 FP64 和通用 frame 的成本，但两者采样噪声和准确性不同。
- 大量设计：本项目预计明显领先。DeepLens 需要重复 Python 调用、修改镜头状态并启动追迹，而本项目可以在每个表面内核中同时处理多个设计。
- 多波长：本项目进一步占优，因为材料折射率和追迹都包含 wavelength 维度；DeepLens 当前按波长循环。

## 5. PSF 对比

### 5.1 两者默认 PSF 并不等价

optics_core 当前公开 PSF 是 Zemax 风格的惠更斯 PSF：

1. 在规则入瞳网格追迹到像面。
2. 使用像面交点、OPL 和方向余弦构造局部平面波相位。
3. 在共同像面网格直接积分复振幅。
4. 支持单波长或全部波长强度混合。
5. 默认同时计算理想 PSF 和 Strehl，不对最终 PSF 归一化。

DeepLens 提供三种 PSF：

1. `geometric`：几何光线落点的双线性 scatter，默认模型。
2. `coherent`：出口瞳复场加 ASM 传播。
3. `huygens`：把出口瞳光线作为球面次波源，使用 `OPL+r`、倾斜因子和 `1/r` 衰减直接积分。

DeepLens 源码明确说明其 Huygens 实现不同于 Zemax Huygens。它只支持单物点、单波长，并把 PSF 能量归一化为 1。因此公平对比必须指定 `model="huygens"`，并统一采样、网格和输出语义。

### 5.2 计算复杂度

直接惠更斯积分的主项为：

```text
O(D · F · W · P · I²)
```

其中 `I × I` 为像面网格。

本项目默认 `P≈32²`、`I=32`。DeepLens Huygens 的配置写为 `SPP_COHERENT = 2 << 23`，Python 实际值是 `16,777,216`，尽管源码行尾注释误写为 `8,388,608`；默认 `I=64`。对单设计、单视场、单波长，仅比较相位样本数：

```text
optics_core 实际 PSF:  32² × 32²        ≈ 1.05 × 10⁶
DeepLens Huygens:      16,777,216 × 64² ≈ 6.87 × 10¹⁰
```

DeepLens 默认约多 65536 倍相位样本。本项目公开 PSF 还会计算一次理想 PSF，因此主积分约翻倍；即便如此，DeepLens 默认主积分量仍约高 32768 倍。这个比例只说明默认计算量，不能证明两种采样具有相同误差。

### 5.3 内核与显存

本项目：

- 同时批量 design 和波长。
- 为控制峰值显存，按视场循环积分并自动拆分 design。
- 当前会完整物化一个 design minibatch 的 `[B, W, P, I, I]` 相位和 complex128 kernel。
- 尚未按 pupil ray 分块；高 pupil/image 采样下，即使 `design_batch_size=1` 也可能 OOM。

DeepLens：

- Huygens 只处理一个镜头、一个物点和一个波长。
- pupil ray 固定按最多 10000 条分块，峰值显存近似 `O(10000 · I²)`，可处理非常高的总光线数。
- 每个 ray chunk 都有 Python 循环，并计算 `sqrt`、`fmod`、`1/r` 和复指数，单位相位样本的算术量高于本项目的局部平面波形式。

### 5.4 理论性能判断

- 与 DeepLens 默认几何 PSF 比：DeepLens 会快很多，因为几何 scatter 是 `O(P)`，本项目直接惠更斯积分是 `O(P·I²)`；但前者不包含衍射相干叠加。
- 与 DeepLens Huygens 默认配置比：本项目预计大幅领先，主要来自采样量差异、较轻的相位公式，以及 design/波长批量。
- 统一 `P`、`I`、FP64 且只算实际 PSF：本项目的局部平面波内核算术更少，理论上仍有优势；但公开 PSF 默认额外计算理想 PSF，会削弱这一优势。
- 超高采样单设计：DeepLens 的 pupil chunking 更稳健；本项目当前可能先遇到单设计 OOM。
- 若允许使用 ASM/FFT 而不要求 Zemax Huygens 语义，DeepLens 的 coherent 路径具有更好的渐近复杂度，尤其在高像面分辨率下可能优于直接积分。

## 6. MTF 对比

### 6.1 optics_core

本项目 MTF 链路为：

```text
多设计、多视场、单波长/全波长惠更斯 PSF
→ 横纵向积分得到 LSF
→ 约 8 倍补零
→ GPU torch.fft.rfft
→ 复 OTF 插值到目标频率
```

MTF 路径会跳过理想 PSF 和 Strehl。PSF、LSF、FFT 和频率插值都在 GPU 端按 `[design, field]` 批量执行，最后统一回收 CPU。

### 6.2 DeepLens

DeepLens `mtf()` 调用未指定模型的 `self.psf()`，因此默认使用几何 PSF，然后：

```text
几何 PSF
→ 同步到 CPU NumPy
→ 横纵向积分
→ 未补零的 numpy.fft.rfft
```

`mtf()` 只处理一个视场和一个波长；`draw_mtf()` 在 Python 中循环视场、深度和波长。它返回原生频率点，不执行本项目的 8 倍补零和复数频率插值。

### 6.3 理论性能判断

- 当前默认调用的绝对耗时：DeepLens 很可能明显更快，因为它计算的是几何 MTF，不是惠更斯 MTF。
- 等价惠更斯 MTF：总耗时由 PSF 主导，本项目在大量设计、多视场和全波长场景预计明显领先。
- 只比较“已有 PSF → MTF”：DeepLens 的单个小 PSF 使用 NumPy FFT 可能更轻；本项目有 GPU 启动、8 倍补零和插值开销。批量增大后，本项目的 GPU Tensor 路径更有优势，也避免每个视场/波长的 GPU→CPU 同步。
- DeepLens 当前没有“全部波长先在共同网格混合，再计算 MTF”的等价公开路径。

## 7. GPU 架构影响

### V100

本项目所有几何和波动路径使用 FP64，能够利用 V100 较强的 FP64 算力。DeepLens 默认几何 spot/PSF/MTF 使用 FP32，不能特别利用 V100 的 FP64 优势；若切换到 DeepLens Huygens，则双方都会使用 FP64/complex128。

本项目已有 `examples/output/batch_mtf_v100.json` 记录：

```text
design=6561, field=3, wavelength=3
pupil=32×32, image=32×32
elapsed=55.339 s
throughput=118.560 designs/s
design_batch_size=32
```

该记录证明当前 design batch 能稳定扩展到数千设计，但没有使用相同配置运行 DeepLens，不能据此计算二者加速比。

### 消费级 GPU

在 4090D 等 FP64 吞吐较弱的 GPU 上，DeepLens 默认 FP32 几何分析的单镜头优势会更明显。本项目的大 design batch 可以提高 GPU 利用率，但无法消除 FP64 峰值算力差异。等价 Huygens 对比时双方都受 FP64/complex128 限制，batch 组织和相位样本数会重新成为主要因素。

## 8. 综合判断

| 场景 | 预计占优方 | 原因 |
| --- | --- | --- |
| 单镜头、单波长、FP32 几何 spot | DeepLens | 轻量表面参数、轴对称路径、FP32 |
| 单镜头、多视场几何 PSF map | DeepLens | 批量 field + `index_put_` scatter |
| 数百至数千同拓扑设计的 spot | optics_core | 原生 design batch，避免逐镜头 Python 循环 |
| DeepLens 几何 PSF 与本项目惠更斯 PSF | DeepLens 更快但不可比 | 算法复杂度和物理语义不同 |
| 同采样直接 Huygens，单设计 | optics_core 倾向占优 | 相位公式更轻；公开 PSF 的理想核会抵消部分优势 |
| 默认配置直接 Huygens | optics_core 明显占优 | DeepLens 默认 pupil ray 数极大 |
| 高采样、单设计、显存受限 Huygens | DeepLens 更稳健 | 已有 pupil ray chunking |
| 多设计、多视场、全波长 Huygens MTF | optics_core 明显占优 | design/field/wavelength batch、GPU FFT、无逐项 CPU 同步 |
| 单镜头可微优化 | DeepLens | 自动微分参数链路和优化器已完整实现 |
| 大规模装配公差 Monte Carlo 前向评价 | optics_core | 参数矩阵直接形成多系统 batch |

本项目并非在所有分析上都比 DeepLens 更快。更准确的定位是：

> optics_core 面向共享拓扑的大规模 FP64 多系统仿真；DeepLens 面向单镜头可微设计、几何图像仿真和神经网络训练工作流。

在本项目当前重点场景——大量装配误差设计、多个视场、全部波长、32×32 惠更斯 PSF/MTF——本项目具有明显的架构领先。DeepLens 值得借鉴的重点是参数原生 Tensor 化、轴对称快速路径、pupil ray chunking，以及几何/ASM 快速近似模型。

## 9. 建议的公平实测方案

若需要得到可信的加速比，应拆成以下基准，不要直接运行两边默认接口：

### 9.1 公共设置

1. 使用双方都支持的同一个纯球面 Double Gauss，暂不加入非球面和复杂坐标间断。
2. 统一 FP64、视场、波长、孔径、像面位置和有效光线规则。
3. 注入相同的确定性 pupil 坐标，避免规则采样与随机采样误差不同。
4. 关闭绘图和文件导出。
5. CUDA warm-up 后同步计时，报告中位数、峰值显存和有效光线数。

### 9.2 分层基准

1. 纯追迹：相同 `F×W×P`，分别测试 `D=1、32、1024`。
2. Spot：统一质心或主光线参考，只统计 RMS/GEO。
3. Huygens PSF：统一 `P=1024、I=32`，双方只计算实际 PSF；DeepLens 显式选择 `model="huygens"`。
4. PSF 后处理：向双方输入相同 PSF，单独测试 LSF 和 FFT。
5. 完整 Huygens MTF：统一频率输出和补零策略，再测试 `D=1、32、1024`。

DeepLens 没有原生 design batch，因此 `D>1` 应报告两组结果：

- 当前源码逐设计循环的真实吞吐。
- 若未来实现 stacked lens parameters 后的实验吞吐。

只有完成这些等价设置，才适合报告“倍数领先”；当前源码分析足以判断架构方向，但不适合给出一个统一速度倍率。

## 10. 主要源码依据

optics_core：

- `optics_core/system.py`：MultiOpticalSystem、design view。
- `optics_core/tracing/_core.py`：多维顺序追迹。
- `optics_core/spot_diagram.py`：spot 采样和 RMS 统计。
- `optics_core/huygens_psf.py`：design minibatch 和惠更斯积分。
- `optics_core/huygens_mtf.py`：GPU LSF、FFT 和频率插值。
- `optics_core/first_order.py`：当前参数读取与 Tensor 构造。

DeepLens：

- `deeplens/ray.py`：任意前导 batch 维度的 Ray。
- `deeplens/geolens.py`：单 GeoLens 的采样和逐面追迹。
- `deeplens/geometric_surface/base.py`：局部坐标、Newton 求交和折射。
- `deeplens/geolens_pkg/psf_compute.py`：几何、ASM 和 Huygens PSF。
- `deeplens/imgsim/monte_carlo.py`：批量视场 PSF scatter。
- `deeplens/geolens_pkg/eval.py`：spot 和几何 MTF。
- `deeplens/geolens_pkg/eval_tolerance.py`：逐 trial Monte Carlo。
- `deeplens/geolens_pkg/optim.py`、`optim_2nd.py`：自动微分与二阶优化。
