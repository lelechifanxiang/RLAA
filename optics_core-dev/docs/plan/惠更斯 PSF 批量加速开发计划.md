# 惠更斯 PSF/MTF 自适应批量开发计划

## 目标

本轮只解决以下案例：

```text
对多个光学设计并行计算多个视场、全部波长的 Huygens MTF。
光瞳采样和像面采样均为 32×32。
```

目标调用保持现有接口：

```python
result = system.analysis.mtf(
    MTFSettings(
        field_indices=(0, 1, 2),
        wavelength_index=-1,
        pupil_sample_count=32,
        image_sample_count=32,
    )
).run()
```

用户不设置 batch size。框架根据当前设备和分析配置自动选择 design minibatch 大小。

## 范围

本轮只实现：

1. 自动探测 PSF/MTF 可用的 design batch size。
2. 沿 design 维执行内部 minibatch。
3. MTF 分批消费 PSF，只汇总最终 MTF。

不增加以下公开参数：

```text
PSFSettings.design_batch_size
MTFSettings.design_batch_size
```

32×32 采样下单设计可以放入 V100 16G，因此不实现 pupil/image 积分分块。若 batch size 为 1 仍然 OOM，直接给出明确错误。

## 总体流程

```text
多设计系统完成 prepare
        ↓
用一个设计测量目标 PSF 的峰值显存
        ↓
估算安全 design batch size
        ↓
按 design 连续分批
        ↓
每批一次追迹全部视场和全部波长
        ↓
计算并混合该批多波长 PSF
        ↓
立即计算该批 S/T MTF
        ↓
MTF 回收到 CPU，释放该批 PSF
```

`wavelength_index=-1` 保持现有语义：

1. 使用主波长主光线像点建立共同网格。
2. 所有波长在共同网格计算 PSF。
3. 先混合 PSF，再计算 MTF。

## 1. 轻量 Design Batch View

### 1.1 目的

minibatch 只需要让现有追迹和分析代码看到一段连续设计：

```python
batch_system = system.design_batch_view(start, stop)
```

它不是一个独立复制的新光学系统，而是原系统 `[start, stop)` 区间的只读计算视图。

### 1.2 共享与切片规则

以下静态对象直接共享引用，不复制：

```text
architecture
materials
surface 对象
fields
wavelengths
aperture
config
tracer
```

这些对象在分析过程中只读，与 design 区间无关。

参数批量使用轻量区间视图：

```text
ParameterVectorBatchRange(parent_parameters, start, stop)
```

该对象只保存：

```text
parent_parameters
start
stop
```

并实现现有计算需要的最小接口：

```text
schema
system_count
parameter_count
__len__
__iter__
__getitem__
```

读取第 `i` 个局部设计时，实际访问：

```python
parent_parameters[start + i]
```

禁止通过以下方式构造参数批量：

```python
ParameterVectorBatch(vectors=parent.parameters[start:stop])
```

因为当前 `ParameterVectorBatch.__post_init__()` 会对每个参数向量执行 `list(vector)`，产生不必要的参数复制。

已准备的 tensor 使用普通连续切片：

```python
tensor[start:stop]
```

包括：

```text
frame_data.rotations
frame_data.origins
first_order_data 的全部字段
clear_aperture_data 中按 design 保存的 tensor
```

PyTorch 基础切片与原 tensor 共享 storage，不复制底层数据。这里禁止调用：

```text
clone()
contiguous()
to()
torch.tensor(...)
```

### 1.3 视图构造

`design_batch_view()` 不调用 `MultiOpticalSystem.__init__()`，也不重新执行 `prepare()`。

建议实现方式：

1. 创建一个浅层 `MultiOpticalSystem` 容器视图。
2. 参数替换为 `ParameterVectorBatchRange`。
3. `surfaces` 只创建新的 `SurfaceSequence` 包装器，内部 surface 列表仍共享。
4. prepared data 替换为 `[start:stop]` tensor view。
5. `_analysis_hub` 置空，按 batch view 延迟创建。
6. metadata 记录原始 `[start, stop)`，不修改父系统。

视图只增加少量 Python 包装对象，不复制镜头结构、参数矩阵或 GPU tensor。

### 1.4 生命周期

batch view 引用父系统数据，因此父系统必须在分析完成前保持存活。这符合当前调用方式：

```text
父系统.analysis.mtf(...).run()
```

视图仅供内部同步计算，不长期缓存，也不暴露修改接口。

## 2. Batch Size 自动探测

新增内部函数：

```python
resolve_huygens_design_batch_size(
    system,
    field_indices,
    wavelength_index,
    pupil_sample_count,
    image_sample_count,
) -> int
```

CUDA 下：

1. 取第一个设计的轻量 batch view。
2. 完整运行一次目标 PSF作为 warmup。
3. 释放 warmup 结果并同步 GPU。
4. 记录当前 allocated memory，重置峰值统计。
5. 再运行一次目标 PSF并同步。
6. 用峰值减去执行前 allocated memory，得到单设计增量。
7. 读取当前空闲显存。
8. 使用 75% 空闲显存预算估算：

```text
batch_size = floor(空闲显存 × 0.75 / 单设计显存增量)
```

9. 将结果限制在 `[1, system.system_count]`。
10. 本次分析固定复用该结果，不增加跨调用缓存。

CPU 不探测显存，直接使用全部设计。

如果实际 minibatch 发生 CUDA OOM：

1. batch size 减半。
2. 重新执行当前设计区间。
3. batch size 为 1 仍失败时抛错。

只捕获 CUDA OOM。仅在 OOM 后调用 `torch.cuda.empty_cache()`。

## 3. 共享 PSF Minibatch 框架

新增内部迭代器：

```python
iter_huygens_psf_design_batches(
    system,
    field_indices,
    wavelength_index,
    pupil_sample_count,
    image_sample_count,
    image_delta_um,
)
```

返回：

```text
start, stop, HuygensPSFBatch
```

职责：

1. 自动探测 batch size。
2. 创建连续 design batch view。
3. 调用现有 `compute_huygens_psf_batch()`。
4. OOM 时减半并重试。
5. 按原始 design 顺序返回结果。

一个 minibatch 内必须一次追迹全部目标视场和波长，不能再按 field、wavelength 或 design 拆分追迹。

## 4. PSF 与 MTF 接入

### PSF

`run_huygens_psf()` 消费共享迭代器。每批完成后，将最终结果一次性回收到 CPU，并按 design 维拼接：

```text
psf
strehl_ratio
psf_by_wavelength
strehl_by_wavelength
pixel_pitch_um
```

图片导出仍只支持单设计。

### MTF

`run_huygens_mtf()` 使用同一个 PSF minibatch 迭代器：

```text
当前批 PSF
    ↓
当前批 S/T MTF
    ↓
MTF 一次性回收到 CPU
    ↓
释放当前批 PSF
```

最终只拼接：

```text
sagittal:       (design, field, frequency)
tangential:     (design, field, frequency)
pixel_pitch_um: (design,)
```

禁止保存全部设计的 PSF 后再统一计算 MTF。

## 5. 最小测试集

只增加四组关键 contract 测试：

1. **轻量视图测试**
   - 参数索引和 design 数量正确。
   - prepared tensor 与父系统共享 storage。
   - 不调用 `prepare()`，不复制 parameter vector。

2. **分批结果一致性**
   - 使用多设计、多视场、`wavelength_index=-1`。
   - 强制拆成两个 minibatch。
   - PSF 和 MTF 与一次性计算结果一致。
   - 同时覆盖最后一批不足 batch size 的情况。

3. **一次追迹测试**
   - counting tracer 验证每个 minibatch 只追迹一次全部视场和波长。

4. **OOM 降级测试**
   - 模拟首个 batch CUDA OOM。
   - 验证 batch size 减半并成功重试。
   - batch size 为 1 仍失败时错误信息明确。

因为只改变 design 分组，不改变单设计积分顺序，结果应满足：

```text
atol=1e-12
rtol=1e-12
```

现有 PSF、MTF contract 和 Zemax 回归测试必须无新增失败。

## 6. V100 示例

新增：

```text
examples/batch_mtf.py
```

复用 `examples/batch_spot.py` 的多设计双高斯构造，固定：

```text
field_indices=(0, 1, 2)
wavelength_index=-1
pupil_sample_count=32
image_sample_count=32
save_path=None
```

命令行只保留：

```text
--device
--surfaces
--summary-json
```

打印：

```text
设计总数
探测及最终 batch size
minibatch 数量
总耗时
designs/s
峰值 allocated/reserved 显存
MTF 输出 shape
```

不提供用户 batch size 参数。

## 实施顺序

1. 实现轻量 `ParameterVectorBatchRange` 和 `design_batch_view()`。
2. 实现单设计显存探测。
3. 实现共享 PSF minibatch 迭代器。
4. 接入 PSF 结果拼接和 MTF 流式计算。
5. 增加 OOM 减半重试。
6. 增加四组关键 contract 测试。
7. 增加 `examples/batch_mtf.py`。
8. 运行现有 PSF、MTF 和全量测试。

## 完成条件

1. 用户无需设置 batch size。
2. 多设计、多视场、全部波长 MTF 能自动分批运行。
3. design batch view 不复制参数矩阵和 prepared tensor。
4. 每个 minibatch 只追迹一次全部目标视场和波长。
5. MTF 不保存全部设计的 PSF。
6. OOM 时自动减半；单设计仍失败时明确报错。
7. 分批前后 PSF/MTF 数值一致。
8. V100 示例输出吞吐率和峰值显存。
9. 全量测试无新增失败。
