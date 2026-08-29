# 惠更斯 MTF 第一阶段冗余删除执行方案

## Goal 模式目标

在不改变公开 PSF 行为和 MTF 数值定义的前提下，完成 `docs/plan/惠更斯 MTF GPU 加速开发计划.md` 中 4.1、4.2、4.3 的低风险优化：

1. MTF 路径不再计算理想 PSF 和 Strehl。
2. 惠更斯积分删除重复 mask 和不必要的大型无效张量。
3. MTF minibatch 结果尽量保留在 GPU 端，最后一次性回收 CPU。

本方案不实施 4.4 的吞吐率 batch size 选择，也不实施第二阶段可分离相位 GEMM。

## 当前问题定位

当前 `run_huygens_mtf()` 通过 `iter_huygens_psf_design_batches()` 复用 PSF 计算路径。该路径会同时计算：

```text
psf_by_wavelength
ideal_psf_by_wavelength
strehl_by_wavelength
strehl_ratio
```

但 MTF 最终只需要：

```text
mixed_psf
pixel_pitch_um
wavelength_indices
```

此外，`_huygens_integral()` 中有效光瞳点已经通过零积分权重表达，后续再对 `kernel` 和 `ideal_kernel` 执行 `torch.where()` 会额外物化大型 complex128 张量。`run_huygens_mtf()` 也在每个 minibatch 后立即 `.cpu()`，形成同步点。

## 修改范围

主要修改文件：

1. `optics_core/huygens_psf.py`
2. `optics_core/huygens_mtf.py`
3. `tests/contract/test_huygens_batching_contract.py`
4. `tests/contract/test_huygens_mtf_contract.py`

必要时少量更新 benchmark 或 example 输出字段，但不新增公开用户参数。

## 1. MTF 跳过理想 PSF 和 Strehl

### 实现方式

给内部 PSF 批处理链路增加参数：

```python
compute_ideal_psf: bool = True
```

传递路径：

```text
iter_huygens_psf_design_batches()
HuygensPSFDesignBatchIterator
resolve_huygens_design_batch_size()
compute_huygens_psf_batch()
compute_huygens_psf()
_huygens_integral()
```

`run_huygens_psf()` 保持默认 `compute_ideal_psf=True`，公开 PSF 行为不变。

`run_huygens_mtf()` 调用 minibatch iterator 时设置：

```python
compute_ideal_psf=False
```

### 返回结构

保持现有 `HuygensPSFBatch`，但允许以下字段在 MTF 路径为 `None`：

```python
strehl_ratio: torch.Tensor | None
psf_by_wavelength: torch.Tensor | None
strehl_by_wavelength: torch.Tensor | None
```

MTF 路径只读取：

```python
psf_batch.psf
psf_batch.pixel_pitch_um
psf_batch.wavelength_indices
```

PSF 路径在返回 `PSFResult` 前显式检查这些字段非空。这样不复制算法，也不增加新的公开结果类型。

### 计算逻辑

`compute_huygens_psf_batch(..., compute_ideal_psf=False)` 时：

1. `compute_huygens_psf()` 只返回实际 `psf_by_wavelength`。
2. 不调用 `_strehl_from_psfs()`。
3. `_reduce_huygens_psf()` 只混合 PSF，不混合 Strehl。
4. `HuygensPSFBatch.strehl_ratio` 等字段置为 `None`。

建议新增一个小函数：

```python
_reduce_huygens_psf_only(system, psf_by_wavelength)
```

或者让 `_reduce_huygens_psf()` 接受 `strehl_by_wavelength: torch.Tensor | None`。优先选择更直观、分支更少的实现。

## 2. 删除重复 mask 和大型无效张量

### 保留一次权重 mask

`_huygens_integral()` 中保留：

```python
integration_weight = torch.where(valid_points, pupil_weights.reshape(1, 1, 1, -1), 0.0)
integration_weight = integration_weight / integration_weight.sum(...).clamp_min(...)
```

后续不再执行：

```python
kernel = torch.where(valid_points[..., None, None], kernel, zeros)
ideal_kernel = torch.where(valid_points[..., None, None], ideal_kernel, zeros)
```

因为无效光线的积分权重已经为 0。

### 避免理想相位无效构造

当 `compute_ideal_psf=False` 时，不构造：

```text
ideal_phase
ideal_kernel
ideal_amplitude
ideal_psf
```

当 `compute_ideal_psf=True` 时，理想 PSF 仍按当前数学定义计算。

### 保持数值含义

需要特别确认：

1. `integration_weight` 的归一化维度不变。
2. 全无效光瞳时仍返回稳定结果，不引入 inf。
3. 删除 `kernel where` 后，PSF/MTF 与修改前在正常有效光瞳条件下保持一致。

## 3. MTF 结果延迟回收 CPU

### 第一版实现

`run_huygens_mtf()` 中每批计算得到：

```python
batch_sagittal
batch_tangential
psf_batch.pixel_pitch_um
```

不要立即 `.cpu()`，改为先保存在 GPU 端列表：

```python
sagittal_batches.append(batch_sagittal.detach())
tangential_batches.append(batch_tangential.detach())
pixel_pitch_batches.append(psf_batch.pixel_pitch_um.detach())
```

循环结束后先在 GPU 端拼接：

```python
sagittal_gpu = torch.cat(sagittal_batches, dim=0)
tangential_gpu = torch.cat(tangential_batches, dim=0)
pixel_pitch_gpu = torch.cat(pixel_pitch_batches, dim=0)
```

最后一次性回收：

```python
sagittal = sagittal_gpu.cpu()
tangential = tangential_gpu.cpu()
pixel_pitch_um = pixel_pitch_gpu.cpu()
frequencies_cpu = frequencies_lp_per_mm.cpu()
```

### 显存边界

固定案例的最终 MTF 输出约 90 MiB，V100 可以承受。第一版不实现 pinned CPU 双缓冲。

如果后续出现显存压力，再单独实施 pinned CPU 双缓冲；本阶段不提前引入复杂数据搬运框架。

## 验证计划

### 功能测试

运行：

```powershell
python -m pytest tests\contract\test_huygens_mtf_contract.py
python -m pytest tests\contract\test_huygens_batching_contract.py
```

重点验证：

1. 多视场 MTF shape 不变。
2. minibatch 与 full batch 结果一致。
3. OOM 减半逻辑仍有效。
4. 单设计 MTF 图片导出不变。

### 新增或调整测试

新增一个轻量 contract 测试，证明 MTF 路径没有计算 Strehl：

1. monkeypatch `_strehl_from_psfs()`，若被 MTF 调用则抛错。
2. 执行 `system.analysis.mtf(...).run()`。
3. 断言 MTF 正常返回。

如实现中 `_huygens_integral()` 支持 `compute_ideal_psf=False`，可再用 monkeypatch 或返回字段检查确认 `psf_batch.strehl_ratio is None`。

### 数值回归

运行：

```powershell
python -m pytest tests\regression\test_huygens_mtf_against_zemax.py
```

如果 Zemax 环境不可用，至少运行现有 contract 测试，并记录 Zemax 回归未执行。

### 性能验证

在 V100 上运行：

```powershell
python examples\batch_mtf.py --device cuda:0 --surfaces 1 2 10 11 --summary-json examples\output\batch_mtf_v100_after_phase1.json
```

记录并对比：

```text
elapsed_seconds
designs_per_second
detected_design_batch_size
design_batch_size
minibatch_count
peak_allocated_gib
peak_reserved_gib
```

预期结果：

1. `peak_allocated_gib` 下降。
2. `detected_design_batch_size` 可能上升。
3. `designs_per_second` 上升。
4. MTF 数值输出不发生可见变化。

## 完成条件

1. `run_huygens_psf()` 公开行为不变，仍返回 PSF、逐波长 PSF、Strehl。
2. `run_huygens_mtf()` 不再计算 ideal PSF 和 Strehl。
3. `_huygens_integral()` 中删除重复 complex kernel mask。
4. MTF minibatch 不再每批 `.cpu()` 同步回收。
5. 相关 contract 测试通过。
6. V100 批量 MTF benchmark 有新的性能摘要文件。

## 不做事项

1. 不实现 4.4 batch size 按吞吐率选择。
2. 不实现可分离相位 GEMM。
3. 不实现 pinned CPU 双缓冲。
4. 不改变 PSF/MTF 对 Zemax 的数值定义。
5. 不增加用户可见的 `MTFSettings` 或 `PSFSettings` 参数。
