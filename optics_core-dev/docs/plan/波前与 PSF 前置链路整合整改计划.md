# 波前与 PSF 前置链路整合整改计划

## 目标

整理当前 Wavefront Map 与 Huygens PSF 中重复的 pupil 追迹、像面光线数据整理和 OPD helper，使共享逻辑只有一处实现。

整改后：

1. Wavefront Map 继续放在 `optics_core/wavefront.py`。
2. PSF 不再自己维护重复的 pupil trace / 有效 pupil mask / 方向余弦归一化 / 出瞳参考 OPD helper。
3. PSF 调用 `wavefront.py` 中的底层数据整理函数。
4. PSF 的惠更斯积分、像面网格、多波长 PSF 混合逻辑仍保留在 `huygens_psf.py`。
5. 不新增复杂抽象层，不新增 `huygens_pupil.py` 之类的中间模块。

---

## 当前问题

当前实现中，Wavefront Map 代码基本重新写了一套 PSF 已有的前置流程，主要重复点包括：

1. 按 pupil sample 构造输入光线并追迹到像面。
2. 从 `TraceResult` 中提取：
   - 像面交点 `image_points`
   - 方向余弦 `ray_directions`
   - OPL
   - 波长
   - 主光线像点 `chief_points`
3. 根据 pupil 坐标和 trace valid 标记生成有效 pupil 点。
4. 方向余弦归一化。
5. 出瞳参考球 OPD 计算。

其中 `_exit_pupil_reference_data()` 目前在 `huygens_psf.py` 和 `wavefront.py` 中各有一份，应该合并。

---

## PSF 与 Wavefront 的共同点

二者都需要下面的底层数据：

```text
输入:
system
fields
wavelengths
pupil sample

输出:
trace_result
image_points
ray_directions
opl
wavelength_mm
chief_points
valid_points
pupil_coordinates
```

这些数据都来自同一次 pupil 光线追迹，和最终分析类型无关。

因此这部分应放在 `optics_core/wavefront.py` 中，作为低层公共函数。

---

## PSF 与 Wavefront 的不同点

不能把 Wavefront Map 的最终 OPD grid 直接喂给 PSF。

PSF 当前惠更斯积分使用的是像面局部平面波相位：

```text
OPL_j + dot(direction_j, P - P_image_j) - piston
```

Wavefront Map 输出的是 pupil 网格上的 OPD / wavefront error：

```text
exit-pupil-reference OPD at pupil sample
```

二者共享像面光线数据，但最终计算不同：

1. PSF 使用像面网格和惠更斯复振幅叠加。
2. Wavefront Map 使用 pupil 网格 OPD。

所以整改重点是共享前置数据链路，而不是让 PSF 依赖 Wavefront Map 的最终 `opd` 输出。

---

## 推荐代码结构

只保留两个核心模块：

```text
optics_core/wavefront.py
optics_core/huygens_psf.py
```

### `wavefront.py` 负责

1. Wavefront Map 公开分析入口：

```python
run_wavefront(system, settings)
compute_wavefront_batch(...)
iter_wavefront_design_batches(...)
plot_wavefront(...)
```

2. Wavefront Map 采样器：

```python
sample_zemax_wavefront_pupil(sample_count)
```

3. PSF 与 Wavefront 共享的底层 helper：

```python
trace_pupil_to_image(system, fields, wavelengths, sample) -> TraceResult
extract_image_wave_data(system, trace_result, sample) -> ImageWaveData
exit_pupil_referenced_opd(...) -> tuple[torch.Tensor, torch.Tensor]
normalized(vector) -> torch.Tensor
```

其中 `ImageWaveData` 建议为轻量 dataclass：

```python
@dataclass(slots=True)
class ImageWaveData:
    image_points: torch.Tensor
    ray_directions: torch.Tensor
    opl: torch.Tensor
    wavelength_mm: torch.Tensor
    chief_points: torch.Tensor
    valid_points: torch.Tensor
    pupil_coordinates: torch.Tensor
    pupil_weights: torch.Tensor
    ordinary_ray_count: int
    chief_ray_index: int
```

保持字段直接、明确，不再额外封装 analysis 语义。

### `huygens_psf.py` 负责

1. PSF settings/result 组织。
2. PSF design minibatch。
3. PSF 像面采样间隔和像面网格。
4. 惠更斯积分 `_huygens_integral()`。
5. 单波长 / 全波长 PSF 混合。
6. PSF 图片导出。

---

## 整改步骤

### 1. 在 `wavefront.py` 中整理公共数据结构

新增：

```python
@dataclass(slots=True)
class ImageWaveData:
    ...
```

新增或重命名：

```python
trace_pupil_to_image(...)
extract_image_wave_data(...)
exit_pupil_referenced_opd(...)
normalized(...)
```

要求：

1. `trace_pupil_to_image()` 替代当前 `trace_wavefront_rays()`。
2. `extract_image_wave_data()` 统一完成 `TraceResult` 张量提取、方向归一化、有效 pupil mask、主光线像点提取。
3. `exit_pupil_referenced_opd()` 替代两个文件里的 `_exit_pupil_reference_data()`。

### 2. 修改 Wavefront Map 使用公共 helper

`compute_wavefront_batch()` 流程改为：

```text
选择 field / wavelength
生成 zemax wavefront pupil sample
trace_pupil_to_image()
extract_image_wave_data()
exit_pupil_referenced_opd()
reshape 为 N×N OPD grid
计算 RMS
```

Wavefront Map 对外行为不变：

1. 不返回 `pupil_mask`。
2. 无效 pupil 区域 `opd=0`。
3. `sample_count=N` 输出 `N×N`。
4. 不支持 `wavelength_indices=(-1,)`。

### 3. 修改 PSF 使用公共 helper

`compute_huygens_psf_batch()` 中保留 PSF 自己的采样器：

```python
sample = SquarePupilSampler(nx=pupil_sample_count, ny=pupil_sample_count).sample()
```

但追迹和数据整理改为：

```python
trace_result = trace_pupil_to_image(system, fields, wavelengths, sample)
wave_data = extract_image_wave_data(system, trace_result, sample)
```

`compute_huygens_psf()` 内部不再重复：

1. 手动读取 `x/y/z/l/m/n/opl/wavelength`。
2. 手动计算 `valid_points`。
3. 手动计算 `image_points`。
4. 手动调用本地 `_normalized()`。

而是使用 `ImageWaveData`：

```python
image_points = wave_data.image_points
ray_directions = wave_data.ray_directions
opl = wave_data.opl
wavelength_mm = wave_data.wavelength_mm
chief_points = wave_data.chief_points
valid_points = wave_data.valid_points
pupil_weights = wave_data.pupil_weights
```

PSF 的 `_huygens_integral()` 不改算法。

### 4. 删除重复代码

从 `huygens_psf.py` 删除：

1. `trace_huygens_psf_rays()`
2. `_normalized()`
3. `_exit_pupil_reference_data()`

如果外部测试仍需要出瞳参考 OPD，改为从 `optics_core.wavefront` 导入：

```python
from optics_core.wavefront import exit_pupil_referenced_opd, normalized
```

同时修改相关测试导入。

### 5. 保留必要差异

不要强行合并以下内容：

1. PSF 的 `wavelength_index=-1` 全波长混合。
2. Wavefront Map 的 `wavelength_indices` 多个单波长。
3. PSF 的像面网格 `_image_grid()`。
4. PSF 的 `_huygens_integral()`。
5. Wavefront Map 的 Zemax DataGrid 采样器。

这些逻辑语义不同，合并会降低可读性。

---

## 测试调整

### Contract 测试

继续运行：

```powershell
python -m pytest tests\contract\test_wavefront_contract.py
python -m pytest tests\contract\test_huygens_psf_contract.py
python -m pytest tests\contract\test_huygens_mtf_contract.py
python -m pytest tests\contract\test_huygens_batching_contract.py
```

新增或调整一个轻量测试：

```text
PSF 和 Wavefront 使用同一个 trace/extract helper 时，输出 shape 和已有 PSF 数值不变。
```

不需要新增大量数值测试。

### Regression 测试

继续运行：

```powershell
python -m pytest tests\regression\test_wavefront_map_against_zemax.py
```

该测试当前允许数值未完全对齐时 xfail，但 shape、Zemax grid size、center point 必须正常打印。

如本机有 Zemax，也运行：

```powershell
python -m pytest tests\regression\test_huygens_psf_against_zemax.py
python -m pytest tests\regression\test_huygens_mtf_against_zemax.py
```

---

## 完成条件

1. `wavefront.py` 中只有一份 pupil trace / image wave data / exit pupil OPD helper。
2. `huygens_psf.py` 调用 `wavefront.py` 的底层 helper。
3. `huygens_psf.py` 不再定义 `_exit_pupil_reference_data()` 和 `_normalized()`。
4. PSF、MTF、Wavefront contract 测试通过。
5. Wavefront Zemax 回归保持 xfail 或通过，但不能失败。
6. PSF / MTF 现有数值行为不变化。
7. 代码量减少，模块边界清晰，没有新增多余适配层。

---

## 备注

这次整改不是让 PSF 调用 `run_wavefront()`，也不是让 PSF 使用 Wavefront Map 的最终 OPD grid。

正确方向是：

```text
PSF 和 Wavefront Map 共享像面光线波前数据。
PSF 继续做惠更斯积分。
Wavefront Map 继续做 pupil OPD map。
```
