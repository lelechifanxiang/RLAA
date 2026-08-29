# 批量波前图 Goal 模式实施计划

## 目标

实现批量 Wavefront Map 分析功能，对标 Zemax Wavefront Map。

首版只支持：

1. 多设计。
2. 多视场并行。
3. 多个单波长并行。
4. 采样率，默认 `32`。
5. 单设计图片保存路径。

不支持：

1. `wavelength_index=-1` 全波长混合。
2. 用户选择评价表面，内部固定 `Surface=Image`。
3. `remove_tilt`、`use_exit_pupil`、`reference_to_primary` 等 Zemax 高级选项。
4. Zernike、干涉图、任意子孔径。

---

## 公开接口

新增或整理：

```python
@dataclass(slots=True)
class WavefrontSettings:
    field_indices: Sequence[int] | None = None
    wavelength_indices: Sequence[int] | None = None
    sample_count: int = 32
    save_path: str | None = None
```

语义：

1. `field_indices=None` 表示全部视场。
2. `wavelength_indices=None` 表示主波长。
3. `wavelength_indices` 只接受非负波长索引。
4. 传入 `-1` 直接报错。
5. `sample_count=N` 输出 `N×N` 波前图。
6. `save_path` 只支持单设计、单视场、单波长导图。

结果对象至少包含：

```text
opd:           (design, field, wavelength, sample_count, sample_count), 单位 waves
rms_wavefront: (design, field, wavelength), 单位 waves
pupil_x/y:     (sample_count, sample_count)
field_indices
wavelength_indices
sample_count
save_path
```

无效 pupil 区域不单独返回 mask，`opd` 对应位置直接填 0。

---

## Zemax 对标约定

Zemax helper 固定使用 Wavefront Map，并固定设置：

```text
Surface = Image
Show As = Surface
Rotation = 0
Sampling = f"{sample_count}x{sample_count}"
Polarization = None
Reference To Primary = False
Use Exit Pupil Shape = False
Remove Tilt = False
Scale = 1.0
Subaperture X/Y = 0.0
Subaperture R = 1.0
```

注意：

1. `Surface=Image` 是评价表面固定为像面。
2. exit pupil 只属于 OPD 参考球和显示形状语义，不作为公开表面参数。
3. Zemax 原生数据设置 `N×N` 时按 `N×N` 对标。
4. helper 需要打印 Zemax 原始 grid shape、pupil grid size、center point 行列信息。
5. 若 ZOSPy wrapper 和 Zemax 原生 datagrid shape 不一致，以 Zemax 原生 datagrid 为准。

---

## 实现步骤

### 1. 接通分析入口

修改 `optics_core/analysis.py`：

1. 新增或整理 `WavefrontSettings`。
2. 新增或整理 `WavefrontResult`。
3. `system.analysis.wavefront(settings).run()` 调用 `run_wavefront()`。

### 2. 新增 Wavefront 模块

新增：

```text
optics_core/wavefront.py
```

从现有 PSF 实现中抽取的波前图前置逻辑也放入该文件。

提供：

```python
run_wavefront(system, settings)
compute_wavefront_batch(...)
iter_wavefront_design_batches(...)
sample_zemax_wavefront_pupil(sample_count)
plot_wavefront(...)
```

核心职责：

1. field / wavelength 选择。
2. 构造 pupil rays。
3. 追迹到像面。
4. 整理 OPL、方向余弦、主光线和内部有效 pupil mask。
5. 计算 OPD grid 和 RMS。
6. 导出单张波前图。

### 3. 实现 Wavefront Map 采样器

在 `optics_core/wavefront.py` 中新增内部函数：

```python
sample_zemax_wavefront_pupil(sample_count)
```

要求：

1. 生成 `sample_count × sample_count` pupil 采样点。
2. 采样顺序、中心点位置和内部 mask 通过 Zemax 回归测试确认。
3. 追加 reference chief ray。

### 4. 实现计算逻辑

1. 固定像面评价。
2. 输出 OPD 单位为 waves。
3. 无效 pupil 区域 `opd=0`。
4. RMS 只统计内部有效 pupil 点。
5. 多设计使用内部 minibatch；CUDA OOM 时自动减半重试。
6. 不增加用户手动 batch size 参数。

### 5. 增加 Zemax helper

新增：

```text
tests/zemax/wavefront_map.py
```

提供：

```python
fetch_zemax_wavefront_map_from_spec(...)
```

返回 Zemax OPD grid、pupil 坐标、RMS、field/wavelength 信息和原始 grid 元数据。

### 6. 增加批量示例

新增：

```text
examples/batch_wavefront.py
```

复用 `scripts/batch_tolerance_common.py` 的 4D 公差扫描。

支持参数：

```text
--device
--tolerance-sample-count
--sample-count
--output-dir
--summary-json
--skip-images
--field-indices
--wavelength-indices
```

打印设计数、视场数、波长数、采样率、minibatch 数、总耗时、每秒 wavefront map 数、保存图片数。

---

## 测试要求

### Contract 测试

新增：

```text
tests/contract/test_wavefront_contract.py
```

只测试关键契约：

1. 返回 shape 正确。
2. `wavelength_indices=None` 使用主波长。
3. 多视场、多波长索引并行可运行。
4. `wavelength_indices=(-1,)` 抛错。
5. `sample_count=32` 输出 `32×32`。
6. RMS 只统计有效 pupil 点。
7. 单设计、单视场、单波长图片导出成功。

### Zemax 回归测试

新增：

```text
tests/regression/test_wavefront_map_against_zemax.py
```

首批对标：

```text
sample_count = 32
field_index = 0, 1, 2
wavelength_index = primary
```

测试内容：

1. shape 与 Zemax 原生 Wavefront Map 一致。
2. 打印 Zemax pupil grid size 和 center point。
3. 打印 RMS、最大绝对误差、平均绝对误差。
4. field 0 主波长尽量严格通过。
5. 非零视场若暂未对齐，使用 xfail。

---

## 最小验收命令

```powershell
python -m pytest tests\contract\test_wavefront_contract.py
python -m pytest tests\contract\test_huygens_psf_contract.py
python -m pytest tests\contract\test_huygens_mtf_contract.py
python examples\batch_wavefront.py --device cpu --tolerance-sample-count 1 --sample-count 32 --skip-images
```

有 Zemax 时额外运行：

```powershell
python -m pytest tests\regression\test_wavefront_map_against_zemax.py
```

有 GPU 时额外运行：

```powershell
python examples\batch_wavefront.py --device cuda:0 --tolerance-sample-count 3 --sample-count 32 --skip-images
```

---

## 完成条件

1. `system.analysis.wavefront(...).run()` 可用。
2. 支持多设计、多视场、多个具体波长索引。
3. 不支持 `wavelength_index=-1`。
4. 公开 settings 只包含视场索引、波长索引、采样率、保存路径。
5. `sample_count=N` 输出 `N×N`。
6. Zemax helper 明确调用 Wavefront Map，且固定 `Surface=Image`。
7. Zemax 回归测试建立 shape、RMS、OPD grid 对标链路。
8. PSF / MTF 现有测试不退化。
9. 批量示例能输出耗时、吞吐率和可选图片。
