# 惠更斯 MTF 开发计划

## 1. 结论

查阅 OpticStudio 官方文档后，可以确认：

1. Zemax Huygens MTF 是对 Huygens PSF 做 FFT 得到的。
2. 若只需要 Sagittal 和 Tangential 两条轴向曲线，可以先把二维 PSF 沿正交方向积分为一维 LSF，再分别进行一维 FFT。根据傅里叶投影切片定理，这与提取二维 PSF FFT 的 `fx`、`fy` 中心切片等价。
3. 补零不是 MTF 定义的一部分，也不会增加物理分辨率；它只会细化 FFT 的频率采样，便于在指定频率上插值。OpticStudio 官方文档没有公开其内部补零倍数，因此不能假定某个固定倍数就是 Zemax 算法。
4. Zemax 的方向约定必须严格遵守：
   - Tangential：局部像面 `y` 方向的空间频率。
   - Sagittal：局部像面 `x` 方向的空间频率。
5. 多波长 MTF 必须先对各波长 PSF 做非相干合成，再对混合 PSF 求 MTF；不能对单色 MTF 的模值直接加权平均。

因此，本项目适合采用“复用 Huygens PSF → 横纵投影 → 对称补零 → 一维 FFT → DC 归一化 → 频率插值”的实现。

## 2. Zemax 官方计算语义

### 2.1 Huygens MTF 来源

OpticStudio Huygens MTF 文档明确说明：

- Huygens MTF 对 Huygens PSF 执行 FFT。
- `Image Sampling` 和 `Image Delta` 与 Huygens PSF 完全相同。
- PSF 图像范围由 `Image Sampling × Image Delta` 决定。
- `Max Frequency` 仅控制显示到哪个空间频率。
- 当前只需要支持 `Modulation` 类型。

本项目不应重新实现另一套 Huygens 衍射积分，而应直接复用现有 PSF 计算结果。

### 2.2 S/T 方向

MTF 内部使用的批量 PSF 张量约定为：

```text
(design, field, image_y, image_x)
```

对应关系为：

```text
Sagittal MTF
    = x 方向空间频率响应
    = 先沿 y 积分，得到 x 方向 LSF
    = FFT(sum(PSF, dim=image_y))

Tangential MTF
    = y 方向空间频率响应
    = 先沿 x 积分，得到 y 方向 LSF
    = FFT(sum(PSF, dim=image_x))
```

需要专门增加非对称 PSF 测试，防止 S/T 方向写反。

Huygens MTF 的 S/T 是局部像面坐标，而不是根据视场坐标重新计算的径向和切向方向。像面旋转会改变 Huygens MTF 的方向定义。

### 2.3 OTF 和归一化

离散 PSF 为 `I(x, y)`，则：

```text
OTF(fx, fy) = FFT2(I)
MTF(fx, fy) = abs(OTF(fx, fy) / OTF(0, 0))
```

对一维投影：

```text
LSF_s(x) = sum_y I(x, y)
LSF_t(y) = sum_x I(x, y)

MTF_s(fx) = abs(FFT(LSF_s) / FFT(LSF_s)[0])
MTF_t(fy) = abs(FFT(LSF_t) / FFT(LSF_t)[0])
```

由此得到：

- `MTF(0)` 应严格为 `1`。
- PSF 是否做峰值归一化不影响 MTF。
- PSF 的整体强度倍数会在 DC 归一化时约掉。
- PSF 平移只改变 OTF 相位，不改变 MTF 模值；但有限图像窗口发生截断时仍会影响结果。
- 不应在 PSF 或 LSF 上额外应用 Hann 等窗函数，否则会改变 Zemax 所定义的 MTF。

### 2.4 频率轴

若像面间隔为 `delta_um`，补零后的 FFT 长度为 `n_fft`：

```text
delta_mm = delta_um × 1e-3
frequency_step = 1 / (n_fft × delta_mm)
nyquist_frequency = 1 / (2 × delta_mm)
```

使用 `lp/mm` 表示：

```text
frequency_step_lp_per_mm = 1000 / (n_fft × delta_um)
nyquist_lp_per_mm = 500 / delta_um
```

所有请求频率必须小于等于各设计对应的 Nyquist 频率。

原始 `N` 点 FFT 的频率间隔为：

```text
1000 / (N × delta_um)
```

补零到更长的 `n_fft` 只降低频率间隔，不会补回被像面采样或图像截断丢失的信息。

### 2.5 图像窗口和采样

Huygens MTF 继承 Huygens PSF 的采样要求：

1. `Image Delta=0` 时继续使用现有 Zemax 自动间隔公式。
2. 全波长分析使用最长波长确定自动 Image Delta。
3. 图像范围必须覆盖主要 PSF 能量，否则矩形截断会引入 MTF 波动。
4. 增大 `Image Sampling` 会扩大图像范围，并在不补零时提高原生频率分辨率。
5. 减小 `Image Delta` 会提高 Nyquist 频率，但会缩小固定点数下的图像范围。

因此测试中不能只提高频率上限，还要同时检查 PSF 是否在图像边缘被明显截断。

### 2.6 多波长

Zemax 先构造多波长混合 PSF，再计算 MTF：

```text
mixed_psf = sum_w(spectral_weight[w] × psf[w])
mixed_otf = FFT(mixed_psf)
mixed_mtf = abs(mixed_otf / mixed_otf[0])
```

以下算法是错误的：

```text
sum_w(spectral_weight[w] × abs(FFT(psf[w]) / FFT(psf[w])[0]))
```

原因是复 OTF 在取模前仍包含符号和相位信息，模值不满足线性叠加。

MTF 应直接消费现有 PSF 框架最终输出的单色 PSF 或混合 PSF，避免复制波长权重逻辑。

### 2.7 像面坐标和 Huygens 相位参考

OpticStudio 在主光线像点处建立与像面相切的计算平面，该平面法向取像面法向，而不是主光线方向。Huygens MTF 使用该局部像面坐标。

当前 `_image_grid()` 使用全局 `x/y` 和固定 `z` 网格，对普通未倾斜像面可与 Zemax 对标；对倾斜像面并不完整。

首版回归可继续使用当前双高斯系统，但在宣称支持一般倾斜像面前，需要：

1. 从像面 frame 读取局部 `x/y` 基向量和法向。
2. 在主光线像点处沿局部 `x/y` 构造图像网格。
3. 确认 Sagittal 对应局部 `x`，Tangential 对应局部 `y`。

此外，OpticStudio 会根据像面范围自动在平面波和球面波 Huygens 相位参考之间切换。MTF 不应单独实现该逻辑；将来应在 PSF 引擎中统一补齐，MTF 自动继承。

## 3. 功能范围

首版支持：

1. 焦点系统，频率单位固定为 `lp/mm`。
2. Modulation MTF。
3. 多设计并行。
4. 一次分析多个视场。
5. 指定单波长、主波长或全部波长混合。
6. Sagittal 和 Tangential 两条曲线。
7. 不考虑偏振。
8. 不支持多配置和 afocal MTF。

## 4. 接口设计

建议扩展现有 `MTFSettings`：

```python
@dataclass(slots=True)
class MTFSettings:
    pupil_sample_count: int = 32
    image_sample_count: int = 32
    image_delta_um: Scalar = 0.0
    frequencies_lp_per_mm: Sequence[Scalar] = field(default_factory=tuple)
    field_indices: Sequence[int] | None = None
    wavelength_index: int | None = None
    save_path: str | None = None
```

语义：

- `field_indices=None`：计算系统全部视场。
- `wavelength_index=None`：主波长。
- `wavelength_index=-1`：全部波长混合。
- `frequencies_lp_per_mm`：最终返回的目标频率；保持所有设计使用相同频率轴。

补零长度属于内部数值策略，不建议首版暴露为公开设置。初始可使用不小于原 LSF 长度 8 倍的二次幂长度，再由 Zemax 回归决定是否调整。

建议扩展 `MTFResult`：

```python
@dataclass(slots=True)
class MTFResult:
    frequencies_lp_per_mm: ArrayLike | None = None
    sagittal: ArrayLike | None = None
    tangential: ArrayLike | None = None
    field_indices: tuple[int, ...] = ()
    wavelength_indices: tuple[int, ...] = ()
    pixel_pitch_um: ArrayLike | None = None
    figure: Any | None = None
    axes: Any | None = None
    save_path: str | None = None
```

形状：

```text
frequencies_lp_per_mm: (frequency,)
sagittal:              (design, field, frequency)
tangential:            (design, field, frequency)
pixel_pitch_um:         (design,)
```

## 5. PSF 框架重构

当前 `run_huygens_psf()` 公开接口只处理一个视场，并在 `compute_huygens_psf()` 中去掉 field 维。

为了避免 MTF 循环构造分析系统或重复追迹，建议提取内部批量函数：

```python
compute_huygens_psf_batch(
    system,
    *,
    field_indices,
    wavelength_index,
    pupil_sample_count,
    image_sample_count,
    image_delta_um,
) -> HuygensPSFBatch
```

内部结果建议包含：

```text
psf:                    (design, field, image_y, image_x)
psf_by_wavelength:      (design, field, wavelength, image_y, image_x)
pixel_pitch_um:         (design,)
field_indices
wavelength_indices
```

执行方式：

1. 所有选定视场和波长组成一次 batch ray trace。
2. 每个视场单独使用对应主光线像点建立图像网格。
3. 全波长时，每个视场分别使用主波长主光线建立自己的共同网格。
4. 在统一 batch 维度上执行 Huygens 积分。
5. 单色直接返回该波长 PSF。
6. 全波长先混合 PSF，再返回最终 PSF。

现有 `run_huygens_psf()` 调用该函数并选择一个 field 输出；新的 `run_huygens_mtf()` 直接消费全部 field 的最终 PSF。

不要在 MTF 模块复制以下逻辑：

- 光瞳采样和追迹
- Image Delta 自动计算
- 主光线像点选择
- Huygens 积分
- 多波长权重和 PSF 混合

## 6. MTF 数值实现

新增 `optics_core/huygens_mtf.py`。

核心函数：

```python
def compute_huygens_mtf(
    psf: torch.Tensor,
    *,
    pixel_pitch_um: torch.Tensor,
    frequencies_lp_per_mm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

实现步骤：

1. 输入 PSF 形状为 `(design, field, y, x)`。
2. 计算投影：

   ```python
   sagittal_lsf = psf.sum(dim=-2)
   tangential_lsf = psf.sum(dim=-1)
   ```

3. 选择内部 FFT 长度并在 LSF 两端对称补零，保证原 PSF 中心落在补零数组中心。
4. 对中心化 LSF 执行 `ifftshift`。
5. 使用 `torch.fft.rfft` 得到非负频率复 OTF。
6. 分别除以零频复 OTF。
7. 根据每个设计的 `pixel_pitch_um` 构造实际频率轴。
8. 将复 OTF 的实部和虚部分别线性插值到目标频率，再计算模值。
9. 将零频结果显式设置为 `1.0`，消除浮点误差。

使用 FP64 PSF 时，`torch.fft.rfft` 会生成 complex128，符合项目精度约定。

插值复 OTF优于直接插值 MTF 模值，因为它更接近连续傅里叶变换后再取模的定义。

如果后续发现固定补零仍无法满足 Zemax 误差要求，可以增加一个仅用于验证的直接离散傅里叶计算：

```text
OTF(f) = sum_j LSF[j] × exp(-i 2π f x[j])
```

它可在任意目标频率上直接求值，用来区分“PSF 不一致”和“FFT 插值误差”，但首版公开实现仍使用 FFT。

## 7. 绘图

首版绘图只支持单设计，与 PSF 和点列图约定一致。

一张图绘制所有选定视场：

- 每个视场两条曲线：S 和 T。
- 横轴：`Spatial Frequency (lp/mm)`。
- 纵轴：`Modulation`。
- 范围固定为 `[0, 1]`。
- 文件名包含波长模式；如由测试批量导出，再附加设计和视场索引。

绘图只消费 `MTFResult`，不参与数值计算。

## 8. Zemax 获取函数

新增：

```text
tests/zemax/huygens_mtf.py
```

使用：

```python
zp.analyses.mtf.HuygensMTF(
    pupil_sampling=f"{pupil_sample_count}x{pupil_sample_count}",
    image_sampling=f"{image_sample_count}x{image_sample_count}",
    image_delta=image_delta_um,
    wavelength=zemax_wavelength,
    field=field_index + 1,
    mtf_type="Modulation",
    maximum_frequency=maximum_frequency_lp_per_mm,
    use_polarization=False,
)
```

要求：

1. 直接读取 Zemax 返回的频率、Sagittal 和 Tangential 数据。
2. 不通过 Zemax PSF 再反算 MTF作为期望值。
3. helper 返回 Zemax 的实际频率轴。
4. 记录输入 Image Delta；若设为零，同时记录根据 PSF 规则得到的实际 Image Delta。
5. 单次加载双高斯系统，所有视场和波长 case 复用同一个 `oss`。

## 9. 测试计划

### 9.1 数学契约测试

1. 单像素脉冲 PSF：S/T MTF 全部等于 1。
2. 各向同性高斯 PSF：S/T 曲线一致。
3. 各向异性高斯 PSF：验证 S/T 方向没有交换。
4. PSF 乘任意常数：MTF 不变。
5. PSF 整数像素平移且不截断：MTF 不变。
6. 一维投影 FFT 与二维 FFT 中心切片一致。
7. 补零前后在原生 FFT 频率点上的结果一致。
8. `MTF(0)=1`。
9. 超过 Nyquist 的请求频率应直接报错。
10. 多波长结果等于“混合 PSF 后求 MTF”，且不等于一般情况下的“单色 MTF 模值加权平均”。

### 9.2 Zemax 回归矩阵

使用现有双高斯结构，不重复构造系统：

```text
field_index:      0, 1, 2
wavelength_index: 0, 1, 2, -1
```

建议首轮：

```text
pupil sampling: 64 × 64
image sampling: 64 × 64
image delta: 0
maximum frequency: 150 lp/mm
```

逐 case 输出：

- Zemax 与 OpticsCore Image Delta。
- S/T 最大绝对误差。
- S/T 平均绝对误差。
- 0、50、100、150 lp/mm 附近的对比值。
- PSF 边缘能量比例，用于识别图像窗口截断。
- 两端计算耗时。

建议初始阈值：

```text
MTF 最大绝对误差 <= 0.02
MTF 平均绝对误差 <= 0.005
```

未达到阈值的 case 使用 `pytest.xfail`，并打印原始误差。

### 9.3 误差定位顺序

若与 Zemax 不一致，按以下顺序排查：

1. 使用同一份 OpticsCore PSF，对比一维投影 FFT 与二维 FFT 切片。
2. 使用直接 DFT 排除补零和插值误差。
3. 比较 OpticsCore 和 Zemax 的 Image Delta、图像范围及频率轴。
4. 比较单色 PSF；若 PSF 已不一致，MTF 模块不单独修正。
5. 检查多波长是否先混合 PSF 再求 MTF。
6. 检查 S/T 是否使用局部像面 `x/y`。
7. 检查 PSF 是否在图像边缘被截断。
8. 最后检查 Zemax 是否切换到了球面波 Huygens 相位参考。

## 10. 实施顺序

1. 将现有 Huygens PSF 内核扩展为多视场 batch，不改变公开 PSF 结果语义。
2. 增加 PSF batch contract 测试，确认一次追迹覆盖所有视场。
3. 实现 LSF 投影、补零、FFT、归一化和频率插值。
4. 接通 `ModulationTransferFunction.run()`。
5. 增加数学契约测试。
6. 增加 `tests/zemax/huygens_mtf.py`。
7. 增加双高斯多视场、多波长回归矩阵。
8. 根据回归结果确定默认补零倍数；不要依据猜测硬编码 Zemax 内部行为。
9. 最后实现单设计多视场曲线导出。

## 11. 验收标准

1. 一次系统构造和一次 batch PSF 追迹可完成全部目标视场。
2. MTF 模块不包含重复的 Huygens 积分或波长混合逻辑。
3. S/T 方向契约测试能检测轴交换。
4. 单色和全波长混合均有 Zemax 直接结果对标。
5. 所有 MTF 输出使用 FP64/complex128。
6. 频率轴单位和 Nyquist 限制明确。
7. 未对齐 case 进入 xfail，且输出足够的定位信息。

## 12. 官方资料

1. [OpticStudio User Guide: Huygens MTF](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v261/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Huygens_MTF.html)
2. [OpticStudio User Guide: Huygens PSF](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v261/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Huygens_PSF.html)
3. [OpticStudio User Guide: Units](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v261/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Units.html)
4. [Ansys: Methods for analyzing MTF in OpticStudio](https://optics.ansys.com/hc/en-us/articles/42661772706835-Methods-for-analyzing-MTF-in-OpticStudio)
5. [Ansys: Why are FFT and Huygens MTF results different on tilted image surfaces?](https://optics.ansys.com/hc/en-us/articles/42661833682707-Why-are-FFT-and-Huygens-MTF-results-different-on-tilted-image-surfaces)
6. [Ansys: Contrast Loss Map in OpticStudio](https://optics.ansys.com/hc/en-us/articles/42661976342035-Contrast-Loss-Map-in-OpticStudio)
