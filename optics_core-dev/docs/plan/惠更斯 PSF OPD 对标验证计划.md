# 惠更斯 PSF OPD 对标验证计划

## 目标

固定相同的系统、视场、波长和归一化 pupil 坐标，分别取得 Zemax 与 OpticsCore 的出瞳参考 OPD，判断当前 PSF 差异发生在：

1. 光线追迹和 OPD 计算阶段
2. pupil 采样与权重阶段
3. 惠更斯积分阶段

本轮只验证 0 视场、主波长和单设计，使用：

```text
tests/zemax/zmx_files/Double Gauss 28 degree field real.zmx
```

## 对标量定义

两侧统一比较出瞳参考 OPD：

```text
OPD_exit = distance(image point, exit-pupil reference sphere) - OPL_image
```

比较前统一：

1. 单位转换为 mm
2. 关闭 OPD modulo 2 pi
3. 使用 exit pupil reference
4. 不去除 tilt
5. 采用 Zemax 的“参考光程减当前光程”符号，并以主光线为零点或统一去 piston

不能把 OpticsCore 的像面 `OPL - chief OPL` 直接与 Zemax Wavefront Map 比较。两者参考面不同。

## Zemax 数据获取

### 1. 逐点 OPD：主要验证方式

在 `tests/zemax/opd.py` 增加：

```python
fetch_zemax_exit_pupil_opd(
    spec,
    oss,
    pupil_coordinates,
    *,
    field_index,
    wavelength_index,
)
```

使用 ZOS-API Batch Ray Trace：

```text
OpenBatchRayTrace()
CreateNormUnpol(...)
AddRay(..., OPDMode.CurrentAndChief)
```

直接传入与 OpticsCore 完全相同的：

```text
Hx, Hy, Px, Py, wavelength number
```

其中：

```text
OpticsCore wavelength_index = 0-based
Zemax wavelength number     = wavelength_index + 1
```

`CurrentAndChief` 应直接返回当前光线相对主光线的 OPD。首次实现时需要通过以下检查确认返回值字段和单位：

1. `(Px, Py) = (0, 0)` 的 OPD 应接近 0
2. 输出应与 Zemax Wavefront Map 同位置数值一致
3. 若返回单位为 waves，则转换为：

```text
opd_mm = opd_waves * wavelength_um * 1e-3
```

逐点 Batch Ray Trace 是主要参考，因为它无需对 Wavefront Map 做插值，也不会产生 pupil 网格位置不一致。

### 2. Wavefront Map：辅助验证方式

再增加：

```python
fetch_zemax_wavefront_map(...)
```

调用 zospy：

```python
zp.analyses.wavefront.WavefrontMap(
    field=field_index + 1,
    wavelength=wavelength_index + 1,
    surface="Image",
    sampling="32x32",
    reference_to_primary=False,
    use_exit_pupil=True,
    remove_tilt=False,
)
```

Wavefront Map 用于检查：

1. OPD 是否近似旋转对称
2. x/y 方向是否交换或翻转
3. 边缘和中心 OPD 的数量级
4. Batch Ray Trace 返回的 OPD 字段和单位是否正确

Wavefront Map 的网格不应直接假定与 `SquarePupilSampler` 完全相同。若坐标不一致，只比较共同采样点，或仅用于图形和径向趋势验证。

## OpticsCore 数据获取

使用与 PSF 完全相同的采样结果：

```python
sample = SquarePupilSampler(nx=32, ny=32).sample()
```

Zemax 和 OpticsCore 都使用：

```python
pupil_coordinates = sample.pupil_coordinates[:sample.sample_ray_count]
```

OpticsCore 流程：

1. `system.prepare()`
2. 使用 `build_input_rays_from_sample(...)` 构建光线
3. 正向追迹到像面并取得 `x, y, z, l, m, n, opl`
4. 使用 PSF 当前的出瞳参考球反向求交计算：

```text
P_exit
OPD_exit
```

验证代码应复用生产实现，不要在测试中复制第二套 OPD 公式。若测试直接调用私有函数不便，可将 `_exit_pupil_reference_data()` 移至 `optics_core/opd.py`，但暂不增加新的类或结果结构。

## 第一阶段：少量 pupil 点

先使用容易检查的显式坐标：

```text
( 0.0,  0.0)
( 0.5,  0.0)
(-0.5,  0.0)
( 0.0,  0.5)
( 0.0, -0.5)
( 0.5,  0.5)
(-0.5,  0.5)
( 0.5, -0.5)
(-0.5, -0.5)
```

逐点打印：

```text
Px, Py
Zemax OPD (mm)
OpticsCore OPD (mm)
absolute error (mm)
error (waves)
valid/error/vignette code
```

同时验证旋转对称系统应满足：

```text
OPD(+Px, 0) ~= OPD(-Px, 0)
OPD(0, +Py) ~= OPD(0, -Py)
OPD(Px, Py) ~= OPD(Py, Px)
```

若这一阶段失败，应先检查：

1. Zemax OPD 返回字段和单位
2. OPD 正负号
3. Zemax Reference OPD 设置
4. 出瞳位置和参考球半径
5. `t_back` 求交根和方向符号
6. OpticsCore OPL 累积

此时不要继续调整惠更斯积分。

## 第二阶段：完整 32x32 pupil

复用 PSF 的 `SquarePupilSampler(32, 32)`，裁掉单位圆外点和末尾 reference chief ray。

输出：

1. Zemax OPD pupil map
2. OpticsCore OPD pupil map
3. OPD difference map
4. x 轴、y 轴和 45 度方向剖面
5. 最大误差、平均绝对误差和 RMS 误差

建议同时报告 mm 和 waves：

```text
error_waves = error_mm / (wavelength_um * 1e-3)
```

初始目标：

```text
max abs error < 1e-4 waves
RMS error     < 1e-5 waves
```

若误差呈固定常数，说明只是 piston；若呈线性斜面，说明 tilt/reference point 不一致；若呈径向高阶分布，重点检查参考球面和 OPL；若呈四重对称，重点检查方形 pupil 采样映射和 x/y 坐标。

## 第三阶段：隔离积分误差

当 OPD 对齐后，在两侧使用同一份 OpticsCore pupil 数据包：

```text
pupil coordinates
valid mask
OPD_exit
image point
ray direction
```

执行以下实验：

1. `OPD = 0`，验证无像差 PSF 是否为圆对称
2. 使用 Zemax OPD 输入 OpticsCore 积分器
3. 使用 OpticsCore OPD 输入 OpticsCore 积分器
4. 比较两者 PSF

判断原则：

| 结果 | 结论 |
|---|---|
| OPD 已对齐，PSF 仍不一致 | 积分公式、采样权重或归一化问题 |
| Zemax OPD 输入后 PSF 对齐 | OpticsCore OPD 问题 |
| Zemax OPD 输入后仍不对齐 | 积分或 pupil 权重问题 |
| `OPD=0` 仍明显非圆形 | pupil 网格、mask、权重或像面相位项问题 |

## 文件安排

建议新增：

```text
tests/zemax/opd.py
tests/regression/test_huygens_opd_against_zemax.py
tests/output/zemax_opd_map.png
tests/output/optics_core_opd_map.png
tests/output/opd_difference_map.png
```

`tests/zemax/opd.py` 只负责 Zemax 数据获取；采样、OpticsCore 追迹、误差统计和绘图放在 regression 测试中。

## 完成标准

1. Zemax 与 OpticsCore 使用完全相同的 pupil 坐标
2. 主光线、单位、波长和 OPD 参考设置均有打印
3. 少量 pupil 点可以逐行核对
4. 完整 pupil map 可以观察误差空间分布
5. 能明确判断 PSF 差异来自 OPD 之前还是惠更斯积分阶段
