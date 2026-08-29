# 惠更斯 PSF 计算原理说明

本文档对应当前 `optics_core/huygens_psf.py` 的实现，重点说明代码中的计算流程和张量含义，不展开基础衍射理论。

## 当前实现语义

当前 PSF 采用接近 Zemax Huygens PSF 描述的像面局部平面波叠加方式：

1. 从入瞳采样一组光线。
2. 正向追迹到像面，记录每根光线的像面坐标、方向余弦和 OPL。
3. 将像面 OPL 反向修正到出瞳参考球面，得到用于相位计算的 OPD。
4. 在以主光线像点为中心的像面网格上，把每根光线看作一束局部平面波并相干叠加。
5. 对复振幅取模平方得到 PSF。

当前实现不使用 `exp(i k r) / r`，也不使用 `K_factor`。这意味着代码不把每个出瞳采样点当成球面子波点源，而是使用像面局部平面波近似。

## 光线采样

入口函数是：

```python
run_huygens_psf(system, settings)
```

其中 `settings.pupil_sample_count` 控制入瞳方形采样数量。例如默认值 32 表示先生成 `32 x 32` 个 pupil 坐标。自动采样器会额外在末尾补一根主光线，主光线用于参考，不参与积分。

当前采样器是：

```python
SquarePupilSampler(nx=pupil_sample_count, ny=pupil_sample_count)
```

后续会通过：

```python
inside_pupil = pupil_x * pupil_x + pupil_y * pupil_y <= 1.0 + 1e-12
```

过滤圆形归一化入瞳外的采样点。

## 正向追迹与 OPL

`trace_huygens_psf_rays(...)` 会调用：

```python
build_input_rays_from_sample(...)
system.tracer.trace(...)
```

生成并追迹光线。追迹结果中 PSF 需要的字段包括：

```python
x, y, z          # 光线在像面的交点，单位 mm
l, m, n          # 光线在像面的方向余弦
valid            # 光线是否有效
opl              # 从物方起点累计到像面的光程，单位 mm
wavelength_um    # 波长，单位 um
```

当前 PSF 要求追迹结果形状为：

```text
(design, field=1, wavelength, ray)
```

因此代码会先去掉 field 维度，得到：

```text
design x wavelength x ray
```

## 有效积分光线

参与积分的光线由三类条件共同决定：

1. 光线追迹有效：`valid == True`
2. pupil 坐标在单位圆内
3. 不包含末尾额外补充的主光线

对应代码：

```python
valid_points = valid & inside_pupil.reshape(1, 1, -1)
valid_points[..., int(sample_ray_count):] = False
```

这里 `sample_ray_count` 表示真实采样光线数量，末尾的 reference chief ray 不参与积分。

方形 pupil 网格还会为每个节点计算面积积分权重。内部节点权重相同，圆周附近节点
按照网格单元与单位圆的相交面积降低权重，单位圆外权重为 0。所有面积权重归一化
后保留在 `SamplingResult` 中，并由 PSF 分析流程显式传入积分器。

## 像面网格

`_image_grid(...)` 以主光线像面交点为中心生成方形网格：

```python
axis = (arange(image_sample_count) - image_sample_count // 2) * pixel_pitch_mm
```

默认像素间隔为：

```python
HUYGENS_PSF_PIXEL_PITCH_UM = 0.5
```

因此默认 `32 x 32` 网格的采样中心为 `-8.0 ~ 7.5 um`，采样间隔为
`0.5 um`，索引 16 对应主光线位置 `0 um`。绘图时按照像素边界显示为
`-8.25 ~ 7.75 um`，与 Zemax Data Grid 的范围一致。

输出网格为：

```python
image_x, image_y, image_z
```

形状为：

```text
design x wavelength x image_y x image_x
```

## 出瞳参考 OPD

PSF 相位不能直接使用像面 OPL 差。当前代码先从像面沿反向光线求交出瞳参考球面：

```python
exit_pupil_points, opd_exit = _exit_pupil_reference_data(...)
```

参考球面的球心在轴上：

```text
(0, 0, exit_pupil_z)
```

半径取主光线像点到该球心的距离：

```python
sphere_radius = norm(chief_points - axial_exit_center)
```

对每根光线，设反向方向为：

```python
backward_direction = -ray_directions
```

求交参数 `t_back` 后，当前按 Zemax 的“参考光程减当前光程”约定定义 OPD：

```python
exit_pupil_points = image_points - t_back * ray_directions
opd_exit = t_back - opl
```

然后对同一 `design x wavelength` 下的有效光线去 piston：

```python
opd_exit = opd_exit - mean(opd_exit)
```

去 piston 不改变 PSF 强度分布，只去掉共同相位，便于数值稳定和调试。

## 局部平面波叠加

`_huygens_integral(...)` 是当前 PSF 的核心。

对第 `j` 根光线，它在像面网格点 `P` 上的相位由两部分组成：

```text
phase_j(P) = k * (OPD_Zemax_j + dot(d_j, P - P_exit_j))
```

其中：

```text
k      = 2*pi / wavelength_mm
OPD_Zemax_j = 按“参考光程减当前光程”定义的出瞳参考 OPD
d_j      = 光线在像面的单位方向向量
P_exit_j = 反向求交得到的出瞳参考球面交点
P        = 当前像面网格点
```

对应代码：

```python
propagation = image_grid[:, :, None] - exit_pupil_points[:, :, :, None, None, :]
phase_tilt = torch.sum(ray_directions[:, :, :, None, None, :] * propagation, dim=-1)
phase = wave_number[:, :, None, None, None] * (opd_exit[:, :, :, None, None] + phase_tilt)
kernel = torch.complex(torch.cos(phase), torch.sin(phase))
```

这里必须保证 OPD 和传播坐标使用同一个参考面。若使用出瞳参考 OPD，传播项必须从
`P_exit` 出发，不能从像面交点 `P_image` 出发。此前混用 `OPD_exit` 和 `P_image`
会额外残留逐光线变化的相位。

当前复指数采用 `exp(+i*phase)`，并按 Zemax 返回的 OPD 符号直接写入相位。若未来更换
时间因子或复指数符号，OPD 与传播项的符号必须成组调整，不能只翻转其中一项。

所有有效光线相干叠加：

```python
weighted_kernel = kernel * pupil_weights
amplitude = weighted_kernel.sum(dim=2)
psf = real(amplitude * conj(amplitude))
```

这里 `dim=2` 是 ray 维度。

`pupil_weights` 表示 pupil 面积积分权重，不表示光线能量，因此直接乘复振幅，不取
平方根。实际 PSF 和理想 PSF 使用完全相同的权重。

## 理想 PSF 与 Strehl

Strehl 需要一个无像差参考峰值。当前代码用同一组有效光线、同一组方向余弦，但去掉 OPD 项：

```python
ideal_phase = wave_number[:, :, None, None, None] * phase_tilt
ideal_kernel = torch.complex(torch.cos(ideal_phase), torch.sin(ideal_phase))
```

然后同样相干叠加并取峰值：

```python
strehl_ratio = max(psf) / max(ideal_psf)
```

这一定义便于内部保持同采样、同 mask、同方向余弦。它是否能严格复刻 Zemax 的 Strehl，还需要继续通过中间变量对标确认。

## 当前实现和球面子波版本的差异

当前实现没有以下项：

```text
K_factor
exp(i k r) / r
```

因此它不是严格的出瞳点源球面波积分。这样做的原因是 Zemax 对 Huygens PSF 的公开说明强调：OpticStudio 会把每根光线转换为像面处的 planar wavefront 后相干叠加。为了优先对标 Zemax，当前代码采用平面波形式。

如果后续需要研究更接近参考代码 `reference/GYOptics/DiffOptics.py` 的球面子波实现，可以单独增加一个内部模式，但不建议和 Zemax 对标主链混在一起。

## 后续对标重点

当前 PSF 是否能和 Zemax 对齐，主要取决于以下中间量：

1. 同一 pupil 坐标下的像面 `x, y, z, l, m, n`
2. 像面 OPL 与出瞳参考 OPD
3. Zemax 的 pupil 采样、圆孔边界和权重语义
4. 像面网格原点、行列方向和偶数采样半像素偏移
5. Zemax 的 Strehl 参考峰值定义

建议优先增加诊断测试，逐项打印和比较这些中间变量，不要只看最终 PSF 图像。
