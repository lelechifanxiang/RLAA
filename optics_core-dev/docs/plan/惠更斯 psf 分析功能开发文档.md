# 惠更斯 PSF 分析功能开发文档

## 1. 目标

为 `optics_core` 增加一版可尽快落地的 `Huygens PSF` 数值分析能力，优先满足下面几件事：

1. 和 Zemax 的 Huygens PSF 建立稳定对标链路
2. 尽量复用现有 `first_order`、`trace`、`analysis`、`tests/zemax` 能力
3. 不为首版开放可选参数，不为了“以后可能扩展”先做框架
4. 代码尽量短，语义尽量直接，可读性优先

当前阶段不追求：

1. 通用波前分析框架
2. 通用 Zernike 分解框架
3. 用户自定义 pupil/image sampling
4. 用户自定义 wavelength / field / centroid / normalize / polarization
5. True Color、log PSF、real/imaginary amplitude、phase map
6. 自动切换 Zemax 的平面/球面 Huygens phase reference

---

## 2. 对当前四步计划的判断

你给出的四步主线是对的，但还缺几条关键前提。如果不先补上，后面即使代码跑通，也很难和 Zemax 真正对齐。

原计划：

1. 完成出瞳估计功能
2. 完成 OPD 功能开发，和 Zemax 对标
3. 完成出瞳到像面的惠更斯计算过程
4. 和 Zemax 的 Huygens PSF 对标

主要漏洞有 5 个：

1. **缺少固定分析语义这一步**
   Huygens PSF 不是只要有 OPD 就能算。必须先固定：
   - 用哪个波长
   - 像面中心参考是 chief ray 还是 centroid
   - image sampling / image delta 是多少
   - normalize 是否打开
   - 是否考虑 polarization
   - Huygens integral 用平面参考还是球面参考

2. **“OPD” 需要先明确是哪个 OPD**
   这里不能只做“到像面的光程差”。
   Zemax 的衍射相关 OPD 默认是**以出瞳 reference sphere 为参考**的 OPD，这和简单的像面 OPL 差不是一回事。

3. **出瞳估计不只是位置，还包括半径和 stop 成像关系**
   入口瞳当前是由 `SystemAperture(kind="entrance_pupil_diameter")` 直接给半径。
   但出瞳是 stop 在像空间的近轴像，半径不能直接复用 entrance pupil 的逻辑，必须明确 stop 物理孔径半径的来源。

4. **缺少像面采样策略**
   Huygens PSF 必须有固定 image grid，才能和 Zemax 一格一格对比。
   如果沿用 Zemax `Image Delta = 0` 的自动公式，会额外引入 working F/# 语义和默认步长复现问题，不适合首版快闭环。

5. **缺少 phase reference 策略**
   Zemax 在 Huygens PSF 中会在平面参考和球面参考之间自动切换。
   这件事如果首版也照搬，会明显拉长开发时间。

结论：

- 你的四步方向没有问题
- 但在“出瞳估计”之前，必须先加一个“固定 Zemax 语义”的第 0 步
- 在“OPD 开发”里，必须明确做的是**exit pupil referenced OPD**
- 首版建议**主动简化 Huygens 设置**，不要追 Zemax 的所有自动行为

---

## 3. 首版范围建议

首版目标建议收缩为：

1. 固定主波长 `primary wavelength`
2. 固定 `Use Centroid = False`，始终以 chief ray 为中心
3. 固定 `Type = Linear`
4. 固定 `Normalize = False`
5. 固定 `Use Polarization = False`
6. 固定 Huygens phase reference 为**平面参考**
7. 固定 pupil sampling、image sampling、image delta
8. 对系统全部 field 逐个输出 PSF
9. 输出 `psf`、`strehl_ratio`、`pixel_pitch_um`

这样做的好处：

1. Zemax 对标边界清晰
2. OPD 和 Huygens 积分的实现都会简单很多
3. 不需要为了以后扩展先做一层参数系统

---

## 4. 固定 Zemax 语义

首版建议把 Zemax 侧和 `optics_core` 侧都固定成下面这组设置：

1. `Analysis = Huygens PSF`
2. `Surface = Image`
3. `Field = 系统全部 field，逐个循环`
4. `Wavelength = Primary`
5. `Use Centroid = False`
6. `Type = Linear`
7. `Normalize = False`
8. `Use Polarization = False`
9. `Rotation = 0`
10. `Pupil Sampling = 64 x 64`
11. `Image Sampling = 128 x 128`
12. `Image Delta = 0.5 um`
13. `Method to Compute Huygens Integral = Planar`

这里有两个刻意的取舍：

### 4.1 不做 polychromatic

Zemax 的 polychromatic Huygens PSF 会引入：

1. wavelength weight
2. 不同波长的非相干叠加
3. UI 和 API 上的额外设置语义

这对首版不是必须项。先把 primary wavelength 的单色 PSF 做通，再扩展多波长。

### 4.2 不用 Zemax 的自动 image delta

Zemax 官方说明里，`Image Delta = 0` 会按默认公式自动给步长，这个公式依赖 pupil sample、波长和 working F/#。  
这会额外引入 working F/# 语义，不利于快速闭环。

因此首版直接固定一个显式值，例如 `0.5 um`，同时 Zemax helper 也使用同样的固定值。

---

## 5. 和 Zemax 语义相关的关键事实

下面这些语义会直接影响实现方式。

### 5.1 Huygens PSF 的计算平面

Zemax 官方说明指出，Huygens PSF 不是直接在原始 image surface 上做，而是在：

- **chief ray 像面截点处**
- **与 image surface 相切**
- **法向沿 image surface 局部法向**

定义的虚拟平面上计算。

这意味着首版实现里不能直接假设“像面永远是全局 `z = const` 的平面”。  
但因为当前追迹已经能记录 image surface 交点和法向，所以可以直接复用现有交点链路。

### 5.2 Zemax 的衍射相关 OPD 参考在出瞳

Zemax 官方关于 Exit Pupil 的说明给出的定义是：

- 先计算物方到像面的绝对 OPL 差
- 再加上从像面回到 exit pupil reference sphere 的 correction term
- 最终得到的是对衍射计算真正有意义的 OPD

因此这里不能把“到像面的光程差”直接拿去做 Huygens phase。

### 5.3 Zemax 会在平面/球面 phase reference 间自动切换

官方说明指出，若像区较大，Zemax 会改用 spherical wave reference。  
这套自动判据首版不建议复现。

首版建议：

1. 在 Zemax 测试侧固定为 planar
2. 在 `optics_core` 侧只实现 planar

这样最适合快速闭环。

---

## 6. 推荐开发顺序

建议把原来的四步改成下面六步。

### 6.0 先固定语义和测试边界

先把第 4 节的固定 Zemax 设置写死，不开放用户参数。  
这是后面所有实现和回归测试的前提。

建议同步收缩接口：

```python
@dataclass(slots=True)
class PSFSettings:
    pass
```

或者最多只保留：

```python
@dataclass(slots=True)
class PSFSettings:
    save_path: str | None = None
```

不要保留 `grid_size`、`oversampling` 这种首版不会开放的字段。

---

### 6.1 完成 exit pupil 位置和半径估计

目标：

1. 计算 `exit_pupil_z`
2. 计算 `exit_pupil_radius`
3. 和 Zemax 的 paraxial exit pupil 对标

建议实现方式：

1. 在 [optics_core/first_order.py](/abs/path/c:/Users/huweijian/Project/optics_core/optics_core/first_order.py:1) 中新增一套与 entrance pupil 对称的 helper
2. 从 stop 面发出两条近轴探测光线，正向追迹到像方
3. 用两条光线反解 exit pupil 位置和 pupil magnification
4. 用 stop 物理孔径半径乘 pupil magnification 得到 `exit_pupil_radius`

这里要注意：

1. `exit_pupil_radius` 的物方来源应是 **stop surface 的实际孔径半径**
2. 不能直接复用 `SystemAperture.value / 2`

建议新增函数：

1. `stop_aperture_radius(system)`
2. `build_exit_pupil_probe_rays(...)`
3. `solve_exit_pupil_from_probes(...)`
4. `compute_exit_pupil(system)`

Zemax 测试建议：

1. 扩展 [tests/zemax/first_order.py](/abs/path/c:/Users/huweijian/Project/optics_core/tests/zemax/first_order.py:1)
2. 直接读取 `EXPP`
3. 直接读取 `EXPD` 或等价出瞳直径操作数
4. 新增 `test_exit_pupil_against_zemax.py`

---

### 6.2 先补通用 OPL 累计，再做 exit-pupil OPD

这一步建议拆成两层：

#### 第一层：通用 OPL

目标：

1. 在顺序追迹过程中累计每条 ray 的绝对 optical path length
2. 不在 tracer 内部引入 Huygens 专属语义

建议最省代码的做法：

1. 复用现有 `RayBundle.opd`
2. 在 `record_opd=True` 时，把它当成**累计 OPL，单位 mm**

理由：

1. 当前 `opd` 字段已经存在，但主链未真正使用
2. 这样不用额外新建结果结构
3. Huygens 模块后处理时再把它转换成真正的 OPD

不建议现在做的事情：

1. 新建一套独立 optical path result dataclass
2. 先做通用 wavefront analysis 框架

#### 第二层：exit-pupil referenced OPD

有了 OPL 之后，再在 Huygens/OPD 专属模块里做：

1. chief ray 到像面的绝对 OPL
2. 各 ray 到像面的绝对 OPL
3. 从像面回到 exit pupil reference plane 的 correction term
4. 转成最终的 exit-pupil referenced OPD

建议新增一个局部模块，例如：

- `optics_core/opd.py`

但不要为了它再套一层大分析框架。

Zemax 测试建议：

1. 先做 **absolute OPL / per-ray OPD** 的小回归
2. 再做 pupil OPD map 回归

这会比直接一步跳到 PSF 更容易排错。

---

### 6.3 先做独立的 pupil phase 数据提取

在做 Huygens 积分前，建议先拿到一个稳定的 pupil 数据包：

1. `pupil_x`
2. `pupil_y`
3. `valid_mask`
4. `opd_mm` 或 `phase_rad`
5. `amplitude`
6. `chief_ray_image_point`
7. `image_surface_normal`
8. `exit_pupil_z`
9. `exit_pupil_radius`

建议这个数据包只在 `huygens_psf.py` 内部流转，不要升级成公共框架对象。  
首版里用 `dict[str, Any]` 就够了。

这样有两个好处：

1. Huygens 积分可以和 OPD 计算解耦
2. 调试时可以单独打印 pupil phase，而不必每次都跑完整 PSF

---

### 6.4 实现从 exit pupil 到像面切平面的 Huygens 积分

建议新建模块：

- `optics_core/huygens_psf.py`

主链建议非常直接：

1. 固定 square pupil 采样
2. 构建入射光线并追迹到像面
3. 取 image surface 上 chief ray 截点和局部法向
4. 计算 exit-pupil referenced OPD
5. 在 chief ray 截点处构建像面切平面 grid
6. 对每个像素做 Huygens 复振幅叠加
7. 取模平方得到 PSF
8. 用同一 pupil mask、同一采样、零像差相位生成 reference PSF peak
9. 计算 `strehl_ratio`

首版建议的幅值模型：

1. 每条有效 ray 幅值先统一设为 `1.0`
2. pupil 外和 vignetted ray 直接置零
3. 不做 polarization
4. 不做额外 apodization / obliquity 修正

原因：

1. 这是最容易和固定 Zemax 设置闭环的起点
2. 如果后续发现和 Zemax 仍存在系统性偏差，再局部引入振幅修正

---

### 6.5 做 Zemax Huygens PSF 回归

建议新增：

1. [tests/zemax/huygens_psf.py](/abs/path/c:/Users/huweijian/Project/optics_core/tests/zemax/huygens_psf.py)
2. [tests/regression/test_huygens_psf_against_zemax.py](/abs/path/c:/Users/huweijian/Project/optics_core/tests/regression/test_huygens_psf_against_zemax.py)

Zemax helper 建议直接调用 Huygens PSF analysis：

1. 固定 `Field`
2. 固定 `Wavelength = Primary`
3. 固定 `Pupil Sampling = 64 x 64`
4. 固定 `Image Sampling = 128 x 128`
5. 固定 `Image Delta = 0.5 um`
6. 固定 `Use Centroid = False`
7. 固定 `Normalize = False`
8. 固定 `Use Polarization = False`
9. 固定 `Type = Linear`

回归不要只看一项，至少检查：

1. `strehl_ratio`
2. `pixel_pitch_um`
3. `psf` 峰值位置是否落在中心像素附近
4. 中心行 / 中心列截面
5. 完整 2D PSF grid 的最大绝对误差

首版阈值建议比 spot 宽松一些，先保证趋势和峰形正确，再逐步收紧。

---

## 7. 推荐接口

建议继续沿用现有入口：

```python
result = system.analysis.psf().run()
```

首版不建议开放任何用户参数。  
如果确实要保留一个设置对象，也建议只保留 `save_path`。

建议结果结构：

```python
@dataclass(slots=True)
class PSFResult(AnalysisResult):
    psf: ArrayLike | None = None
    strehl_ratio: ArrayLike | None = None
    pixel_pitch_um: float | None = None
```

其中：

1. `psf`：建议形状为 `(system, field, image_y, image_x)`
2. `strehl_ratio`：建议形状为 `(system, field)`
3. `pixel_pitch_um`：固定标量

不建议首版加入：

1. `centroid`
2. `encircled_energy`
3. `complex_amplitude`
4. `phase_map`

---

## 8. 推荐代码落点

### 8.1 需要新增或修改的模块

1. [optics_core/analysis.py](/abs/path/c:/Users/huweijian/Project/optics_core/optics_core/analysis.py:1)
   - 收缩 `PSFSettings`
   - 接通 `PointSpreadFunction.run()`

2. [optics_core/first_order.py](/abs/path/c:/Users/huweijian/Project/optics_core/optics_core/first_order.py:1)
   - 增加 `compute_exit_pupil(...)`
   - 增加 stop aperture 相关 helper
   - 扩展 `compute_first_order_batch(...)`

3. `optics_core/huygens_psf.py`
   - Huygens PSF 主链

4. 视情况新增 `optics_core/opd.py`
   - 只放 exit-pupil referenced OPD 相关 helper

5. [optics_core/tracing/_core.py](/abs/path/c:/Users/huweijian/Project/optics_core/optics_core/tracing/_core.py:1) 及相关追迹模块
   - 接通 `record_opd=True`
   - 累计通用 OPL

### 8.2 需要新增的 Zemax helper

1. `tests/zemax/huygens_psf.py`
2. 扩展 [tests/zemax/first_order.py](/abs/path/c:/Users/huweijian/Project/optics_core/tests/zemax/first_order.py:1)
3. 视情况新增 `tests/zemax/opd.py`

### 8.3 需要新增的回归测试

1. `tests/regression/test_exit_pupil_against_zemax.py`
2. `tests/regression/test_opd_against_zemax.py`
3. `tests/regression/test_huygens_psf_against_zemax.py`

---

## 9. 具体开发建议

### 9.1 不要先做 wavefront analysis

虽然 Huygens PSF 需要 OPD，但这不代表要先把 `system.analysis.wavefront()` 整套做完。  
那会把目标扩成另一个 feature。

更合适的做法是：

1. 先做 Huygens PSF 自己需要的 OPL / OPD helper
2. 以后如果真要做 wavefront analysis，再从这里抽公共部分

### 9.2 不要先做通用 pupil framework

首版只需要固定 square pupil 采样。  
不要为了以后支持 hexapolar、annular、user mask，先写一层复杂采样抽象。

### 9.3 不要一开始就追求 Zemax 自动行为全复刻

首版最容易拖慢进度的两件事是：

1. 自动 image delta
2. 自动平面/球面 phase reference 切换

这两项都应该先固定掉。

### 9.4 先对简单系统，再对 Double Gauss

建议测试顺序：

1. `paraxial_single_lens.zmx`
2. `four_surface_spherical.zmx`
3. `Double Gauss 28 degree field.zmx`

原因：

1. 单透镜更容易看清 OPD 参考是否写对
2. Double Gauss 更适合作为最终回归，不适合作为首个排错对象

---

## 10. 最终建议的执行计划

建议最终执行顺序是：

1. 固定 Huygens PSF 的 Zemax 语义，不开放用户参数
2. 补 `exit_pupil_z` 和 `exit_pupil_radius`，先和 Zemax 对齐
3. 接通 tracer 中的通用 OPL 累计
4. 基于 exit pupil reference 实现 OPD，并和 Zemax 对齐
5. 实现固定参数的单色 Huygens PSF 主链
6. 用 Zemax Huygens PSF 做 2D grid + strehl 回归

如果要再压缩范围，我建议首版目标就定成：

1. `primary wavelength`
2. `square pupil`
3. `chief-ray centered`
4. `planar reference`
5. `linear / unpolarized / non-normalized`
6. `all fields`

这个版本最符合你当前“尽快实现功能、代码精简、复用已有能力、不要过度封装”的目标。

---

## 11. 参考

官方资料中，对本功能最关键的两点是：

1. Huygens PSF 在 chief ray 截点处、沿 image surface 局部法向的切平面上计算
2. Zemax 的衍射相关 OPD 默认参考在 exit pupil / reference sphere

参考链接：

1. Huygens PSF  
   https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v25101/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Huygens_PSF.html
2. Exit Pupil  
   https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v252/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Exit_Pupil.html
3. ZOS-API Huygens PSF settings interface  
   https://developer.ansys.com/docs/zemax-opticstudio-zos-api-2026-r1/reference/interface_z_o_s_a_p_i_1_1_analysis_1_1_settings_1_1_psf_1_1_i_a_s___huygens_psf.md
