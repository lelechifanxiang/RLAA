# Spot Diagram 分析功能开发文档

## 1. 目标

为 `optics_core` 增加一个简洁可用的 `spot diagram` 分析能力，当前阶段只做最常用、最容易和 Zemax 对齐的部分，不预留过多自由参数。

本阶段功能目标：

1. 支持 `hexapolar` 和 `square` 两种 pupil 采样方式
2. 默认采样方式为 `hexapolar`
3. 支持一个统一的 `ray_density` 参数，默认值与 Zemax Standard Spot 默认值保持一致
4. 不支持自由选择波长、视场、Refer To 等额外参数
5. 输出每个视场的 `RMS radius` 和 `GEO radius`
6. 支持保存 spot 图；保存图片时额外输出原始散点，便于绘图和人工检查

---

## 2. Zemax 对齐基线

### 2.1 本阶段对齐对象

本功能建议直接对齐 Zemax 的 **Standard Spot Diagram**，而不是对齐 `reference/DiffOptics.py`。

原因：

1. `DiffOptics.py` 本身并没有和 Zemax 严格对齐
2. Spot Diagram 对采样模式、参考点、波长组合方式非常敏感
3. 既然最终目标是和 Zemax 对齐，就应该直接以 Zemax Standard Spot 为准

---

### 2.2 Zemax Standard Spot 的关键默认行为

根据 Ansys 官方帮助和本机 ZOSAPI 实测，Standard Spot 的关键默认行为如下：

1. `Pattern = Hexapolar`
2. `RayDensity = 30`
3. `Refer To = ChiefRay`
4. 默认按系统定义的全部波长参与 polychromatic 计算
5. 每个视场单独输出一个 spot
6. vignetted rays 不参与最终 RMS/GEO 计算

---

### 2.3 Ray Density 的 Zemax 语义

官方文档对 `Ray Density` 的定义非常明确：

#### Hexapolar

`Ray Density` 表示 **hexapolar rings 的层数**。

若密度记为 `N`，总光线数为：

`1 + 6 + 12 + ... + 6N = 1 + 3N(N + 1)`

例如：

- `N = 3` -> `37` 条光线
- `N = 30` -> `2791` 条光线

#### Square

`Ray Density` 表示 **方阵横向和纵向的采样数**。

若密度记为 `N`，候选光线数为：

`N * N`

例如：

- `N = 30` -> `900` 条候选光线

注意：  
在圆形 pupil 下，square 模式发出的部分角点光线会落在 pupil 外，后续会被视为无效或 vignetted。

---

### 2.4 Chief Ray 参考点语义

本阶段 `Refer To` 固定为 `ChiefRay`，并与 Zemax 保持一致。

对 polychromatic Standard Spot，Zemax 的 chief-ray reference 实际上是：

- **该视场主波长 chief ray 的像面截点**

也就是说：

1. 先获得每个视场、每个设计的主波长 chief ray 像面截点
2. 取其在像面上的 `(x, y)` 作为参考点
3. 所有波长的 spot 坐标都相对这个参考点做半径统计

当前阶段建议把支持的两种 sampler 都处理成“保证包含主光线”：

1. `hexapolar`：中心光线天然存在
2. `square`：先生成规则方阵网格；若网格中缺少 `(0, 0)`，则额外补一条主光线

这样 `SamplingResult` 可以稳定提供：

- `chief_ray_index`

后续 spot 分析就不需要再单独做一轮 chief-ray trace，而是可以直接从同一批 sampled trace 结果里提取主光线参考点。

需要明确的是：  
这种 square 语义在实现上等价于 **“square grid + chief ray”**，而不再是严格的 `N x N` 纯方阵。  
它会带来很小的采样差异，但换来更直接、更统一的实现流程。

---

## 3. 功能范围

## 3.1 支持的功能

### 采样方式

- `hexapolar`
- `square`

默认：

- `hexapolar`

### 采样密度

- 参数名建议固定为 `ray_density`
- 默认值固定为 `30`
- 最小值建议与 Zemax 一致，限制为 `>= 3`

### 波长与视场

不开放自由设置：

- 波长固定使用系统当前定义的 **全部波长**
- 视场固定使用系统当前定义的 **全部视场**
- 输出结果按“每个视场单独一组”组织

### 参考点

不开放自由设置：

- `Refer To` 固定为 `ChiefRay`

### 输出数值

输出每个视场：

- `RMS radius`
- `GEO radius`

单位建议固定为：

- 内部计算：`mm`
- 最终结果输出：`um`

这样更方便和 Zemax Standard Spot 直接比对。

### 图片保存

支持保存图片。

当 `save_path` 非空时：

1. 生成 spot diagram 图片
2. 额外保留原始散点数据，供绘图和人工检查

---

## 3.2 当前阶段不支持的功能

以下能力当前阶段都不建议开放：

1. 自定义 `wavelength` 选择
2. 自定义 `field` 筛选
3. 自定义 `Refer To`
4. 自定义 `surface`
5. through-focus spot
6. Airy disk、symbol、show scale 等 UI 风格参数
7. dithered pattern
8. 用户直接传入任意 sampler

理由很简单：  
这些能力都会扩大接口面，但对当前“先做出稳定、可对标 Zemax 的 spot radius 分析”帮助不大。

---

## 4. 推荐接口

建议沿用当前分析入口风格：

`system.analysis.spot_diagram(settings).run()`

建议把数值分析默认做成 **支持 `system_count > 1` 的批量模式**。  
也就是说，同一架构的多个设计可以一次性并行计算 spot RMS / GEO。

但“保存图片”建议仍保持保守：

1. 数值分析支持多设计 batch
2. 图片输出优先只面向单个 `design_view`
3. 如果后续确实需要批量出图，再单独扩展

---

### 4.1 推荐 settings

把现有的 `SpotDiagramSettings` 从“可传入 sampler”改成最小参数集：

```python
@dataclass(slots=True)
class SpotDiagramSettings:
    pattern: Literal["hexapolar", "square"] = "hexapolar"
    ray_density: int = 30
    save_path: str | None = None
```

说明：

- 不再暴露 `sampler`
- 采样方式通过 `pattern` 控制
- 采样密度统一通过 `ray_density` 控制
- 只有保存图片时才触发散点保留

---

### 4.2 推荐 result

建议输出结构保持精简：

```python
@dataclass(slots=True)
class SpotDiagramResult(AnalysisResult):
    rms_radius_um: ArrayLike | None = None
    geo_radius_um: ArrayLike | None = None
    field_points: tuple[tuple[float, float], ...] = ()
    figure: Any | None = None
    axes: Any | None = None
    save_path: str | None = None
    scatter_points: dict[str, Any] | None = None
```

其中：

- `rms_radius_um`：形状 `[field_count]`
- `geo_radius_um`：形状 `[field_count]`
- `scatter_points`：仅在 `save_path` 非空时填充

---

## 5. 推荐实现方案

建议新增独立实现模块：

- `optics_core/spot_diagram.py`

`analysis.py` 中只保留入口类和结果结构，不把采样、追迹、统计、绘图都塞进去。

---

### 5.1 采样器实现

#### Square

当前仓库已有 `SquarePupilSampler`，但它的接口是 `nx/ny`。

本功能不建议直接把这个接口暴露给用户。  
建议在 `spot_diagram.py` 内部增加一个构造函数，把 `ray_density` 直接映射成：

- `nx = ray_density`
- `ny = ray_density`

#### Hexapolar

当前仓库中的 `HexapolarPupilSampler` 需要按 Zemax rings 语义补齐。

推荐语义：

- `rings = ray_density`
- pupil 归一化坐标包含中心点
- 第 `k` 圈有 `6k` 条光线
- 半径按 `k / rings` 均匀分布

这样就能与 Zemax 的 hexapolar density 语义一致。

---

### 5.2 Spot 追迹主流程

推荐流程如下：

1. 根据 `pattern` 和 `ray_density` 构造 pupil 采样器，并保证 `chief_ray_index` 非空
2. 对全部视场、全部波长执行 sampled trace，shape 保持为 `design x field x wavelength x ray`
3. 从 sampled trace 中直接提取每个设计、每个视场的主波长 chief ray
4. 收集像面交点
5. 去掉无效 / vignetted rays
6. 所有有效点减去 chief reference
7. 统计每个设计、每个视场的 `RMS radius` 和 `GEO radius`
8. 若需要保存图片，则仅对单个 `design_view` 保留原始散点并绘图

---

### 5.3 RMS / GEO 统计建议

设某视场、某波长下有效 spot 点为 `(x_i, y_i)`，参考点为 `(x_ref, y_ref)`。

先定义相对坐标：

`dx_i = x_i - x_ref`

`dy_i = y_i - y_ref`

`r_i = sqrt(dx_i^2 + dy_i^2)`

#### GEO radius

建议直接定义为：

`max(r_i)`

#### RMS radius

若系统所有波长权重都相等，可直接对所有有效 ray 的 `r_i^2` 做平均再开方。

更一般地，建议保留 Zemax 语义：

1. 每个波长内部先对有效 ray 求均值
2. 再按波长权重加权平均
3. 最后开方

即：

`RMS = sqrt( sum_w weight_w * mean(r_i^2 | wave=w) / sum_w weight_w )`


---

## 6. 绘图方案

建议使用 `matplotlib`。

绘图规则保持朴素即可：

1. 每个视场一个 subplot
2. 横轴 `x (um)`，纵轴 `y (um)`
3. 不同波长使用不同颜色，按照波长顺序，默认使用蓝、绿、红、黄... 的zemax默认颜色顺序
4. 坐标相对 chief ray reference 绘制
5. 保持等比例显示
6. 标题中写出视场坐标、RMS、GEO

当前阶段用户只要求“能保存图”和“输出散点”，因此不必额外做复杂样式参数。

---

## 7. 对 `reference/DiffOptics.py` 的参考方式

`reference/DiffOptics.py` 可作为流程参考，但不建议照搬其数值定义。

当前观察到的几个特征：

1. 它的 spot 结果按全部波长拼接后统计
2. spot 的横向中心使用了整体 `mean x` 或 chief 相关位移做中心化
3. 不同函数中采样规模并不完全一致
4. 它的结果并未与 Zemax 严格对齐

因此建议只参考其“功能流程”，不要把它作为数值基准。

本项目的数值基准应始终是 Zemax Standard Spot。

---

## 8. 单元测试计划

## 8.1 测试基线

复用：

- `tests/zemax/zmx_files/Double Gauss 28 degree field.zmx`

原因：

1. 已经有成熟的 zmx 读取链路
2. 当前项目里已经用于 direct ray trace / benchmark / first-order 回归
3. 实测该文件 3 个波长的权重均为 `1.0`

---

## 8.2 Zemax 参考值获取方案

建议新增：

- `tests/zemax/spot_diagram.py`

提供一个轻量 helper，例如：

`fetch_zemax_standard_spot_metrics(zmx_path, pattern, ray_density)`

推荐实现方式：

1. 直接调用 Zemax `Standard Spot`
2. 把设置固定为：
   - 全波长
   - 全视场
   - `Refer To = ChiefRay`
   - `Pattern = square / hexapolar`
   - `RayDensity = 指定值`
3. 从 `SpotData` 直接读取：
   - `GetRMSSpotSizeFor(...)`
   - `GetGeoSpotSizeFor(...)`

这样拿到的是 **Standard Spot 原生结果**，不会再引入额外的 operand 语义差异。

---

## 8.2.1 多设计 batch 的现实边界

当前 `SequentialSurfaceRayTracer` 在数值层面已经支持：

1. 同一架构下多个设计的曲率批量变化
2. 厚度批量变化
3. 孔径半径批量变化
4. `design x field x wavelength x ray` 形状的并行追迹

但仍有一个明确边界：

**逐设计不同的材料配置，目前在 batched trace 主链里还没有完整兑现。**

原因是当前折射内核读取介质时，还是直接从共享 `system.surfaces[surface_index].gap.medium` 取值，而不是按 `system_index` 分发不同 medium。

因此，当前阶段如果要直接开始“同架构多设计并行 spot RMS 分析”，建议约束为：

1. 同一架构
2. 同一套材料
3. 设计之间只变化几何参数、厚度、孔径等数值参数

在这个前提下，直接开始 batch spot RMS / GEO 分析是可行的。

---

## 8.3 合同测试

建议先补一组轻量 contract test，验证采样器本身。

### Hexapolar 采样器

检查：

1. `ray_density = 3` 时总点数是否为 `37`
2. 是否包含中心点 `(0, 0)`
3. 所有点是否都在单位 pupil 内

### Square 采样器

检查：

1. `ray_density = 5` 时总点数是否为 `25`
2. 采样是否为规则网格
3. 坐标范围是否覆盖 `[-1, 1]`

---

## 8.4 回归测试

建议新增：

- `tests/regression/test_spot_diagram_against_zemax.py`

至少包含两组回归：

### 回归 1：默认 hexapolar

设置：

- `pattern = "hexapolar"`
- `ray_density = 30`

验证：

1. 每个视场的 `RMS radius` 与 Zemax 对齐
2. 每个视场的 `GEO radius` 与 Zemax 对齐

容差：

- `abs_tol = 1e-3 um`

### 回归 2：square

设置：

- `pattern = "square"`
- `ray_density = 30`

验证：

1. 每个视场的 `RMS radius` 与 Zemax 对齐
2. 每个视场的 `GEO radius` 与 Zemax 对齐

容差：

- `abs_tol = 1e-3 um`

---

## 8.5 图片导出测试

在默认 hexapolar 回归中同时验证保存图片：

1. 调用 `save_path`
2. 输出 PNG
3. 断言文件存在且大小大于 0
4. 断言 `result.scatter_points` 不为空

你会自己确认点图形状，因此当前阶段不做 golden image 像素对比。

---

## 8.6 测试顺序建议

尽管默认模式是 `hexapolar`，但从对齐难度看，建议实际开发顺序是：

1. **先做 square 并和 Zemax 对齐**
2. 再做 hexapolar
3. 最后再接图片输出

原因：

1. square 的采样语义最直接
2. 对 `DiffOptics.py` 与 Zemax 的差异，square 更容易排查
3. 一旦 square 与 Zemax 对齐，再收敛 hexapolar 会更稳

但对外默认值仍保持：

- `pattern = "hexapolar"`
- `ray_density = 30`

---

## 9. 最终建议

如果只给当前阶段一句建议，我会建议：

**把 Spot Diagram 当成“固定 Zemax 语义下的数值分析工具”来做，而不是做成一个高度可配置的通用绘图器。**

具体落实就是：

1. 只开放 `pattern` 和 `ray_density`
2. 默认 `hexapolar + 30`
3. 固定 `Refer To = ChiefRay`
4. 固定全波长、全视场、每个视场单独输出
5. 数值基准直接对齐 Zemax Standard Spot

这样代码最少，也最符合你当前的目标：提高可读性、提高写测效率、尽快把数值主链做实。

---

## 10. 参考资料

1. Standard Spot Diagram  
   https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v25101/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Standard_Spot_Diagram.html

2. Hexapolar Rings  
   https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Hexapolar_Rings.html

3. Why is the RMS spot size listed in Spot Diagram different than the values reported by RSCE and RSRE operands?  
   https://optics.ansys.com/hc/en-us/articles/43071095528979-Why-is-the-RMS-spot-size-listed-in-Spot-Diagram-different-than-the-values-reported-by-RSCE-and-RSRE-operands

4. 本机 ZOSAPI 实测（2026-06-06）  
   `New_StandardSpot().GetSettings()` 默认值：
   - `Pattern = Hexapolar`
   - `RayDensity = 30`
   - `ReferTo = ChiefRay`
