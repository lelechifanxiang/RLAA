# optics_core 接口定义

## 1. 设计原则

公开层采用三层模型：

1. `OpticalArchitecture`：描述共享的光学拓扑。
2. `ParameterSchema`：描述参数向量中的每个槽位如何映射到具体光设路径。
3. `MultiOpticalSystem`：组合 architecture、parameter schema、parameter vectors、fields、wavelengths、aperture、tracer 和 analysis 的公开主入口。

这套模型强调两件事：

1. 结构拓扑只定义一次，便于批量系统复用。
2. 多系统差异统一表达为参数向量列表，便于批处理、扫描、Monte Carlo 和优化器接入。

当前参数系统采用“一个 `ParameterSpec` 对应一个参数槽位”的思路。参数向量是 `list[Any]`，其中每一项可以是标量，也可以是后续模块需要的结构化 payload。具体执行层如果只支持标量，需要在读取时显式转换和报错。

### 1.1 当前开发目标

当前项目处在“建立稳定测试基线和最小可运行追迹链路”的阶段，开发重点不是一次性完成完整真实光线追迹内核，而是先把以下基础边界固定下来：

1. 共享光学拓扑和批量参数向量如何组合。
2. 单设计系统如何从批量系统中物化出来。
3. 视场、波长、孔径、采样器和追迹器之间如何传递数据。
4. 近轴面追迹如何作为最小数值闭环被测试。
5. Zemax/ZOS-API 参考值如何进入回归测试。

因此当前实现会保留一些明确的 `NotImplementedError` 入口。它们不是遗漏，而是为了在接口稳定后逐步补齐真实球面、非球面、坐标间断和更复杂介质追迹。

## 2. 模块结构

当前代码包位于 `optics_core/`，公开模块分为：

1. `system`：`MultiOpticalSystem`、builder、单设计物化视图。
2. `system_specs`：`FieldPoint`、`Wavelength`、`SystemAperture`、`ParameterSpec`、`ParameterSchema`、`ParameterVectorBatch`、`OpticalArchitecture`。
3. `surfaces`：顺序面对象、间隔介质、坐标间断、面序列容器。
4. `geometries`：几何形状定义，例如 `StandardGeometry`、`PlaneGeometry`、`EvenAsphereGeometry`、`ParaxialGeometry`。
5. `materials`：材料接口、Abbe 材料、RealMaterial（Sellmeier 1）材料、材料库。
6. `rays`：显式光线束、追迹选项、追迹结果、光线瞄准结果。
7. `sampling`：瞳孔采样器和 ray aiming 接口。
8. `tracing`：顺序追迹器抽象接口和当前的最小 surface-based 实现。
9. `analysis`：Spot、Wavefront、Zernike、PSF、MTF、Distortion、First Order、Merit 的分析类。
10. `imaging`：面向后续图像形成扩展的预留接口。

## 3. 核心对象关系

### 3.1 OpticalArchitecture

`OpticalArchitecture` 负责描述多系统共享的结构信息：

1. 面序列及其类型。
2. stop 面位置。
3. 材料库与面拓扑。

它只表达“结构长什么样”，不负责存放每个系统实例的具体参数值。

### 3.2 ParameterSpec 与 ParameterSchema

`ParameterSpec` 描述参数向量中的一个槽位：

1. `name`：逻辑名称，例如 `zoom_radius`。
2. `path`：落到共享拓扑上的参数路径，例如 `surface[1].geometry.radius`。
3. `default`：生成默认参数向量时使用的基线值。
4. `metadata`：保留给后续扩展的附加信息。

当前没有 `size` 字段，也不再支持一个 spec 占多个连续分量。`ParameterSchema.parameter_count` 等于 `ParameterSpec` 的数量。

`ParameterSchema` 负责管理整张映射表，并提供：

1. `parameter_count`：参数向量槽位数量。
2. `spec_names`：按顺序返回参数名称。
3. `index_of(name)`：查询某个参数在向量中的槽位下标。
4. `slice_of(name)`：返回单槽位切片，主要用于兼容已有调用习惯。
5. `default_vector()`：生成一组默认参数向量，返回 `list[Any]`。
6. `vector_from_mapping(...)`：从命名参数构造完整参数向量，返回 `list[Any]`。

这次合并前，参数系统曾经支持 `ParameterSpec.size`，即一个参数可以占据多个连续分量。合并后已经改成单槽位模型，因此旧代码中依赖 `spec.size` 的地方都应改为使用 `schema.index_of(spec.name)` 读取参数槽位。

对于执行层，这个变化有两点影响：

1. 标量参数，例如焦距、厚度、曲率半径，可以直接读取并 `float(...)` 转换。
2. 非标量参数，例如材料对象、系数列表、tensor payload，不应由通用标量读取函数强行转换；后续需要按具体路径单独处理。

### 3.3 ParameterVectorBatch 与 helper

`ParameterVectorBatch` 保存一组共享映射表下的参数向量，核心属性包括：

1. `schema`
2. `vectors: list[list[Any]]`
3. `system_count`
5. `parameter_count`

构造时会把每条向量归一成 `list`，并检查向量长度等于 `schema.parameter_count`。推荐使用方式有两类：

1. 如果参数差异较少且呈规则扫描，使用 `build_parameter_vector_grid(...)` 生成 mesh-grid 风格的参数向量列表。
2. 如果每个系统都有独立参数，直接提供原始参数向量列表。

例如三结构近轴测试中，schema 有两个槽位：

1. `surface[0].geometry.focal_length`
2. `surface[0].gap.thickness`

对应的参数向量可以写成：

```python
[
    [40.0, 40.0],
    [50.0, 50.0],
    [40.0, 36.0],
]
```

此时 `system_count == 3`，追迹器会把每个结构的焦距和像距分别广播到 `[system_count, ray_count]` 的光线张量上。


### 3.4 MultiOpticalSystem

`MultiOpticalSystem` 负责聚合：

1. `architecture: OpticalArchitecture`
2. `parameter_schema: ParameterSchema`
3. `parameters: ParameterVectorBatch`
4. `fields: FieldSequence`
5. `wavelengths: WavelengthSequence`
6. `aperture: SystemAperture | None`
7. `materials: MaterialLibrary`
8. `tracer: SequentialSurfaceRayTracer`
9. `analysis: AnalysisHub`

调用方式：

1. `trace(...)`
2. `design_view(index)`

当前公开系统层只保留 batched `trace(...)` 主入口。显式光线追迹与反向追迹不再作为系统层公开接口；反向追迹仅作为 tracer 内部辅助函数服务入瞳定位。

`design_view(index)` 会把第 `index` 条参数向量物化成单设计系统：它会深拷贝 surface 列表，按 `ParameterSpec.path` 写入该设计的参数值，并解析材料引用。这个接口用于测试、Zemax 对标和后续单结构分析。

单系统只是多系统的退化场景。默认情况下，`MultiOpticalSystem` 会持有一条默认参数向量，因此 `system_count == 1`。

### 3.5 SurfaceSequence 与 Surface

`SurfaceSequence` 负责顺序管理光学面。

每个 `Surface` 由以下几部分组成：

1. `geometry`
2. `gap`
3. `frame`
4. `interaction`
5. `is_stop`

当前标准球面统一用 `StandardGeometry` 表示；`PlaneGeometry` 和 `EvenAsphereGeometry` 继承自标准几何语义。

### 3.6 Sampling、RayAimer、Tracer

1. `PupilSampler` 生成归一化 pupil 坐标。
2. 默认从 `first_order_data` 读取入瞳位置和半径；显式传入 `RayAimer` 时使用自定义瞄准逻辑。
3. `SequentialSurfaceRayTracer` 负责顺序追迹。

`RayAimingResult.aimed_rays` 存储所有瞄准后的采样光线，`pupil_coordinates[i]` 对应第 `i` 条光线。主光线、边缘光线不单独保存；如果采样中包含 `(0, 0)` 或边缘坐标，可由 `pupil_coordinates` 索引对应光线。若采样不包含中心点，本次结果中就没有可索引的主光线。

当前 `SequentialSurfaceRayTracer` 的能力边界：

1. `trace(system, rays, options)` 接收显式 `RayBundle`，按 surface 顺序更新交点和方向余弦。
2. `batched_trace(...)` 当前只支持 `system.fields.field_type == "angle"`，会把视场角转换为入射斜率。
3. 由 sampler 或 aimer 生成的 pupil 坐标会写入 `RayBundle.metadata["pupil_coordinates"]`，用于测试中与 Zemax 参考结果按 `(px, py)` 对齐。
4. 当前真实实现覆盖平面 pass-through、近轴面、标准球面/圆锥面正向多面折射、像面记录和反射面。
5. `SphereSurface` 走 `StandardGeometry` 解析求交；空系数 `EvenAsphereSurface` 暂时退化为同一条标准面路径。
6. 非空 `EvenAsphereGeometry.coefficients` 当前显式抛出 `NotImplementedError`，为后续非球面 Newton 求交和法线实现预留接口。
7. `CoordinateBreak` 暂不支持真实坐标变换，遇到时显式抛出 `NotImplementedError`。
8. `record_opd`、`record_ray_angles`、`warm_start` 暂不支持，真实 tracer 会显式抛出 `NotImplementedError`。
9. 参数覆盖读取适配当前 schema：球面追迹会读取 `geometry.radius`、`geometry.conic`、`semi_diameter` 和 `gap.thickness` 的多系统参数。

正向标准球面/圆锥面追迹的当前计算流程：

1. sampler 生成归一化 pupil 坐标 `(px, py)`。
2. 入口瞳半径由系统孔径定义解析：ENPD 直接给定，FLOA 由 stop 半径和前组瞳放大率换算，FNO 由 `|EFFL| / (2 * FNO)` 换算。
3. 初始光线位置为 `x = px * entrance_pupil_radius`，`y = py * entrance_pupil_radius`。
4. 若视场类型为 `angle`，视场角转换为入射斜率：`l = tan(field.x)`，`m = tan(field.y)`，`n` 由方向余弦归一化链路在追迹中持续更新。
5. 每个球面使用 `FrameData` 中对应的 frame 转入 surface-local 坐标，再调用标准面解析求交。
6. 命中后计算局部法线，按面前介质和面后 `gap.medium` 的折射率执行 Snell 折射。
7. 求交失败、孔径外和折射失败都会传播到 `TraceResult.valid=False`。
8. 输出 `TraceResult`，其中 `rays` 是最终像面光线，`intersections` 在开启 `record_intersections=True` 时记录各面交点。


## 5. 一条典型调用路径

```python
import optics_core as oc

architecture = oc.OpticalArchitecture(name="zoom_lens")
architecture.materials.register(oc.AbbeModelMaterial(name="BK7", nd=1.5168, vd=64.17))

architecture.surfaces.add_object()
architecture.surfaces.add_sphere(radius=22.0, thickness=3.5, medium="BK7", is_stop=True)
architecture.surfaces.add_even_asphere(
    radius=-30.0,
    thickness=1.2,
    medium="AIR",
    coefficients=(1e-5, 2e-7),
)
architecture.surfaces.add_image()

parameter_schema = oc.ParameterSchema(
    [
        oc.ParameterSpec(name="focus_distance", path="surface[0].gap.thickness", default=1000.0),
        oc.ParameterSpec(name="zoom_radius", path="surface[1].geometry.radius", default=22.0),
    ]
)
parameters = oc.build_parameter_vector_grid(
    parameter_schema,
    [
        oc.ParameterSweepAxis(parameter="zoom_radius", values=[22.0, 28.0, 35.0]),
        oc.ParameterSweepAxis(parameter="focus_distance", values=[1000.0, 500.0, 250.0]),
    ],
)

systems = oc.MultiOpticalSystem(
    architecture=architecture,
    name="demo",
    parameter_schema=parameter_schema,
    parameters=parameters,
)
systems.fields.set_type("angle")
systems.fields.add(y=0.0)
systems.fields.add(y=14.0)
systems.wavelengths.add(0.4861)
systems.wavelengths.add(0.5876, is_primary=True)
systems.wavelengths.add(0.6563)
systems.set_aperture("entrance_pupil_diameter", 10.0)
```

如果只需要单系统，可以直接省略 `parameters`，或者传入仅包含一条向量的 `ParameterVectorBatch`。

## 6. 当前测试基线

当前测试按三类组织：

1. `tests/contract`：公开 API、几何对象和基础契约。
2. `tests/regression`：端到端链路和数值行为基线。
3. `tests/zemax`：Zemax/ZOS-API 对标 helper 和临时参考结构。

新增或合并后的重点测试包括：

1. `test_basic_paraxial_focus.py`：用 `SequentialSurfaceRayTracer` 跑单近轴面最小流程，并与 Zemax 参考结果比较。
2. `test_multifield_paraxial_trace.py`：构造多视场、多结构参数批次的近轴系统；用 `FieldSequence(field_type="angle")` 定义边缘视场；用 `SquarePupilSampler(nx=3, ny=3)` 生成 pupil 光线；按 `(px, py)` 对齐 `actual` 与 Zemax `reference`。
3. `test_multispectral_sphere_material.py`：验证 `StandardGeometry`、材料折射率和 `design_view()` 的单设计物化能力。

Zemax 对标测试需要当前环境安装 `zospy`，且 OpticStudio 能通过注册表或 ZOS-API helper 被发现。若本机缺少 OpticStudio 注册表项，相关测试会在 `zp.ZOS()` 初始化阶段失败。

### 6.1 多视场近轴测试的数据流

`test_multifield_paraxial_trace.py` 的测试目的不是临时写一个测试追迹器，而是验证 `optics_core` 公开对象能完整表达这个场景：

1. 用 `OpticalArchitecture` 构建近轴面加像面的共享拓扑。
2. 用 `ParameterSchema + ParameterVectorBatch` 描述多个焦距/像距结构。
3. 用 `FieldSequence(field_type="angle")` 记录边缘视场角，例如 x 方向 20 度、y 方向 30 度。
4. 调用 `MultiOpticalSystem.trace(Hx=..., Hy=...)`，其中 `Hx/Hy` 是归一化视场坐标。
5. `SequentialSurfaceRayTracer` 从系统视场中取边缘角，把归一化视场转换成实际入射斜率。
6. `TraceResult` 被整理成 `ParaxialTraceReference`，再与 Zemax 参考值比较。

结果对齐使用 `(px, py)`，而不是近轴面上的 `(x, y)` 大小排序。原因是 `(px, py)` 是 sampler 生成光线时的身份信息，更适合作为 actual/reference 的对应关系；如果以后真实曲面追迹导致近轴面坐标不完全一致，也不会破坏光线匹配逻辑。

### 6.2 Zemax helper 的角色

`tests/zemax` 中的临时结构体和 helper 只服务于测试参考值：

1. `ParaxialFocusCaseSpec` 描述 Zemax 侧需要构建的近轴系统规格。
2. `ParaxialTraceReference` 是 actual/reference 共用的扁平比较结构。
3. `fetch_zemax_paraxial_trace_mu_par(...)` 负责创建 Zemax 顺序系统、设置角度视场、逐条 pupil 光线运行 `SingleRayTrace`，再返回参考数据。

这些结构目前放在 `tests/zemax`，还不是正式公开 API。后续如果 Zemax 对标场景增多，可以再把公共配置逻辑继续收敛到 `tests/zemax/common.py`。

## 7. 当前代码状态

当前阶段重点是冻结公开 API，并建立三类测试基线：

1. `OpticalArchitecture`、`ParameterSchema`、`ParameterVectorBatch`、`MultiOpticalSystem` 的对象关系已经固定。
2. 参数批次模型已经从旧的 flattened tuple/size 模型切换为 `list[Any]` 单槽位模型。
3. `MultiOpticalSystem.design_view()` 支持将批量系统中的单个结构物化，方便材料、Zemax 对标和后续单结构分析。
4. `optics_core/first_order.py` 已经承担读参数读取、stop 查询和 surface 轴向位置这类只读查询能力，供 aimer、tracer 与后续分析模块复用。
5. 默认采样追迹直接复用 `first_order_data` 中缓存的入瞳位置和半径，不再经过额外的默认 aimer。
6. `SequentialSurfaceRayTracer` 现在按 surface class 分发局部追迹，并把 plane hit、sag hit 以及通用折射/反射内核收敛到共享实现里；SphereSurface 与 EvenAsphereSurface 先复用同一条真实面主链，ParaxialSurface 与 CoordinateBreak 保持独立 kernel。
7. GPU 并行 tracing、更完整的坐标间断、以及更高精度的真实非球面内核，仍由后续实现层补齐。

这意味着当前公开层已经把“共享拓扑”“参数批次”“顺序追迹接口”和“测试参考链路”稳定分离，同时没有把完整物理追迹内核提前写死进高层 API。

## 8. 后续实现建议

后续扩展建议按风险从低到高推进：

1. 先清理 `tests/zemax/paraxial_focus.py` 中合并后残留的重复 import 和旧 helper，保证 Zemax helper 层风格一致。
2. 继续扩展 `optics_core/first_order.py`，把当前只读查询层向更多 first-order 分析共用能力收敛，避免再把参数读取 helper 放回 tracer。
3. 在 `TraceResult.cache` 或 `RayBundle.metadata` 中保留更多追迹上下文，例如 field、wavelength、sampler 信息，方便测试和分析模块复用。
4. 继续完善 `EvenAsphereGeometry` 的专用 sag/normal/intersect 数值实现，再补充与 Zemax 的真实球面/非球面/多波长端到端对标。
5. 在真实曲面追迹稳定后，再考虑 GPU batched kernel 和更高阶的 ray aiming。
