# DiffOptics 球面正向追迹迁移计划

## 目标

本阶段目标不是复刻旧版 `DiffOptics.py` 的内部行为，而是在 `optics_core` 当前架构中实现可验证的正向多球面追迹能力，并直接与 Zemax 结果对齐。

迁移范围限定为：

1. 正向顺序追迹，不迁移反向追迹。
2. 多个标准球面/圆锥面连续追迹。
3. 玻璃材料采用阿贝模型，不迁移旧版真实玻璃库和材料表。
4. 光线、几何和折射计算保持 FP64 tensor。
5. 架构上为后续非球面求交和法线计算预留接口。

本阶段不包含：

1. 旧版 DiffOptics 中间结果对齐。
2. 反向追迹、入瞳反求和 backward probe。
3. 非球面 Newton 迭代真实落地。
4. OPD、ray angle 统计、warm start、优化器状态迁移。
5. 坐标断点和倾斜偏心面。

## 旧代码参考位置

旧版代码只作为算法参考，不作为数值验收基准。

`../DiffOptics.py` 中与追迹主流程相关的位置：

1. `../DiffOptics.py:3665`：`_refract(wi, n, eta)`，Snell 折射公式。
2. `../DiffOptics.py:3678`：`_trace(...)`，正向/反向追迹分发。
3. `../DiffOptics.py:3692`：`_forward_tracing(...)`，正向逐面追迹主循环。

`../SurfaceClass.py` 中与球面几何相关的位置：

1. `../SurfaceClass.py:57`：`ray_surface_intersection(ray)`，求交后叠加孔径有效性。
2. `../SurfaceClass.py:90`：`newtons_method_new(...)`，球面/圆锥面解析求交和非球面 Newton 入口。
3. `../SurfaceClass.py:31`：`normal(x, y)`，由曲面导数计算单位法线。
4. `../SurfaceClass.py:207`：`Aspheric`，旧版同时承载球面、圆锥面和偶次非球面表达。

## 新项目保留的设计

迁移前不需要把新项目统一成旧版形式。旧版代码只提供公式线索，新项目继续保留当前公开结构：

### 数据结构

继续使用 `RayBundle.x/y/z/l/m/n`，不恢复旧版 `ray.o`、`ray.d` 的 `(3, ...)` 堆叠结构。

原因：

1. 当前结构更适合 batch-first 的公开接口。
2. 单分量 tensor 更容易保持 shape、device 和 dtype 的显式约定。
3. 后续分析模块可以直接读取像面 `x/y` 和方向余弦。

旧版 `ray.o`、`ray.d` 只在阅读旧公式或临时对照时用 `torch.stack` 理解，不进入正式接口。

### 面位置

继续使用 `surface.gap.thickness` 累加得到 surface vertex z，不恢复旧版每个 surface 自带绝对 `d` 的形式。

实现要求：

1. `_sag_surface_hit()` 通过 `surface_position(system, surface_index)` 获取当前面顶点 z。
2. 求交前将全局 ray 坐标转换到当前面的局部坐标：`local_z = z - surface_z`。
3. `geometry.intersect()` 只处理局部面形，不读取 system、gap 或材料。

### 半径和曲率

继续使用 `radius` 表达曲率半径，不恢复旧版 `c = 1 / R` 的曲率参数。

实现要求：

1. 从 Zemax 或旧公式迁入时明确使用 `radius = 1 / curvature`。
2. `radius = 0` 继续表示平面退化，不沿用旧版 `abs(R) > 1e9` 近似平面逻辑。
3. 正负半径、近轴光线、边缘孔径光线必须分别与 Zemax 对齐。

## 推荐架构

继续增强当前 tracing 模块，不新增一套平行 tracer。

1. `optics_core/tracing/_core.py`
   保留 `SequentialSurfaceRayTracer.trace()` 和 `_trace_surfaces()` 作为正向主入口。第一阶段只保证 `TraceOptions.direction="forward"`。

2. `optics_core/tracing/_dispatch.py`
   `_trace_surface_local()` 继续按 surface 类型分发。`SphereSurface` 和无非球面系数的 `EvenAsphereSurface` 进入同一条球面/圆锥面路径。

3. `optics_core/tracing/_hits.py`
   `_sag_surface_hit()` 负责局部坐标转换、几何求交、孔径裁剪和交点快照。

4. `optics_core/geometries.py`
   `StandardGeometry` 承载球面/圆锥面的 `sag()`、`normal()`、`intersect()`。后续非球面应扩展 `EvenAsphereGeometry`，不要把非球面逻辑塞进 tracer 主循环。

5. `optics_core/tracing/_interactions.py`
   负责折射率读取、法线方向选择和 Snell 折射。阿贝模型玻璃应通过 `Material.refractive_index(wavelength_um)` 提供折射率。

## 为非球面预留的接口

本阶段不实现非球面 Newton 求交，但需要避免把架构写死成“只能解析球面”。

建议接口边界： 

1. `BaseGeometry.intersect(ray_origin, ray_direction)` 保留为几何求交统一入口。
2. `StandardGeometry.intersect()` 使用解析球面/圆锥面公式。
3. `EvenAsphereGeometry.intersect()` 后续单独实现 Newton 或其他迭代求交。
4. `EvenAsphereGeometry.normal()` 后续单独实现偶次非球面导数。
5. `_trace_sag_surface()` 不关心具体几何类型，只调用 geometry 的 `intersect()` 和 `normal()`。

短期保护：

1. 如果 `EvenAsphereSurface.coefficients` 非空，应显式 `NotImplementedError` 或在构造阶段 fail fast。
2. 系数为空时可以退化为 `StandardGeometry` 行为。
3. 测试中必须覆盖“非空非球面系数当前不支持”，避免静默按球面计算。

## Zemax 对标策略

Zemax 是本阶段唯一数值验收基准。

### 参考数据获取

在 `tests/zemax` 中新增球面正向追迹 helper，集中负责：

1. 创建顺序系统。
2. 设置多波长。
3. 设置阿贝模型玻璃或与阿贝模型等价的玻璃参数。
4. 设置多个球面曲率、厚度、孔径。
5. 直接获取各面交点、方向余弦和折射率。

不要通过最终像点反算中间面数据。能直接从 Zemax ray trace data 或 operand 获取的值，应优先直接获取。

### 测试系统

建议最少准备三组系统：

1. 单球面折射
   空气到阿贝玻璃，覆盖轴上光线、离轴光线、边缘孔径光线。

2. 双球面厚透镜
   空气 -> 玻璃 -> 空气，覆盖正负半径组合和像面截距。

3. 多球面系统
   至少 4 个折射球面，包含两种阿贝玻璃和多个波长，验证连续折射、材料顺序和 batch 传播。

### 关键比较值

每个测试至少比较：

1. 每个球面的交点 `x/y/z`。
2. 每个球面的入射或出射方向余弦 `l/m/n`。
3. 每个波长下的介质折射率。
4. 最终像面坐标。
5. `valid` 与 Zemax 是否截光/全反射的判断一致。

测试中需要打印关键验证数据，包括系统曲率半径、厚度、波长、玻璃参数、Zemax 参考值和 `optics_core` 计算值。

## 实施阶段

### 阶段 1：收窄公开行为

1. 明确 `SequentialSurfaceRayTracer` 当前阶段只承诺正向追迹。
2. 对 `TraceOptions.direction != "forward"` 的路径暂时 fail fast，或至少不纳入本阶段验收。
3. 对非空 `EvenAsphereGeometry.coefficients` 显式报错。
4. 对 `record_opd`、`record_ray_angles`、`warm_start` 等未实现选项给出明确行为。

交付物：

1. 公开接口说明更新。
2. contract 测试覆盖不支持功能的 fail-fast 行为。

### 阶段 2：Zemax 参考生成

1. 新增 `tests/zemax/spherical_forward_trace.py`。
2. 实现单球面、双球面、多球面系统的 Zemax 构建函数。
3. 统一阿贝模型玻璃参数生成和 Zemax 写入逻辑。
4. 输出每个 surface 的交点、方向余弦、折射率和有效性参考。

交付物：

1. Zemax helper。
2. 参考数据结构 dataclass。
3. 可打印关键数据的 regression 测试骨架。

### 阶段 3：球面几何 kernel 对齐

1. 校验 `StandardGeometry.sag()` 与 Zemax sag 一致。
2. 校验 `StandardGeometry.normal()` 的方向约定与 Zemax 追迹结果一致。
3. 校验 `StandardGeometry.intersect()` 的根选择，覆盖正半径、负半径和近似平面。
4. 保持所有输入输出为 FP64 tensor，不增加 scalar/list/tuple 兼容。

交付物：

1. `tests/contract/test_geometry_contracts.py` 增加球面/圆锥面几何测试。
2. 必要时调整 `optics_core/geometries.py`。

### 阶段 4：正向多球面追迹

1. 在 `_sag_surface_hit()` 中补齐孔径裁剪。
2. 确保求交失败、孔径外、折射失败都会进入 `TraceResult.valid`。
3. 在 `_apply_refractive_interaction()` 中对齐 Zemax 的折射方向和全反射有效性。
4. 确认材料顺序为“面前介质 -> 面后 gap 介质”。
5. 跑单球面、双球面、多球面正向端到端测试。

交付物：

1. `tests/regression/test_spherical_forward_trace_against_zemax.py`。
2. `tests/contract/test_surface_trace_kernels.py` 增加孔径、折射失败和多面传播用例。

### 阶段 5：batch 和性能收口

1. 保持 surface 循环在 Python 层，ray、field、wavelength、system 维度全部 tensor 化。
2. 所有临时 tensor 使用输入 ray 的 device，避免 CPU-GPU 搬运。
3. 多系统参数读取继续走 `surface_value()`/`surface_position()`，后续再收敛到 `TraceRuntime`。
4. 大 batch 下默认减少交点快照，必要时提供显式开启策略。

交付物：

1. 多 field、多 wavelength、多 system 的 batch regression。
2. CUDA 可用时的 smoke 测试。
3. benchmark 中加入多球面正向追迹场景。

## 注意事项

1. 旧版代码只作为公式参考，不作为测试 oracle。
2. 不要迁移旧版 `Ray`、`Surfaces`、`Aspheric` 类本体。
3. 不要把旧版绝对 `d`、曲率 `c`、`ray.o/d` 改成新项目公开接口。
4. 几何模块只处理局部面形；材料、面位置和系统拓扑留在 tracing/system 层。
5. 阿贝模型玻璃必须用 Zemax 可复现的参数表达，测试里打印 `nd`、`vd` 和各波长折射率。
6. 孔径裁剪要进入 `TraceResult.valid`，不能只靠交点 NaN 暗示。
7. 全反射策略必须与 Zemax 对齐；如果选择将全反射作为 invalid，需要写入接口文档。
8. 波长单位保持 `um`，所有 Zemax 返回值进入断言前先统一单位。
9. 非空非球面系数短期必须 fail fast，避免静默按球面追迹。
10. 验证优先取 Zemax 直接输出值，避免从最终像面结果反算中间量。

## 验收标准

1. 单球面系统的交点、方向余弦、最终像面坐标与 Zemax 对齐。
2. 双球面厚透镜的两面交点和最终像面截距与 Zemax 对齐。
3. 多球面系统在多波长、两种阿贝玻璃下与 Zemax 对齐。
4. 正负半径根选择稳定，无错误命中远端根。
5. 孔径外光线 `valid=False`，且不会污染后续面追迹。
6. 非空 `EvenAsphereGeometry.coefficients` 明确报错。
7. 所有新增数值测试打印关键验证数据。
8. `python -m pytest` 通过；Zemax 环境不可用时，Zemax 测试按既有标记跳过或单独运行。

## 最小实施顺序

1. 先收窄公开行为：正向追迹、非球面 fail fast、未实现选项明确化。
2. 再新增 Zemax 正向球面参考 helper。
3. 再对齐 `StandardGeometry` 的 sag、normal、intersect。
4. 再补 `_sag_surface_hit()` 的孔径裁剪和 valid 传播。
5. 再对齐 `_apply_refractive_interaction()` 的阿贝玻璃折射。
6. 最后补多球面、多波长、多系统 regression 和 benchmark。
