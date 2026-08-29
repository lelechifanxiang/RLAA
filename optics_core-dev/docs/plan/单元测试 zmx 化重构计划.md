# 单元测试 zmx 化重构计划

## 目标

将当前以“测试规格参数 + zospy 构建 Zemax 参考系统 + optics_core 构建本地系统”的系统级测试，逐步重构为：

1. 读取 `.zmx` 文件，获取系统设置、视场、波长、表面和材料信息
2. 在 `optics_core` 中根据 `.zmx` 构建同样的系统
3. 用 `zospy` 只负责读取参考数值，不再重复手工搭建 Zemax 系统
4. 对比 `optics_core` 结果和 Zemax 结果

这样做的核心收益：

- 减少重复的 Zemax 搭建代码
- 让系统规格以 `.zmx` 文件为唯一来源，便于人工调系统
- 让测试更接近真实设计输入
- 为后续更多真实镜头 case 复用同一条测试路径


## 结论

我认同将“系统级、Zemax 对标型”测试逐步迁移到 `.zmx` 读取路径。

但不建议把所有测试都改成 `.zmx` 读取。更合适的边界是：

- **应迁移**：
  依赖 Zemax 参考数值、关注真实光学系统行为的测试
- **应保留现状**：
  几何 kernel、公开接口契约、内部算法、异常路径、smoke tracer 这类纯单元测试

换句话说，`.zmx` 路径更适合做“系统级真值驱动测试”，不适合替代所有底层单元测试。


## 现有测试规格盘点

### 1. 近轴单透镜规格

规格来源：

- [ParaxialFocusCaseSpec](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/zemax/temp_structures.py:8>)

主要字段：

- `focal_length_mm`
- `aperture_radius_mm`
- `wavelength_um`
- `field_hx/field_hy`
- `edge_field_x_deg/edge_field_y_deg`
- `pupil_grid_shape`
- `image_plane_distance_mm`

使用位置：

- [test_basic_paraxial_focus.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/regression/test_basic_paraxial_focus.py:98>)
- [test_multifield_paraxial_trace.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/regression/test_multifield_paraxial_trace.py:107>)
- Zemax helper: [tests/zemax/paraxial_focus.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/zemax/paraxial_focus.py:35>)

当前特点：

- Zemax 系统由 helper 直接创建
- 本地系统由 `build_basic_paraxial_focus_system()` / `build_multifield_multistructure_system()` 创建

迁移判断：

- **适合迁移**
- 可准备 `paraxial_single_lens.zmx`
- 多视场、多结构可拆成多个 `.zmx` 或保留“一个基础 `.zmx` + 本地参数扰动”的混合模式


### 2. 单球面多材料规格

规格来源：

- [MultispectralSphereMaterialCaseSpec](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/zemax/temp_structures.py:40>)
- [BASE_MULTISPECTRAL_SPHERE_SPEC](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/fixtures/cases.py:53>)
- [DECLARED_MULTISPECTRAL_SYSTEM_CASES](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/fixtures/cases.py:31>)

主要使用位置：

- [test_multispectral_sphere_material.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/regression/test_multispectral_sphere_material.py:77>)
- Zemax helper: [tests/zemax/multispectral_sphere_material.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/zemax/multispectral_sphere_material.py:18>)

当前特点：

- 主要验证：
  - 球面矢高
  - 阿贝模型折射率
  - `design_view` 参数物化
- Zemax 参考值来自 helper 动态搭建

迁移判断：

- **适合部分迁移**
- 建议准备 `single_sphere_material_reference.zmx`
- 但 `design_view` / 参数 schema / 多系统参数物化 仍然是本地结构测试，不建议强行 zmx 化


### 3. 四面球面正向追迹规格

规格来源：

- [SphericalForwardTraceCaseSpec](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/zemax/temp_structures.py:76>)
- [DEFAULT_FORWARD_MULTI_SPHERE_CASE](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/fixtures/cases.py:56>)
- [DEFAULT_ZEMAX_SPHERICAL_FORWARD_CASE](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/fixtures/cases.py:91>)
- [DEFAULT_ZEMAX_EXTREME_PUPIL_CASE](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/fixtures/cases.py:128>)

主要使用位置：

- [test_spherical_forward_trace_against_zemax.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/regression/test_spherical_forward_trace_against_zemax.py:35>)
- [test_spherical_extreme_pupil_trace_against_zemax.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/regression/test_spherical_extreme_pupil_trace_against_zemax.py:19>)
- [test_clear_aperture_against_zemax.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/regression/test_clear_aperture_against_zemax.py:18>)
- Zemax helpers:
  - [tests/zemax/spherical_forward_trace.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/zemax/spherical_forward_trace.py:12>)
  - [tests/zemax/builders.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/zemax/builders.py:45>)

当前特点：

- 是当前最完整的 Zemax 对标链路
- 同时覆盖：
  - 多视场
  - 多波长
  - 多 pupil 采样
  - 极限 pupil
  - clear aperture

迁移判断：

- **强烈建议迁移**
- 应优先把这套 case 替换为 `4_surface_spherical_reference.zmx`
- 这是最适合成为“标准 zmx 对标模板”的一类测试


### 4. Double Gauss zmx 规格

规格来源：

- [Double Gauss 28 degree field.zmx](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/zemax/zmx_files/Double%20Gauss%2028%20degree%20field.zmx>)

主要使用位置：

- [test_double_gauss_zmx_batch_ray_trace_benchmark.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/benchmark/test_double_gauss_zmx_batch_ray_trace_benchmark.py:27>)
- [test_spherical_forward_trace_against_zemax.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/regression/test_spherical_forward_trace_against_zemax.py:126>)
- 读取/构建 helper:
  - [tests/zemax/zmx_loader.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/zemax/zmx_loader.py:52>)
  - [tests/zemax/batch_ray_trace.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/zemax/batch_ray_trace.py:39>)

当前特点：

- 已经走通了“从 `.zmx` 读取系统 -> 本地构建 -> direct rays 双端对比”的完整路径
- 目前是 zmx 化测试的样板

迁移判断：

- **已经是目标形态**
- 后续应作为其它 zmx 化测试的参考模板


### 5. 纯本地/非 Zemax 规格

主要使用位置：

- [test_forward_multi_sphere_trace.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/regression/test_forward_multi_sphere_trace.py:149>)
- [test_backward_trace_and_entrance_pupil.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/regression/test_backward_trace_and_entrance_pupil.py:14>)
- [test_surface_trace_kernels.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/contract/test_surface_trace_kernels.py:18>)
- [test_geometry_contracts.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/contract/test_geometry_contracts.py:9>)
- [test_public_api_contracts.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/contract/test_public_api_contracts.py:10>)
- [test_demo_pipeline_regression.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/regression/test_demo_pipeline_regression.py:10>)
- [test_demo_pipeline_benchmark.py](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/benchmark/test_demo_pipeline_benchmark.py:9>)

迁移判断：

- **不建议迁移**

原因：

- 它们验证的是内部行为、几何 kernel、接口契约、mock tracer 流程或独立参考实现
- `.zmx` 文件不会让这些测试更清楚，反而会增加阅读成本和外部依赖


## 重构分类建议

### A 类：优先改造为 zmx 驱动

这些测试最值得迁移：

1. `test_basic_paraxial_focus.py`
2. `test_multifield_paraxial_trace.py`
3. `test_multispectral_sphere_material.py`
4. `test_spherical_forward_trace_against_zemax.py`
5. `test_spherical_extreme_pupil_trace_against_zemax.py`
6. `test_clear_aperture_against_zemax.py`

共同特点：

- 已经依赖 Zemax
- 本地和 Zemax 两边都在重复搭系统
- 迁到 zmx 后收益明显


### B 类：保持本地构造，但可引入 zmx 对照补充

1. `test_forward_multi_sphere_trace.py`
2. `test_backward_trace_and_entrance_pupil.py`

建议：

- 保留现有测试作为内部算法/独立参考测试
- 未来可新增额外 zmx case 做系统级补充，但不要替换掉现有测试


### C 类：明确不做 zmx 化

1. `test_surface_trace_kernels.py`
2. `test_geometry_contracts.py`
3. `test_public_api_contracts.py`
4. `test_demo_pipeline_regression.py`
5. `test_demo_pipeline_benchmark.py`

原因：

- 它们不是 Zemax 真值测试
- 关注点是契约、异常、mock tracer、轻量 smoke
- zmx 化不会减少复杂度


## 推荐的重构目标形态

后续系统级测试统一遵循下面这条路径：

1. `.zmx` 文件保存系统规格
2. `load_zmx_sequential_system_spec(...)` 读取：
   - 表面
   - 材料
   - 视场
   - 波长
   - 孔径
   - stop
3. `build_optics_core_system_from_zmx_spec(...)` 构建本地系统
4. `zospy` helper 只负责读取参考结果：
   - 单光线追迹
   - BatchRayTrace
   - SemiDiameter
   - GetIndex
   - Merit operands
5. 回归测试只负责：
   - 选取采样
   - 调用两端
   - 对比结果

这样一来，测试文件本身会更短，Zemax helper 也会更像“读数模块”，不再承担过多系统构建逻辑。


## 推荐目录演进

建议沿着现有结构轻量演进，不做大重写：

- `tests/zemax/zmx_files/`
  - 存放参考设计文件
- `tests/zemax/zmx_loader.py`
  - 负责从 `.zmx` 读取系统规格
- `tests/zemax/readers.py`
  - 负责读取 Zemax 数据
- `tests/zemax/batch_ray_trace.py`
  - 负责 direct rays / BatchRayTrace
- `tests/zemax/*.py`
  - 保留按功能分文件的读取 helper

建议新增的 `.zmx` 文件族：

1. `paraxial_single_lens.zmx`
2. `single_sphere_material_reference.zmx`
3. `four_surface_spherical_reference.zmx`
4. `Double Gauss 28 degree field.zmx` 已存在


## 分阶段实施计划

### 第一阶段：固定基础设施

目标：

- 以 `Double Gauss` 路径为模板，稳定 `zmx_loader + readers + batch_ray_trace`

动作：

1. 保持 `.zmx -> spec -> optics_core system` 作为标准入口
2. 明确当前支持边界：
   - `Standard` surface
   - `Angle` fields
   - `EntrancePupilDiameter`
   - Zemax model glass / 已加载折射率表
3. 对新 helper 增补最少量契约测试


### 第二阶段：迁移四面球面 Zemax 对标链路

目标：

- 用 `four_surface_spherical_reference.zmx` 取代 `DEFAULT_ZEMAX_SPHERICAL_FORWARD_CASE`

动作：

1. 新建四面球面 `.zmx`
2. 将以下测试迁到 zmx 驱动：
   - `test_spherical_forward_trace_against_zemax.py`
   - `test_spherical_extreme_pupil_trace_against_zemax.py`
   - `test_clear_aperture_against_zemax.py`
3. 逐步减少 `SphericalForwardTraceCaseSpec` 在 Zemax 对标场景中的使用


### 第三阶段：迁移近轴系统

目标：

- 用 `.zmx` 驱动近轴 Zemax 对标测试

动作：

1. 新建 `paraxial_single_lens.zmx`
2. 将：
   - `test_basic_paraxial_focus.py`
   - `test_multifield_paraxial_trace.py`
   迁到 zmx 驱动
3. 保留多结构参数扫描部分作为本地测试，不强制用 zmx 替代


### 第四阶段：迁移单球面多材料对标

目标：

- 让矢高和材料折射率对标也改成 zmx 读取系统

动作：

1. 新建 `single_sphere_material_reference.zmx`
2. 将 `test_multispectral_sphere_material.py` 中的 Zemax 构建逻辑迁到 zmx 驱动
3. 保留 `design_view`、parameter schema、多系统参数物化测试为本地构造


### 第五阶段：清理旧的“同规格双构建”路径

目标：

- 只保留必要的手工 spec/helper

动作：

1. 清理只剩 Zemax 对标用途的旧 spec
2. 保留纯本地测试仍然需要的 fixture
3. 在文档中明确：
   - 哪些测试是 zmx 驱动
   - 哪些测试是本地构造


## 实施原则

### 原则 1：不是所有测试都 zmx 化

只迁移“系统级 Zemax 对标测试”。


### 原则 2：`.zmx` 是系统规格源，不是断言逻辑源

`.zmx` 负责描述系统；断言逻辑仍留在 pytest 里，避免把测试含义埋进外部文件。


### 原则 3：材料优先保留 Zemax 实际折射率表

对于 `.zmx` 导入路径，优先像 [LoadedZemaxMaterial](</c:/Users/Huwei/Project/auto_od_web/worker/auto_od/GYOptics/optics_core/tests/zemax/zmx_loader.py:14>) 这样保留 Zemax 当前系统波长上的实际折射率，避免再次压缩成近似模型后引入误差。


### 原则 4：测试文件保持短，helper 承担读取细节

理想状态下，一个系统级回归测试只需要：

1. 读取 `.zmx`
2. 选采样
3. 跑两端
4. 对比


## 建议的近期执行顺序

建议按下面顺序推进，收益最高：

1. 四面球面 Zemax 对标链路 zmx 化
2. 近轴单透镜和多视场近轴链路 zmx 化
3. 单球面多材料链路 zmx 化
4. 最后再清理旧 spec/helper

这样能先把重复最多、收益最大的那部分拿下。

