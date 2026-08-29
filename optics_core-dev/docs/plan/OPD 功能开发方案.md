# OPD 功能开发方案

## 目标

补齐 OPD 能力，为后续惠更斯 PSF 和波前分析提供稳定的数据基础。

核心原则：

1. 追迹层只累计绝对光程 `OPL`
2. OPD 参考语义放到独立后处理模块中实现
3. 首版只服务正向序列追迹、PSF 和波前基础能力
4. 不新增复杂框架，不提前实现通用波前/Zernike 分析
5. 所有数值计算保持 FP64 tensor，避免 CPU/GPU 频繁搬运

这里要特别区分两个概念：

- `OPL`：光线从起点到当前面的绝对 optical path length，单位 mm
- `OPD`：相对某条参考光线或某个参考波面的 optical path difference，单位 mm

追迹层建议使用 `RayBundle.opl` 保存 OPL。最终用于 PSF/波前的 OPD 由 `optics_core/opd.py` 后处理得到。

## 当前状态

当前代码中已经具备部分入口：

1. `RayBundle.opl`
2. `TraceOptions.record_opd`
3. `PSFResult.opd`
4. `WavefrontResult.opd`

当前 `TraceOptions.record_opd` 默认开启，顺序追迹过程会累计 OPL。若某些性能路径暂时不需要 OPL，可显式设置 `record_opd=False`。

## 总体架构

推荐拆成两层：

### 1. 追迹层：累计 OPL

修改 `optics_core/tracing`，在逐面追迹过程中累计：

```text
OPL += n_medium(wavelength) * geometric_distance
```

其中：

1. `geometric_distance` 使用当前光线位置到当前 surface 交点的三维欧氏距离
2. `n_medium` 使用光线到达当前 surface 前所在介质的折射率
3. 累计结果写回 `TraceResult.rays.opl`
4. 单位固定为 mm

追迹层不关心 chief ray、出瞳、reference sphere、像面切平面等分析语义。

### 2. OPD 后处理层：转换参考语义

新增 `optics_core/opd.py`，只放和 OPD 参考有关的 helper。

首版建议实现两类 OPD：

1. `chief-ray referenced image OPL difference`
   - 同一视场、同一波长内，以 chief ray 的像面 OPL 为零点
   - 适合最小回归和排查

2. `exit-pupil referenced OPD`
   - 面向惠更斯 PSF
   - 读取 `system.first_order_data.exit_pupil_z/radius`
   - 基于出瞳参考补偿，把像面 OPL 差转换成衍射计算需要的 OPD

不要在首版实现完整 `WavefrontMap.run()`。波前功能后续可以复用 `opd.py`，但不应反过来拖慢 OPD 开发。

## 追迹层实现方案

### 修改点

主要修改：

1. `optics_core/tracing/_core.py`
   - 接通 `record_opd` 对 OPL 累计的控制
   - 在 `_trace_surfaces()` 中初始化并传递 `opl`
   - 每个 surface 追迹后累计本段 OPL

2. `optics_core/tracing/_dispatch.py`
   - `_trace_surface_with_frame()` 返回交点后，需要让上层能拿到本段几何距离
   - 最小实现可以不改返回结构，在 `_core.py` 中用追迹前后的全局 position 计算距离

3. `optics_core/tracing/_interactions.py`
   - 复用或暴露当前 surface 的 incident medium 解析逻辑
   - 建议新增一个简短 helper，例如 `_incident_medium(system, surface_index, direction)`

### 累计流程

在 `_trace_surfaces()` 中：

1. 初始化 `opl`
   - 默认开启 OPL 累计
   - 如果 `options.record_opd=False`，不做任何额外计算
   - 如果开启 OPL 累计且 `rays.opl is None`，初始化为 `torch.zeros_like(rays.x)`
   - 如果 `rays.opl` 已存在，作为初始 OPL 继续累计

2. 每个 surface 追迹前保存旧位置

3. 调用 `_trace_surface_with_frame(...)`

4. 追迹后计算几何距离

```python
dx = new_x - old_x
dy = new_y - old_y
dz = new_z - old_z
distance = torch.sqrt(dx * dx + dy * dy + dz * dz)
```

5. 获取当前段介质折射率

```python
medium = _incident_medium(system, surface_index, direction)
n = medium.refractive_index(wavelength_um)
```

6. 累计 OPL

```python
opl = torch.where(torch.isfinite(distance), opl + n * distance, torch.full_like(opl, torch.nan))
```

7. 构造 `final_rays` 时写入：

```python
opl=opl if options.record_opd else rays.opl
```

### 介质选择

介质选择应和折射计算一致：

- 正向追迹到 surface `i` 前：
  - `i == 0` 时介质为 `AIR`
  - 否则介质为 `system.surfaces[i - 1].gap.medium`

- 反向追迹到 surface `i` 前：
  - 介质为 `system.surfaces[i].gap.medium`

这一逻辑和 `_surface_media()` 的 incident medium 一致，建议避免复制两套实现。

### 坐标间断

CB 面不改变 OPD 语义：

1. 光线到 CB 面的几何距离照常累计
2. CB 后到下一面的几何距离照常累计
3. CB 本身是 pass surface，不引入额外相位
4. `ignore_coordinate_breaks=True` 时按忽略 CB 后的几何路径累计

首版不支持反向穿越 CB 的 OPD，因为当前反向 CB 追迹本身尚未完整支持。

## OPD 后处理方案

新增 `optics_core/opd.py`。

首版建议只保留少量函数，不新增复杂 dataclass。

### 1. chief ray 参考 OPD

输入：

1. `TraceResult`
2. `SamplingResult.chief_ray_index`
3. batch shape 元数据

输出：

1. `opd_mm`
2. `chief_opl_mm`
3. `valid`

计算方式：

```text
opd = ray_opl - chief_ray_opl
```

这个结果主要用于单元测试和早期排查，不直接作为最终 Huygens PSF 相位。

### 2. exit pupil 参考 OPD

输入：

1. `system.first_order_data.exit_pupil_z`
2. `TraceResult.rays.opd`
3. 像面交点
4. chief ray 像面交点
5. 波长

输出：

1. `opd_mm`
2. `phase_rad = 2*pi*opd_mm/(wavelength_um*1e-3)`
3. `valid`

首版建议先实现平面参考版本，服务 PSF 文档中固定的 planar Huygens 语义。

若后续要严格复刻 Zemax 的 reference sphere，可在该模块中增加 sphere correction，不要改 tracer。

## 和 PSF 的关系

惠更斯 PSF 的主链建议如下：

1. 使用固定 pupil sampler 采样
2. 正向追迹到像面，默认获得 OPL
3. 从追迹结果中取 chief ray 像面点和每根光线像面点
4. 调用 `opd.py` 计算 exit-pupil referenced OPD
5. 转换为相位 `phase_rad`
6. 执行 Huygens 积分

这样 PSF 模块不需要关心逐面介质、路径长度和 CB frame，只消费追迹后的 OPL/OPD 数据。

## 单元测试计划

### 1. contract 测试

新增或修改：

- `tests/contract/test_surface_trace_kernels.py`

验证：

1. 默认追迹会输出 OPL
2. 空气中平面传播时，`OPL == geometric distance`
3. 均匀玻璃中传播时，`OPL == n * geometric distance`
4. `record_opd=False` 时不额外修改 `rays.opl`

### 2. regression 测试：OPL 小系统

新增：

- `tests/regression/test_opl_trace.py`

建议用本项目构建极简系统，不需要 Zemax：

1. 空气中两平面传播
2. 单玻璃段传播
3. 多波长真实材料传播

这些测试主要验证追迹层 OPL 累计公式，排查成本低。

### 3. regression 测试：和 Zemax 对标

新增：

- `tests/zemax/opd.py`
- `tests/regression/test_opd_against_zemax.py`

测试文件建议优先使用：

1. `tests/zemax/zmx_files/paraxial_single_lens.zmx`
2. `tests/zemax/zmx_files/four_surface_spherical.zmx`
3. `tests/zemax/zmx_files/Double Gauss 28 degree field.zmx`

对标顺序：

1. 先对 chief-ray referenced OPL difference
2. 再对 pupil OPD map
3. 最后服务 Huygens PSF 对标

Zemax 侧尽可能直接读取 OPD/OPL 分析结果，不通过本项目光线追迹反算。

## 开发步骤

建议按下面顺序执行：

1. 接通默认开启的 OPL 累计
2. 新增 `_incident_medium(...)`，保证 OPL 和折射使用同一介质语义
3. 增加最小 contract 测试，验证空气/玻璃段 OPL
4. 新增 `optics_core/opd.py`，实现 chief ray 参考 OPD
5. 和 Zemax 做第一轮 OPD 对标
6. 增加 exit pupil referenced OPD helper
7. 接入惠更斯 PSF 开发链路

## 注意事项

1. 不要在 tracer 中直接减 chief ray OPL
2. 不要在 tracer 中实现 reference sphere correction
3. 不要把 `RayBundle.opl` 改成新的复杂结构
4. 不要为了 OPD 首版实现完整 `WavefrontMap`
5. 不要在每个 surface 把 tensor 拉回 CPU
6. 对无效光线，OPL 应该同步变成 `nan`，避免后续 PSF 误用
7. `wavelength_um` 需要保持和光线 batch shape 一致，材料折射率计算直接吃 tensor
8. 如果后续觉得 `RayBundle.opl` 命名仍不够直观，可以在一次单独重构中改名为 `optical_path_mm`，不要和本次功能混在一起

## 预期结果

完成本阶段后：

1. `system.trace(...)` 默认返回累计 OPL
2. `opd.py` 能把 OPL 转换成 chief ray 参考 OPD
3. PSF 模块可以直接基于追迹结果构造 pupil phase
4. 后续波前功能可以复用同一套 OPD helper，而不需要重写追迹逻辑
