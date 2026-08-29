# P0-6 异构镜头 Batch 开发方案

## 1. 目标和范围

SurfaceGenerator 需要把固定拓扑、不同处方的镜头组成一个 batch。每个 design 允许不同的：

- 曲率半径和厚度；
- 玻璃材料序列；
- 实际视场角；
- 入瞳位置和半径。

所有 design 必须具有相同的表面数量、表面类型序列、stop 位置、Coordinate Break 布局、视场数量和波长集合。不同拓扑应由上游先分组，不在本轮通过 padding 混合。

本轮不处理不同波长集合、材料可微优化、数据库解析和异构 surface type。

## 2. 五个 ZMX 文件的检查结论

测试文件位于：

```text
tests/zemax/zmx_files/same_arch_diff_materials/
```

### 2.1 结构检查

五个文件均可被当前 ZMX loader 正常解析，公共结构如下：

| 项目 | 检查结果 |
| --- | --- |
| Zemax 总表面数 | 15：物面 + 前置光阑 + 12 个折射面 + 像面 |
| Loader `spec.surfaces` | 13：前置光阑 + 12 个折射面，不包含物面和像面 |
| OpticsCore surface 数 | 14：上述 13 面 + ImageSurface |
| Surface type | 全部为 Standard，五份文件一致 |
| Stop | Zemax 第 1 面，对应内部索引 0 |
| 视场 | 均为 5 个 Angle 视场，具体角度允许逐 design 不同 |
| 有效波长 | 0.486、0.588、0.656 μm，0.588 μm 为主波长 |
| 系统孔径 | 均为 Image Space F/#，数值允许逐 design 不同 |
| 物方 | 无穷远 |
| Coordinate Break | 无 |

文件中保留了第 4～24 条 0.55 μm 的 `WAVM` 槽位，但 `FTYP` 声明的有效波长数是 3。当前 loader 正确只读取前三个有效波长，因此不构成问题。

五份文件的表面类型序列、stop 位置、视场数量和波长集合一致，拓扑签名兼容；半径、厚度、材料和固定面半口径均可不同。三个原文件保持 HFOV 5°、F/# 6，两个补充文件分别为 HFOV 25°、F/# 3.0 和 HFOV 34°、F/# 4.8。

### 2.2 材料检查

| 文件 | 6 片玻璃材料序列 |
| --- | --- |
| `sg6_material_a.zmx` | H-ZK3, H-LAF10LA, H-ZF5, H-K9L, H-LAK52, H-ZF13 |
| `sg6_material_b.zmx` | H-ZPK2A, H-ZK9B, H-ZK21, H-ZF72A, H-K51, H-ZK21 |
| `sg6_material_c.zmx` | H-ZLAF52A, H-ZLAF4LA, H-K9L, H-ZF72A, H-LAF50B, H-K9L |
| `sg6_hfov25_f3p0.zmx` | H-ZLAF4LA, H-ZF7LA, H-ZLAF55D, H-ZPK5, H-LAK52, H-ZLAF50E |
| `sg6_hfov34_f4p8.zmx` | H-K51, H-ZF5, H-ZK21, H-ZPK1A, H-LAF50B, H-ZLAF75A |

所有材料均能从当前内置 `MaterialLibrary` 解析，没有缺失玻璃或材料名冲突。

### 2.3 单系统运行结果

五个文件均可构建 `MultiOpticalSystem` 并完成 `prepare()`。使用 5 个视场、3 个波长和 3 个 pupil 点追迹时，每个系统均为 `45/45` 光线有效。

| 文件 | HFOV | F/# | EFFL / mm | 入瞳半径 / mm | TTL / mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sg6_material_a.zmx` | 5° | 6.0 | 248.00023450 | 20.66668621 | 307.96195547 |
| `sg6_material_b.zmx` | 5° | 6.0 | 247.99894416 | 20.66657868 | 307.98701622 |
| `sg6_material_c.zmx` | 5° | 6.0 | 247.99941338 | 20.66661778 | 308.00039125 |
| `sg6_hfov25_f3p0.zmx` | 25° | 3.0 | 47.00167947 | 7.83361324 | 81.21380409 |
| `sg6_hfov34_f4p8.zmx` | 34° | 4.8 | 31.60534446 | 3.29222338 | 56.49966825 |

P0-4 完成后，五份系统的 Working F/# 均可正常计算，并与 Zemax 基准在 `1.3e-10` 内一致。一阶 probe 按 System Data 语义忽略处方固定半口径裁剪；普通光线追迹仍保留硬孔径行为，且未引入真实光线瞄准。

补充文件 README 中的 EFL、TTL、BFL 标签与 ZMX 当前处方的实际一阶量或轨长并不完全相同。P0-6 回归应以 ZMX 解析结果和 Zemax API 读取值为准；这不影响 HFOV、F/# 和拓扑覆盖。

### 2.4 Zemax direct ray 对比

每份文件选取轴上/边缘两个视场、三个有效波长和三个 pupil 点，共 18 根显式光线。结果如下：

- 五份文件均为 Zemax 与 OpticsCore `18/18` 有效性完全一致；
- 最大像面横向位置误差：`2.0e-13 mm`；
- 最大方向余弦误差：`1.6e-15`。

结论：五个文件本身没有阻塞问题，可以直接用于 P0-6 的单系统 Zemax 基准和异构 batch 验收。两个新文件补齐了逐 design HFOV、F/# 和入瞳差异的真实系统覆盖；显式 tensor 光线构造仍需 contract 测试独立验证。

## 3. 当前实现缺口

半径和厚度已经可以通过 `ParameterSchema` 与 `ParameterVectorBatch` 随 design 变化，例如：

```text
surface[i].geometry.radius
surface[i].gap.thickness
```

材料还不能真正 batch。当前折射和 OPL 都读取共享对象：

```python
system.surfaces[surface_index].gap.medium
```

即使参数向量保存了 `surface[i].gap.medium`，也只有 `design_view()` 深拷贝表面后才会生效。批量追迹仍会对全部 design 使用同一材料。

另外，`FieldSequence` 和 `SystemAperture` 也是共享定义。逐 design HFOV 和入瞳应通过显式 tensor 构造输入光线，不应改成嵌套 Python 对象列表。

## 4. 推荐设计

### 4.1 参数表示

继续复用现有参数系统，不增加另一套公开处方容器：

```python
ParameterSpec(
    name="surface_1_medium",
    path="surface[1].gap.medium",
    default="H-ZK3",
)
```

同一条参数向量可以同时保存半径、厚度和材料名。`design_view()` 保留现有材料物化逻辑，作为独立正确性基准。

五文件回归还通过现有 `surface[i].aperture_radius` 参数路径携带固定面半口径，避免所有 design 错用第一份文件的硬孔径。

### 4.2 材料执行态

新增私有模块 `optics_core/_material_batch.py`，核心数据为：

```python
@dataclass(slots=True)
class BatchedMaterialData:
    material_index: torch.Tensor          # int64 [design, surface]
    refractive_index_table: torch.Tensor  # float64 [material, wavelength]
    wavelength_um: torch.Tensor           # float64 [wavelength]
    material_names: tuple[str, ...]
    device: torch.device
```

约定：

- `material_index[:, i]` 表示第 `i` 面之后 gap 的材料；
- AIR 固定为索引 0，`None` 也按 AIR 处理；
- 折射率表在准备阶段调用现有 `Material.refractive_index()` 生成；
- tracing 热路径只使用 GPU tensor，不再逐面访问 Python `Material`；
- `design_batch_view()` 切片 `material_index`，共享折射率表。

`prepare()` 应在 first order 之前编译材料数据；未 prepare 的直接追迹允许按 device 懒编译一次。首轮把系统构建后视为冻结状态，不支持原地修改参数向量后继续复用旧缓存。

### 4.3 折射和 OPL

正向追迹时：

```text
incident(i)    = AIR, i == 0；否则为 material_index[:, i - 1]
transmitted(i) = material_index[:, i]
```

反向追迹时两者交换。

`_interactions.py` 应直接取得逐 design 的 incident/transmitted 折射率 tensor。`_core.py::_accumulate_optical_path()` 必须复用相同的 incident 折射率，避免出现方向正确但 OPL 仍使用共享材料的问题。

首轮只允许使用 `system.wavelengths` 中声明的波长。波长列号在 trace 入口解析一次，未声明波长直接报错。

### 4.4 HFOV 和入瞳

这部分与 P0-2 的公开光线构造接口重叠，应复用同一入口：

```python
build_pupil_rays(
    system,
    *,
    field_angles: torch.Tensor,           # [design, field, 2]
    entrance_pupil_z: torch.Tensor,       # [design]
    entrance_pupil_radius: torch.Tensor,  # [design]
    pupil_coordinates: torch.Tensor,      # [ray, 2]
    wavelength_um: torch.Tensor,          # [wavelength]
) -> RayBundle
```

输入统一为 FP64 tensor，输出形状为 `[design, field, wavelength, ray]`，不依赖 `first_order_data`。SurfaceGenerator 负责把 HFOV 转换为实际 field angle tensor。

## 5. 测试方案

### 5.1 Contract

只保留三个核心契约：

1. 两个 design 使用不同半径、厚度和材料时，batch 的光线、valid 和 OPL 等于两个 `design_view()`；
2. `material_index` 的 shape、dtype、device 和 `design_batch_view()` 切片正确；
3. 不同 field angle 和入瞳 tensor 构造出的批量光线，等于逐 design 构造后 stack。

### 5.2 五文件 Zemax Regression

回归目标改为验证五个异构系统并行计算的点列图 RMS 半径。测试辅助代码只放在 `tests/zemax`，不把“合并多个 ZMX”做成公共 API。

#### Zemax 基准生成

ZMX 加载和 Zemax 分析较慢，因此不在日常 pytest 中重复调用。提供一个手动基准生成脚本，依次加载五个 ZMX，通过 Standard Spot 记录每个文件、每个视场的 RMS 半径。固定设置为：

- `pattern="hexapolar"`、`ray_density=30`；
- 全部视场、全部波长、像面；
- `ReferTo=ChiefRay`，单位为 μm。

结果保存到 `tests/zemax/reference_data/same_arch_diff_materials_spot_rms.json`。文件同时记录 ZMX 文件名、文件哈希、视场、波长、采样设置、Zemax RMS 和 OpticsCore 单文件 RMS；只有 ZMX、采样设置或单文件基准变化时才重新生成。

```powershell
python -m tests.zemax.heterogeneous_spot
```

#### 日常回归

1. 读取 JSON、校验五个 ZMX 的文件哈希，再加载 spec、检查拓扑签名并组成五条参数向量；
2. 使用逐 design field angle 和入瞳 tensor，一次运行五个 design 的并行点列图分析；
3. 取得形状为 `[design, field]` 的 RMS 半径；
4. 将第 `i` 个 design 与 JSON 中对应的 OpticsCore 单文件 RMS 严格比较；
5. 打印每个文件、每个视场的 Zemax RMS、单文件 RMS、batch RMS 和两类绝对误差。

batch 与单文件 RMS 使用 `1e-9 μm` 绝对容差。由于真实光线瞄准不在当前范围，Zemax RMS 用于记录精度差异，不作为 P0-6 的严格通过门槛。该测试只读取预存基准，不要求本机安装或启动 Zemax。Working F/# 留给 P0-4，不纳入本轮通过条件。

### 5.3 并行点列图示例

`examples/batch_spot_multiple_zmx.py` 直接解析五个本地 ZMX 文本并读取预存 JSON，不依赖 Zemax 或 ZOSPy。example 只执行并行点列图分析，不再构造逐设计串行基线或比较加速比。

五个 design 可循环扩展到指定规模。分析后将每个 design 的逐视场 RMS 半径与预存单文件结果比较，使用 `1e-9 μm` 绝对容差，并打印五类镜头的视场角、RMS 半径和最大差异。

本机 CPU 冒烟方法：

```powershell
python examples/batch_spot_multiple_zmx.py
```

Linux CUDA 设备上的大规模运行方法：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python examples/batch_spot_multiple_zmx.py --device cuda:0 --design-count 300
python examples/batch_spot_multiple_zmx.py --device cuda:0 --design-count 3600 --summary-json heterogeneous_3600.json
```

## 6. 实施顺序

1. 增加同拓扑异材料失败测试，证明当前 batch 错误复用共享 medium；
2. 实现 `_material_batch.py` 和准备态切片；
3. 让折射与 OPL 统一读取批量折射率；
4. 实现逐 design field/入瞳 tensor builder；
5. 生成五文件 Zemax spot RMS 基准，并完成并行点列图回归；
6. 本机运行全量测试和 example CPU 冒烟，在 CUDA 设备运行大规模并行点列图分析。

预计涉及：

- `optics_core/_material_batch.py`；
- `optics_core/system.py`；
- `optics_core/tracing/_interactions.py`；
- `optics_core/tracing/_core.py`；
- `optics_core/tracing/_sampled_rays.py`；
- `tests/contract/test_heterogeneous_batch.py`；
- `tests/zemax/heterogeneous_spot.py`；
- `tests/zemax/reference_data/same_arch_diff_materials_spot_rms.json`；
- `tests/regression/test_heterogeneous_batch_against_zemax.py`；
- `examples/batch_spot_multiple_zmx.py`。

## 7. 完成定义

- `material_index` 为 GPU `int64 [design, surface]`；
- batch 折射和 OPL 不再读取共享 `surface.gap.medium`；
- 半径、厚度和材料可逐 design 不同；
- 显式 tensor 接口支持逐 design HFOV 和入瞳；
- 五文件并行点列图的逐 design/逐视场 RMS 半径与预存单文件基准一致，并打印 Zemax 差异；
- 正向、反向、OPL、负厚度和 Coordinate Break 回归通过；
- CUDA 环境下 3,600 design example 可运行且无逐面 CPU-GPU 材料搬运；
- 全量 pytest 无新增失败。
