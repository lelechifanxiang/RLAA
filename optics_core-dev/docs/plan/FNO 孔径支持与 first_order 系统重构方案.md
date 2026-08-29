# FNO 孔径支持与 first_order 系统重构方案

## 1. 背景

当前 `optics_core/first_order.py` 将 4 根近轴基底光线和 1 根真实边缘光线统一放在 `build_first_order_probe_rays()` 中构造。这个设计在无限物距 ENPD 系统中可以用一次 trace 同时得到 EFFL、入瞳和 Working F/#，但扩展到 FLOA、有限物距和 Image Space F/# 后出现了明显问题：

1. `run()` 根据孔径类型设置临时值，承担了过多策略判断。
2. FLOA 首次把 stop 半径临时当作入瞳半径，只是为了满足统一 builder 的参数要求。
3. 有限物距或 FLOA 会第二次调用同一个 builder，再次追迹全部 5 根光线，但只使用第 5 根结果。
4. 各求解函数通过固定列号区分光线语义，近轴量和真实边缘量耦合。
5. Image Space F/# 必须先知道 EFFL 才能换算入瞳半径，因此无法在首次追迹前正确构造第 5 根光线。
6. `entrance_pupil_radius()` 只认识 ENPD，其职责已经不足以覆盖多种系统孔径定义。

本方案取代早期的 `docs/plan/一阶量 Probe Rays 合并方案.md` 中“所有物方 probe 固定合并”的假设。早期方案仍可作为 ENPD 无限物距场景的性能优化参考，但不再作为统一的数据流设计。

## 2. 目标与非目标

### 2.1 目标

1. 让 `_FirstOrderCalculator.run()` 只表达清晰的一阶计算阶段，不直接处理各孔径类型分支。
2. 删除对 `build_first_order_probe_rays()` 的重复调用和临时入瞳半径。
3. 支持现有 `ApertureType` 中的 `image_f_number`，并解析 ZMX `FNUM`。
4. 保持 ENPD、FLOA、有限物距和无焦像空间的现有行为。
5. 保持 batch-first、FP64 tensor 和 GPU 批量执行方式。
6. 继续使用真实边缘光线计算 Working F/#，不把系统设置的 Image Space F/# 直接当作 Working F/#。
7. 为将来使用解析近轴矩阵替换 4 根近轴 probe 留出接口，但本次不要求实现近轴矩阵。

### 2.2 非目标

1. 不实现真实光线瞄准或视场相关的 pupil aiming。
2. 不在本次支持 `object_na`。
3. 不实现 Zemax 的 Paraxial Working F/#、Object Cone Angle 等额外孔径类型。
4. 不追求离轴或非轴对称系统完整的 Zemax Working F/# 定义。
5. 不改变 `FirstOrderData` 的公开字段。

本次 Working F/# 继续采用轴上主波长的一根径向真实边缘光线。该范围与当前实现一致。

## 3. FNO 的定义边界

项目内部继续使用已有的规范名称：

```python
SystemAperture(kind="image_f_number", value=f_number)
```

ZMX 中对应：

```text
FNUM 4.0
```

解析后的 Zemax 名称建议使用 `ImageSpaceFNumber`，再映射为内部的 `image_f_number`。不新增 `fno`、`f_number` 等同义枚举，避免公共接口出现多套命名。

### 3.1 Image Space F/#

OpticStudio 将 Image Space F/# 定义为无限共轭近轴有效焦距与近轴入瞳直径的比值，即使系统实际用于有限共轭也仍采用该定义：

\[
F_{\mathrm{image}}=\frac{|\mathrm{EFFL}|}{\mathrm{ENPD}}
\]

因此：

\[
R_{\mathrm{EP}}=\frac{|\mathrm{EFFL}|}{2F_{\mathrm{image}}}
\]

### 3.2 Working F/#

Working F/# 基于实际共轭下的真实边缘光线像方角度，不能直接令：

```python
working_f_number = system.aperture.value
```

即使系统孔径类型是 `image_f_number`，仍要先由 EFFL 换算入瞳半径，再构造真实边缘光线并追迹得到 Working F/#。对于有限物距、像差明显或无焦输出的系统，二者可能明显不同。

当前 `solve_working_f_number_from_probe()` 使用：

\[
F_{\mathrm{working}}\approx\frac{1}{2\sin\theta'}
\]

这隐含像空间折射率为 1。本次可先保持已有的空气像空间行为；若要同时完善非空气像空间，应显式读取主波长的像空间折射率并使用：

\[
F_{\mathrm{working}}=\frac{1}{2n'\sin\theta'}
\]

非空气像空间建议作为独立提交处理，避免 FNUM 重构同时扩大介质语义范围。

## 4. 总体设计

将一阶计算拆成四个有明确依赖关系的阶段：

```text
阶段 A：求与孔径大小无关的前组近轴量
  ├── EFFL
  ├── entrance_pupil_z
  └── front_pupil_magnification
              │
              ▼
阶段 B：按系统孔径类型解析实际口径
  ├── entrance_pupil_radius
  └── stop_radius
              │
              ▼
阶段 C：构造并追迹一根真实边缘光线
  └── working_f_number

阶段 D：求后组近轴量
  ├── exit_pupil_z
  ├── rear_pupil_magnification
  └── exit_pupil_radius
```

阶段 A 和 D 只描述系统的一阶几何性质，与系统孔径数值无关。阶段 B 集中处理 ENPD、FLOA 和 FNO 的换算。阶段 C 是唯一依赖实际入瞳半径和有限物距的真实光线阶段。

## 5. 内部数据结构

建议增加三个仅供一阶模块内部使用的结果类型，所有 tensor 形状均为 `(system_count,)`：

```python
@dataclass(slots=True)
class FrontParaxialData:
    effl: torch.Tensor
    entrance_pupil_z: torch.Tensor
    front_pupil_magnification: torch.Tensor


@dataclass(slots=True)
class ResolvedAperture:
    entrance_pupil_radius: torch.Tensor
    stop_radius: torch.Tensor


@dataclass(slots=True)
class ExitPupilData:
    exit_pupil_z: torch.Tensor
    rear_pupil_magnification: torch.Tensor
```

这些类型不放入 `system_state.py`，因为它们只是构造最终 `FirstOrderData` 的中间态，不属于准备完成后的公开缓存。

## 6. Probe 光线接口重构

### 6.1 前组近轴 probe

将当前 builder 中的前 4 根光线拆成独立函数：

```python
def build_front_paraxial_probe_rays(
    system: MultiOpticalSystem,
    probe_height: torch.Tensor,
) -> RayBundle:
    """构造正负高度、正负斜率共 4 根前组近轴探测光线。"""
```

该函数不接收入瞳半径和入瞳位置。一次完整正向追迹后，由纯 tensor 函数生成 `FrontParaxialData`：

```python
def solve_front_paraxial_data(
    result: TraceResult,
    *,
    stop_index: int,
    probe_height: torch.Tensor,
) -> FrontParaxialData:
    ...
```

EFFL 继续使用正负高度光线的像方斜率；入瞳位置和前组瞳放大率继续通过 stop 面的中心差分恢复，不在本次改变其数学定义。

### 6.2 真实边缘光线

将第 5 根光线拆成只返回一根光线的 builder：

```python
def build_marginal_probe_ray(
    system: MultiOpticalSystem,
    *,
    entrance_pupil_z: torch.Tensor,
    entrance_pupil_radius: torch.Tensor,
) -> RayBundle:
    """构造轴上物点到入瞳边缘的真实边缘光线。"""
```

返回的 ray 维度固定为 `(system_count, 1)`。无限物距时生成平行边缘光线；有限物距时连接轴上物点和入瞳边缘。

相应求解函数不再硬编码第 5 列：

```python
def solve_working_f_number_from_marginal_ray(
    result: TraceResult,
) -> torch.Tensor:
    """由单根真实边缘光线的像方数值孔径求 Working F/#。"""
```

函数直接读取索引 0，因为输入契约明确为单根边缘光线。

### 6.3 出瞳 probe

现有两根出瞳 probe 可以保留，但求解函数返回 `ExitPupilData`，避免在 `run()` 中传递裸 tuple。出瞳半径在实际 stop 半径解析完成后统一计算：

\[
R_{\mathrm{XP}}=|M_{\mathrm{rear}}|R_{\mathrm{stop}}
\]

## 7. 统一的孔径解析器

删除或收窄当前只支持 ENPD 的 `entrance_pupil_radius()`，新增集中式解析函数：

```python
def resolve_system_aperture(
    system: MultiOpticalSystem,
    *,
    effl: torch.Tensor,
    front_pupil_magnification: torch.Tensor,
) -> ResolvedAperture:
    ...
```

孔径分支只允许出现在这个函数中。FLOA 所需的 stop 面固定半口径也由该函数内部读取，不能要求 `run()` 预先判断孔径类型并准备参数：

### 7.1 Entrance Pupil Diameter

\[
R_{\mathrm{EP}}=\frac{\mathrm{aperture.value}}{2}
\]

\[
R_{\mathrm{stop}}=\frac{R_{\mathrm{EP}}}{|M_{\mathrm{front}}|}
\]

### 7.2 Float By Stop Size

\[
R_{\mathrm{stop}}=R_{\mathrm{stop,surface}}
\]

\[
R_{\mathrm{EP}}=|M_{\mathrm{front}}|R_{\mathrm{stop}}
\]

FLOA 不读取 `SystemAperture.value` 作为真实来源，stop 面的固定半口径仍是唯一来源。

### 7.3 Image Space F/#

\[
R_{\mathrm{EP}}=\frac{|\mathrm{EFFL}|}{2\,\mathrm{aperture.value}}
\]

\[
R_{\mathrm{stop}}=\frac{R_{\mathrm{EP}}}{|M_{\mathrm{front}}|}
\]

`aperture.value` 必须大于 0。FNO 在同一批设计中是同一个系统规格，但 EFFL 可以随 design 参数变化，因此 `entrance_pupil_radius` 和 `stop_radius` 必须保持逐 design 的 FP64 tensor，不能先转为 Python 标量。

### 7.4 暂不支持的类型

`object_na` 在解析器中保留明确的 `NotImplementedError` 分支。这样以后增加新类型时只扩展孔径解析器，不再改写 `run()`。

## 8. 重构后的 `run()`

目标代码结构如下：

```python
def run(self) -> FirstOrderData:
    ttl = self.total_track_length()
    front = self.solve_front_paraxial()
    aperture = self.resolve_aperture(front)
    working_f_number = self.trace_working_f_number(front, aperture)
    exit_pupil = self.solve_exit_pupil()

    return FirstOrderData(
        ttl=ttl,
        effl=front.effl,
        working_f_number=working_f_number,
        entrance_pupil_z=front.entrance_pupil_z,
        entrance_pupil_radius=aperture.entrance_pupil_radius,
        stop_radius=aperture.stop_radius,
        exit_pupil_z=exit_pupil.exit_pupil_z,
        exit_pupil_radius=(
            exit_pupil.rear_pupil_magnification.abs()
            * aperture.stop_radius
        ),
    )
```

`run()` 中不再包含：

- `if aperture.kind == ...`；
- 临时把 stop 半径写入入瞳半径；
- 对同一个 probe builder 的二次调用；
- 依赖第 5 列的结果读取；
- 对“是否需要重新追迹”的判断。

## 9. Trace 调度与并行性能

### 9.1 基线实现

重构后的统一基线为：

1. 4 根前组近轴光线，一次完整正向 trace；
2. 1 根真实边缘光线，一次完整正向 trace；
3. 2 根出瞳光线，一次 stop 到像面的 trace。

总计 3 次 trace，共处理 7 根光线。有限物距和 FLOA 当前也是 3 次 trace，但物方两次各追迹 5 根光线；重构后物方部分从 10 根降为 5 根。

FNO 必须先得到 EFFL 才能计算入瞳半径，因此自然采用 `4 + 1 + 2` 的调度。

### 9.2 ENPD 无限物距的可选融合优化

ENPD 无限物距系统在首次追迹前已经知道边缘光线的完整初始条件。当前 `5 + 2` 两次 trace 可能比统一基线的 `4 + 1 + 2` 三次 trace 更快。

因此，接口应保持逻辑分离，但在基准测试证明第三次调度有明显开销时，可以增加私有融合层：

```text
build_front_paraxial_probe_rays() ─┐
                                   ├─ concatenate → 一次 trace
build_marginal_probe_ray() ────────┘
```

融合只允许出现在调度层，不能重新把两类光线合并回同一个 builder。求解函数通过明确的 slice 或私有 layout 读取各自结果。

推荐先实现结构清晰的 3 次 trace 版本并建立基准，再决定是否加入融合 fast path。若加入 fast path，目标调度为：

| 孔径与物距 | 物方 probe | 出瞳 probe | trace 次数 |
|---|---:|---:|---:|
| ENPD、无限物距 | `4+1` 融合 | 2 | 2 |
| ENPD、有限物距 | 4 后再 1 | 2 | 3 |
| FLOA | 4 后再 1 | 2 | 3 |
| FNO | 4 后再 1 | 2 | 3 |

不要为了保持固定 trace 次数，再向未知的入瞳参数写入占位值。

### 9.3 后续解析近轴模块

如果一阶量成为性能瓶颈，可参考 `reference/GYOptics/DiffOptics.py` 增加批量近轴传递矩阵，用解析递推替代前 4 根和出瞳 2 根 probe。届时通用追迹器只需要处理 1 根真实边缘光线。

解析近轴模块属于后续阶段。本次先消除职责耦合，避免同时改变一阶数学方法和孔径类型语义。

## 10. 文件组织建议

为避免 `first_order.py` 继续增长，建议拆分私有实现：

```text
optics_core/
├── first_order.py               # _FirstOrderCalculator 与公开入口
├── _first_order_probes.py       # 三类 probe builder 和 TraceResult 求解函数
└── _first_order_aperture.py     # ResolvedAperture 与孔径类型换算
```

`compute_first_order_data()` 和现有需要公开使用的 helper 继续从 `optics_core.first_order` 提供，避免破坏导入路径。若当前重构规模尚小，也可以先保持单文件，但必须维持上述职责边界。

## 11. ZMX 加载改动

### 11.1 头部解析

在 `zemax_utils/zmx_loader.py::_parse_zmx_header()` 中增加：

```python
if command == "FNUM":
    aperture_kind = "ImageSpaceFNumber"
    aperture_value = _parse_zmx_float(tokens[1])
    continue
```

`OBNA` 继续保持未支持。

### 11.2 系统构建映射

增加映射：

```python
aperture_kind_by_zemax = {
    "EntrancePupilDiameter": "entrance_pupil_diameter",
    "ImageSpaceFNumber": "image_f_number",
    "FloatByStopSize": "float_by_stop_size",
}
```

无需修改 `ApertureType`，因为 `image_f_number` 已经存在。`SystemAperture` 结构也无需新增字段。

## 12. 测试设计

所有准确性测试均应直接通过 ZOS-API/ZOSPy 读取 Zemax 一阶量，不通过其他追迹结果反算期望值。

### 12.1 Contract 测试

1. `build_front_paraxial_probe_rays()` 输出 `(system_count, 4)`。
2. `build_marginal_probe_ray()` 输出 `(system_count, 1)`。
3. 有限物距边缘光线连接轴上物点和入瞳边缘。
4. `solve_working_f_number_from_marginal_ray()` 不依赖固定第 5 列。
5. ENPD、FLOA、FNO 三种解析公式分别使用 FP64 tensor 验证。
6. FNO 多设计批量中，固定 FNO、不同 EFFL 应得到不同入瞳半径。
7. `FNUM` 头部能够解析为 `ImageSpaceFNumber`，构建系统后得到 `image_f_number`。
8. `OBNA` 仍明确进入未支持分支。
9. 将现有“仅 ENPD 受支持”的测试替换为 FNO 正向测试，并改用 `object_na` 验证未支持分支。

### 12.2 Trace 调度测试

使用计数 tracer 同时记录调用次数和每次的 ray count：

- 基线重构：`4、1、2`；
- 若实现 ENPD 无限物距融合：`5、2`；
- 有限物距、FLOA、FNO 不允许出现第二个 5-ray trace；
- 所有一阶 trace 继续设置 `ignore_coordinate_breaks=True`；
- 只有前组近轴 trace 需要 `record_intersections=True`。

真实边缘光线和出瞳 trace 不需要记录全部逐面交点，应显式设置 `record_intersections=False`。三类一阶 probe 都不消费 OPL，应统一设置 `record_opd=False`，减少不必要的计算和显存占用。

### 12.3 Zemax 回归测试

至少准备两份 FNUM ZMX：

1. 无限物距、普通成像系统；
2. 有限物距系统。

扩展 `tests/zemax/first_order.py` 和 `ZemaxFirstOrderReference`，直接读取并打印：

- `EFFL`；
- `ENPP`；
- `EPDI`；
- `EXPP`；
- `EXPD`；
- `ISFN`；
- `WFNO`。

重点验证：

- OpticsCore 的 `entrance_pupil_radius` 与 Zemax `EPDI/2` 一致；
- OpticsCore 的 `stop_radius` 符合前组瞳放大率关系；
- OpticsCore 的 `working_f_number` 对比 Zemax `WFNO`，而不是对比系统设置的 FNUM；
- 有限物距测试中打印 `ISFN` 与 `WFNO`，明确展示二者允许不同。

现有 ENPD、FLOA、双高斯和出瞳回归必须继续通过。

### 12.4 Benchmark

在 `tests/benchmark` 增加一阶量基准，至少覆盖：

- `system_count=1`：观察 trace 调度开销；
- 中等多系统 batch：观察 `4+1` 与融合 5-ray 的差异；
- 大规模 GPU batch：观察吞吐和显存峰值；
- ENPD、FLOA、FNO 三种孔径类型。

基准需要同时报告 trace 次数、总 ray 数、耗时和 CUDA 峰值显存。是否加入 ENPD 无限物距融合 fast path 应由该基准决定。

## 13. 实施顺序

### 阶段 1：建立行为基线

1. 增加 FNUM ZMX fixture。
2. 扩展 Zemax 一阶量读取，加入 `ISFN` 和 `WFNO`。
3. 记录当前 ENPD、FLOA 的数值结果和 trace 调度。

### 阶段 2：拆分 probe 职责

1. 新增 4-ray 前组 builder。
2. 新增 1-ray 边缘 builder。
3. 将 Working F/# 求解改为消费单根边缘光线结果。
4. 保留现有数学公式，先不引入解析近轴矩阵。

### 阶段 3：集中孔径解析

1. 增加 `ResolvedAperture`。
2. 将 ENPD、FLOA 迁移到统一解析器。
3. 删除 FLOA 的临时入瞳半径和二次 5-ray trace。
4. 重写 `run()` 为阶段式编排。

### 阶段 4：加入 FNO

1. 实现 `image_f_number` 解析公式。
2. 支持 ZMX `FNUM`。
3. 完成无限物距和有限物距 Zemax 回归。

### 阶段 5：性能收敛

1. 运行 CPU/GPU benchmark。
2. 仅在基准证明需要时加入 ENPD 无限物距融合 fast path。
3. 后续再评估是否实现解析近轴传递矩阵。

## 14. 验收标准

1. `_FirstOrderCalculator.run()` 中没有孔径类型分支。
2. `build_first_order_probe_rays()` 被删除，不再有临时 stop 半径冒充入瞳半径。
3. 有限物距、FLOA、FNO 路径只追迹一次真实边缘光线。
4. `image_f_number` 可以通过 Python API 和 ZMX `FNUM` 使用。
5. FNO 的 ENPP、EPDI、EXPP、EXPD、EFFL 和 WFNO 与 Zemax 对标。
6. 所有一阶数值接口继续返回 batch-first FP64 tensor，计算过程中不增加 CPU-GPU 往返。
7. ENPD、FLOA 现有回归无精度退化。
8. 文档、注释和测试输出使用中文。

## 15. 风险与处理

| 风险 | 处理方式 |
|---|---|
| 将 Image Space F/# 错当成 Working F/# | 两者使用独立字段、公式和 Zemax operand 验证 |
| FNO 有限物距仍错误使用设定值作为 WFNO | 强制通过真实边缘光线计算 WFNO |
| 拆分后 ENPD 小 batch 性能下降 | 建立 benchmark，必要时只在调度层融合 `4+1` |
| 前组瞳放大率接近零 | 沿用明确的退化判定，不用无意义常数替代 |
| 非空气像空间 WFNO 缺少折射率 | 本次声明空气像空间范围，后续独立补充 `n'` |
| 无焦像空间边缘光线几乎平行 | 保持现有行为并与 Zemax WFNO 打印对比，不在本次擅自增加 10000 上限 |
| 旧计划与新方案冲突 | 以本文为多孔径类型重构依据，旧文档仅保留历史背景 |

## 16. 参考定义

- [Ansys OpticStudio：Aperture Type](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v252/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Aperture_Type.html)
- [Ansys OpticStudio：Image Space F/#](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v252/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Image_Space_F.html)
- [Ansys OpticStudio：Working F/#](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v252/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Working_F.html)
- [Ansys OpticStudio：System Aperture](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/System_Aperture.html)
