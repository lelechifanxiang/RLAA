# 一阶量 Probe Rays 合并方案

## 目标

当前一阶量计算分别为 EFFL、入瞳、出瞳和 Working F/# 构造 probe rays，通常需要 4 次追迹。

建议先合并为 **2 次追迹**，在保持现有追迹接口不变的情况下覆盖当前全部一阶量；后续若补充逐面方向状态记录，可进一步合并为 **1 次追迹**。

## 推荐方案：2 次追迹

### 第一次：物方到像方的共享探测光线

构造一个 `FirstOrderProbeBundle`，一次性包含：

1. `+h/-h` 两根近轴平行光线
2. `+u/-u` 两根近轴倾斜光线
3. 一根轴上主波长真实边缘光线

使用一次完整正向追迹，并记录逐面交点：

```python
TraceOptions(
    record_intersections=True,
    ignore_coordinate_breaks=True,
)
```

该结果可计算：

- `ttl`：继续直接读取 frame，不依赖 probe rays
- `effl`：由 `+h/-h` 光线的像方斜率计算
- `working_f_number`：由真实边缘光线的像方数值孔径计算
- `entrance_pupil_z`：由四根近轴基底光线在 stop 面的交点恢复前组矩阵的 `A、B`
- `stop_radius`：由 `stop_radius = abs(A) * entrance_pupil_radius` 计算
- `entrance_pupil_radius`：继续使用系统声明的 Entrance Pupil Diameter

前组矩阵只需 stop 面交点即可恢复：

```text
A = (x_stop(+h) - x_stop(-h)) / (2h)
B = (x_stop(+u) - x_stop(-u)) / (2u)

entrance_pupil_z = B / A
stop_radius = abs(A) * entrance_pupil_radius
```

使用正负探测光线做中心差分，可以保留当前实现对数值误差和非严格线性行为的抵抗能力。

### 第二次：stop 到像方的出瞳探测光线

保留当前 stop 面出发的两根近轴光线，一次正向追迹到像面：

```python
TraceOptions(
    start_surface=stop_index,
    stop_surface=image_surface_index,
    direction="forward",
    record_intersections=False,
    ignore_coordinate_breaks=True,
)
```

由两根光线的像方位置和方向求交，得到：

- `exit_pupil_z`
- rear pupil magnification
- `exit_pupil_radius`

这样无需修改通用追迹结果结构，风险较低，追迹次数由最多 4 次降为固定 2 次。

## 可选方案：进一步合并为 1 次追迹

若为 `SurfaceIntersection` 增加 stop 面的入射、出射方向快照，则第一次完整追迹可以恢复：

1. 物方到 stop 的前组近轴矩阵 `M_front`
2. 物方到像面的总矩阵 `M_total`
3. stop 到像面的后组矩阵：

```text
M_rear = M_total @ inverse(M_front)
```

再由 `M_rear` 直接计算出瞳位置和放大率，不再需要第二次追迹。

该方案性能最好，但会扩大通用 tracing 接口，因此建议作为第二阶段处理，避免为了当前一阶量过早增加全局记录开销。

## 代码重构建议

在 `first_order.py` 中增加以下内部结构：

```python
@dataclass(slots=True)
class FirstOrderProbeTrace:
    result: TraceResult
    height_positive_index: int
    height_negative_index: int
    slope_positive_index: int
    slope_negative_index: int
    marginal_index: int
```

主流程调整为：

```text
build shared probe rays
        │
        ▼
第一次完整正向追迹
        ├── EFFL
        ├── Working F/#
        ├── Entrance pupil
        └── Stop radius

第二次 stop→image 追迹
        ├── Exit pupil position
        └── Exit pupil radius
```

删除独立的：

- `build_effective_focal_length_probe_rays`
- `build_working_f_number_probe_rays`
- 入瞳反向 probe trace

各一阶量求解函数继续保持为无追迹副作用的 tensor 计算函数，统一消费共享结果。

## 验证

1. 现有一阶量 contract 和 regression 测试必须全部通过。
2. 双高斯系统继续与 Zemax 对标：
   - EFFL
   - ENPP / ENPD
   - EXPP / EXPD
   - WFNO
3. 增加多设计 batch 测试，确认所有矩阵和结果维度均为 `(system_count, ...)`。
4. 增加计数 tracer 测试，断言 `system.prepare()` 的一阶量计算最多调用 2 次 `trace()`。

## 建议实施顺序

1. 先实现两次追迹版本并完成 Zemax 回归。
2. 确认一阶矩阵拆分在球面、近轴面和坐标间断系统上稳定。
3. 只有在一阶计算确实构成性能瓶颈时，再扩展逐面方向记录并收敛为一次追迹。
