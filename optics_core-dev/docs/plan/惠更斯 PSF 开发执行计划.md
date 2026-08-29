# 惠更斯 PSF 开发执行计划

## 前置条件判断

当前已经具备实现惠更斯 PSF 的主要前置条件：

1. `system.prepare()` 已缓存 `first_order_data.exit_pupil_z` 和 `exit_pupil_radius`
2. 顺序追迹默认累计 `RayBundle.opl`
3. 采样追迹结果中已有 `chief_ray_index`
4. `SurfaceIntersection` 可记录像面交点和法向
5. `system.analysis.psf()` 入口已经存在，只需接通实现

仍需在 PSF 功能内部补齐：

1. exit-pupil referenced OPD
2. Huygens 积分
3. Zemax Huygens PSF helper 和回归测试

这些不是阻塞项，可以作为本轮 PSF 开发内容。

## 首版目标

实现单色、单视场、固定语义的惠更斯 PSF：

1. 默认使用主波长，支持通过 `wavelength_index` 指定单个波长
2. 视场由 `field_index` 指定，默认 `0`
3. 光瞳采样为 square grid，默认 `32 x 32`
4. 图像采样为 square grid，默认 `32 x 32`
5. 像面中心固定为 chief ray 像面交点
6. 不支持 centroid、polarization、normalize、自动 image delta、球面参考自动切换

首版输出：

1. `psf`
2. `strehl_ratio`
3. `pixel_pitch_um`

## 接口计划

修改 `PSFSettings`：

```python
@dataclass(slots=True)
class PSFSettings:
    pupil_sample_count: int = 32
    image_sample_count: int = 32
    field_index: int = 0
    wavelength_index: int | None = None
```

`wavelength_index` 语义：

1. `None`：使用系统主波长
2. `0, 1, ...`：使用指定波长
3. `-1`：计算全部波长

`PSFResult.psf` 形状建议：

```text
(system_count, wavelength_count, image_y, image_x)
```

`strehl_ratio` 形状建议：

```text
(system_count, wavelength_count)
```

`pixel_pitch_um` 首版固定为内部常量，例如 `0.5 um`，暂不开放参数。

## 实现步骤

### 1. 接通分析入口

修改 `PointSpreadFunction.run()`：

```python
from .huygens_psf import run_huygens_psf
return run_huygens_psf(self.system, self.settings)
```

### 2. 新增 `optics_core/huygens_psf.py`

主流程：

1. 检查 `system.prepare()` 已执行
2. 取 `settings.field_index` 对应视场
3. 按 `wavelength_index` 选择主波长、指定波长或全部波长
4. 构造 `SquarePupilSampler(nx=N, ny=N)`
5. 使用 `build_input_rays_from_sample(...)` 组装光线
6. `TraceOptions(record_intersections=False)` 追迹到像面
7. 提取 chief ray 像面交点、每根光线的 OPL
8. 计算 exit-pupil referenced OPD
9. 在 chief ray 像面点附近构造 `image_sample_count x image_sample_count` 网格
10. 执行 Huygens 复振幅叠加
11. 输出 PSF 和 Strehl ratio

### 3. OPD 计算

首版采用“主光线参考 OPL + 出瞳几何距离修正”：

```text
opd = OPL差 + 像面点到出瞳参考球面的几何修正
```

如果后续和 Zemax 偏差明显，再继续检查参考球面、采样权重和 Zemax Huygens PSF 的归一化语义。

### 4. Huygens 积分

首版采用最小模型：

1. 每条有效 pupil ray 振幅为 `1.0`
2. 无效光线不参与
3. 相位为 `2*pi*opd_mm/(wavelength_um*1e-3)`
4. 对每个 image grid 点做复振幅叠加
5. `psf = abs(amplitude) ** 2`
6. 用零 OPD 情况计算 reference peak，得到 `strehl_ratio`

## 测试计划

### 1. contract 测试

新增 `tests/contract/test_huygens_psf_contract.py`：

1. 默认参数能运行
2. 输出 shape 正确
3. `pupil_sample_count/image_sample_count/field_index` 生效
4. 未执行 `prepare()` 时直接报错

### 2. Zemax 回归测试

新增：

1. `tests/zemax/huygens_psf.py`
2. `tests/regression/test_huygens_psf_against_zemax.py`

优先使用：

1. `tests/zemax/zmx_files/paraxial_single_lens.zmx`
2. `tests/zemax/zmx_files/four_surface_spherical.zmx`

Zemax 设置固定：

1. Huygens PSF
2. Field = `field_index`
3. Wavelength = Primary
4. Pupil Sampling = `pupil_sample_count`
5. Image Sampling = `image_sample_count`
6. Use Centroid = False
7. Normalize = False
8. Polarization = False

首轮回归先检查：

1. PSF peak 是否在中心附近
2. `strehl_ratio` 数量级是否一致
3. 中心行/列形状是否一致

如果 OPD reference 与 Zemax 仍存在系统偏差，先保留诊断打印，不急于收紧阈值。

## 开发宗旨

1. 优先跑通主链，不做通用波前框架
2. 参数只支持 `pupil_sample_count`、`image_sample_count`、`field_index`、`wavelength_index`
3. 能用现有采样、追迹、first order 数据就直接复用
4. 所有核心计算保持 tensor 化
5. 首版先对齐简单系统，再扩展到复杂镜头
