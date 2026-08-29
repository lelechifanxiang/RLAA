# spot tracing 显存与性能分析

## 1. 目标与背景

本文整理 `python examples/batch_spot.py --device cuda:0 --surfaces 1 2 10 11 --ray-density 13` 这一批量点列图场景下的：

- 光线 tracing 显存占用分析
- 当前 tracing 耗时分布
- 可执行的优化计划

当前目标不是做通用框架优化，而是服务于快速实现 spot / PSF 相关分析能力。优化策略应优先满足：

- 尽量复用现有 tracing 能力
- 代码保持精简可读
- 不为了“通用性”引入额外封装
- 参数与 Zemax 保持一致，优先支持当前固定分析流程

## 2. 测试场景

### 2.1 用户执行日志

```text
zmx 文件: C:\Users\huweijian\Project\optics_core\tests\zemax\zmx_files\Double Gauss 28 degree field.zmx
设备: cuda:0
Zemax 扰动面号: [1, 2, 10, 11]
内部 surface 索引: [0, 1, 9, 10]
视场点: ((0.0, 0.0), (0.0, 10.0), (0.0, 14.0))
波长(um): [0.4861, 0.5876, 0.6563]
总设计数: 6561
固定厚度扰动: +/- 0.100000 mm
固定曲率扰动: +/- 1.000000e-04 1/mm
并行 spot 信息:
pattern=hexapolar, ray_density=13, pupil_ray_count=547
design_count=6561, field_count=3, wavelength_count=3
total_ray_count=32299803, valid_ray_count=32130747
spot 分析耗时: 2777.979 ms
CSV 已保存: C:\Users\huweijian\Project\optics_core\examples\output\batch_spot.csv
批量点列图测试完成。
pupil_ray_count=547
valid_ray_count=32130747
elapsed_ms=2777.979
```

### 2.2 数据规模

- 扰动表面数：4 个
- 每个表面扰动参数：厚度、半径，共 2 个
- 参数轴总数：8 个
- 设计数：`3^8 = 6561`
- 视场数：3
- 波长数：3
- 六边采样光线数：`1 + 3 * rings * (rings + 1)`
- 当 `ray_density=13` 时，`pupil_ray_count = 547`
- 总并行光线数：`6561 * 3 * 3 * 547 = 32,299,803`

这个数量级本身就足以解释为什么 16G V100 很快接近显存上限。

## 3. 显存占用分析

### 3.1 基础量级

当前 tracing 主链路统一使用 `float64`，单个全尺寸张量大小约为：

- `32,299,803 * 8 bytes ≈ 246.4 MB`

若只看最基础的显式光线状态：

- `x/y/z/l/m/n` 六个张量：约 `1.44 GB`
- `RayBundle` 当前实际还包含 `wavelength_um/intensity/opd`
- 因此仅 `RayBundle` 这 9 个 `float64` 张量就约 `2.17 GB`

spot 后处理还会继续创建：

- `dx/dy`：约 `2 * 246.4 MB`
- `radius_squared_mm`：约 `246.4 MB`
- `radius_mm`：约 `246.4 MB`

只看这些“看得见”的张量，显存已经达到数 GB。再叠加 tracing 过程中每个表面的中间临时量，达到 12G 以上是正常现象。

### 3.2 tracing 过程中的主要显存来源

#### 3.2.1 光线组装阶段

`optics_core/tracing/_sampled_rays.py::build_input_rays_from_sample()`

当前会一次性构造：

- `x/y/z/l/m/n`
- `wavelength_um`
- `intensity=torch.ones_like(x)`
- `opd=torch.zeros_like(x)`

其中：

- `intensity` 在当前 spot 分析链路中未使用
- `opd` 在当前 spot 分析链路中未使用
- `wavelength_um` 被展开为完整四维张量，而不是更紧凑的波长维表达

这部分是明确可优化的冗余显存。

#### 3.2.2 sag 面求交阶段

`optics_core/tracing/_hits.py::_sag_surface_hit()`

当前标准球面 / 标准几何路径里会额外创建：

- `local_origin = stack(x, y, z - surface_z)`
- `local_direction = stack(travel_l, travel_m, travel_n)`
- `path`
- `local_hit_x/local_hit_y/local_hit_z`
- `hit_x/hit_y/hit_z`
- `normal_x/normal_y/normal_z`
- `valid`
- `axial_offset`

其中 `local_origin/local_direction` 是两个 3 通道大张量，对当前 3200 万条光线规模来说是明显的大块瞬时显存。

#### 3.2.3 折射相互作用阶段

`optics_core/tracing/_interactions.py::_apply_refractive_interaction()`

该阶段还会派生：

- `incident_index`
- `transmitted_index`
- `unit_l/unit_m/unit_n`
- `oriented_normal_x/y/z`
- `cos_i`
- `eta`
- `k`
- `sqrt_k`
- `refracted_l/m/n`
- `outgoing_l/m/n`

这些张量都与主光线批次同形状，因此峰值显存远高于“只存一份光线状态”的直觉估计。

### 3.3 为什么 `ray_density` 稍微增加就会打满显存

六边采样光线数不是线性增长，而是二次增长：

- `ray_density=13` 时：547 条 pupil 光线
- `ray_density=15` 时：721 条，约为 `1.318x`
- `ray_density=20` 时：1261 条，约为 `2.305x`
- `ray_density=25` 时：1951 条，约为 `3.567x`
- `ray_density=30` 时：2791 条，约为 `5.102x`

因此，当 `ray_density=13` 已经接近 16G 的 80% 时，继续提升密度非常容易 OOM，这与当前实现完全一致，不是异常行为。

## 4. 显存优化计划

### 4.1 第一优先级：按 design 维分块

这是当前最直接、最稳妥、收益最高的方案。

思路：

- 不一次跑完 6561 个设计
- 按 design 维拆成多个 chunk
- 每个 chunk 分别做入瞳估计、光线组装、trace、spot 统计
- 最后再拼接 CSV 输出

优点：

- 峰值显存近似按 chunk 大小线性下降
- 对现有架构侵入最小
- 几乎不影响结果正确性
- 可以立刻支持更高 `ray_density`

结论：

- 如果目标是“尽快把功能跑起来”，这是必须优先做的优化

### 4.2 第二优先级：去掉 spot 路径中未使用的 `intensity/opd`

当前 `RayBundle` 已允许：

- `intensity: ArrayLike | None`
- `opd: ArrayLike | None`

因此 spot tracing 路径完全可以不分配这两个全尺寸张量，或者仅在确实需要时再创建。

预期收益：

- 直接减少两份全尺寸 `float64` 张量
- 对当前规模大约可省 `2 * 246.4 MB ≈ 493 MB`

### 4.3 第三优先级：避免标准几何路径中的大 `stack`

对于当前项目最常见的标准球面路径，可直接传递分量：

- `ox/oy/oz`
- `dx/dy/dz`

避免先 `stack` 再 `_split_local_vector()`。

预期收益：

- 降低瞬时峰值显存
- 降低额外张量构造开销

### 4.4 第四优先级：`save_path=None` 时直接计算统计量

当前 spot 后处理先构造：

- `dx`
- `dy`
- `radius_squared_mm`
- `radius_mm`

若只是为了拿：

- RMS 半径
- GEO 半径

则可以考虑直接从最终像面坐标计算，少保留一部分大中间量。

### 4.5 第五优先级：预打包 tracing 所需的 surface 参数

当前 tracing 内部频繁调用：

- `surface_position()`
- `surface_value()`

它们会反复按 design 遍历参数并重新构造 tensor。若后续需要继续提速，可在 trace 开始前一次性预打包：

- 各 surface 的 vertex z
- 半径
- 圆锥常数
- aperture radius

优点：

- 减少 Python 循环
- 减少重复 tensor 构造
- 代码依然可以保持相对简单

## 5. tracing 耗时分布分析

## 5.1 结论先行

当前 batch spot 场景下：

- 主要耗时几乎全部在 `system.tracer.trace()`
- 入瞳估计有一定开销，但远小于主 trace
- 光线组装与 spot 后处理耗时很小

### 5.2 粗粒度分段计时

在当前环境对同一场景做一次 warmup 后，再分段同步计时，得到：

- `entrance_pupil_ms = 97.95`
- `build_rays_ms = 4.53`
- `trace_kernel_ms = 2322.53`
- `extract_spot_ms = 1.64`
- `compute_metrics_ms = 7.03`
- `trace_spot_total_ms = 2425.00`
- `full_pipeline_ms = 2433.67`

说明：

- `trace_spot_total_ms` 对应 `trace_spot_diagram_rays()` 这一主链路，和用户日志中的 `2777.979 ms` 属于同一量级
- 计时差异来自 warmup、GPU 状态、缓存和同步位置不同，属正常波动

按上述测量，`trace_spot_diagram_rays()` 内部大致占比为：

- 入瞳估计：约 `4.0%`
- 光线组装：约 `0.2%`
- 主 tracing：约 `95.8%`

因此如果目标是提速，必须优先优化主 tracing 内核。

### 5.3 逐面耗时

对 `SequentialSurfaceRayTracer._trace_surfaces()` 做同步插桩后，单次 trace 大致分布为：

- surface 0: `204.93 ms`
- surface 1: `209.64 ms`
- surface 2: `212.14 ms`
- surface 3: `209.22 ms`
- surface 4: `217.55 ms`
- surface 5: `226.05 ms`
- surface 6: `221.53 ms`
- surface 7: `217.18 ms`
- surface 8: `226.21 ms`
- surface 9: `228.38 ms`
- surface 10: `231.57 ms`
- surface 11(image): `53.42 ms`

结论：

- 前 11 个折射面耗时非常接近，说明瓶颈不是某一个特殊表面
- tracing 主要是“每个折射面都要完整做一遍大规模求交 + 折射”
- 像面只有平面求交，无折射，因此明显更快

### 5.4 tracing 内部热点分布

对 `_trace_sag_surface()` 和 `_apply_surface_interaction()` 做同步插桩后：

- `sag_hit_ms = 1603.60`
- `interaction_ms = 800.88`
- `plane_hit_ms = 53.77`

可见主 trace 内部大致可分为：

- 几何求交相关：约三分之二
- 折射相互作用：约三分之一

进一步拆分后得到：

- `_standard_geometry_intersect()`：`857.72 ms`
- `_standard_geometry_normal()`：`286.21 ms`
- `_apply_refractive_interaction()`：`799.73 ms`

剩余时间主要来自：

- `local_hit`、`hit`、`valid` 等中间张量组装
- 孔径裁剪
- `torch.where` / `isfinite` / `clamp` / `sqrt` 等逐元素算子
- tracing 结束后的 `valid` 汇总

### 5.5 参数读取开销

对 tracing 中通过 Python 循环构造参数 tensor 的路径做粗测，得到：

- `surface_position()`：12 次，共约 `167.97 ms`
- `surface_value()`：55 次，共约 `98.39 ms`

说明：

- 这部分不是最大热点
- 但已经不是可以完全忽略的量级
- 后续若要继续提速，预打包 surface 参数是有意义的

注意：

- 上述逐面 / 逐函数耗时是通过显式同步插桩得到的粗测值
- 插桩本身会带来额外同步开销
- 因此这些数值适合判断热点顺序，不应用作最终 benchmark 报告

## 6. 是否还有加速空间

结论：有，而且空间不小，但优化方式应分层处理。

### 6.1 短期可做，收益明确

#### 6.1.1 design chunking

这是最优先项，主要解决显存问题，同时避免高密度采样时 OOM。

对速度的影响：

- 总时间可能略增，因为多了多次 launch
- 但在高显存压力场景下，chunking 反而更稳定，整体吞吐未必更差

#### 6.1.2 删掉未使用的大张量

重点是：

- `intensity`
- `opd`
- 一部分不必要的 spot 中间量

这是低风险优化，建议尽快做。

#### 6.1.3 标准几何 fast path 去掉 `stack/split`

当前双高斯场景全部是标准球面 + 像面，这类场景完全值得走更直接的实现路径。

### 6.2 中期可做，兼顾提速和可维护性

#### 6.2.1 trace 前一次性预打包 surface 参数

将每个 surface 的：

- `vertex_z`
- `radius`
- `conic`
- `aperture_radius`

提前准备好，trace 内部直接取 tensor。

这会减少：

- Python 层循环
- 重复 path 查找
- 反复创建小 tensor

#### 6.2.2 固定波长场景预计算折射率

当前 spot / PSF 分析中的波长集合是固定的。对每个 surface，可提前按波长计算：

- 入射介质折射率
- 出射介质折射率

trace 时直接广播使用，不必每个表面在大 batch 上重复调用材料接口。

### 6.3 长期可做，但当前不建议优先

#### 6.3.1 更激进的算子融合 / 自定义 CUDA 内核

理论上还可以继续提速，但问题是：

- 开发成本高
- 正确性验证成本高
- 不符合当前“尽快实现功能、代码保持简洁”的目标

在 spot / Huygens PSF 功能尚未完成前，不建议优先走这条路。

#### 6.3.2 改成 FP32

不建议。

原因：

- 当前项目约定优先使用 FP64
- spot / PSF 都需要和 Zemax 对标
- 精度风险大于当前收益

## 7. 推荐执行顺序

建议按下面顺序推进：

1. 给 `batch_spot.py` 增加 design chunking，先解决高密度采样显存问题
2. spot tracing 路径去掉未使用的 `intensity/opd`
3. 标准几何路径去掉 `stack/split` 大中间量
4. `save_path=None` 时压缩 spot 后处理的中间张量
5. 若还需要继续提速，再做 surface 参数预打包和折射率预计算

## 8. 下一步验证方式

每做一步优化，都建议固定用下面的场景回归：

```bash
python examples/batch_spot.py --device cuda:0 --surfaces 1 2 10 11 --ray-density 13
```

至少记录：

- `elapsed_ms`
- `valid_ray_count`
- CSV 中各视场 `RMS/GEO` 半径
- `torch.cuda.max_memory_allocated()`
- `torch.cuda.max_memory_reserved()`

判定标准：

- spot 结果与优化前一致
- 有效光线数一致
- 显存下降或相同显存下可支持更高 `ray_density`
- tracing 总耗时不出现明显回退

## 9. 最终判断

针对当前 V100 16G + `6561` 设计 + `3` 视场 + `3` 波长 + `547` pupil 光线的组合：

- `ray_density=13` 时显存接近 80% 是正常现象
- 继续提高采样密度后 OOM 也是符合当前实现特征的
- 当前 tracing 的绝对主耗时在逐面几何求交与折射计算
- 现阶段最值得做的不是大改架构，而是：
  - design chunking
  - 删除未使用大张量
  - 压缩标准球面路径中的中间张量

这三项都符合“快速实现功能、代码尽量精简、尽量复用现有能力”的目标。
