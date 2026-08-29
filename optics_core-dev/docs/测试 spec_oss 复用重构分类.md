# 测试 spec / oss 复用重构分类

## 目标

后续 Zemax 对标型单元测试，尽量统一到下面这条路径：

1. 测试本体先 `load_zmx_sequential_system_spec(...)`
2. 基于同一个 `spec` 构建 `optics_core system`
3. 如需 Zemax 参考值，尽量在同一个 `with loaded_sequential_system(spec.zmx_path) as oss:` 中完成
4. Zemax helper 优先接收 `spec`，必要时再接收 `oss`

这样主要减少两类重复：

- 重复解析 `zmx -> spec`
- 重复执行 `oss.load(...)`


## 分类

### A 类：`spec` 未复用，`oss` 未复用

特征：

- 测试本体自己先 `load_zmx_sequential_system_spec(...)`
- Zemax helper 内部又再次 `load_zmx_sequential_system_spec(...)`
- Zemax helper 内部自己 `loaded_sequential_system(...)`

这是当前最常见、也最值得优先整理的一类。

当前文件：

- [tests/regression/test_layout_2d_against_zemax.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\regression\test_layout_2d_against_zemax.py:30)
- [tests/regression/test_spot_diagram_against_zemax.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\regression\test_spot_diagram_against_zemax.py:28)
- [tests/regression/test_material_refractive_index_against_zemax.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\regression\test_material_refractive_index_against_zemax.py:21)
- [tests/regression/test_clear_aperture_against_zemax.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\regression\test_clear_aperture_against_zemax.py:29)
- [tests/regression/test_single_sphere_sag_against_zemax.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\regression\test_single_sphere_sag_against_zemax.py:38)
- [tests/regression/test_basic_paraxial_focus.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\regression\test_basic_paraxial_focus.py:106)
- [tests/regression/test_spherical_extreme_pupil_trace_against_zemax.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\regression\test_spherical_extreme_pupil_trace_against_zemax.py:30)
- [tests/regression/test_spherical_forward_trace_against_zemax.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\regression\test_spherical_forward_trace_against_zemax.py:40)
- [tests/regression/test_spherical_forward_trace_against_zemax.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\regression\test_spherical_forward_trace_against_zemax.py:248)

建议：

- 先把这类 helper 改成 `fetch_xxx(spec, oss, ...)` 或 `fetch_xxx(spec, ...)`
- 测试本体统一持有 `spec`


### B 类：`spec` 已复用，`oss` 未复用

特征：

- 测试本体已经持有 `spec`
- 但 Zemax helper 仍然自己打开一次 `loaded_sequential_system(...)`

当前代码里这类不算单独很多，因为大多数 helper 还停留在 A 类。

建议：

- 当 A 类 helper 先改成接收 `spec` 后，这一类通常会自然出现
- 下一步再继续把 `oss` 也上提到测试本体


### C 类：`spec` 已复用，`oss` 也复用

特征：

- 测试本体先 `load spec`
- 测试本体自己 `with loaded_sequential_system(...) as oss`
- 后续 direct ray、spot、surface property 等读取都复用同一个 `oss`

这是推荐的标准形态。

当前代表：

- [tests/benchmark/test_double_gauss_zmx_batch_ray_trace_benchmark.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\benchmark\test_double_gauss_zmx_batch_ray_trace_benchmark.py:31)
- [tests/regression/test_spherical_forward_trace_against_zemax.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\regression\test_spherical_forward_trace_against_zemax.py:136)

说明：

- 这类测试结构最清楚
- 也是后续系统级 Zemax 对标测试应优先模仿的模板


### D 类：混合型，局部重复最重

特征：

- 测试本体先 `load spec`
- 先调用一个内部会再次 `load spec + load oss` 的 helper
- 然后测试本体自己又额外 `loaded_sequential_system(...)`

当前最典型：

- [tests/regression/test_spherical_forward_trace_against_zemax.py](C:\Users\Huwei\Project\auto_od_web\worker\auto_od\GYOptics\optics_core\tests\regression\test_spherical_forward_trace_against_zemax.py:195)

这类测试是当前最冗余的一类，应优先处理。


## 推荐整理顺序

1. 先整理 D 类  
   先把最重的重复点拿掉，收益最大。

2. 再整理 A 类  
   先消掉 helper 内部重复 `load spec`。

3. 最后把 A/B 类统一向 C 类收敛  
   形成一套固定模板。


## 推荐模板

系统级 Zemax 对标测试后续尽量写成：

```python
spec = load_zmx_sequential_system_spec(zmx_path)
system = build_optics_core_system_from_zmx_spec(spec)

with loaded_sequential_system(spec.zmx_path) as oss:
    reference = fetch_xxx(spec, oss, ...)
    extra_data = fetch_yyy(spec, oss, ...)

result = run_optics_core(system, ...)
assert_compare(result, reference, extra_data)
```


## 第一批建议整改文件

- `tests/regression/test_spherical_forward_trace_against_zemax.py`
- `tests/regression/test_layout_2d_against_zemax.py`
- `tests/regression/test_spot_diagram_against_zemax.py`
- `tests/regression/test_material_refractive_index_against_zemax.py`
- `tests/regression/test_clear_aperture_against_zemax.py`
