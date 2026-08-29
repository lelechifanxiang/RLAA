# design_batch_view 准备态切片重构计划

## 背景

`design_batch_view(start, stop)` 的目标是为 PSF、MTF 等批量分析提供连续 design 的轻量系统视图。它应该共享原系统的大部分结构和准备态 tensor，仅改变当前视图可见的 design 范围。

当前实现已经基本满足性能目标：参数向量通过 `ParameterVectorBatchRange` 引用父批量，`FrameData`、`FirstOrderData`、`ClearApertureResult` 的 tensor 切片也共享底层 storage，不会复制大 tensor。

但代码可读性存在坏味道：`MultiOpticalSystem.design_batch_view()` 需要直接了解多个准备态对象的内部字段，并在 `system.py` 中维护手工切片逻辑。

```python
view.frame_data = FrameData(...)
view.first_order_data = FirstOrderData(**{...})
view.clear_aperture_data = self._slice_clear_aperture_data(start, stop)
```

这说明“按 design 维度生成视图”没有成为准备态数据对象自身的一等能力。

## 目标

1. 提高 `design_batch_view()` 的可读性。
2. 减少 `system.py` 中针对不同准备态对象的冗余适配代码。
3. 保持当前零拷贝设计，不引入额外 CPU-GPU 数据搬运。
4. 避免过度封装，不新增复杂协议、基类或通用切片框架。

## 非目标

1. 不重构整个参数系统。
2. 不改变 `design_view(index)` 的行为；单设计物化视图仍然可以深拷贝 surface。
3. 不为标量、list、tuple 等输入增加额外兼容逻辑。
4. 不引入用户可见的新接口。

## 当前问题

### 1. system.py 了解过多准备态细节

`FrameData`、`FirstOrderData`、`ClearApertureResult` 都是以 design 为第 0 维的准备态数据，但它们没有统一的局部切片方法，导致 `MultiOpticalSystem` 必须知道每个对象有哪些字段、哪些字段需要切片、哪些字段应该共享。

这会让 `system.py` 逐渐变成 prepare 缓存的适配中心。

### 2. 新增准备态缓存时容易继续堆适配代码

如果后续新增某种缓存，例如像面参考点、光瞳参考量或 PSF/MTF 预计算缓存，开发者很容易继续在 `design_batch_view()` 中补一段专用切片逻辑。

这会让函数越来越长，也更难判断哪些数据共享、哪些数据切片、哪些数据丢弃。

### 3. ClearApertureResult 的切片规则放错位置

`ClearApertureResult.trace_result` 在 batch view 中不参与后续分析，当前通过 `_slice_clear_aperture_data()` 丢弃。

这条规则属于 `ClearApertureResult` 自身，而不应该由 `MultiOpticalSystem` 单独维护。

## 推荐方案

为所有“第 0 维是 design 维”的准备态数据类增加一个简单方法：

```python
def design_slice(self, start: int, stop: int) -> Self:
    ...
```

该方法只负责返回当前对象的连续 design 视图。内部使用 tensor 切片，因此共享底层 storage，不复制大 tensor。

### FrameData

```python
def design_slice(self, start: int, stop: int) -> FrameData:
    return FrameData(
        rotations=self.rotations[start:stop],
        origins=self.origins[start:stop],
        device=self.device,
    )
```

### FirstOrderData

推荐显式列出字段，而不是使用 `__dataclass_fields__` 自动遍历。

虽然自动遍历代码更短，但一阶量字段较少，显式写法更容易阅读，也更符合当前项目“避免过度魔法”的风格。

```python
def design_slice(self, start: int, stop: int) -> FirstOrderData:
    return FirstOrderData(
        ttl=self.ttl[start:stop],
        effl=self.effl[start:stop],
        working_f_number=self.working_f_number[start:stop],
        entrance_pupil_z=self.entrance_pupil_z[start:stop],
        entrance_pupil_radius=self.entrance_pupil_radius[start:stop],
        stop_radius=self.stop_radius[start:stop],
        exit_pupil_z=self.exit_pupil_z[start:stop],
        exit_pupil_radius=self.exit_pupil_radius[start:stop],
    )
```

### ClearApertureResult

```python
def design_slice(self, start: int, stop: int) -> ClearApertureResult:
    return ClearApertureResult(
        semi_diameter=self.semi_diameter[start:stop],
        valid=self.valid[start:stop],
        surface_indices=self.surface_indices,
        trace_result=None,
    )
```

`trace_result=None` 是有意行为：clear aperture 的 trace 细节不参与 PSF/MTF 后续分析，保留它会增加对象引用关系，也不利于释放中间结果。

## design_batch_view 修改后形态

修改后，`design_batch_view()` 中准备态切片部分应简化为：

```python
view.frame_data = self.frame_data.design_slice(start, stop)
view.first_order_data = self.first_order_data.design_slice(start, stop)
view.clear_aperture_data = (
    None if self.clear_aperture_data is None
    else self.clear_aperture_data.design_slice(start, stop)
)
```

随后可以删除 `MultiOpticalSystem._slice_clear_aperture_data()`。

## 数据拷贝和 CPU-GPU 搬运分析

`design_batch_view()` 不应该带来大规模数据拷贝，也不应该触发 CPU-GPU 数据搬运。

当前主要对象的行为如下：

| 对象 | 当前行为 | 是否复制大数据 |
| --- | --- | --- |
| `ParameterVectorBatchRange` | 引用父参数批量，通过索引读取原参数向量 | 否 |
| `SurfaceSequence.bind()` | 新建 `SurfaceSequence` 包装对象，共享 `_items` 列表 | 否 |
| `FrameData.rotations/origins[start:stop]` | tensor view，共享 storage | 否 |
| `FirstOrderData` 各字段 `[start:stop]` | tensor view，共享 storage | 否 |
| `ClearApertureResult.semi_diameter/valid[start:stop]` | tensor view，共享 storage | 否 |
| `MultiOpticalSystem` view | 新建 Python 包装对象 | 是，小型 CPU 对象 |

因此，`design_batch_view()` 的额外开销主要是少量 Python 包装对象创建。准备态 tensor 不会被 clone，不会从 GPU 拷贝到 CPU，也不会从 CPU 拷贝到 GPU。

需要注意的是：tensor 切片是否零拷贝，前提是使用普通切片 `start:stop`。如果未来改成高级索引、布尔 mask 或显式 `clone()`，就可能产生真实拷贝。因此 `design_batch_view()` 应继续只支持连续 design 区间。

## 实施步骤

1. 在 `optics_core/system_state.py` 中为 `FrameData` 和 `FirstOrderData` 增加 `design_slice()`。
2. 在 `optics_core/apertures.py` 中为 `ClearApertureResult` 增加 `design_slice()`。
3. 简化 `MultiOpticalSystem.design_batch_view()` 中的准备态切片逻辑。
4. 删除 `MultiOpticalSystem._slice_clear_aperture_data()`。
5. 保留现有 contract 测试，确认 batch view 仍共享准备态 tensor 的 storage。

## 验证重点

1. `view.parameters[0] is system.parameters[start]` 仍成立。
2. `view.frame_data.rotations` 与原 tensor 共享 storage。
3. `view.first_order_data.working_f_number` 与原 tensor 共享 storage。
4. `view.clear_aperture_data.semi_diameter` 与原 tensor 共享 storage。
5. PSF/MTF minibatch 结果与 full batch 结果一致。

## 结论

这次重构不需要引入复杂架构。最合适的做法是建立一个轻量约定：

> 凡是第 0 维表示 design 的准备态数据对象，都由对象自身提供 `design_slice(start, stop)`。

这样可以把切片规则放回数据对象本地，让 `MultiOpticalSystem` 只负责组装系统视图，减少冗余适配代码，同时保持现有零拷贝、高可读性的 batch-first 设计。
