# tracing 模块拆分计划

## 1. 目标

这份文档回答四个直接问题：

1. `optics_core/tracing.py` 如果继续拆分，每个目标文件分别放什么内容。
2. Python 工程里，同一个类能不能拆到多个文件。
3. 拆分后是否还能保持 batch-first 设计。
4. 这里不再使用 `facade` 这个词，统一改叫“主入口类”。

本文的推荐方案是：

不要把 `SequentialSurfaceRayTracer` 这个类本体拆散到多个文件；保留它作为 tracing 的主入口类，把内部私有算法按职责拆到多个内部模块。

这样做有三个好处：

1. 公开 API 基本不变。
2. 阅读入口仍然只有一个类，不会增加心智负担。
3. batch-first 的张量契约不会因为文件拆分而被打散。

## 2. 当前 tracing.py 实际承载的职责

当前 [optics_core/tracing.py](optics_core/tracing.py) 里混在一起的内容，实际上已经自然分成 5 类：

1. 公开接口与主流程：`SequentialSurfaceRayTracer.trace()`、`batched_trace()`、`_trace_surfaces()`。
2. surface 分发：`_trace_surface_local()`、`_trace_paraxial_surface()`、`_trace_plane_surface()`、`_trace_sag_surface()`，以及 coordinate break 的未实现分支。
3. sampled tracing 输入构造：`_build_input_rays_from_sample()` 以及 pupil/field/broadcast helper。
4. 命中 kernel：`SurfaceHit`、`_plane_hit()`、`_sag_surface_hit()`、`_record_surface_hit()`。
5. 交互 kernel：介质查询、方向归一化、反射、折射。

此前未使用的 `_propagate()` 和 `_edge_field_value()` 已经可以直接清理掉，不再作为拆分对象。

## 3. 推荐拆分原则

### 3.1 原则一：类不拆，职责拆

推荐保留：

1. `SequentialSurfaceRayTracer`

这个公开类继续放在主入口文件里。

不推荐的做法：

1. 把 `SequentialSurfaceRayTracer` 的一部分方法复制到另一个文件里，再通过运行时注入回类里。
2. 用 monkey patch 在 import 时给类动态挂方法。
3. 为了“一个文件一个功能”而强行引入很多 mixin。

这些做法虽然技术上能做，但会让类型提示、跳转、测试定位和维护成本都变差。

### 3.2 原则二：先保持 import 路径稳定

第一阶段拆分时，建议仍保留：

1. `optics_core/tracing.py`
2. `from .tracing import SequentialSurfaceRayTracer`

也就是说，先不急着把 `tracing.py` 改成一个包目录，而是先把复杂的私有逻辑拆到内部模块，例如：

1. `optics_core/_tracing_types.py`
2. `optics_core/_tracing_sampled_rays.py`
3. `optics_core/_tracing_dispatch.py`
4. `optics_core/_tracing_hits.py`
5. `optics_core/_tracing_interactions.py`

这样风险最小。

### 3.3 原则三：只按“计算职责”拆，不按“类方法数量”拆

tracing 是一条数值链路，拆分应该围绕：

1. 谁负责逐面调度。
2. 谁负责构造显式光线输入。
3. 谁负责求交。
4. 谁负责局部交互。

而不是围绕“这个类的方法太多了，所以平均分到几个文件”。

## 4. 推荐目标文件与内容分配

下面是建议的第一阶段目标结构。

### 4.1 optics_core/tracing.py

角色：公开主入口类。

这个文件只保留公开类和高层编排，不再承载具体求交和交互公式。

建议保留内容：

1. `SequentialSurfaceRayTracer`
2. `trace()`
3. `batched_trace()`
4. `_trace_surfaces()`
5. `_surface_bounds()`
6. `_surface_indices()`

像 `_trace_backward()`、`_trace_surface()` 这种只提供过渡分层价值、但已经没有实际调用意义的 helper，可以在拆分前先清理掉。

这个文件应该回答的唯一问题是：

“这次 trace 从哪里开始，按什么顺序逐面推进，以及何时调用下层模块。”

它不应该继续直接展开球面求交、折射公式、pupil 坐标整形等实现细节。

### 4.2 optics_core/_tracing_types.py

角色：tracing 内部共享数据结构。

建议放入：

1. `SurfaceHit`

后续如果需要引入 `TraceRuntime`、`TraceState`、`TraceSlice` 之类的执行态对象，也适合先落在这里。

这样做的意义是把“内部数据结构”从“主入口类”里拔出来，避免主入口文件既定义流程、又定义数据载体。

### 4.3 optics_core/_tracing_sampled_rays.py

角色：sampled tracing 的输入构造模块。

建议放入：

1. `_build_input_rays_from_sample()`
2. `_pupil_coordinate_tensors()`
3. `_field_slopes()`
4. `_angle_to_slope()`
5. `_broadcast_batch_value()`

像 `_edge_field_value()` 这种已经确认无调用的 helper，建议在拆分前直接删除，不再进入模块拆分清单。

这个模块的职责是：

把 sampler 和 aimer 给出的结果，转换成真正送入 `trace()` 的 `RayBundle`，并保持 `design x field x wavelength x ray` 的 batch 组织。

### 4.4 optics_core/_tracing_dispatch.py

角色：surface class 分发与局部 surface 处理。

建议放入：

1. `_trace_surface_local()`
2. `_trace_paraxial_surface()`
3. `_trace_plane_surface()`
4. `_trace_sag_surface()`

`ObjectSurface`、`ImageSurface` 这种只映射到平面 pass 面的场景，不需要额外保留一层 `_trace_*_surface()` 转发 helper。

`CoordinateBreak` 当前还没有真实 kernel，建议先把未实现分支直接留在 `_trace_surface_local()` 里，等坐标变换内核落地后，再决定是否拆成独立函数。

这个模块只负责回答一个问题：

“当前 surface 是什么类型，应该走哪条局部处理链路。”

它不负责：

1. 逐面循环。
2. sampled ray 构造。
3. 介质折射率查询实现。
4. 具体 plane/sag 求交公式。

### 4.5 optics_core/_tracing_hits.py

角色：命中点求解与交点记录。

建议放入：

1. `_surface_vertex_z()`
2. `_plane_hit()`
3. `_sag_surface_hit()`
4. `_record_surface_hit()`

像 `_propagate()` 这种已经确认无调用、但又没有承接当前主链职责的 helper，建议先直接删除；如果后续真实 segment propagation 重新进入实现，再按真实调用点重建专门的传播函数。

这个模块的职责是：

给定当前 ray state 和当前 surface，返回一个统一的 `SurfaceHit`，让上层 dispatch 和 interaction 不必知道 plane 和 sag 的差异细节。

### 4.6 optics_core/_tracing_interactions.py

角色：局部相互作用与方向处理。

建议放入：

1. `_travel_direction()`
2. `_store_travel_direction()`
3. `_normalize_direction()`
4. `_surface_media()`
5. `_apply_surface_interaction()`
6. `_apply_reflective_interaction()`
7. `_apply_refractive_interaction()`

像 `_surface_placeholder()` 这种只负责包装 `NotImplementedError` 的 helper，通常可以直接删掉，把异常信息内联到分支里，阅读成本更低。

这个模块负责：

1. 前向/后向下 travel direction 的统一处理。
2. 反射与折射的局部物理更新。
3. 介质查询和 TIR 回退逻辑。

也就是说，命中点在哪里由 `_tracing_hits.py` 决定，命中之后方向怎么改由 `_tracing_interactions.py` 决定。

## 5. 为什么不建议“把同一个类拆到多个文件”

### 5.1 Python 原生没有 partial class

像某些语言那样，把同一个类定义天然拆成多个文件，Python 没有这个机制。

你在 Python 里看到的“同一个类来自多个文件”，通常只有三种实现手法：

1. 多继承 + mixin
2. 运行时给类挂方法
3. 代码生成

这些方式都不是当前 tracing 模块最合适的第一选择。

### 5.2 这个项目里更合适的做法

在本项目里，更推荐的模式是：

1. `SequentialSurfaceRayTracer` 继续作为唯一主入口类。
2. 复杂私有逻辑变成同包内部的函数模块。
3. 主入口类只负责组织调用，不再自己承载所有算法细节。

换句话说，不是“把一个类拆到多个文件”，而是“让一个类依赖多个内部模块”。

这是 Python 工程里更自然、也更稳的组织方式。

## 6. 拆分后是否还能保持 batch-first

可以，而且应该更容易保持。

关键点不在“一个类还是多个文件”，而在“张量契约是否被保持”。

### 6.1 拆分后必须保持的 batch 契约

以下约束在拆分后必须保持不变：

1. sampled tracing 的主 batch 维仍是 `design x field x wavelength x ray`。
2. `trace()` 接收的显式光线分量仍是 FP64 tensor。
3. surface-local helper 不引入按 ray 的 Python for-loop。
4. `surface_value()`、`surface_position()` 仍只做只读查询，不回流到高层 convenience 接口。
5. 不引入额外的 CPU-GPU 来回搬运。

### 6.2 哪些循环仍然是合理的

拆分后允许继续保留的 Python 级循环只有两类：

1. 逐 surface 的顺序循环。
2. 未来真实非球面求交中，固定次数的小迭代循环。

不应该引入的循环：

1. 逐 design 循环。
2. 逐 field 循环。
3. 逐 wavelength 循环。
4. 逐 ray 循环。

### 6.3 为什么拆分反而有利于 batch-first

拆分后，每个内部模块的职责会更单纯：

1. sampled ray 构造模块专注 shape 和 broadcast。
2. hit 模块专注 tensor 求交。
3. interaction 模块专注 tensor 方向更新。

这样更容易检查有没有人在局部偷偷退回标量思维或对象遍历。

## 7. 建议的分步实施顺序

建议按下面顺序拆，风险最低。

### 第一步：先拆 sampled ray 构造

先把下面几个函数搬走：

1. `_build_input_rays_from_sample()`
2. `_pupil_coordinate_tensors()`
3. `_field_slopes()`
4. `_angle_to_slope()`
5. `_broadcast_batch_value()`

原因：

1. 这部分和 surface local kernel 耦合最弱。
2. 比较容易独立测试。
3. 可以先验证 batch shape 契约是否完整保留。

### 第二步：再拆 hit 与 interaction kernel

第二批迁移：

1. `SurfaceHit`
2. `_plane_hit()`
3. `_sag_surface_hit()`
4. `_record_surface_hit()`
5. `_travel_direction()`
6. `_store_travel_direction()`
7. `_normalize_direction()`
8. `_surface_media()`
9. `_apply_surface_interaction()`
10. `_apply_reflective_interaction()`
11. `_apply_refractive_interaction()`

原因：

1. 这部分已经是比较纯的 tensor kernel。
2. 一旦独立出来，真实球面、真实非球面、TIR、坐标间断等后续扩展都会更清楚。

### 第三步：最后拆 surface dispatch

最后再搬：

1. `_trace_surface_local()`
2. 各个 `_trace_*_surface()`

原因：

1. 它是连接 orchestration 和 kernel 的中间层。
2. 等前两层先稳定后，再搬 dispatch，改动面最小。

## 8. 第一阶段拆分后，tracing.py 应该长什么样

理想状态下，`optics_core/tracing.py` 应该像一个“主入口类 + 调度壳”文件，而不是一个“大而全实现文件”。

拆分后它应该主要保留：

1. 抽象接口定义。
2. `trace()` 和 `batched_trace()` 的高层流程。
3. surface range 和顺序控制。
4. 对内部 helper 模块的调用。

也就是说，别人打开这个文件时，应该先看见 tracing 的主流程，而不是先陷进折射公式和 plane/sag 求交细节。

## 9. 后续可选的第二阶段优化

如果第一阶段拆完后，内部模块仍继续增长，再考虑第二阶段动作：

1. 把 `optics_core/tracing.py` 升级为 `optics_core/tracing/` 包。
2. 把内部模块转入包目录，例如 `optics_core/tracing/interactions.py`。
3. 在包内逐步引入 `TraceRuntime` 或 `TraceState` 之类的执行态对象。

但这不是第一步必须做的事情。

对当前项目，更稳妥的路径是：

先做“单文件拆成若干内部模块”，等职责真正稳定后，再决定是否包化。

## 10. 拆分实施时的验证建议

每完成一小步迁移，都建议至少跑下面几组测试：

1. `tests/regression/test_backward_trace_and_entrance_pupil.py`
2. `tests/contract/test_surface_trace_kernels.py`
3. `tests/contract/test_public_api_contracts.py`

如果要做一次较完整的非 Zemax 回归，再补：

1. `pytest -m "not zemax" --ignore=tests/benchmark/test_demo_pipeline_benchmark.py -q`

验证重点不是“文件拆了没有报 import 错误”，而是：

1. batch shape 有没有变化。
2. backward entrance pupil 链路有没有断。
3. SphereSurface 和 EvenAsphereSurface 当前共享的真实面主链有没有回归。
4. 公开导出和 import 路径有没有被破坏。

## 11. 结论

最推荐的方案不是“把一个类分尸到多个文件”，而是：

1. 保留 `SequentialSurfaceRayTracer` 作为 tracing 的主入口类。
2. 把 sampled ray 构造、surface 分发、hit kernel、interaction kernel 拆成内部模块。
3. 第一阶段先保持 `optics_core/tracing.py` 这个公开模块路径不变。
4. 用“主入口类”这个词代替 `facade`。

如果后续要真正进入更大规模的 tracing 重构，下一层自然演进就是把这些内部模块继续收敛到 `TraceRuntime` 驱动的执行态设计里，但那应当放在本轮拆分之后。
