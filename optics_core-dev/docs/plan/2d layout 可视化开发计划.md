# 2D Layout 可视化开发计划

## 目标

面向单个 `design_view` 生成固定 `yz` 平面的 2D layout 图，显示：

- 每一面的镜片轮廓
- 参与显示的视场光线
- 光线在各面的交点连线

当前阶段坚持精简实现：

- 只支持单个 `design_view`
- 只画 `yz` 平面
- 只支持 `x=0` 的视场
- 每个视场固定采样 7 根光线
- 使用 `matplotlib`
- 不预留过多通用接口

---

## 对需求流程的判断

你给出的流程总体合理，可以直接采用。结合当前项目结构，建议做两点收敛：

1. “确保镜片净口径已计算”建议由 layout 内部直接完成  
   原因：当前 `calculate_clear_apertures(...)` 是只读结果，不写回 `surface.aperture_radius`。  
   因此 layout 最简单的做法是内部先调用一次净口径计算，再直接使用返回值绘制，不新增“先准备再绘图”的外部状态依赖。

2. 第 2 步的“筛选 `x=0` 视场”建议改成显式过滤 + 显式提示  
   规则固定为：保留 `field.x == 0` 的视场；若其余视场存在，在结果对象或日志中提示“已过滤”；若一个都没有，直接报错。

参考 `reference/ParaOptics.py`，其思路也是：

- 先画表面轮廓
- 再追迹选定光线
- 最后把追迹结果叠加到 2D 图上

所以当前项目不需要额外发明新流程。

---

## 推荐架构

建议采用“成员入口 + 独立实现”的现有风格：

- 对外入口：`system.design_view(i).analysis.layout_2d().run(...)`
- 具体实现：新增独立模块 `optics_core/layout_2d.py`

这样有几个好处：

- 用户入口统一挂在 `analysis`
- 算法细节不塞进 `analysis.py`
- 便于后续单测内部函数

建议新增以下最小结构。

### 1. analysis 层

在 `optics_core/analysis.py` 中新增：

- `Layout2DSettings`
- `Layout2DResult`
- `Layout2D`
- `AnalysisHub.layout_2d()`

其中 `Layout2DResult` 只保留必要信息：

- `filtered_field_indices`
- `filtered_field_points`
- `trace_result`
- `clear_aperture_result`
- `figure`
- `axes`
- `save_path`
- `message`

不做更多抽象。

### 2. 计算/绘图层

新增 `optics_core/layout_2d.py`，放以下函数：

- `filter_layout_fields(system)`
- `build_layout_sampler()`
- `trace_layout_rays(system, fields)`
- `sample_surface_profile(system, surface_index, semi_diameter)`
- `plot_layout_2d(system, trace_result, clear_aperture_result, filtered_fields, save_path=None)`
- `run_layout_2d(system, settings)`

其中固定策略如下：

- 视场过滤：只保留 `field.x == 0`
- pupil 采样：固定 7 点 `[(0,-1), (0,-2/3), ..., (0,1)]`
- 表面采样：固定在 `y` 方向均匀采样轮廓点，`z = surface_position + sag(0, y)`
- 坐标系：横轴 `z`，纵轴 `y`

---

## 推荐实现流程

### 阶段 1：最小工作流打通

实现 `Layout2D.run()`：

1. 检查 `system.system_count == 1`
2. 过滤出 `x=0` 视场
3. 内部调用 `calculate_clear_apertures(...)`
4. 构造固定 7 光线 sampler
5. 对过滤后的视场执行正向追迹，并记录交点
6. 使用 `matplotlib` 绘制镜片和光线
7. 可选保存 PNG

### 阶段 2：镜片轮廓绘制

对每个 surface：

1. 从 `clear_aperture_result.semi_diameter` 读取该面的半口径
2. 在 `[-semi_diameter, semi_diameter]` 上采样 `y`
3. 用 `surface.geometry.sag(x=0, y)` 计算矢高
4. 用 `surface_position(...) + sag` 得到全局 `z`
5. 绘制 `(z, y)` 曲线

当前阶段可只支持：

- `SphereSurface`
- `ImageSurface`
- `ObjectSurface`

遇到暂不支持的表面类型直接报 `NotImplementedError`。

### 阶段 3：光线绘制

对 `TraceResult.intersections`：

1. 提取每条光线在每一面的 `(z, y)` 交点
2. 按光线编号连接成折线
3. 每个视场使用固定颜色

注意：

- `yz` 图只接受 `pupil_x = 0`
- 因此 pupil 采样固定为沿 `y` 轴的 7 根光线

---

## 单元测试计划

### 1. 回归测试：交点与 Zemax 对齐

新增一个 layout 回归测试，复用四面结构：

- `tests/zemax/zmx_files/four_surface_spherical.zmx`

测试规则：

1. 只保留 `x=0` 的视场  
   在当前四面结构下，应保留：
   - `(0, 0)`
   - `(0, -3)`

2. pupil 固定为 7 点：
   - `(0, -1.0)`
   - `(0, -2/3)`
   - `(0, -1/3)`
   - `(0, 0.0)`
   - `(0, 1/3)`
   - `(0, 2/3)`
   - `(0, 1.0)`

3. 通过现有 Zemax helper 获取参考交点  
   直接复用 `fetch_zemax_spherical_forward_trace(...)`，只传上述视场对应的 pupil 坐标组合即可，无需另起复杂架构。

4. 对比每一面的 `y/z` 交点与 optics_core 的 layout trace 结果是否一致

### 2. 图片导出测试

同一个测试中额外导出 PNG：

- 建议固定输出到 `tests/output/layout_2d_four_surface.png`
- 测试中打印导出路径，供人工检查

检查项：

- 文件存在
- 文件大小大于 0

不做像素级黄金图比较，当前阶段人工检查足够。

---

## 需要同步调整的现有代码

1. `optics_core/analysis.py`
   - 增加 layout 入口

2. `optics_core/__init__.py`
   - 导出 `Layout2D` / `Layout2DResult`

3. `tests/zemax`
   - 不需要新增复杂 builder
   - 复用现有 `fetch_zemax_spherical_forward_trace(...)`

4. `tests/fixtures`
   - 如有需要，只增加一个固定 7 点 pupil sampler helper
   - 不新增额外 case 层

---

## 不做的内容

当前阶段明确不做：

- 多 design 一次性绘图
- 任意截面选择
- 自动支持任意 field 过滤规则
- 自定义光线数
- 自定义颜色/样式系统
- 非球面专门绘图适配
- 像差扇形图、spot diagram 等附加图
- 黄金图像素回归

---

## 完成标准

满足以下条件即可认为本阶段完成：

1. 可以对单个 `design_view` 成功生成 `yz` layout 图
2. 非 `x=0` 视场会被过滤并给出提示
3. 每个保留视场固定绘制 7 根光线
4. 四面结构 layout 交点与 Zemax 对齐
5. 测试会导出 PNG 文件供人工检查
6. 代码仍保持“小入口 + 少量独立函数”的精简风格
