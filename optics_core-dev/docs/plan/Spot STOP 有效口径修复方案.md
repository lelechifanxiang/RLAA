# Spot STOP 口径语义重构方案

> 状态：已完成实现，并通过五文件 Spot、Cooke Layout、固定口径和全量回归验证。

## 1. 结论

当前 `max(surface_radius, first_order.stop_radius)` 可以通过已有 Spot 和 Layout 回归，但它是兼容性近似，不是 Zemax 的完整口径语义。

五个异构 ZMX 文件的真正问题不是 Zemax 同时设置了两个互相矛盾的硬孔径，而是加载器把下面两个概念合并成了 `aperture_type="fixed"`：

- `DIAM ... 1`：Clear Semi-Diameter/Semi-Diameter 的 solve 为 Fixed；
- `FLAP`、`CLAP` 等：真正参与渐晕判断的 Surface Aperture。

通过 ZOS-API 检查，五个文件的 STOP 均为：

```text
SemiDiameter Solve = Fixed
Surface Aperture    = None
```

因此 Zemax 不会在这些 STOP 面按 `DIAM` 硬裁剪。系统 F/# 负责确定入瞳和发射光束，当前加载器额外制造了一个不存在的硬孔径。

## 2. Zemax 口径模型

### 2.1 System Aperture

F/#、Entrance Pupil Diameter 等属于系统孔径，负责确定入瞳并发射光线。只有 `Float By Stop Size` 反过来使用 STOP Semi-Diameter 定义系统孔径。

### 2.2 Semi-Diameter

Semi-Diameter 描述表面尺寸；Automatic/Fixed 是该数据列的 solve 状态。Fixed 本身不应在核心模型中直接等价为硬裁剪。

### 2.3 Surface Aperture

Surface Aperture 才决定光线是否被渐晕：

- `None`：不按 Semi-Diameter 裁剪；
- `Floating Aperture`：最大半径始终等于当前 Semi-Diameter；
- `Circular Aperture`：按显式最小/最大半径裁剪。

普通非 dummy 面把 Semi-Diameter 改为 Fixed 时，OpticStudio 通常会自动添加 Floating Aperture，所以日常使用时容易误以为“Fixed 就是硬孔径”。dummy 面不会自动得到这一语义；五个文件的 STOP 正属于这种情况。

## 3. 当前实现的问题

`zmx_loader.py` 当前按 `DIAM` 的 solve 状态生成 `aperture_type`：

```python
aperture_type = "fixed" if diameter_mode == 1 else "auto"
```

随后 `_hits.py` 将所有 `fixed` 面当作圆形硬孔径。这会产生三个问题：

1. 无法表示“Fixed Semi-Diameter + Surface Aperture None”；
2. 忽略 ZMX 中已经保存的 `FLAP` 等真实口径信息；
3. 迫使追迹层通过 STOP 特判和取较大值猜测 Zemax 行为。

## 4. 重构方案

### 4.1 分离表面尺寸和追迹口径

将 `Surface` 中当前混合的字段拆分为：

```python
semi_diameter: Scalar | None
semi_diameter_solve: Literal["auto", "fixed"]
aperture_type: Literal["none", "floating"]
```

本阶段只支持 Zemax `None` 和 `Floating Aperture`。Floating Aperture 的追迹半径直接复用 `semi_diameter`，无需再保存一份重复半径。Circular Aperture 等类型留待实际需求出现后扩展。

### 4.2 按 ZMX 原始语义加载

- `DIAM` 只设置 `semi_diameter` 和 `semi_diameter_solve`；
- `FLAP` 设置 `aperture_type="floating"`；
- 没有 Surface Aperture 记录时保持 `aperture_type="none"`；
- 不在加载器中重新推断 dummy 面规则，以 ZMX 已保存的数据为准。

异构 batch 中 Semi-Diameter 仍可作为逐设计参数；Surface Aperture 类型属于共享拓扑。

### 4.3 简化追迹层

追迹时只判断 Surface Aperture：

```text
none      -> 不裁剪
floating  -> 按逐设计 semi_diameter 裁剪
```

删除 STOP 的 `max(surface_radius, first_order.stop_radius)` 特判。System Aperture 只参与入瞳和光线构造，Surface Aperture 只参与逐面渐晕，两者不再在 `_hits.py` 中混合。

保留当前圆孔边界的 FP64 相对裕量，它解决的是边界舍入问题，与口径来源无关。

### 4.4 调整其他使用位置

- `first_order.py` 的 `Float By Stop Size` 从 STOP `semi_diameter` 读取系统孔径；
- 自动净口径计算只更新 `semi_diameter_solve="auto"` 的表面尺寸；
- Layout 使用 `semi_diameter` 绘制表面，不依赖其是否裁剪光线；
- 参数 schema 中的 `surface[*].aperture_radius` 改为 `surface[*].semi_diameter`。

不保留旧字段的长期兼容转发，避免继续维持含义模糊的接口。

## 5. 验证方案

不新增测试文件，在现有测试中完成验证：

1. `test_heterogeneous_batch_against_zemax.py`：五个 Fixed Semi-Diameter、Aperture None 的 STOP 不硬裁剪，Spot RMS 与 Zemax 误差不超过 `1e-3 μm`；
2. `test_zmx_text_loader.py`：分别确认五文件 STOP 为 `fixed + none`、Cooke STOP 为 `fixed + floating`；
3. 现有 surface trace kernel：确认 Floating Aperture 仍会裁剪超出口径的光线；
4. 保留 Double Gauss、Cooke Layout 和负厚度回归；
5. 运行 `batch_spot_multiple_zmx.py --design-count 10 --device cpu` 和全量 pytest。

## 6. 完成标准

- ZMX 导入后 Semi-Diameter solve 与 Surface Aperture 信息不再混淆；
- 五文件 Spot RMS 与 Zemax 保持一致；
- Cooke 等带 `FLAP` 的表面继续产生正确渐晕；
- FLOA 仍由 STOP Semi-Diameter 定义系统孔径；
- 追迹层不再依赖一阶 STOP 半径或孔径类型特判；
- 不增加新的测试文件，整体代码保持简洁。

## 7. 在 Zemax 中设置硬孔径

在 LDE 中把 Clear Semi-Diameter 设置为 Fixed，主要是在固定表面尺寸。要明确设置硬孔径，应进入：

```text
Surface Properties -> Aperture
```

然后选择：

- `Floating Aperture`：硬孔径半径始终等于该面的 Clear Semi-Diameter，适合当前圆形净口径模型；
- `Circular Aperture`：显式设置最小和最大半径，适合口径值需要独立于 Semi-Diameter 的情况。

对普通非 dummy 面，OpticStudio 通常会在 Semi-Diameter 改为 Fixed 时自动设置 Floating Aperture；对 dummy 面不应依赖这一隐式行为，建议在 Surface Properties 中显式选择 Aperture 类型并保存，确认 ZMX 中出现 `FLAP` 或相应的 aperture 记录。

## 8. 参考

- [System Aperture](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/System_Aperture.html)
- [Aperture Type](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v25101/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Aperture_Type.html)
- [Semi-Diameters](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v25101/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Semi_Diameters.html)
- [Surface Aperture](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v252/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Aperture_surface_properties.html)
