## 原始 Zemax 差异具体出现在哪里

| 镜头 | 最大差异视场 | Zemax | optics_core | 绝对差异 | 相对差异 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sg6_material_a.zmx` | −5° | 57.504131 | 54.059597 | 3.444534 µm | 5.99% |
| `sg6_material_b.zmx` | −5° | 60.663452 | 57.296325 | 3.367127 µm | 5.55% |
| `sg6_material_c.zmx` | −5° | 70.010480 | 64.943337 | **5.067144 µm** | 7.24% |
| `sg6_hfov25_f3p0.zmx` | −25° | 14.985651 | 14.522569 | **0.463081 µm** | 3.09% |
| `sg6_hfov34_f4p8.zmx` | −17° | 10.592899 | 9.518117 | 1.074783 µm | 10.15% |

完整的逐视场绝对差异如下，单位均为 µm：

```text
sg6_material_a
视场: [0, 1.25, 2.5, 3.75, 5]
差异: [0.008563, 0.074099, 0.082308, 1.861607, 3.444534]

sg6_material_b
差异: [0.469025, 0.417805, 0.581339, 1.823769, 3.367127]

sg6_material_c
差异: [0.396835, 0.169278, 0.940066, 2.972212, 5.067144]

sg6_hfov25_f3p0
视场: [0, 6.25, 12.5, 18.75, 25]
差异: [0.082058, 0.092550, 0.215143, 0.019793, 0.463081]

sg6_hfov34_f4p8
视场: [0, 8.5, 17, 25.5, 34]
差异: [0.628859, 0.848830, 1.074783, 0.917826, 0.185887]
```

## 3. 已经确认的根因

### 3.1 最外圈 180 条光线被全部误删

当前采用：

```text
Hexapolar, ray_density=30
```

一共有：

```text
1 + 3 × 30 × 31 = 2791 条评价光线
```

实际检查 `valid_points` 后发现，五个镜头的每个视场、每个波长都只有：

```text
2611 / 2791
```

恰好少了：

```text
2791 - 2611 = 180
```

最外层第 30 环正好有：

```text
6 × 30 = 180 条光线
```

因此可以确定：最外圈瞳孔光线被 STOP 面口径裁掉了。

### 3.2 optics_core 内部存在两套 STOP 半径

ZMX 文本加载器将 `DIAM ... 1` 解析成固定口径，见 [zmx_loader.py](/home/huweijian/optics_core/zemax_utils/zmx_loader.py:257)。

但这些镜头的系统孔径类型是 `FNUM`。`optics_core` 的一阶计算已经根据 EFL 和 F/# 正确计算了：

```python
entrance_pupil_radius = abs(effl) / (2 * f_number)
stop_radius = entrance_pupil_radius / front_pupil_magnification
```

实现位于 [first_order.py](/home/huweijian/optics_core/optics_core/first_order.py:69)。

两套值如下：

| 镜头 | ZMX 文本 STOP `DIAM` | `FirstOrderData.stop_radius` |
| --- | ---: | ---: |
| material A | 20.664600 | 20.666686 |
| material B | 20.664600 | 20.666579 |
| material C | 20.664600 | 20.666618 |
| HFOV 25° | 7.832550 | 7.833613 |
| HFOV 34° | 3.228844 | 3.292223 |

Spot 光线根据后一列的入瞳半径生成，但追迹到 STOP 时，又被前一列裁剪。

裁剪发生在 [_hits.py](/home/huweijian/optics_core/optics_core/tracing/_hits.py:277)：

```python
inside_aperture = x² + y² <= aperture_radius²
```

因此，刚刚按照正确系统孔径生成的边缘光线，马上又被较小的 ZMX `DIAM` 判定为无效。

Ansys 文档说明，系统孔径类型和值决定入瞳和相关孔径；只有 `Float By Stop Size` 才以 STOP 面尺寸作为系统孔径来源。[Aperture Value](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v25101/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Aperture_Value.html)、[Aperture Type](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v252/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Aperture_Type.html)

因此，对于 `FNUM`，我认为应以 `FirstOrderData.stop_radius` 作为追迹 STOP 半径；`DIAM` 不应反过来覆盖系统 F/# 所确定的孔径。对于 `Float By Stop Size`，才应继续以表面固定半径为准。

## 4. 验证根因的实验

我只替换了 STOP 面的有效追迹半径：

```text
STOP aperture radius = FirstOrderData.stop_radius × (1 + 1e-9)
```

其他所有表面口径裁剪都保持不变。

结果：

```text
修改前最大 Zemax 差异：5.067143531 µm
修改后最大 Zemax 差异：0.000002406 µm
```

也就是从微米级误差降低到约：

```text
2.4 × 10⁻⁶ µm
```

25 个视场全部基本精确重合。这足以说明问题不是：

- 异构材料索引；
- GPU batch；
- Spot RMS 公式；
- Hexapolar 采样；
- Zemax 参考数据。

问题集中在 STOP 面有效半径的选择。

单纯增加浮点比较容差也不够，因为 HFOV 34° 的两个半径相差约 1.96%，不是数值舍入误差。

## 5. 可以直接转给 optics_core 负责人的修复要求

> 对于 `entrance_pupil_diameter` 和 `image_f_number` 系统孔径，Spot/普通采样追迹在 STOP 面应使用 `FirstOrderData.stop_radius`，而不能继续使用 ZMX 文本导入的 `surface.aperture_radius`。  
> 对于 `float_by_stop_size`，仍以 STOP 面固定半口径为权威来源。  
> 异构 batch 中需要生成 `[design, surface]` 的有效口径 tensor，并用逐设计 `stop_radius` 覆盖 STOP 列。  
> 同时在边界判断中增加很小的 FP64 相对裕量，避免理论上位于 `r=R` 的光线因为三角函数舍入被误删。

建议增加两项验收：

```python
torch.testing.assert_close(
    batch_rms,
    zemax_rms,
    atol=1e-3,
    rtol=0.0,
)
```

并确认：

```text
FNUM 系统中，STOP 边缘 pupil 光线不会因为导入 DIAM 与解析 stop_radius 不一致而全部失效。
```

现有测试只打印 Zemax 差异，没有断言，应在 [test_heterogeneous_batch_against_zemax.py](/home/huweijian/optics_core/tests/regression/test_heterogeneous_batch_against_zemax.py:31) 中补上这个断言。

本次仅分析和运行验证，没有修改或提交代码。