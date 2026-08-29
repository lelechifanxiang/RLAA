# 惠更斯 PSF 数值差异分析

## 当前现象

测试系统：

```text
tests/zemax/zmx_files/Double Gauss 28 degree field real.zmx
```

当前 `32x32` pupil、`32x32` image、主波长、0 视场结果：

```text
Zemax Strehl:                0.153000
OpticsCore Strehl:           0.114173
归一化 PSF 平均绝对误差:    0.048906
归一化 PSF 最大绝对误差:    0.375621
```

两侧 PSF 的环纹数量和整体形状已经基本一致，说明光线追迹、OPD 和主要相位关系没有结构性错误。剩余差异主要来自离散采样、像面网格原点、积分权重和 Strehl 参考值定义。

## 已基本排除的问题

### 1. 波长索引

两侧都使用主波长 `0.5876 um`：

```text
Zemax 波长编号:               2（1-based）
OpticsCore wavelength_index:  1（0-based）
```

Zemax helper 已执行 `wavelength_index + 1`，不存在波长错位。

### 2. 几何光线追迹

同一归一化 pupil 坐标下，像面交点和方向余弦已经与 Zemax 接近 FP64 精度：

```text
x/y 最大误差约 1e-14 mm
l/m/n 最大误差约 1e-15
```

因此 PSF 差异不是球面求交或折射公式造成的。

### 3. 出瞳参考 OPD

使用与 PSF 相同的 `32x32` 端点 pupil 网格，对全部 740 个单位圆内采样点比较 Zemax Batch Ray Trace OPD：

```text
最大 OPD 误差:  2.66e-10 waves
平均 OPD 误差:  8.66e-11 waves
RMS OPD 误差:   1.09e-10 waves
```

可以认为 OPL 累积、出瞳参考球面、反向求交和 OPD 符号均已对齐。后续不应继续把主要精力放在 OPD 公式上。

## 第一优先级：像面网格半像素约定

原 `_image_grid()` 使用：

```python
axis = (
    torch.arange(image_sample_count)
    - 0.5 * (image_sample_count - 1)
) * pixel_pitch
```

当 `image_sample_count=32` 时，中心坐标为：

```text
index 15: -0.25 um
index 16: +0.25 um
```

主光线位于四个像素之间，因此 OpticsCore 中心形成四个近似相同的峰值。Zemax 数值阵列则把 PSF 峰值集中在 `(16, 16)`。

现已将 OpticsCore 网格改为：

```python
axis = (
    torch.arange(image_sample_count)
    - image_sample_count // 2
) * pixel_pitch
```

即让 `index 16` 对应主光线。此时 32 个采样中心为 `-8.0 ~ 7.5 um`，
像素边界范围为 `-8.25 ~ 7.75 um`，与 Zemax 软件中的 Data Grid 一致。
试验中的误差变化为：

```text
平均误差: 0.048906 -> 0.027224
最大误差: 0.375621 -> 0.086789
```

这是目前发现的最大单项误差来源。

zospy 当前返回的 DataFrame 标签为 `-7.75 ... +7.75 um`，与 Zemax 软件显示
的 Data Grid 边界不一致。回归对标应以 Zemax 软件的像素边界和数值阵列索引为准，
不能直接把 zospy DataFrame 标签当成像素中心。

## 第二优先级：pupil 数值积分权重（已实现）

原实现虽然会生成 `SamplingResult.weights`，但：

1. `build_input_rays_from_sample()` 没有把 weights 写入 ray metadata
2. `_huygens_integral()` 没有使用 weights
3. 所有单位圆内采样点均按相同振幅相加
4. 单位圆边缘外的点直接丢弃

这相当于用方形节点网格对圆孔做简单点计数积分。圆周附近的节点实际代表的 pupil 面积不同，直接等权会产生明显的边界离散误差。

现已完成：

1. `SquarePupilSampler` 估算每个网格单元与单位圆的相交面积
2. 仅对圆周附近单元执行 `32x32` 子采样
3. 将面积权重归一化后写入 `SamplingResult.weights`
4. PSF 分析显式保留并传递 `SamplingResult`
5. 实际 PSF 与理想 PSF使用同一积分权重

当前 `weights` 的语义固定为 pupil 面积积分权重，直接乘复振幅，不取平方根。

对每个网格节点估算其 Voronoi 方格与单位圆的相交面积，并把该面积作为复振幅积分权重后：

```text
仅面积权重:
平均误差 0.048906 -> 0.044655

index 16 像面原点 + 面积权重:
平均误差 0.027224 -> 0.022326
最大误差 0.086789 -> 0.069508
```

实现后的实际结果：

```text
                         面积权重前    面积权重后
OpticsCore Strehl        0.114563      0.121856
归一化平均绝对误差       0.027224      0.022333
归一化最大绝对误差       0.086789      0.069529
```

因此 pupil 权重是第二个已确认并已修复的差异。

当前 weights 的物理语义为：

```text
离散积分权重 = pupil 面积权重
复振幅权重   = 面积权重 * pupil amplitude
```

后续如需支持光源强度或 apodization，应新增独立的 pupil amplitude，不能改变
`SamplingResult.weights` 的面积积分语义。

## 第三优先级：Zemax 的振幅传输与 pupil Jacobian

即使关闭 polarization，Zemax 的 Huygens PSF 也未必只对每根光线使用单位振幅。可能还包含：

1. 归一化 pupil 到实际 pupil 面积的 Jacobian
2. entrance/exit pupil 之间的 pupil aberration
3. obliquity 或投影面积因子
4. vignetted pupil 单元的有效面积
5. 系统 apodization

当前代码只使用：

```python
kernel = exp(i * phase)
amplitude = kernel.sum(dim=ray)
```

没有任何逐光线振幅项。对于低 NA、弱 pupil aberration 系统，这通常不改变 PSF 环纹位置，但会改变中心峰值和各级环纹的相对能量，符合当前“形状接近、数值不同”的现象。

建议后续导出或构造以下候选权重并逐项实验：

```text
area_weight
sqrt(area_weight)
area_weight * cos(theta)
area_weight * exit-pupil Jacobian
```

一次只增加一个因子，用归一化网格误差判断其作用。

## 第四优先级：Strehl 参考峰值定义

当前理想 PSF 的构造方式是：

```python
ideal_phase = k * phase_tilt
```

也就是保留实际光线方向、实际出瞳交点和实际 valid mask，只移除 OPD。

这不一定等同于 Zemax 的 diffraction-limited reference PSF。Zemax 可能使用：

1. 同 pupil 采样和同振幅权重下的理想参考球
2. 系统一阶 F/# 对应的理想 Airy 峰值
3. 单独归一化后的无像差 Huygens 计算

当前数据：

```text
有效光线数 N:       740
实际 PSF peak/N^2:  0.103365
当前报告 Strehl:    0.114173
Zemax Strehl:       0.153000
```

说明 Strehl 差异不能仅靠最终 PSF 归一化解释。应把“PSF 网格对齐”和“Strehl 对齐”拆成两个测试，先完成前者。

## 第五优先级：pupil 采样收敛

提高 pupil sampling 后，两侧差异持续下降：

| Pupil sampling | Zemax Strehl | OpticsCore Strehl | 平均网格误差 |
|---|---:|---:|---:|
| 32x32 | 0.153 | 0.1142 | 0.0489 |
| 64x64 | 0.120 | 0.1033 | 0.0438 |
| 128x128 | 0.108 | 0.0997 | 0.0422 |

这进一步说明当前差异具有明显的离散积分特征。

但不能只通过提高采样数规避问题，因为 Zemax 在不同 sampling 下的 Strehl 自身也明显变化。严格对标仍需要复刻同一 sampling 下的积分规则。

## 较低优先级问题

### 1. 传播相位符号与参考点

候选相位组合已经做过比较。使用出瞳参考点 `P_exit` 明显优于使用像面交点 `P_image`，当前符号组合也是已测试候选中误差较小的一组。

这部分仍应保留独立测试，但暂不是主要误差来源。

### 2. 大绝对相位的数值稳定性

当前相位包含从出瞳到像面的较大传播量，波数相乘后可能达到约 `1e6 rad`。FP64 足以避免百分之几级误差，但仍建议在计算 `sin/cos` 前减去主光线公共相位：

```text
phase = phase - phase_chief
```

这不会改变 PSF，可减少大数相位取模带来的数值噪声。

### 3. Zemax Huygens integral method

应确认 Zemax 系统级 `Method to Compute Huygens Integral` 是 Planar 还是 Spherical，并在 helper 中打印。当前 zospy Huygens PSF wrapper 没有显式设置该系统参数。

由于当前 PSF 形状已接近，这项不太可能是最大误差源，但应固定测试环境，避免 zmx 或 Zemax 默认设置变化。

## 建议排查顺序

### 第一步：修正像面网格原点

让偶数网格满足：

```text
axis[image_sample_count // 2] == 0
```

重新记录：

```text
峰值位置
最大误差
平均误差
中心 5x5 数值
```

### 第二步：接通 pupil weights（已完成）

1. PSF 分析已显式传递 `SamplingResult.weights`
2. `_huygens_integral()` 已使用 integration weights
3. 方形裁圆网格已计算边界单元面积
4. weights 已固定为面积权重，直接参与复振幅积分

### 第三步：单独对齐 PSF 网格

暂时不以 Strehl 作为失败条件，只比较：

```text
归一化 PSF 网格
中心剖面
径向能量分布
encircled energy
```

### 第四步：对齐理想 PSF 和 Strehl

打印并比较：

```text
actual peak
ideal peak
actual total energy
ideal total energy
Strehl
```

确认 Zemax 的 ideal peak 定义后，再修改 Strehl 实现。

### 第五步：检查振幅 Jacobian

如果完成像面原点和面积权重后仍有约 `1%` 以上系统性误差，再检查实际 pupil 映射、方向余弦和投影面积因子。

## 当前判断

当前 PSF 差异已经不属于光线追迹或 OPD 的严重错误。现有证据支持以下优先级：

```text
1. 偶数 image grid 半像素原点
2. pupil 边界面积和积分权重
3. Zemax 的逐光线振幅与 pupil Jacobian
4. Strehl 理想参考峰值
5. 传播相位和系统级 Huygens 设置细节
```

前两项完成后，归一化 PSF 平均误差已从约 `0.049` 降至约 `0.022`。
下一步应处理逐光线振幅、pupil Jacobian 和 Strehl 参考峰值定义。
