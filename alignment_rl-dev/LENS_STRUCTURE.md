# Double Gauss 六片镜头结构

训练、评估和 MTF 分析统一使用带 Coordinate Break 的 Double Gauss ZMX
处方：

```text
optics_core-dev/tests/zemax/zmx_files/Double Gauss 28 degree field with CB.ZMX
```

当前处方由 `optics_core` 解析为 24 个 sequential surfaces，其中包含 4 对
Coordinate Break。四个逻辑镜组按 ZMX 中 Coordinate Break pair 的顺序映射，
默认仅开放第二组的 `dx/dy` 动作。

光学设置：

- 角度视场：0°、10°、14°；训练默认使用中心视场。
- 波长：0.4861、0.5876、0.6563 μm，等权重。
- MTF 频率：20、30、50 lp/mm。
- 训练采样：32×32 pupil samples，64×64 image samples。
- 制造公差：通过 Double Gauss 参数向量直接作用于曲率半径和厚度。

单环境与批环境共享相同的处方、Coordinate Break 映射、奖励和终止条件；
批环境只是在一个 `MultiOpticalSystem` 中并行计算多个 design。
