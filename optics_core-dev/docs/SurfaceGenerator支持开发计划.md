# SurfaceGenerator 支持开发计划

## 1. 当前完成度

| 需求 | 状态 | 当前缺口 |
| --- | --- | --- |
| P0-1 逐面状态 | 已完成 | `SurfaceTraceHistory` 可记录全部或指定表面的出射位置、方向、OPL 和 valid。 |
| P0-2 入瞳光线 | 已完成 | `build_pupil_rays()` 已公开，支持逐设计视场/入瞳、共享采样与波长，并设置初始 OPL。 |
| P0-3 中间继续追迹 | 已完成 | 上一段 `result.rays` 可从下一面续追，终态、OPL 和 valid 与完整追迹一致。 |
| P0-4 一阶量 | 已完成 | EFL、Working F/#、TTL、处方像面距离、近轴 BFL 和有效掩码均为 `[design]` tensor。 |
| P0-5 Wavefront/Spot 有效率 | 已完成 | 两类结果均提供有效数/有效率；Wavefront 另提供逐点 mask，无有效光线时评价量为 NaN。 |
| P0-6 异构 batch | 已完成 | 材料、半径、厚度、HFOV 和入瞳可逐设计不同；尚需在 CUDA 环境完成大规模运行确认。 |

## 2. 开发顺序

### 第一阶段：逐面训练状态

1. 新增 `SurfaceTraceHistory`，记录选定表面折射后的 `x/y/z/l/m/n/opl/valid` 和 `surface_indices`。
2. 在 `TraceOptions` 增加可选记录面；关闭时不分配历史 tensor。
3. 明确续追用法：将上一段 `TraceResult.rays` 传入，并从下一面开始追迹。
4. 增加完整追迹、逐面终态、分段续追和相同前缀测试。

### 第二阶段：评价结果补齐

1. 为 `FirstOrderData/Result` 增加处方 `image_plane_distance`、近轴 `bfl` 和 `valid`，输出均为 `[design]`。
2. 为 `WavefrontResult` 增加 valid mask/count/fraction；无有效光线时 RMS 返回 NaN。
3. 为 `SpotDiagramResult` 增加逐设计、视场、波长的 valid count/fraction。
4. 使用现有 ZMX 补充 EFL、Working F/#、BFL、TTL、像面距离和 Spot 对标。

### 第三阶段：规模验收

1. 运行全量 pytest，确认现有单设计和 tolerance analysis 行为不变。
2. 在 CUDA 环境运行：

```bash
python examples/batch_spot_multiple_zmx.py --device cuda:0 --design-count 3600
```

3. 若显存不足，再实现 Spot design minibatch；参数 tensor 缓存和失败原因分类放到 P1。

## 3. 完成标准

- 一次追迹可返回 12 个折射面后的完整状态，且与分别追迹到各面一致；
- 完整追迹与任意分段续追的最终 ray、OPL、valid 一致；
- EFL、Working F/#、BFL、TTL 和像面距离均有 `[design]` 结果及有效掩码；
- Wavefront/Spot 正确报告有效率，无有效光线时返回 NaN；
- 五文件异构 batch 与单文件基准一致，并通过 CUDA 3,600 design 运行。
