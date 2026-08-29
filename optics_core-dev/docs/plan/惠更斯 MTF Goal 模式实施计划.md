# 惠更斯 MTF Goal 模式实施计划

## Goal

实现可与 Zemax Huygens MTF 对标的多视场、多波长 Sagittal/Tangential MTF，并复用现有 Huygens PSF 内核。

## 必须遵守

1. 使用 FP64/complex128 tensor，保持 batch-first。
2. 一次系统构造、一次 batch 光线追迹完成全部目标视场。
3. MTF 不复制光线追迹、Huygens 积分、Image Delta 或多波长混合逻辑。
4. Sagittal 对应局部像面 `x` 频率，Tangential 对应局部像面 `y` 频率。
5. 多波长必须先混合 PSF，再计算 MTF，禁止加权平均单色 MTF。
6. Zemax 期望值必须由 Huygens MTF 分析直接获取。
7. 不增加非必要兼容、保护或抽象。

## 实施步骤

### 1. 将 Huygens PSF 内核扩展为多视场 batch

- 提取内部批量计算函数。
- 输入多个 `field_indices`，一次追迹全部视场和选定波长。
- 输出最终 PSF：

```text
(design, field, image_y, image_x)
```

- 全波长时，每个视场以主波长主光线像点建立共同网格。
- 保持现有单视场 `run_huygens_psf()` 公开行为不变。

### 2. 扩展 MTF 接口

`MTFSettings` 增加：

```python
pupil_sample_count: int = 32
image_sample_count: int = 32
image_delta_um: Scalar = 0.0
frequencies_lp_per_mm: Sequence[Scalar] = ()
field_indices: Sequence[int] | None = None
wavelength_index: int | None = None
save_path: str | None = None
```

`MTFResult` 输出：

```text
frequencies_lp_per_mm: (frequency,)
sagittal:              (design, field, frequency)
tangential:            (design, field, frequency)
pixel_pitch_um:         (design,)
field_indices
wavelength_indices
```

### 3. 实现 PSF 到 MTF

新增 `optics_core/huygens_mtf.py`：

```python
sagittal_lsf = psf.sum(dim=-2)
tangential_lsf = psf.sum(dim=-1)
```

然后：

1. 对 LSF 对称补零。
2. `ifftshift` 后执行 `torch.fft.rfft`。
3. 用零频复 OTF 归一化。
4. 对复 OTF 的实部、虚部分别插值到目标频率。
5. 取复数模值得到 MTF。
6. 显式保证 `MTF(0)=1`。

频率：

```text
frequency_step = 1000 / (n_fft × image_delta_um)
nyquist = 500 / image_delta_um
```

请求频率超过任一设计 Nyquist 时直接报错。

### 4. 接通分析入口和绘图

- `ModulationTransferFunction.run()` 调用 `run_huygens_mtf()`。
- 图片导出只支持单设计。
- 一张图绘制所有视场的 S/T 曲线。

### 5. 增加 Zemax helper

新增 `tests/zemax/huygens_mtf.py`，调用：

```python
zp.analyses.mtf.HuygensMTF(...)
```

直接返回：

- Zemax 频率轴
- Sagittal MTF
- Tangential MTF
- 输入参数和实际 Image Delta

### 6. 测试

契约测试至少包括：

1. 脉冲 PSF 的 S/T MTF 恒为 1。
2. 各向同性高斯的 S/T 相同。
3. 各向异性高斯验证 S/T 未交换。
4. PSF 强度缩放不改变 MTF。
5. 一维投影 FFT 与二维 FFT 中心切片一致。
6. `MTF(0)=1`。
7. 超过 Nyquist 报错。
8. 多波长验证“先混合 PSF，再计算 MTF”。
9. 多视场只执行一次 PSF 光线追迹。

Zemax 回归矩阵：

```text
field_index:      0, 1, 2
wavelength_index: 0, 1, 2, -1
pupil sampling:   64 × 64
image sampling:   64 × 64
image delta:      0
frequency range:  0–150 lp/mm
```

逐 case 打印 S/T 最大和平均绝对误差。初始阈值：

```text
最大绝对误差 <= 0.02
平均绝对误差 <= 0.005
```

未达标 case 标记为 `xfail`。

## 完成条件

1. 全部新增 contract 测试通过。
2. Zemax 12 个回归 case 已运行，通过或带明确误差进入 xfail。
3. 现有 PSF 接口和测试不退化。
4. `python -m pytest` 无新增失败。
5. 代码中不存在重复的 Huygens PSF 或多波长混合实现。
