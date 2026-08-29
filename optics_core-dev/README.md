# optics_core

`optics_core` 是一个基于 PyTorch 的并行光线追迹与光学分析基础库，主要面向多光学设计的 GPU 批量仿真。当前示例覆盖光线追迹、点列图、Huygens MTF、Huygens PSF 和 Wavefront Map。

## 环境准备

```bash
pip install -r requirements.txt
pip install torch
```

说明：

- GPU 加速需要安装可用的 CUDA 版 PyTorch。
- Zemax 对标测试需要本机已安装并可调用 Zemax。

## 快速运行

### 基础示例
加载测试 Zemax 文件 `tests/zemax/zmx_files/Double Gauss 28 degree field real.zmx`，构造多设计系统，执行并行追迹、点列图和 layout 导出。

```bash
python examples/basic.py
```

### 批量点列图

```bash
python examples/batch_spot.py --device cuda:0 --ray-density 30
```

### 批量 Huygens MTF

Monte Carlo 生成 1024 个装配公差设计，并保存 CSV：

```bash
python examples/batch_mtf.py --device cuda:0 --design-count 1024 
```

### 批量 Huygens PSF
导出每个设计、每个视场的 PSF 图片：

```bash
python examples/batch_psf.py --device cuda:0 --design-count 1024 
```

### 批量 Wavefront Map

```bash
python examples/batch_wavefront.py --device cuda:0 --design-count 1024 
```

默认输出目录为：

```text
examples/output/
```

## 代码使用方式

核心流程可以参考 `examples/basic.py`：

```python
spec = load_zmx_sequential_system_spec(zmx_path)
base_system = build_optics_core_system_from_zmx_spec(spec)

system = oc.MultiOpticalSystem(
    architecture=base_system.architecture,
    parameter_schema=schema,
    parameters=parameters,
    config=base_system.config,
    tracer=base_system.tracer,
    materials=base_system.materials,
    fields=base_system.fields,
    wavelengths=base_system.wavelengths,
    aperture=base_system.aperture,
)
system.prepare()

spot = system.analysis.spot_diagram().run()
```

多设计仿真的关键是 `ParameterSchema` 和 `ParameterVectorBatch`：所有设计共享同一个光学结构，只改变参数向量，从而获得 batch-first 的并行计算能力。

## 测试

```bash
python -m pytest
```

Zemax 回归测试依赖 Zemax / ZOS API 环境；没有 Zemax 的机器只适合运行不依赖 Zemax 的测试和示例。
