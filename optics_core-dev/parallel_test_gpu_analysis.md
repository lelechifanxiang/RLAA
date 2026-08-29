# 并行测试 GPU 使用情况分析

## 关键发现

### ❌ 并行测试**没有使用 GPU/CUDA**

---

## 证据

### 1. 当前环境检查

```bash
$ python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
CUDA available: False
Device count: 0
```

**当前测试环境没有 CUDA 支持。**

---

### 2. 默认设备配置

#### RuntimeConfig 默认值
```python
@dataclass(slots=True)
class BackendConfig:
    name: BackendName = "torch"
    device: str | None = None    # 默认为 None
    dtype: str | None = None
    enable_autodiff: bool = True
```

#### default_device() 函数逻辑

位置: [`optics_core/_runtime.py:11-15`](optics_core/_runtime.py:11-15)

```python
def default_device(system: MultiOpticalSystem) -> torch.device:
    configured_device = system.config.backend.device
    if configured_device is None:
        return torch.device("cpu")    # 未配置时默认 CPU
    return torch.device(configured_device)
```

**关键逻辑**：
- `config.backend.device = None` → 返回 `cpu`
- 只有显式配置才会使用 GPU

---

### 3. 测试代码未配置 GPU

#### 测试文件检查

```bash
$ pytest tests/contract/test_huygens_mtf_contract.py \
         tests/contract/test_huygens_psf_contract.py \
         tests/contract/test_wavefront_map_contract.py -v
```

这些测试文件：
- ✅ 使用 `build_backward_paraxial_system()` 等 fixture
- ❌ **没有设置** `system.config.backend.device = "cuda"`
- ❌ **没有传入** GPU 设备配置

#### 典型测试代码

```python
def test_huygens_mtf_impulse_is_one() -> None:
    psf = torch.zeros((1, 1, 17, 17), dtype=torch.float64)
    psf[0, 0, 8, 8] = 1.0
    frequencies = _frequency_tensor((0.0, 100.0, 300.0, 500.0))
    
    # 直接调用，未指定设备
    sagittal, tangential = compute_huygens_mtf(
        psf,
        pixel_pitch_um=torch.tensor([0.5], dtype=torch.float64),
        frequencies_lp_per_mm=frequencies,
    )
    # 所有 tensor 默认在 CPU 上
```

---

### 4. 对比：示例脚本 vs 测试代码

| 项目 | 示例脚本 | 测试代码 |
|------|---------|---------|
| **默认设备** | `cuda:0` | `None` (→ `cpu`) |
| **配置方式** | 命令行参数 `--device` | 无配置 |
| **系统初始化** | 显式传入 `device=torch.device("cuda:0")` | 使用默认 fixture |
| **GPU 监控** | ✅ 内存统计、同步计时 | ❌ 无 |
| **CUDA 检查** | ✅ `resolve_device()` 检查可用性 | ❌ 无 |

---

## 为什么并行测试运行在 CPU 上？

### 原因 1：环境没有 CUDA
```
torch.cuda.is_available() = False
torch.cuda.device_count() = 0
```

当前测试运行环境：
- 可能是 CI/CD 环境
- 可能是 CPU-only PyTorch 安装
- 可能是没有 GPU 的开发机器

### 原因 2：测试代码未配置 GPU

即使有 GPU，测试代码也不会使用，因为：

```python
# tests/fixtures/systems.py
def build_tracing_system(...) -> oc.MultiOpticalSystem:
    system = oc.MultiOpticalSystem(
        architecture=architecture,
        # 没有传入 config 参数
        # 使用默认 RuntimeConfig()
        # 默认 device = None → CPU
    )
```

### 原因 3：pytest-xdist 进程隔离

`pytest-xdist` 使用多进程并行：
- 每个 worker 是独立的 Python 进程
- 每个进程独立初始化 PyTorch
- 如果没有显式配置 GPU，所有 worker 都使用 CPU

---

## GPU 相关的测试代码

### Benchmark 测试（有 CUDA 配置）

位置: [`tests/benchmark/test_double_gauss_zmx_batch_ray_trace_benchmark.py`](tests/benchmark/test_double_gauss_zmx_batch_ray_trace_benchmark.py)

```python
# 这个测试显式配置了 CUDA
system.config.backend.device = "cuda"

if optics_core_rays.x.device.type == "cuda":
    torch.cuda.synchronize()
```

### Contract 测试（有 CUDA 检查）

位置: [`tests/contract/test_public_api_contracts.py`](tests/contract/test_public_api_contracts.py)

```python
def test_trace_uses_configured_runtime_device_when_cuda_available() -> None:
    showcase_system.config.backend.device = "cuda"
    
    assert trace.rays.x.device.type == "cuda"
    assert trace.rays.y.device.type == "cuda"
```

**但这些测试需要 CUDA 可用才能运行。**

---

## 性能影响

### 并行测试为什么慢？

前面的性能报告显示：
- 串行 (CPU): **3.79秒**
- 并行 16 workers (CPU): **9.67秒**

原因分析：

| 因素 | 影响 |
|------|------|
| **CPU 计算** | 轻量测试，单次 < 0.01s |
| **进程开销** | 启动 16 个 Python 进程 |
| **PyTorch 初始化** | 每个 worker 初始化 PyTorch (CPU) |
| **进程通信** | 序列化/反序列化测试结果 |
| **资源竞争** | 16 个进程争抢 CPU 资源 |

### 如果使用 GPU 会怎样？

**理论优势**：
- ✅ GPU 并行计算能力强
- ✅ 批量操作效率高

**实际问题**：
- ⚠️ pytest-xdist 多进程共享 GPU 可能冲突
- ⚠️ GPU 初始化开销（每个 worker）
- ⚠️ 测试太轻量，无法发挥 GPU 优势

---

## 建议

### 当前状态（CPU 测试）

✅ **保持现状** - 对于单元/契约测试：
- 测试轻量快速
- 串行执行最优（3.79s）
- 无需 GPU 依赖

### 如果要启用 GPU 测试

需要以下改动：

#### 1. 环境准备
```bash
# 安装 CUDA 版本的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 验证 CUDA 可用
python -c "import torch; assert torch.cuda.is_available()"
```

#### 2. 修改测试 fixture
```python
# tests/fixtures/systems.py
def build_tracing_system(..., device: str = "cpu") -> oc.MultiOpticalSystem:
    config = oc.RuntimeConfig(
        backend=oc.BackendConfig(device=device)
    )
    system = oc.MultiOpticalSystem(
        architecture=architecture,
        config=config,  # 传入配置
    )
    ...
```

#### 3. 添加 pytest 配置
```python
# conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="PyTorch device: cpu, cuda, cuda:0, etc."
    )

@pytest.fixture(scope="session")
def device(request):
    return request.config.getoption("--device")
```

#### 4. 运行 GPU 测试
```bash
# 串行 GPU 测试
pytest tests/ --device cuda:0

# 并行 GPU 测试（需要多 GPU 或小心调度）
pytest tests/ --device cuda:0 -n 4
```

---

## 总结

### 当前状态

| 测试类型 | 设备 | 并行方式 | 性能 |
|---------|------|---------|------|
| **契约测试** | CPU | pytest-xdist (16 workers) | 9.67s |
| **契约测试** | CPU | 串行 | **3.79s** ✅ 最快 |
| **示例脚本** | GPU (cuda:0) | 批量并行（PyTorch） | 设计用于大规模仿真 |

### 核心结论

1. ❌ **并行测试未使用 GPU** - 运行在 CPU 上
2. 🔧 **原因是配置** - 非环境问题
   - 测试代码未设置 `config.backend.device`
   - `default_device()` 默认返回 CPU
3. ✅ **这是合理的** - 单元测试不需要 GPU
4. 🚀 **GPU 用于生产** - 示例脚本默认使用 GPU 进行大规模批量仿真
