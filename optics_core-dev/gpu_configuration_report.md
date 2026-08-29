# GPU 配置报告

## 概述

`optics_core` 是一个基于 PyTorch 的并行光线追迹库，支持 GPU 加速的批量光学仿真。

---

## GPU 配置机制

### 1. 设备配置位置

#### RuntimeConfig / BackendConfig
定义在 [`optics_core/types.py`](optics_core/types.py:28-34)：

```python
@dataclass(slots=True)
class BackendConfig:
    name: BackendName = "torch"
    device: str | None = None          # GPU 设备配置
    dtype: str | None = None
    enable_autodiff: bool = True

@dataclass(slots=True)
class RuntimeConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    units: UnitSystem = field(default_factory=UnitSystem)
    default_batch_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### MultiOpticalSystem 初始化
在 [`optics_core/system.py`](optics_core/system.py:30-50)：

```python
class MultiOpticalSystem:
    def __init__(
        self,
        architecture: OpticalArchitecture,
        *,
        config: RuntimeConfig | None = None,  # 包含设备配置
        ...
    ):
        self.config = config or RuntimeConfig()
        ...
```

---

### 2. 示例脚本中的 GPU 配置

所有批量分析示例都通过命令行参数 `--device` 指定设备。

#### batch_mtf.py
```python
DEFAULT_DEVICE = "cuda:0"

def parse_args() -> argparse.Namespace:
    parser.add_argument(
        "--device", 
        default=DEFAULT_DEVICE, 
        help="运行设备，默认 cuda:0；CPU 可手动指定 cpu"
    )
```

#### batch_psf.py
```python
DEFAULT_DEVICE = "cuda:0"
```

#### batch_wavefront.py
```python
DEFAULT_DEVICE = "cuda:0"
```

#### batch_spot.py
```python
DEFAULT_DEVICE = "cuda:0"

def resolve_device(device_text: str) -> torch.device:
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"请求的设备 {device_text!r} 不可用，"
            f"当前环境未检测到 CUDA。"
        )
    return device
```

---

## 使用方式

### 默认配置（GPU）

```bash
# 使用默认 cuda:0
python examples/batch_mtf.py --design-count 1024

python examples/batch_psf.py --design-count 1024

python examples/batch_wavefront.py --design-count 1024
```

### 指定 GPU 设备

```bash
# 使用 GPU 0
python examples/batch_mtf.py --device cuda:0 --design-count 1024

# 使用 GPU 1
python examples/batch_mtf.py --device cuda:1 --design-count 1024
```

### 使用 CPU

```bash
# 强制使用 CPU
python examples/batch_mtf.py --device cpu --design-count 1024
```

---

## GPU 性能监控

### 内存统计

所有批量分析脚本都包含 GPU 内存监控（以 `batch_mtf.py` 为例）：

```python
def run_batch_mtf_analysis(...):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    started_at = synchronized_now(device)
    result = system.analysis.mtf(settings).run()
    elapsed_seconds = synchronized_now(device) - started_at
    
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0
    if device.type == "cuda":
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    
    print(f"peak_allocated_gib={peak_allocated_bytes / 2**30:.3f}")
    print(f"peak_reserved_gib={peak_reserved_bytes / 2**30:.3f}")
```

### 同步计时

```python
def synchronized_now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)  # 确保 GPU 操作完成
    return time.perf_counter()
```

---

## 配置建议

### GPU 环境

✅ **推荐配置**
- 安装 CUDA 版本的 PyTorch
- 使用 `--device cuda:0` 或让程序使用默认值
- 适合：大规模批量仿真（100+ 设计）

### CPU 环境

⚠️ **备用配置**
- 无 GPU 或 CUDA 不可用时
- 使用 `--device cpu`
- 性能显著低于 GPU
- 适合：小规模测试、开发调试

### 多 GPU 环境

🔄 **高级配置**
- 手动指定设备：`cuda:0`, `cuda:1`, 等
- 当前版本不支持自动多 GPU 并行
- 可通过脚本级别并行（不同进程使用不同 GPU）

---

## 检查 GPU 可用性

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

---

## 配置总结

| 组件 | 默认设备 | 配置方式 |
|------|---------|---------|
| **batch_mtf.py** | `cuda:0` | `--device` 参数 |
| **batch_psf.py** | `cuda:0` | `--device` 参数 |
| **batch_wavefront.py** | `cuda:0` | `--device` 参数 |
| **batch_spot.py** | `cuda:0` | `--device` 参数 |
| **RuntimeConfig** | `None` (自动) | `BackendConfig.device` |

**核心设计理念**：所有批量分析默认使用 GPU (`cuda:0`)，提供命令行参数灵活切换设备。
