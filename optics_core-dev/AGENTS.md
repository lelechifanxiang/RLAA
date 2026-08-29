# optics_core 并行光线追迹和仿真基础库
使用GPU加速的多光学系统大规模并行光线追迹和评价基础库。

## 技术栈
- Python 项目，使用torch实现并行加速
- 测试框架使用 pytest。

## 核心编程约定
- 在数据处理时优先使用tensor，并且使用FP64精度。
- 避免频繁CPU-GPU数据搬运。
- 几何、材料和追迹相关的数值计算接口统一采用 FP64 tensor 输入，不额外兼容标量、tuple 或 list。

## 目录

- `optics_core/`：主包，包含系统建模、数据准备、光线追迹和光学评价。
  - `optics_core/tracing/`：顺序光线追迹、表面求交、光线交互及入瞳光线构造。
  - `optics_core/first_order.py`、`optics_core/_first_order_probes.py`：一阶量计算及 probe 光线构造。
  - `optics_core/system.py`、`optics_core/system_state.py`：多系统模型及追迹准备态数据。
  - `optics_core/analysis.py` 及各评价模块：Spot、Wavefront、PSF/MTF 等分析接口与实现。
- `zemax_utils/`：ZMX 文本解析和 OpticsCore 系统构建，不依赖 Zemax 运行环境。
- `tests/`：契约、回归、性能测试及 Zemax 对标辅助代码。
- `examples/`：可直接运行的批量分析和性能示例。
- `docs/`：设计说明、开发计划和问题分析。
- `reference/`：外部参考实现，仅用于研究和对照，不作为主包运行依赖。

## 命令

- 运行全部测试：`python -m pytest`

## 单元测试约定
- 每个特性都需要和zemax对比准确性，采用zospy库调用zemax获取期望值，可参考`tests\regression\test_spot_diagram_against_zemax.py`的调用方法。
- zospy获取zemax数值的功能统一放置于`tests/zemax`目录，要做好架构设计，减少代码重复。
- 获zemax期望数值时，尽可能采用直接获取的方式，不要通过光线追迹结果反算的形式
- 除了检验结果正确，也需要打印关键验证数据，提示用户测试情况

## 编码规范与风格
- 注释、文档字符串、日志打印使用**中文**。
- 使用简要的注释说明当前进行的操作，如`# 1. 设置多波长`、`# 2. 计算光线和表面交点`
- 所有新增或重构函数要有简要的docstring
- 保持 batch-first 设计思路：新增接口时优先考虑多系统模型，单系统作为退化场景处理。
- 只在必要的位置添加保护代码，程序异常崩溃并非坏事，避免代码过于冗长是更重要的。
- 不要添加过多的contract测试，这可能会产生过多的异常产生和保护逻辑，导致代码长度过长，不易阅读。
- 避免单个文件过长，必要时刻可进行文件拆分和大规模重构
- 无需显示添加del param以声明param参数未使用