# optics\_core 重构计划

## 1. 目标

本计划的目标是把当前项目中与自动优化器强耦合的“GPU 并行光线追迹 + 像质评估”能力剥离成一个可复用基础库，供后续不同算法模块调用。

基础库的目标能力分三层：

1.  第一阶段 并行光线追迹
    
2.  第二阶段 像质评估
    
3.  第三阶段 图像仿真扩展能力
    

当前不作为基础库首批范围的内容：

*   种群初始化
    
*   变异策略
    
*   AdamO / GlobalOpti
    
*   结果导出和绘图
    

这些内容仍然应该保留在算法层。

## 2. 总体阶段规划

建议分成 6 个阶段推进。

## 3. 阶段 0：边界确认与现状冻结

### 3.1 目标

明确基础库边界，冻结第一版要保留和不保留的能力范围。

### 3.2 建议纳入基础库的内容

#### 已有模块

*   几何曲面表示
    
    *   球面、偶次非球面
        
    *   近轴面（待开发）
        
*   材料模型
    
    *   色散、密度、成本
        
    *   热膨胀系数（待开发）
        
*   系统装配
    
    *   光学系统表示
        
    *   支持坐标间断（待开发）
        
*   光线采样
    
    *   高斯、平方
        
    *   六边、杂乱（待开发）
        
*   前向/后向追迹
    
*   入瞳估计（光线瞄准）
    
*   像质评估（分析）
    
    *   点列图
        
    *   波前
        
    *   PSF
        
    *   MTF
        
    *   对比度（待开发）
        
*   物理约束评估
    
    *   一阶参数
        
    *   CRA
        
    *   畸变
        
    *   其他一系列操作数
        

#### 待添加模块

*   多重结构
    
*   公差计算（依赖坐标间断）
    
*   膜层
    

### 3.3 暂不纳入基础库的内容

*   run 主调度
    
*   init\_pop
    
*   init\_opti
    
*   mutation\_all
    
*   mutation\_n
    
*   mutation\_asph
    
*   AdamO
    
*   GlobalOpti
    
*   draw\_all
    
*   write\_optics\_data
    

### 3.4 交付物

1.  一份基础库边界说明
    
2.  一份初版模块划分图
    
3.  一份待迁移函数清单
    

## 4. 阶段 1：定义初步接口

这一阶段不急着重写实现，先把“未来想让别人怎么调用”定义清楚。

### 4.1 初版模块划分建议

建议把基础库暂命名为 optics\_core，内部按以下模块划分：

1.  optics\_core.geometry 曲面与光线几何对象
    
2.  optics\_core.materials 材料库、折射率和色散模型
    
3.  optics\_core.system 光学系统定义和装配
    
4.  optics\_core.sampling 光线采样
    
5.  optics\_core.tracing GPU 并行追迹
    
6.  optics\_core.evaluation 像质评估与工程约束评估
    
7.  optics\_core.imaging 未来扩展：PSF、MTF、图像仿真
    

### 4.2 初步接口建议

下面是一版偏工程化、可逐步落地的初始接口。

#### 4.2.1 system 数据结构

建议把当前 self.basics 和 self.order1 中“真正用于求值”的部分拆成显式对象。

示意接口：

```python
from dataclasses import dataclass
from typing import Optional, Sequence, Literal
import torch

@dataclass
class SurfaceSpec:
    kind: str
    curvature: torch.Tensor
    distance: torch.Tensor
    aperture_radius: torch.Tensor
    conic: Optional[torch.Tensor] = None
    asphere_coeffs: Optional[torch.Tensor] = None

@dataclass
class MaterialSpec:
    name: Optional[str]
    nd: torch.Tensor
    vd: torch.Tensor
    density: Optional[torch.Tensor] = None
    price: Optional[torch.Tensor] = None

@dataclass
class OpticalSystem:
    surfaces: list[SurfaceSpec]
    materials: list[MaterialSpec]
    stop_index: int
    wavelengths_nm: torch.Tensor
    wavelength_weights: torch.Tensor
    sensor_z: torch.Tensor

```

建议后续支持两种构建方式：

1.  from\_parameter\_vector(...)
    
2.  from\_explicit\_specs(...)
    

第一种兼容当前项目，第二种兼容未来其他算法模块。

#### 4.2.2 ray bundle 数据结构

```python
@dataclass
class RayBundle:
    origins: torch.Tensor
    directions: torch.Tensor
    wavelengths_nm: torch.Tensor
    weights: Optional[torch.Tensor] = None

```

约定：

*   origins shape 为 \[3, ray\_count, batch\]
    
*   directions shape 为 \[3, ray\_count, batch\]
    
*   wavelengths\_nm shape 为 \[ray\_count, batch\] 或广播兼容格式
    

#### 4.2.3 tracing 结果结构

```python
@dataclass
class TraceResult:
    valid_mask: torch.Tensor
    final_origins: torch.Tensor
    final_directions: torch.Tensor
    surface_intersections: torch.Tensor
    opd: Optional[torch.Tensor] = None
    ray_angle_in_deg: Optional[torch.Tensor] = None
    ray_angle_out_deg: Optional[torch.Tensor] = None

```

#### 4.2.4 tracing 模块的建议接口

```python
def trace_rays(
    system: OpticalSystem,
    rays: RayBundle,
    start_surface: int = 0,
    stop_surface: int | None = None,
    record_opd: bool = False,
    record_ray_angles: bool = False,
    warm_start: Optional[torch.Tensor] = None,
) -> TraceResult:
    ...


def trace_rays_backward(
    system: OpticalSystem,
    rays: RayBundle,
    start_surface: int,
    stop_surface: int,
    warm_start: Optional[torch.Tensor] = None,
) -> TraceResult:
    ...

```

#### 4.2.5 sampling 模块的建议接口

```python
def sample_aperture_rays(... ) -> RayBundle:
    ...


def sample_pupil_rays(... ) -> RayBundle:
    ...


def sample_spot_rays(... ) -> RayBundle:
    ...

```

建议不要直接保留当前 mode='init0'、mode='init'、mode='draw'、mode='spot' 这种字符串分支接口，而是拆成多个显式函数。

#### 4.2.6 evaluation 模块的建议接口

```python
@dataclass
class EvaluationConfig:
    use_vignetting: bool
    field_samples: torch.Tensor
    field_weights: torch.Tensor
    image_height_target: Optional[torch.Tensor] = None
    cra_target_deg: Optional[torch.Tensor] = None
    distortion_bound: Optional[tuple[float, float, float]] = None
    use_wavefront_error: bool = True
    use_spot_size: bool = True


@dataclass
class EvaluationResult:
    merit: torch.Tensor
    rms_spot_size: torch.Tensor
    rms_wavefront_error: torch.Tensor
    distortion: torch.Tensor
    cra_deg: torch.Tensor
    lca: Optional[torch.Tensor] = None
    aca: Optional[torch.Tensor] = None
    image_height: Optional[torch.Tensor] = None
    constraints: Optional[dict[str, torch.Tensor]] = None


def evaluate_image_quality(
    system: OpticalSystem,
    config: EvaluationConfig,
    cache: Optional[dict] = None,
) -> EvaluationResult:
    ...

```

#### 4.2.7 imaging 模块的未来接口预留

既然你已经考虑未来加图像仿真，建议现在就预留这一层，但不要求第一轮实现。

```python
@dataclass
class PSFResult:
    psf: torch.Tensor
    pixel_size_mm: float
    center_shift_mm: torch.Tensor


def simulate_psf(... ) -> PSFResult:
    ...


def simulate_mtf(... ) -> torch.Tensor:
    ...


def render_sensor_image(... ) -> torch.Tensor:
    ...

```

这样后面做图像仿真时，不需要再推翻前面的 tracing/evaluation 结构。

### 4.3 阶段 1 的交付物

1.  一份模块边界文档
    
2.  一份接口定义文档
    
3.  一份类型约定说明
    
4.  一份输入输出 shape 约定说明
    

## 5. 阶段 2：建立三类测试基线

这一阶段非常关键，建议在正式剥离之前完成。

### 5.1 测试类型 A：现有行为基线测试

目的：

保证第一次接口剥离之前，先记录当前实现的行为。

测试内容建议包括：

1.  单曲面追迹 平面/球面/非球面交点是否合理
    
2.  双面折射 简单透镜的出射方向是否合理
    
3.  多波长折射率一致性 同一材料在不同波长下折射率是否符合当前实现
    
4.  完整系统追迹 对一个固定结构，valid\_mask、交点、出射方向是否稳定
    
5.  merit 基线 对几个固定镜头，记录 merit 和关键指标
    

这些测试的价值是“冻结现状”，不是判断物理上是否最优。

### 5.2 测试类型 B：接口契约测试

目的：

保证新接口本身清晰可用。

建议覆盖：

1.  输入 shape 错误时是否给出明确异常
    
2.  batch 维和 ray 维是否支持广播或明确限制
    
3.  device 和 dtype 是否在接口间一致传递
    
4.  tracing/evaluation 结果字段是否完整
    
5.  warm\_start/cached data 缺失时是否能安全退化运行
    

### 5.3 测试类型 C：性能基准测试

目的：

建立后续优化的量化基线。

建议至少记录：

1.  单次 forward tracing 耗时
    
2.  单次 backward tracing 耗时
    
3.  calc\_enp\_all\_fov 耗时
    
4.  calc\_fit 耗时
    
5.  merit\_func\_diff 总耗时
    
6.  GPU 显存峰值
    
7.  不同 batch size 的吞吐变化
    

### 5.4 测试样例建议

建议至少准备三组固定测试系统：

#### 用例 1：最小系统

*   单球面或平板
    
*   用于验证求交、折射和法向
    

#### 用例 2：简单成像系统

*   1 到 2 片透镜
    
*   少量视场和波长
    
*   用于验证完整追迹与基础像质输出
    

#### 用例 3：接近真实项目的系统

*   当前项目中选一个中小规模设计
    
*   保留真实 Arch、波长和视场数量
    
*   用于验证重构后行为一致性和性能表现
    

### 5.5 阶段 2 的交付物

1.  regression test 集
    
2.  contract test 集
    
3.  benchmark 脚本
    
4.  一份 baseline 数据记录文件
    

## 7. 阶段 3：第一轮接口剥离

这一阶段的目标不是“彻底重写”，而是“先把可复用内核抠出来，并让旧代码还能跑”。

### 7.1 建议策略

采用适配器式剥离，而不是一次性大重构。

即：

1.  先在新模块里复制或迁移 tracing/evaluation 相关实现。
    
2.  在现有 DiffLensPopulation 中增加薄适配层。
    
3.  先让旧逻辑通过新接口调用新模块。
    
4.  确认行为一致后，再清理冗余代码。
    

### 7.2 第一轮剥离建议顺序

#### 步骤 1

先剥离 geometry 和 materials。

原因：

*   相对独立
    
*   最少依赖 self.order1
    

#### 步骤 2

再剥离 tracing。

目标：

*   trace\_rays
    
*   trace\_rays\_backward
    

#### 步骤 3

剥离 sampling。

把当前一个函数多模式的接口拆成多个显式函数。

#### 步骤 4

剥离 evaluation。

优先保留：

*   入瞳估计
    
*   spot / wavefront / distortion / CRA
    

后续再逐步纳入工程约束、重量和厚度约束。

### 7.3 这一阶段不要急着做的事

*   不要一开始就把所有 order1 缓存都搬进新库
    
*   不要先改优化器逻辑
    
*   不要先做接口美化而忽略回归一致性
    

### 7.4 阶段 3 的交付物

1.  optics\_core 初版目录
    
2.  旧代码到新接口的适配层
    
3.  首批回归通过记录
    

## 8. 阶段 4：围绕接口补测试

### 8.1 重点补哪些测试

1.  其他算法模块模拟调用测试 不通过 DiffLensPopulation，而是直接构建 OpticalSystem 和 RayBundle 调用 tracing/evaluation。
    
2.  配置组合测试 不同波长数、不同视场数、不同 batch 大小。
    
3.  错误处理测试 不完整输入、非法 surface 参数、非法 dtype/device。
    
4.  缓存兼容测试 有 warm\_start 和无 warm\_start 两种路径都应工作正常。
    

### 8.2 阶段 4 的交付物

1.  新接口直接调用示例
    
2.  契约测试覆盖率报告
    
3.  基础库最小使用说明
    

## 9. 阶段 5：性能优化

建议等接口和测试稳定之后再做，不要把“重构”和“优化”混在一起。

### 9.1 优先级最高的优化项

#### 优化 1：缓存

优先缓存：

*   材料在当前波长表下的折射率
    
*   pupil 采样模板
    
*   常用 linspace/index tensor
    
*   warm\_start 追迹中间结果
    

#### 优化 2：减少重复追迹

重点检查：

*   入瞳估计是否每次都必须全量重算
    
*   部分视场是否可以低频更新
    
*   merit 中不同阶段是否可以共享中间结果
    

#### 优化 3：减少 Python 循环

重点改写：

*   多波长 ray generation
    
*   多视场 ray generation
    
*   thickness 采样逻辑
    
*   某些 ray angle 统计逻辑
    

#### 优化 4：分层精度

建议评估：

*   float32 tracing + float64 final evaluation
    
*   float32 warm-up + float64 final confirmation
    

#### 优化 5：编译与图优化

待前面几个完成后，再评估：

*   torch.compile
    
*   CUDA Graph
    
*   Triton / CUDA fused kernel
    

### 9.2 性能优化的验收标准

建议不要只看单一耗时，而是同时看：

1.  单次 merit 耗时
    
2.  tracing 吞吐量
    
3.  GPU 显存占用
    
4.  数值偏差是否在可接受范围内
    
5.  端到端优化迭代总耗时
    

## 10. 阶段 6：图像仿真扩展

既然未来要支持图像仿真，建议从现在开始在接口层预留能力，但不强行在第一轮重构里全部实现。

### 10.1 推荐纳入的未来能力

1.  传感器采样
    
2.  卷积式成像仿真
    
3.  多视场、多波长图像形成
    

### 10.2 对当前设计的影响

为了支持未来图像仿真，当前 tracing/evaluation 接口最好保留以下信息：

*   surface\_intersections
    
*   opd
    
*   pupil sampling weights
    
*   chief ray 信息
    
*   sensor plane 几何信息
    

这些信息今天对成像仿真不是必须全用，但如果一开始就设计成可输出，将来扩展会轻松很多。

## 11. 建议的里程碑顺序

建议按下面 5 个里程碑推进。

### 里程碑 M1：接口定义完成

交付：

*   optics\_core 模块划分
    
*   初始 dataclass 和函数签名
    
*   shape/dtype/device 约定
    

### 里程碑 M2：基线测试完成

交付：

*   3 组测试系统
    
*   regression / contract / benchmark 三类测试
    
*   baseline 数据
    

### 里程碑 M3：第一轮剥离完成

交付：

*   tracing 与 evaluation 从 DiffOptics 中剥离
    
*   旧算法仍能跑通
    
*   回归测试通过
    

### 里程碑 M4：接口稳定化完成

交付：

*   直接调用新接口的示例
    
*   文档和测试完善
    
*   基础库可被其他算法模块直接调用
    

### 里程碑 M5：第一轮性能优化完成

交付：

*   缓存优化
    
*   重复追迹减少
    
*   至少一轮量化加速报告
    

## 12. 我建议你现在立刻做的事

如果只做最务实的下一步，我建议是：

1.  先把接口文档定下来
    
2.  同时准备 3 个固定测试系统
    
3.  写一版最小 benchmark 脚本
    
4.  然后开始第一轮 tracing/evaluation 剥离
    

不要先改优化器，也不要先追求“漂亮架构”。

第一轮的目标应该很明确：

“让旧代码通过新接口跑起来，并且结果与当前版本基本一致。”

只要做到这一点，后续测试完善和性能优化都会顺很多。

## 13. 简化版执行清单

如果要转换成更短的执行清单，可以直接按下面顺序做：

1.  明确 optics\_core 范围，只保留 tracing 和 evaluation 核心。
    
2.  定义 OpticalSystem、RayBundle、TraceResult、EvaluationResult 四类核心接口。
    
3.  建立三类测试：基线回归、接口契约、性能基准。
    
4.  从 geometry/materials 开始剥离，再剥 tracing，再剥 evaluation。
    
5.  用旧算法适配层接新接口，确保回归一致。
    
6.  在新接口稳定后，再做缓存、减少重复追迹和 Python 循环优化。
    
7.  最后再扩展 PSF、MTF 和图像仿真。