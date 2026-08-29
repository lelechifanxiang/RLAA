# auto\_od 项目概览

**仓库地址：** [**https://github.com/LiGpy/auto\_od/tree/aod\_v2\_web**](https://github.com/LiGpy/auto_od/tree/aod_v2_web)

## 1. 项目定位

这个项目实现了一套基于 PyTorch 的自动光学设计流程，核心目标是：

1.  根据输入的镜头规格参数自动生成初始镜头结构。
    
2.  在 GPU 上并行评估大量候选镜头。
    
3.  通过“近轴像差预优化 + 全光线追迹精评 + 变异/局部优化”的组合策略持续改进解。
    
4.  输出可视化结果、结构数据以及可导入 Zemax 的中间结果。
    

从代码实现看，它不是一个通用的机器学习项目，而是一个“几何光学 + 像差理论 + 元启发式优化 + GPU 并行数值计算”的工程化系统。

## 2. 主入口与执行链路

当前主入口是 GYOptics/DiffOptics.py 中的 DiffLensPopulation.run。

完整调用链通常是：

1.  DesignLensV2.py 负责从 Excel 或 JSON 中读取设计规格，组装参数字典。
    
2.  auto\_od\_wrapper.py 对参数做预处理，并把参数写入 GYOptics.DiffLensPopulation.basics。
    
3.  GYOptics/DiffOptics.py 执行真正的自动设计、优化、评估和结果导出。
    

对应的数据流可以概括为：

规格输入 -> 参数预处理 -> 初始化种群 -> 近轴预筛选 -> 精确光线追迹评价 -> 变异/Adam 优化 -> 导出 JSON/Excel/布局图

## 3. 核心算法在做什么

### 3.1 参数化方式

镜头系统被编码成一个一维参数向量，按照表面顺序展开。

每个表面至少包含：

*   曲率 c
    
*   厚度/间距 d
    

如果该表面后面是实体材料（G/M/P），还会包含：

*   折射率 n
    
*   阿贝数 V
    

如果该表面允许非球面，还会继续包含：

*   圆锥系数 k
    
*   更高阶非球面系数 ai
    

Arch 字段决定结构拓扑，典型编码包括：

*   SG0: 玻璃球面
    
*   SM8: 模压类/混合材料高阶面
    
*   SP8: 塑料高阶面
    
*   SA0 / SA8: 空气面或对应阶次的表面
    

因此，这个项目优化的不是离散“镜头模板”，而是固定拓扑下的一组连续光学参数，材料也允许在库中切换。

### 3.2 run 主流程分阶段说明

DiffLensPopulation.run 大致可以拆成下面几个阶段。

#### 阶段 A：初始化或续算

如果 basics.init\_read\_root 不为空：

*   说明从已有结果继续优化。
    
*   init\_pop 会从历史 Excel 中恢复已有结构。
    

如果 basics.init\_read\_root 为空：

*   先随机生成大规模初始种群。
    
*   调用 init\_opti 做一轮基于近轴理论的快速预优化。
    
*   再用 merit\_func\_diff 做第一次全量精评。
    
*   然后进入 AdamO 做局部连续优化，得到一批多样化的可行初始解。
    

#### 阶段 B：epoch 级迭代优化

每个 epoch 中主要执行：

1.  set\_period 根据当前 epoch 决定开放哪些非球面阶次参与优化，并设置局部可变参数掩码。
    
2.  merit\_func\_diff 对当前 parents 进行精确评价。
    
3.  保留更优个体 按评价函数排序，只留下有效且较优的父代。
    
4.  mutation\_all 做材料替换或非球面变异。
    
5.  AdamO 使用有限差分近似梯度的 Adam 优化继续细化局部解。
    
6.  draw\_all 输出当前阶段最优镜头的结构图、JSON、Excel 等结果。
    

整体策略是典型的“启发式搜索 + 局部连续优化”混合框架。

### 3.3 两级评价体系

这个项目有两套明显不同的评价函数。

#### 第一层：近轴快速评价

由 fitness\_func\_para 负责，主要用于初始化阶段的大批量预筛选。

它基于：

*   一阶近轴量
    
*   赛德尔像差
    
*   中心/边缘厚度约束
    
*   焦距、总长、面倾角、光线角度等工程约束
    

特点是：

*   快
    
*   可对超大种群做初筛
    
*   能在完整光线追迹前淘汰掉大量不合理结构
    

#### 第二层：完整精确评价

由 merit\_func\_diff 负责，它是主评价函数。

内部流程是：

1.  set\_basics 把参数向量装配成真实的 surfaces 和 materials。
    
2.  calc\_order1\_data 计算 EFL、入瞳位置、出瞳位置等一阶量。
    
3.  calc\_enp\_all\_fov 或 calc\_enp\_all\_fov\_vig 对不同视场和波长做入瞳/主光线估计，处理渐晕与有效孔径。
    
4.  calc\_fit 进行完整光线追迹，并汇总所有成像指标和工程约束。
    

calc\_fit 中综合了以下几类目标：

*   RMS spot size
    
*   RMS wavefront error
    
*   横向色差 LCA
    
*   轴向色差 ACA
    
*   CRA
    
*   畸变
    
*   像高约束
    
*   总长 TTL
    
*   焦距 EFL
    
*   后焦距 BFL
    
*   面厚、边厚、空气间隔
    
*   径厚比/厚薄比
    
*   表面倾角
    
*   光线最大入射角
    
*   近似重量
    

最终使用加权 RMS 方式，把“成像质量项”和“物理可实现性项”合成为一个标量 merit。

### 3.4 为什么叫 DiffOptics

文件名里的 Diff 更接近“可微分/可数值优化”的含义，而不是传统自动求导端到端训练。

这里的“可微”主要体现在：

*   绝大多数光学量都用 torch.Tensor 表达，可在 GPU 上批量算。
    
*   一部分初始化阶段直接用 autograd 反传。
    
*   主局部优化阶段 AdamO 使用有限差分方式近似梯度。
    

所以它本质上是“基于 torch 张量化实现的自动光学优化引擎”。

## 4. 关键模块划分

### 4.1 外层接口与任务入口

#### auto\_od\_wrapper.py

职责：

*   提供 AutoOdWrapper，作为外部系统调用入口。
    
*   把用户原始参数转成内部可接受格式。
    
*   创建 DataWriter，把结果通过队列回传给上层进程或前端。
    

重点：

*   preprocess\_param 会补齐可选参数。
    
*   自动设置 device、piece\_num、use\_aspheric、COVA 等内部字段。
    
*   run 本身只是转调 self.pop.run()。
    

适合看作：

“算法内核的服务层封装”。

#### DesignLensV2.py

职责：

*   命令行入口。
    
*   从 Excel 或 JSON 文件中读取项目规格。
    
*   把业务层规格转换成内核所需的 Basics 参数字典。
    

重点函数：

*   get\_param\_from\_excel
    
*   get\_design\_from\_json
    
*   do\_design
    

这个文件承担的是：

“规格输入适配层”。

### 4.2 核心算法层

#### GYOptics/DiffOptics.py

这是当前项目的主实现，也是最核心的模块。

主要职责可以再细分为 6 个子层：

1.  种群初始化 init\_pop 根据 Arch、材料库、厚度范围、固定参数约束生成候选参数向量。
    
2.  近轴预优化 init\_opti fitness\_func\_para seidel 用赛德尔像差和近轴约束对大种群做快速筛选和粗优化。
    
3.  光学系统装配 set\_basics calc\_order1\_data 把参数向量映射成具体曲面、材料和一阶光学量。
    
4.  光线采样与追迹 sample\_ray\_gus sample\_ray\_common sample\_ray\_aper \_trace / \_forward\_tracing / \_backward\_tracing 用 GPU 并行方式对多视场、多波长、多光线进行追迹。
    
5.  评价与优化 merit\_func\_diff calc\_fit GlobalOpti AdamO mutation\_all / mutation\_n / mutation\_asph 构成完整的混合优化器。
    
6.  结果导出与可视化 draw\_all plot\_setup2D\_with\_trace\_whole plot\_spot\_diagram write\_optics\_data 导出 png、json、xlsx，并把结果送到 writer。
    

一句话总结这个文件：

它既是求解器，也是评估器，还是结果生成器。

#### GYOptics/BaseClass.py

职责：

*   定义核心状态容器 Basics 和 Order1。
    

Basics 更像“输入规格 + 全局配置”：

*   设计规格，如 EFL、TTL、BFL、FOV、CRA、IH。
    
*   结构约束，如 Arch、A\_Idx、厚度范围、材料库路径。
    
*   优化参数，如 epoch\_num、init\_pop\_num、max\_pop、local\_iter。
    
*   公差参数，如曲率半径公差、厚度公差、偏心公差等。
    

Order1 更像“运行时缓存 + 中间结果仓库”：

*   参数索引表 c\_list / d\_list / n\_list / ai\_list。
    
*   光学系统对象 surfaces / materials。
    
*   有效索引、视场采样、入瞳位置、像面信息。
    
*   评价项缓存，如 rms\_spot\_size、distort、sur\_angle、fit\_all。
    

这两个类把“输入规格”和“运行态结果”分离得比较清楚。

#### GYOptics/SurfaceClass.py

职责：

*   提供底层光学对象和追迹所需的基础数学模型。
    

主要类：

*   Surfaces 抽象曲面基类，包含有效口径判断、法向计算、牛顿法求交。
    
*   Aspheric 具体的轴对称非球面曲面实现，支持圆锥项和高阶偶次项。
    
*   Ray 几何光线对象，保存起点、方向和波长。
    
*   MaterLab 从 Excel 材料库读取折射率、阿贝数、价格、密度与色散公式。
    
*   Material 具体材料对象，负责按波长返回折射率。
    

这一层是整个算法的“光学物理基础设施”。

### 4.3 其他算法分支或历史实现

#### GYOptics/ParaOptics.py

从结构上看，这是一个更早期或平行版本的镜头种群优化器 LensPopulation。

特点：

*   与 DiffOptics.py 有大量相似逻辑。
    
*   同样包含 init\_pop、GlobalOpti、AdamO、mutation 等方法。
    
*   但接口更旧，没有现在的 writer 输出链路。
    
*   主评价函数名是 merit\_func，而不是 merit\_func\_diff。
    

可以把它理解为：

“旧版/备用版自动设计实现”。

#### GYOptics/diff\_optics.py

这个文件定义的是 LensFinetune。

从内容看，它更偏向单镜头微调、渐晕因子探索和局部精修，而不是当前主流程使用的大规模自动设计入口。

可以理解为：

“实验性或专项优化模块”。

#### GYOptics/init.py

这个文件直接导出：

*   ParaOptics
    
*   DiffOptics
    

说明包设计上默认把这两个实现都暴露给外部调用。

### 4.4 结果导出与外部工具层

#### json2zmx.py

职责：

*   把 draw\_all 生成的 JSON 结果恢复成 Zemax OpticStudio 的 zmx 文件。
    
*   通过 ZOS-API 设置系统孔径、视场、波长、每一面曲率、厚度、材料和非球面系数。
    

适合看作：

“JSON -> Zemax 的桥接器”。

#### excel2zmx.py

职责：

*   从 Excel 结果中提取某个镜头结构。
    
*   转写为 Zemax 文件。
    

它主要服务于旧结果格式，因此更像历史兼容工具。

#### PythonStandaloneApplication.py

职责：

*   封装 Zemax ZOS-API 的 Python.NET 初始化过程。
    
*   建立与 OpticStudio 的连接。
    

本身不做光学优化，只是外部 CAD/光学软件接口层。

### 4.5 其他目录

#### gaoptics

当前目录基本为空，没有形成有效代码模块，可能是历史遗留目录或预留命名空间。

## 5. 关键数据结构和状态流

### 5.1 Basics：输入规格中心

Basics 几乎收纳了所有设计目标和工程约束，包括：

*   光学规格：EFL、IH、CRA、FOV、BFL、TTL
    
*   结构规格：Arch、A\_Idx、Max\_R、Filter、材料库
    
*   工艺约束：中心厚度、边缘厚度、径厚比、圆锥系数范围
    
*   优化控制：epoch\_num、local\_iter、global\_iter、max\_pop
    
*   计算控制：device、data\_type、rings、arms、use\_vignetting
    

### 5.2 Order1：运行态结果中心

Order1 负责缓存运行过程中产生的大量中间量，典型包括：

*   surfaces / materials
    
*   effl / enpp / enpd / expp
    
*   valid\_idx / init\_pop\_idx
    
*   enp\_xyz / object\_chief / chief\_img
    
*   rms\_spot\_size / rms\_wavefront\_error / distort / LCA\_value / ACA\_value
    
*   thickness\_data / Weight\_data / conic\_data / fit\_all
    

如果要调试算法，大部分运行态信息都在这里找。

## 6. 输入、输出和落盘结果

### 6.1 输入

项目支持两种主要输入方式：

1.  Excel 规格表 由 DesignLensV2.get\_param\_from\_excel 解析。
    
2.  JSON 项目参数 由 DesignLensV2.get\_design\_from\_json 解析。
    

输入内容主要包括：

*   架构 Arch
    
*   孔径定义 A\_Typ / A\_Val / A\_Idx
    
*   视场定义 F\_Typ / F\_Val / F\_Spl / F\_Wt
    
*   波长及权重
    
*   EFL、IH、TTL、BFL、CRA、Dist、Ray\_Ang、Sur\_Ang
    
*   厚度、材料库、公差、重量等约束
    

### 6.2 输出

主输出由 DiffOptics.draw\_all 产生，包含：

1.  Excel data\_epochX.xlsx 保存每轮最优结构的表面参数。
    
2.  JSON data\_epochX\_idxY\_mfvZ.json 保存系统参数、像质指标、表面数据、MTF、点列图数据。
    
3.  PNG layout\_epochX\_idxY\_mfvZ.png 保存镜头布局图和光线示意。
    
4.  队列消息 由 DataWriter.output\_data 发送给上层系统。
    

值得注意的是：

*   run 没有直接 return 设计结果。
    
*   它的结果主要通过文件输出和 writer 回调传播。
    

## 7. 项目分层理解建议

如果把整个项目按工程角色分层，可以这样理解：

### 第 1 层：业务入口层

*   DesignLensV2.py
    
*   auto\_od\_wrapper.py
    

负责把外部规格转成内核参数，并启动任务。

### 第 2 层：算法调度层

*   GYOptics/DiffOptics.py 中的 DiffLensPopulation.run
    

负责控制初始化、迭代、变异、局部优化、结果导出。

### 第 3 层：物理建模与评价层

*   set\_basics
    
*   calc\_order1\_data
    
*   calc\_enp\_all\_fov
    
*   calc\_fit
    
*   fitness\_func\_para
    

负责把参数变成真实光学系统，并计算像质和工程约束。

### 第 4 层：基础光学对象层

*   GYOptics/SurfaceClass.py
    
*   GYOptics/BaseClass.py
    

负责曲面、材料、光线、状态缓存等基础建模。

### 第 5 层：外部生态接口层

*   json2zmx.py
    
*   excel2zmx.py
    
*   PythonStandaloneApplication.py
    

负责和 Zemax/OpticStudio 互通。

## 8. 对这个项目的整体判断

从当前代码结构看，这个项目的主旨可以总结为：

“在给定镜头拓扑和工程约束下，利用 GPU 并行和混合优化策略，自动搜索一批满足像质与制造约束的镜头设计方案，并输出可继续在 Zemax 中分析的结果。”

它不是简单的遗传算法，也不是单纯的可微光线追迹，而是三种思想的混合：

*   大规模并行候选生成
    
*   近轴理论快速预筛
    
*   精确光线追迹驱动的局部迭代优化
    

因此，最值得优先掌握的代码顺序是：

1.  auto\_od\_wrapper.py
    
2.  DesignLensV2.py
    
3.  GYOptics/DiffOptics.py 中的 run / init\_pop / merit\_func\_diff / calc\_fit / AdamO
    
4.  GYOptics/SurfaceClass.py
    
5.  GYOptics/BaseClass.py
    

按这个顺序读，最容易先建立全局理解，再逐步深入细节。

## 9. 如果要抽取“GPU 并行光线追迹和像质评价”基础库

如果目标是把现有项目中的“GPU 并行光线追迹 + 像质评价”抽成一个可复用基础库，最值得优先关注的不是 run 本身，而是 run 内部调用的这条核心链：

参数向量或镜头结构 -> 光学系统装配 -> 光线采样 -> 批量追迹 -> 像质指标计算 -> 评价值输出

换句话说，真正应该抽出去的是“求值内核”，而不是“优化调度器”。

### 9.1 最值得重点阅读的代码

#### 第一优先级：底层光学对象与求交内核

文件：GYOptics/SurfaceClass.py

重点看：

*   Surfaces
    
*   Aspheric
    
*   Ray
    
*   Material
    
*   MaterLab
    
*   ray\_surface\_intersection
    
*   newtons\_method\_new
    
*   newtons\_method\_impl
    
*   surface\_and\_derivatives\_dot\_D
    
*   ior
    

这部分决定了：

*   曲面几何如何表达
    
*   光线如何表示
    
*   光线与曲面的交点如何求解
    
*   折射率如何随波长计算
    

如果要抽基础库，这一层基本是必须保留的，因为它是追迹器的物理核心。

#### 第二优先级：系统装配层

文件：GYOptics/DiffOptics.py

重点看：

*   set\_basics
    
*   calc\_order1\_data
    

这两部分负责把参数向量转换成真正可追迹的系统对象：

*   surfaces 列表
    
*   materials 列表
    
*   一阶量，如 EFL、ENPP、ENPD、EXPP、TTL
    

如果未来其他算法模块不一定使用当前的参数向量编码，也可以把这一层改造成两种接口：

1.  从参数向量构建系统
    
2.  直接从显式 surface/material 定义构建系统
    

这样基础库就不会被当前这套优化器耦死。

#### 第三优先级：光线采样与追迹主循环

文件：GYOptics/DiffOptics.py

重点看：

*   sample\_ray\_gus
    
*   sample\_ray\_common
    
*   sample\_ray\_aper
    
*   \_refract
    
*   \_trace
    
*   \_forward\_tracing
    
*   \_backward\_tracing
    

这部分是“GPU 并行追迹引擎”的直接主体。

其中：

*   sample\_ray\_gus 负责用于像质评价的高斯型 pupil/ray 采样。
    
*   sample\_ray\_common 负责入瞳搜索、绘图、spot 等多种采样模式。
    
*   sample\_ray\_aper 负责为入瞳反推构造光线。
    
*   \_trace / \_forward\_tracing / \_backward\_tracing 负责真正的批量追迹。
    

这部分抽出去后，应当能形成一个更通用的接口，例如：

*   trace\_rays(system, ray\_bundle)
    
*   trace\_rays\_backward(system, ray\_bundle, stop\_surface)
    
*   sample\_rays\_from\_pupil(...)
    

#### 第四优先级：像质评价层

文件：GYOptics/DiffOptics.py

重点看：

*   calc\_enp\_all\_fov
    
*   calc\_enp\_all\_fov\_vig
    
*   calc\_fit
    
*   merit\_func\_diff
    

这四部分合起来，构成了“从可追迹系统到像质 merit”的完整闭环。

职责分工大致是：

*   calc\_enp\_all\_fov / calc\_enp\_all\_fov\_vig  估计多视场、多波长下的入瞳位置、主光线与有效孔径。
    
*   calc\_fit  基于完整追迹结果计算 spot、wavefront、CRA、distortion、厚度、重量等指标。
    
*   merit\_func\_diff  把系统装配、入瞳估计和最终评价串起来，输出统一标量评价值。
    

如果希望基础库兼容不同优化算法，建议把 calc\_fit 再往下拆成：

*   几何像质指标模块
    
*   波前指标模块
    
*   工程约束模块
    
*   总体汇总模块
    

这样上层算法就可以自由选择只调用某一部分指标，而不是被迫使用当前项目定义好的综合 merit。

### 9.2 不建议优先抽出的部分

如果目标是抽“基础库”，下面这些反而不应作为第一优先级：

*   run
    
*   init\_pop
    
*   init\_opti
    
*   GlobalOpti
    
*   AdamO
    
*   mutation\_all
    
*   mutation\_n
    
*   mutation\_asph
    
*   draw\_all
    
*   write\_optics\_data
    

原因很简单：

*   这些逻辑强耦合于当前的元启发式搜索框架。
    
*   它们更像“算法策略层”，不是“通用物理计算层”。
    
*   其他算法模块真正需要复用的，通常是追迹和评价，不是当前这套搜索器。
    

### 9.3 推荐的基础库拆分方式

如果做模块化重构，比较合理的拆法是：

#### 1. optics\_core.geometry

包含：

*   Surfaces
    
*   Aspheric
    
*   Ray
    

#### 2. optics\_core.materials

包含：

*   MaterLab
    
*   Material
    

#### 3. optics\_core.system

包含：

*   从参数向量或结构定义装配 system 的逻辑
    
*   当前 set\_basics 和 calc\_order1\_data 的核心能力
    

#### 4. optics\_core.sampling

包含：

*   sample\_ray\_gus
    
*   sample\_ray\_common
    
*   sample\_ray\_aper
    

#### 5. optics\_core.tracing

包含：

*   \_refract
    
*   \_trace
    
*   \_forward\_tracing
    
*   \_backward\_tracing
    

#### 6. optics\_core.evaluation

包含：

*   calc\_enp\_all\_fov
    
*   calc\_enp\_all\_fov\_vig
    
*   calc\_fit
    
*   各单项指标计算函数
    

#### 7. algorithms

保留当前：

*   DiffLensPopulation.run
    
*   初始化、变异、局部优化、全局优化
    

这样其他算法模块只依赖 optics\_core，而不必依赖整个 DiffLensPopulation。

### 9.4 抽库时最需要先解耦的问题

当前代码最大的问题不是“算不出来”，而是状态耦合比较重。

主要耦合点包括：

1.  大量函数依赖 self.basics 和 self.order1 的隐式状态。
    
2.  \_trace 读取 self.order1.surfaces 和 self.order1.materials，而不是显式传参。
    
3.  calc\_fit 会读写很多运行时缓存，既做计算，又做状态保存。
    
4.  merit\_func\_diff 既负责调度，又负责容错和结果回填。
    

因此抽库时最先要做的不是重写数学，而是把函数改成更像纯函数：

*   输入明确
    
*   输出明确
    
*   少依赖 self 上的隐藏状态
    
*   把“缓存”和“计算”拆开
    

这是后续支持其他算法模块复用的关键。

## 10. 现有 GPU 并行光线追迹是否还有加速空间

结论是：有，而且空间不小。

当前实现已经把“多光线、多波长、多候选镜头”批量张量化到了 GPU 上，这一点是正确方向；但代码里仍然保留了不少 Python 控制流、重复张量构造和可缓存却未缓存的计算，因此还有明显优化余地。

### 10.1 已经做得比较好的部分

当前实现已经具备以下优势：

1.  光线、波长、候选镜头总体上是批量张量化计算，而不是单条光线逐条追迹。
    
2.  曲面求交使用 torch 张量，Newton 迭代是在 GPU 上跑的。
    
3.  折射、法向、像面交点、RMS spot、RMS wavefront 等关键量大多是并行算的。
    
4.  整个 merit\_func\_diff 面向的是候选种群，而不是单个镜头。
    

这意味着整体架构已经具备“可继续深挖 GPU 性能”的基础。

### 10.2 主要瓶颈在哪里

#### 瓶颈 A：一次完整评价内部要做多次追迹

calc\_enp\_all\_fov 本身就包含三段追迹：

1.  反向追迹，估计入瞳位置。
    
2.  正向追迹，估计有效孔径和 chief ray 信息。
    
3.  再次正向追迹，修正多视场入瞳边界。
    

之后 calc\_fit 还会再做一次全量追迹。

也就是说，一次 merit\_func\_diff 并不是“一次追迹”，而是“多阶段追迹组合”。

如果外层优化器频繁调用 merit\_func\_diff，那么实际总耗时会被快速放大。

#### 瓶颈 B：Python 层循环仍然很多

虽然底层算子在 GPU 上，但上层仍然存在不少 Python 循环，例如：

*   对波长循环
    
*   对视场循环
    
*   对表面循环
    
*   对厚度采样点循环
    

尤其是在：

*   calc\_enp\_all\_fov
    
*   calc\_enp\_all\_fov\_vig
    
*   calc\_fit
    
*   \_forward\_tracing
    
*   \_backward\_tracing
    

这些循环会带来：

*   kernel launch 次数增加
    
*   Python 调度开销增加
    
*   后续 torch.compile 难以发挥最大效果
    

#### 瓶颈 C：Material.ior 中有重复构造张量的开销

Material.ior 在按波长求折射率时，会反复构造和搬运与波长表相关的 Tensor。

而 \_trace 在每个表面上都会调用两次 ior：

*   当前介质一次
    
*   下一介质一次
    

这意味着在“表面数 × 追迹次数 × 优化迭代次数”这个规模下，会产生不少本可避免的重复开销。

#### 瓶颈 D：Newton 迭代包含可能触发同步的控制流

newtons\_method\_impl 中使用了这种条件：

*   residual 是否还有元素超过阈值
    
*   是否达到最大迭代次数
    

这种带 any/reduction 的 while 控制流，容易在 GPU 上引入同步点，也不利于进一步编译优化。

#### 瓶颈 E：很多采样模板每次都在重建

例如：

*   rings/arms 对应的采样半径和权重
    
*   linspace 网格
    
*   某些中间索引张量
    

在 sample\_ray\_gus、sample\_ray\_common、calc\_enp\_all\_fov 中，这些张量会反复创建。

如果评估次数很多，这部分开销会累计得很明显。

#### 瓶颈 F：默认精度偏高

项目默认使用 torch.float64。

这在光学计算中有合理性，但从 GPU 吞吐上看，float64 通常明显慢于 float32。对于某些阶段，如果允许混合精度或分阶段精度策略，仍有明显提速空间。

### 10.3 最值得优先做的加速方向

如果按投入产出比排序，优先建议如下。

#### 优先级 1：缓存材料色散结果和采样模板

建议缓存：

*   每个 material 在当前 wave\_sample 下的折射率表
    
*   rings/arms 对应的 pupil 采样模板
    
*   反复使用的 linspace/index tensor
    

这是改动风险较低、收益通常很稳定的一类优化。

#### 优先级 2：把追迹器改成更“纯”的批量接口

当前 \_trace 和 \_forward\_tracing 强依赖 self.order1。建议改成：

*   输入 system
    
*   输入 ray bundle
    
*   返回 trace result
    

一旦函数边界更清晰，就更容易：

*   做缓存
    
*   做 benchmark
    
*   用 torch.compile
    
*   用 CUDA Graph
    

#### 优先级 3：减少评价中的重复追迹

要重点评估：

*   calc\_enp\_all\_fov 的三阶段追迹是否都必须每次完整执行
    
*   某些阶段能否沿用上一轮 Adam 迭代的结果做 warm start
    
*   某些视场/波长的入瞳估计能否低频更新，而不是每次 merit 都完全重算
    

当前代码已经有 oss\_all\_1 到 oss\_all\_4 这样的 warm-start 痕迹，说明作者已经意识到这个方向是有效的，但还没有完全发展成系统级缓存机制。

#### 优先级 4：减少 Python 循环，提升算子融合机会

比较值得改写的部分包括：

*   多波长、多视场的光线生成
    
*   部分 thickness 采样逻辑
    
*   ray angle 统计逻辑
    

目标不是“完全无循环”，而是把高频、大张量路径尽量合并，让 GPU 做更长的连续计算。

#### 优先级 5：重新审视 float64 的使用范围

可以考虑分层精度策略：

1.  入瞳估计、初筛、采样生成用 float32。
    
2.  完整像质评价保留 float64。
    
3.  或者 Adam 内部多次局部迭代用 float32，最终确认阶段再回到 float64。
    

这类策略通常需要用数值误差实验验证，但潜在收益很大。

### 10.4 可以进一步考虑的中长期优化方向

#### 方向 A：torch.compile

在减少动态控制流、减少 .item() 和 shape 变动之后，可以尝试把追迹和评价主路径交给 torch.compile。

当前代码里动态分支不少，直接上 compile 收益不一定稳定，但重构后值得测试。

#### 方向 B：CUDA Graph

在 AdamO 这类“同一形状反复评估”的阶段，CUDA Graph 可能有价值，因为它可以减少重复 launch 开销。

前提是：

*   输入 shape 基本固定
    
*   控制流足够稳定
    

#### 方向 C：自定义 CUDA / Triton 内核

如果未来确认瓶颈主要集中在：

*   曲面求交迭代
    
*   法向与折射更新
    
*   多表面串行追迹
    

可以进一步考虑针对交点求解和追迹主循环写更底层的 fused kernel。

这一步改造成本高，不建议作为第一阶段动作，但如果要把它做成长期复用的基础库，这是一条明确可行的路线。

### 10.5 实际上的优先建议

如果你的目标是“先抽库，再让其他算法调用”，建议按下面顺序推进：

1.  先把 SurfaceClass.py 和 DiffOptics.py 中的追迹/评价核心函数从优化器逻辑中剥离出来。
    
2.  把 self.basics / self.order1 的隐式依赖改成显式输入输出。
    
3.  对抽出的 tracing 和 evaluation 做独立 benchmark。
    
4.  先做缓存类优化和模板复用。
    
5.  再考虑 compile、mixed precision 和更深层的 kernel 优化。
    

这样做的原因是：

*   先抽库，才能独立测性能。
    
*   先建立稳定接口，后面的性能优化才不会反复返工。
    
*   当前最大的收益，很可能先来自“减少重复计算和重复分配”，而不是一开始就改底层数学。