# AlignmentRL

AlignmentRL 是一个用于精密光学镜头对准的强化学习实验项目。目标是在虚拟仿真环境中训练智能体，通过尽可能少的测量步数完成高精度对准。

当前项目聚焦于 Double Gauss 六片镜头的主动对准任务：默认场景为第二个 Coordinate Break 镜组的二维偏心对准，质量指标为 episode 基线相对 log MTF 增益。

项目以 Gymnasium 环境为核心，配套提供 RL 训练、基线算法、评估与可视化工具。详细使用方式、命令示例和扩展说明请直接查看相关文件。

## 目录结构

```text
AlignmentRL/
├── config.py                     # 全局配置
├── train.py                      # 训练入口
├── evaluate.py                   # 评估入口
├── visualize_episode.py          # 单次 episode 可视化
├── env/                          # 物理模型与 Gymnasium 环境
├── agents/                       # 传统基线算法
├── utils/                        # 通用工具与绘图函数
├── visualization/                # 任务专用可视化渲染
├── script/                       # 批处理与汇报脚本
├── docs/                         # 补充文档
├── models/                       # 训练输出模型
├── logs/                         # TensorBoard 日志
├── results/                      # 评估结果与可视化输出
└── figures/                      # 评估生成的图像
```

## 进一步阅读

- docs/operations.md：环境准备、训练、评估和可视化命令
- docs/algorithm_notes.md：强化学习相关说明
- config.py：环境、镜片组与训练超参数定义
