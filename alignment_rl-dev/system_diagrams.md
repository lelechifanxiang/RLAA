# RLAA系统架构图

## 图1：光学建模体系

```mermaid
flowchart TB
    subgraph INPUT["输入"]
        ZMX["光学处方\n(ZMX)"]
        ALIGN["对准参数\n(dx, dy)"]
        TOL["制造公差"]
    end
    
    subgraph OPTICAL["光学计算引擎"]
        direction LR
        SYS["光学系统\n构建"]
        TRACE["光线追迹"]
        PSF["PSF计算"]
        MTF["MTF提取"]
    end
    
    subgraph OUTPUT["输出"]
        RAW["原始MTF\n多视场×多频点"]
        REL["相对质量\nq=log(MTF/MTF_ref)"]
    end
    
    ZMX --> SYS
    ALIGN --> SYS
    TOL --> SYS
    SYS --> TRACE
    TRACE --> PSF
    PSF --> MTF
    MTF --> RAW
    RAW --> REL
    
    style INPUT fill:#e1f5ff,stroke:#0288d1,stroke-width:3px
    style OPTICAL fill:#fff9e1,stroke:#f57c00,stroke-width:3px
    style OUTPUT fill:#e8f5e9,stroke:#388e3c,stroke-width:3px
```

## 图2：RL训练体系

```mermaid
flowchart TB
    subgraph ENV["环境层"]
        direction LR
        RESET["Episode初始化\n公差+基线+初始状态"]
        STATE["状态管理\n对准参数+公差"]
        OBS["观测生成\nMTF历史+动作历史"]
        REWARD["奖励计算\n质量改进+成功奖励"]
        
        RESET --> STATE
        STATE --> OBS
        STATE --> REWARD
    end
    
    subgraph PHYSICS["物理层"]
        OPTICS["光学建模\n(见图1)"]
    end
    
    subgraph ALGO["算法层"]
        direction LR
        POLICY["策略网络\n观测→动作"]
        VALUE["价值网络\n状态价值估计"]
        BUFFER["经验回放\n样本存储与采样"]
        UPDATE["梯度更新\n策略优化"]
        
        POLICY --> BUFFER
        VALUE --> BUFFER
        BUFFER --> UPDATE
        UPDATE --> POLICY
        UPDATE --> VALUE
    end
    
    subgraph PARALLEL["并行训练"]
        ENV_BATCH["12个环境\n共享物理引擎"]
    end
    
    subgraph EVAL["评估层"]
        METRICS["训练指标\nep_len/rew/success"]
        BASELINE["Baseline对比\nHill Climbing等"]
        
        METRICS --> BASELINE
    end
    
    OBS --> POLICY
    POLICY --> |动作| STATE
    STATE --> OPTICS
    OPTICS --> OBS
    OPTICS --> REWARD
    REWARD --> BUFFER
    
    ENV -.实例化.-> ENV_BATCH
    ENV_BATCH --> ALGO
    ALGO --> EVAL
    
    style ENV fill:#fff9e1,stroke:#f57c00,stroke-width:3px
    style PHYSICS fill:#e1f5ff,stroke:#0288d1,stroke-width:3px
    style ALGO fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    style PARALLEL fill:#ffe1e1,stroke:#d32f2f,stroke-width:3px
    style EVAL fill:#e8f5e9,stroke:#388e3c,stroke-width:3px
```

## 系统架构说明

### 光学建模体系
- **输入层**：光学设计、对准参数、制造误差
- **计算核心**：光线追迹 → PSF → MTF
- **输出层**：原始MTF → 归一化质量指标

### RL训练体系
- **物理层**：光学建模提供环境反馈
- **环境层**：状态管理、观测生成、奖励计算
- **算法层**：SAC算法（Actor-Critic + 经验回放）
- **并行层**：多环境共享物理引擎
- **评估层**：性能统计与对比
