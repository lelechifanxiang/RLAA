# P0-6 异构镜头 batch 测试文件

这 3 个 ZMX 来自同一规格目录 `LensDatabase/DS_1`，用于开发和验证 `optics_core` 的异构材料 batch。

共同条件：

- 架构：`SGAGAGAGAGAGA`
- 表面数：15（物面 + 前置光阑 + 12 个折射面 + 像面）
- HFOV：5°
- Fno：6
- EFL：248 mm
- TTL：308 mm
- BFL：188 mm
- 波长：486、588、656 nm

材料序列：

| 文件 | 6 片玻璃材料序列 |
|---|---|
| `sg6_material_a.zmx` | H-ZK3, H-LAF10LA, H-ZF5, H-K9L, H-LAK52, H-ZF13 |
| `sg6_material_b.zmx` | H-ZPK2A, H-ZK9B, H-ZK21, H-ZF72A, H-K51, H-ZK21 |
| `sg6_material_c.zmx` | H-ZLAF52A, H-ZLAF4LA, H-K9L, H-ZF72A, H-LAF50B, H-K9L |

说明：

- 三个文件拓扑和目标规格相同，均为独立的真实数据库设计，因此半径和厚度也存在差异。
- 文件保持原始 Zemax UTF-16 LE 编码。
- 可用于验证异构 batch 与逐文件单系统追迹是否一致。

## 不同 HFOV 和 F/# 的补充文件

以下两个文件仍采用相同架构 `SGAGAGAGAGAGA` 和相同的 15 面组织，但视场、F 数及其他处方参数不同：

| 文件 | 数据来源 | HFOV | F/# | EFL | TTL | BFL | 6 片玻璃材料序列 |
|---|---|---:|---:|---:|---:|---:|---|
| `sg6_hfov25_f3p0.zmx` | `DS_106220/design1.zmx` | 25° | 3.0 | 47 mm | 107 mm | 20 mm | H-ZLAF4LA, H-ZF7LA, H-ZLAF55D, H-ZPK5, H-LAK52, H-ZLAF50E |
| `sg6_hfov34_f4p8.zmx` | `DS_7980/design1.zmx` | 34° | 4.8 | 31 mm | 91 mm | 20 mm | H-K51, H-ZF5, H-ZK21, H-ZPK1A, H-LAF50B, H-ZLAF75A |

这两个文件可用于进一步验证每个 design 具有不同 HFOV、F/#、入瞳和材料序列时的 batch 行为。
