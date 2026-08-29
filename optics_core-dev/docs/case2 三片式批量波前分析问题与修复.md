# case2 三片式批量波前分析问题与修复

## 结论

修复前，`case2_3p_center.zmx` 不能通过只替换文件路径和 CB 面号得到正确波前；修复后只需修改 `batch_wavefront.py` 顶部的两个常量：

```python
ZMX_PATH = REPO_ROOT / "tests/zemax/zmx_files/case2_3p_center.zmx"
COORDINATE_BREAK_PAIRS = ((1, 4), (7, 10))
```

随后按普通参数执行脚本：

```powershell
python examples\batch_wavefront.py `
  --device cpu `
  --design-count 2 `
  --sample-count 32 `
  --field-indices 0 `
  --wavelength-indices 0 `
  --output-dir "tests\output\wavefront\case2_user_simulation"
```

`1 4` 和 `7 10` 分别表示两组 `first/return` 坐标间断面。每组 first 面独立生成偏心和倾斜随机量，return 面显式写入相反数。

## 发现并修复的问题

| 问题 | 原有影响 | 处理方式 |
|---|---|---|
| ZMX 文本加载器只支持 `ENPD` | 遇到该文件的 `FLOA` 直接报错 | 支持 Float By Stop Size，并从固定 stop 半口径取得口径 |
| 物面厚度被丢弃 | 有限物点被错误地当成无限远平面波 | 保留物距，从有限物点向入瞳发射球面波并初始化 OPL |
| 无焦像空间标志被丢弃 | 使用球面参考波前，产生几十波量级的伪球差 | 解析 `FTYP` 中的无焦标志，改用垂直主光线的平面参考波前 |
| `___BLANK` 材料 pickup 被当成 `nd=1.5` 的玻璃 | 第 5、11 面后折射率错误 | 解析 GLAS pickup 并复用目标面的介质；本文件两处均解析为空气 |
| 零厚度 CB 后紧接负 sag 球面 | 非主光线被错误判为负路径并失效 | 顺序求交统一接受有符号路径并采用局部最近交点 |
| 公差构造器固定只有一组 CB | 无法独立扰动前两个镜片 | 公共构造器支持任意数量的 CB 对，随机记录和参数向量统一生成 |
| 偶数 Wavefront 网格没有采到 pupil 原点 | 波前中心和 Zemax 错位，边缘误差被放大 | 实际追迹网格向负方向偏移半个显示步长，使 `(N/2, N/2)` 严格为零 |
| 无效光线参与 RMS 中间运算 | 部分配置可能得到 `NaN` RMS | 平方前先把无效位置置零 |

无焦系统采用平面参考波前与 OpticStudio 的定义一致，参见 [Ansys：How to design afocal systems](https://optics.ansys.com/hc/en-us/articles/42661707601683-How-to-design-afocal-systems) 和 [Afocal Image Space](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v252/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Afocal_Image_Space.html)。

## 验证结果

无公差、32×32、视场 0、波长 0 的 Wavefront Map 对比：

| 指标 | Zemax | OpticsCore |
|---|---:|---:|
| RMS（waves） | 0.06980000 | 0.06997518 |
| 最小 OPD（waves） | -0.26383739 | -0.26761429 |
| 最大 OPD（waves） | 0 | 0 |

在 Zemax 的 709 个有效点上，OPD 最大/平均绝对误差为 `0.05180771 / 0.01309306 waves`。

两设计 CPU 批量导出实测成功：输出 shape 为 `(2, 1, 1, 32, 32)`，生成 2 张 PNG；计算耗时约 `0.028 s`。小批量数据只用于确认链路，不代表性能基准。

测试结果：

- `tests/contract`：65 passed，1 skipped（本机无 CUDA）。
- Wavefront Zemax 回归：图片导出测试通过；3 组严格数值测试保持既有 `xfail` 状态。

## 尚存差异

该文件在 Zemax 中启用了 ray aiming。OpticsCore 当前仍按一阶入瞳位置直接构造采样光线，因此边缘遮拦规则尚未完全一致：Zemax 有 709 个有效网格点，OpticsCore 为 749 个。内置材料库缺少该文件所用的精确 N-BK7 色散数据时，也会回退到 Abbe 模型。这两点不妨碍批量导图，但若要继续压缩逐像素误差，应优先补齐真实光线瞄准和材料数据。
