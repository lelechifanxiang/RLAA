# 测试性能对比报告

## 测试范围
针对以下三个核心组件的契约测试：
- **Huygens MTF** (11个测试)
- **Huygens PSF** (13个测试)  
- **Wavefront Map** (9个测试)

**总计：33个测试用例**

---

## 性能对比

### 串行执行（默认模式）
```bash
pytest tests/contract/test_huygens_mtf_contract.py \
       tests/contract/test_huygens_psf_contract.py \
       tests/contract/test_wavefront_contract.py
```

**总耗时：3.79秒**

最慢的测试：
- `test_huygens_mtf_impulse_is_one`: 1.02s
- `test_huygens_psf_all_wavelengths_exports_single_mixed_image`: 0.19s
- `test_wavefront_exports_single_image`: 0.19s
- `test_huygens_mtf_exports_single_design_curves`: 0.16s

---

### 并行执行（pytest-xdist，16个worker）
```bash
pytest tests/contract/test_huygens_mtf_contract.py \
       tests/contract/test_huygens_psf_contract.py \
       tests/contract/test_wavefront_contract.py \
       -n auto
```

**总耗时：9.67秒**

最慢的测试（并行环境下）：
- `test_huygens_image_grid_follows_local_image_plane_axes`: 2.15s
- `test_huygens_mtf_impulse_is_one`: 2.06s
- `test_wavefront_sets_invalid_pupil_area_to_zero_and_rms_uses_valid_points`: 2.06s
- `test_huygens_mtf_projection_matches_2d_fft_axes`: 2.05s

---

## 分析

### 结果解读

**串行执行更快** - 在这个测试集上，串行模式（3.79秒）比并行模式（9.67秒）快约 **2.5倍**。

### 原因分析

1. **测试用例轻量化**
   - 多数测试在 0.01-0.02秒内完成
   - 只有少数几个测试超过 0.1秒
   - 33个测试的总计算时间很短

2. **并行开销占主导**
   - 启动16个worker进程的开销
   - 进程间通信和同步成本
   - PyTorch模块在每个worker中的初始化

3. **单个测试的执行时间在并行环境下变长**
   - 串行：`test_huygens_mtf_impulse_is_one` 用时 1.02s
   - 并行：同一测试用时 2.06s（**慢了2倍**）
   - 可能的原因：CPU资源竞争、缓存失效、内存带宽瓶颈

---

## 建议

### 适合串行执行的场景
✅ **当前测试集** - 测试数量少、单个测试快  
✅ 快速验证和开发调试  
✅ CI/CD中的快速检查

### 适合并行执行的场景
🔄 大规模测试套件（100+ 测试）  
🔄 包含长时间运行的集成测试  
🔄 回归测试套件（例如 Zemax 对标测试）

### 优化建议

1. **对于当前测试集：继续使用串行模式**
2. **对于大规模测试：**
   - 使用 `-n 4` 或 `-n 8` 而不是 `-n auto`（16个worker过多）
   - 将快速测试和慢速测试分组
   - 只对耗时测试使用并行

3. **针对性优化：**
   ```bash
   # 快速测试（串行）
   pytest tests/contract/ -k "not export"
   
   # 慢速测试（并行）
   pytest tests/contract/ -k "export" -n 4
   ```

---

## 结论

对于这三个核心组件的契约测试（33个用例），**串行执行是最优选择**，速度快2.5倍。并行执行适合更大规模的测试套件。
