"""
最小化测试：只测试 optics_core MTF 计算
"""
import sys
from pathlib import Path
import io

# 设置输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加 optics_core 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "optics_core-dev"))

import torch
import optics_core as oc


def main():
    print("=" * 60)
    print("测试 optics_core MTF 计算")
    print("=" * 60)

    # 加载 ZMX 文件 - 使用和 batch_mtf.py 相同的文件
    zmx_path = "../optics_core-dev/tests/zemax/zmx_files/Double Gauss 28 degree field with CB.ZMX"
    print(f"\n1. 加载 ZMX: {zmx_path}")

    try:
        from zemax_utils import load_zmx_sequential_system_spec, build_optics_core_system_from_zmx_spec

        spec = load_zmx_sequential_system_spec(zmx_path)
        print("   ✓ ZMX 规格加载成功")

        base_system = build_optics_core_system_from_zmx_spec(spec)
        print("   ✓ 系统构建成功")

        # 使用和 batch_mtf.py 示例相同的方式构建 MultiOpticalSystem
        import copy

        system = oc.MultiOpticalSystem(
            architecture=base_system.architecture,
            name=base_system.name,
            parameter_schema=oc.ParameterSchema([]),
            parameters=[{}],
            config=copy.deepcopy(base_system.config),
            tracer=base_system.tracer,
            materials=base_system.materials,
            fields=copy.deepcopy(list(base_system.fields)),
            wavelengths=copy.deepcopy(list(base_system.wavelengths)),
            aperture=copy.deepcopy(base_system.aperture),
        )

        # 设置为 CPU，更简单测试
        system.config.backend.device = "cpu"

        device = system.config.backend.device
        print(f"   ✓ MultiOpticalSystem 创建成功 (device: {device})")

    except Exception as e:
        print(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 计算 MTF
    print("\n2. 计算 MTF")
    try:
        system.prepare()
        print("   ✓ 系统准备完成")

        settings = oc.MTFSettings(
            pupil_sample_count=32,
            image_sample_count=32,
            frequencies_lp_per_mm=(20.0, 30.0, 50.0),
            field_indices=(0,),
            wavelength_index=-1,
        )

        result = system.analysis.mtf(settings).run()
        print("   ✓ MTF 计算完成")

        # 提取并显示结果
        sag = result.sagittal.detach().cpu().numpy()
        tan = result.tangential.detach().cpu().numpy()

        print(f"\n3. 结果:")
        print(f"   - Sagittal shape: {sag.shape}")
        print(f"   - Sagittal range: [{sag.min():.4f}, {sag.max():.4f}]")
        print(f"   - Tangential shape: {tan.shape}")
        print(f"   - Tangential range: [{tan.min():.4f}, {tan.max():.4f}]")

        if sag.ndim == 3:
            sag = sag[0, 0, :]  # 第一个设计，第一个视场
            tan = tan[0, 0, :]

        print(f"\n   中心视场 MTF:")
        print(f"   频率 [lp/mm]  |  Sagittal  |  Tangential")
        print(f"   " + "-" * 45)
        for i, freq in enumerate([20.0, 30.0, 50.0]):
            print(f"   {freq:12.1f}  |  {sag[i]:8.4f}  |  {tan[i]:8.4f}")

        print("\n" + "=" * 60)
        print("✓ 测试成功！optics_core MTF 计算正常工作")
        print("=" * 60)

    except Exception as e:
        print(f"   ✗ MTF 计算失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
