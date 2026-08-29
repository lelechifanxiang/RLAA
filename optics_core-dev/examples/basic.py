from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optics_core as oc
from zemax_utils import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


DOUBLE_GAUSS_ZMX_PATH = Path("tests/zemax/zmx_files/Double Gauss 28 degree field real.zmx")
OUTPUT_DIR = REPO_ROOT / "examples/output"
PUPIL_RAY_COUNT = 100000
PUPIL_SEED = 0
LAST_THICKNESS_DELTAS_MM = (-0.1, 0.0, 0.1)
CURVATURE_DELTA_INV_MM = 1e-4


def synchronized_now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def build_multi_system(
    zmx_path: Path,
) -> tuple[oc.MultiOpticalSystem, list[dict[str, float]]]:
    """从 zmx_path 构建 6 个同架构设计。"""

    # 1. 从 zmx 文件中读取规格
    spec = load_zmx_sequential_system_spec(zmx_path)

    # 2. 在 optics core 中构建同样的系统 base_system
    base_system = build_optics_core_system_from_zmx_spec(spec)

    # 3. 设置多重结构规则
    last_surface_index = len(spec.surfaces) - 1
    last_surface_spec = spec.surfaces[last_surface_index]
    base_thickness_mm = float(last_surface_spec.thickness_mm)
    base_radius_mm = float(last_surface_spec.radius_mm)
    schema = oc.ParameterSchema(
        [
            oc.ParameterSpec(
                name="last_thickness_mm",
                path=f"surface[{last_surface_index}].gap.thickness",
            ),
            oc.ParameterSpec(
                name="last_radius_mm",
                path=f"surface[{last_surface_index}].geometry.radius",
            ),
        ]
    )

    # 4. 构造多重结构参数参数
    # 对最后一面的厚度d微扰 [d-0.1, d, d+0.1]
    # 最后一个面曲率微扰 [c, c+1e-4]
    # 总共构建 3x2=6 个设计
    thickness_values_mm = [base_thickness_mm + delta for delta in LAST_THICKNESS_DELTAS_MM]
    base_curvature = 1.0 / base_radius_mm
    radius_values_mm = [
        base_radius_mm,
        1.0 / (base_curvature + CURVATURE_DELTA_INV_MM),
    ]
    parameters = oc.build_parameter_vector_grid(
        schema,
        axes=[
            oc.ParameterSweepAxis("last_thickness_mm", thickness_values_mm),
            oc.ParameterSweepAxis("last_radius_mm", radius_values_mm),
        ]
    )

    # 5. 将扰动数据填入系统，重新构建多重结构系统 system
    system = oc.MultiOpticalSystem(
        architecture=base_system.architecture,
        name=f"{base_system.name}",
        parameter_schema=schema,
        parameters=parameters,
        config=copy.deepcopy(base_system.config),
        tracer=base_system.tracer,
        materials=base_system.materials,
        fields=copy.deepcopy(list(base_system.fields)),
        wavelengths=copy.deepcopy(list(base_system.wavelengths)),
        aperture=copy.deepcopy(base_system.aperture),
    )
    if torch.cuda.is_available():
        system.config.backend.device = "cuda"

    # 6. 打印多重结构信息
    print(f"zmx 文件: {zmx_path}")
    print(f"共构建 {system.system_count} 个设计")
    for idx, parameter_vector in enumerate(system.parameters):
        print(
            "design {}: last_thickness_mm={:.6f}, last_radius_mm={:.6f}".format(
                idx,
                float(parameter_vector[schema.index_of("last_thickness_mm")]),
                float(parameter_vector[schema.index_of("last_radius_mm")]),
            )
        )

    return system


def run_parallel_trace(system: oc.MultiOpticalSystem) -> None:
    # 1. 构造随机采样器
    sampler = oc.RandomPupilSampler(ray_count=PUPIL_RAY_COUNT, seed=PUPIL_SEED)
    device = torch.device(system.config.backend.device or "cpu")

    # 2. 进行追迹，并统计耗时
    started_at = synchronized_now(device)
    result = system.trace(
        sampler=sampler,
        options=oc.TraceOptions(record_intersections=False),
    )
    elapsed_ms = (synchronized_now(device) - started_at) * 1000.0

    total_ray_count = int(result.valid.numel())
    valid_ray_count = int(torch.as_tensor(result.valid, dtype=torch.bool).sum().item())
    print(f"并行追迹 device: {device}")
    print(f"并行追迹 design_count={system.system_count}, field_count={len(system.fields)}, wavelength_count={len(system.wavelengths)}")
    print(f"并行追迹 pupil_ray_count={PUPIL_RAY_COUNT}, total_ray_count={total_ray_count}, valid_ray_count={valid_ray_count}")
    print(f"并行追迹耗时: {elapsed_ms:.3f} ms")


def run_parallel_spot(system: oc.MultiOpticalSystem) -> None:
    # 1. 构造点列图分析器，执行点列图
    result = system.analysis.spot_diagram().run()

    # 2. 打印点列图结果
    rms_radius_um = torch.as_tensor(result.rms_radius_um, dtype=torch.float64).detach().cpu()
    geo_radius_um = torch.as_tensor(result.geo_radius_um, dtype=torch.float64).detach().cpu()

    print("并行 spot RMS 半径 (um):")
    for design_index in range(system.system_count):
        print(f"design {design_index}: {rms_radius_um[design_index].tolist()}")
    print("并行 spot GEO 半径 (um):")
    for design_index in range(system.system_count):
        print(f"design {design_index}: {geo_radius_um[design_index].tolist()}")


def export_layouts(system: oc.MultiOpticalSystem) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("开始逐设计导出 2D layout 图...")
    for design_index in range(system.system_count):
        design = system.design_view(design_index)
        design.prepare()
        output_path = OUTPUT_DIR / f"layout_design_{design_index:02d}.png"
        result = design.analysis.layout_2d(
            oc.Layout2DSettings(save_path=str(output_path)),
        ).run()
        print(
            "design {} layout 导出: {}".format(
                design_index,
                result.save_path,
            )
        )


def main() -> None:
    # 1. 加载双高斯 zmx 文件，读取规格，通过扰动厚度和曲率，构建多重结构系统
    zmx_path = REPO_ROOT / DOUBLE_GAUSS_ZMX_PATH
    system = build_multi_system(zmx_path)
    system.prepare()

    # 2. 随机生成大量光线，进行并行光线追迹，统计耗时
    run_parallel_trace(system)

    # 3. 点列图
    run_parallel_spot(system)

    # 4. 绘制 layout
    export_layouts(system)


if __name__ == "__main__":
    main()
