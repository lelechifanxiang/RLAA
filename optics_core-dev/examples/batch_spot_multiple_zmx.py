from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optics_core as oc
from examples.batch_spot import resolve_device, save_summary_json
from tests.zemax.heterogeneous_spot import (
    ZMX_FILENAMES,
    build_heterogeneous_system,
    load_heterogeneous_specs,
    load_spot_rms_reference,
    run_heterogeneous_spot_rms,
)


DEFAULT_DEVICE = "cpu"
DEFAULT_DESIGN_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多 ZMX 异构镜头并行点列图分析")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="运行设备，默认 cpu")
    parser.add_argument("--design-count", type=int, default=DEFAULT_DESIGN_COUNT, help="并行设计数，默认 5")
    parser.add_argument("--summary-json", default=None, help="将运行结果保存到 JSON")
    return parser.parse_args()


def repeat_system(base: oc.MultiOpticalSystem, design_count: int) -> oc.MultiOpticalSystem:
    """循环复用五个基础设计，构造指定规模的异构 batch。"""
    design_indices = torch.arange(design_count) % base.system_count
    vectors = [list(base.parameters[index]) for index in design_indices.tolist()]
    aperture = copy.deepcopy(base.aperture)
    aperture.value = torch.as_tensor(aperture.value)[design_indices]
    return oc.MultiOpticalSystem(
        base.architecture,
        name=f"heterogeneous_{design_count}",
        parameter_schema=base.parameter_schema,
        parameters=oc.ParameterVectorBatch(schema=base.parameter_schema, vectors=vectors),
        config=copy.deepcopy(base.config),
        tracer=base.tracer,
        materials=base.materials,
        fields=base.fields,
        wavelengths=base.wavelengths,
        aperture=aperture,
    )


def run_multiple_zmx_batch_spot(
    *,
    device: torch.device,
    design_count: int,
) -> dict[str, object]:
    # 1. 加载五份 ZMX 文本并循环扩展为指定规模的异构 batch。
    specs = load_heterogeneous_specs()
    if design_count < len(specs):
        raise ValueError(f"design_count 不能小于 ZMX 文件数 {len(specs)}。")
    reference = load_spot_rms_reference()
    base = build_heterogeneous_system(specs)
    base.config.backend.device = str(device)
    system = repeat_system(base, design_count)

    # 2. 一次并行点列图分析，并与五个 Zemax 预存结果比较。
    batch_rms = run_heterogeneous_spot_rms(system, specs, reference).cpu()
    zemax_rms = torch.tensor(
        [item["zemax_rms_radius_um"] for item in reference["systems"]],
        dtype=torch.float64,
    )
    design_indices = torch.arange(design_count) % len(specs)
    expected_rms = zemax_rms[design_indices]
    max_error_um = torch.abs(batch_rms - expected_rms).max().item()
    torch.testing.assert_close(batch_rms, expected_rms, atol=1e-3, rtol=0.0)

    rms_by_zmx = {
        name: batch_rms[index].tolist()
        for index, name in enumerate(ZMX_FILENAMES)
    }
    summary = {
        "device": str(device),
        "zmx_files": [Path(spec.zmx_path).name for spec in specs],
        "design_count": design_count,
        "field_count": len(base.fields),
        "wavelength_count": len(base.wavelengths),
        "rms_radius_um": rms_by_zmx,
        "max_abs_error_um": max_error_um,
    }

    print(f"并行设计数: {design_count}")
    for index, name in enumerate(ZMX_FILENAMES):
        print(f"{name} 视场角 (deg): {specs[index].field_points}")
        print(f"{name} RMS 半径 (um): {batch_rms[index].tolist()}")
    print(f"与 Zemax 结果的最大差异 (um): {max_error_um:.3e}")
    print("并行点列图结果校验通过。")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_multiple_zmx_batch_spot(
        device=resolve_device(args.device),
        design_count=args.design_count,
    )
    if args.summary_json is not None:
        save_summary_json(Path(args.summary_json), summary)


if __name__ == "__main__":
    main()
