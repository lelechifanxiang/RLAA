from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

import optics_core as oc
from optics_core.spot_diagram import build_spot_diagram_sampler, compute_spot_metrics, extract_spot_data
from zemax_utils import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
ZMX_DIRECTORY = REPO_ROOT / "tests/zemax/zmx_files/same_arch_diff_materials"
ZMX_FILENAMES = (
    "sg6_material_a.zmx",
    "sg6_material_b.zmx",
    "sg6_material_c.zmx",
    "sg6_hfov25_f3p0.zmx",
    "sg6_hfov34_f4p8.zmx",
)
REFERENCE_PATH = REPO_ROOT / "tests/zemax/reference_data/same_arch_diff_materials_spot_rms.json"


def load_heterogeneous_specs() -> tuple[Any, ...]:
    """加载五个同拓扑 ZMX 规格。"""
    return tuple(load_zmx_sequential_system_spec(ZMX_DIRECTORY / name) for name in ZMX_FILENAMES)


def topology_signature(spec: Any) -> tuple[Any, ...]:
    """提取允许组成同一 batch 的拓扑信息。"""
    return (
        tuple(
            (surface.surface_type, surface.semi_diameter_solve, surface.aperture_type)
            for surface in spec.surfaces
        ),
        spec.stop_surface_index,
        spec.aperture_kind,
        spec.field_type,
        len(spec.field_points),
        spec.wavelengths_um,
        spec.primary_wavelength_index,
    )


def build_heterogeneous_system(specs: Sequence[Any]) -> oc.MultiOpticalSystem:
    """将五份规格整理为共享结构和逐设计参数。"""
    base = build_optics_core_system_from_zmx_spec(specs[0])
    schema_specs: list[oc.ParameterSpec] = []
    for surface_index, surface in enumerate(specs[0].surfaces):
        schema_specs.extend(
            (
                oc.ParameterSpec(
                    name=f"surface_{surface_index}_radius",
                    path=f"surface[{surface_index}].geometry.radius",
                    default=surface.radius_mm,
                ),
                oc.ParameterSpec(
                    name=f"surface_{surface_index}_thickness",
                    path=f"surface[{surface_index}].gap.thickness",
                    default=surface.thickness_mm,
                ),
                oc.ParameterSpec(
                    name=f"surface_{surface_index}_medium",
                    path=f"surface[{surface_index}].gap.medium",
                    default=surface.material_name or "AIR",
                ),
                oc.ParameterSpec(
                    name=f"surface_{surface_index}_semi_diameter",
                    path=f"surface[{surface_index}].semi_diameter",
                    default=surface.semi_diameter_mm,
                ),
            )
        )
    schema = oc.ParameterSchema(schema_specs)
    vectors = [
        [
            value
            for surface in spec.surfaces
            for value in (
                surface.radius_mm,
                surface.thickness_mm,
                surface.material_name or "AIR",
                surface.semi_diameter_mm,
            )
        ]
        for spec in specs
    ]
    return oc.MultiOpticalSystem(
        base.architecture,
        name="same_arch_diff_materials",
        parameter_schema=schema,
        parameters=oc.ParameterVectorBatch(schema=schema, vectors=vectors),
        config=base.config,
        tracer=base.tracer,
        materials=base.materials,
        fields=base.fields,
        wavelengths=base.wavelengths,
        aperture=oc.SystemAperture(
            kind=base.aperture.kind,
            value=torch.tensor([spec.aperture_value for spec in specs], dtype=torch.float64),
            stop_surface=base.aperture.stop_surface,
            label=base.aperture.label,
        ),
    )


def load_spot_rms_reference() -> dict[str, Any]:
    """读取并校验预存 Zemax Standard Spot 基准。"""
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    for item in reference["systems"]:
        zmx_path = ZMX_DIRECTORY / item["zmx_file"]
        if hashlib.sha256(zmx_path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Zemax spot reference is stale for {zmx_path.name}.")
    return reference


def run_heterogeneous_spot_rms(
    system: oc.MultiOpticalSystem,
    specs: Sequence[Any],
    reference: dict[str, Any],
) -> torch.Tensor:
    """一次追迹计算异构 design batch 的逐视场 RMS 半径。"""
    system.prepare()
    settings = oc.SpotDiagramSettings(**reference["settings"])
    sample = build_spot_diagram_sampler(settings).sample()
    design_indices = torch.arange(system.system_count) % len(specs)
    first_order = system.first_order_data
    rays = oc.build_pupil_rays(
        system,
        field_angles=torch.tensor([spec.field_points for spec in specs], dtype=torch.float64)[design_indices],
        entrance_pupil_z=first_order.entrance_pupil_z,
        entrance_pupil_radius=first_order.entrance_pupil_radius,
        pupil_coordinates=sample.pupil_coordinates,
        wavelength_indices=torch.arange(len(specs[0].wavelengths_um), dtype=torch.int64),
    )
    trace_result = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(record_intersections=False),
    )
    spot_data = extract_spot_data(system, trace_result, sample)
    rms_radius_um, _ = compute_spot_metrics(system, spot_data)
    return rms_radius_um


def generate_spot_rms_reference() -> dict[str, Any]:
    """调用 Zemax 生成五文件 Standard Spot 基准。"""
    from tests.zemax.common import loaded_sequential_system
    from tests.zemax.spot_diagram import fetch_zemax_standard_spot_metrics_from_spec

    systems: list[dict[str, Any]] = []
    for name, spec in zip(ZMX_FILENAMES, load_heterogeneous_specs(), strict=True):
        with loaded_sequential_system(spec.zmx_path) as oss:
            spot = fetch_zemax_standard_spot_metrics_from_spec(
                spec,
                oss,
                pattern="hexapolar",
                ray_density=30,
            )
        zmx_path = ZMX_DIRECTORY / name
        systems.append(
            {
                "zmx_file": name,
                "sha256": hashlib.sha256(zmx_path.read_bytes()).hexdigest(),
                "field_points_deg": [list(point) for point in spec.field_points],
                "wavelengths_um": list(spec.wavelengths_um),
                "zemax_rms_radius_um": spot.rms_radius_um,
            }
        )
    return {
        "schema_version": 2,
        "settings": {"pattern": "hexapolar", "ray_density": 30},
        "zemax_settings": {"field": "all", "wavelength": "all", "surface": "image", "refer_to": "chief_ray"},
        "systems": systems,
    }


if __name__ == "__main__":
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(
        json.dumps(generate_spot_rms_reference(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
