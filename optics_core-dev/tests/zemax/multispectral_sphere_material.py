from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.zemax.common import get_merit_operand_value, get_surface_indices, loaded_sequential_system, zp
from tests.zemax.temp_structures import ZemaxSurfaceRefractiveIndicesReference, ZemaxSurfaceSagReference
from tests.zemax.zmx_loader import load_zmx_sequential_system_spec


def fetch_zemax_surface_sag_reference(
    zmx_path: str | Path,
    *,
    surface_index: int = 1,
    sag_sample_xy_mm: tuple[float, float],
) -> ZemaxSurfaceSagReference:
    """从 zmx 文件读取指定表面的球面矢高参考值。"""

    spec = load_zmx_sequential_system_spec(zmx_path)
    with loaded_sequential_system(spec.zmx_path) as oss:
        return fetch_zemax_surface_sag_reference_from_spec(
            spec,
            oss,
            surface_index=surface_index,
            sag_sample_xy_mm=sag_sample_xy_mm,
        )


def fetch_zemax_surface_sag_reference_from_spec(
    spec: Any,
    oss: Any,
    *,
    surface_index: int = 1,
    sag_sample_xy_mm: tuple[float, float],
) -> ZemaxSurfaceSagReference:
    """基于已加载的 spec / oss 读取指定表面的球面矢高参考值。"""

    if surface_index < 1 or surface_index > len(spec.surfaces):
        raise ValueError("surface_index is out of range.")

    surface_spec = spec.surfaces[surface_index - 1]
    sample_x_mm, sample_y_mm = sag_sample_xy_mm
    pupil_x = sample_x_mm / surface_spec.semi_diameter_mm
    pupil_y = sample_y_mm / surface_spec.semi_diameter_mm
    if pupil_x * pupil_x + pupil_y * pupil_y > 1.0:
        raise ValueError("sag_sample_xy_mm must lie inside the entrance pupil.")

    surface_sag_mm = get_merit_operand_value(
        oss,
        zp.constants.Editors.MFE.MeritOperandType.REAZ,
        surface_index=surface_index,
        wavelength_index=spec.primary_wavelength_index + 1,
        field_hx=0.0,
        field_hy=0.0,
        pupil_x=pupil_x,
        pupil_y=pupil_y,
    )

    return ZemaxSurfaceSagReference(
        surface_sag_mm=surface_sag_mm,
        metadata={
            "source": "zospy.MFE.REAZ",
            "zmx_path": spec.zmx_path,
            "surface_index": surface_index,
            "sag_sample_xy_mm": sag_sample_xy_mm,
        },
    )


def fetch_zemax_surface_refractive_indices(
    zmx_path: str | Path,
    *,
    surface_index: int = 1,
) -> ZemaxSurfaceRefractiveIndicesReference:
    """从 zmx 文件读取指定表面的多波长折射率参考值。"""

    spec = load_zmx_sequential_system_spec(zmx_path)
    with loaded_sequential_system(spec.zmx_path) as oss:
        return fetch_zemax_surface_refractive_indices_from_spec(
            spec,
            oss,
            surface_index=surface_index,
        )


def fetch_zemax_surface_refractive_indices_from_spec(
    spec: Any,
    oss: Any,
    *,
    surface_index: int = 1,
) -> ZemaxSurfaceRefractiveIndicesReference:
    """基于已加载的 spec / oss 读取指定表面的多波长折射率参考值。"""

    if surface_index < 1 or surface_index > len(spec.surfaces):
        raise ValueError("surface_index is out of range.")

    refractive_indices = get_surface_indices(
        oss,
        surface_index,
        wavelength_count=len(spec.wavelengths_um),
    )

    return ZemaxSurfaceRefractiveIndicesReference(
        wavelengths_um=[float(wavelength_um) for wavelength_um in spec.wavelengths_um],
        refractive_indices=refractive_indices,
        metadata={
            "source": "ILensDataEditor.GetIndex",
            "zmx_path": spec.zmx_path,
            "surface_index": surface_index,
        },
    )
