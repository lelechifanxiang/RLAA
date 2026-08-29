from __future__ import annotations

from pathlib import Path
from typing import Any

from zospy.analyses.old.reports import system_data

from tests.zemax.common import get_merit_operand_value, loaded_sequential_system, zp
from tests.zemax.temp_structures import ZemaxFirstOrderReference
from tests.zemax.zmx_loader import load_zmx_sequential_system_spec


def fetch_zemax_first_order_from_zmx(
    zmx_path: str | Path,
) -> ZemaxFirstOrderReference:
    """直接从 zmx 文件读取一阶参考量。"""

    spec = load_zmx_sequential_system_spec(zmx_path)
    with loaded_sequential_system(spec.zmx_path) as oss:
        return fetch_zemax_first_order_from_spec(spec, oss)


def fetch_zemax_first_order_from_spec(
    spec: Any,
    oss: Any,
) -> ZemaxFirstOrderReference:
    """基于已加载的 spec / oss 读取一阶参考量。"""

    wavelength_index = spec.primary_wavelength_index + 1
    lens_data = system_data(oss).Data.GeneralLensData
    total_track_length_mm = float(lens_data["Total Track"])
    back_focal_length_mm = float(lens_data["Back Focal Length"])
    last_refractive_surface = max(
        index for index, surface in enumerate(spec.surfaces) if surface.surface_type != "CoordinateBreak"
    )
    image_plane_distance_mm = sum(
        surface.thickness_mm for surface in spec.surfaces[last_refractive_surface:]
    )

    effective_focal_length_mm = get_merit_operand_value(
        oss,
        zp.constants.Editors.MFE.MeritOperandType.EFFL,
        surface_index=0,
        wavelength_index=wavelength_index,
    )
    image_f_number = get_merit_operand_value(
        oss,
        zp.constants.Editors.MFE.MeritOperandType.ISFN,
        surface_index=0,
        wavelength_index=wavelength_index,
    )
    working_f_number = get_merit_operand_value(
        oss,
        zp.constants.Editors.MFE.MeritOperandType.WFNO,
        surface_index=0,
        wavelength_index=wavelength_index,
    )
    entrance_pupil_z_mm = get_merit_operand_value(
        oss,
        zp.constants.Editors.MFE.MeritOperandType.ENPP,
        surface_index=0,
        wavelength_index=wavelength_index,
    )
    entrance_pupil_radius_mm = 0.5 * get_merit_operand_value(
        oss,
        zp.constants.Editors.MFE.MeritOperandType.EPDI,
        surface_index=0,
        wavelength_index=wavelength_index,
    )
    exit_pupil_z_mm = get_merit_operand_value(
        oss,
        zp.constants.Editors.MFE.MeritOperandType.EXPP,
        surface_index=0,
        wavelength_index=wavelength_index,
    )
    exit_pupil_radius_mm = 0.5 * get_merit_operand_value(
        oss,
        zp.constants.Editors.MFE.MeritOperandType.EXPD,
        surface_index=0,
        wavelength_index=wavelength_index,
    )

    return ZemaxFirstOrderReference(
        effective_focal_length_mm=effective_focal_length_mm,
        image_f_number=image_f_number,
        working_f_number=working_f_number,
        entrance_pupil_z_mm=entrance_pupil_z_mm,
        entrance_pupil_radius_mm=entrance_pupil_radius_mm,
        exit_pupil_z_mm=exit_pupil_z_mm,
        exit_pupil_radius_mm=exit_pupil_radius_mm,
        total_track_length_mm=total_track_length_mm,
        image_plane_distance_mm=image_plane_distance_mm,
        back_focal_length_mm=back_focal_length_mm,
        metadata={
            "source": "zospy MFE + System Data",
            "zmx_path": spec.zmx_path,
            "primary_wavelength_um": spec.wavelengths_um[spec.primary_wavelength_index],
            "stop_surface_index": spec.stop_surface_index,
        },
    )
