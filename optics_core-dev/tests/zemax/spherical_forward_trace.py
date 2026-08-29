from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.zemax.common import (
    get_surface_indices,
    loaded_sequential_system,
    normalized_field_coordinate,
    surface_row,
    zp,
)
from tests.zemax.readers import read_surface_semi_diameters, row_direction
from tests.zemax.temp_structures import (
    SphericalClearApertureReference,
    SphericalForwardTraceReference,
)
from tests.zemax.zmx_loader import load_zmx_sequential_system_spec


def fetch_zemax_spherical_forward_trace(
    zmx_path: str | Path,
    *,
    pupil_coordinates: tuple[tuple[float, float], ...],
    field_points: tuple[tuple[float, float], ...] | None = None,
    wavelength_indices: tuple[int, ...] | None = None,
) -> SphericalForwardTraceReference:
    """从 zmx 文件读取单光线追迹参考值。"""

    spec = load_zmx_sequential_system_spec(zmx_path)
    with loaded_sequential_system(spec.zmx_path) as oss:
        return fetch_zemax_spherical_forward_trace_from_spec(
            spec,
            oss,
            pupil_coordinates=pupil_coordinates,
            field_points=field_points,
            wavelength_indices=wavelength_indices,
        )


def fetch_zemax_spherical_forward_trace_from_spec(
    spec: Any,
    oss: Any,
    *,
    pupil_coordinates: tuple[tuple[float, float], ...],
    field_points: tuple[tuple[float, float], ...] | None = None,
    wavelength_indices: tuple[int, ...] | None = None,
) -> SphericalForwardTraceReference:
    """基于已加载的 spec / oss 读取单光线追迹参考值。"""

    if not spec.surfaces:
        raise ValueError("zmx spec must contain at least one surface.")

    selected_field_points = spec.field_points if field_points is None else field_points
    selected_wavelength_indices = (
        tuple(range(1, len(spec.wavelengths_um) + 1))
        if wavelength_indices is None
        else wavelength_indices
    )

    surface_indices = list(range(1, len(spec.surfaces) + 1)) + [spec.image_surface_index]
    x_mm: list[list[float]] = [[] for _ in surface_indices]
    y_mm: list[list[float]] = [[] for _ in surface_indices]
    z_mm: list[list[float]] = [[] for _ in surface_indices]
    direction_l: list[list[float]] = [[] for _ in surface_indices]
    direction_m: list[list[float]] = [[] for _ in surface_indices]
    direction_n: list[list[float]] = [[] for _ in surface_indices]
    refractive_indices_by_surface: dict[int, list[float]] = {}

    edge_field_x_deg = max((abs(field_x) for field_x, _field_y in spec.field_points), default=0.0)
    edge_field_y_deg = max((abs(field_y) for _field_x, field_y in spec.field_points), default=0.0)

    for surface_index, surface_spec in enumerate(spec.surfaces, start=1):
        has_material = (
            surface_spec.material_name is not None
            or surface_spec.refractive_indices is not None
            or surface_spec.nd is not None
            or surface_spec.vd is not None
        )
        if not has_material:
            continue
        refractive_indices_by_surface[surface_index] = get_surface_indices(
            oss,
            surface_index,
            wavelength_count=len(spec.wavelengths_um),
        )

    # 枚举视场、波长和入瞳坐标，逐条读取各面的交点与方向
    for field_x_deg, field_y_deg in selected_field_points:
        normalized_hx = normalized_field_coordinate(field_x_deg, edge_field_x_deg)
        normalized_hy = normalized_field_coordinate(field_y_deg, edge_field_y_deg)
        for wavelength_index in selected_wavelength_indices:
            for pupil_x, pupil_y in pupil_coordinates:
                result = zp.analyses.raysandspots.SingleRayTrace(
                    hx=normalized_hx,
                    hy=normalized_hy,
                    px=pupil_x,
                    py=pupil_y,
                    wavelength=wavelength_index,
                    field=0,
                    raytrace_type="TangentAngle",
                    global_coordinates=True,
                ).run(oss)
                ray_data = result.data.real_ray_trace_data
                if ray_data is None or ray_data.empty:
                    raise ValueError("real_ray_trace_data was not returned.")

                for output_index, surface_index in enumerate(surface_indices):
                    row = surface_row(ray_data, surface_index)
                    l_value, m_value, n_value = row_direction(row)
                    x_mm[output_index].append(float(row["X-coordinate"]))
                    y_mm[output_index].append(float(row["Y-coordinate"]))
                    z_mm[output_index].append(float(row["Z-coordinate"]))
                    direction_l[output_index].append(l_value)
                    direction_m[output_index].append(m_value)
                    direction_n[output_index].append(n_value)

    return SphericalForwardTraceReference(
        surface_indices=surface_indices,
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        direction_l=direction_l,
        direction_m=direction_m,
        direction_n=direction_n,
        refractive_indices_by_surface=refractive_indices_by_surface,
        wavelengths_um=[float(spec.wavelengths_um[wavelength_index - 1]) for wavelength_index in selected_wavelength_indices],
        field_points=list(selected_field_points),
        pupil_coordinates=list(pupil_coordinates),
        metadata={
            "source": "zospy.SingleRayTrace",
            "zmx_path": spec.zmx_path,
            "surface_count": len(spec.surfaces),
            "field_count": len(selected_field_points),
            "ray_count": len(selected_field_points) * len(selected_wavelength_indices) * len(pupil_coordinates),
            "edge_field_x_deg": edge_field_x_deg,
            "edge_field_y_deg": edge_field_y_deg,
        },
    )


def fetch_zemax_spherical_clear_apertures(
    zmx_path: str | Path,
) -> SphericalClearApertureReference:
    """直接从 zmx 文件读取各球面的 SemiDiameter。"""

    spec = load_zmx_sequential_system_spec(zmx_path)
    with loaded_sequential_system(spec.zmx_path) as oss:
        return fetch_zemax_spherical_clear_apertures_from_spec(spec, oss)


def fetch_zemax_spherical_clear_apertures_from_spec(
    spec: Any,
    oss: Any,
) -> SphericalClearApertureReference:
    """基于已加载的 spec / oss 读取各球面的 SemiDiameter。"""

    surface_indices = list(range(1, len(spec.surfaces) + 1))
    semi_diameter_mm = read_surface_semi_diameters(oss, surface_indices)

    return SphericalClearApertureReference(
        surface_indices=surface_indices,
        semi_diameter_mm=semi_diameter_mm,
        field_points=list(spec.field_points),
        pupil_coordinates=[],
        wavelengths_um=list(spec.wavelengths_um),
        metadata={
            "source": "ILDERow.SemiDiameter",
            "zmx_path": spec.zmx_path,
            "method": "direct_surface_property",
            "surface_count": len(surface_indices),
        },
    )
