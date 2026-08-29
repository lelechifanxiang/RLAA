from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.zemax.common import loaded_sequential_system, normalized_field_coordinate, surface_row, zp
from tests.zemax.temp_structures import ParaxialTraceReference
from tests.zemax.zmx_loader import load_zmx_sequential_system_spec


def _configure_paraxial_angle_fields(
    oss: Any,
    *,
    edge_field_x_deg: float,
    edge_field_y_deg: float,
) -> None:
    """为近轴参考系统设置矩形归一化视场。"""

    fields = oss.SystemData.Fields
    fields.DeleteAllFields()
    fields.SetFieldType(zp.constants.SystemData.FieldType.Angle)
    fields.Normalization = zp.constants.SystemData.FieldNormalizationType.Rectangular
    fields.AddField(0.0, 0.0, 1.0)
    if edge_field_x_deg != 0.0:
        fields.AddField(edge_field_x_deg, 0.0, 1.0)
    if edge_field_y_deg != 0.0:
        fields.AddField(0.0, edge_field_y_deg, 1.0)


def fetch_zemax_paraxial_trace_from_zmx(
    zmx_path: str | Path,
    *,
    pupil_grid_shape: tuple[int, int] = (3, 3),
    focal_length_mm: float | None = None,
    image_plane_distance_mm: float | None = None,
    field_point_deg: tuple[float, float] | None = None,
    edge_field_deg: tuple[float, float] | None = None,
) -> ParaxialTraceReference:
    """从 zmx 文件读取近轴参考系统，并支持少量参数覆写。"""

    spec = load_zmx_sequential_system_spec(zmx_path)
    with loaded_sequential_system(spec.zmx_path) as oss:
        return fetch_zemax_paraxial_trace_from_spec(
            spec,
            oss,
            pupil_grid_shape=pupil_grid_shape,
            focal_length_mm=focal_length_mm,
            image_plane_distance_mm=image_plane_distance_mm,
            field_point_deg=field_point_deg,
            edge_field_deg=edge_field_deg,
        )


def fetch_zemax_paraxial_trace_from_spec(
    spec: Any,
    oss: Any,
    *,
    pupil_grid_shape: tuple[int, int] = (3, 3),
    focal_length_mm: float | None = None,
    image_plane_distance_mm: float | None = None,
    field_point_deg: tuple[float, float] | None = None,
    edge_field_deg: tuple[float, float] | None = None,
) -> ParaxialTraceReference:
    """基于已加载的 spec / oss 读取近轴参考系统，并支持少量参数覆写。"""

    if len(spec.surfaces) != 1 or spec.surfaces[0].surface_type != "Paraxial":
        raise ValueError("zmx reference must contain exactly one paraxial surface.")

    base_surface = spec.surfaces[0]
    resolved_focal_length_mm = base_surface.focal_length_mm if focal_length_mm is None else focal_length_mm
    resolved_image_plane_distance_mm = base_surface.thickness_mm if image_plane_distance_mm is None else image_plane_distance_mm
    resolved_field_point_deg = spec.field_points[0] if field_point_deg is None else field_point_deg

    if edge_field_deg is None:
        edge_field_x_deg = max((abs(value[0]) for value in spec.field_points), default=0.0)
        edge_field_y_deg = max((abs(value[1]) for value in spec.field_points), default=0.0)
    else:
        edge_field_x_deg, edge_field_y_deg = edge_field_deg

    rows, cols = pupil_grid_shape
    py_axis = [0.0] if rows == 1 else [-1.0 + 2.0 * index / (rows - 1) for index in range(rows)]
    px_axis = [0.0] if cols == 1 else [-1.0 + 2.0 * index / (cols - 1) for index in range(cols)]
    pupil_coordinates = [(px, py) for py in py_axis for px in px_axis]

    normalized_hx = normalized_field_coordinate(resolved_field_point_deg[0], edge_field_x_deg)
    normalized_hy = normalized_field_coordinate(resolved_field_point_deg[1], edge_field_y_deg)

    paraxial_x_mm: list[float] = []
    paraxial_y_mm: list[float] = []
    paraxial_z_mm: list[float] = []
    image_x_mm: list[float] = []
    image_y_mm: list[float] = []
    image_z_mm: list[float] = []
    direction_l: list[float] = []
    direction_m: list[float] = []
    direction_n: list[float] = []
    valid: list[bool] = []

    surface = oss.LDE.GetSurfaceAt(1)
    surface.SurfaceData.FocalLength = resolved_focal_length_mm
    surface.Thickness = resolved_image_plane_distance_mm
    _configure_paraxial_angle_fields(
        oss,
        edge_field_x_deg=edge_field_x_deg,
        edge_field_y_deg=edge_field_y_deg,
    )
    oss.update_status()

    # 枚举入瞳采样，逐条读取近轴面和像面的结果
    for px, py in pupil_coordinates:
        result = zp.analyses.raysandspots.SingleRayTrace(
            hx=normalized_hx,
            hy=normalized_hy,
            px=px,
            py=py,
            wavelength=spec.primary_wavelength_index + 1,
            field=0,
            raytrace_type="TangentAngle",
            global_coordinates=True,
        ).run(oss)

        ray_data = result.data.real_ray_trace_data
        if ray_data is None or ray_data.empty:
            raise ValueError("real_ray_trace_data was not returned.")

        paraxial_row = surface_row(ray_data, 1)
        image_row = surface_row(ray_data, spec.image_surface_index)
        paraxial_x_mm.append(float(paraxial_row["X-coordinate"]))
        paraxial_y_mm.append(float(paraxial_row["Y-coordinate"]))
        paraxial_z_mm.append(float(paraxial_row["Z-coordinate"]))
        image_x_mm.append(float(image_row["X-coordinate"]))
        image_y_mm.append(float(image_row["Y-coordinate"]))
        image_z_mm.append(float(image_row["Z-coordinate"]))
        direction_l.append(float(image_row["X-tangent"]))
        direction_m.append(float(image_row["Y-tangent"]))
        direction_n.append(1.0)
        valid.append(True)

    return ParaxialTraceReference(
        paraxial_x_mm=paraxial_x_mm,
        paraxial_y_mm=paraxial_y_mm,
        paraxial_z_mm=paraxial_z_mm,
        image_x_mm=image_x_mm,
        image_y_mm=image_y_mm,
        image_z_mm=image_z_mm,
        direction_l=direction_l,
        direction_m=direction_m,
        direction_n=direction_n,
        valid=valid,
        metadata={
            "source": "zospy.SingleRayTrace",
            "zmx_path": spec.zmx_path,
            "pupil_coordinates": pupil_coordinates,
            "ray_count": len(valid),
            "field_point_deg": resolved_field_point_deg,
            "edge_field_deg": (edge_field_x_deg, edge_field_y_deg),
            "focal_length_mm": resolved_focal_length_mm,
            "image_plane_distance_mm": resolved_image_plane_distance_mm,
        },
    )
