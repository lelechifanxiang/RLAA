from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch

from .._runtime import default_device
from ..rays import RayAimingResult, RayBundle
from ..sampling import SamplingResult
from ..system_specs import FieldPoint

if TYPE_CHECKING:
    from ..system import MultiOpticalSystem


def build_pupil_rays(
    system: MultiOpticalSystem,
    *,
    field_angles: torch.Tensor,
    entrance_pupil_z: torch.Tensor,
    entrance_pupil_radius: torch.Tensor,
    pupil_coordinates: torch.Tensor,
    wavelength_indices: torch.Tensor,
) -> RayBundle:
    """由逐设计视场和可广播的入瞳数据构造显式批量光线。"""
    tensors = (field_angles, entrance_pupil_z, entrance_pupil_radius, pupil_coordinates)
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float64 for value in tensors):
        raise TypeError("build_pupil_rays inputs must be FP64 tensors.")
    if not isinstance(wavelength_indices, torch.Tensor) or wavelength_indices.dtype != torch.int64:
        raise TypeError("wavelength_indices must be an int64 tensor.")
    if field_angles.ndim != 3 or field_angles.shape[0] != system.system_count or field_angles.shape[-1] != 2:
        raise ValueError("field_angles must have shape (design, field, 2).")
    if pupil_coordinates.ndim != 2 or pupil_coordinates.shape[-1] != 2:
        raise ValueError("pupil_coordinates must have shape (ray, 2).")
    if wavelength_indices.ndim != 1:
        raise ValueError("wavelength_indices must have shape (wavelength,).")

    device = default_device(system)
    fields = field_angles.to(device=device)
    wavelength_indices = wavelength_indices.to(device=device)
    field_count = fields.shape[1]
    wavelength_count = wavelength_indices.shape[0]
    batch_shape = (system.system_count, field_count, wavelength_count)
    return _assemble_pupil_rays(
        system,
        field_slopes=_field_slopes(system, fields),
        entrance_pupil_z=_broadcast_batch_value(
            entrance_pupil_z,
            batch_shape,
            name="entrance_pupil_z",
            device=device,
        ),
        entrance_pupil_radius=_broadcast_batch_value(
            entrance_pupil_radius,
            batch_shape,
            name="entrance_pupil_radius",
            device=device,
        ),
        pupil_coordinates=pupil_coordinates.to(device=device),
        wavelength_indices=wavelength_indices,
    )


def build_input_rays_from_sample(
    system: MultiOpticalSystem,
    fields: Sequence[FieldPoint],
    wavelength_indices: Sequence[int],
    sample: SamplingResult,
    entrance_pupil: RayAimingResult,
) -> RayBundle:
    """把入瞳采样点映射成显式光线。"""
    # 1. 读取归一化 pupil 点和入瞳求解结果。
    if sample.pupil_coordinates is None:
        raise ValueError("sample.pupil_coordinates is required to build traced rays.")
    if entrance_pupil.entrance_pupil_z is None or entrance_pupil.entrance_pupil_radius is None:
        raise ValueError("entrance pupil z/radius is required to build traced rays.")

    device = default_device(system)
    entrance_pupil_z = torch.as_tensor(
        entrance_pupil.entrance_pupil_z,
        dtype=torch.float64,
        device=device,
    )
    entrance_pupil_radius = torch.as_tensor(
        entrance_pupil.entrance_pupil_radius,
        dtype=torch.float64,
        device=device,
    )

    # 2. 将系统的视场和波长索引整理为 batch-first tensor。
    field_angles = torch.tensor(
        [(float(field.x), float(field.y)) for field in fields],
        dtype=torch.float64,
        device=device,
    ).reshape(1, len(fields), 2).expand(system.system_count, -1, -1)
    wavelength_index = torch.tensor(
        wavelength_indices,
        dtype=torch.int64,
        device=device,
    )

    # 3. 复用公开构造接口，统一完成广播和光线组装。
    return build_pupil_rays(
        system,
        field_angles=field_angles,
        entrance_pupil_z=entrance_pupil_z,
        entrance_pupil_radius=entrance_pupil_radius,
        pupil_coordinates=sample.pupil_coordinates.to(dtype=torch.float64, device=device),
        wavelength_indices=wavelength_index,
    )


def _assemble_pupil_rays(
    system: MultiOpticalSystem,
    *,
    field_slopes: torch.Tensor,
    entrance_pupil_z: torch.Tensor,
    entrance_pupil_radius: torch.Tensor,
    pupil_coordinates: torch.Tensor,
    wavelength_indices: torch.Tensor,
) -> RayBundle:
    """组装 design x field x wavelength x ray 光线张量。"""
    device = field_slopes.device
    px, py = _pupil_coordinate_tensors(pupil_coordinates, device=device)
    design_count, field_count, _ = field_slopes.shape
    wavelength_count = wavelength_indices.shape[0]
    ray_count = px.shape[0]
    field_l = field_slopes[..., 0].reshape(design_count, field_count, 1, 1)
    field_m = field_slopes[..., 1].reshape(design_count, field_count, 1, 1)
    pupil_z = entrance_pupil_z.unsqueeze(-1)
    pupil_radius = entrance_pupil_radius.unsqueeze(-1)
    pupil_x = px.reshape(1, 1, 1, ray_count) * pupil_radius
    pupil_y = py.reshape(1, 1, 1, ray_count) * pupil_radius

    # 在首面顶点平面 z=0 上组装无限远平面波或有限物距点源光线。
    object_distance_mm = float(system.architecture.object_distance_mm)
    if math.isfinite(object_distance_mm):
        object_z = -object_distance_mm
        object_to_pupil = pupil_z - object_z
        object_x = -field_l * object_to_pupil
        object_y = -field_m * object_to_pupil
        l_tensor = (pupil_x - object_x) / object_to_pupil
        m_tensor = (pupil_y - object_y) / object_to_pupil
        x = object_x + l_tensor * (-object_z)
        y = object_y + m_tensor * (-object_z)
    else:
        l_tensor = field_l.expand(design_count, -1, wavelength_count, ray_count)
        m_tensor = field_m.expand(design_count, -1, wavelength_count, ray_count)
        x = -pupil_z * l_tensor + pupil_x
        y = -pupil_z * m_tensor + pupil_y
    z = torch.zeros_like(x)

    # 方向分量采用 (slope_x, slope_y, 1) 形式，追迹过程中与 z=0 起点组合后可得到正确的空间直线。
    n_tensor = torch.ones_like(x)
    wavelength_tensor = wavelength_indices.reshape(1, 1, wavelength_count, 1).expand(
        design_count, field_count, -1, ray_count
    )

    if math.isfinite(object_distance_mm):
        # 有限物距点源在首面前已经积累了随 pupil 变化的球面波光程。
        initial_opl = torch.sqrt(
            (x - object_x) * (x - object_x)
            + (y - object_y) * (y - object_y)
            + object_distance_mm * object_distance_mm
        )
    else:
        # 倾斜平面波在 z=0 输入平面上具有线性初始相位。
        direction_norm = torch.sqrt(l_tensor * l_tensor + m_tensor * m_tensor + n_tensor * n_tensor)
        initial_opl = (l_tensor * x + m_tensor * y + n_tensor * z) / direction_norm

    return RayBundle(
        x=x,
        y=y,
        z=z,
        l=l_tensor,
        m=m_tensor,
        n=n_tensor,
        wavelength_index=wavelength_tensor,
        intensity=torch.ones_like(x),
        opl=initial_opl,
    )


def _pupil_coordinate_tensors(
    pupil_coordinates: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 pupil 坐标规整成 `(px, py)` 张量。"""
    coordinates = torch.as_tensor(pupil_coordinates, dtype=torch.float64)
    if coordinates.device != device:
        coordinates = coordinates.to(device=device)
    if coordinates.ndim == 1:
        if coordinates.shape[0] != 2:
            raise ValueError("pupil_coordinates must end with (px, py).")
        coordinates = coordinates.reshape(1, 2)
    elif coordinates.shape[-1] != 2:
        raise ValueError("pupil_coordinates must end with (px, py).")

    flattened = coordinates.reshape(-1, 2)
    return flattened[:, 0], flattened[:, 1]


def _field_slopes(system: MultiOpticalSystem, field_angles: torch.Tensor) -> torch.Tensor:
    """把视场角 tensor 转换为 x/y 方向斜率。"""
    if system.fields.field_type != "angle":
        raise NotImplementedError(
            "SequentialSurfaceRayTracer only supports angle fields for sampled tracing."
        )
    if system.config.units.angle == "deg":
        field_angles = torch.deg2rad(field_angles)
    elif system.config.units.angle != "rad":
        raise ValueError(f"Unsupported angle unit: {system.config.units.angle!r}")
    return torch.tan(field_angles)


def _broadcast_batch_value(
    value,
    batch_shape: tuple[int, int, int],
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    """把标量、design 向量或 3-D tensor 广播到 batch 形状。"""
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.device != device:
        tensor = tensor.to(device=device)
    design_count, _, _ = batch_shape

    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1, 1)
    elif tensor.ndim == 1:
        if tensor.shape[0] != design_count:
            raise ValueError(f"{name} must be scalar, design vector, or 3-D batch tensor.")
        tensor = tensor.reshape(design_count, 1, 1)
    elif tensor.ndim != 3:
        raise ValueError(f"{name} must be scalar, design vector, or 3-D batch tensor.")

    try:
        return torch.broadcast_to(tensor, batch_shape).clone()
    except RuntimeError as exc:
        raise ValueError(f"{name} shape is not compatible with batch shape {batch_shape}.") from exc
