from __future__ import annotations

import torch

from .._material_batch import BatchedMaterialData
from ._types import SurfaceHit
from ..surfaces import CoordinateBreak, ImageSurface, ObjectSurface, Surface
from ..types import TraceDirection


def _travel_direction(
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
    *,
    direction: TraceDirection,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if direction == "forward":
        return l, m, n
    return -l, -m, -n


def _store_travel_direction(
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
    *,
    direction: TraceDirection,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if direction == "forward":
        return l, m, n
    return -l, -m, -n


def _normalize_direction(
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    norm = torch.sqrt(l * l + m * m + n * n)
    safe_norm = torch.where(norm > 0.0, norm, torch.ones_like(norm))
    return l / safe_norm, m / safe_norm, n / safe_norm


def _apply_surface_interaction(
    surface: Surface,
    surface_index: int,
    hit: SurfaceHit,
    material_data: BatchedMaterialData,
    wavelength_index: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
    *,
    direction: TraceDirection,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(surface, (ObjectSurface, ImageSurface, CoordinateBreak)):
        return l, m, n
    if hit.normal is None:
        raise ValueError("surface interaction requires a valid surface normal.")
    return _apply_refractive_interaction(
        material_data,
        surface_index,
        hit,
        wavelength_index,
        l,
        m,
        n,
        direction=direction,
    )


def _apply_refractive_interaction(
    material_data: BatchedMaterialData,
    surface_index: int,
    hit: SurfaceHit,
    wavelength_index: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
    *,
    direction: TraceDirection,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    incident_material, transmitted_material = material_data.surface_indices(surface_index, direction=direction)
    incident_index = material_data.refractive_index(incident_material, wavelength_index)
    transmitted_index = material_data.refractive_index(transmitted_material, wavelength_index)

    travel_l, travel_m, travel_n = _travel_direction(l, m, n, direction=direction)
    unit_l, unit_m, unit_n = _normalize_direction(travel_l, travel_m, travel_n)
    normal_x, normal_y, normal_z = hit.normal
    dot_product = unit_l * normal_x + unit_m * normal_y + unit_n * normal_z
    oriented_normal_x = torch.where(dot_product > 0.0, -normal_x, normal_x)
    oriented_normal_y = torch.where(dot_product > 0.0, -normal_y, normal_y)
    oriented_normal_z = torch.where(dot_product > 0.0, -normal_z, normal_z)

    cos_i = -(unit_l * oriented_normal_x + unit_m * oriented_normal_y + unit_n * oriented_normal_z)
    eta = incident_index / transmitted_index
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    sqrt_k = torch.sqrt(torch.clamp_min(k, 0.0))

    refracted_l = eta * unit_l + (eta * cos_i - sqrt_k) * oriented_normal_x
    refracted_m = eta * unit_m + (eta * cos_i - sqrt_k) * oriented_normal_y
    refracted_n = eta * unit_n + (eta * cos_i - sqrt_k) * oriented_normal_z

    valid_refraction = (k >= 0.0) & hit.valid
    nan_tensor = torch.full_like(refracted_l, torch.nan)
    outgoing_l = torch.where(valid_refraction, refracted_l, nan_tensor)
    outgoing_m = torch.where(valid_refraction, refracted_m, nan_tensor)
    outgoing_n = torch.where(valid_refraction, refracted_n, nan_tensor)
    outgoing_l, outgoing_m, outgoing_n = _normalize_direction(outgoing_l, outgoing_m, outgoing_n)
    return _store_travel_direction(outgoing_l, outgoing_m, outgoing_n, direction=direction)
