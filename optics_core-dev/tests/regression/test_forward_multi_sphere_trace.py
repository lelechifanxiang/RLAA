from __future__ import annotations

import math

import pytest
import torch

import optics_core as oc
from tests.fixtures import (
    DEFAULT_FORWARD_MULTI_SPHERE_CASE,
    build_multi_sphere_system,
    build_multi_system_multi_sphere_system,
)


pytestmark = pytest.mark.regression


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vector))
    return vector[0] / norm, vector[1] / norm, vector[2] / norm


def _intersect_standard_surface(
    radius: float,
    conic: float,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> float:
    ox, oy, oz = origin
    dx, dy, dz = direction
    if radius == 0.0:
        return -oz / dz

    conic_factor = 1.0 + conic
    quadratic_a = dx * dx + dy * dy + conic_factor * dz * dz
    half_b = ox * dx + oy * dy + (conic_factor * oz - radius) * dz
    quadratic_c = ox * ox + oy * oy + conic_factor * oz * oz - 2.0 * radius * oz
    discriminant = half_b * half_b - quadratic_a * quadratic_c
    if discriminant < 0.0:
        return math.nan

    sqrt_discriminant = math.sqrt(discriminant)
    roots = (
        (-half_b - sqrt_discriminant) / quadratic_a,
        (-half_b + sqrt_discriminant) / quadratic_a,
    )
    positive_roots = [root for root in roots if root >= 0.0]
    if not positive_roots:
        return math.nan
    return min(positive_roots)


def _standard_surface_normal(radius: float, conic: float, x: float, y: float) -> tuple[float, float, float]:
    if radius == 0.0:
        return 0.0, 0.0, 1.0

    rho_sq = x * x + y * y
    sqrt_term = math.sqrt(1.0 - (1.0 + conic) * rho_sq / (radius * radius))
    dz_dx = x / (radius * sqrt_term)
    dz_dy = y / (radius * sqrt_term)
    return _normalize((-dz_dx, -dz_dy, 1.0))


def _material_index(material: oc.Material | None, wavelength_um: float) -> float:
    medium = oc.AIR if material is None else material
    return float(medium.refractive_index(torch.tensor(wavelength_um, dtype=torch.float64)).item())


def _refract(
    direction: tuple[float, float, float],
    normal: tuple[float, float, float],
    incident_index: float,
    transmitted_index: float,
) -> tuple[float, float, float]:
    dot_product = sum(direction[index] * normal[index] for index in range(3))
    oriented_normal = tuple(-component for component in normal) if dot_product > 0.0 else normal
    cos_i = -sum(direction[index] * oriented_normal[index] for index in range(3))
    eta = incident_index / transmitted_index
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    if k < 0.0:
        return math.nan, math.nan, math.nan

    sqrt_k = math.sqrt(k)
    refracted = tuple(
        eta * direction[index] + (eta * cos_i - sqrt_k) * oriented_normal[index]
        for index in range(3)
    )
    return _normalize(refracted)


def _surface_positions(surfaces: list[oc.Surface]) -> list[float]:
    positions: list[float] = []
    z_position = 0.0
    for surface in surfaces:
        positions.append(z_position)
        thickness = float(surface.gap.thickness)
        if math.isfinite(thickness):
            z_position += thickness
    return positions


def _reference_forward_trace(
    surfaces: list[oc.Surface],
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    wavelength_um: float,
) -> tuple[list[tuple[float, float, float]], tuple[float, float, float]]:
    positions = _surface_positions(surfaces)
    current_origin = origin
    current_direction = _normalize(direction)
    hits: list[tuple[float, float, float]] = []

    for surface_index, surface in enumerate(surfaces):
        surface_z = positions[surface_index]
        if isinstance(surface, oc.ImageSurface):
            path = (surface_z - current_origin[2]) / current_direction[2]
            hit = (
                current_origin[0] + current_direction[0] * path,
                current_origin[1] + current_direction[1] * path,
                surface_z,
            )
            hits.append(hit)
            current_origin = hit
            continue

        assert isinstance(surface.geometry, oc.StandardGeometry)
        local_origin = (
            current_origin[0],
            current_origin[1],
            current_origin[2] - surface_z,
        )
        path = _intersect_standard_surface(
            float(surface.geometry.radius),
            float(surface.geometry.conic),
            local_origin,
            current_direction,
        )
        local_hit = (
            local_origin[0] + current_direction[0] * path,
            local_origin[1] + current_direction[1] * path,
            local_origin[2] + current_direction[2] * path,
        )
        hit = (local_hit[0], local_hit[1], surface_z + local_hit[2])
        hits.append(hit)

        normal = _standard_surface_normal(
            float(surface.geometry.radius),
            float(surface.geometry.conic),
            local_hit[0],
            local_hit[1],
        )
        incident_medium = None if surface_index == 0 else surfaces[surface_index - 1].gap.medium
        transmitted_medium = surface.gap.medium
        current_direction = _refract(
            current_direction,
            normal,
            _material_index(incident_medium, wavelength_um),
            _material_index(transmitted_medium, wavelength_um),
        )
        current_origin = hit

    return hits, current_direction


def test_forward_multi_sphere_trace_matches_independent_reference() -> None:
    system = build_multi_sphere_system(DEFAULT_FORWARD_MULTI_SPHERE_CASE)
    ray_count = 3
    initial_l = torch.tensor([[0.0, 0.035, -0.025]], dtype=torch.float64)
    initial_m = torch.tensor([[0.0, -0.015, 0.02]], dtype=torch.float64)
    initial_n = torch.sqrt(1.0 - initial_l * initial_l - initial_m * initial_m)
    rays = oc.RayBundle(
        x=torch.tensor([[0.0, 1.2, -1.0]], dtype=torch.float64),
        y=torch.tensor([[0.0, -0.8, 0.7]], dtype=torch.float64),
        z=torch.full((1, ray_count), -15.0, dtype=torch.float64),
        l=initial_l,
        m=initial_m,
        n=initial_n,
        wavelength_index=torch.arange(3, dtype=torch.int64).reshape(1, 3),
    )

    result = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(record_intersections=True),
    )

    expected_hits_by_surface = [[] for _ in range(len(system.surfaces))]
    expected_directions = []
    for ray_index in range(ray_count):
        hits, direction = _reference_forward_trace(
            list(system.surfaces),
            (
                float(rays.x[0, ray_index]),
                float(rays.y[0, ray_index]),
                float(rays.z[0, ray_index]),
            ),
            (
                float(rays.l[0, ray_index]),
                float(rays.m[0, ray_index]),
                float(rays.n[0, ray_index]),
            ),
            float(system.wavelengths[int(rays.wavelength_index[0, ray_index])].value_um),
        )
        for surface_index, hit in enumerate(hits):
            expected_hits_by_surface[surface_index].append(hit)
        expected_directions.append(direction)

    print(f"球面数量: {len(system.surfaces) - 1}")
    print(f"波长 (um): {[float(wavelength.value_um) for wavelength in system.wavelengths]}")
    print(f"最终像面 x: {result.rays.x.tolist()[0]}")
    print(f"最终像面 y: {result.rays.y.tolist()[0]}")

    assert torch.all(result.valid)
    assert len(result.intersections) == len(system.surfaces)
    for surface_index, expected_hits in enumerate(expected_hits_by_surface):
        expected = torch.tensor(expected_hits, dtype=torch.float64).T.unsqueeze(0)
        hit = result.intersections[surface_index]
        actual = torch.stack(hit.position, dim=1)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0.0)

    expected_direction_tensor = torch.tensor(expected_directions, dtype=torch.float64).T.unsqueeze(0)
    actual_direction_tensor = torch.stack((result.rays.l, result.rays.m, result.rays.n), dim=1)
    torch.testing.assert_close(actual_direction_tensor, expected_direction_tensor, atol=1e-10, rtol=0.0)


def test_forward_multi_sphere_trace_preserves_multi_system_field_wavelength_batch() -> None:
    # 初始化多结构系统
    system = build_multi_system_multi_sphere_system()

    # 构造多视场、多波长、多光线输入
    shape = (system.system_count, 2, 3, 2)
    field_l = torch.tensor([0.0, 0.025], dtype=torch.float64).reshape(1, 2, 1, 1)
    field_m = torch.tensor([0.0, -0.015], dtype=torch.float64).reshape(1, 2, 1, 1)
    ray_offset_x = torch.tensor([-0.8, 0.9], dtype=torch.float64).reshape(1, 1, 1, 2)
    ray_offset_y = torch.tensor([0.5, -0.4], dtype=torch.float64).reshape(1, 1, 1, 2)
    l = field_l.expand(shape)
    m = field_m.expand(shape)
    n = torch.sqrt(1.0 - l * l - m * m)
    rays = oc.RayBundle(
        x=ray_offset_x.expand(shape).clone(),
        y=ray_offset_y.expand(shape).clone(),
        z=torch.full(shape, -15.0, dtype=torch.float64),
        l=l,
        m=m,
        n=n,
        wavelength_index=torch.arange(3, dtype=torch.int64)
        .reshape(1, 1, 3, 1)
        .expand(shape),
    )

    # 批量追迹
    batched = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(record_intersections=True),
    )

    assert torch.all(batched.valid)
    assert tuple(batched.rays.x.shape) == shape
    assert len(batched.intersections) == len(system.surfaces)

    # 与单结构 design_view 对齐
    for system_index in range(system.system_count):
        design = system.design_view(system_index)
        design_rays = oc.RayBundle(
            x=rays.x[system_index : system_index + 1],
            y=rays.y[system_index : system_index + 1],
            z=rays.z[system_index : system_index + 1],
            l=rays.l[system_index : system_index + 1],
            m=rays.m[system_index : system_index + 1],
            n=rays.n[system_index : system_index + 1],
            wavelength_index=rays.wavelength_index[system_index : system_index + 1],
        )
        design_result = design.tracer.trace(
            design,
            design_rays,
            options=oc.TraceOptions(record_intersections=True),
        )

        print(f"系统 {system_index} 参数: {system.parameter_vectors[system_index]}")
        print(f"批量像面 x: {batched.rays.x[system_index].reshape(-1).tolist()}")
        print(f"单系统像面 x: {design_result.rays.x.reshape(-1).tolist()}")
        torch.testing.assert_close(batched.rays.x[system_index : system_index + 1], design_result.rays.x)
        torch.testing.assert_close(batched.rays.y[system_index : system_index + 1], design_result.rays.y)
        torch.testing.assert_close(batched.rays.z[system_index : system_index + 1], design_result.rays.z)
        torch.testing.assert_close(batched.rays.l[system_index : system_index + 1], design_result.rays.l)
        torch.testing.assert_close(batched.rays.m[system_index : system_index + 1], design_result.rays.m)
        torch.testing.assert_close(batched.rays.n[system_index : system_index + 1], design_result.rays.n)
        for surface_index, intersection in enumerate(batched.intersections):
            design_intersection = design_result.intersections[surface_index]
            for component_index in range(3):
                torch.testing.assert_close(
                    intersection.position[component_index][system_index : system_index + 1],
                    design_intersection.position[component_index],
                )
