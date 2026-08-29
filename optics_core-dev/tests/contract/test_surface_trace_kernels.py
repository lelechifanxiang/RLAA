from __future__ import annotations

import pytest
import torch

import optics_core as oc
from tests.fixtures import build_surface_trace_system, prepare_trace_materials


pytestmark = pytest.mark.contract


def build_test_rays(*, x: float = 1.0, z: float = -10.0) -> oc.RayBundle:
    return oc.RayBundle(
        x=torch.tensor([[x]], dtype=torch.float64),
        y=torch.zeros((1, 1), dtype=torch.float64),
        z=torch.full((1, 1), z, dtype=torch.float64),
        l=torch.zeros((1, 1), dtype=torch.float64),
        m=torch.zeros((1, 1), dtype=torch.float64),
        n=torch.ones((1, 1), dtype=torch.float64),
        wavelength_index=torch.zeros((1, 1), dtype=torch.int64),
    )


def test_even_asphere_surface_reuses_same_sag_trace_kernel_as_sphere() -> None:
    material = oc.AbbeModelMaterial(name="LENS", nd=1.5, vd=60.0)
    sphere_system = build_surface_trace_system(
        oc.SphereSurface(
            radius=50.0,
            thickness=15.0,
            medium=material,
            semi_diameter=10.0,
            label="S1",
        )
    )
    even_system = build_surface_trace_system(
        oc.EvenAsphereSurface(
            radius=50.0,
            thickness=15.0,
            medium=material,
            semi_diameter=10.0,
            coefficients=(),
            label="A1",
        )
    )

    rays = build_test_rays(x=1.0)
    sphere_result = sphere_system.tracer.trace(
        sphere_system,
        rays,
        options=oc.TraceOptions(record_intersections=True),
    )
    even_result = even_system.tracer.trace(
        even_system,
        rays,
        options=oc.TraceOptions(record_intersections=True),
    )

    assert torch.all(sphere_result.valid)
    assert torch.all(even_result.valid)
    torch.testing.assert_close(sphere_result.intersections[0].position[0], even_result.intersections[0].position[0])
    torch.testing.assert_close(sphere_result.intersections[0].position[1], even_result.intersections[0].position[1])
    torch.testing.assert_close(sphere_result.intersections[0].position[2], even_result.intersections[0].position[2])
    torch.testing.assert_close(sphere_result.rays.l, even_result.rays.l)
    torch.testing.assert_close(sphere_result.rays.m, even_result.rays.m)
    torch.testing.assert_close(sphere_result.rays.n, even_result.rays.n)


def test_sag_surface_aperture_clipping_marks_ray_invalid() -> None:
    system = build_surface_trace_system(
        oc.SphereSurface(
            radius=50.0,
            thickness=0.0,
            semi_diameter=1.0,
            semi_diameter_solve="fixed",
            aperture_type="floating",
            label="S1",
        )
    )

    result = system.tracer.trace(
        system,
        build_test_rays(x=2.0),
        options=oc.TraceOptions(stop_surface=0, record_intersections=True),
    )

    assert not bool(result.valid.item())
    assert torch.isnan(result.intersections[0].position[0]).all()
    assert torch.isnan(result.rays.l).all()


def test_nonempty_even_asphere_coefficients_fail_fast() -> None:
    system = build_surface_trace_system(
        oc.EvenAsphereSurface(
            radius=50.0,
            thickness=0.0,
            semi_diameter=10.0,
            coefficients=(1e-5,),
            label="A1",
        )
    )

    with pytest.raises(NotImplementedError, match="EvenAsphereGeometry"):
        system.tracer.trace(
            system,
            build_test_rays(x=1.0),
            options=oc.TraceOptions(stop_surface=0, record_intersections=True),
        )


def test_total_internal_refraction_marks_refractive_ray_invalid() -> None:
    glass = oc.AbbeModelMaterial(name="GLASS", nd=1.5, vd=60.0)
    architecture = oc.OpticalArchitecture(name="tir_surface")
    architecture.surfaces.add_sphere(
        radius=0.0,
        thickness=10.0,
        medium=glass,
        semi_diameter=100.0,
        label="ENTRY",
    )
    architecture.surfaces.add_sphere(
        radius=0.0,
        thickness=0.0,
        medium=None,
        semi_diameter=100.0,
        label="EXIT",
    )
    system = oc.MultiOpticalSystem(
        architecture=architecture,
        tracer=oc.SequentialSurfaceRayTracer(),
    )
    system.wavelengths.add(0.5875618, is_primary=True)
    prepare_trace_materials(system)
    direction_l = torch.tensor([[0.9]], dtype=torch.float64)
    direction_n = torch.sqrt(1.0 - direction_l * direction_l)
    rays = oc.RayBundle(
        x=torch.zeros((1, 1), dtype=torch.float64),
        y=torch.zeros((1, 1), dtype=torch.float64),
        z=torch.full((1, 1), 9.0, dtype=torch.float64),
        l=direction_l,
        m=torch.zeros((1, 1), dtype=torch.float64),
        n=direction_n,
        wavelength_index=torch.zeros((1, 1), dtype=torch.int64),
    )

    result = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(start_surface=1, stop_surface=1, record_intersections=True),
    )

    assert not bool(result.valid.item())
    assert torch.isfinite(result.intersections[0].position[2]).all()
    assert torch.isnan(result.rays.l).all()
    assert torch.isnan(result.rays.opl).all()


def test_record_opd_accumulates_air_geometric_path() -> None:
    system = build_surface_trace_system(
        oc.SphereSurface(
            radius=0.0,
            thickness=0.0,
            semi_diameter=10.0,
            label="S1",
        )
    )

    result = system.tracer.trace(
        system,
        build_test_rays(x=0.0, z=-10.0),
        options=oc.TraceOptions(stop_surface=0),
    )

    torch.testing.assert_close(result.rays.opl, torch.full((1, 1), 10.0, dtype=torch.float64))


def test_record_opd_accumulates_glass_optical_path_to_image() -> None:
    wavelength = 0.5875618
    glass = oc.AbbeModelMaterial(name="GLASS", nd=1.5, vd=60.0)
    system = build_surface_trace_system(
        oc.SphereSurface(
            radius=0.0,
            thickness=15.0,
            medium=glass,
            semi_diameter=10.0,
            label="S1",
        )
    )
    rays = build_test_rays(x=0.0, z=-10.0)

    result = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(),
    )

    glass_index = glass.refractive_index(torch.full((1, 1), wavelength, dtype=torch.float64))
    expected_opl = 10.0 + glass_index * 15.0
    torch.testing.assert_close(result.rays.opl, expected_opl)


def test_record_opd_uses_signed_path_for_negative_thickness() -> None:
    """负厚度传播应扣减光程，避免坐标恢复结构引入假相位。"""
    architecture = oc.OpticalArchitecture(name="negative_thickness")
    architecture.surfaces.add_sphere(
        radius=0.0,
        thickness=-5.0,
        semi_diameter=10.0,
        label="S1",
    )
    architecture.surfaces.add_sphere(
        radius=0.0,
        thickness=0.0,
        semi_diameter=10.0,
        label="S2",
    )
    system = oc.MultiOpticalSystem(
        architecture=architecture,
        tracer=oc.SequentialSurfaceRayTracer(),
    )
    system.wavelengths.add(0.5875618, is_primary=True)
    prepare_trace_materials(system)

    result = system.tracer.trace(
        system,
        build_test_rays(x=0.0, z=-10.0),
        options=oc.TraceOptions(stop_surface=1),
    )

    torch.testing.assert_close(result.rays.opl, torch.full((1, 1), 5.0, dtype=torch.float64))


def test_negative_sag_surface_after_coordinate_break_accepts_local_negative_path() -> None:
    """零厚度 CB 后的负 sag 面应允许回到顶点平面之前完成求交。"""
    architecture = oc.OpticalArchitecture(name="coordinate_break_negative_sag")
    architecture.surfaces.add_coordinate_break(thickness=0.0, label="CB")
    surface = architecture.surfaces.add_sphere(
        radius=-10.0,
        thickness=0.0,
        semi_diameter=5.0,
        semi_diameter_solve="fixed",
        aperture_type="floating",
        label="S1",
    )
    system = oc.MultiOpticalSystem(
        architecture=architecture,
        tracer=oc.SequentialSurfaceRayTracer(),
    )
    system.wavelengths.add(0.5875618, is_primary=True)
    prepare_trace_materials(system)

    result = system.tracer.trace(
        system,
        build_test_rays(x=1.0, z=-1.0),
        options=oc.TraceOptions(stop_surface=1),
    )

    expected_z = surface.geometry.sag(
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
    )
    assert torch.all(result.valid)
    torch.testing.assert_close(result.rays.z, expected_z)


def test_real_tracer_rejects_unimplemented_record_options() -> None:
    system = build_surface_trace_system(
        oc.SphereSurface(
            radius=50.0,
            thickness=0.0,
            semi_diameter=10.0,
            label="S1",
        )
    )

    with pytest.raises(NotImplementedError, match="record_ray_angles"):
        system.tracer.trace(system, build_test_rays(), options=oc.TraceOptions(record_ray_angles=True))
