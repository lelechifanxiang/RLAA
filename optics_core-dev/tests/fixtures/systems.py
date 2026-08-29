from __future__ import annotations

import optics_core as oc
from optics_core._material_batch import compile_batched_material_data
from optics_core._runtime import default_device
from tests.fixtures.cases import (
    DEFAULT_FORWARD_MULTI_SPHERE_CASE,
    MULTIFIELD_PARAXIAL_FIELD_CASES,
)
from tests.fixtures.recipes import (
    build_forward_multi_sphere_parameter_schema,
    build_forward_multi_sphere_parameters,
    build_multifield_paraxial_parameter_schema,
    build_multifield_paraxial_parameters,
    build_paraxial_architecture,
    build_spherical_forward_architecture,
    build_surface_trace_architecture,
)
from tests.zemax.temp_structures import SphericalForwardTraceCaseSpec


def prepare_trace_materials(system: oc.MultiOpticalSystem) -> oc.MultiOpticalSystem:
    system._material_data = compile_batched_material_data(system, device=default_device(system))
    return system


def build_tracing_system(
    architecture: oc.OpticalArchitecture,
    *,
    name: str | None = None,
    parameter_schema: oc.ParameterSchema | None = None,
    parameters: oc.ParameterVectorBatch | None = None,
    field_points: tuple[tuple[float, float], ...] = (),
    wavelengths_um: tuple[float, ...] = (),
    wavelength_labels: tuple[str | None, ...] | None = None,
    primary_wavelength_index: int = 0,
    aperture_diameter: float | None = None,
    stop_surface: int | None = None,
) -> oc.MultiOpticalSystem:
    """按测试常用配置装配顺序追迹系统。"""
    system = oc.MultiOpticalSystem(
        architecture=architecture,
        name=name,
        parameter_schema=parameter_schema,
        parameters=parameters,
        tracer=oc.SequentialSurfaceRayTracer(),
    )
    if stop_surface is not None:
        system.surfaces.set_stop(stop_surface)
    if field_points:
        system.fields.set_type("angle")
        for field_index, (field_x_deg, field_y_deg) in enumerate(field_points):
            label = "on_axis" if field_index == 0 and field_x_deg == 0.0 and field_y_deg == 0.0 else f"field_{field_index}"
            system.fields.add(x=field_x_deg, y=field_y_deg, label=label)
    if wavelengths_um:
        for index, wavelength_um in enumerate(wavelengths_um):
            system.wavelengths.add(
                wavelength_um,
                is_primary=index == primary_wavelength_index,
                label=None if wavelength_labels is None or index >= len(wavelength_labels) else wavelength_labels[index],
            )
    if aperture_diameter is not None:
        system.set_aperture(
            "entrance_pupil_diameter",
            aperture_diameter,
            label="EPD",
        )
    return system


def build_surface_trace_system(surface: oc.Surface) -> oc.MultiOpticalSystem:
    """构造单面 trace kernel 测试系统。"""
    return prepare_trace_materials(
        build_tracing_system(build_surface_trace_architecture(surface), wavelengths_um=(0.5875618,))
    )


def build_backward_paraxial_system() -> oc.MultiOpticalSystem:
    """构造反向近轴测试系统。"""
    architecture = build_paraxial_architecture(
        focal_length=40.0,
        thickness=20.0,
        semi_diameter=6.0,
        name="backward_paraxial",
        label="PX",
    )
    return prepare_trace_materials(
        build_tracing_system(
            architecture,
            field_points=((0.0, 0.0), (0.0, 1.0)),
            wavelengths_um=(0.4861, 0.5876),
            wavelength_labels=("F", "d"),
            primary_wavelength_index=1,
            aperture_diameter=12.0,
            stop_surface=1,
        )
    )


def build_multifield_multistructure_system() -> oc.MultiOpticalSystem:
    """构造多视场、多结构的近轴测试系统。"""
    architecture = build_paraxial_architecture(
        focal_length=40.0,
        thickness=40.0,
        semi_diameter=6.0,
        name="multifield_paraxial_architecture",
        is_stop=True,
    )
    parameter_schema = build_multifield_paraxial_parameter_schema()
    parameters = build_multifield_paraxial_parameters(parameter_schema)
    field_points = tuple((20.0 * hx, 30.0 * hy) for hx, hy in MULTIFIELD_PARAXIAL_FIELD_CASES)
    return prepare_trace_materials(
        build_tracing_system(
            architecture,
            name="multifield_paraxial",
            parameter_schema=parameter_schema,
            parameters=parameters,
            field_points=field_points,
            wavelengths_um=(0.5876,),
            wavelength_labels=("d",),
            aperture_diameter=12.0,
            stop_surface=0,
        )
    )


def build_multi_sphere_system(
    spec: SphericalForwardTraceCaseSpec = DEFAULT_FORWARD_MULTI_SPHERE_CASE,
) -> oc.MultiOpticalSystem:
    """构造独立参考回归使用的多球面系统。"""
    architecture = build_spherical_forward_architecture(
        spec,
        name="forward_multi_sphere",
    )
    return prepare_trace_materials(build_tracing_system(architecture, wavelengths_um=spec.wavelengths_um))


def build_multi_system_multi_sphere_system(
    spec: SphericalForwardTraceCaseSpec = DEFAULT_FORWARD_MULTI_SPHERE_CASE,
) -> oc.MultiOpticalSystem:
    """构造多结构多球面系统。"""
    architecture = build_spherical_forward_architecture(
        spec,
        name="forward_multi_sphere",
    )
    parameter_schema = build_forward_multi_sphere_parameter_schema(spec)
    parameters = build_forward_multi_sphere_parameters(parameter_schema)
    return prepare_trace_materials(
        build_tracing_system(
            architecture,
            parameter_schema=parameter_schema,
            parameters=parameters,
            wavelengths_um=spec.wavelengths_um,
        )
    )
