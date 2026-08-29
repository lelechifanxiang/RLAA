from __future__ import annotations

import optics_core as oc
from tests.fixtures.cases import MULTIFIELD_PARAXIAL_PARAMETER_VECTORS
from tests.zemax.temp_structures import SphericalForwardTraceCaseSpec


def build_surface_trace_architecture(surface: oc.Surface, *, name: str = "surface_trace") -> oc.OpticalArchitecture:
    """构造单面 kernel 测试用的最小顺序系统。"""
    architecture = oc.OpticalArchitecture(name=name)
    architecture.surfaces.add(surface)
    architecture.surfaces.add_image(label="IMG")
    return architecture


def build_paraxial_architecture(
    *,
    focal_length: float,
    thickness: float,
    semi_diameter: float,
    name: str,
    label: str | None = None,
    is_stop: bool = False,
) -> oc.OpticalArchitecture:
    """构造单近轴面配方。"""
    architecture = oc.OpticalArchitecture(name=name)
    architecture.surfaces.add_paraxial(
        focal_length=focal_length,
        thickness=thickness,
        semi_diameter=semi_diameter,
        label=label,
        is_stop=is_stop,
    )
    architecture.surfaces.add_image(label="IMG")
    return architecture


def build_spherical_forward_architecture(
    spec: SphericalForwardTraceCaseSpec,
    *,
    name: str,
) -> oc.OpticalArchitecture:
    """按球面 case 构造多球面配方。"""
    architecture = oc.OpticalArchitecture(name=name)
    for surface_index, surface_spec in enumerate(spec.surfaces):
        medium = None
        if surface_spec.nd is not None and surface_spec.vd is not None:
            medium = oc.AbbeModelMaterial(
                name=f"MODEL_{surface_index + 1}",
                nd=surface_spec.nd,
                vd=surface_spec.vd,
            )
        architecture.surfaces.add_sphere(
            radius=surface_spec.radius_mm,
            thickness=surface_spec.thickness_mm,
            medium=medium,
            semi_diameter=surface_spec.aperture_radius_mm,
            label=surface_spec.comment,
            is_stop=surface_index == 0,
        )
    architecture.surfaces.add_image(label="IMG")
    return architecture


def build_multifield_paraxial_parameter_schema() -> oc.ParameterSchema:
    """定义多结构近轴系统的参数槽位。"""
    return oc.ParameterSchema(
        [
            oc.ParameterSpec(
                name="focal_length",
                path="surface[0].geometry.focal_length",
                default=40.0,
            ),
            oc.ParameterSpec(
                name="image_distance",
                path="surface[0].gap.thickness",
                default=40.0,
            ),
        ]
    )


def build_multifield_paraxial_parameters(
    schema: oc.ParameterSchema,
) -> oc.ParameterVectorBatch:
    """构造多结构近轴测试的参数批。"""
    return oc.ParameterVectorBatch(
        schema=schema,
        vectors=[list(vector) for vector in MULTIFIELD_PARAXIAL_PARAMETER_VECTORS],
        grid_shape=(len(MULTIFIELD_PARAXIAL_PARAMETER_VECTORS),),
    )


def build_forward_multi_sphere_parameter_schema(
    spec: SphericalForwardTraceCaseSpec,
) -> oc.ParameterSchema:
    """定义多球面批量系统的参数槽位。"""
    return oc.ParameterSchema(
        [
            oc.ParameterSpec(
                name="s1_radius",
                path="surface[0].geometry.radius",
                default=spec.surfaces[0].radius_mm,
            ),
            oc.ParameterSpec(
                name="s3_radius",
                path="surface[2].geometry.radius",
                default=spec.surfaces[2].radius_mm,
            ),
        ]
    )


def build_forward_multi_sphere_parameters(
    schema: oc.ParameterSchema,
) -> oc.ParameterVectorBatch:
    """构造多球面批量系统的参数批。"""
    vectors = (
        [45.0, 60.0],
        [52.0, 68.0],
    )
    return oc.ParameterVectorBatch(
        schema=schema,
        vectors=[list(vector) for vector in vectors],
        grid_shape=(len(vectors),),
    )
