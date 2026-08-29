from __future__ import annotations

from tests.zemax.temp_structures import (
    SphericalForwardSurfaceSpec,
    SphericalForwardTraceCaseSpec,
)


MULTIFIELD_PARAXIAL_FIELD_CASES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (0.0, 0.5),
    (-0.5, 0.5),
)

MULTIFIELD_PARAXIAL_PARAMETER_VECTORS: tuple[tuple[float, float], ...] = (
    (40.0, 40.0),
    (50.0, 50.0),
    (40.0, 36.0),
)


DEFAULT_FORWARD_MULTI_SPHERE_CASE = SphericalForwardTraceCaseSpec(
    surfaces=(
        SphericalForwardSurfaceSpec(
            radius_mm=45.0,
            thickness_mm=8.0,
            aperture_radius_mm=18.0,
            nd=1.5168,
            vd=64.17,
            comment="S1",
        ),
        SphericalForwardSurfaceSpec(
            radius_mm=-35.0,
            thickness_mm=4.0,
            aperture_radius_mm=18.0,
            comment="S2",
        ),
        SphericalForwardSurfaceSpec(
            radius_mm=60.0,
            thickness_mm=7.0,
            aperture_radius_mm=18.0,
            nd=1.62,
            vd=45.0,
            comment="S3",
        ),
        SphericalForwardSurfaceSpec(
            radius_mm=-42.0,
            thickness_mm=12.0,
            aperture_radius_mm=18.0,
            comment="S4",
        ),
    ),
    wavelengths_um=(0.4861327, 0.5875618, 0.6562725),
)
