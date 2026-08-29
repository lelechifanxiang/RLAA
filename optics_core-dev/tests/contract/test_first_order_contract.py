from __future__ import annotations

import math

import torch

import optics_core as oc
from tests.fixtures.recipes import build_paraxial_architecture
from tests.fixtures.systems import build_tracing_system


def test_first_order_multi_design_shape_validity_and_slice() -> None:
    """集中验证一阶量的 batch shape、无效语义和切片。"""
    architecture = build_paraxial_architecture(
        focal_length=40.0,
        thickness=40.0,
        semi_diameter=6.0,
        name="first_order_contract",
        is_stop=True,
    )
    schema = oc.ParameterSchema(
        [
            oc.ParameterSpec("focal_length", "surface[0].geometry.focal_length", 40.0),
            oc.ParameterSpec("image_distance", "surface[0].gap.thickness", 40.0),
        ]
    )
    parameters = oc.ParameterVectorBatch(
        schema=schema,
        vectors=((40.0, 40.0), (50.0, 45.0), (math.inf, 40.0)),
    )
    system = build_tracing_system(
        architecture,
        parameter_schema=schema,
        parameters=parameters,
        field_points=((0.0, 0.0),),
        wavelengths_um=(0.5876,),
        aperture_diameter=12.0,
        stop_surface=0,
    ).prepare()

    result = system.analysis.first_order().run()
    names = ("effl", "working_f_number", "ttl", "image_plane_distance", "bfl")
    for name in names:
        value = getattr(result, name)
        assert isinstance(value, torch.Tensor)
        assert value.dtype == torch.float64
        assert value.shape == (3,)
        assert torch.isfinite(value[:2]).all()
        assert torch.isnan(value[2])

    assert result.valid.dtype == torch.bool
    assert result.valid.tolist() == [True, True, False]

    sliced = system.first_order_data.design_slice(1, 3)
    assert sliced.valid.tolist() == [True, False]
    torch.testing.assert_close(sliced.bfl, result.bfl[1:3], equal_nan=True)
