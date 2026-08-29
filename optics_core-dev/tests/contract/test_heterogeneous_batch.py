from __future__ import annotations

import torch

import optics_core as oc


def _build_two_design_system() -> oc.MultiOpticalSystem:
    materials = oc.MaterialLibrary({"AIR": oc.AIR})
    materials.register(oc.AbbeModelMaterial(name="GLASS_A", nd=1.52, vd=64.0))
    materials.register(oc.AbbeModelMaterial(name="GLASS_B", nd=1.68, vd=32.0))
    architecture = oc.OpticalArchitecture(name="heterogeneous_contract", materials=materials)
    architecture.surfaces.add_sphere(radius=30.0, thickness=5.0, medium="GLASS_A", is_stop=True)
    architecture.surfaces.add_sphere(radius=-30.0, thickness=20.0, medium="AIR")
    architecture.surfaces.add_image()

    schema = oc.ParameterSchema(
        [
            oc.ParameterSpec("front_radius", "surface[0].geometry.radius", 30.0),
            oc.ParameterSpec("center_thickness", "surface[0].gap.thickness", 5.0),
            oc.ParameterSpec("glass", "surface[0].gap.medium", "GLASS_A"),
            oc.ParameterSpec("rear_radius", "surface[1].geometry.radius", -30.0),
            oc.ParameterSpec("image_distance", "surface[1].gap.thickness", 20.0),
        ]
    )
    parameters = oc.ParameterVectorBatch(
        schema=schema,
        vectors=[
            [30.0, 5.0, "GLASS_A", -30.0, 20.0],
            [36.0, 6.0, "GLASS_B", -42.0, 24.0],
        ],
    )
    system = oc.MultiOpticalSystem(
        architecture,
        parameter_schema=schema,
        parameters=parameters,
        tracer=oc.SequentialSurfaceRayTracer(),
        materials=materials,
    )
    system.fields.add(x=0.0, y=0.0)
    system.wavelengths.add(0.55, is_primary=True)
    system.set_aperture("entrance_pupil_diameter", 4.0)
    return system


def _pupil_inputs(design_count: int) -> dict[str, torch.Tensor]:
    return {
        "field_angles": torch.tensor(
            [[[0.0, 0.0], [0.0, 3.0]], [[0.0, 0.0], [0.0, 8.0]]],
            dtype=torch.float64,
        )[:design_count],
        "entrance_pupil_z": torch.zeros(design_count, dtype=torch.float64),
        "entrance_pupil_radius": torch.tensor([2.0, 1.5], dtype=torch.float64)[:design_count],
        "pupil_coordinates": torch.tensor(
            [[0.0, 0.0], [0.5, 0.0], [-0.25, 0.4]],
            dtype=torch.float64,
        ),
        "wavelength_indices": torch.tensor([0], dtype=torch.int64),
    }


def _design_inputs(inputs: dict[str, torch.Tensor], design_index: int) -> dict[str, torch.Tensor]:
    sliced = dict(inputs)
    for name in ("field_angles", "entrance_pupil_z", "entrance_pupil_radius"):
        sliced[name] = inputs[name][design_index : design_index + 1]
    return sliced


def test_heterogeneous_batch_matches_design_views() -> None:
    system = _build_two_design_system().prepare()
    inputs = _pupil_inputs(system.system_count)
    rays = oc.build_pupil_rays(system, **inputs)
    batch = system.tracer.trace(system, rays, options=oc.TraceOptions(record_intersections=False))

    for design_index in range(system.system_count):
        design = system.design_view(design_index)
        design_rays = oc.build_pupil_rays(design, **_design_inputs(inputs, design_index))
        actual = design.tracer.trace(design, design_rays, options=oc.TraceOptions(record_intersections=False))

        for name in ("x", "y", "z", "l", "m", "n", "opl"):
            torch.testing.assert_close(
                getattr(batch.rays, name)[design_index : design_index + 1],
                getattr(actual.rays, name),
            )
        assert torch.equal(batch.valid[design_index : design_index + 1], actual.valid)


def test_material_data_and_design_batch_view_keep_design_axis() -> None:
    system = _build_two_design_system().prepare()
    material_data = system._material_data

    assert material_data is not None
    assert material_data.material_index.shape == (2, len(system.surfaces))
    assert material_data.material_index.dtype == torch.int64
    assert material_data.material_index.device.type == "cpu"
    assert material_data.material_index[0, 0] != material_data.material_index[1, 0]

    view = system.design_batch_view(1, 2)
    assert view._material_data is not None
    assert torch.equal(view._material_data.material_index, material_data.material_index[1:2])
    assert view._material_data.refractive_index_table.data_ptr() == material_data.refractive_index_table.data_ptr()


def test_build_pupil_rays_matches_stacked_designs() -> None:
    system = _build_two_design_system()
    inputs = _pupil_inputs(system.system_count)
    batch = oc.build_pupil_rays(system, **inputs)

    designs: list[oc.RayBundle] = []
    for design_index in range(system.system_count):
        design = system.design_view(design_index)
        designs.append(oc.build_pupil_rays(design, **_design_inputs(inputs, design_index)))

    for name in ("x", "y", "z", "l", "m", "n", "wavelength_index", "intensity", "opl"):
        expected = torch.cat([getattr(rays, name) for rays in designs], dim=0)
        torch.testing.assert_close(getattr(batch, name), expected)
