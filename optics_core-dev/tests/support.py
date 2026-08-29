from __future__ import annotations

import torch

import optics_core as oc


class ShowcaseSmokeTracer(oc.SequentialSurfaceRayTracer):
    """用于接口回归的轻量 smoke tracer。"""

    def trace(
        self,
        system: oc.MultiOpticalSystem,
        rays: oc.RayBundle,
        options: oc.TraceOptions | None = None,
    ) -> oc.TraceResult:
        options = options or oc.TraceOptions()
        x = torch.as_tensor(rays.x, dtype=torch.float64)
        y = torch.as_tensor(rays.y, dtype=torch.float64)
        z = torch.as_tensor(rays.z, dtype=torch.float64)
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)

        intersections = ()
        if options.record_intersections:
            intersections = tuple(
                oc.SurfaceIntersection(
                    surface_index=index,
                    position=(x.clone(), y.clone(), torch.full_like(z, float(index))),
                    normal=(zeros, zeros, ones),
                )
                for index, _surface in enumerate(system.surfaces)
            )

        ray_angles = zeros if options.record_ray_angles else None
        return oc.TraceResult(
            rays=rays,
            valid=torch.ones_like(x, dtype=torch.bool),
            intersections=intersections,
            ray_angles_in_deg=ray_angles,
            ray_angles_out_deg=ray_angles,
            cache={
                "mode": options.direction,
                "surface_count": len(system.surfaces),
                "system_count": system.system_count,
                "batch_shape": tuple(x.shape),
            },
        )


def build_showcase_system() -> oc.MultiOpticalSystem:
    architecture = oc.OpticalArchitecture(name="showcase")
    architecture.surfaces.add_object(label="OBJ")
    architecture.surfaces.add_sphere(
        radius=60.0,
        thickness=5.0,
        semi_diameter=6.0,
        label="S1",
        is_stop=True,
    )
    architecture.surfaces.add_sphere(
        radius=-80.0,
        thickness=8.0,
        semi_diameter=6.0,
        label="S2",
    )
    architecture.surfaces.add_paraxial(
        focal_length=35.0,
        thickness=10.0,
        semi_diameter=6.0,
        label="PX",
    )
    architecture.surfaces.add_coordinate_break(thickness=2.0, label="CB")
    architecture.surfaces.add_sphere(
        radius=100.0,
        thickness=0.0,
        semi_diameter=6.0,
        label="GEN",
    )
    architecture.surfaces.add_image(label="IMG")

    fields = oc.FieldSequence(field_type="angle")
    fields.add(x=0.0, y=0.0, label="on_axis")
    fields.add(x=0.0, y=12.0, label="upper")

    wavelengths = oc.WavelengthSequence()
    wavelengths.add(0.4861, label="F")
    wavelengths.add(0.5876, is_primary=True, label="d")

    return oc.MultiOpticalSystem(
        architecture=architecture,
        name="showcase",
        fields=fields,
        wavelengths=wavelengths,
        aperture=oc.SystemAperture(kind="entrance_pupil_diameter", value=12.0, stop_surface=1),
        tracer=ShowcaseSmokeTracer(),
    )
