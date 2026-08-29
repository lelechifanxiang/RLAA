from __future__ import annotations

import codecs
import math
from pathlib import Path

import optics_core as oc
import torch

from .specs import ZemaxSequentialSurfaceSpec, ZemaxSequentialSystemSpec


class LoadedZemaxMaterial(oc.Material):
    """按 zmx 当前系统波长表直接读取的折射率材料。"""

    def __init__(
        self,
        *,
        name: str,
        wavelengths_um: tuple[float, ...],
        refractive_indices: tuple[float, ...],
    ) -> None:
        super().__init__(name=name)
        self.wavelengths_um = wavelengths_um
        self.refractive_indices = refractive_indices

    def refractive_index(self, wavelength_um: torch.Tensor) -> torch.Tensor:
        wavelength_tensor = wavelength_um.to(dtype=torch.float64)
        wavelength_table = torch.tensor(self.wavelengths_um, dtype=torch.float64, device=wavelength_tensor.device)
        refractive_index_table = torch.tensor(self.refractive_indices, dtype=torch.float64, device=wavelength_tensor.device)

        difference = torch.abs(wavelength_tensor.unsqueeze(-1) - wavelength_table)
        min_difference, nearest = torch.min(difference, dim=-1)
        if wavelength_tensor.device.type == "cpu" and torch.any(min_difference > 1e-9).item():
            value = float(wavelength_tensor[min_difference > 1e-9][0])
            raise ValueError(f"LoadedZemaxMaterial does not support wavelength {value}.")
        return refractive_index_table[nearest]


def _parse_zmx_float(value_text: str) -> float:
    upper_text = value_text.strip().upper()
    if upper_text in {"INFINITY", "+INFINITY"}:
        return math.inf
    if upper_text == "-INFINITY":
        return -math.inf
    return float(value_text)


def _read_zmx_lines(resolved_path: Path) -> list[str]:
    """按 BOM 自动识别 zmx 文本编码。"""

    raw = resolved_path.read_bytes()
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return raw.decode("utf-16").splitlines()
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig").splitlines()
    return raw.decode("utf-8", errors="ignore").splitlines()


def _split_zmx_surface_blocks(lines: list[str]) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """把 zmx 文本拆成头部和 SURF 块。"""

    header_lines: list[str] = []
    surface_blocks: list[tuple[int, list[str]]] = []
    current_surface_number: int | None = None
    current_block_lines: list[str] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        if stripped.startswith("SURF "):
            if current_surface_number is not None:
                surface_blocks.append((current_surface_number, current_block_lines))
            current_surface_number = int(stripped.split()[1])
            current_block_lines = []
            continue

        if current_surface_number is None:
            header_lines.append(stripped)
        else:
            current_block_lines.append(stripped)

    if current_surface_number is not None:
        surface_blocks.append((current_surface_number, current_block_lines))

    return header_lines, surface_blocks


def _field_type_from_ftyp(ftyp_tokens: list[str]) -> str:
    if len(ftyp_tokens) < 5:
        raise ValueError(f"FTYP line is malformed: {ftyp_tokens!r}")

    field_type_code = int(ftyp_tokens[1])
    if field_type_code != 0:
        raise NotImplementedError(f"当前仅支持 Angle 视场，收到 FTYP code {field_type_code}。")
    return "angle"


def _surface_type_from_text(surface_type_text: str) -> str:
    normalized = surface_type_text.strip().upper()
    if normalized == "STANDARD":
        return "Standard"
    if normalized == "PARAXIAL":
        return "Paraxial"
    if normalized == "COORDBRK":
        return "CoordinateBreak"
    raise NotImplementedError(f"不支持表面类型 {surface_type_text!r}")


def _glas_line_to_material_spec(line: str) -> tuple[str | None, float | None, float | None, int | None]:
    """从 GLAS 行解析材料名、nd/vd 或材料 pickup 面号。"""

    tokens = line.split()
    if len(tokens) < 6:
        raise NotImplementedError(f"当前 zmx 文本解析仅支持包含材料名与 nd/vd 的 GLAS 行，收到 {line!r}。")

    material_name = tokens[1].strip()
    solve_type = int(tokens[2])
    if solve_type == 2:
        return None, None, None, int(tokens[3])

    try:
        nd = _parse_zmx_float(tokens[4])
        vd = _parse_zmx_float(tokens[5])
    except ValueError as exc:
        raise NotImplementedError(f"当前 zmx 文本解析无法读取 GLAS 的 nd/vd，收到 {line!r}。") from exc
    return material_name or None, nd, vd, None


def _parse_zmx_header(
    header_lines: list[str],
    *,
    resolved_path: Path,
) -> tuple[tuple[float, ...], int, str, tuple[tuple[float, float], ...], str, float | None, bool]:
    """解析 zmx 头部中的波长、视场和孔径。"""

    aperture_kind: str | None = None
    aperture_value: float | None = None
    field_type: str | None = None
    field_count: int | None = None
    wavelength_count: int | None = None
    x_field_values: list[float] = []
    y_field_values: list[float] = []
    wavelength_by_index: dict[int, float] = {}
    primary_wavelength_index = 0
    afocal_image_space = False

    for line in header_lines:
        tokens = line.split()
        if not tokens:
            continue

        command = tokens[0]
        if command == "ENPD":
            aperture_kind = "EntrancePupilDiameter"
            aperture_value = _parse_zmx_float(tokens[1])
            continue
        if command == "FNUM":
            aperture_kind = "ImageSpaceFNumber"
            aperture_value = _parse_zmx_float(tokens[1])
            continue
        if command == "FLOA":
            aperture_kind = "FloatByStopSize"
            aperture_value = None
            continue
        if command == "OBNA":
            raise NotImplementedError(f"Unsupported aperture command {command!r}.")
        if command == "FTYP":
            field_type = _field_type_from_ftyp(tokens)
            field_count = int(tokens[3])
            wavelength_count = int(tokens[4])
            afocal_image_space = len(tokens) > 7 and bool(int(tokens[7]))
            continue
        if command == "XFLN":
            x_field_values = [_parse_zmx_float(token) for token in tokens[1:]]
            continue
        if command == "YFLN":
            y_field_values = [_parse_zmx_float(token) for token in tokens[1:]]
            continue
        if command == "WAVM":
            wavelength_by_index[int(tokens[1])] = _parse_zmx_float(tokens[2])
            continue
        if command == "PWAV":
            primary_wavelength_index = int(tokens[1]) - 1

    if aperture_kind is None:
        raise ValueError(f"zmx file {resolved_path} is missing a supported aperture header.")
    if field_type is None or field_count is None or wavelength_count is None:
        raise ValueError(f"zmx file {resolved_path} is missing FTYP header.")
    if len(x_field_values) < field_count or len(y_field_values) < field_count:
        raise ValueError(f"zmx file {resolved_path} field header is incomplete.")

    wavelengths = tuple(wavelength_by_index[index] for index in range(1, wavelength_count + 1))
    field_points = tuple(
        (x_field_values[field_index], y_field_values[field_index])
        for field_index in range(field_count)
    )
    return (
        wavelengths,
        primary_wavelength_index,
        field_type,
        field_points,
        aperture_kind,
        aperture_value,
        afocal_image_space,
    )


def _parse_zmx_surface_block(
    surface_number: int,
    block_lines: list[str],
) -> ZemaxSequentialSurfaceSpec:
    """解析单个 SURF 块。"""

    surface_type: str | None = None
    comment: str | None = None
    curvature = 0.0
    thickness_mm = 0.0
    semi_diameter_mm = 0.0
    semi_diameter_solve = "auto"
    aperture_type = "none"
    focal_length_mm: float | None = None
    material_name: str | None = None
    nd: float | None = None
    vd: float | None = None
    material_pickup_surface_number: int | None = None
    is_stop = False
    decenter_x_mm = 0.0
    decenter_y_mm = 0.0
    tilt_x_deg = 0.0
    tilt_y_deg = 0.0
    tilt_z_deg = 0.0
    order_flag = 0

    for line in block_lines:
        tokens = line.split()
        if not tokens:
            continue

        command = tokens[0]
        if command == "COMM":
            comment = line[len("COMM") :].strip() or None
            continue
        if command == "STOP":
            is_stop = True
            continue
        if command == "TYPE":
            surface_type = _surface_type_from_text(tokens[1])
            continue
        if command == "CURV":
            curvature = _parse_zmx_float(tokens[1])
            continue
        if command == "DISZ":
            thickness_mm = _parse_zmx_float(tokens[1])
            continue
        if command == "DIAM":
            semi_diameter_mm = _parse_zmx_float(tokens[1])
            diameter_mode = int(round(_parse_zmx_float(tokens[2]))) if len(tokens) > 2 else 0
            semi_diameter_solve = "fixed" if diameter_mode == 1 else "auto"
            continue
        if command == "FLAP":
            aperture_type = "floating"
            continue
        if command == "PARM" and len(tokens) >= 3:
            parameter_index = int(tokens[1])
            parameter_value = _parse_zmx_float(tokens[2])
            if surface_type == "Paraxial" and parameter_index == 1:
                focal_length_mm = parameter_value
            elif surface_type == "CoordinateBreak":
                if parameter_index == 1:
                    decenter_x_mm = parameter_value
                elif parameter_index == 2:
                    decenter_y_mm = parameter_value
                elif parameter_index == 3:
                    tilt_x_deg = parameter_value
                elif parameter_index == 4:
                    tilt_y_deg = parameter_value
                elif parameter_index == 5:
                    tilt_z_deg = parameter_value
                elif parameter_index == 6:
                    order_flag = int(round(parameter_value))
            continue
        if command == "GLAS":
            material_name, nd, vd, material_pickup_surface_number = _glas_line_to_material_spec(line)

    if surface_type is None:
        raise ValueError(f"surface {surface_number} is missing TYPE.")

    radius_mm = 0.0
    if math.isfinite(curvature) and abs(curvature) > 1e-12:
        radius_mm = 1.0 / curvature

    if surface_type == "Paraxial" and focal_length_mm is None:
        raise ValueError(f"Paraxial surface {surface_number} is missing PARM 1 focal length.")

    return ZemaxSequentialSurfaceSpec(
        radius_mm=radius_mm,
        thickness_mm=thickness_mm,
        semi_diameter_mm=semi_diameter_mm,
        semi_diameter_solve=semi_diameter_solve,
        aperture_type=aperture_type,
        surface_type=surface_type,
        focal_length_mm=focal_length_mm,
        material_name=material_name,
        nd=nd,
        vd=vd,
        material_pickup_surface_number=material_pickup_surface_number,
        refractive_indices=None,
        comment=comment,
        is_stop=is_stop,
        decenter_x_mm=decenter_x_mm,
        decenter_y_mm=decenter_y_mm,
        tilt_x_deg=tilt_x_deg,
        tilt_y_deg=tilt_y_deg,
        tilt_z_deg=tilt_z_deg,
        order_flag=order_flag,
    )


def _load_zmx_sequential_system_text_spec(resolved_path: Path) -> ZemaxSequentialSystemSpec:
    lines = _read_zmx_lines(resolved_path)
    header_lines, surface_blocks = _split_zmx_surface_blocks(lines)
    if len(surface_blocks) < 2:
        raise ValueError(f"zmx file {resolved_path} does not contain enough SURF blocks.")

    (
        wavelengths,
        primary_wavelength_index,
        field_type,
        field_points,
        aperture_kind,
        aperture_value,
        afocal_image_space,
    ) = _parse_zmx_header(header_lines, resolved_path=resolved_path)

    image_surface_index = surface_blocks[-1][0]
    object_distance_mm = math.inf
    surfaces: list[ZemaxSequentialSurfaceSpec] = []
    stop_surface_index: int | None = None
    for surface_number, block_lines in surface_blocks:
        if surface_number == 0:
            object_distance_mm = _parse_zmx_surface_block(surface_number, block_lines).thickness_mm
            continue
        if surface_number == image_surface_index:
            continue

        surface_spec = _parse_zmx_surface_block(surface_number, block_lines)
        surfaces.append(surface_spec)
        if surface_spec.is_stop:
            stop_surface_index = len(surfaces) - 1

    if stop_surface_index is None:
        raise ValueError(f"zmx file {resolved_path} is missing STOP surface.")
    if aperture_kind == "FloatByStopSize":
        stop_surface = surfaces[stop_surface_index]
        if stop_surface.semi_diameter_solve != "fixed" or stop_surface.semi_diameter_mm <= 0.0:
            raise ValueError(f"zmx file {resolved_path} uses FLOA but its STOP surface has no fixed semi-diameter.")
        aperture_value = 2.0 * stop_surface.semi_diameter_mm
    if aperture_value is None:
        raise ValueError(f"zmx file {resolved_path} has no aperture value.")

    return ZemaxSequentialSystemSpec(
        name=resolved_path.stem,
        zmx_path=str(resolved_path),
        surfaces=tuple(surfaces),
        wavelengths_um=wavelengths,
        primary_wavelength_index=primary_wavelength_index,
        field_type=field_type,
        field_points=field_points,
        aperture_kind=aperture_kind,
        aperture_value=aperture_value,
        object_distance_mm=object_distance_mm,
        afocal_image_space=afocal_image_space,
        stop_surface_index=stop_surface_index,
        image_surface_index=image_surface_index,
    )

def load_zmx_sequential_system_spec(
    zmx_path: str | Path,
) -> ZemaxSequentialSystemSpec:
    """从 zmx 文件读取顺序系统规格。"""

    resolved_path = Path(zmx_path).resolve()
    return _load_zmx_sequential_system_text_spec(resolved_path)


def _surface_medium_from_spec(
    spec: ZemaxSequentialSystemSpec,
    surface_index: int,
    surface_spec: ZemaxSequentialSurfaceSpec,
    *,
    material_library: oc.MaterialLibrary,
    use_real_materials: bool,
) -> oc.Material | str | None:
    if surface_spec.material_pickup_surface_number is not None:
        pickup_index = surface_spec.material_pickup_surface_number - 1
        if pickup_index < 0 or pickup_index >= len(spec.surfaces):
            raise ValueError(f"材料 pickup 面号 {surface_spec.material_pickup_surface_number} 超出规格范围。")
        return _surface_medium_from_spec(
            spec,
            pickup_index,
            spec.surfaces[pickup_index],
            material_library=material_library,
            use_real_materials=use_real_materials,
        )

    material_name = surface_spec.material_name
    has_real_material_name = (
        use_real_materials
        and material_name is not None
        and material_name in material_library
    )
    has_valid_abbe_parameters = (
        surface_spec.nd is not None
        and surface_spec.vd is not None
        and float(surface_spec.nd) > 0.0
        and float(surface_spec.vd) > 0.0
    )

    if has_real_material_name:
        return material_name
    if surface_spec.refractive_indices is not None:
        return LoadedZemaxMaterial(
            name=f"LOADED_{surface_index + 1}",
            wavelengths_um=spec.wavelengths_um,
            refractive_indices=surface_spec.refractive_indices,
        )
    if has_valid_abbe_parameters:
        return oc.AbbeModelMaterial(
            name=f"MODEL_{surface_index + 1}",
            nd=surface_spec.nd,
            vd=surface_spec.vd,
        )
    return None


def build_optics_core_system_from_zmx_spec(
    spec: ZemaxSequentialSystemSpec,
    *,
    materials: oc.MaterialLibrary | None = None,
    use_real_materials: bool = True,
) -> oc.MultiOpticalSystem:
    material_library = materials or oc.MaterialLibrary({"AIR": oc.AIR})
    if use_real_materials:
        material_library.load_builtin_real_materials()

    architecture = oc.OpticalArchitecture(
        name=spec.name,
        materials=material_library,
        object_distance_mm=spec.object_distance_mm,
        afocal_image_space=spec.afocal_image_space,
    )
    current_medium: oc.Material | str | None = None

    for surface_index, surface_spec in enumerate(spec.surfaces):
        medium = _surface_medium_from_spec(
            spec,
            surface_index,
            surface_spec,
            material_library=material_library,
            use_real_materials=use_real_materials,
        )

        if surface_spec.surface_type == "CoordinateBreak":
            architecture.surfaces.add_coordinate_break(
                thickness=surface_spec.thickness_mm,
                medium=current_medium,
                frame=oc.CoordinateFrame(
                    x=surface_spec.decenter_x_mm,
                    y=surface_spec.decenter_y_mm,
                    rx=surface_spec.tilt_x_deg,
                    ry=surface_spec.tilt_y_deg,
                    rz=surface_spec.tilt_z_deg,
                ),
                order_flag=surface_spec.order_flag,
                label=surface_spec.comment,
            )
            continue

        if surface_spec.surface_type == "Paraxial":
            if surface_spec.focal_length_mm is None:
                raise ValueError("Paraxial surface requires focal_length_mm.")
            architecture.surfaces.add_paraxial(
                focal_length=surface_spec.focal_length_mm,
                thickness=surface_spec.thickness_mm,
                medium=medium,
                semi_diameter=surface_spec.semi_diameter_mm,
                semi_diameter_solve=surface_spec.semi_diameter_solve,
                aperture_type=surface_spec.aperture_type,
                label=surface_spec.comment,
                is_stop=surface_spec.is_stop,
            )
        else:
            architecture.surfaces.add_sphere(
                radius=surface_spec.radius_mm,
                thickness=surface_spec.thickness_mm,
                medium=medium,
                semi_diameter=surface_spec.semi_diameter_mm,
                semi_diameter_solve=surface_spec.semi_diameter_solve,
                aperture_type=surface_spec.aperture_type,
                label=surface_spec.comment,
                is_stop=surface_spec.is_stop,
            )
        current_medium = medium

    architecture.surfaces.add_image(label="IMG")

    system = oc.MultiOpticalSystem(
        architecture=architecture,
        tracer=oc.SequentialSurfaceRayTracer(),
        materials=material_library,
    )
    system.fields.set_type(spec.field_type)
    for field_index, (field_x_deg, field_y_deg) in enumerate(spec.field_points):
        system.fields.add(x=field_x_deg, y=field_y_deg, label=f"field_{field_index}")
    for wavelength_index, wavelength_um in enumerate(spec.wavelengths_um):
        system.wavelengths.add(
            wavelength_um,
            is_primary=wavelength_index == spec.primary_wavelength_index,
            label=f"wave_{wavelength_index}",
        )
    aperture_kind_by_zemax = {
        "EntrancePupilDiameter": "entrance_pupil_diameter",
        "ImageSpaceFNumber": "image_f_number",
        "FloatByStopSize": "float_by_stop_size",
    }
    system.set_aperture(
        aperture_kind_by_zemax[spec.aperture_kind],
        spec.aperture_value,
        stop_surface=spec.stop_surface_index,
        label=spec.aperture_kind,
    )
    return system
