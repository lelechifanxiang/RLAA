from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optics_core as oc
from zemax_utils import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


DOUBLE_GAUSS_CB_ZMX_PATH = Path("tests/zemax/zmx_files/Double Gauss 28 degree field with CB.ZMX")
FIRST_CB_SURFACE_NUMBER = 3
RETURN_CB_SURFACE_NUMBER = 7
DEFAULT_COORDINATE_BREAK_PAIRS = ((FIRST_CB_SURFACE_NUMBER, RETURN_CB_SURFACE_NUMBER),)
DEFAULT_TOLERANCE_SAMPLE_COUNT = 3
DEFAULT_MONTE_CARLO_DESIGN_COUNT = DEFAULT_TOLERANCE_SAMPLE_COUNT**4
DEFAULT_RANDOM_SEED = 0
DECENTER_RANGE_MM = (-0.01, 0.01)
TILT_RANGE_DEG = (-0.01, 0.01)
TOLERANCE_PARAMETER_FIELDNAMES = (
    "cb_decenter_x_mm",
    "cb_decenter_y_mm",
    "cb_tilt_x_deg",
    "cb_tilt_y_deg",
)


def tolerance_parameter_fieldnames(pair_count: int) -> tuple[str, ...]:
    """返回一组或多组独立 CB 公差记录字段。"""
    names: list[str] = []
    for pair_index in range(int(pair_count)):
        prefix = "cb" if pair_index == 0 else f"cb{pair_index + 1}"
        names.extend(
            (
                f"{prefix}_decenter_x_mm",
                f"{prefix}_decenter_y_mm",
                f"{prefix}_tilt_x_deg",
                f"{prefix}_tilt_y_deg",
            )
        )
    return tuple(names)


def parse_coordinate_break_pairs(surface_numbers: list[int]) -> tuple[tuple[int, int], ...]:
    """把连续 CB 面号解析为 first/return 对。"""
    if len(surface_numbers) == 0 or len(surface_numbers) % 2 != 0:
        raise ValueError("coordinate_break_pairs must contain first/return surface-number pairs.")
    return tuple(
        (int(surface_numbers[index]), int(surface_numbers[index + 1]))
        for index in range(0, len(surface_numbers), 2)
    )


def tolerance_axis_values(start: float, stop: float, sample_count: int) -> list[float]:
    """生成装配公差扫描轴；单点扫描默认取中心值。"""
    count = int(sample_count)
    if count <= 0:
        raise ValueError("tolerance_sample_count must be positive.")
    if count == 1:
        return [0.5 * (float(start) + float(stop))]
    return [float(value) for value in torch.linspace(float(start), float(stop), count, dtype=torch.float64).tolist()]


def build_assembly_tolerance_design_records(sample_count: int) -> list[dict[str, float]]:
    """构造第一个坐标间断面的 x/y 偏心和 x/y 倾斜 4D 参数扫描。"""
    decenter_values = tolerance_axis_values(*DECENTER_RANGE_MM, sample_count)
    tilt_values = tolerance_axis_values(*TILT_RANGE_DEG, sample_count)

    design_records: list[dict[str, float]] = []
    parameter_axes = (decenter_values, decenter_values, tilt_values, tilt_values)
    for design_index, (decenter_x, decenter_y, tilt_x, tilt_y) in enumerate(itertools.product(*parameter_axes)):
        design_records.append(
            {
                "design_index": design_index,
                "cb_decenter_x_mm": float(decenter_x),
                "cb_decenter_y_mm": float(decenter_y),
                "cb_tilt_x_deg": float(tilt_x),
                "cb_tilt_y_deg": float(tilt_y),
            }
        )
    return design_records


def build_random_assembly_tolerance_design_records(
    design_count: int,
    random_seed: int,
    coordinate_break_pair_count: int = 1,
) -> list[dict[str, float]]:
    """构造一组或多组独立坐标间断面的 Monte Carlo 参数。"""
    count = int(design_count)
    if count <= 0:
        raise ValueError("design_count must be positive.")
    pair_count = int(coordinate_break_pair_count)
    if pair_count <= 0:
        raise ValueError("coordinate_break_pair_count must be positive.")

    generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
    decenters = torch.empty((count, pair_count, 2), dtype=torch.float64).uniform_(
        float(DECENTER_RANGE_MM[0]),
        float(DECENTER_RANGE_MM[1]),
        generator=generator,
    )
    tilts = torch.empty((count, pair_count, 2), dtype=torch.float64).uniform_(
        float(TILT_RANGE_DEG[0]),
        float(TILT_RANGE_DEG[1]),
        generator=generator,
    )

    parameter_fieldnames = tolerance_parameter_fieldnames(pair_count)
    design_records: list[dict[str, float]] = []
    for design_index in range(count):
        record: dict[str, float] = {"design_index": design_index}
        for pair_index in range(pair_count):
            fieldnames = parameter_fieldnames[pair_index * 4 : pair_index * 4 + 4]
            values = (*decenters[design_index, pair_index].tolist(), *tilts[design_index, pair_index].tolist())
            record.update({name: float(value) for name, value in zip(fieldnames, values, strict=True)})
        design_records.append(record)
    return design_records


def coordinate_break_surface_indices(
    zmx_path: Path,
    coordinate_break_pairs: tuple[tuple[int, int], ...] = DEFAULT_COORDINATE_BREAK_PAIRS,
) -> tuple[tuple[int, int], ...]:
    """校验 CB 面号并返回 OpticsCore 内部索引对。"""
    spec = load_zmx_sequential_system_spec(zmx_path)
    index_pairs: list[tuple[int, int]] = []
    for first_surface_number, return_surface_number in coordinate_break_pairs:
        first_index = int(first_surface_number) - 1
        return_index = int(return_surface_number) - 1
        for surface_number, surface_index in (
            (first_surface_number, first_index),
            (return_surface_number, return_index),
        ):
            if surface_index < 0 or surface_index >= len(spec.surfaces):
                raise ValueError(f"Zemax 面号 {surface_number} 超出当前规格范围。")
            if spec.surfaces[surface_index].surface_type != "CoordinateBreak":
                raise ValueError(f"Zemax 面号 {surface_number} 不是 Coordinate Break。")
        index_pairs.append((first_index, return_index))
    return tuple(index_pairs)


def build_assembly_tolerance_multi_system_from_design_records(
    zmx_path: Path,
    *,
    device: torch.device,
    design_records: list[dict[str, float]],
    grid_shape: tuple[int, ...] | None = None,
    scan_metadata: dict[str, object] | None = None,
    coordinate_break_pairs: tuple[tuple[int, int], ...] = DEFAULT_COORDINATE_BREAK_PAIRS,
) -> tuple[oc.MultiOpticalSystem, list[dict[str, float]], tuple[tuple[float, float], ...], list[int]]:
    """根据装配公差记录构造一组或多组独立 CB 公差多系统。"""
    spec = load_zmx_sequential_system_spec(zmx_path)
    base_system = build_optics_core_system_from_zmx_spec(spec)
    cb_index_pairs = coordinate_break_surface_indices(zmx_path, coordinate_break_pairs)

    schema = oc.ParameterSchema()
    for pair_index, (first_cb_index, return_cb_index) in enumerate(cb_index_pairs, start=1):
        for role, surface_index in (("first", first_cb_index), ("return", return_cb_index)):
            prefix = f"cb_pair_{pair_index}_{role}"
            surface = base_system.surfaces[surface_index]
            schema.add(f"{prefix}_decenter_x_mm", f"surface[{surface_index}].frame.x", default=surface.frame.x)
            schema.add(f"{prefix}_decenter_y_mm", f"surface[{surface_index}].frame.y", default=surface.frame.y)
            schema.add(f"{prefix}_tilt_x_deg", f"surface[{surface_index}].frame.rx", default=surface.frame.rx)
            schema.add(f"{prefix}_tilt_y_deg", f"surface[{surface_index}].frame.ry", default=surface.frame.ry)

    vectors: list[list[float]] = []
    for record in design_records:
        values: dict[str, float] = {}
        all_fieldnames = tolerance_parameter_fieldnames(len(cb_index_pairs))
        for pair_index in range(len(cb_index_pairs)):
            decenter_x, decenter_y, tilt_x, tilt_y = (
                float(record[name]) for name in all_fieldnames[pair_index * 4 : pair_index * 4 + 4]
            )
            prefix = f"cb_pair_{pair_index + 1}"
            values.update(
                {
                    f"{prefix}_first_decenter_x_mm": decenter_x,
                    f"{prefix}_first_decenter_y_mm": decenter_y,
                    f"{prefix}_first_tilt_x_deg": tilt_x,
                    f"{prefix}_first_tilt_y_deg": tilt_y,
                    f"{prefix}_return_decenter_x_mm": -decenter_x,
                    f"{prefix}_return_decenter_y_mm": -decenter_y,
                    f"{prefix}_return_tilt_x_deg": -tilt_x,
                    f"{prefix}_return_tilt_y_deg": -tilt_y,
                }
            )
        vectors.append(schema.vector_from_mapping(values))

    parameters = oc.ParameterVectorBatch(
        schema=schema,
        vectors=vectors,
        grid_shape=grid_shape,
        metadata={
            "parameter_names": list(tolerance_parameter_fieldnames(len(cb_index_pairs))),
            "coordinate_break_pairs": [list(pair) for pair in coordinate_break_pairs],
            **dict(scan_metadata or {}),
        },
    )
    system = oc.MultiOpticalSystem(
        architecture=base_system.architecture,
        name=base_system.name,
        parameter_schema=schema,
        parameters=parameters,
        config=copy.deepcopy(base_system.config),
        tracer=base_system.tracer,
        materials=base_system.materials,
        fields=copy.deepcopy(list(base_system.fields)),
        wavelengths=copy.deepcopy(list(base_system.wavelengths)),
        aperture=copy.deepcopy(base_system.aperture),
    )
    system.config.backend.device = str(device)

    field_points = tuple((float(field.x), float(field.y)) for field in system.fields)
    print(f"zmx 文件: {zmx_path}")
    print(f"设备: {device}")
    internal_surface_indices = [index for pair in cb_index_pairs for index in pair]
    print(f"坐标间断面对: {coordinate_break_pairs}")
    print(f"内部 surface 索引: {internal_surface_indices}")
    print(f"视场点: {field_points}")
    print(f"波长(um): {[float(wavelength.value_um) for wavelength in system.wavelengths]}")
    print(f"总设计数: {system.system_count}")
    if scan_metadata is not None and "random_seed" in scan_metadata:
        print(f"Monte Carlo 随机种子: {scan_metadata['random_seed']}")
    if scan_metadata is not None and "tolerance_sample_count" in scan_metadata:
        print(f"装配公差网格采样点数: {scan_metadata['tolerance_sample_count']}")
    print(f"偏心范围(mm): {DECENTER_RANGE_MM}")
    print(f"倾斜范围(deg): {TILT_RANGE_DEG}")
    return system, design_records, field_points, internal_surface_indices


def build_assembly_tolerance_multi_system(
    zmx_path: Path,
    *,
    device: torch.device,
    tolerance_sample_count: int,
) -> tuple[oc.MultiOpticalSystem, list[dict[str, float]], tuple[tuple[float, float], ...], list[int]]:
    """构造第 2-3 胶合镜片装配公差 4D 扫描多系统。"""
    design_records = build_assembly_tolerance_design_records(tolerance_sample_count)
    return build_assembly_tolerance_multi_system_from_design_records(
        zmx_path,
        device=device,
        design_records=design_records,
        grid_shape=(int(tolerance_sample_count),) * 4,
        scan_metadata={
            "scan_type": "assembly_tolerance_coordinate_break_grid",
            "tolerance_sample_count": int(tolerance_sample_count),
            "axes": list(TOLERANCE_PARAMETER_FIELDNAMES),
        },
    )


def build_random_assembly_tolerance_multi_system(
    zmx_path: Path,
    *,
    device: torch.device,
    design_count: int,
    random_seed: int,
    coordinate_break_pairs: tuple[tuple[int, int], ...] = DEFAULT_COORDINATE_BREAK_PAIRS,
) -> tuple[oc.MultiOpticalSystem, list[dict[str, float]], tuple[tuple[float, float], ...], list[int]]:
    """构造一组或多组独立 CB 装配公差 Monte Carlo 多系统。"""
    design_records = build_random_assembly_tolerance_design_records(
        design_count=int(design_count),
        random_seed=int(random_seed),
        coordinate_break_pair_count=len(coordinate_break_pairs),
    )
    return build_assembly_tolerance_multi_system_from_design_records(
        zmx_path,
        device=device,
        design_records=design_records,
        scan_metadata={
            "scan_type": "assembly_tolerance_monte_carlo",
            "design_count": int(design_count),
            "random_seed": int(random_seed),
        },
        coordinate_break_pairs=coordinate_break_pairs,
    )


def _surface_column(oss, parameter_number: int):
    """获取 Zemax LDE 参数列常量。"""
    column_name = f"Par{int(parameter_number)}"
    zosapi_candidates = (
        getattr(oss, "ZOSAPI", None),
        getattr(getattr(oss, "ZOS", None), "ZOSAPI", None),
        getattr(getattr(oss, "_System", None), "ZOSAPI", None),
    )
    for zosapi in zosapi_candidates:
        editors = getattr(zosapi, "Editors", None)
        lde_constants = getattr(editors, "LDE", None)
        surface_columns = getattr(lde_constants, "SurfaceColumn", None)
        if surface_columns is not None and hasattr(surface_columns, column_name):
            return getattr(surface_columns, column_name)

    try:
        import zospy as zp

        surface_columns = zp.api.constants.Editors.LDE.SurfaceColumn
        return getattr(surface_columns, column_name)
    except Exception as exc:
        raise RuntimeError(f"无法获取 Zemax SurfaceColumn.{column_name} 常量。") from exc


def _set_coordinate_break_parameter(oss, surface_number: int, parameter_number: int, value: float) -> None:
    """设置 Zemax 坐标间断面参数。"""
    surface = oss.LDE.GetSurfaceAt(int(surface_number))
    column = _surface_column(oss, parameter_number)
    if hasattr(surface, "GetSurfaceCell"):
        cell = surface.GetSurfaceCell(column)
    elif hasattr(surface, "GetCellAt"):
        cell = surface.GetCellAt(column)
    else:
        raise RuntimeError("当前 Zemax API surface 对象不支持 GetSurfaceCell/GetCellAt。")
    cell.DoubleValue = float(value)


def apply_assembly_tolerance_to_zemax(
    oss,
    design_record: dict[str, float],
    coordinate_break_pairs: tuple[tuple[int, int], ...] = DEFAULT_COORDINATE_BREAK_PAIRS,
) -> None:
    """把一组或多组装配公差设计写入 Zemax。

    Zemax 文件中返回坐标间断面已经设置 pickup=-1，因此这里只写第一个坐标间断面。
    OpticsCore 多系统构造中会显式写入返回面为相反数。
    """
    parameter_fieldnames = tolerance_parameter_fieldnames(len(coordinate_break_pairs))
    for pair_index, (first_surface_number, _return_surface_number) in enumerate(coordinate_break_pairs):
        values = (
            float(design_record[name])
            for name in parameter_fieldnames[pair_index * 4 : pair_index * 4 + 4]
        )
        for parameter_number, value in enumerate(values, start=1):
            _set_coordinate_break_parameter(oss, first_surface_number, parameter_number, value)


def tolerance_summary_fields(tolerance_sample_count: int) -> dict[str, object]:
    """返回装配公差扫描摘要字段。"""
    return {
        "zmx_path": str(REPO_ROOT / DOUBLE_GAUSS_CB_ZMX_PATH),
        "scan_type": "assembly_tolerance_coordinate_break_grid",
        "tolerance_sample_count": int(tolerance_sample_count),
        "parameter_names": list(TOLERANCE_PARAMETER_FIELDNAMES),
        "decenter_range_mm": list(DECENTER_RANGE_MM),
        "tilt_range_deg": list(TILT_RANGE_DEG),
        "first_cb_surface_number": FIRST_CB_SURFACE_NUMBER,
        "return_cb_surface_number": RETURN_CB_SURFACE_NUMBER,
    }


def monte_carlo_tolerance_summary_fields(
    design_count: int,
    random_seed: int,
    *,
    zmx_path: Path = REPO_ROOT / DOUBLE_GAUSS_CB_ZMX_PATH,
    coordinate_break_pairs: tuple[tuple[int, int], ...] = DEFAULT_COORDINATE_BREAK_PAIRS,
) -> dict[str, object]:
    """返回装配公差 Monte Carlo 摘要字段。"""
    return {
        "zmx_path": str(zmx_path),
        "scan_type": "assembly_tolerance_monte_carlo",
        "design_count": int(design_count),
        "random_seed": int(random_seed),
        "parameter_names": list(tolerance_parameter_fieldnames(len(coordinate_break_pairs))),
        "decenter_range_mm": list(DECENTER_RANGE_MM),
        "tilt_range_deg": list(TILT_RANGE_DEG),
        "coordinate_break_pairs": [list(pair) for pair in coordinate_break_pairs],
    }
