from __future__ import annotations

import atexit
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any


_standalone_lock = RLock()
_standalone_zos: Any | None = None
_standalone_oss: Any | None = None
_cleanup_registered = False


def _require_zospy():
    try:
        import zospy as zp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("需要先安装 zospy，才能读取 Zemax zmx 文件。") from exc
    return zp


def _shutdown_standalone_connection() -> None:
    global _standalone_zos, _standalone_oss

    if _standalone_zos is None:
        return

    try:
        _standalone_zos.disconnect()
    except Exception:
        pass
    finally:
        _standalone_zos = None
        _standalone_oss = None


def _get_standalone_system() -> Any:
    global _standalone_zos, _standalone_oss, _cleanup_registered

    if _standalone_zos is None or _standalone_oss is None:
        zp = _require_zospy()
        _standalone_zos = zp.ZOS()
        _standalone_oss = _standalone_zos.connect(mode="standalone")
        if not _cleanup_registered:
            atexit.register(_shutdown_standalone_connection)
            _cleanup_registered = True

    return _standalone_oss


@contextmanager
def loaded_sequential_system(zmx_path: str | Path) -> Iterator[Any]:
    """加载给定 zmx 文件，并复用当前 standalone Zemax 连接。"""

    with _standalone_lock:
        oss = _get_standalone_system()
        oss.load(str(Path(zmx_path).resolve()))
        yield oss


def normalized_field_coordinate(value_deg: float, edge_deg: float) -> float:
    """将角度视场换算为 Zemax Hx/Hy 归一化坐标。"""

    if edge_deg == 0.0:
        return 0.0
    return float(value_deg) / float(edge_deg)


def surface_row(ray_data: Any, surface_index: int) -> Any:
    matches = ray_data[ray_data["Surf"] == surface_index]
    if matches.empty:
        raise ValueError(f"Surface {surface_index} was not present in ray trace data.")
    return matches.iloc[0]


def get_merit_operand_value(
    oss: Any,
    operand_type: Any,
    *,
    surface_index: int,
    wavelength_index: int,
    field_hx: float = 0.0,
    field_hy: float = 0.0,
    pupil_x: float = 0.0,
    pupil_y: float = 0.0,
    extra_x: float = 0.0,
    extra_y: float = 0.0,
) -> float:
    """通过 Zemax Merit Function operand 直接获取单个标量结果。"""

    return float(
        oss.MFE.GetOperandValue(
            operand_type,
            surface_index,
            wavelength_index,
            field_hx,
            field_hy,
            pupil_x,
            pupil_y,
            extra_x,
            extra_y,
        )
    )


def get_surface_indices(oss: Any, surface_index: int, *, wavelength_count: int) -> list[float]:
    """通过 ILensDataEditor.GetIndex 读取指定面的多波长折射率。"""

    if wavelength_count <= 0:
        raise ValueError("wavelength_count must be greater than zero.")

    try:
        system = __import__("System")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 pythonnet 的 System 模块，无法读取 Zemax 折射率表。") from exc

    index_buffer = system.Array[system.Double]([0.0] * wavelength_count)
    resolved_count = int(oss.LDE.GetIndex(surface_index, wavelength_count, index_buffer))
    if resolved_count != wavelength_count:
        raise ValueError("GetIndex did not return the requested number of wavelengths.")
    return [float(index_buffer[index]) for index in range(resolved_count)]
