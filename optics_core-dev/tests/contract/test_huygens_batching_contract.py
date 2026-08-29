from __future__ import annotations

import pytest
import torch

import optics_core as oc
import optics_core.huygens_psf as huygens_psf
from optics_core.huygens_mtf import compute_huygens_mtf
from tests.fixtures.systems import build_multifield_multistructure_system


def _build_polychromatic_multi_design_system() -> oc.MultiOpticalSystem:
    """构造多设计、多视场、三波长测试系统。"""
    system = build_multifield_multistructure_system()
    system.wavelengths.add(0.4861, label="F")
    system.wavelengths.add(0.6563, label="C")
    return system.prepare()


def test_design_batch_view_shares_parameters_and_prepared_tensors() -> None:
    """连续 design 视图不应复制参数向量和准备态 tensor。"""
    system = _build_polychromatic_multi_design_system()
    view = system.design_batch_view(1, 3)

    assert view.system_count == 2
    assert view.parameters[0] is system.parameters[1]
    assert view.parameters[1] is system.parameters[2]
    assert view.architecture is system.architecture
    assert view.materials is system.materials
    assert view.tracer is system.tracer
    assert view.fields is system.fields
    assert view.wavelengths is system.wavelengths
    assert view.frame_data.rotations.untyped_storage().data_ptr() == (
        system.frame_data.rotations.untyped_storage().data_ptr()
    )
    assert view.first_order_data.working_f_number.untyped_storage().data_ptr() == (
        system.first_order_data.working_f_number.untyped_storage().data_ptr()
    )
    assert view.clear_aperture_data.semi_diameter.untyped_storage().data_ptr() == (
        system.clear_aperture_data.semi_diameter.untyped_storage().data_ptr()
    )


def test_huygens_psf_and_mtf_minibatches_match_full_batch(monkeypatch) -> None:
    """多视场全波长分批结果应与一次性计算一致。"""
    system = _build_polychromatic_multi_design_system()
    field_indices = (0, 1)
    frequencies = torch.tensor((0.0, 25.0), dtype=torch.float64)
    expected_psf = huygens_psf.compute_huygens_psf_batch(
        system,
        field_indices=(0,),
        wavelength_index=-1,
        pupil_sample_count=4,
        image_sample_count=5,
        image_delta_um=0.5,
    )
    expected_mtf_psf = huygens_psf.compute_huygens_psf_batch(
        system,
        field_indices=field_indices,
        wavelength_index=-1,
        pupil_sample_count=4,
        image_sample_count=5,
        image_delta_um=0.5,
    )
    expected_sagittal, expected_tangential = compute_huygens_mtf(
        expected_mtf_psf.psf,
        pixel_pitch_um=expected_mtf_psf.pixel_pitch_um,
        frequencies_lp_per_mm=frequencies,
    )
    monkeypatch.setattr(huygens_psf, "resolve_huygens_design_batch_size", lambda *args, **kwargs: 2)

    psf_result = system.analysis.psf(
        oc.PSFSettings(
            pupil_sample_count=4,
            image_sample_count=5,
            image_delta_um=0.5,
            wavelength_index=-1,
        )
    ).run()

    def fail_strehl_calculation(*args, **kwargs):
        raise AssertionError("MTF 路径不应计算 Strehl")

    monkeypatch.setattr(huygens_psf, "_strehl_from_psfs", fail_strehl_calculation)
    optimized_mtf_psf = huygens_psf.compute_huygens_psf_batch(
        system,
        field_indices=field_indices,
        wavelength_index=-1,
        pupil_sample_count=4,
        image_sample_count=5,
        image_delta_um=0.5,
        compute_ideal_psf=False,
    )
    mtf_result = system.analysis.mtf(
        oc.MTFSettings(
            pupil_sample_count=4,
            image_sample_count=5,
            image_delta_um=0.5,
            frequencies_lp_per_mm=tuple(frequencies.tolist()),
            field_indices=field_indices,
            wavelength_index=-1,
        )
    ).run()

    assert psf_result.design_batch_size == 2
    assert psf_result.minibatch_count == 2
    assert mtf_result.design_batch_size == 2
    assert mtf_result.minibatch_count == 2
    torch.testing.assert_close(torch.as_tensor(psf_result.psf), expected_psf.psf)
    torch.testing.assert_close(
        torch.as_tensor(psf_result.psf_by_wavelength),
        expected_psf.psf_by_wavelength[:, 0],
    )
    torch.testing.assert_close(optimized_mtf_psf.psf, expected_mtf_psf.psf)
    assert optimized_mtf_psf.strehl_ratio is None
    assert optimized_mtf_psf.psf_by_wavelength is None
    assert optimized_mtf_psf.strehl_by_wavelength is None
    torch.testing.assert_close(torch.as_tensor(mtf_result.sagittal), expected_sagittal)
    torch.testing.assert_close(torch.as_tensor(mtf_result.tangential), expected_tangential)


class _CountingTracer(oc.SequentialSurfaceRayTracer):
    """记录显式 PSF 光线追迹次数。"""

    def __init__(self) -> None:
        self.trace_count = 0

    def trace(
        self,
        system: oc.MultiOpticalSystem,
        rays: oc.RayBundle,
        options: oc.TraceOptions | None = None,
    ) -> oc.TraceResult:
        self.trace_count += 1
        return super().trace(system, rays, options)


def test_huygens_mtf_traces_once_per_design_minibatch(monkeypatch) -> None:
    """每个 design minibatch 应一次追迹全部视场和波长。"""
    system = build_multifield_multistructure_system()
    system.wavelengths.add(0.4861, label="F")
    system.wavelengths.add(0.6563, label="C")
    tracer = _CountingTracer()
    system.set_tracer(tracer)
    system.prepare()
    tracer.trace_count = 0
    monkeypatch.setattr(huygens_psf, "resolve_huygens_design_batch_size", lambda *args, **kwargs: 2)

    result = system.analysis.mtf(
        oc.MTFSettings(
            pupil_sample_count=4,
            image_sample_count=5,
            image_delta_um=0.5,
            frequencies_lp_per_mm=(0.0, 25.0),
            field_indices=(0, 1),
            wavelength_index=-1,
        )
    ).run()

    assert tracer.trace_count == 2
    assert result.minibatch_count == 2
    assert tuple(torch.as_tensor(result.sagittal).shape) == (system.system_count, 2, 2)


def test_huygens_minibatch_halves_after_oom(monkeypatch) -> None:
    """CUDA OOM 后应减半 batch size，单设计仍失败时给出明确错误。"""
    system = _build_polychromatic_multi_design_system()
    original_compute = huygens_psf.compute_huygens_psf_batch
    monkeypatch.setattr(huygens_psf, "resolve_huygens_design_batch_size", lambda *args, **kwargs: 4)

    def fail_large_batch(batch_system, **kwargs):
        if batch_system.system_count > 2:
            raise torch.OutOfMemoryError("模拟 CUDA out of memory")
        return original_compute(batch_system, **kwargs)

    monkeypatch.setattr(huygens_psf, "compute_huygens_psf_batch", fail_large_batch)
    batches = huygens_psf.iter_huygens_psf_design_batches(
        system,
        field_indices=(0, 1),
        wavelength_index=-1,
        pupil_sample_count=4,
        image_sample_count=5,
        image_delta_um=0.5,
    )
    completed = list(batches)

    assert batches.design_batch_size == 2
    assert [(start, stop) for start, stop, _ in completed] == [(0, 2), (2, 3)]

    monkeypatch.setattr(huygens_psf, "resolve_huygens_design_batch_size", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        huygens_psf,
        "compute_huygens_psf_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(torch.OutOfMemoryError("模拟 CUDA out of memory")),
    )
    failed_batches = huygens_psf.iter_huygens_psf_design_batches(
        system,
        field_indices=(0, 1),
        wavelength_index=-1,
        pupil_sample_count=4,
        image_sample_count=5,
        image_delta_um=0.5,
    )
    with pytest.raises(RuntimeError, match="design_batch_size=1"):
        next(failed_batches)
