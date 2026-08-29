from __future__ import annotations

import csv
from pathlib import Path

import torch

from examples.batch_mtf import run_batch_mtf_analysis
from scripts.batch_analysis_scaling import ANALYSIS_SPECS


def test_batch_mtf_analysis_exports_frequency_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "batch_mtf.csv"
    summary = run_batch_mtf_analysis(
        device=torch.device("cpu"),
        design_count=1,
        random_seed=0,
        pupil_sample_count=4,
        image_sample_count=4,
        frequencies_lp_per_mm=(50.0, 100.0),
        csv_path=csv_path,
    )

    assert summary["analysis_type"] == "mtf"
    assert summary["sample_count"] == 4
    assert summary["case_name"] == "sample_count"
    assert summary["case_value"] == 4
    assert summary["scan_type"] == "assembly_tolerance_monte_carlo"
    assert summary["random_seed"] == 0
    assert summary["design_count"] == 1
    assert summary["csv_path"] == str(csv_path)
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert len(rows) == 1
    assert "cb_decenter_x_mm" in reader.fieldnames
    assert "cb_decenter_y_mm" in reader.fieldnames
    assert "cb_tilt_x_deg" in reader.fieldnames
    assert "cb_tilt_y_deg" in reader.fieldnames
    assert "field_0_freq_50_sagittal" in reader.fieldnames
    assert "field_0_freq_50_tangential" in reader.fieldnames
    assert "field_2_freq_100_sagittal" in reader.fieldnames
    assert "field_2_freq_100_tangential" in reader.fieldnames


def test_batch_analysis_scaling_uses_mtf_sample_count_command() -> None:
    spec = ANALYSIS_SPECS["mtf"]
    summary_path = Path("dummy_summary.json")
    command = spec.build_command("cpu", [1, 2], 16, summary_path)

    assert Path(command[1]).resolve() == spec.script_path.resolve()
    assert "--surfaces" not in command
    assert "--pupil-sample-count" in command
    assert "--image-sample-count" in command
    assert command[command.index("--pupil-sample-count") + 1] == "16"
    assert command[command.index("--image-sample-count") + 1] == "16"
    assert command[command.index("--summary-json") + 1] == str(summary_path)
