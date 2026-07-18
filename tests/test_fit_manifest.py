# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for per-fit provenance manifests."""

import json
import subprocess
from pathlib import Path

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd

from vocab_growth.models.common import (
    ModelFitContext,
    _git_metadata,
    write_fit_manifest,
)
from vocab_growth.models.definitions import VG01


def test_write_fit_manifest_records_data_code_and_sampling(tmp_path):
    reporting_config = reporting.ReportingConfiguration(
        model_name="VG01",
        config_name="test-manifest",
        output_root_dir=str(tmp_path),
        ci_prob=0.90,
        interval_kind="hdi",
    )
    output_dir = Path(reporting_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = ModelFitContext(
        reporting=reporting_config,
        sampling=sampling.get_sampling_configuration("dev"),
        sampling_config_name="dev",
    )
    analysis_df = pd.DataFrame(
        {
            "study": ["A", "A", "B"],
            "age": [12.0, 18.0, 24.0],
            "spoken": [1, 5, 20],
        }
    )
    context.set_model_data(
        model_data.BinomialModelData(
            X_obs=analysis_df[["age"]].to_numpy(),
            y_obs=np.array([1, 5, 20]),
            n_trials=810,
        ),
        analysis_df,
    )

    write_fit_manifest(context, VG01)

    manifest = json.loads(
        (output_dir / "fit_manifest.json").read_text()
    )
    assert manifest["model"]["model_id"] == "VG01"
    assert manifest["sampling"]["configuration_name"] == "dev"
    assert manifest["data"]["rows"] == 3
    assert manifest["data"]["source_row_counts"] == {"A": 2, "B": 1}
    assert manifest["data"]["analysis_frame_hash"].startswith("sha256:")
    assert "commit" in manifest["code"]
    assert "pymc" in {name.lower() for name in manifest["runtime"]["packages"]}
    assert isinstance(manifest["runtime"]["direct_package_origins"], dict)


def test_git_metadata_records_detached_head_as_null(monkeypatch):
    outputs = {
        ("rev-parse", "HEAD"): "abc123\n",
        ("branch", "--show-current"): "\n",
        ("status", "--porcelain", "--untracked-files=normal"): "",
    }

    def fake_run(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=outputs[tuple(command[1:])],
            stderr="",
        )

    monkeypatch.setattr("vocab_growth.models.common.subprocess.run", fake_run)

    metadata = _git_metadata()

    assert metadata == {
        "commit": "abc123",
        "branch": None,
        "detached": True,
        "dirty": False,
    }


def test_git_metadata_distinguishes_unavailable_git(monkeypatch):
    def fail_run(command, **kwargs):
        del command, kwargs
        raise OSError("git unavailable")

    monkeypatch.setattr("vocab_growth.models.common.subprocess.run", fail_run)

    metadata = _git_metadata()

    assert metadata == {
        "commit": None,
        "branch": None,
        "detached": None,
        "dirty": None,
    }
