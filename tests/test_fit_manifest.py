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

from vocab_growth.fit_artifacts import git_metadata
from vocab_growth.models.common import (
    ModelFitContext,
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


def _git(repository: Path, *arguments: str) -> str:
    """Run one Git command in ``repository`` with no ambient identity."""
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    """A one-commit repository, so a commit and a branch both exist."""
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main", "--quiet")
    (repository / "tracked.txt").write_text("one", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "--quiet", "-m", "initial")
    return repository


def test_git_metadata_records_a_clean_checkout(tmp_path):
    """The four recorded keys, against a real repository rather than a stub.

    ``validate_fit_output`` turns ``dirty is not False`` into a refusal to
    resume or publish, so "clean" has to be recorded as exactly ``False`` --
    not ``None``, which is what an unavailable query records.
    """
    repository = _repository(tmp_path)

    assert git_metadata(str(repository)) == {
        "commit": _git(repository, "rev-parse", "HEAD"),
        "branch": "main",
        "detached": False,
        "dirty": False,
    }


def test_git_metadata_records_detached_head_as_null(tmp_path):
    repository = _repository(tmp_path)
    commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "checkout", "--quiet", "--detach", commit)

    assert git_metadata(str(repository)) == {
        "commit": commit,
        "branch": None,
        "detached": True,
        "dirty": False,
    }


def test_git_metadata_counts_an_untracked_file_as_dirty(tmp_path):
    """Untracked entries are dirt, and ignored ones are not.

    Both halves matter to ``require_clean_fit``: a fit whose output directory
    happens to sit inside the checkout must not be recorded as dirty for that
    reason, while a genuinely uncommitted source file must be.
    """
    repository = _repository(tmp_path)
    (repository / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    _git(repository, "add", ".gitignore")
    _git(repository, "commit", "--quiet", "-m", "ignore")
    (repository / "ignored").mkdir()
    (repository / "ignored" / "trace.nc").write_bytes(b"artefact")
    assert git_metadata(str(repository))["dirty"] is False

    (repository / "untracked.py").write_text("x = 1\n", encoding="utf-8")
    assert git_metadata(str(repository))["dirty"] is True


def test_git_metadata_distinguishes_unavailable_git(tmp_path):
    """A directory that is not a working tree records nothing, not a clean one.

    ``dirty`` must stay ``None`` here. ``False`` would let an unverifiable
    checkout satisfy the publication provenance check.
    """
    outside = tmp_path / "not-a-repo"
    outside.mkdir()

    assert git_metadata(str(outside)) == {
        "commit": None,
        "branch": None,
        "detached": None,
        "dirty": None,
    }
    assert git_metadata(str(tmp_path / "absent")) == {
        "commit": None,
        "branch": None,
        "detached": None,
        "dirty": None,
    }
