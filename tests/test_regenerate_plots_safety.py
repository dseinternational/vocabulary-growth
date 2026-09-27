# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for what ``regenerate_plots.py`` may and may not change in a fit.

The script's contract is that a redraw changes figures and their companion
CSVs and nothing else, and only after the whole redraw has succeeded. Two
defects found on 2026-09-27, while figures were redrawn for the
dse-research-utils 0.16.0 upgrade, broke it:

* **Staging bypass.** The rebuilt context wrote into the promoted fit
  directory until the plot stage, so the prior-density figures and the model
  graph from ``priors`` and ``build`` replaced the fit's copies before staging
  applied. An aborted VG24 redraw left them replaced.
* **Cross-stack draws.** Engines without a samples extractor re-run the
  posterior predictive from the sampling seed, which reproduces the stored
  draws only on the fit's own numerical stack. VG13 came back from a newer
  numpy and PyTensor with 17 companion CSVs moved by Monte Carlo noise.

The engine here is a stub whose stages write into whatever reporting directory
they are given, so any write that escapes staging lands in the fit directory
where a test can see it. The fit directory holds a real ``trace.nc``, because
the reproduction check reads the stored draws from it lazily, one chain at a
time.
"""

import gc
import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from rich.console import Console

from vocab_growth.models import implementation_identity
from vocab_growth.models.definitions import MODEL_REGISTRY

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "regenerate_plots.py"
_SPEC = importlib.util.spec_from_file_location(
    "regenerate_plots_safety_script", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

# A real registration, for its output directory; the engine is replaced below.
MODEL_ID = "vg13"
CONFIG = "rep"

# What the promoted fit holds before a redraw.
ORIGINAL = {
    "descriptive_statistics.csv": "original statistics\n",
    "eta_dist.png": "original prior figure",
    "gp_model_graph.svg": "<svg>original graph</svg>",
    "posterior_predictive_u.csv": "original predictive table\n",
    "p_u_plot.png": "original plot",
}

# What each stub stage writes, stage by stage.
PREPARE = {"descriptive_statistics.csv": "new statistics\n"}
PRIORS = {"eta_dist.png": "new prior figure"}
BUILD = {"gp_model_graph.svg": "<svg>new graph</svg>"}
PREDICTIVE = {"posterior_predictive_calibration.csv": "new calibration\n"}
PLOTS = {"p_u_plot.png": "new plot", "posterior_predictive_u.csv": "new predictive table\n"}


def _stored_draws() -> xr.Dataset:
    rng = np.random.default_rng(20260927)
    return xr.Dataset(
        {
            "y_u_plot": (
                ("chain", "draw", "plot_id"),
                rng.integers(0, 810, size=(2, 5, 3)),
            ),
            "p_u_query_subject_marginal": (
                ("chain", "draw", "query_id"),
                rng.uniform(size=(2, 5, 4)),
            ),
        },
        coords={"chain": [0, 1], "draw": np.arange(5)},
    )


def _write(directory: str, files: dict[str, str]) -> None:
    for name, text in files.items():
        Path(directory, name).write_bytes(text.encode("utf-8"))


def _snapshot(directory: str) -> dict[str, bytes]:
    # The fit's trace is opened lazily; let it close before reading it back.
    gc.collect()
    return {name: Path(directory, name).read_bytes() for name in sorted(os.listdir(directory))}


class _StubEngine:
    """A bivariate-RE-shaped engine: no extractor, so the predictive is re-run."""

    samples_extractor = None
    plots_call = "context"

    def __init__(self, *, drift: bool = False, fail_in: str | None = None):
        self.drift = drift
        self.fail_in = fail_in
        self.stages = {
            "prepare": lambda context, definition: self._stage("prepare", context, PREPARE),
            "priors": lambda context, definition: self._stage("priors", context, PRIORS),
            "build": lambda context, definition: self._stage("build", context, BUILD),
            "posterior_predictive": self._posterior_predictive,
            "plots": lambda context: self._stage("plots", context, PLOTS),
        }

    def resolve(self, key):
        return self.stages[key]

    def _stage(self, name, context, files):
        _write(context.reporting.output_dir, files)
        if self.fail_in == name:
            raise RuntimeError(f"stub {name} failed")

    def _posterior_predictive(self, context, definition):
        # As PyMC 6 does with `extend_inferencedata=True`: the group is replaced
        # in the trace the context already holds.
        regenerated = _stored_draws()
        if self.drift:
            regenerated["y_u_plot"][1, 4, 2] += 1
        context.trace.update({"posterior_predictive": xr.DataTree(dataset=regenerated)})
        self._stage("posterior_predictive", context, PREDICTIVE)


@pytest.fixture
def fit(tmp_path, monkeypatch):
    """A promoted fit whose validation passes, and a console to read."""
    definition = MODEL_REGISTRY[MODEL_ID]
    target = _MODULE._reporting_configuration(definition, str(tmp_path)).output_dir
    os.makedirs(target)
    _write(target, ORIGINAL)
    xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"mu": (("chain", "draw"), np.zeros((2, 5)))},
                coords={"chain": [0, 1], "draw": np.arange(5)},
            ),
            "posterior_predictive": _stored_draws(),
        }
    ).to_netcdf(os.path.join(target, "trace.nc"))

    # Recorded as fitted on an older numpy, so a refusal has a cause to name.
    recorded = implementation_identity.implementation_signature()
    recorded["packages"] = {**recorded["packages"], "numpy": {"version": "2.4.6", "commit": None}}
    recorded["sha256"] = "recorded-on-another-stack"
    Path(target, "fit_manifest.json").write_text(
        json.dumps({"model": {"implementation": recorded}}), encoding="utf-8"
    )
    Path(target, "fit_state.json").write_text('{"state": "complete"}', encoding="utf-8")

    monkeypatch.setattr(_MODULE.env, "output_root", lambda: str(tmp_path))
    # Validation has its own tests; these are about what happens after it passes.
    monkeypatch.setattr(_MODULE, "require_valid_fit", lambda *args, **kwargs: None)
    monkeypatch.setattr(_MODULE, "source_data_hash", lambda data_dir: "sha256:raw")
    monkeypatch.setattr(
        _MODULE, "expected_analysis_frame_hash", lambda key, definition: "sha256:frame"
    )
    output = io.StringIO()
    monkeypatch.setattr(_MODULE, "console", Console(file=output, width=10_000))

    def use(engine):
        monkeypatch.setitem(_MODULE.ENGINES, "stub", engine)
        monkeypatch.setitem(_MODULE.ENGINE_BY_MODEL, MODEL_ID, "stub")

    return types.SimpleNamespace(
        root=str(tmp_path), target=target, before=_snapshot(target), use=use, output=output
    )


def _regenerate(dry_run=False):
    return _MODULE.regenerate(MODEL_ID, CONFIG, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["build", "posterior_predictive", "plots"])
def test_a_failed_redraw_leaves_the_fit_exactly_as_it_was(fit, stage):
    """The defect: ``priors`` and ``build`` wrote into the fit before staging.

    A failure in the build is the VG24 case. Later failures show that no
    earlier stage's writes survive either.
    """
    fit.use(_StubEngine(fail_in=stage))

    assert _regenerate() is False
    assert _snapshot(fit.target) == fit.before
    assert "[failed]" in fit.output.getvalue()
    assert not os.path.exists(os.path.join(fit.root, ".replot", MODEL_ID))


def test_a_dry_run_leaves_the_fit_exactly_as_it_was(fit):
    fit.use(_StubEngine())

    assert _regenerate(dry_run=True) is False
    assert _snapshot(fit.target) == fit.before
    assert "[dry-run]" in fit.output.getvalue()


def test_a_redraw_replaces_every_staged_figure_and_table_and_nothing_else(fit):
    fit.use(_StubEngine())

    assert _regenerate() is True
    after = _snapshot(fit.target)
    for name, text in {**PREPARE, **PRIORS, **BUILD, **PREDICTIVE, **PLOTS}.items():
        assert after[name] == text.encode("utf-8"), name
    for name in ("trace.nc", "fit_manifest.json", "fit_state.json"):
        assert after[name] == fit.before[name], name
    assert not os.path.exists(os.path.join(fit.root, ".replot", MODEL_ID))


def test_the_staging_root_is_keyed_by_model_id_not_label(fit, monkeypatch):
    """The label is already in the staged path, and ``dot`` obeys MAX_PATH."""
    seen = []
    engine = _StubEngine()
    plots = engine.stages["plots"]

    def record_then_plot(context):
        seen.append(context.reporting.output_dir)
        plots(context)

    engine.stages["plots"] = record_then_plot
    fit.use(engine)

    assert _regenerate() is True
    label = os.path.basename(fit.target)
    assert seen == [os.path.join(fit.root, ".replot", MODEL_ID, "models", label)]


# ---------------------------------------------------------------------------
# Reproducing the stored posterior-predictive draws
# ---------------------------------------------------------------------------


def test_a_re_run_that_does_not_reproduce_the_stored_draws_is_refused(fit):
    """VG13's case: one differing draw is enough to keep every CSV as it was."""
    fit.use(_StubEngine(drift=True))

    assert _regenerate() is False
    assert _snapshot(fit.target) == fit.before
    message = fit.output.getvalue()
    assert "[refused]" in message
    assert "did not reproduce the fit's stored draws (y_u_plot)" in message
    assert "numpy 2.4.6 -> " in message
    assert not os.path.exists(os.path.join(fit.root, ".replot", MODEL_ID))


def test_a_dry_run_reports_a_re_run_that_does_not_reproduce(fit):
    fit.use(_StubEngine(drift=True))

    assert _regenerate(dry_run=True) is False
    assert "[refused]" in fit.output.getvalue()
    assert "[dry-run]" not in fit.output.getvalue()


def test_an_engine_with_an_extractor_reads_the_stored_draws(fit):
    """Nothing is re-run, so there is nothing to compare and nothing to refuse."""
    engine = _StubEngine(drift=True)
    engine.samples_extractor = "extract_model_samples"
    engine.stages["samples_extractor"] = lambda trace: "samples"
    del engine.stages["posterior_predictive"]
    fit.use(engine)

    assert _regenerate() is True
    assert "[refused]" not in fit.output.getvalue()


class TestPredictiveDifferences:
    def test_identical_draws_reproduce(self):
        assert _MODULE._predictive_differences(_stored_draws(), _stored_draws()) == []

    def test_a_single_differing_draw_in_a_later_chain_is_found(self):
        regenerated = _stored_draws()
        regenerated["p_u_query_subject_marginal"][1, 4, 3] += 1e-12
        assert _MODULE._predictive_differences(_stored_draws(), regenerated) == [
            "p_u_query_subject_marginal"
        ]

    def test_missing_values_in_the_same_places_reproduce(self):
        stored = _stored_draws()
        stored["p_u_query_subject_marginal"][0, 0, 0] = np.nan
        regenerated = stored.copy(deep=True)
        assert _MODULE._predictive_differences(stored, regenerated) == []

    def test_a_variable_the_fit_never_stored_cannot_be_vouched_for(self):
        """A predictive variable added after the fit, like the by-sex draws."""
        stored = _stored_draws().drop_vars("p_u_query_subject_marginal")
        assert _MODULE._predictive_differences(stored, _stored_draws()) == [
            "p_u_query_subject_marginal (not stored)"
        ]

    def test_a_stored_variable_the_re_run_no_longer_draws_is_ignored(self):
        """Nothing downstream reads it, so it cannot move a figure or a table."""
        regenerated = _stored_draws().drop_vars("p_u_query_subject_marginal")
        assert _MODULE._predictive_differences(_stored_draws(), regenerated) == []

    def test_a_changed_shape_is_a_difference(self):
        regenerated = _stored_draws().isel(draw=slice(0, 4))
        assert _MODULE._predictive_differences(_stored_draws(), regenerated) == [
            "p_u_query_subject_marginal (shape (2, 5, 4) -> (2, 4, 4))",
            "y_u_plot (shape (2, 5, 3) -> (2, 4, 3))",
        ]

    def test_a_fit_with_no_stored_group_cannot_be_vouched_for(self):
        assert _MODULE._predictive_differences(None, _stored_draws()) == [
            "posterior_predictive (not stored in the fit's trace)"
        ]
