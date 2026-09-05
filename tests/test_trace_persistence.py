# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import json

import numpy as np
import pytest
import xarray as xr

from vocab_growth.fit_artifacts import (
    TRACE_PERSISTENCE_ENV_VAR,
    FitValidationError,
    TracePersistence,
    configured_trace_persistence,
    plan_trace_persistence,
    require_full_trace,
    save_trace,
    set_trace_persistence,
)


def _trace(*, subject_re: bool = True, raw_effect: bool = True, joint: bool = False) -> xr.DataTree:
    """A trace shaped like the real ones: same group and dimension names."""
    nc, nd, nobs, nplot, nall, nsub = 2, 5, 7, 4, 13, 3
    r = np.random.default_rng(0)

    def post(dims_sizes):
        return (("chain", "draw", *dims_sizes[0]), r.normal(size=(nc, nd, *dims_sizes[1])))

    data = {
        # observation-sized deterministics: droppable
        "f_obs": post((("obs_id",), (nobs,))),
        "p_obs": post((("obs_id",), (nobs,))),
        "kappa_obs": post((("obs_id",), (nobs,))),
        # the concatenated obs+plot+query grid: droppable
        "f_all": post((("all_id",), (nall,))),
        "g_unit": post((("all_id",), (nall,))),
        # reporting grid: must be kept
        "p_plot": post((("plot_id",), (nplot,))),
        "kappa_plot": post((("plot_id",), (nplot,))),
        # free scalars: must be kept
        "eta": (("chain", "draw"), r.normal(size=(nc, nd))),
        "tau_subject": (("chain", "draw"), abs(r.normal(size=(nc, nd)))),
    }
    if subject_re:
        data["delta_subject"] = post((("subject_id",), (nsub,)))
        if raw_effect:
            data["delta_subject_raw"] = post((("subject_id",), (nsub,)))
    groups = {
        "/posterior": xr.Dataset(data),
        # observation-dimensioned but must never be touched
        "/observed_data": xr.Dataset({"y_obs": (("obs_id",), r.integers(0, 9, nobs))}),
        "/constant_data": xr.Dataset(
            {"X_obs": (("obs_id",), r.normal(size=nobs)),
             "X_plot": (("plot_id",), np.linspace(8, 30, nplot))}
        ),
        "/sample_stats": xr.Dataset({"diverging": (("chain", "draw"), np.zeros((nc, nd), bool))}),
        "/log_likelihood": xr.Dataset(
            {("y_u_obs" if joint else "y_obs"): (("chain", "draw", "obs_u_id" if joint else "obs_id"),
                                                 r.normal(size=(nc, nd, nobs)))}
        ),
        "/posterior_predictive": xr.Dataset(
            {"y_obs": (("chain", "draw", "obs_id"), r.integers(0, 9, (nc, nd, nobs))),
             "y_plot": (("chain", "draw", "plot_id"), r.integers(0, 9, (nc, nd, nplot)))}
        ),
    }
    return xr.DataTree.from_dict(groups)


# ---- policy ----
def test_full_drops_nothing():
    assert plan_trace_persistence(_trace(), TracePersistence.FULL) == {}


def test_compact_drops_observation_sized_and_all_grid():
    plan = plan_trace_persistence(_trace(), "compact")
    assert set(plan) == {"posterior"}
    assert set(plan["posterior"]) >= {"f_obs", "p_obs", "kappa_obs", "f_all", "g_unit"}


def test_compact_keeps_the_reporting_grid_and_free_scalars():
    dropped = plan_trace_persistence(_trace(), "compact")["posterior"]
    for keep in ("p_plot", "kappa_plot", "eta", "tau_subject"):
        assert keep not in dropped


def test_compact_drops_the_scaled_random_effect_but_keeps_its_raw_draw():
    # delta = tau * delta_raw, so the scaled copy is exactly recoverable.
    dropped = plan_trace_persistence(_trace(), "compact")["posterior"]
    assert "delta_subject" in dropped
    assert "delta_subject_raw" not in dropped


def test_a_random_effect_without_a_raw_counterpart_is_kept():
    # The centred branch samples `delta` directly; there the scaled copy is the
    # only record of it and dropping it would lose the effect.
    trace = _trace(subject_re=True, raw_effect=False)
    assert "delta_subject" not in plan_trace_persistence(trace, "compact")["posterior"]


def test_protected_groups_are_never_planned_for_dropping():
    # observed_data/y_obs and constant_data/X_obs are obs_id-dimensioned; an
    # unscoped dimension rule would delete the data itself.
    plan = plan_trace_persistence(_trace(), "minimal")
    for group in ("observed_data", "constant_data", "sample_stats"):
        assert group not in plan


def test_minimal_also_drops_stored_likelihood_and_predictive():
    plan = plan_trace_persistence(_trace(), "minimal")
    assert plan["log_likelihood"] == ["y_obs"]
    assert plan["posterior_predictive"] == ["y_obs"]        # y_plot is grid-sized
    assert "y_plot" not in plan["posterior_predictive"]


def test_joint_models_per_outcome_likelihood_dims_are_recognised():
    # Joint models index log_likelihood as obs_u_id / obs_s_id, not obs_id.
    plan = plan_trace_persistence(_trace(joint=True), "minimal")
    assert plan["log_likelihood"] == ["y_u_obs"]


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        plan_trace_persistence(_trace(), "smallish")


# ---- writing ----
def test_save_trace_does_not_mutate_the_in_memory_trace(tmp_path):
    # Later pipeline stages read context.trace after the save; extract_model_samples
    # needs f_obs, which compact drops from the file.
    trace = _trace()
    before = set(trace["posterior"].to_dataset().data_vars)
    save_trace(trace, str(tmp_path), persistence="compact")
    assert set(trace["posterior"].to_dataset().data_vars) == before
    assert "f_obs" in before


def test_saved_compact_trace_keeps_the_data_and_the_reporting_grid(tmp_path):
    save_trace(_trace(), str(tmp_path), persistence="compact")
    written = xr.open_datatree(tmp_path / "trace.nc")
    try:
        post = written["posterior"].to_dataset()
        assert "f_obs" not in post and "f_all" not in post
        assert {"p_plot", "kappa_plot", "eta", "tau_subject"} <= set(post.data_vars)
        assert "y_obs" in written["observed_data"].to_dataset()
        assert "X_plot" in written["constant_data"].to_dataset()
        assert "y_obs" in written["log_likelihood"].to_dataset()   # kept at compact
    finally:
        written.close()


def test_save_trace_full_is_byte_for_byte_what_it_always_was(tmp_path):
    trace = _trace()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    record = save_trace(trace, str(a), persistence="full")
    trace.to_netcdf(b / "trace.nc")                              # the pre-change call
    assert record == {"persistence": "full", "dropped": {}, "dropped_count": 0}
    left, right = xr.open_datatree(a / "trace.nc"), xr.open_datatree(b / "trace.nc")
    try:
        assert set(left.children) == set(right.children)
        for group in left.children:
            xr.testing.assert_identical(
                left[group].to_dataset(), right[group].to_dataset()
            )
    finally:
        left.close()
        right.close()


def test_save_trace_reports_what_it_dropped(tmp_path):
    record = save_trace(_trace(), str(tmp_path), persistence="minimal")
    assert record["persistence"] == "minimal"
    assert record["dropped_count"] == sum(len(v) for v in record["dropped"].values())
    assert "f_obs" in record["dropped"]["posterior"]


def test_a_tier_that_cannot_be_applied_fails_rather_than_silently_writing_full(tmp_path):
    # The dangerous failure is not a crash, it is a `compact` request that finds
    # no groups, plans nothing, and writes the full trace anyway: the artifact
    # then looks correct and the policy has quietly done nothing.
    class NotATrace:
        def to_netcdf(self, path):  # pragma: no cover - must not be reached
            raise AssertionError("wrote the trace despite an inapplicable tier")

    with pytest.raises(TypeError, match="no readable 'posterior' group"):
        save_trace(NotATrace(), str(tmp_path), persistence="compact")


def test_full_still_writes_anything_that_can_write_itself(tmp_path):
    # FULL applies no policy, so it must not require a DataTree.
    written = []

    class Minimal:
        def to_netcdf(self, path):
            written.append(path)

    record = save_trace(Minimal(), str(tmp_path), persistence="full")
    assert written and record["dropped_count"] == 0


# ---- configuration ----
def test_the_default_is_full_so_existing_behaviour_is_unchanged(monkeypatch):
    monkeypatch.delenv(TRACE_PERSISTENCE_ENV_VAR, raising=False)
    set_trace_persistence(None)
    assert configured_trace_persistence() is TracePersistence.FULL


def test_the_environment_variable_is_honoured(monkeypatch):
    set_trace_persistence(None)
    monkeypatch.setenv(TRACE_PERSISTENCE_ENV_VAR, "  COMPACT  ")
    assert configured_trace_persistence() is TracePersistence.COMPACT


def test_an_explicit_override_beats_the_environment(monkeypatch):
    monkeypatch.setenv(TRACE_PERSISTENCE_ENV_VAR, "compact")
    set_trace_persistence("minimal")
    try:
        assert configured_trace_persistence() is TracePersistence.MINIMAL
    finally:
        set_trace_persistence(None)


def test_a_bad_environment_value_is_rejected_by_name(monkeypatch):
    set_trace_persistence(None)
    monkeypatch.setenv(TRACE_PERSISTENCE_ENV_VAR, "smallish")
    with pytest.raises(ValueError, match=TRACE_PERSISTENCE_ENV_VAR):
        configured_trace_persistence()


def test_save_trace_follows_the_configured_tier_without_being_told(tmp_path, monkeypatch):
    # This is the plumbing: engines call save_trace(trace, dir) with no tier.
    monkeypatch.delenv(TRACE_PERSISTENCE_ENV_VAR, raising=False)
    set_trace_persistence("compact")
    try:
        record = save_trace(_trace(), str(tmp_path))
    finally:
        set_trace_persistence(None)
    assert record["persistence"] == "compact"
    assert "f_obs" in record["dropped"]["posterior"]


def test_an_explicit_tier_pins_a_save_against_the_configuration(tmp_path, monkeypatch):
    # The convergence-failure save relies on this: it must stay full even when
    # the run is configured for compact, because it exists to be investigated.
    monkeypatch.delenv(TRACE_PERSISTENCE_ENV_VAR, raising=False)
    set_trace_persistence("minimal")
    try:
        record = save_trace(_trace(), str(tmp_path), persistence=TracePersistence.FULL)
    finally:
        set_trace_persistence(None)
    assert record["persistence"] == "full"
    assert record["dropped_count"] == 0


# ---- manifest ----
def test_the_manifest_records_what_was_actually_written(tmp_path):
    manifest = tmp_path / "fit_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "model": {"model_id": "VGxx"}}))
    save_trace(_trace(), str(tmp_path), persistence="compact")
    payload = json.loads(manifest.read_text())
    assert payload["artefacts"]["trace"]["persistence"] == "compact"
    assert "f_obs" in payload["artefacts"]["trace"]["dropped"]["posterior"]
    assert payload["model"] == {"model_id": "VGxx"}      # untouched


def test_the_manifest_records_the_pinned_tier_not_the_configured_one(tmp_path, monkeypatch):
    # A manifest claiming `compact` beside a full trace would be worse than none.
    manifest = tmp_path / "fit_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1}))
    monkeypatch.delenv(TRACE_PERSISTENCE_ENV_VAR, raising=False)
    set_trace_persistence("compact")
    try:
        save_trace(_trace(), str(tmp_path), persistence=TracePersistence.FULL)
    finally:
        set_trace_persistence(None)
    assert json.loads(manifest.read_text())["artefacts"]["trace"]["persistence"] == "full"


def test_a_fit_that_writes_no_manifest_is_not_an_error(tmp_path):
    # VG17 writes its trace without a manifest.
    record = save_trace(_trace(), str(tmp_path), persistence="compact")
    assert record["persistence"] == "compact"
    assert not (tmp_path / "fit_manifest.json").exists()


# ---- guard for consumers that need a full trace ----
def _fit_dir(tmp_path, manifest: dict | None):
    if manifest is not None:
        (tmp_path / "fit_manifest.json").write_text(json.dumps(manifest))
    return str(tmp_path)


def test_require_full_trace_passes_a_full_fit(tmp_path):
    save_trace(_trace(), _fit_dir(tmp_path, {"schema_version": 1}), persistence="full")
    require_full_trace(str(tmp_path), purpose="Leave-one-study-out")


def test_require_full_trace_passes_a_fit_that_predates_the_setting(tmp_path):
    # Every fit written before this existed was full and carries no record;
    # those must keep working rather than be refused for lacking a field.
    require_full_trace(_fit_dir(tmp_path, {"schema_version": 1}), purpose="LOSO")
    require_full_trace(_fit_dir(tmp_path, None), purpose="LOSO")


def test_require_full_trace_rejects_a_compacted_fit_and_says_how_to_fix_it(tmp_path):
    save_trace(_trace(), _fit_dir(tmp_path, {"schema_version": 1}), persistence="compact")
    with pytest.raises(FitValidationError) as excinfo:
        require_full_trace(str(tmp_path), purpose="Leave-one-study-out")
    message = str(excinfo.value)
    assert "Leave-one-study-out" in message
    assert "'compact'" in message
    assert "--trace-persistence full" in message
    assert "f_obs" in message            # names what is actually absent


def test_require_full_trace_rejects_a_minimal_fit(tmp_path):
    save_trace(_trace(), _fit_dir(tmp_path, {"schema_version": 1}), persistence="minimal")
    with pytest.raises(FitValidationError, match="'minimal'"):
        require_full_trace(str(tmp_path), purpose="LOSO")


def test_the_guard_reads_the_manifest_without_opening_the_trace(tmp_path):
    # The point of checking the manifest is to fail before reading tens of GB,
    # so it must not need the trace to be present at all.
    _fit_dir(tmp_path, {"artefacts": {"trace": {"persistence": "compact", "dropped": {}}}})
    assert not (tmp_path / "trace.nc").exists()
    with pytest.raises(FitValidationError):
        require_full_trace(str(tmp_path), purpose="LOSO")


# ---- non-centred effects named for their distribution (z_*) ----
def _joint_modality_trace() -> xr.DataTree:
    """The joint-modality / trivariate naming: `delta_u` beside `z_u`, no `_raw`."""
    nc, nd, nstudy, nsub, nsign = 2, 5, 6, 4, 3
    r = np.random.default_rng(1)

    def var(dim, size):
        return (("chain", "draw", dim), r.normal(size=(nc, nd, size)))

    return xr.DataTree.from_dict({
        "/posterior": xr.Dataset({
            "delta_u": var("study_id", nstudy),
            "z_u": var("study_id", nstudy),
            "delta_subj_u": var("subject_id", nsub),
            "z_subj_u": var("subject_id", nsub),
            # delta_sign scatters z_sign (sign-informed studies only) into a
            # zero-filled vector over every study: not an elementwise scaling.
            "delta_sign": var("study_id", nstudy),
            "z_sign": var("z_sign_dim_0", nsign),
            "tau_u": (("chain", "draw"), abs(r.normal(size=(nc, nd)))),
        }),
        "/observed_data": xr.Dataset({"y_obs": (("obs_id",), r.integers(0, 9, 3))}),
    })


def test_effects_named_z_are_recognised_as_raw_counterparts():
    # The joint-modality engines name the offset for its distribution, not the
    # effect. Without this the duplication stays and the saving does not happen.
    dropped = plan_trace_persistence(_joint_modality_trace(), "compact")["posterior"]
    assert "delta_u" in dropped
    assert "delta_subj_u" in dropped
    assert "z_u" not in dropped and "z_subj_u" not in dropped


def test_a_scattered_effect_is_kept_because_it_is_not_a_scaling():
    # delta_sign is built by scattering z_sign into a wider vector, so it cannot
    # be rebuilt from z_sign and a scale. Dimensions differ, so it must survive.
    dropped = plan_trace_persistence(_joint_modality_trace(), "compact")["posterior"]
    assert "delta_sign" not in dropped
    assert "z_sign" not in dropped


def test_nutpie_backend_resolves_override_then_environment_then_numba(monkeypatch):
    """``configured_nutpie_backend`` follows trace persistence's resolution (#289 4.1).

    The backend is nutpie's compiler, not a sampling parameter: it changes
    nothing about the posterior and is recorded in the manifest's runtime
    block. ``numba`` is the default every fit of record used; ``jax`` is the
    escape hatch for a graph numba cannot compile on a platform.
    """
    from vocab_growth.fit_artifacts import (
        NUTPIE_BACKEND_ENV_VAR,
        configured_nutpie_backend,
        set_nutpie_backend,
    )

    monkeypatch.delenv(NUTPIE_BACKEND_ENV_VAR, raising=False)
    set_nutpie_backend(None)
    try:
        assert configured_nutpie_backend() == "numba"
        monkeypatch.setenv(NUTPIE_BACKEND_ENV_VAR, "  JAX ")
        assert configured_nutpie_backend() == "jax"
        set_nutpie_backend("numba")
        assert configured_nutpie_backend() == "numba", "the override beats the environment"
        set_nutpie_backend(None)
        monkeypatch.setenv(NUTPIE_BACKEND_ENV_VAR, "cuda")
        with pytest.raises(ValueError, match="expected one of numba, jax"):
            configured_nutpie_backend()
        with pytest.raises(ValueError, match="--nutpie-backend"):
            set_nutpie_backend("torch")
    finally:
        set_nutpie_backend(None)
