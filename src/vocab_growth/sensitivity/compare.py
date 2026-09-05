# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare registered sensitivity fits with their model of record.

A robustness verdict requires compatible definitions, executable code and
prepared data, complete fit lifecycles, clean convergence of both fits, and
coverage of the engine's expected quantities. Structural parameters are checked
alongside age trajectories. The criterion is descriptive containment within the
baseline's 89% intervals, not proof that a model is insensitive to every prior.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dse_research_utils.statistics.diagnostics import ESS_THRESHOLD, RHAT_MAX

from vocab_growth.fit_artifacts import (
    ACCEPTED_EXCEPTION_KEY,
    DIAGNOSTICS_SUMMARY_FILENAME,
    convergence_caveats,
    validate_fit_output,
)

#: Definition fields that always differ between a baseline and its variant, and
#: whose difference carries no information: the variant's directory name and the
#: banner printed at the top of its fit log.
NAMING_FIELDS = frozenset({"config_name", "banner", "model_id"})

#: Coverage below this fraction of the baseline's own comparable rows means the
#: age grids disagree badly enough that the verdict is not about priors.
MIN_COVERAGE = 0.9

# (quantity, filename, median_col, ci_lo_col, ci_hi_col).
# Missing files are left for the required-quantity coverage check. Present files
# must carry complete, finite summaries.
_SERIES: tuple[tuple[str, str, str, str, str], ...] = (
    ("Ey_understood", "posterior_summary_u.csv", "Ey_median", "Ey_ci_lo", "Ey_ci_hi"),
    ("Ey_spoken", "posterior_summary_s.csv", "Ey_median", "Ey_ci_lo", "Ey_ci_hi"),
    ("Ey_signed", "posterior_summary_sign.csv", "Ey_median", "Ey_ci_lo", "Ey_ci_hi"),
    ("Ey", "posterior_summary.csv", "Ey_median", "Ey_ci_lo", "Ey_ci_hi"),
    ("q", "posterior_summary_q.csv", "q_median", "q_ci_lo", "q_ci_hi"),
    ("r", "posterior_summary_r.csv", "r_median", "r_ci_lo", "r_ci_hi"),
    ("p_any", "posterior_summary_p_any.csv", "p_any_median", "p_any_ci_lo", "p_any_ci_hi"),
    ("Ey_any", "posterior_summary_p_any.csv", "Ey_any_median", "Ey_any_ci_lo", "Ey_any_ci_hi"),
    ("gap", "comprehension_production_gap.csv", "gap_median", "ci_lo", "ci_hi"),
)


def _read(dirpath: str, name: str) -> pd.DataFrame | None:
    path = os.path.join(dirpath, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def load_headlines(dirpath: str) -> dict[str, pd.DataFrame]:
    """Return ``{quantity: DataFrame[age_months, median, ci_lo, ci_hi]}`` for
    every headline series present in ``dirpath`` (missing series are skipped)."""
    out: dict[str, pd.DataFrame] = {}
    for qty, fname, mcol, lo, hi in _SERIES:
        df = _read(dirpath, fname)
        if df is None:
            continue
        # VG15 prefixes the count columns by outcome; older engines do not.
        prefix = {"Ey_understood": "u", "Ey_spoken": "s", "Ey_signed": "sign"}.get(qty)
        if mcol not in df and prefix:
            mcol, lo, hi = (f"Ey_{prefix}_{part}" for part in ("median", "ci_lo", "ci_hi"))
        required = {"age_months", mcol, lo, hi}
        if not required.issubset(df.columns):
            raise ValueError(f"{fname} lacks required columns: {sorted(required - set(df.columns))}")
        frame = pd.DataFrame({"age_months": df["age_months"], "median": df[mcol],
                              "ci_lo": df[lo], "ci_hi": df[hi]})
        _validate_values(frame, f"{dirpath}/{fname}")
        out[qty] = frame
    return out


def load_psi(dirpath: str) -> dict[str, float] | None:
    """The VG15 association scalar summary, or ``None`` if not present."""
    df = _read(dirpath, "posterior_summary_psi.csv")
    if df is None:
        return None
    columns = ["psi_median", "psi_ci_lo", "psi_ci_hi", "P_psi_gt_1"]
    if len(df) != 1 or not set(columns).issubset(df.columns):
        raise ValueError("posterior_summary_psi.csv must contain one complete scalar summary")
    if not np.isfinite(df[columns].to_numpy(dtype=float)).all():
        raise ValueError("posterior_summary_psi.csv contains non-finite values")
    r = df.iloc[0]
    return {
        "psi_median": float(r["psi_median"]),
        "psi_ci_lo": float(r["psi_ci_lo"]),
        "psi_ci_hi": float(r["psi_ci_hi"]),
        "P_psi_gt_1": float(r["P_psi_gt_1"]),
    }


def _validate_values(frame: pd.DataFrame, label: str) -> None:
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains empty or non-finite summary values")
    if frame["age_months"].duplicated().any():
        raise ValueError(f"{label} contains duplicate ages")
    if (frame["ci_lo"] > frame["ci_hi"]).any():
        raise ValueError(f"{label} contains reversed intervals")


_PARAMETER = re.compile(
    r"^(beta_lag|rho_uq|conc|tau_psi|v_total(?:_[uq])?|subject_variance_share(?:_[uq])?|"
    r"tau_subject(?:_(?:young|old))?|tau_subj_(?:u|q|s|sign)(?:_(?:0|1|rho|young|old))?|"
    r"log_tau_(?:subject|subj_(?:u|q))_ratio)$"
)


def load_parameters(dirpath: str) -> dict[str, pd.DataFrame]:
    """Structural parameters from diagnostics, explicitly compared as means.

    The existing scalar summary stores means, not medians. Retaining that
    distinction lets historical summaries be inspected without inventing a
    posterior median from the mean. Both members of a pair use the same statistic.
    """
    df = _read(dirpath, "diagnostics.csv")
    if df is None or not {"mean", "eti89_lb", "eti89_ub"}.issubset(df.columns):
        return {}
    out = {}
    for _, row in df.iterrows():
        name = str(row.iloc[0])
        if not _PARAMETER.fullmatch(name):
            continue
        if name in out:
            raise ValueError(f"Duplicate parameter {name} in {dirpath}/diagnostics.csv")
        frame = pd.DataFrame({"age_months": [-1], "estimate": [row["mean"]],
                              "ci_lo": [row["eti89_lb"]], "ci_hi": [row["eti89_ub"]]})
        _validate_values(frame, f"{dirpath}/diagnostics.csv: {name}")
        out[name] = frame.assign(
            estimate_kind="mean", interval_kind="eti")
    return out


def required_quantities(model_key: str, definition) -> set[str]:
    """Expected outputs, derived from the engine and the child-effect plan.

    Coverage must check this schema even when both fits omit the same output.
    """
    from vocab_growth.models.catalogue import CATALOGUE
    from vocab_growth.models.subject_effects import SubjectEffectKind, resolve

    engine = CATALOGUE[model_key].engine.name
    required = ({"Ey"} if engine.startswith("univariate")
                else {"Ey_understood", "Ey_spoken", "q"})
    if engine in {"trivariate", "joint"}:
        required |= {"Ey_signed", "r", "p_any", "Ey_any"}
    if engine == "joint":
        required |= {"psi", "conc"}
    if getattr(definition, "use_cross_lag", False):
        required.add("beta_lag")
    plan = resolve(definition)
    if plan.correlation_eta is not None or plan.factor is not None:
        required.add("rho_uq")
    if plan.variance_partition is not None:
        required |= {"v_total", "subject_variance_share"}
    for effect in plan.effects:
        if not effect.is_active:
            continue
        name = effect.scale_name
        required.add(name)
        if effect.kind in {SubjectEffectKind.CHILD_SLOPE, SubjectEffectKind.FACTOR}:
            required |= {f"{name}_0", f"{name}_1"}
        if effect.kind is SubjectEffectKind.CHILD_SLOPE:
            required.add(f"{name}_rho")
        if effect.kind is SubjectEffectKind.AGE_VARYING:
            required |= {f"{name}_young", f"{name}_old", f"log_{name}_ratio"}
    return required


def _comparable(dirpath: str) -> dict[str, pd.DataFrame]:
    out = {
        name: frame.rename(columns={"median": "estimate"}).assign(
            estimate_kind="median", interval_kind="eti")
        for name, frame in load_headlines(dirpath).items()
    }
    out.update(load_parameters(dirpath))
    psi = load_psi(dirpath)
    if psi is not None:
        frame = pd.DataFrame({"age_months": [-1], "estimate": [psi["psi_median"]],
                              "ci_lo": [psi["psi_ci_lo"]], "ci_hi": [psi["psi_ci_hi"]]})
        _validate_values(frame, f"{dirpath}/posterior_summary_psi.csv")
        out["psi"] = frame.assign(estimate_kind="median", interval_kind="hdi")
        out["P_psi_gt_1"] = pd.DataFrame({
            "age_months": [-1], "estimate": [psi["P_psi_gt_1"]],
            "ci_lo": [np.nan], "ci_hi": [np.nan],
            "estimate_kind": ["probability"], "interval_kind": [None],
        })
    return out


def pairing_errors(baseline_dir: str, variant_dir: str, model_key: str, name: str) -> list[str]:
    """Validate both fits against current definitions and their own prepared data.

    Rebuilding each frame admits a registered data restriction while rejecting
    unrelated changes to that frame. For prior-only variants the rebuilt frames
    coincide. This checks actual override values, not just permitted field names.
    """
    from dse_research_utils.statistics.models.sampling import get_sampling_configuration

    from vocab_growth.analysis_frames import expected_analysis_frame_hash
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.implementation_identity import implementation_signature
    from vocab_growth.sensitivity.registry import build_variant

    errors = []
    signature = implementation_signature()
    base = MODEL_REGISTRY[model_key]
    variant, = build_variant(model_key, name)
    sampling_names = []
    for label, directory, definition in (("baseline", baseline_dir, base),
                                          ("variant", variant_dir, variant)):
        try:
            manifest = _manifest(directory) or {}
            config = manifest.get("sampling", {}).get("configuration_name")
            sampling_names.append(config)
            if not config:
                raise ValueError("sampling configuration is missing")
            if not manifest.get("code", {}).get("commit"):
                errors.append(f"{label}: fit commit is missing")
            parameters = get_sampling_configuration(config)
            frame_hash = expected_analysis_frame_hash(model_key, definition)
            problems = validate_fit_output(
                directory, expected_definition=definition,
                expected_implementation=signature,
                expected_sampling_config_name=config,
                expected_sampling_parameters=parameters,
                expected_analysis_frame_hash=frame_hash,
                require_clean_fit=True,
            )
            errors.extend(f"{label}: {problem}" for problem in problems)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{label}: cannot verify fit: {exc}")
    if len(sampling_names) == 2 and sampling_names[0] != sampling_names[1]:
        errors.append("baseline and variant use different sampling configurations")
    return errors


#: Separator for the ``caveats`` column. Not ``"; "`` because the caveat strings
#: themselves may contain semicolons (the accepted-exception ``decided`` field).
CAVEATS_SEPARATOR = " | "

#: Caveat recorded when a fit predates ``diagnostics_summary.json`` and the
#: verdict had to fall back to ``diagnostics.csv``.
CSV_FALLBACK_CAVEAT = (
    "convergence assessed from diagnostics.csv only (pre-payload fit): rounded, "
    "scalars-only R-hat/ESS; divergences, energy BFMI and unassessable "
    "parameters were never checked"
)


@dataclass(frozen=True)
class DiagnosticsGate:
    """Convergence verdict for one fit directory (:func:`diagnostics_gate`).

    ``converged`` is the hard R-hat/ESS verdict (``None`` when nothing is
    recorded at all). ``clean`` is the gate payload's ``passed`` — every check,
    hard and soft, together — and is what the "robust"/"recovered" verdicts are
    reserved for; it is ``None`` when no payload exists. ``caveats`` carries the
    soft-tier problems (divergent transitions, low energy BFMI, unassessable
    parameters, a recorded R-hat exception) and, for a pre-payload fit, the
    fallback note itself. Iterating yields ``(converged, max_rhat, min_ess)``
    so existing triple-unpacking callers keep working.
    """

    converged: bool | None
    max_rhat: float | None
    min_ess: float | None
    caveats: tuple[str, ...] = ()
    source: str | None = None
    clean: bool | None = None

    def __iter__(self):
        return iter((self.converged, self.max_rhat, self.min_ess))

    @property
    def caveats_text(self) -> str:
        return CAVEATS_SEPARATOR.join(self.caveats)


def _diagnostics_payload(dirpath: str) -> dict | None:
    path = os.path.join(dirpath, DIAGNOSTICS_SUMMARY_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def diagnostics_gate(dirpath: str) -> DiagnosticsGate:
    """Convergence verdict for the fit in ``dirpath``.

    Reads the canonical gate payload (``diagnostics_summary.json``) when it is
    present: the per-check booleans supply the hard R-hat/ESS verdict (a
    recorded, accepted R-hat exception is treated as caveated rather than
    failed, matching the fit pipeline's own gate), ``passed`` supplies
    :attr:`DiagnosticsGate.clean`, and
    :func:`vocab_growth.fit_artifacts.convergence_caveats` plus the
    unassessable-parameter list supply the soft tier. Only when the payload is
    absent — a fit made before it existed — does this fall back to scanning
    ``diagnostics.csv``, whose values are rounded and scalars-only and which
    records nothing about divergences, BFMI or unassessable parameters; the
    fallback is recorded as a caveat in the returned record, so such a fit can
    never be scored as cleanly converged.
    """
    payload = _diagnostics_payload(dirpath)
    if payload is not None:
        checks = payload.get("checks") or {}
        rhat_ok = bool(checks.get("rhat"))
        if not rhat_ok and payload.get(ACCEPTED_EXCEPTION_KEY) is not None:
            # A registered exception accepts a narrowly-scoped R-hat failure;
            # convergence_caveats reports it, so it belongs to the soft tier.
            rhat_ok = True
        converged = rhat_ok and bool(checks.get("ess"))
        caveats = list(convergence_caveats(payload))
        unassessable = payload.get("unassessable_parameters") or []
        if unassessable:
            shown = ", ".join(unassessable[:6])
            if len(unassessable) > 6:
                shown += f", ... ({len(unassessable)} in total)"
            caveats.append(
                f"R-hat/ESS could not be assessed for {shown}: an unmeasured "
                "parameter is not a passing one"
            )
        max_rhat = payload.get("max_rhat")
        min_ess = payload.get("min_ess")
        return DiagnosticsGate(
            converged=converged,
            max_rhat=None if max_rhat is None else float(max_rhat),
            min_ess=None if min_ess is None else float(min_ess),
            caveats=tuple(caveats),
            source=DIAGNOSTICS_SUMMARY_FILENAME,
            clean=bool(payload.get("passed")),
        )

    df = _read(dirpath, "diagnostics.csv")
    if df is None or "r_hat" not in df.columns:
        return DiagnosticsGate(converged=None, max_rhat=None, min_ess=None)
    max_rhat = float(np.nanmax(df["r_hat"].values))
    ess_cols = [c for c in ("ess_bulk", "ess_tail") if c in df.columns]
    min_ess = float(np.nanmin(df[ess_cols].min(axis=1).values)) if ess_cols else None
    converged = bool(
        max_rhat <= RHAT_MAX and min_ess is not None and min_ess >= ESS_THRESHOLD
    )
    return DiagnosticsGate(
        converged=converged,
        max_rhat=max_rhat,
        min_ess=min_ess,
        caveats=(CSV_FALLBACK_CAVEAT,),
        source="diagnostics.csv",
        clean=None,
    )


def _manifest(dirpath: str) -> dict | None:
    """The fit manifest in ``dirpath``, or ``None`` if it has none."""
    path = os.path.join(dirpath, "fit_manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fit_created_at(dirpath: str) -> str | None:
    """When the fit in ``dirpath`` was made, from its manifest."""
    manifest = _manifest(dirpath)
    return None if manifest is None else manifest.get("created_at_utc")


def definition_mismatch(
    baseline_dir: str, variant_dir: str, override_keys: set[str]
) -> list[str]:
    """Definition fields that differ but should not — empty when the pair is sound.

    A variant fit is only a controlled comparison if it differs from the baseline
    in the fields the registry overrides and nothing else. ``override_keys`` is
    that expected set; anything else in the diff means the two fits were made
    under different model definitions, so the delta is not attributable to the
    variant. This is the check that catches a **stale pairing** — a variant
    fitted before the model of record was refitted under a changed definition,
    which produces a perfectly well-formed and completely misleading matrix row.

    Returns ``[]`` when either manifest is unreadable: an unverifiable pairing is
    reported through :func:`pairing_errors` rather than as a false mismatch.
    This helper alone is not a pairing validator.
    """
    base_m, var_m = _manifest(baseline_dir), _manifest(variant_dir)
    if base_m is None or var_m is None:
        return []
    base = (base_m.get("model") or {}).get("definition") or {}
    var = (var_m.get("model") or {}).get("definition") or {}
    if not base or not var:
        return []
    allowed = NAMING_FIELDS | set(override_keys)
    return sorted(
        k
        for k in set(base) | set(var)
        if k not in allowed and base.get(k) != var.get(k)
    )


def coverage_report(baseline_dir: str, variant_dir: str, *, required: set[str] | None = None) -> tuple[int, int, list[str]]:
    """``(baseline_rows, shared_rows, missing_series)`` for a variant pairing.

    ``baseline_rows`` counts every age the baseline reports across its headline
    series; ``shared_rows`` counts those :func:`compare_dirs` will actually be
    able to pair up. ``missing_series`` names quantities the baseline reports and
    the variant does not at all.

    **This must use exactly the matching rule** :func:`compare_dirs` **uses**, or
    it reports coverage the comparison does not have. The rule is an exact match
    on ``age_months``, which is safe for the query-grid series (integer months
    from ``ages_query``) and consequential for the plot-grid ones: ``gap`` is
    emitted on ``np.linspace(min_age, max_age, n_plot)``, so two fits share its
    ages only if they share an age range. A variant that restricts the pool —
    ``dse-native-only`` is the live example — gets a different linspace, and the
    intersection collapses to the handful of points that coincide by arithmetic
    accident: 3 of 355 in the 2026-08-16 run, which had been reported as a normal
    "sensitive: gap" verdict. Measuring coverage the same way turns that into the
    partial-coverage status it is.
    """
    base, var = _comparable(baseline_dir), _comparable(variant_dir)
    baseline_rows = sum(len(frame) for frame in base.values())
    shared_rows = 0
    for qty, frame in base.items():
        if qty not in var:
            continue
        b_ages = pd.Index(frame["age_months"])
        v_ages = pd.Index(var[qty]["age_months"])
        shared_rows += len(b_ages.intersection(v_ages))
    missing = sorted((set(base) | (required or set())) - (set(base) & set(var)))
    return baseline_rows, shared_rows, missing


def failed_fit_dir(failed_root: str, model_id: str, config_name: str) -> str | None:
    """The most recent retained failed fit for ``<model_id>-<config_name>``.

    A fit stopped by the convergence gate is moved to ``output/failed/`` with a
    UTC timestamp appended, so it is invisible to a ``models/`` lookup. Finding it
    is what lets a non-converged variant appear in the matrix **with its reason**
    rather than as a blank — the requirement recorded in
    ``notes/202608142000-refit-run-record-and-disk-failure.md`` §5b.
    """
    if not os.path.isdir(failed_root):
        return None
    matches = sorted(glob.glob(os.path.join(failed_root, f"{model_id}-{config_name}-*")))
    return matches[-1] if matches else None


def summarise_absent(label: str, status: str, reason: str, variant_dir: str | None = None) -> dict:
    """A matrix row for a variant that produced no comparable summaries.

    ``status`` is ``"not-fitted"`` or ``"failed"``. A failed fit still has its
    diagnostics, so its R-hat, ESS and failing parameters are reported: that is
    the difference between "this variant does not sample" (an informative
    negative) and "nobody ran it".
    """
    row = {
        "variant": label,
        "status": status,
        "converged": None,
        "max_rhat": None,
        "min_ess": None,
        "n_within_ci": 0,
        "n_checked": 0,
        "coverage": None,
        "quantities_outside_ci": "",
        "max_abs_delta": None,
        "caveats": "",
        "verdict": reason,
    }
    if variant_dir:
        gate = diagnostics_gate(variant_dir)
        row["converged"] = gate.converged
        row["max_rhat"] = gate.max_rhat
        row["min_ess"] = gate.min_ess
        row["caveats"] = gate.caveats_text
    return row


def compare_dirs(baseline_dir: str, variant_dir: str) -> pd.DataFrame:
    """Compare shared ages and parameters, with the point statistic labelled.

    ``age_months=-1`` marks a scalar. Parameter estimates from diagnostics are
    means; trajectory and psi summaries are medians. Fractional ages are retained.
    This calculates differences only; :func:`summarise` requires validation
    evidence before making a robustness claim.
    """
    base, var = _comparable(baseline_dir), _comparable(variant_dir)
    rows = []
    for qty in sorted(set(base) & set(var)):
        b, v = base[qty].set_index("age_months"), var[qty].set_index("age_months")
        for age in b.index.intersection(v.index):
            bm, vm = float(b.loc[age, "estimate"]), float(v.loc[age, "estimate"])
            lo, hi = b.loc[age, "ci_lo"], b.loc[age, "ci_hi"]
            kind = b.loc[age, "estimate_kind"]
            if kind != v.loc[age, "estimate_kind"]:
                raise ValueError(f"{qty} uses different point statistics")
            within = bool(lo <= vm <= hi) if pd.notna(lo) and pd.notna(hi) else None
            rows.append({
                "quantity": qty, "age_months": float(age),
                "base_estimate": bm, "var_estimate": vm, "estimate_kind": kind,
                "delta": vm - bm, "base_ci_lo": lo, "base_ci_hi": hi,
                "within_baseline_ci": within, "interval_kind": b.loc[age, "interval_kind"],
            })
    return pd.DataFrame(rows, columns=[
        "quantity", "age_months", "base_estimate", "var_estimate", "estimate_kind",
        "delta", "base_ci_lo", "base_ci_hi", "within_baseline_ci", "interval_kind",
    ])


def summarise(
    comparison: pd.DataFrame,
    variant_dir: str,
    label: str,
    *,
    baseline_dir: str | None = None,
    validation_errors: list[str] | None = None,
    mismatch: list[str] | None = None,
    coverage: tuple[int, int, list[str]] | None = None,
) -> dict:
    """One-row robustness verdict for a variant (feeds the §7 matrix).

    ``validation_errors`` must come from :func:`pairing_errors`, and
    ``baseline_dir`` supplies the other fit's convergence gate. Missing checks
    cannot yield a robustness verdict. ``coverage`` includes required quantities
    even when both fits omit them. Soft convergence caveats from either fit are
    carried into the verdict; hard failures prevent assessment.
    """
    gate = diagnostics_gate(variant_dir)
    base_gate = diagnostics_gate(baseline_dir) if baseline_dir else None
    converged, max_rhat, min_ess = gate
    checked = comparison.dropna(subset=["within_baseline_ci"])
    # The column mixes Python bools with None (P_psi_gt_1 / four-cell rows), so it
    # is object dtype even after dropna; ~ on object bools yields -2/-1, not a
    # mask. Coerce before inverting.
    within = checked["within_baseline_ci"].astype(bool)
    n_within = int(within.sum())
    n_checked = int(len(checked))
    outside = sorted(checked.loc[~within, "quantity"].unique().tolist())
    max_abs_delta = float(comparison["delta"].abs().max()) if len(comparison) else 0.0

    coverage_frac = None
    if coverage is not None:
        baseline_rows, shared_rows, missing = coverage
        coverage_frac = (shared_rows / baseline_rows) if baseline_rows else None

    caveats = list(gate.caveats)
    if base_gate:
        caveats.extend(f"baseline: {caveat}" for caveat in base_gate.caveats)
    caveated = converged is True and bool(caveats)
    status = "compared"
    if mismatch:
        status = "stale-pairing"
        verdict = (
            "STALE PAIRING (not assessed): baseline and variant differ in "
            + ", ".join(mismatch)
            + " — refit the variant against the current model of record"
        )
    elif converged is False or (base_gate and base_gate.converged is False):
        status = "non-converged"
        verdict = "NON-CONVERGED (not assessed): baseline or variant failed the gate"
    elif coverage is not None and (coverage[2] or (coverage_frac is not None and coverage_frac < MIN_COVERAGE)):
        status = "partial-coverage"
        missing_note = (
            f"; missing series: {', '.join(coverage[2])}" if coverage[2] else ""
        )
        verdict = (
            f"PARTIAL COVERAGE (not assessed): only {coverage[1]} of "
            f"{coverage[0]} baseline rows are shared{missing_note}"
        )
    elif validation_errors is None or validation_errors or base_gate is None:
        status = "unverified-pairing"
        verdict = "UNVERIFIED PAIRING (not assessed): " + "; ".join(
            validation_errors or ["both fits require provenance and convergence checks"])
    elif converged is None or base_gate.converged is None:
        status = "unverified-convergence"
        verdict = "CONVERGENCE NOT VERIFIED (not assessed): baseline or variant has no gate"
    elif not n_checked:
        status = "no-comparable-output"
        verdict = "NO COMPARABLE OUTPUT (not assessed)"
    else:
        if caveated:
            status = "converged-with-caveats"
        if outside:
            verdict = "sensitive: " + ", ".join(outside)
            if caveated:
                verdict += " (converged with caveats)"
        elif gate.clean is True and base_gate.clean is True and coverage is not None:
            verdict = "robust (all within baseline 89% interval)"
        elif caveated:
            verdict = (
                "within baseline 89% interval, but converged with caveats — "
                "not scored robust (see caveats)"
            )
        else:
            # Convergence was never recorded at all. The containment holds, but
            # a robustness claim needs a clean gate payload behind it.
            verdict = (
                "within baseline 89% interval, but no recorded convergence "
                "gate — not scored robust"
            )
    return {
        "variant": label,
        "status": status,
        "converged": converged,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "n_within_ci": n_within,
        "n_checked": n_checked,
        "coverage": coverage_frac,
        "quantities_outside_ci": ", ".join(outside),
        "max_abs_delta": max_abs_delta,
        "baseline_converged": base_gate.converged if base_gate else None,
        "baseline_max_rhat": base_gate.max_rhat if base_gate else None,
        "baseline_min_ess": base_gate.min_ess if base_gate else None,
        "caveats": CAVEATS_SEPARATOR.join(caveats),
        "verdict": verdict,
    }
