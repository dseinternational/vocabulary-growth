# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared blocks the model reports print, computed from the fit on disk.

Every model report is a Quarto template copied into its fitted output directory
and rendered there. Blocks that were hand-written into those fifteen templates
have drifted from the fits they describe, and a review of all fifteen found the
same failure repeatedly: a prior stated in prose is a copy of a number that
lives in ``definitions.py``, and when the registered value moves the prose does
not. VG10's report described ``eta_q`` as ``HalfNormal(0.20)`` in three places
after it was widened to 0.8; VG15 made the same error in four; VG02 quoted a
frame size of 346 rows directly above a table rendering 987.

The remedy is not to correct the copies but to stop copying. Each helper here
reads ``fit_manifest.json`` -- which records the definition the fit actually
used, not the one registered today -- and prints the block from it. Field
descriptions are stable prose held in this module; every number comes from the
file. A refit therefore updates the report, and a stale number becomes
impossible rather than merely unlikely.

All helpers are written for a report cell with ``#| output: asis``, the pattern
:func:`vocab_growth.models.calibration.render_calibration_section` established.
"""

from __future__ import annotations

import json
import math
import os

from scipy import stats

from vocab_growth.glossary import render_glossary  # noqa: F401  (re-exported)
from vocab_growth.models.diagnostics_utils import (  # noqa: F401  (re-exported)
    render_convergence_caveats,
)

MANIFEST_FILENAME = "fit_manifest.json"

# Sampling configurations that are reporting-grade. Read from the manifest's
# recorded name rather than inferred from chains x draws: the old templates
# carried a hard-coded {(chains, draws): label} table, and every fit run at
# rep-hightune (6 x 8,000 and 6 x 10,000 draws -- *more* effort than the table's
# "reporting" entry of 6 x 6,000) fell through it to the default and published
# the sentence "It was not fitted in reporting mode". Five models of record said
# that about themselves: VG08, VG09, VG11, VG12 and VG13.
REPORTING_CONFIGS = frozenset({"rep", "rep-hightune", "rep-lite"})

CONFIG_LABELS = {
    "rep": "reporting",
    "rep-hightune": "reporting (high-tune)",
    "rep-lite": "reporting-lite",
    "test": "test",
    "dev": "development",
}


def read_manifest(directory: str = ".") -> dict:
    """The fit manifest for a rendered report, or an empty dict when absent."""
    path = os.path.join(directory, MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _trace_dimensions(directory: str) -> tuple[int | None, int | None]:
    """Chains and draws from the trace header, without loading the posterior."""
    try:
        import h5netcdf
    except ImportError:  # pragma: no cover - h5netcdf is a hard dependency of the reports
        return None, None
    path = os.path.join(directory, "trace.nc")
    if not os.path.isfile(path):
        return None, None
    try:
        with h5netcdf.File(path, "r") as handle:
            posterior = handle["posterior"]
            return (
                posterior.dimensions["chain"].size,
                posterior.dimensions["draw"].size,
            )
    except (OSError, KeyError):
        return None, None


def render_sampling_banner(directory: str = ".") -> None:
    """Print the preliminary-output warning, naming the configuration actually used.

    The configuration name is authoritative: it is what the fit was launched
    with and what the publication gate checks. Chains and draws are reported
    alongside it as corroborating detail, not as the thing the label is derived
    from.
    """
    manifest = read_manifest(directory)
    config = (manifest.get("sampling") or {}).get("configuration_name")
    chains, draws = _trace_dimensions(directory)

    if config is None:
        label, approximate = "an unrecorded", True
    else:
        label = CONFIG_LABELS.get(config, config)
        approximate = config not in REPORTING_CONFIGS

    print('::: {.callout-warning title="Warning: Preliminary model output"}')
    print()
    print(
        "This model output is preliminary and likely to change as the model "
        "evolves and further data are received."
    )
    print()
    effort = ""
    if chains is not None and draws is not None:
        effort = f" ({chains:,} chains × {draws:,} draws per chain)"
    print(f"This model was fitted in **{label}** configuration{effort}.")
    if approximate:
        print()
        print(
            "It was **not** fitted at reporting quality, so these results are "
            "more approximate than a full reporting-quality fit."
        )
    print(":::")


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

# Each entry maps a *fitted parameter name* to the definition fields carrying
# its prior, plus stable prose describing it. Keying on the parameter rather
# than the definition field is what makes the table self-gating: the definition
# dataclass records defaults for machinery a given model does not instantiate
# (VG05's manifest carries `tau_u_sigma` though VG05 has no study effects, and
# every bivariate manifest carries `beta_lag_sigma` though only VG16 uses it),
# so listing priors from the definition alone would describe parameters that do
# not exist in the fit. The fitted parameter set is read from `diagnostics.csv`,
# which is the gate's own record of what was sampled.
#
# `kind` selects the plain-scale reading: proportions get a word count against
# the reference inventory, ratios stay on their own scale, scales get an odds
# multiplier, and amplitudes are departures from the straight-line trend.
_PRIOR_SPECS: list[tuple[str, str, str, str]] = [
    # (parameter, description, definition field stem, kind)
    ("p_slope_low", "Expected proportion at the low age anchor", "p_slope_low", "words"),
    ("p_slope_hi", "Expected proportion at the high age anchor", "p_slope_hi", "words"),
    ("p_slope_low_u", "Understood proportion at the low age anchor", "p_slope_low_u", "words"),
    ("p_slope_hi_u", "Understood proportion at the high age anchor", "p_slope_hi_u", "words"),
    ("p_slope_low_q", "Production ratio $q$ at the low age anchor", "p_slope_low_q", "ratio"),
    ("p_slope_hi_q", "Production ratio $q$ at the high age anchor", "p_slope_hi_q", "ratio"),
    ("p_slope_low_sign", "Signed ratio at the low age anchor", "p_slope_low_sign", "ratio"),
    ("p_slope_mid_sign", "Signed ratio at the peak anchor", "p_slope_mid_sign", "ratio"),
    ("p_slope_hi_sign", "Signed ratio at the high age anchor", "p_slope_hi_sign", "ratio"),
    ("peak_unit_sign", "Age at which the signed ratio peaks", "peak_unit_sign", "peak"),
    ("ell_unit", "GP length-scale", "ell_unit", "lengthscale"),
    ("ell_unit_u", "GP length-scale, understood", "ell_unit_u", "lengthscale"),
    ("ell_unit_q", "GP length-scale, production ratio $q$", "ell_unit_q", "lengthscale"),
    ("ell_unit_sign", "GP length-scale, signing", "ell_unit_sign", "lengthscale"),
    ("eta", "GP amplitude", "eta_sigma", "amplitude"),
    ("eta_u", "GP amplitude, understood", "eta_u_sigma", "amplitude"),
    ("eta_q", "GP amplitude, production ratio $q$", "eta_q_sigma", "amplitude"),
    ("eta_sign", "GP amplitude, signing", "eta_sign_sigma", "amplitude"),
    ("tau", "Between-study SD", "tau_study_sigma", "odds"),
    ("tau_u", "Between-study SD, understood", "tau_u_sigma", "odds"),
    ("tau_q", "Between-study SD, production ratio $q$", "tau_q_sigma", "odds"),
    ("tau_sign", "Between-study SD, signing", "tau_sign_sigma", "odds"),
    ("tau_subject", "Between-child SD", "tau_subject_sigma", "odds"),
    ("tau_subj_u", "Between-child SD, understood", "tau_subj_u_sigma", "odds"),
    ("tau_subj_q", "Between-child SD, production ratio $q$", "tau_subj_q_sigma", "odds"),
    ("tau_subj_sign", "Between-child SD, signing", "tau_subj_sign_sigma", "odds"),
    ("tau_psi", "Between-study SD of the sign–speech association", "tau_psi_sigma", "odds"),
    ("log_psi", "Sign–speech association $\\psi$ (log scale)", "log_psi", "log_psi"),
    ("beta_lag", "Cross-lag coefficient $\\beta$", "beta_lag", "lag"),
]


def fitted_parameters(directory: str = ".") -> set[str]:
    """Names of the parameters this fit actually sampled, from its diagnostics."""
    path = os.path.join(directory, "diagnostics.csv")
    if not os.path.isfile(path):
        return set()
    import pandas as pd

    try:
        return {str(name) for name in pd.read_csv(path, index_col=0).index}
    except (OSError, ValueError):
        return set()


def _prior_row(
    parameter: str,
    description: str,
    stem: str,
    kind: str,
    definition: dict,
) -> tuple[str, str, str] | None:
    """One table row, or None when the definition does not carry this prior."""
    n_trials = definition.get("n_trials")
    anchors = definition.get("slope_anchors") or []

    if kind == "peak":
        # The signed peak has no Beta hyper-parameters in the definition: it is a
        # unit-interval parameter mapped onto a stated window of years, and the
        # window is the whole of the prior. Reported here because VG15's report
        # asserted the peak was "fixed at the middle anchor by construction",
        # which was true of an earlier definition and is not true of this fit --
        # `peak_unit_sign` is in the sampled parameter set.
        window = definition.get("sign_peak_prior")
        if not window:
            return None
        return (
            description,
            f"uniform over {window[0]:g}–{window[1]:g} years",
            f"{window[0] * 12:.0f}–{window[1] * 12:.0f} months — **estimated, not fixed**",
        )

    if kind in {"words", "ratio", "lengthscale"}:
        alpha = definition.get(f"{stem}_alpha")
        beta = definition.get(f"{stem}_beta")
        if alpha is None or beta is None:
            return None
        distribution = f"Beta({alpha:g}, {beta:g})"
        median = float(stats.beta.ppf(0.5, alpha, beta))
        lo = float(stats.beta.ppf(0.05, alpha, beta))
        hi = float(stats.beta.ppf(0.95, alpha, beta))

        if kind == "lengthscale":
            window = definition.get("ell_months_range")
            if not window:
                return description, distribution, "on the unit interval"
            span = window[1] - window[0]
            reading = (
                f"maps to {window[0]:g}–{window[1]:g} months; "
                f"median ≈ {window[0] + median * span:.1f} months"
            )
            return description, distribution, reading

        # Signing carries its own three anchor ages, which are not the understood
        # and q anchors; labelling them with `slope_anchors` would put the signed
        # low anchor at 24 months when the fit places it at 15.
        if stem.endswith("_sign"):
            anchors = definition.get("sign_anchor_ages") or []
        label = description
        if anchors:
            if "low" in stem:
                age = anchors[0]
            elif "mid" in stem and len(anchors) > 2:
                age = anchors[1]
            else:
                age = anchors[-1]
            label = f"{description} ({age:g} months)"
        if kind == "ratio" or not n_trials:
            return label, distribution, f"median {median:.3f}, 5–95% {lo:.3f}–{hi:.3f}"
        return (
            label,
            distribution,
            f"median {median:.3f} ({median * n_trials:.0f} words), "
            f"5–95% {lo:.3f}–{hi:.3f} ({lo * n_trials:.0f}–{hi * n_trials:.0f} words)",
        )

    if kind in {"odds", "amplitude"}:
        sigma = definition.get(stem)
        if sigma is None:
            return None
        median = float(stats.halfnorm.ppf(0.5, scale=sigma))
        if kind == "amplitude":
            reading = f"median {median:.2f} logits of departure from the straight-line trend"
        else:
            reading = f"median {median:.2f} logits (odds ×{math.exp(median):.2f} at +1 SD)"
        return description, f"HalfNormal({sigma:g})", reading

    if kind in {"lag", "log_psi"}:
        mu = definition.get(f"{stem}_mu", 0.0)
        sigma = definition.get(f"{stem}_sigma")
        if sigma is None:
            return None
        if kind == "log_psi":
            reading = (
                f"$\\psi$ prior median {math.exp(mu):.2f}; "
                f"P($\\psi>1$) = {1 - stats.norm.cdf(0, mu, sigma):.3f}"
            )
        else:
            reading = "on the logit scale, per unit of the lag predictor"
        return description, f"Normal({mu:g}, {sigma:g})", reading

    return None


def render_priors_table(directory: str = ".") -> None:
    """Print the fitted model's priors as a table read from its own manifest.

    Anchored proportions are given a word-count reading against the reference
    inventory, and random-effect scales an odds-multiplier reading, because a
    number on the logit scale is not something a reader can picture and every
    review of these reports said so.

    Only parameters this fit actually sampled appear: see :data:`_PRIOR_SPECS`.
    """
    manifest = read_manifest(directory)
    definition = (manifest.get("model") or {}).get("definition") or {}
    if not definition:
        print(
            "_No fit manifest for this fit (`fit_manifest.json` absent), so the "
            "priors cannot be read from the fitted definition._"
        )
        return

    present = fitted_parameters(directory)
    rows: list[tuple[str, str, str]] = []
    for parameter, description, stem, kind in _PRIOR_SPECS:
        if present and parameter not in present:
            continue
        row = _prior_row(parameter, description, stem, kind, definition)
        if row is not None:
            rows.append(row)

    for field in ("kappa", "kappa_u", "kappa_s", "kappa_sign"):
        kappa = definition.get(field)
        if not isinstance(kappa, dict):
            continue
        ages = kappa.get("anchor_ages") or []
        outcome = {
            "kappa_u": ", understood",
            "kappa_s": ", spoken",
            "kappa_sign": ", signed",
        }.get(field, "")
        form = (
            f"anchored at {' and '.join(f'{a:g}' for a in ages)} months"
            if ages
            else "intercept-and-slope form"
        )
        rows.append(
            (
                f"Dispersion $\\kappa${outcome}",
                form,
                "larger $\\kappa$ means *less* between-child spread",
            )
        )

    if not rows:
        print("_This fit's manifest records no recognised prior fields._")
        return

    print("| Parameter | Prior | Reading |")
    print("| --- | --- | --- |")
    for label, distribution, reading in rows:
        print(f"| {label} | `{distribution}` | {reading} |")
    print()
    n_trials = definition.get("n_trials")
    caption = ": Priors as recorded in this fit's `fit_manifest.json`."
    if n_trials:
        caption += f" Word counts are against the {n_trials:,}-word reference inventory."
    print(caption)


def render_model_at_a_glance(directory: str = ".") -> None:
    """Print the model's structural facts, read from the fitted definition."""
    manifest = read_manifest(directory)
    definition = (manifest.get("model") or {}).get("definition") or {}
    data = manifest.get("data") or {}
    if not definition:
        print("_No fit manifest for this fit, so the model summary cannot be read._")
        return

    population = {"ds": "children with Down syndrome", "td": "typically-developing children"}.get(
        definition.get("population"), definition.get("population", "unrecorded")
    )
    items: list[tuple[str, str]] = [("Population", population)]

    n_trials = definition.get("n_trials")
    if n_trials:
        items.append(("Reference inventory", f"{n_trials:,} words"))

    anchors = definition.get("slope_anchors")
    if anchors:
        items.append(("Age anchors", " and ".join(f"{a:g}" for a in anchors) + " months"))

    domain = definition.get("gp_domain_months")
    if domain:
        items.append(("Modelled age range", f"{domain[0]:g}–{domain[1]:g} months"))

    hierarchy = []
    if definition.get("tau_u_sigma") is not None or definition.get("tau_sigma") is not None:
        hierarchy.append("study random intercepts")
    if definition.get("use_subject_re_u"):
        hierarchy.append("child random intercept on understood")
    if definition.get("use_subject_re_q"):
        hierarchy.append("child random intercept on $q$")
    items.append(
        (
            "Hierarchy",
            "; ".join(hierarchy)
            if hierarchy
            else "none — study and repeated-child effects are not modelled here",
        )
    )

    anchor_age = definition.get("gp_anchor_age_months")
    if anchor_age:
        items.append(("GP anchor", f"{anchor_age:g} months"))

    clamp = definition.get("clamp_mean_above_hi_anchor")
    if clamp:
        clamp_text = {
            "q_only": "the production ratio $q$ only",
            True: "all means",
        }.get(clamp, str(clamp))
        items.append(("Mean held level above the high anchor", clamp_text))

    caps = []
    if definition.get("report_max_age_understood"):
        caps.append(f"understood and ratios to {definition['report_max_age_understood']:g} months")
    if definition.get("report_max_age_signed"):
        caps.append(f"signed to {definition['report_max_age_signed']:g} months")
    ages_query = definition.get("ages_query")
    if ages_query:
        caps.append(f"spoken to {max(ages_query):g} months")
    if caps:
        items.append(("Reported ages", "; ".join(caps)))

    rows = data.get("rows")
    if rows:
        items.append(("Observations in the fitted frame", f"{rows:,}"))
    counts = data.get("observed_outcome_counts")
    if isinstance(counts, dict) and counts:
        items.append(
            (
                "Outcome rows",
                ", ".join(f"{k} {v:,}" for k, v in counts.items()),
            )
        )

    print('::: {.callout-note title="Model at a glance"}')
    print()
    for label, value in items:
        print(f"- **{label}:** {value}")
    print()
    print(":::")
