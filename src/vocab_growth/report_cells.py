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
from collections.abc import Mapping

from scipy import stats

from vocab_growth.administration_loo import ADMINISTRATION_LABEL
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
    # The knot, not the curve maximum: the anchor heights are sampled
    # independently and a GP departure is then added, so the fitted r(a) can
    # peak elsewhere (#238). The full-curve peak is `signed_ratio_peak.csv`.
    ("peak_unit_sign", "Age of the signed tent's peak anchor (knot)", "peak_unit_sign", "peak"),
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
    ("v_total", "Total logit-scale scatter at the young dispersion anchor", "", "vp_total"),
    (
        "subject_variance_share",
        "Share of that scatter attributable to persistent child differences",
        "",
        "vp_share",
    ),
    ("tau_subj_u", "Between-child SD, understood", "tau_subj_u_sigma", "odds"),
    ("tau_subj_q", "Between-child SD, production ratio $q$", "tau_subj_q_sigma", "odds"),
    ("tau_subj_sign", "Between-child SD, signing", "tau_subj_sign_sigma", "odds"),
    ("tau_psi", "Between-study SD of the sign–speech association", "tau_psi_sigma", "odds"),
    # VG20's correlation had no entry here at all, so its priors table omitted
    # the one prior the model exists to place (#233). It is a top-level scalar
    # field rather than part of a subject-scale block, so it needs its own kind
    # -- `_subject_scale_row` never sees it.
    (
        "rho_uq",
        "Correlation between a child's understood and $q$ offsets $\\rho_{uq}$",
        "subject_re_correlation_eta",
        "lkj",
    ),
    ("log_psi", "Sign–speech association $\\psi$ (log scale)", "log_psi", "log_psi"),
    ("beta_lag", "Cross-lag coefficient $\\beta$", "beta_lag", "lag"),
    # VG15 samples this and its own page names it a prior-sensitivity target,
    # but the table had no row for it -- the same omission as VG22's factor
    # block, found by the coverage check written for that one (#273).
    (
        "log_conc",
        "Dirichlet-Multinomial concentration (log scale)",
        "log_conc",
        "log_concentration",
    ),
    # Sampled only under `spoken_fallback="separate_dispersion"`, a registered
    # sensitivity variant -- and a variant fit renders the model of record's
    # template, so a prior with no row shows up on a real page. One per nested
    # outcome: the signing engines apply the treatment to their signed rows too,
    # so VG14 and VG15 sample `log_kappa_sign_fallback` as well (#266 finding 8).
    (
        "log_kappa_s_fallback",
        "Dispersion offset, spoken rows with no usable understood count",
        "spoken_fallback_kappa",
        "log_multiplier",
    ),
    (
        "log_kappa_sign_fallback",
        "Dispersion offset, signed rows with no usable understood count",
        "spoken_fallback_kappa",
        "log_multiplier",
    ),
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

    if kind in {"vp_total", "vp_share"}:
        # VG11 and VG12 reparameterise the child scale and the young dispersion
        # anchor into one shared budget, so `tau_subject` becomes a deterministic
        # function of these two and its HalfNormal prior is never used. Reporting
        # that HalfNormal as though it governed the fit -- which this table did --
        # describes a prior with no effect on the posterior.
        partition = definition.get("subject_variance_partition")
        if not partition:
            return None
        if kind == "vp_total":
            mu = partition.get("total_mu", 0.0)
            sigma = partition.get("total_sigma")
            if sigma is None:
                return None
            return (
                description,
                f"LogNormal({mu:g}, {sigma:g})",
                f"median {math.exp(mu):.2f} on the logit variance scale",
            )
        alpha = partition.get("share_alpha")
        beta = partition.get("share_beta")
        if alpha is None or beta is None:
            return None
        median = float(stats.beta.ppf(0.5, alpha, beta))
        return (
            description,
            f"Beta({alpha:g}, {beta:g})",
            f"median {median:.2f}; the remainder is within-child noise ($\\kappa$)",
        )

    if kind == "peak":
        # `sign_peak_prior` is (alpha, beta) of a Beta on the peak's POSITION
        # between the OUTER signed anchors -- not a window of ages. Reading the
        # pair as a year range gave "uniform over 2-4 years"; it is Beta(2, 4)
        # over 15-96 months, whose median sits above 40 months.
        #
        # Reported at all because VG15's report asserted the peak was "fixed at
        # the middle anchor by construction". That was true of an earlier
        # definition; this fit samples `peak_unit_sign`, and the fixed knot was
        # abandoned because it made the peak's age an assertion rather than an
        # estimate (see the field's docstring in definitions.py).
        params = definition.get("sign_peak_prior")
        anchors = definition.get("sign_anchor_ages")
        if not params or not anchors:
            return None
        alpha, beta = params[0], params[1]
        low, high = float(anchors[0]), float(anchors[-1])
        span = high - low
        median = low + float(stats.beta.ppf(0.5, alpha, beta)) * span
        lo = low + float(stats.beta.ppf(0.05, alpha, beta)) * span
        hi = low + float(stats.beta.ppf(0.95, alpha, beta)) * span
        return (
            description,
            f"Beta({alpha:g}, {beta:g}) on its position between "
            f"{low:g} and {high:g} months",
            f"prior median {median:.0f} months, 5–95% {lo:.0f}–{hi:.0f} — "
            "**estimated, not fixed**",
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
        # The subject-scale fields are overloaded. A float is the constant
        # between-child scale every model of record carries, but VG19 puts a
        # child intercept-and-rate block there (`SubjectSlopePriorParams`) and
        # Proposal A1 an age-varying scale (`AgeVaryingSubjectScale`). Once the
        # definition has been through `asdict` both arrive as mappings, and
        # scipy then raises a bare `TypeError: '>' not supported between
        # instances of 'dict' and 'int'` from inside `ppf` -- which surfaces as
        # an unrenderable report page rather than as anything diagnosable.
        if isinstance(sigma, Mapping):
            return _subject_scale_row(description, sigma)
        median = float(stats.halfnorm.ppf(0.5, scale=sigma))
        if kind == "amplitude":
            reading = f"median {median:.2f} logits of departure from the straight-line trend"
        else:
            reading = f"median {median:.2f} logits (odds ×{math.exp(median):.2f} at +1 SD)"
        return description, f"HalfNormal({sigma:g})", reading

    if kind == "lkj":
        # `(rho + 1) / 2 ~ Beta(eta, eta)` is exactly LKJ(eta) for a 2x2, which
        # is how the build writes it so the correlation stays a named variable.
        # Named as LKJ here for the same reason `_subject_scale_row` does.
        eta = definition.get(stem)
        if eta is None:
            return None
        eta = float(eta)
        lo = float(stats.beta.ppf(0.05, eta, eta)) * 2.0 - 1.0
        hi = float(stats.beta.ppf(0.95, eta, eta)) * 2.0 - 1.0
        if eta == 1.0:
            emphasis = "flat over (-1, 1), so no size of correlation is favoured"
        elif eta > 1.0:
            emphasis = "pulled toward zero, so a correlation has to be evidenced"
        else:
            emphasis = "pushed toward ±1, which favours a strong correlation"
        return (
            description,
            f"LKJ({eta:g}), i.e. $(\\rho_{{uq}}+1)/2 \\sim$ Beta({eta:g}, {eta:g})",
            f"centred on zero and {emphasis}; 5–95% {lo:+.2f} to {hi:+.2f}",
        )

    if kind == "log_multiplier":
        sigma = definition.get(f"{stem}_sigma")
        if sigma is None:
            return None
        hi = math.exp(1.598 * sigma)  # 89% equal-tailed, centred on zero
        return (
            description,
            f"Normal(0, {sigma:g})",
            f"a multiplier on the shared age-varying $\\kappa$; centred on 1 "
            f"(no separate dispersion), 89% {1 / hi:.2f}–{hi:.2f}",
        )

    if kind == "log_concentration":
        mu = definition.get(f"{stem}_mu")
        sigma = definition.get(f"{stem}_sigma")
        if mu is None or sigma is None:
            return None
        lo = math.exp(mu - 1.645 * sigma)
        hi = math.exp(mu + 1.645 * sigma)
        return (
            description,
            f"Normal({mu:g}, {sigma:g})",
            f"concentration median {math.exp(mu):.0f}, 5–95% {lo:.0f}–{hi:.0f}; "
            "larger means the four-cell counts cluster more tightly around the "
            "predicted composition",
        )

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



def _subject_scale_row(description: str, spec: Mapping) -> tuple[str, str, str] | None:
    """A priors-table row for a subject-scale field holding a block, not a scalar.

    Both overloads describe the same thing -- how far children sit from the
    population trajectory -- but neither is a single ``HalfNormal``, so the
    scalar path cannot summarise either. Returning ``None`` would silently drop
    the row and leave the reader thinking the model has no between-child prior
    at all, which is worse than a crash; these rows say what was actually put on
    the block.
    """
    if "tau0_sigma" in spec:
        # VG19: a child intercept and rate, correlated. `(rho + 1) / 2 ~
        # Beta(eta, eta)` is exactly LKJ(eta) for a 2x2, so it is named as LKJ.
        t0 = float(spec["tau0_sigma"])
        t1 = float(spec["tau1_sigma"])
        eta = float(spec.get("rho_eta", 2.0))
        m0 = float(stats.halfnorm.ppf(0.5, scale=t0))
        m1 = float(stats.halfnorm.ppf(0.5, scale=t1))
        return (
            f"{description} — intercept and rate",
            f"HalfNormal({t0:g}) and HalfNormal({t1:g}); "
            f"$\\rho_{{01}} \\sim$ LKJ({eta:g})",
            f"median {m0:.2f} logits at the reference age; rate median "
            f"{m1:.2f} logits per year; correlation centred on zero",
        )
    if "log_ratio_sigma" in spec:
        # Proposal A1: one deviate scaled by a curve through two anchor ages.
        young = float(spec["young_sigma"])
        ratio = float(spec["log_ratio_sigma"])
        ages = spec.get("anchor_ages") or ()
        m = float(stats.halfnorm.ppf(0.5, scale=young))
        where = f" at {ages[0]:g} months" if len(ages) else ""
        return (
            f"{description} — age-varying",
            f"HalfNormal({young:g}); $\\log(\\tau_{{old}}/\\tau_{{young}}) "
            f"\\sim$ Normal(0, {ratio:g})",
            f"median {m:.2f} logits{where}; zero widening is the prior centre",
        )
    return None


#: Fitted parameters a priors table is not expected to carry a row for, and why.
#: Checked by :func:`prior_coverage`, which is the graph-to-report contract that
#: VG22's missing factor block escaped: a whole parameter family had no entry in
#: :data:`_PRIOR_SPECS` and nothing said so (issue #273).
#:
#: Each entry is a predicate on the parameter name. The exemptions are the
#: reparameterisation machinery and the derived quantities, never a prior a
#: reader would want and cannot find.
PRIOR_EXEMPTIONS: tuple[tuple[str, str], ...] = (
    (
        "*_raw",
        "non-centred offset: the prior a reader wants is on the scale that "
        "multiplies it, which carries its own row",
    ),
    (
        "*_z",
        "standard-normal deviate of a non-centred block, reported through its "
        "scale",
    ),
    (
        "derived",
        "deterministic function of sampled parameters, not something a prior is "
        "placed on",
    ),
)

#: Deterministics that appear in ``diagnostics.csv`` alongside the sampled
#: parameters. Named explicitly because the diagnostics table does not
#: distinguish the two, and a coverage check that treated every row as a
#: parameter would demand priors for quantities that have none.
#:
#: The trend stems are the straight-line trajectory written on its natural
#: scale: ``slope``/``intercept`` are the logit-scale line implied by the two
#: anchored proportions, and ``ell`` is the length-scale in months implied by
#: ``ell_unit``. All three are functions of priors that carry their own rows,
#: and each appears once per outcome (``slope_u``, ``ell_q``, and the signed
#: tent's ``slope_up_sign`` / ``slope_dn_sign``).
_OUTCOME_SUFFIXES = ("", "_u", "_q", "_s", "_sign")
_TREND_DETERMINISTICS = frozenset(
    f"{stem}{suffix}"
    for stem in ("slope", "intercept", "ell")
    for suffix in _OUTCOME_SUFFIXES
) | {"slope_up_sign", "slope_dn_sign"}

_DERIVED_PREFIXES = (
    "b0_",
    "b1_",
    "delta_",
    "subject_factor_corr",
    "subject_factor_loadings",
)
#: Unconditionally derived: no registered model places a prior on any of these.
#:
#: `tau_subject`, `tau_subj_u`, `tau_subj_q` and `rho_uq` are deliberately
#: **absent**. Each is a sampled parameter in some models and a deterministic in
#: others -- `tau_subject` becomes a function of the variance budget in VG11/VG12,
#: the two child scales become the factor block's level scales in VG22, and
#: `rho_uq` is sampled in VG20 and implied by the loadings in VG22 -- so an
#: unconditional exemption would absorb the loss of a row that other models do
#: need. They are handled by `_prior_rows`'s `inert` set instead, which is
#: computed from the definition and is therefore right per model.
_DERIVED_NAMES = (
    frozenset(
        {
            # exp(log_psi) and exp(log_conc); both priors are placed and
            # reported on the log scale.
            "psi",
            "conc",
        }
    )
    | _TREND_DETERMINISTICS
)


def _is_exempt(parameter: str) -> str | None:
    """The reason ``parameter`` needs no prior row, or ``None`` if it needs one."""
    if parameter.endswith("_raw"):
        return PRIOR_EXEMPTIONS[0][1]
    if parameter.endswith("_z") or "_z_" in parameter or parameter.startswith("z_"):
        return PRIOR_EXEMPTIONS[1][1]
    if parameter in _DERIVED_NAMES or parameter.startswith(_DERIVED_PREFIXES):
        return PRIOR_EXEMPTIONS[2][1]
    return None


def _dispersion_parameters(present: set[str], field: str) -> list[str]:
    """The ``kappa`` parameters one dispersion row accounts for.

    The definition field is ``kappa`` / ``kappa_u`` / ``kappa_s`` / ``kappa_sign``,
    but the graph names carry the outcome as a **suffix** on several different
    stems -- ``kappa_min_u``, ``kappa_excess_young_u``, ``a_kappa_u``,
    ``b_kappa_mag_s``, ``kappa_old_sign`` -- so a prefix match on the field name
    finds none of them. One dispersion row describes that whole
    parameterisation for its outcome, which is what it is a *block* row for.
    """
    suffix = field.removeprefix("kappa")  # "", "_u", "_s", "_sign"
    other = {"_u", "_q", "_s", "_sign"} - {suffix}
    return [
        name
        for name in present
        if "kappa" in name
        and (name.endswith(suffix) if suffix else not name.endswith(tuple(other)))
    ]


def prior_coverage(directory: str = ".") -> dict[str, list[str]]:
    """Which of this fit's parameters the priors table covers, and which it does not.

    Returns ``{"rendered": [...], "exempt": [...], "uncovered": [...]}``. A
    non-empty ``uncovered`` means the rendered table is silently incomplete --
    the VG22 failure, where the four factor scales, the nine loading directions
    and the per-child factor scores had no entry in :data:`_PRIOR_SPECS` and the
    page said only that the table "omits every prior this block adds".

    Pure and cheap: it reads the same manifest and ``diagnostics.csv`` the table
    itself reads, so it can be called from a report cell or a test without a
    model build.
    """
    manifest = read_manifest(directory)
    definition = (manifest.get("model") or {}).get("definition") or {}
    present = fitted_parameters(directory)
    rendered = set(_prior_rows(definition, present)[0])

    covered: list[str] = []
    exempt: list[str] = []
    uncovered: list[str] = []
    for parameter in sorted(present):
        if parameter in rendered:
            covered.append(parameter)
        elif _is_exempt(parameter):
            exempt.append(parameter)
        else:
            uncovered.append(parameter)
    return {"rendered": covered, "exempt": exempt, "uncovered": uncovered}


def _factor_rows(
    definition: dict, present: set[str]
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Rows for VG22's low-rank factor block, and the parameters they cover.

    The block replaces the parent's two child-intercept priors with four scale
    priors, the sampled entries of the raw loading matrix and the per-child
    factor scores. None of them fits :data:`_PRIOR_SPECS`'s
    one-parameter-one-definition-field shape: the loading entries are a family
    whose size depends on ``rank``, and the two rate scales live on the factor
    spec while the two level scales are inherited from the parent's own scalar
    fields. Rendering them by hand in the model's prose is what left VG22's
    table describing a model it was not fitted under.
    """
    spec = definition.get("subject_factor")
    if not isinstance(spec, Mapping):
        return [], []

    rank = int(spec.get("rank", 0))
    rows: list[tuple[str, str, str]] = []
    covered: list[str] = []

    scales = (
        ("tau_subj_u_0", "Between-child SD, understood — level", definition.get("tau_subj_u_sigma")),
        ("tau_subj_u_1", "Between-child SD, understood — rate", spec.get("tau1_u_sigma")),
        ("tau_subj_q_0", "Between-child SD, ratio $q$ — level", definition.get("tau_subj_q_sigma")),
        ("tau_subj_q_1", "Between-child SD, ratio $q$ — rate", spec.get("tau1_q_sigma")),
    )
    reference = spec.get("ref_age_months")
    for parameter, description, sigma in scales:
        if not isinstance(sigma, (int, float)):
            # A level scale that is itself a block belongs to a different
            # structure; say nothing rather than describe the wrong one.
            continue
        if present and parameter not in present:
            continue
        median = float(stats.halfnorm.ppf(0.5, scale=sigma))
        if parameter.endswith("_1"):
            reading = f"median {median:.2f} logits per year of age"
        else:
            where = f" at {reference:g} months" if isinstance(reference, (int, float)) else ""
            reading = f"median {median:.2f} logits{where} (odds ×{math.exp(median):.2f} at +1 SD)"
        rows.append((description, f"HalfNormal({sigma:g})", reading))
        covered.append(parameter)

    # `rho_uq` is sampled directly since #266 finding 5: the first anchor row is
    # exactly e_0, so the second row's first coordinate IS the correlation, and a
    # prior there is a prior on it with no approximation.
    if not present or "rho_uq_raw" in present:
        eta = float(spec.get("rho_uq_eta", 2.0))
        lo = float(stats.beta.ppf(0.05, eta, eta)) * 2.0 - 1.0
        hi = float(stats.beta.ppf(0.95, eta, eta)) * 2.0 - 1.0
        rows.append(
            (
                "Correlation between a child's understood and $q$ levels "
                "$\\rho_{uq}$",
                f"LKJ({eta:g}), i.e. $(\\rho_{{uq}}+1)/2 \\sim$ Beta({eta:g}, {eta:g})",
                "centred on zero and pulled toward it, so a correlation has to be "
                f"evidenced; 5–95% {lo:+.2f} to {hi:+.2f}. The same prior VG20 "
                "places on the same quantity, so the two are comparable",
            )
        )
        covered.extend(["rho_uq_raw", "rho_uq"])
    elif rank == 1:
        # Rank 1 gives every child one deviate scaled four ways, so every
        # implied correlation is exactly +/-1 and no prior over (-1, 1) can be
        # placed on one. Said rather than omitted: a reader who finds `rho_uq`
        # in the diagnostics and no row for it would reasonably assume the table
        # had lost one.
        rows.append(
            (
                "Correlation between a child's understood and $q$ levels "
                "$\\rho_{uq}$",
                "none — $\\pm 1$ by construction at rank 1",
                "one latent dimension makes every child's four effects one "
                "deviate scaled four ways, so the correlations are degenerate "
                "and only their signs are estimated",
            )
        )
        covered.append("rho_uq")

    loadings = sorted(p for p in present if p.startswith("subject_factor_w_"))
    if loadings or not present:
        # What remains after the two leading anchor rows: one direction per
        # non-anchor row, still sampled as entries and normalised. Each such row
        # spends one parameter on a magnitude that cancels, which is stated
        # rather than hidden -- removing them needs a chart on the sphere, and
        # that was measured sampling far worse (#266 finding 5).
        anchors = sum(1 for name in loadings if name.endswith(("22", "32")))
        rows.append(
            (
                f"Loading directions ({len(loadings) or 'rank-dependent'} sampled entries)",
                "HalfNormal(1) on a remaining anchor diagonal, Normal(0, 1) elsewhere",
                f"{anchors} anchor entr{'y' if anchors == 1 else 'ies'} fix the "
                "rotation and each factor's sign; rows are normalised to unit "
                "length, so $\\tau$ carries the whole marginal scale and one "
                "magnitude per row is prior-only",
            )
        )
        covered.extend(loadings)

    if not present or "subject_factor_z" in present:
        rows.append(
            (
                f"Per-child factor scores $z$ (rank {rank})" if rank else "Per-child factor scores $z$",
                "Normal(0, I)",
                "standard normal by construction; the scale lives in the loadings",
            )
        )
        covered.append("subject_factor_z")

    if rows:
        rows.append(
            (
                "The other five implied correlations",
                "induced by the loading directions, uniform on $(-1, 1)$",
                "$\\rho_{uq}$ is designed above; the remaining five are whatever "
                "the unit-row geometry induces, which is the $\\mathrm{LKJ}(1)$ "
                "marginal — flat rather than extreme, and not separately chosen",
            )
        )
    return covered, rows


def _prior_rows(
    definition: dict, present: set[str]
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Every priors-table row for this definition, and the parameters they cover.

    Split out of :func:`render_priors_table` so :func:`prior_coverage` can ask
    what the table would say without printing it.
    """
    covered: list[str] = []
    rows: list[tuple[str, str, str]] = []

    factor_covered, factor_rows = _factor_rows(definition, present)
    # A factor block reports the two level scales under their own names, so the
    # parent's scalar rows would restate them; and under a variance partition the
    # child scale is a deterministic function of the budget, so its own prior
    # never enters the model.
    inert = set(factor_covered)
    if definition.get("subject_variance_partition"):
        inert.add("tau_subject")
    if factor_rows:
        inert.update({"tau_subj_u", "tau_subj_q"})

    for parameter, description, stem, kind in _PRIOR_SPECS:
        if parameter in inert:
            continue
        if present and parameter not in present:
            continue
        row = _prior_row(parameter, description, stem, kind, definition)
        if row is not None:
            rows.append(row)
            covered.append(parameter)
            # A subject-scale field holding a block renders one row for the
            # block, so the row accounts for the block's own parameters. The
            # two blocks name them differently: VG19's child slope emits
            # `{name}_0`, `_1` and `_rho`, while Proposal A1's age-varying scale
            # emits `log_{name}_ratio` -- a prefix rule alone finds only the
            # first.
            if isinstance(definition.get(stem), Mapping):
                covered.extend(
                    name
                    for name in present
                    if name.startswith(f"{parameter}_")
                    or name == f"log_{parameter}_ratio"
                )

    rows.extend(factor_rows)
    covered.extend(factor_covered)
    # An inert parameter is a *positive* statement, not a gap: the model makes it
    # a deterministic function of something else, and the rows that replaced it
    # are in the table. Counting it as covered is what keeps the coverage check
    # from demanding a prior that does not exist.
    covered.extend(name for name in inert if name in present)

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
        covered.extend(_dispersion_parameters(present, field))

    return covered, rows


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
    covered, rows = _prior_rows(definition, present)

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

    # Say so on the page when the table is incomplete. VG22 fitted a factor
    # block this table could not render, and the only record of it was a
    # sentence someone had written by hand into that model's template -- which
    # is the same failure mode as the copied priors this module exists to
    # replace. A gap now announces itself wherever it occurs.
    uncovered = sorted(
        parameter
        for parameter in present
        if parameter not in set(covered) and not _is_exempt(parameter)
    )
    if uncovered:
        print()
        print(
            "::: {.callout-warning title=\"Incomplete priors table\"}\n\n"
            "This fit sampled "
            + ", ".join(f"`{name}`" for name in uncovered)
            + ", for which this table has no row. The priors in force are the "
            "ones the model definition records; the gap is in the table, not in "
            "the fit.\n:::"
        )


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

    # Gated on the fitted parameter set, not the definition, for the same reason
    # the priors table is: the dataclass carries a non-None default for every
    # scale, so VG05 -- which instantiates no study effect at all -- records
    # `tau_u_sigma` and was announcing study random intercepts it does not have,
    # contradicting its own prose two sections later. The univariate and joint
    # engines also name these parameters differently (`tau` / `tau_subject`
    # against `tau_u` / `tau_subj_u`), so a definition-field test missed the
    # hierarchy VG11 and VG12 genuinely do have.
    present = fitted_parameters(directory)
    hierarchy = []
    study_effects = {
        "tau": "study",
        "tau_u": "study, on understood",
        "tau_q": "study, on the production ratio $q$",
        "tau_sign": "study, on signing",
    }
    child_effects = {
        "tau_subject": "child",
        "tau_subj_u": "child, on understood",
        "tau_subj_q": "child, on the production ratio $q$",
        "tau_subj_sign": "child, on signing",
    }
    for group in (study_effects, child_effects):
        found = [label for name, label in group.items() if name in present]
        if found:
            hierarchy.extend(found)
    items.append(
        (
            "Random intercepts",
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

    # Which outcomes this fit actually has, so a comprehension-only model is not
    # told it reports spoken vocabulary. VG02 and VG04 both were: the query
    # grid's maximum was labelled "spoken to N months" unconditionally.
    outcomes = set(data.get("observed_outcome_counts") or {})
    caps = []
    understood_cap = definition.get("report_max_age_understood")
    signed_cap = definition.get("report_max_age_signed")
    if understood_cap and "understood" in outcomes:
        caps.append(f"understood and ratios to {understood_cap:g} months")
    if signed_cap and "signed" in outcomes:
        caps.append(f"signed to {signed_cap:g} months")
    ages_query = definition.get("ages_query")
    if ages_query:
        furthest = max(ages_query)
        if caps and "spoken" in outcomes:
            caps.append(f"spoken to {furthest:g} months")
        elif not caps:
            caps.append(f"to {furthest:g} months")
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


# ---------------------------------------------------------------------------
# Headline quantities
# ---------------------------------------------------------------------------

# Ten independent reviews of the fifteen model reports each reached the same
# conclusion: the reports display their figures and state no results. VG10's
# comprehension-production gap has its own section and its peak is never given;
# VG16 never prints the cross-lag coefficient it exists to estimate.
#
# The remedy has to survive a refit, so these numbers are computed from the
# summary tables the fit already writes rather than typed into the template.
# Ages are read off the median curve and are therefore point readings, not
# posterior medians of the crossing -- the caption says so, because the median
# of a set of crossings is not the crossing of the median and the difference is
# large enough to matter (see notes on the signing milestones, where the two
# differ by months).

_OUTCOME_LABELS = {
    "u": "words understood",
    "s": "words spoken",
    "sign": "words signed",
    None: "words",
}


def _read(directory: str, name: str):
    """A summary CSV, or None when this engine does not write it."""
    import pandas as pd

    path = os.path.join(directory, f"{name}.csv")
    if not os.path.isfile(path):
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    return frame if not frame.empty else None


def _peak_row(frame, column: str):
    """The row where `column` is largest, or None."""
    if frame is None or column not in frame:
        return None
    return frame.loc[frame[column].idxmax()]


def _first_crossing(frame, column: str, threshold: float):
    """First age at which `column` reaches `threshold`, by linear interpolation."""
    if frame is None or column not in frame or "age_months" not in frame:
        return None
    values = frame[column].to_numpy()
    ages = frame["age_months"].to_numpy()
    above = values >= threshold
    if not above.any() or above[0]:
        return None
    i = int(above.argmax())
    v0, v1 = values[i - 1], values[i]
    if v1 == v0:
        return float(ages[i])
    return float(ages[i - 1] + (threshold - v0) * (ages[i] - ages[i - 1]) / (v1 - v0))


def render_headline_quantities(directory: str = ".") -> None:
    """Print the model's headline quantities as a table computed from its own tables.

    Silent about anything this engine did not write, so a univariate model does
    not advertise a production ratio and a model without random effects does not
    report a between-study scale.
    """
    rows: list[tuple[str, str, str]] = []

    for suffix in (None, "u", "s", "sign"):
        stem = "expected_learning_rate" if suffix is None else f"expected_learning_rate_{suffix}"
        frame = _read(directory, stem)
        peak = _peak_row(frame, "median_rate")
        if peak is None:
            continue
        label = _OUTCOME_LABELS[suffix]
        # A maximum on the first or last grid age says only that the peak was
        # not located *within* the reported range — the true peak lies at or
        # beyond its edge — so it must not be reported as "fastest growth at
        # age X" (#234).
        peak_position = frame["median_rate"].idxmax()
        at_boundary = peak_position in (frame.index[0], frame.index[-1])
        peak_info = _read(directory, f"{stem}_peak")
        if at_boundary:
            rows.append(
                (
                    f"Growth rate in {label}",
                    f"highest estimated rate ({peak['median_rate']:.1f} "
                    f"words/month) occurs at the boundary of the reported "
                    f"range ({peak['age_months']:.0f} months); the peak age is "
                    "not located within it",
                    f"rate {peak['ci_lo']:.1f} – {peak['ci_hi']:.1f}",
                )
            )
        elif peak_info is not None and "peak_age_median_months" in peak_info:
            info = peak_info.iloc[0]
            share = float(info.get("boundary_draw_share", 0.0))
            share_note = (
                f"; {share:.0%} of draws peak at the range edge" if share >= 0.05 else ""
            )
            rows.append(
                (
                    f"Fastest growth in {label}",
                    f"{peak['median_rate']:.1f} words/month around "
                    f"{info['peak_age_median_months']:.0f} months",
                    f"peak age {info['peak_age_ci_lo_months']:.0f} – "
                    f"{info['peak_age_ci_hi_months']:.0f} months; rate "
                    f"{peak['ci_lo']:.1f} – {peak['ci_hi']:.1f}{share_note}",
                )
            )
        else:
            # A fit predating the draw-wise peak table: the age is read off the
            # median curve and carries no location uncertainty.
            rows.append(
                (
                    f"Fastest growth in {label}",
                    f"{peak['median_rate']:.1f} words/month at "
                    f"{peak['age_months']:.0f} months (age read off the median "
                    "curve; refit for peak-age uncertainty)",
                    f"rate {peak['ci_lo']:.1f} – {peak['ci_hi']:.1f}",
                )
            )

    gap = _read(directory, "comprehension_production_gap")
    peak = _peak_row(gap, "gap_median")
    if peak is not None:
        rows.append(
            (
                "Largest gap between words understood and spoken",
                f"{peak['gap_median']:.0f} words at {peak['age_months']:.0f} months",
                f"{peak['ci_lo']:.0f} – {peak['ci_hi']:.0f}",
            )
        )

    q = _read(directory, "posterior_summary_q")
    crossing = _first_crossing(q, "q_median", 0.5)
    if crossing is not None:
        rows.append(
            (
                "Age at which half the understood words are also spoken",
                f"{crossing:.0f} months",
                "read off the median curve",
            )
        )

    # "Between administrations", not "between children": the Beta-Binomial
    # kappa is count dispersion at an age. Where the outcome's mean carries no
    # child effect it is marginal — it mixes between-child, between-study and
    # occasion variation; where the mean is fully conditioned on child effects
    # it is the residual within-child spread, and labelling it as the spread
    # across (different children's) administrations would hand it tau_subject's
    # job. Neither is a between-child quantity (#234, #240). The check is per
    # outcome because the conditioning is: the nested spoken mean is p_u * q,
    # so its kappa is within-child only when both sides carry a child effect
    # (VG08 carries one on understood alone, and its kappa_s keeps the
    # marginal-style label as the conservative reading).
    present = fitted_parameters(directory)
    child_scales_for_suffix = {
        None: {"tau_subject"},
        "u": {"tau_subj_u"},
        "s": {"tau_subj_u", "tau_subj_q"},
        "sign": {"tau_subj_sign"},
    }
    for suffix in (None, "u", "s", "sign"):
        stem = "posterior_kappa" if suffix is None else f"posterior_kappa_{suffix}"
        kappa = _read(directory, stem)
        if kappa is None or "vif_median" not in kappa:
            continue
        label_stem = (
            "Within-child spread between same-age administrations"
            if child_scales_for_suffix[suffix] <= present
            else "Spread across same-age administrations"
        )
        label = f"{label_stem}, {_OUTCOME_LABELS[suffix]}"
        trend = _read(directory, f"{stem}_trend")
        if trend is not None and "p_widens" in trend:
            info = trend.iloc[0]
            p_widens = float(info["p_widens"])
            if p_widens >= 0.95:
                direction = "widens"
            elif p_widens <= 0.05:
                direction = "narrows"
            else:
                direction = "does not change clearly"
            rows.append(
                (
                    label,
                    f"{direction} with age "
                    f"({info['vif_young_median']:.0f}× the Binomial variance at "
                    f"{info['age_young_months']:.0f} months, "
                    f"{info['vif_old_median']:.0f}× at "
                    f"{info['age_old_months']:.0f})",
                    f"ratio ×{info['vif_ratio_median']:.2f} "
                    f"({info['vif_ratio_ci_lo']:.2f} – "
                    f"{info['vif_ratio_ci_hi']:.2f}); "
                    f"P(widens) = {p_widens:.2f}",
                )
            )
        else:
            # A fit predating the draw-wise endpoint contrast: the direction is
            # a comparison of two plug-in medians and carries no uncertainty.
            first, last = kappa.iloc[0], kappa.iloc[-1]
            direction = (
                "widens" if last["vif_median"] > first["vif_median"] else "narrows"
            )
            rows.append(
                (
                    label,
                    f"{direction} with age "
                    f"({first['vif_median']:.0f}× the Binomial variance at "
                    f"{first['age_months']:.0f} months, "
                    f"{last['vif_median']:.0f}× at {last['age_months']:.0f})",
                    "read off the median curve; refit for a draw-wise contrast",
                )
            )

    if not rows:
        print("_This fit writes none of the summary tables this block reads._")
        return

    print("| Quantity | Estimate | 89% interval |")
    print("| --- | --- | --- |")
    for label, estimate, interval in rows:
        print(f"| {label} | {estimate} | {interval} |")
    print()
    print(
        ": Computed at render time from this fit's own summary tables, so these "
        "figures cannot drift from the model they describe. Where a draw-wise "
        "summary exists (peak age, dispersion ratio) its posterior interval is "
        "shown; crossing ages are still read off the median curve rather than "
        "being posterior medians of the crossing, and rate intervals are on the "
        "quantity at the stated age."
    )


def _child_slope_blocks(frame, manifest: dict) -> list[tuple[str, float, float, float]]:
    """The (name, tau0, tau1, rho01) of each child intercept-and-rate block present.

    VG19 and VG22 keep the constant-offset names ``tau_subj_u`` / ``tau_subj_q``
    as deterministic aliases for ``tau0`` -- the between-child scale **at the
    reference age** -- so that every consumer written against VG10 keeps working
    (``gp_utils.build_child_slope``, ``build_child_factor``). The cost is that a
    table which prints the alias under the label "Between children, understood"
    is describing one age and implying every age, which is what #233 flagged:
    under a rate the between-child SD is age-varying by construction.

    Only blocks whose within-outcome correlation is a *named scalar* qualify.
    VG19 emits ``{name}_rho``; VG22's factor form carries the same quantity as
    an element of the ``subject_factor_corr`` matrix, and its element naming in
    ``diagnostics.csv`` is not relied on here -- those models get the relabelled
    alias without the age table rather than a scale computed from a guess.
    """
    column = next((c for c in ("mean", "Mean", "median") if c in frame.columns), None)
    if column is None:
        return []
    blocks = []
    for name in ("tau_subject", "tau_subj_u", "tau_subj_q"):
        needed = (f"{name}_0", f"{name}_1", f"{name}_rho")
        if not all(key in frame.index for key in needed):
            continue
        blocks.append((
            name,
            float(frame.loc[f"{name}_0", column]),
            float(frame.loc[f"{name}_1", column]),
            float(frame.loc[f"{name}_rho", column]),
        ))
    return blocks


def _slope_scale_ages(manifest: dict, ref_age: float) -> list[float]:
    """Ages to evaluate an age-varying child scale at, inside the reporting cap.

    Both scales here belong to comprehension or to a ratio of it, so both take
    the comprehension cap; reporting either past it would quote a between-child
    spread at an age the model declines to report a mean for.
    """
    definition = (manifest.get("model") or {}).get("definition") or {}
    ages = [float(a) for a in (definition.get("ages_query") or [])]
    if not ages:
        return []
    cap = definition.get("report_max_age_understood")
    if cap is not None:
        ages = [a for a in ages if a <= float(cap)]
    if not ages:
        return []
    ages = sorted(set(ages))
    lo, hi = ages[0], ages[-1]
    # Four evenly spaced grid ages plus the reference age, which is where tau0
    # is read and so has to appear even when the spacing would miss it.
    n = len(ages)
    span = max(n - 1, 1)
    picked = {ages[round(i * span / 3)] for i in range(4)}
    if lo <= float(ref_age) <= hi:
        picked.add(float(ref_age))
    return sorted(picked)


def render_variation_table(directory: str = ".") -> None:
    """Print the fitted random-effect scales, with an odds reading.

    Answers the question the hierarchical models exist to answer -- how much do
    children differ, and how much do studies -- which no report currently states.

    Under a child intercept-and-rate block the between-child scale is not one
    number, so the alias row is labelled with the age it refers to and a second
    table gives the scale across the reported ages (#233).
    """
    import pandas as pd

    path = os.path.join(directory, "diagnostics.csv")
    if not os.path.isfile(path):
        return
    try:
        frame = pd.read_csv(path, index_col=0)
    except (OSError, ValueError):
        return

    labels = {
        "tau": "Between studies",
        "tau_u": "Between studies, understood",
        "tau_q": "Between studies, production ratio $q$",
        "tau_sign": "Between studies, signing",
        "tau_subject": "Between children",
        "tau_subj_u": "Between children, understood",
        "tau_subj_q": "Between children, production ratio $q$",
        "tau_subj_sign": "Between children, signing",
        "tau_psi": "Between studies, sign–speech association",
    }
    column = next((c for c in ("mean", "Mean", "median") if c in frame.columns), None)
    if column is None:
        return

    manifest = read_manifest(directory)
    definition = (manifest.get("model") or {}).get("definition") or {}
    ref_age = definition.get("subject_slope_ref_age_months")
    slope_names = {
        name for name in labels if f"{name}_1" in frame.index and f"{name}_0" in frame.index
    }
    if slope_names and ref_age is not None:
        for name in slope_names:
            labels[name] = f"{labels[name]} (at {float(ref_age):g} months)"

    rows = [
        (label, float(frame.loc[name, column]))
        for name, label in labels.items()
        if name in frame.index
    ]
    if not rows:
        return

    print("| Source of variation | SD (logits) | A child or study 1 SD above average |")
    print("| --- | --- | --- |")
    for label, value in sorted(rows, key=lambda r: -r[1]):
        print(f"| {label} | {value:.2f} | ×{math.exp(value):.1f} the odds |")
    print()
    caption = (
        ": Posterior means of the random-effect scales, read from `diagnostics.csv`. "
        "Larger means that group differs more from the population average."
    )
    if slope_names:
        caption += (
            " This model gives each child a **rate** as well as an offset, so its "
            "between-child scales are the spread at the reference age only — they "
            "are aliases for $\\tau_0$, not a single spread that holds at every age. "
            "The table below gives the rest."
        )
    print(caption)

    blocks = _child_slope_blocks(frame, manifest)
    if not blocks or ref_age is None:
        if slope_names and not blocks:
            print()
            print(
                "_The age-varying between-child scale is not tabulated for this "
                "model: it needs the within-outcome offset-rate correlation as a "
                "named scalar, and this fit records it only inside a correlation "
                "matrix._"
            )
        return

    ages = _slope_scale_ages(manifest, float(ref_age))
    if not ages:
        return

    print()
    print("| Source of variation | " + " | ".join(f"{a:g} mo" for a in ages) + " |")
    print("| --- |" + " --- |" * len(ages))
    for name, tau0, tau1, rho01 in blocks:
        cells = []
        for age in ages:
            d = (age - float(ref_age)) / 12.0
            variance = tau0**2 + 2.0 * rho01 * tau0 * tau1 * d + (tau1 * d) ** 2
            cells.append(f"{math.sqrt(max(variance, 0.0)):.2f}")
        label = labels[name].split(" (at ")[0]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print()
    print(
        ": Between-child SD in logits at each age, as "
        "$\\sqrt{\\tau_0^2 + 2\\rho_{01}\\tau_0\\tau_1 D + \\tau_1^2 D^2}$ with $D$ the "
        f"distance from the {float(ref_age):g}-month reference age in years. **Plug-in, "
        "not a posterior summary**: it is evaluated at the posterior means of "
        "$\\tau_0$, $\\tau_1$ and $\\rho_{01}$ rather than draw by draw, so it "
        "carries no interval and is not the posterior mean of the SD. Ages stop at "
        "the comprehension reporting cap, because both scales belong to "
        "comprehension or to a ratio of it."
    )


def render_loo_section(directory: str = ".") -> None:
    """Print the leave-one-out cross-validation result for a report cell.

    Every fit computed this and printed it to the console only, while the
    predictive-calibration section of every report told the reader that
    leave-one-out is the out-of-sample counterpart to its in-sample checks. The
    number they were sent to find was not in the output directory at all.

    The wording above the table branches on how many rows it has, because the
    unit being held out is not the same in the two cases (issue #266, finding
    4). A univariate fit has one unnamed likelihood over administration rows, so
    the estimate really is leave-one-administration-out. A multi-outcome fit has
    one likelihood term per outcome and the engines compute a separate LOO for
    each, so a row holds out one *term*, not an administration -- and because
    the expressive likelihoods take the same administration's observed
    comprehension count as their trial count, neither the held-out spoken score
    nor the held-out understood score is free of that row's observed
    comprehension. Printing the administration wording over such a table told
    the reader the estimate was something it is not.

    Prints an explanatory line rather than failing when the fit predates the
    table, so the section is never silently empty -- the same contract
    :func:`vocab_growth.models.calibration.render_calibration_section` keeps.
    """
    import pandas as pd

    path = os.path.join(directory, "loo_summary.csv")
    if not os.path.isfile(path):
        print(
            "_No leave-one-out summary for this fit (`loo_summary.csv` absent — "
            "it was added on 2026-08-16, so fits made before then need a refit "
            "to produce it)._"
        )
        return
    try:
        table = pd.read_csv(path)
    except (OSError, ValueError):
        print("_The leave-one-out summary for this fit could not be read._")
        return
    if table.empty:
        print("_The leave-one-out summary for this fit is empty._")
        return

    parameters = fitted_parameters(directory)
    # One row means one unnamed likelihood over administration rows; more than
    # one means the engine computed a separate LOO per outcome likelihood, which
    # is a different held-out unit and has to be described as one.
    scale_sentences = (
        "`elpd_loo` is that estimate on the log scale: **higher is better**, "
        "and it is only meaningful when compared with another model fitted to "
        "the same observations — it has no absolute interpretation on its own. "
        "`p_loo` is the effective number of parameters, a measure of how much "
        "flexibility the fit is using.\n"
    )
    if len(table) <= 1:
        print(
            "Leave-one-out cross-validation estimates how well this model would "
            "predict an observation it had not seen. An observation here is a single "
            "**administration** of a checklist — repeated administrations of the "
            "same child are separate observations — so this is "
            "leave-one-administration-out: it scores prediction of another "
            "administration like those in the frame, possibly from a child or study "
            "the model has already seen, not generalisation to a new child or "
            "study. " + scale_sentences
        )
    else:
        # `emit_loo_summary` writes the label into an `outcome` column -- the
        # name predates there being anything in the table that is not one.
        has_administration_row = ADMINISTRATION_LABEL in set(
            table.get("outcome", pd.Series(dtype=str)).astype(str)
        )
        print(
            "Leave-one-out cross-validation estimates how well this model would "
            "predict an observation it had not seen. This model carries a separate "
            "likelihood term for each outcome, and the table below reports a "
            "separate estimate for each of them, so **each per-outcome row is "
            "leave-one-likelihood-term-out for that outcome, not "
            "leave-one-administration-out**. The difference is not a technicality "
            "here, because the expressive outcomes are nested inside "
            "comprehension: the trial count of the spoken (and, where present, "
            "signed) likelihood on a row *is* that administration's observed "
            "words-understood count. Holding out a spoken term therefore scores "
            "prediction of the spoken count **conditional on the same "
            "administration's observed comprehension**, which is a conditional "
            "estimand rather than the prediction of a withheld administration. "
            "And on a paired row, holding out the understood term leaves that "
            "same observed comprehension count in the spoken term's denominator, "
            "so the held-out value has not left the conditioning set and the two "
            "rows are not independent held-out units. Comparing models on these "
            "per-outcome numbers is still sound — every model is scored on the "
            "same conditional units, computed the same way — but they must not be "
            "read as whole-administration predictive accuracy. " + scale_sentences
        )
        if has_administration_row:
            print(
                f"The **{ADMINISTRATION_LABEL}** row is the one that can be: it "
                "sums every likelihood factor belonging to one administration "
                "into a single held-out case, so a paired administration is one "
                "observation with one importance weight rather than two, and "
                "nothing of the held-out row remains in the conditioning set. "
                "Repeated administrations of the same child are still separate "
                "cases, so it scores prediction of another administration like "
                "those in the frame — not generalisation to a new child, which "
                "is what grouped leave-one-subject-out answers. It is the row to "
                "read as this model's predictive accuracy, and the one to compare "
                "against another model of the same administrations.\n"
            )
        else:
            print(
                "This fit carries **no administration-level row**: it was made "
                "before that score was computed (issue #266, finding 4), so the "
                "conditional rows above are all it has. A refit produces one.\n"
            )
        if {"psi", "conc"} <= parameters:
            # The joint sign/speech engine is the only one that adds
            # Dirichlet-Multinomial composition terms, and `psi` (the Plackett
            # odds ratio) with `conc` (the composition concentration) is the
            # pair only it samples, so their presence identifies the model
            # without the report cell having to be told which model it is in.
            composition = (
                "Two further likelihood terms — the within-understood four-cell "
                "composition and the within-produced three-cell composition — are "
                "excluded from every **per-outcome** row above, because a "
                "Dirichlet-Multinomial over cells is not the per-observation "
                "Beta-Binomial the other terms are. Those are precisely the terms "
                "that identify the sign–speech association $\\psi$."
            )
            if has_administration_row:
                print(
                    composition
                    + " They **are** included in the administration row, which is "
                    "why that row is the only one that scores $\\psi$ at all: a "
                    "cross-tabulation row's composition factor is summed into the "
                    "same held-out case as its words-understood factor, rather "
                    "than left in the conditioning set.\n"
                )
            else:
                print(
                    composition
                    + " so **$\\psi$ is not scored by leave-one-out at all** in "
                    "this fit. A cross-tabulation row does still contribute its "
                    "words-understood term, so the words-understood row holds out "
                    "one of that row's two factors while leaving its composition "
                    "factor in the conditioning set.\n"
                )

    display = table.copy()
    for column in ("elpd_loo", "se", "p_loo", "good_k_threshold"):
        if column in display:
            display[column] = display[column].astype(float).round(2)
    print(display.to_markdown(index=False))
    print()

    # The estimate is not usable without its reliability diagnostic, so the
    # verdict is printed rather than left for the reader to derive.
    bad = int(table.get("pareto_k_bad", pd.Series(dtype=int)).fillna(0).sum())
    very_bad = int(table.get("pareto_k_very_bad", pd.Series(dtype=int)).fillna(0).sum())
    total = int(table.get("n_data_points", pd.Series(dtype=int)).fillna(0).sum())
    print(
        "The Pareto $k$ counts say whether that estimate can be trusted. "
        "Leave-one-out here is approximated by importance sampling rather than "
        "by refitting the model without each observation in turn, and the "
        "approximation degrades for observations the model finds surprising. "
        "Counts are against the threshold in the table, which ArviZ sets from "
        "the sample size.\n"
    )
    if bad == 0 and very_bad == 0:
        print(
            f"For this fit **every one of the {total:,} observations is within "
            "the threshold**, so the estimate above is reliable."
        )
    else:
        share = (bad + very_bad) / total if total else 0.0
        print(
            f"For this fit **{bad + very_bad} of {total:,} observations "
            f"({share:.0%}) exceed the threshold** ({very_bad} of them above 1), "
            "so the estimate above should be treated as indicative rather than "
            "as a precise quantity."
        )
        # Why, rather than only that. A high share here is the expected
        # signature of leaving out one *observation* from a model carrying
        # per-child effects, not evidence that the model fits badly -- and the
        # distinction decides what the reader should do about it.
        if parameters & {
            "tau_subject",
            "tau_subj_u",
            "tau_subj_q",
            "tau_subj_sign",
        }:
            print()
            print(
                "A large share is expected here and is not by itself evidence "
                "of misfit. This model gives each child their own random "
                "intercept, and a child's intercept is informed mostly by that "
                "child's own observations — so removing one observation can move "
                "the posterior substantially, which is precisely the situation "
                "importance sampling approximates poorly. Leave-one-observation-out "
                "is the wrong unit of prediction for a model with per-child "
                "parameters. The question it half-answers — how well does this "
                "generalise beyond the data it saw — is better put to the "
                "project's leave-one-study-out and k-fold checks "
                "(`scripts/kfold_loso.py`, `scripts/loso_compare.py`), which hold "
                "out whole studies or whole children rather than single rows."
            )

    dropped = int(
        table.get("n_dropped_degenerate", pd.Series(dtype=int)).fillna(0).sum()
    )
    if dropped:
        print()
        print(
            f"{dropped} observation(s) were excluded as degenerate — their "
            "pointwise log-likelihood is constant across draws, contributing "
            "nothing to the estimate. The exclusion is deterministic, so the "
            "per-outcome values stay comparable across models."
        )


# ---------------------------------------------------------------------------
# Reader-facing blocks (2026-09-02 template review)
# ---------------------------------------------------------------------------
#
# A review of all twenty templates against this run's fitted output found the
# same shape of gap on every page: the numbers a family or a practitioner would
# use exist in the summary CSVs and are shown as raw DataFrames, the checks a
# researcher would want are delegated to the reader ("compare each posterior
# with its prior figure earlier in this report"), and no page says which of
# the report's three audiences it serves. Each block below turns one of those
# into something computed from the fit on disk. All are fail-soft: a fit that
# predates an artefact gets a sentence saying so, never an empty section.

#: How a report describes its own role. Owner decisions live in
#: `docs/models/README.md`; a template passes the role it already states.
READING_ROLES = frozenset({"record", "reference", "candidate", "development"})


def _has_child_effects(present: set[str]) -> bool:
    return any(
        name in present
        for name in ("tau_subject", "tau_subj_u", "tau_subj_q", "tau_subj_sign", "tau_subj_u_0")
    ) or "subject_variance_share" in present


def render_reading_routes(
    role: str,
    *,
    instead: str | None = None,
    joint: bool = False,
    signing: bool = False,
    robustness: bool = False,
    recovery: bool = False,
    loo: bool = True,
) -> None:
    """Print a "three ways to read this page" callout for the report's audiences.

    The technical report says it is written for families, practitioners and
    researchers, and each model page renders standalone, so a reader who lands
    on one gets no other routing. ``role`` is the page's own stated role;
    a development step or candidate routes every non-research reader to the
    model of record named in ``instead`` rather than offering them numbers a
    superseded page should not supply.
    """
    if role not in READING_ROLES:
        raise ValueError(f"unknown reading role {role!r}; expected one of {sorted(READING_ROLES)}")

    print('::: {.callout-tip title="Three ways to read this page"}')
    print()
    if role in {"development", "candidate"}:
        what = (
            "a development step" if role == "development" else "a candidate that is not yet a model of record"
        )
        target = f" **{instead}**" if instead else " the model of record named in the inventory"
        print(
            f"This page describes {what}. It is written for **researchers** following how the "
            f"model family was built. **Families and practitioners** should read{target} instead: "
            "nothing on this page is a reported number, and its own callouts say so."
        )
        print()
        print(":::")
        return

    reference = role == "reference"
    if reference:
        print(
            "**Families.** This is a typically-developing *reference* curve — a comparison "
            "scale for the Down syndrome pages, not a target any child should meet. If you "
            "are here from one of those pages, the [expectations table](#sec-predictions) "
            "shows the same quantities on the same scale; the note "
            "[Reading this page for one child](#sec-one-child) says what the ranges mean."
        )
    else:
        print(
            "**Families.** Start with the [expectations tables](#sec-predictions): expected "
            "words at each age, and the range a single child might fall in. Then read "
            "[Reading this page for one child](#sec-one-child), which says what those "
            "ranges do and do not mean for the child you know. Everything else on this "
            "page is how the numbers were made."
        )
    print()
    practitioner = [
        "**Practitioners.** The [expectations tables](#sec-predictions) carry the "
        "single-child ranges and the probability of scoring at or below common "
        "thresholds"
    ]
    if joint:
        practitioner.append(
            "; [Receptive to expressive](#sec-spoken-given-understood) maps a child's "
            "words understood to the words spoken the model expects"
        )
    if signing:
        practitioner.append(
            "; [What signing is worth to the child](#what-signing-is-worth-to-the-child) gives the "
            "sign-only vocabulary and what counting sign adds"
        )
    practitioner.append(
        ". The [monthly tables](#sec-monthly) show, in their `n_obs` column, where the "
        "evidence runs out."
    )
    print("".join(practitioner))
    print()
    researcher = [
        "**Researchers.** [Priors](#sec-priors) (the table is read from the fit's manifest), "
        "[Diagnostics](#sec-diagnostics) with the rendered verdict and the prior-to-posterior "
        "contraction table"
    ]
    if loo:
        researcher.append(", [Out-of-sample prediction](#sec-loo)")
    if robustness:
        researcher.append(", [Robustness](#sec-robustness)")
    if recovery:
        researcher.append(" and parameter recovery")
    researcher.append(", then [Limits](#sec-limits).")
    print("".join(researcher))
    print()
    print(":::")


def render_family_notes(directory: str = ".") -> None:
    """Print the three things a non-specialist needs before reading any table.

    What a count is (a parent-report checklist harmonised to the reference
    inventory), what the spread means for one child, and that nobody is being
    ranked. The typically-developing pages already carried the last of these
    ("a reference, not a target"); the Down syndrome pages carried none.
    """
    manifest = read_manifest(directory)
    definition = (manifest.get("model") or {}).get("definition") or {}
    population = definition.get("population")
    n_trials = definition.get("n_trials")
    present = fitted_parameters(directory)
    child = _has_child_effects(present)

    inventory = f"the {n_trials:,}-word reference inventory" if n_trials else "a common reference inventory"
    # An id on a callout is read by Quarto as a cross-reference and must carry
    # one of its own prefixes (`nte-`, `wrn-`, ...); anything else is a fatal
    # filter error. The anchor the reading routes link to therefore goes on a
    # plain wrapping div, and the callout inside it stays unlabelled.
    print("::: {#sec-one-child}")
    print('::: {.callout-note title="Reading this page for one child"}')
    print()
    print(
        f"**What a count is.** Every number here is a parent- or carer-reported checklist "
        f"count, placed on {inventory} so that different studies' checklists can be "
        "compared. No child was given that whole inventory; it is a unit of measurement, "
        "not a test they sat."
    )
    print()
    if child:
        print(
            "**What the spread means.** Two kinds of range appear below. The narrower one is "
            "uncertainty about the *typical* child's curve. The wider one — the single-child "
            "range — is where individual children actually fall, and it is the one to read for "
            "a child you know. Children of the same age differ from one another far more than "
            "the average curve moves between ages, so a child outside the inner range is not "
            "unusual, and a child outside the outer range is uncommon rather than wrong."
        )
    else:
        print(
            "**What the spread means.** The ranges below describe where one more checklist "
            "from this pooled sample might fall. This model does not separate differences "
            "between children from differences between studies or between visits, so read "
            "the width as the spread of administrations, not as a clean statement about "
            "how much children differ."
        )
    print()
    if population == "td":
        print(
            "**Nobody is being ranked.** This is a typically-developing *reference*, not a "
            "target — what a different population did. It is not a norm any child is "
            "expected to meet and not a goal for an individual child; a gap between the two "
            "curves is a description, not a deficit to be closed on a timetable."
        )
    else:
        print(
            "**Nobody is being ranked.** These curves describe a pooled research sample of "
            "children with Down syndrome from several countries and studies — not the "
            "population, and not a norm. A child below the curve is not behind a standard; "
            "a child above it is not ahead of one. The curves say what these children did, "
            "with the range that came with it."
        )
    print()
    print(":::")
    print(":::")


def _first_present(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def render_expectations_table(
    outcome: str | None = None,
    *,
    directory: str = ".",
    thresholds: tuple[int, ...] = (0, 10, 50),
) -> None:
    """Print expected words at the reported ages in plain-language columns.

    The summary CSVs carry up to forty columns and the templates displayed them
    as raw DataFrames — float64 to sixteen places under ``P(Y<=5)``. This reads
    the same file and prints the columns a reader asks for: expected words, the
    single-child ranges, the probability of scoring at or below a threshold,
    and the number of nearby observations from the monthly table.

    ``outcome`` is the engine's suffix: ``None`` for a single-outcome fit,
    ``"u"``, ``"s"`` or ``"sign"`` otherwise. Columns are resolved by name, so an
    engine that writes ``Ey_u_median`` (the joint modality engine) is read the
    same way as one that writes ``Ey_median``; a column the engine does not
    write is left out and the caption says so.
    """
    suffix = "" if outcome is None else f"_{outcome}"
    summary = _read(directory, f"posterior_summary{suffix}")
    if summary is None or "age_months" not in summary.columns:
        print(
            f"_No `posterior_summary{suffix}.csv` for this fit, so the expectations table "
            "cannot be shown._"
        )
        return
    monthly = _read(directory, f"posterior_summary_monthly{suffix}")
    columns = list(summary.columns)
    label = _OUTCOME_LABELS.get(outcome, "words")

    expected = _first_present(
        columns, ["Ey_population_median", f"Ey{suffix}_median", "Ey_median"]
    )
    population = expected == "Ey_population_median"
    inner = (_first_present(columns, ["Y_ci50_lo"]), _first_present(columns, ["Y_ci50_hi"]))
    outer = (_first_present(columns, ["Y_ci_lo"]), _first_present(columns, ["Y_ci_hi"]))
    subject_marginal = all(inner) and all(outer)
    if not subject_marginal:
        # The joint modality engine writes no predictive count columns for its
        # per-outcome tables; fall back to the expected-count interval and say so.
        inner = (
            _first_present(columns, [f"Ey{suffix}_ci50_lo", "Ey_ci50_lo"]),
            _first_present(columns, [f"Ey{suffix}_ci50_hi", "Ey_ci50_hi"]),
        )
        outer = (
            _first_present(columns, [f"Ey{suffix}_ci_lo", "Ey_ci_lo"]),
            _first_present(columns, [f"Ey{suffix}_ci_hi", "Ey_ci_hi"]),
        )
    threshold_columns = []
    for k in thresholds:
        name = "P(Y=0)" if k == 0 else f"P(Y<={k})"
        if name in columns:
            threshold_columns.append((k, name))

    n_obs_by_age: dict[int, int] = {}
    last_observed = None
    if monthly is not None and {"age_months", "n_obs"} <= set(monthly.columns):
        for age, n in zip(monthly["age_months"], monthly["n_obs"], strict=True):
            try:
                n_obs_by_age[int(round(float(age)))] = int(n)
            except (TypeError, ValueError):
                continue
        observed = [a for a, n in n_obs_by_age.items() if n > 0]
        last_observed = max(observed) if observed else None

    header = ["Age (months)", f"Expected {label}"]
    if all(inner):
        header.append("Half of children" if subject_marginal else "50% interval")
    if all(outer):
        header.append("Nine in ten children" if subject_marginal else "89% interval")
    for k, _ in threshold_columns:
        header.append("P(no words)" if k == 0 else f"P(≤{k} words)")
    if n_obs_by_age:
        header.append("Observations nearby")

    print("| " + " | ".join(header) + " |")
    print("|" + " ---: |" * len(header))
    beyond = False
    for _, row in summary.iterrows():
        try:
            age = int(round(float(row["age_months"])))
        except (TypeError, ValueError):
            continue
        n_here = n_obs_by_age.get(age)
        extrapolated = last_observed is not None and age > last_observed
        beyond = beyond or extrapolated
        cells = [f"{age}†" if extrapolated else f"{age}"]
        cells.append(f"{float(row[expected]):.0f}" if expected else "—")
        if all(inner):
            cells.append(f"{float(row[inner[0]]):.0f}–{float(row[inner[1]]):.0f}")
        if all(outer):
            cells.append(f"{float(row[outer[0]]):.0f}–{float(row[outer[1]]):.0f}")
        for _, name in threshold_columns:
            cells.append(f"{100 * float(row[name]):.0f}%")
        if n_obs_by_age:
            cells.append("—" if n_here is None else f"{n_here:,}")
        print("| " + " | ".join(cells) + " |")
    print()

    caption = [": "]
    if population:
        caption.append(
            f"Expected {label} is the **population-level** median — the typical child in the "
            "average study, with study and child effects at zero. "
        )
    else:
        caption.append(f"Expected {label} is the median of the fitted trajectory. ")
    if subject_marginal:
        caption.append(
            "The two ranges are **single-child** ranges: where half, and where nine in ten, "
            "of individual children of that age are expected to fall, counting both how much "
            "children differ and the uncertainty in the curve. "
        )
    elif all(outer):
        caption.append(
            "This engine writes no single-child predictive columns for this outcome, so the "
            "ranges are 50% and 89% intervals on the *expected* count — uncertainty about the "
            "curve, not the spread of children, and narrower than a single-child range would be. "
        )
    if threshold_columns:
        caption.append(
            "Threshold columns give the probability that a single child scores at or below "
            "that count. "
        )
    if n_obs_by_age:
        caption.append(
            "*Observations nearby* is the number of administrations within the monthly "
            "window around that age, read from the monthly table — the honest guide to where "
            "these expectations rest on data. "
        )
    if beyond and last_observed is not None:
        caption.append(
            f"† Ages above {last_observed} months lie beyond the last observation for this "
            "outcome: those rows are the model extrapolating, not evidence."
        )
    print("".join(caption).rstrip())


def render_diagnostic_verdict(directory: str = ".") -> None:
    """Print the gate result as one table and one sentence.

    The caveats block discloses exceptions and the styled table colours cells,
    but a reader has to scan thirty rows to learn the worst R-hat, the smallest
    effective sample size, the divergence count and the minimum BFMI, and no
    page states the sampling effort beyond chains and draws. This reads
    ``diagnostics_summary.json``, the manifest's sampling parameters and
    ``diagnostics.csv`` (for which parameter set each extreme) and says it.
    """
    path = os.path.join(directory, "diagnostics_summary.json")
    if not os.path.isfile(path):
        print("_No `diagnostics_summary.json` for this fit, so the gate verdict cannot be shown._")
        return
    try:
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, ValueError):
        print("_The diagnostics summary for this fit could not be read._")
        return

    thresholds = summary.get("thresholds") or {}
    rhat_max = thresholds.get("rhat_max", 1.01)
    ess_min = thresholds.get("ess_threshold", 400)
    bfmi_min = thresholds.get("bfmi_threshold", 0.3)
    checks = summary.get("checks") or {}

    max_rhat = summary.get("max_rhat")
    min_ess = summary.get("min_ess")

    # The gate screens every parameter, including the per-child and per-study
    # random-effect elements the diagnostics table does not list. The table's
    # own extreme is named only when it is the gate's extreme; otherwise the
    # value belongs to an element the table omits, and saying so is the honest
    # reading rather than attaching the nearest listed name.
    unlisted = "an element the table does not list"
    worst_rhat_name = best_ess_name = None
    table = _read(directory, "diagnostics")
    if table is not None:
        import pandas as pd

        try:
            named = pd.read_csv(os.path.join(directory, "diagnostics.csv"), index_col=0)
            if "r_hat" in named.columns and max_rhat is not None:
                column = named["r_hat"].astype(float)
                worst_rhat_name = (
                    str(column.idxmax())
                    if abs(float(column.max()) - float(max_rhat)) <= 5e-4
                    else unlisted
                )
            ess_columns = [c for c in ("ess_bulk", "ess_tail") if c in named.columns]
            if ess_columns and min_ess is not None:
                stacked = named[ess_columns].astype(float)
                table_min = float(stacked.min().min())
                if abs(table_min - float(min_ess)) <= 0.01 * max(float(min_ess), 1.0):
                    best_ess_name = str(stacked.stack().idxmin()[0])
                else:
                    best_ess_name = unlisted
        except (OSError, ValueError, TypeError, IndexError):
            pass

    divergences = summary.get("divergences")
    bfmi = summary.get("bfmi_per_chain") or []
    min_bfmi = min(bfmi) if bfmi else None

    def mark(ok):
        return "pass" if ok else "**fail**"

    print("| Check | This fit | Threshold | Tier | Result |")
    print("| --- | --- | --- | --- | --- |")
    if max_rhat is not None:
        where = (
            f" ({worst_rhat_name})" if worst_rhat_name == unlisted
            else f" (`{worst_rhat_name}`)" if worst_rhat_name else ""
        )
        print(
            f"| Largest R-hat | {max_rhat:.6f}{where} | ≤ {rhat_max:g} | hard | "
            f"{mark(checks.get('rhat', max_rhat <= rhat_max))} |"
        )
    if min_ess is not None:
        where = (
            f" ({best_ess_name})" if best_ess_name == unlisted
            else f" (`{best_ess_name}`)" if best_ess_name else ""
        )
        print(
            f"| Smallest effective sample size | {min_ess:,.0f}{where} | ≥ {ess_min:,} | hard | "
            f"{mark(checks.get('ess', min_ess >= ess_min))} |"
        )
    if divergences is not None:
        print(
            f"| Divergent transitions | {divergences:,} | 0 | soft | "
            f"{mark(checks.get('divergences', divergences == 0))} |"
        )
    if min_bfmi is not None:
        print(
            f"| Smallest energy BFMI across chains | {min_bfmi:.3f} | ≥ {bfmi_min:g} | soft | "
            f"{mark(checks.get('bfmi', min_bfmi >= bfmi_min))} |"
        )
    unassessable = summary.get("unassessable_parameters") or []
    if unassessable:
        print(f"| Parameters the gate could not assess | {len(unassessable):,} | 0 | — | noted |")
    print()

    manifest = read_manifest(directory)
    params = (manifest.get("sampling") or {}).get("parameters") or {}
    effort = []
    if params.get("chains") and params.get("draws"):
        effort.append(f"{params['chains']:,} chains × {params['draws']:,} draws")
    if params.get("tune"):
        effort.append(f"after {params['tune']:,} tuning draws")
    if params.get("target_accept"):
        effort.append(f"at target acceptance {params['target_accept']:g}")
    hard = bool(checks.get("rhat", True)) and bool(checks.get("ess", True))
    soft = bool(checks.get("divergences", True)) and bool(checks.get("bfmi", True))
    if hard and soft:
        verdict = "This fit clears both tiers of the convergence gate."
    elif hard:
        verdict = (
            "This fit clears the **hard** tier (R-hat and effective sample size) but not the "
            "**soft** tier; the caveat block above says what was recorded and the reported "
            "intervals should be read with that in mind."
        )
    else:
        verdict = (
            "This fit **does not clear the hard tier** and reaches this page only under a "
            "recorded exception, which the caveat block above states."
        )
    sentence = verdict
    if effort:
        sentence += " Sampled with " + ", ".join(effort) + "."
    print(": " + sentence + " Read from `diagnostics_summary.json` and the fit manifest.")


def render_prior_posterior_contraction(directory: str = ".") -> None:
    """Print how much each prior was narrowed by the data, from the fit's own table.

    Every template asks the reader to "compare the marginal posteriors above with
    the prior figures earlier in this report", across fifteen to thirty
    parameters, by eye. Where a page then states the answer it states it from
    memory. ``scripts/prior_vs_posterior.py --table`` computes contraction
    (1 − posterior SD / prior SD) and the prior CDF at the posterior mean from
    the trace and the definition; this renders the copy it writes into the fit
    directory. Fail-soft, like the LOO section.
    """
    table = _read(directory, "prior_posterior_contraction")
    if table is None:
        print(
            "_No prior-to-posterior contraction table for this fit "
            "(`prior_posterior_contraction.csv` absent — run "
            "`scripts/prior_vs_posterior.py --table` against the fitted output to produce it)._"
        )
        return
    needed = {"parameter", "posterior_mean", "posterior_sd", "prior_cdf", "contraction"}
    if not needed <= set(table.columns):
        print("_The contraction table for this fit does not carry the expected columns._")
        return

    descriptions = {parameter: description for parameter, description, _, _ in _PRIOR_SPECS}
    has_prior = {"prior_median", "prior_sd"} <= set(table.columns)
    rows = table.sort_values("contraction").itertuples(index=False)

    header = ["Parameter", "Prior median", "Prior SD"] if has_prior else ["Parameter"]
    header += ["Posterior mean", "Posterior SD", "Contraction", "Prior CDF at the mean", "Reading"]
    print("| " + " | ".join(header) + " |")
    print("| --- |" + " ---: |" * (len(header) - 2) + " --- |")
    prior_driven = []
    pressing = []
    for row in rows:
        name = str(row.parameter)
        label = descriptions.get(name, f"`{name}`")
        contraction = float(row.contraction)
        cdf = float(row.prior_cdf)
        if contraction <= 0.1:
            reading = "prior-driven"
            prior_driven.append(name)
        elif cdf >= 0.95 or cdf <= 0.05:
            reading = "pressing against the prior"
            pressing.append(name)
        else:
            reading = "informed by the data"
        cells = [label]
        if has_prior:
            cells += [f"{float(row.prior_median):.3g}", f"{float(row.prior_sd):.3g}"]
        cells += [
            f"{float(row.posterior_mean):.3g}",
            f"{float(row.posterior_sd):.3g}",
            f"{contraction:.2f}",
            f"{cdf:.2f}",
            reading,
        ]
        print("| " + " | ".join(cells) + " |")
    print()
    summary = (
        ": Contraction is 1 − posterior SD / prior SD: 1 means the data fixed the parameter, "
        "0 means the posterior is no narrower than the prior, and a value at or below 0.1 "
        "is marked **prior-driven** — the reported value restates the prior rather than "
        "estimating anything. The prior CDF at the posterior mean says where inside the "
        "prior the data landed; near 0 or 1 the prior is a wall the likelihood is pushing "
        "against. Computed from this fit's trace and its recorded definition by "
        "`scripts/prior_vs_posterior.py`."
    )
    if prior_driven:
        summary += (
            f" Prior-driven here: {', '.join(f'`{n}`' for n in prior_driven)} — any "
            "conclusion resting on one of these is a restatement of its prior."
        )
    if pressing:
        summary += f" Pressing against the prior: {', '.join(f'`{n}`' for n in pressing)}."
    print(summary)


def _print_age_band_coverage(detail) -> None:
    """The administrations carrying each outcome, by age band, from a rebuilt frame."""
    outcomes = [c for c in ("understood", "spoken", "signed") if c in detail.columns]
    if not outcomes or "age" not in detail.columns:
        return
    bands = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 1000)]
    print()
    print("| Age band (months) | " + " | ".join(outcomes) + " |")
    print("| --- |" + " ---: |" * len(outcomes))
    for lo, hi in bands:
        band = detail[(detail["age"] >= lo) & (detail["age"] < hi)]
        if band.empty:
            continue
        label = f"{lo}–{hi}" if hi < 1000 else f"{lo} and above"
        print(
            f"| {label} | "
            + " | ".join(f"{int(band[c].notna().sum()):,}" for c in outcomes)
            + " |"
        )
    print()
    print(
        ": Administrations carrying each outcome, by age band. The bands with few rows "
        "are where the curves rest on the model rather than on observations."
    )


def _verified_frame(manifest: dict):
    """The fit's analysis frame rebuilt from current data, or ``None`` and a reason.

    A block that wants more than the manifest records has to rebuild the frame,
    and must only use it if it still hashes to the one the fit recorded -- a
    loader-rule change since the fit would otherwise be described as if it had
    been fitted. Shared so the frame-composition and dispersion-scope blocks
    cannot apply that guard differently.
    """
    data = manifest.get("data") or {}
    try:
        from vocab_growth.analysis_frames import (
            analysis_frame_hash,
            build_analysis_frame,
        )
        from vocab_growth.models.definitions import MODEL_REGISTRY

        model_id = str((manifest.get("model") or {}).get("model_id") or "").lower()
        recorded = data.get("analysis_frame_hash")
        registered = MODEL_REGISTRY.get(model_id)
        if registered is None or not recorded:
            return None, "the registered definition or the recorded frame hash is unavailable"
        frame, _ = build_analysis_frame(model_id, registered)
        if analysis_frame_hash(frame) != recorded:
            return None, "the frame the current loader rules build no longer hashes to the fitted one"
        return frame, None
    except Exception as exc:  # pragma: no cover - render robustness, not model logic
        return None, f"it could not be rebuilt here (`{type(exc).__name__}`)"


def render_frame_composition(directory: str = ".") -> None:
    """Print what the fitted frame is made of, exactly, from the manifest.

    The Data section on every page was ``describe()`` with normality tests
    appended, followed by a callout apologising for the normality tests. What a
    reader needs is the number of children and administrations, the share of
    children seen more than once (the quantity that decides child-effect
    identification on every hierarchical page), and rows per study. The manifest
    records the first three exactly; the per-study children and age spans need
    the frame, which is rebuilt through :mod:`vocab_growth.analysis_frames` and
    used **only if its hash matches the one the fit recorded** — otherwise the
    manifest-only table is printed and the page says why.
    """
    manifest = read_manifest(directory)
    data = manifest.get("data") or {}
    rows = data.get("rows")
    children = data.get("children")
    counts = data.get("observed_outcome_counts") or {}
    per_study = data.get("source_row_counts") or {}
    if not rows and not per_study:
        print("_No fit manifest for this fit, so the frame composition cannot be read._")
        return

    facts = []
    if rows:
        facts.append(f"**{rows:,} administrations**")
    if children:
        facts.append(f"from **{children:,} children**")
        if rows:
            facts.append(f"({rows / children:.2f} administrations per child on average)")
    if counts:
        facts.append(
            "; outcome rows: " + ", ".join(f"{k} {v:,}" for k, v in counts.items())
        )
    print(" ".join(facts).replace(" ;", ";") + ".")
    print()

    detail, reason = _verified_frame(manifest)

    if detail is not None and "age" in detail.columns and "subject_id" not in detail.columns:
        # The engines without child effects build a frame with no child key --
        # which is a fact about the model, not a gap in the record: every row
        # enters the likelihood as an independent administration. The plain
        # bivariate frame carries no study key either; the trivariate one keeps
        # `study` for its signing-source masks. Say which, and give what the
        # frame does carry.
        if "study" in detail.columns:
            print("| Study | Administrations | Ages (months) |")
            print("| --- | ---: | --- |")
            for study, group in sorted(detail.groupby("study"), key=lambda kv: -len(kv[1])):
                print(f"| `{study}` | {len(group):,} | {group['age'].min():.0f}–{group['age'].max():.0f} |")
            print()
            print(
                ": Rebuilt from the current data through the same loader rules the fit used, and "
                "verified to hash to the frame this fit recorded. **This frame carries no child "
                "key**: the model treats every administration as independent, so the share of "
                "children seen more than once is not part of what it fitted, and the study "
                "column above labels rows without entering the likelihood."
            )
        else:
            print(
                ": Rebuilt from the current data through the same loader rules the fit used, and "
                "verified to hash to the frame this fit recorded. **This frame carries no study "
                "or child key**: the model treats every administration as independent, so "
                "neither a per-study breakdown nor the share of children seen more than once is "
                "part of what it fitted."
            )
        _print_age_band_coverage(detail)
        return

    if detail is not None and {"study", "subject_id"} <= set(detail.columns):
        by_study = detail.groupby("study")
        print("| Study | Administrations | Children | Ages (months) |")
        print("| --- | ---: | ---: | --- |")
        for study, group in sorted(by_study, key=lambda kv: -len(kv[1])):
            ages = group["age"] if "age" in group.columns else None
            span = f"{ages.min():.0f}–{ages.max():.0f}" if ages is not None else "—"
            print(
                f"| `{study}` | {len(group):,} | {group['subject_id'].nunique():,} | {span} |"
            )
        print()
        visits = detail.groupby("subject_id").size()
        repeaters = visits[visits > 1]
        repeat = float((visits > 1).mean()) if len(visits) else float("nan")
        typical = f" (median {repeaters.median():.0f} visits for those who were)" if len(repeaters) else ""
        print(
            f": Rebuilt from the current data through the same loader rules the fit used, and "
            f"verified to hash to the frame this fit recorded. "
            f"**{repeat:.0%} of children were seen more than once**{typical}; that share is "
            "what identifies persistent between-child differences separately from residual "
            "spread."
        )
        _print_age_band_coverage(detail)
    elif per_study:
        print("| Study | Administrations |")
        print("| --- | ---: |")
        for study, n in sorted(per_study.items(), key=lambda kv: -int(kv[1])):
            print(f"| `{study}` | {int(n):,} |")
        print()
        note = ": Rows per contributing study, as this fit's manifest records them."
        if reason:
            note += (
                f" Children and age spans per study are not shown because {reason}; "
                "the totals above are exact."
            )
        print(note)


#: Which sampled kappa parameter governs which part of a two-anchor curve. The
#: engines build ``kappa(age) = kappa_min + exp(a + b z)`` and give the age term
#: priors at two reference ages, so a flagged parameter implicates a *region* of
#: the curve rather than the whole of it -- which is the difference between "do
#: not read this figure" and "do not read this end of it".
_KAPPA_ROLE_REGIONS = {
    "min": "the floor the curve cannot fall below, at every age",
    "excess_young": "the curve at and below the younger reference age",
    "excess_old": "the curve at and above the older reference age",
    # The legacy intercept-and-slope form, still carried by VG05, VG07 and VG08
    # on both outcomes and by VG14 and VG15 on signing. Its parameters are named
    # `a_kappa_s` and `b_kappa_mag_s`, which do not begin with "kappa" -- the
    # first cut of this block matched on that prefix and silently dropped them,
    # so VG08's page reported one caveat and omitted the strongest prior-data
    # conflict in the suite.
    "level": "the height of the curve at the pool's mean age",
    "slope": "how steeply the curve falls with age",
}

#: Name prefix to role. Ordered longest-first so `kappa_min` is not read as
#: `kappa`.
_KAPPA_ROLE_PREFIXES = (
    ("kappa_excess_young", "excess_young"),
    ("kappa_excess_old", "excess_old"),
    ("kappa_min", "min"),
    ("b_kappa_mag", "slope"),
    ("a_kappa", "level"),
)


def _kappa_role_and_suffix(name: str):
    """Split a dispersion parameter name into ``(role, outcome suffix)``.

    Returns ``(None, None)`` for a name that is not a dispersion parameter.
    """
    if "kappa" not in name:
        return None, None
    for prefix, role in _KAPPA_ROLE_PREFIXES:
        if name.startswith(prefix):
            rest = name[len(prefix):]
            for candidate in ("_sign", "_u", "_s"):
                if rest == candidate:
                    return role, candidate.lstrip("_")
            return role, None if rest else None
    return None, None


def _kappa_curve_scope(suffix, n_trials) -> str:
    """One sentence naming the denominator a kappa's dispersion is measured on."""
    if suffix == "u" or suffix is None:
        pool = f"the {n_trials:,}-item reference inventory" if n_trials else "the reference inventory"
        return f"counts out of {pool}"
    conditional = {"s": "spoken among the words that child understands",
                   "sign": "signed among the words that child understands"}[suffix]
    return (
        f"the **conditional** ratio — {conditional} — measured on that child's own "
        "understood count as the denominator"
    )


def render_dispersion_scope(directory: str = ".") -> None:
    """State what each dispersion parameter is a dispersion *of*, and where it is silent.

    The two kappa figures sit side by side on every bivariate page under headings
    that differ by one word, which invites three readings the model does not
    support: that the two are on one scale (they are not -- ``kappa_u`` is
    marginal on the item pool, ``kappa_s`` is conditional on the child's own
    understood count), that a kappa curve is comparable across models (it is
    residual after whatever child structure that model carries, and the models
    carry different structure), and that the whole curve is estimated (a
    two-anchor kappa can have one end the data never informed).

    Every claim here is read from this fit -- the manifest's ``n_trials`` and
    kappa anchor ages, the diagnostics' parameter list, and the contraction
    table's flags -- so a page cannot assert a scope its own fit contradicts.
    """
    manifest = read_manifest(directory)
    definition = ((manifest.get("model") or {}).get("definition")) or {}
    n_trials = (manifest.get("data") or {}).get("n_trials") or definition.get("n_trials")
    suffixes = [
        suffix
        for suffix in (None, "u", "s", "sign")
        if _read(directory, "posterior_kappa" if suffix is None else f"posterior_kappa_{suffix}")
        is not None
    ]
    if not suffixes:
        print("_This fit writes no posterior dispersion curve, so there is no per-age scope to state._")
        return

    print("| Curve | What its dispersion is measured on |")
    print("| --- | --- |")
    for suffix in suffixes:
        name = "$\\kappa$" if suffix is None else f"$\\kappa_{{{suffix}}}$"
        print(f"| {name} ({_OUTCOME_LABELS[suffix]}) | {_kappa_curve_scope(suffix, n_trials)} |")
    print()

    if {"u", "s"} <= set(suffixes):
        count = {2: "two", 3: "three", 4: "four"}.get(len(suffixes), str(len(suffixes)))
        print(
            f": These {count} curves sit on **different denominators** — one marginal on the "
            "item pool, the rest conditional on each child's own understood count — so their "
            "levels are not comparable with each other. A higher $\\kappa$ on a production "
            "outcome than on understood at some age says nothing about which outcome is more "
            "variable. Compare each curve with itself across age, which is what the findings "
            "table above reports as a variance inflation factor."
        )
        print()

    _print_nested_outcome_split(manifest, directory)
    _print_kappa_identification(directory, definition, suffixes)

    print()
    print(
        "Neither curve is comparable across models. $\\kappa$ is what is left once "
        "that model's own mean curve and child effects have taken their share, and the "
        "models differ in how much child structure they carry — a model with child "
        "*slopes* leaves less in $\\kappa$ than one with child intercepts alone, on the "
        "same data. Read a $\\kappa$ curve against the same model's other ages, never "
        "against another model's curve."
    )


def _print_nested_outcome_split(manifest: dict, directory: str) -> None:
    """How many production rows enter conditionally, and how many via the fallback.

    ``kappa_s`` is the conditional ratio's dispersion only on rows whose understood
    count is usable as a denominator; the rest enter through the treatment named by
    ``spoken_fallback``, where the concentration is a derived quantity rather than
    ``kappa_s`` itself. Where that share is large the curve is a blend, and saying
    so is the difference between reading it and over-reading it.
    """
    treatment = ((manifest.get("model") or {}).get("definition") or {}).get("spoken_fallback")
    if not treatment:
        return
    frame, reason = _verified_frame(manifest)
    if frame is None or not {"understood", "spoken"} <= set(frame.columns):
        if reason:
            print(f"_The conditional/fallback split is not shown because {reason}._")
            print()
        return

    import pandas as pd

    understood = pd.to_numeric(frame["understood"], errors="coerce")
    spoken = pd.to_numeric(frame["spoken"], errors="coerce")
    has_spoken = spoken.notna()
    total = int(has_spoken.sum())
    if not total:
        return
    conditional = int((has_spoken & understood.notna() & (spoken <= understood)).sum())
    fallback = total - conditional
    share = fallback / total

    if not fallback:
        print(
            f"Every one of the {total:,} production rows carries a usable understood count, "
            f"so $\\kappa_s$ is the conditional ratio's dispersion throughout — the "
            f"`{treatment}` fallback is declared but never reached."
        )
        print()
        return

    print(
        f"Of the {total:,} production rows, **{conditional:,} ({1 - share:.0%})** carry a usable "
        f"understood count and enter as the conditional ratio; the remaining "
        f"**{fallback:,} ({share:.0%})** have none and enter through the `{treatment}` "
        "treatment, whose concentration is derived from the understood dispersion and the "
        "ratio rather than being $\\kappa_s$ itself. The curve below is therefore a blend "
        "over those two branches, weighted as the ages of those rows fall."
    )
    print()


def _print_kappa_identification(directory: str, definition: dict, suffixes: list) -> None:
    """Name the parts of each kappa curve the data did not inform, or is straining.

    Two readings, and the order matters. A parameter sitting in the far tail of
    its prior is **pressing**, whatever its contraction: contraction is
    ``1 - posterior sd / prior sd``, so a posterior that is *tightly* determined a
    long way outside its prior has a spread much like the prior's and scores as
    though the data never touched it. VG05, VG07 and VG08 put ``b_kappa_mag_s``
    7.6 to 7.9 prior standard deviations out with a relative posterior spread of
    13%, and the contraction statistic labels all three ``uninformed``. Only a
    parameter that is *both* barely contracted *and* sitting mid-prior is
    genuinely unestimated -- VG22's ``kappa_excess_young_s`` is that case.

    The legacy form's intercept and slope are coupled through
    ``kappa_min + exp(a - b_mag z)``, so when both are straining they are one
    finding and are reported as one.
    """
    table = _read(directory, "prior_posterior_contraction")
    if table is None or "flags" not in table.columns:
        return

    anchor_ages = {}
    for suffix in suffixes:
        field = "kappa" if suffix is None else f"kappa_{suffix}"
        block = definition.get(field)
        if isinstance(block, dict) and block.get("anchor_ages"):
            anchor_ages[suffix] = tuple(block["anchor_ages"])

    found = []
    for row in table.itertuples(index=False):
        name = str(row.parameter)
        role, suffix = _kappa_role_and_suffix(name)
        if role is None:
            continue
        contraction = float(row.contraction)
        cdf = float(row.prior_cdf)
        # The two-sided test, on the numbers rather than the CSV's `flags`
        # column: a prior acting as a floor is the same finding as one acting as
        # a ceiling, and tables written before 2026-09-02 carry a one-sided flag.
        pressing = cdf >= 0.95 or cdf <= 0.05
        unestimated = contraction <= 0.05 and not pressing
        if not (pressing or unestimated):
            continue
        region = _KAPPA_ROLE_REGIONS.get(role, "this curve")
        if suffix in anchor_ages and role in ("excess_young", "excess_old"):
            young, old = anchor_ages[suffix][0], anchor_ages[suffix][-1]
            age = young if role == "excess_young" else old
            region = (
                f"the curve at and {'below' if role == 'excess_young' else 'above'} "
                f"{age:.0f} months"
            )
        found.append(
            {"name": name, "role": role, "suffix": suffix, "region": region,
             "cdf": cdf, "contraction": contraction, "pressing": pressing}
        )

    notes = []
    paired = set()
    for item in found:
        # The legacy intercept and slope move together; two separate caveats
        # would read as two problems where the fit has one.
        if item["role"] != "level":
            continue
        mate = next(
            (o for o in found if o["role"] == "slope" and o["suffix"] == item["suffix"]),
            None,
        )
        if mate is None:
            continue
        paired |= {item["name"], mate["name"]}
        outcome = _OUTCOME_LABELS.get(item["suffix"], "words")
        notes.append(
            f"- **The shape of this curve** for {outcome} is set by a prior the data is "
            f"straining against on both sides. `{item['name']}` (its height at the pool's "
            f"mean age) sits at prior CDF {item['cdf']:.2f} and `{mate['name']}` (how "
            f"steeply it falls with age) at prior CDF {mate['cdf']:.2f}. The two are "
            "coupled — the curve is a floor plus an exponential in age — so this is one "
            "finding, not two: the data wants a steeper decline than the slope prior "
            "allows, and the height is pulled down to compensate. Read the curve's overall "
            "level with more confidence than its steepness."
        )

    for item in found:
        if item["name"] in paired:
            continue
        outcome = _OUTCOME_LABELS.get(item["suffix"], "words")
        if item["pressing"]:
            direction = "above" if item["cdf"] >= 0.95 else "below"
            note = (
                f"- **{item['region']}** for {outcome} is **pressing against its prior**: "
                f"`{item['name']}` sits at prior CDF {item['cdf']:.2f}, so the data wants a "
                f"value {direction} what the prior comfortably allows. Read the level as a "
                "bound the prior is setting rather than as an estimate, and see the "
                "sensitivity analysis."
            )
            if item["contraction"] <= 0.05:
                note += (
                    f" Its contraction ({item['contraction']:.2f}) is low, which here does "
                    "**not** mean the data was silent: a posterior held tightly a long way "
                    "outside its prior has a spread much like the prior's, and the "
                    "contraction statistic cannot tell the two apart."
                )
            notes.append(note)
        else:
            notes.append(
                f"- **{item['region']}** for {outcome} is **not estimated from this data**: "
                f"`{item['name']}` has contraction {item['contraction']:.2f} while sitting "
                f"mid-prior (CDF {item['cdf']:.2f}), so the posterior is no narrower than "
                "the prior it started from. The figure draws a median there because the "
                "curve is continuous, not because the data placed it."
            )

    if not notes:
        print(
            "Every dispersion parameter in this fit is informed by the data and sits "
            "within its prior, so both ends of each curve can be read."
        )
        print()
        return

    print('::: {.callout-warning title="Part of this dispersion curve is prior, not data"}')
    print()
    for note in notes:
        print(note)
    print()
    print(
        "Read from the fit's own prior-to-posterior contraction table, not asserted here."
    )
    print()
    print(":::")
    print()
