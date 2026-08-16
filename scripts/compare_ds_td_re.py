# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
DS-vs-TD population contrasts between RE-based models (separate-model, per-draw).

Honest comparators only: the DS side is the spoken / understood **sub-curve of a
random-effects DS model** (VG10 by default — study + subject REs), not the
no-RE VG01/VG02. The TD side is the univariate study-RE models VG11 (spoken) /
VG12 (understood).

Why separate models suffice for credible intervals
--------------------------------------------------
The DS and TD datasets are disjoint (no shared studies or children), so the
joint posterior factorises and a per-draw difference gives an *exact* credible
interval for any contrast — no joint model required. Accordingly every estimand
here is computed **draw-by-draw** (contrast the draws, never the summaries),
on a **common age grid restricted to the empirical overlap**, using the
**population-level (RE-excluded) outcome-scale** curve on both sides so the
estimand is identical. A joint/stacked model that makes the TD-DS gap itself a
generative object (partial pooling, a directly-estimated delay, difference-in-
differences, the "delayed/scaled TD trajectory" hypothesis) is intentionally NOT
built here, and **is not currently registered against any model number**. It was
reserved as VG16 until that number was taken by the DS within-child cross-lag;
nothing has replaced the reservation, so a reader should not expect this gap to
be closed by an existing model.

Estimands, per outcome, written to the configured comparisons dir (default
``output/comparisons/``; see ``vocab_growth.environment.output_root``):

* ``ds_td_<outcome>_re_expected_words.csv``   — TD & DS expected words, the
  difference δ(a)=TD-DS (+89/50% interval, P(TD>DS)) and the ratio.
* ``ds_td_<outcome>_re_learning_rate.csv``    — dY/da for each population and
  the difference Δ(a) (words/month).
* ``ds_td_<outcome>_re_attainment_delay.csv`` — D(v)=age_DS(v)-age_TD(v): how
  many months behind TD the DS population reaches each vocabulary level v
  (a flat D(v) ⇒ pure shift; a rising D(v) ⇒ developmental stretch).
* ``ds_td_<outcome>_re_dispersion.csv``        — concentration κ, implied word
  SD σ_Y, and the overdispersion factor φ for each population, with the contrasts
  Δκ, Δσ_Y, φ_TD/φ_DS. All three are *observation*-level: in these models κ is
  what the Beta-Binomial layer carries once the study and subject random effects
  have taken their share, so none of them is the between-child contrast.
* ``ds_td_<outcome>_re_subject_heterogeneity.csv`` — the between-child contrast
  proper: τ, the SD across children of the child's own logit for this outcome,
  and the spread in expected words σ_child it induces, per population, with
  Δτ, τ_TD/τ_DS and Δσ_child. See ``comparison.subject_heterogeneity`` for why
  this is not simply VG10's ``tau_subj_q`` read against VG11's ``tau_subject``.

Each panel is emitted as its own standalone figure (linear axes, no subplot
grids) so the figures are usable individually:
``ds_td_<outcome>_re_{expected_words,learning_rate,attainment_delay,spread,
spread_contrast,overdispersion,subject_tau,subject_spread}.{png,svg}`` and, for
the comprehension-matched view,
``ds_td_comprehension_{q_at_U,dq,latency,q_at_age}.{png,svg}``.

Usage::

    python scripts/compare_ds_td_re.py                 # spoken + understood
    python scripts/compare_ds_td_re.py spoken          # one outcome
    python scripts/compare_ds_td_re.py --verify        # self-checks then run
"""

from __future__ import annotations

import os
import sys

import dse_research_utils.plot.styles as plot_styles
import numpy as np
import pandas as pd

from vocab_growth import comparison as C
from vocab_growth import environment as env

# DS comparator: VG10 spoken/understood sub-curve (study+subject REs). One-line
# swap to vg07/vg08/vg09 for a sensitivity check (all carry random effects).
DS_KEY = "vg10"
# Dispersion must contrast a kappa that means the same thing on both sides. What
# kappa means depends on whether a subject random effect is present to absorb
# between-child variance: without one, kappa carries that variance; with one, it
# does not.
#
# This used to select VG07 (study-RE only) because VG11/VG12 were study-RE only
# too, so VG07-vs-TD was the like-for-like pairing and VG10 was not. #164 added
# child random effects to VG11/VG12/VG13, which inverted that: VG07's kappa now
# carries child variance while the TD models' kappa does not, so the old pairing
# contrasts incommensurable quantities and overstates DS dispersion. VG10 has
# subject REs on both outcomes and is now the model that satisfies the original
# criterion. Corrected 2026-08-05 during the full reporting refit; the July 2026
# published dispersion contrast is affected and superseded.
#
# Known residual, not addressed here: VG10's kappa_s is the dispersion of the
# production ratio q conditional on understood, whereas VG11's is the dispersion
# of spoken counts marginally. That joint-vs-univariate difference predates this
# correction and has not been audited — see the comparison book's caveat.
#
# (Mean/rate/delay keep VG10 — subject REs are mean-zero, so the population
# trajectory is unaffected.)
DISP_DS_KEY = "vg10"
# TD comparator per outcome: the univariate models, which since #164 carry both
# study and subject random effects.
TD_KEYS = {"spoken": "vg11", "understood": "vg12"}

OUT_DIR = env.comparisons_output_dir()
SEED = 20260616
GRID_STEP = 0.5  # months
# Vocabulary levels (words) for the attainment-delay D(v) curve.
N_GRID = np.array(
    [10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 200,
     250, 300, 350, 400, 450, 500, 550, 600],
    dtype=float,
)
MIN_COVERAGE = 0.80
KEY_AGES = [12, 18, 24, 30]

# Comprehension-matched lens needs JOINT models (U and S coupled per draw): the
# DS joint VG10 vs the TD joint VG13 (RE-based, 8-18 mo). VG13 is the only valid
# TD joint model (wide-age TD comprehension is not validly measured; VG06 was
# excluded as invalid).
JOINT_DS_KEY = "vg10"
JOINT_TD_KEY = "vg13"
# Comprehension levels N (words understood) for the q(U=N) view. Small-N tail is
# noisy (S/U unstable when U barely exceeds N); high-N tail is coverage-limited.
N_GRID_Q = np.array(
    [10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400],
    dtype=float,
)

COL_TD = plot_styles.COLOUR_ORANGE
COL_DS = plot_styles.COLOUR_BLUE
COL_D = plot_styles.COLOUR_GREEN


# ----------------------------------------------------------------------------
# Frame assembly
# ----------------------------------------------------------------------------
def _merge(grid: np.ndarray, grid_name: str, **frames: pd.DataFrame) -> pd.DataFrame:
    """Combine several summarise_draws frames (same grid) into one wide frame."""
    base = pd.DataFrame({grid_name: grid.astype(float)})
    for name, fr in frames.items():
        rename = {c: f"{name}_{c}" for c in fr.columns if c != grid_name}
        base = base.merge(fr.rename(columns=rename), on=grid_name, how="left")
    return base


def _at_age(frame: pd.DataFrame, age: float, col: str) -> float:
    """Nearest-grid-point value of ``col`` at ``age`` (for console summaries)."""
    i = int((frame["age_months"] - age).abs().idxmin())
    nearest = float(frame.loc[i, "age_months"])
    if abs(nearest - age) > 1.0:
        print(f"    [warn] no grid point near {age:g} mo for '{col}'; "
              f"snapped to {nearest:g} mo")
    return float(frame.loc[i, col])


# ----------------------------------------------------------------------------
# Per-outcome analysis
# ----------------------------------------------------------------------------
def run_outcome(outcome: str) -> None:
    td_key = TD_KEYS[outcome]
    print(f"\n=== {outcome.upper()}: DS={C.model_label(DS_KEY)} vs "
          f"TD={C.model_label(td_key)} ===", flush=True)

    ages_ds, p_ds, k_ds, n_ds = C.load_outcome_trajectory(DS_KEY, outcome)
    ages_td, p_td, k_td, n_td = C.load_outcome_trajectory(td_key, outcome)
    if n_ds != n_td:
        raise ValueError(f"n_trials mismatch: DS={n_ds}, TD={n_td}.")
    n = n_ds
    w_ds, w_td = p_ds * n, p_td * n

    # Empirical overlap window and shared grid.
    lo = max(ages_ds.min(), ages_td.min())
    hi = min(ages_ds.max(), ages_td.max())
    grid = np.arange(np.ceil(lo / GRID_STEP) * GRID_STEP, hi + 1e-9, GRID_STEP)
    print(f"  DS draws={w_ds.shape[0]} ages {ages_ds.min():.0f}-{ages_ds.max():.0f} | "
          f"TD draws={w_td.shape[0]} ages {ages_td.min():.0f}-{ages_td.max():.0f} | "
          f"overlap {lo:.0f}-{hi:.0f} mo, n_trials={n}", flush=True)

    # Pair the two independent posteriors (valid under disjointness).
    ia, ib = C.align_draws(w_ds.shape[0], w_td.shape[0], seed=SEED)
    w_ds, p_ds, k_ds = w_ds[ia], p_ds[ia], k_ds[ia]
    w_td, p_td, k_td = w_td[ib], p_td[ib], k_td[ib]

    # ---- 1. Expected words: levels + difference + ratio ----
    W_ds = C.interp_draws(ages_ds, w_ds, grid)
    W_td = C.interp_draws(ages_td, w_td, grid)
    dW = W_td - W_ds
    ratio = np.where(W_ds > 0, W_td / np.where(W_ds == 0, np.nan, W_ds), np.nan)
    ew = _merge(
        grid, "age_months",
        TD=C.summarise_draws(W_td, grid), DS=C.summarise_draws(W_ds, grid),
        delta=C.summarise_draws(dW, grid, with_p_gt0=True),
    )
    ew["ratio_median"] = np.nanmedian(ratio, axis=0)

    # ---- 2. Learning rate (words/month): derivative on native grid ----
    R_ds = C.interp_draws(ages_ds, C.learning_rate(ages_ds, w_ds), grid)
    R_td = C.interp_draws(ages_td, C.learning_rate(ages_td, w_td), grid)
    lr = _merge(
        grid, "age_months",
        TD=C.summarise_draws(R_td, grid), DS=C.summarise_draws(R_ds, grid),
        delta=C.summarise_draws(R_td - R_ds, grid, with_p_gt0=True),
    )

    # ---- 3. Attainment delay D(v) = age_DS(v) - age_TD(v) (per native grid) ----
    A_ds = np.column_stack([C.first_crossing_age(w_ds, ages_ds, v) for v in N_GRID])
    A_td = np.column_stack([C.first_crossing_age(w_td, ages_td, v) for v in N_GRID])
    ad = C.summarise_draws(A_ds - A_td, N_GRID, "words", with_p_gt0=True)

    # ---- 4. Dispersion: kappa, implied SD, overdispersion factor ----
    # DS side uses the model whose kappa means the same thing as the TD models'
    # — i.e. one that also carries subject REs, so neither side's kappa absorbs
    # between-child variance (see DISP_DS_KEY note). Pair its draws with the same
    # TD subset; independence makes any pairing valid.
    a_disp, p_disp, k_disp, _ = C.load_outcome_trajectory(DISP_DS_KEY, outcome)
    i_disp = C.align_draws(p_disp.shape[0], p_td.shape[0], seed=SEED + 1)[0]
    P_ds = C.interp_draws(a_disp, p_disp[i_disp], grid)
    K_ds = C.interp_draws(a_disp, k_disp[i_disp], grid)
    P_td, K_td = C.interp_draws(ages_td, p_td, grid), C.interp_draws(ages_td, k_td, grid)
    SD_ds, SD_td = C.implied_sd_y(P_ds, K_ds, n), C.implied_sd_y(P_td, K_td, n)
    PHI_ds, PHI_td = C.overdispersion_factor(K_ds, n), C.overdispersion_factor(K_td, n)
    disp = _merge(
        grid, "age_months",
        kappa_TD=C.summarise_draws(K_td, grid), kappa_DS=C.summarise_draws(K_ds, grid),
        dkappa=C.summarise_draws(K_td - K_ds, grid, with_p_gt0=True),
        sdY_TD=C.summarise_draws(SD_td, grid), sdY_DS=C.summarise_draws(SD_ds, grid),
        dsdY=C.summarise_draws(SD_td - SD_ds, grid, with_p_gt0=True),
        phi_TD=C.summarise_draws(PHI_td, grid), phi_DS=C.summarise_draws(PHI_ds, grid),
        phi_ratio=C.summarise_draws(PHI_td / PHI_ds, grid),
    )

    # ---- 5. Between-child heterogeneity: the subject random-effect scale ----
    # Everything in block 4 is observation-level. With subject REs on both sides
    # kappa is the *residual* left after persistent child differences are absorbed,
    # so it cannot answer "do DS children differ from one another more or less than
    # TD children do" — the subject scale answers that, and the two are two halves
    # of one scatter budget in the TD models. tau is the SD of the child's own
    # logit for this outcome, which is defined identically in both
    # parameterisations even though VG10 reaches it through p_u and q rather than
    # through a single spoken intercept (comparison.subject_heterogeneity).
    _, TAU_ds, SDC_ds, _ = C.subject_heterogeneity(
        DISP_DS_KEY, outcome, ages=grid, draws=i_disp)
    _, TAU_td, SDC_td, _ = C.subject_heterogeneity(
        td_key, outcome, ages=grid, draws=ib)
    het = _merge(
        grid, "age_months",
        tau_TD=C.summarise_draws(TAU_td, grid), tau_DS=C.summarise_draws(TAU_ds, grid),
        dtau=C.summarise_draws(TAU_td - TAU_ds, grid, with_p_gt0=True),
        tau_ratio=C.summarise_draws(TAU_td / TAU_ds, grid),
        sdchild_TD=C.summarise_draws(SDC_td, grid),
        sdchild_DS=C.summarise_draws(SDC_ds, grid),
        dsdchild=C.summarise_draws(SDC_td - SDC_ds, grid, with_p_gt0=True),
    )

    # ---- Write CSVs ----
    os.makedirs(OUT_DIR, exist_ok=True)
    prefix = os.path.join(OUT_DIR, f"ds_td_{outcome}_re_")
    ew.to_csv(prefix + "expected_words.csv", index=False)
    lr.to_csv(prefix + "learning_rate.csv", index=False)
    ad.to_csv(prefix + "attainment_delay.csv", index=False)
    disp.to_csv(prefix + "dispersion.csv", index=False)
    het.to_csv(prefix + "subject_heterogeneity.csv", index=False)

    _plot_outcome(outcome, td_key, grid,
                  C.summarise_draws(W_td, grid), C.summarise_draws(W_ds, grid),
                  C.summarise_draws(R_td, grid), C.summarise_draws(R_ds, grid),
                  ad,
                  C.summarise_draws(SD_td, grid), C.summarise_draws(SD_ds, grid),
                  C.summarise_draws(SD_td - SD_ds, grid),
                  C.summarise_draws(PHI_td, grid), C.summarise_draws(PHI_ds, grid),
                  C.summarise_draws(TAU_td, grid), C.summarise_draws(TAU_ds, grid),
                  C.summarise_draws(SDC_td, grid), C.summarise_draws(SDC_ds, grid),
                  C.model_label(DISP_DS_KEY))

    _print_summary(outcome, ew, lr, ad, disp, het, C.model_label(DISP_DS_KEY))


# ----------------------------------------------------------------------------
# Plot + console summary
# ----------------------------------------------------------------------------
def _band(ax, frame, x, label, colour, *, cov=0.0):
    C.plot_summary_band(ax, frame, x, label, colour, min_coverage=cov)


def _save_single(filename: str, ax_setup: dict, draw) -> None:
    """Emit one standalone figure to ``OUT_DIR`` (thin wrapper over the shared
    :func:`comparison.save_panel`)."""
    C.save_panel(OUT_DIR, filename, ax_setup, draw)


def _plot_outcome(outcome, td_key, grid, W_td, W_ds, R_td, R_ds, ad,
                  SD_td, SD_ds, dSD, PHI_td, PHI_ds,
                  TAU_td, TAU_ds, SDC_td, SDC_ds, disp_ds_lab) -> None:
    td_lab, ds_lab = C.model_label(td_key), C.model_label(DS_KEY)
    pre = f"ds_td_{outcome}_re_"

    def expected(ax):
        _band(ax, W_td, "age_months", td_lab, COL_TD)
        _band(ax, W_ds, "age_months", ds_lab, COL_DS)

    _save_single(
        pre + "expected_words",
        dict(xlabel="Age (months)", ylabel=f"Expected words {outcome}",
             title=f"Expected vocabulary — words {outcome} (TD vs DS)"),
        expected,
    )

    def rate(ax):
        _band(ax, R_td, "age_months", td_lab, COL_TD)
        _band(ax, R_ds, "age_months", ds_lab, COL_DS)

    _save_single(
        pre + "learning_rate",
        dict(xlabel="Age (months)", ylabel="Words / month",
             title=f"Learning rate — words {outcome} (TD vs DS)"),
        rate,
    )

    def delay(ax):
        _band(ax, ad, "words", "DS - TD", COL_D, cov=MIN_COVERAGE)
        ax.axhline(0, color=plot_styles.LINE_COLOUR, lw=0.6)

    _save_single(
        pre + "attainment_delay",
        dict(xlabel="Vocabulary level v (words)",
             ylabel="Months DS reaches v after TD",
             title=f"Attainment delay D(v) — words {outcome}"),
        delay,
    )

    def spread(ax):
        _band(ax, SD_td, "age_months", td_lab, COL_TD)
        _band(ax, SD_ds, "age_months", disp_ds_lab, COL_DS)

    _save_single(
        pre + "spread",
        dict(xlabel="Age (months)", ylabel=r"Implied $\sigma_Y$ (words)",
             title=f"Between-child spread — words {outcome}"),
        spread,
    )

    def spread_contrast(ax):
        _band(ax, dSD, "age_months", f"{td_lab} - {disp_ds_lab}", COL_D)
        ax.axhline(0, color=plot_styles.LINE_COLOUR, lw=0.6)

    _save_single(
        pre + "spread_contrast",
        dict(xlabel="Age (months)", ylabel=r"$\Delta\sigma_Y$ (words)",
             title=f"Spread contrast (TD - DS) — words {outcome}"),
        spread_contrast,
    )

    def overdispersion(ax):
        _band(ax, PHI_td, "age_months", td_lab, COL_TD)
        _band(ax, PHI_ds, "age_months", disp_ds_lab, COL_DS)

    _save_single(
        pre + "overdispersion",
        # Both halves of the old title were wrong once DISP_DS_KEY moved to VG10.
        # "study-RE only" described VG07; VG10 carries subject random effects too,
        # which is the whole point of the repointing. "mean-independent" overclaims:
        # the factor removes the explicit p(1-p) term, but kappa is itself
        # level-driven in this family, so a cross-population contrast still carries
        # part of the level difference (see comparison.overdispersion_factor).
        dict(xlabel="Age (months)", ylabel=r"Overdispersion $\varphi$",
             title=f"Overdispersion vs Binomial — words {outcome}"),
        overdispersion,
    )

    def subject_tau(ax):
        _band(ax, TAU_td, "age_months", td_lab, COL_TD)
        _band(ax, TAU_ds, "age_months", disp_ds_lab, COL_DS)

    _save_single(
        pre + "subject_tau",
        dict(xlabel="Age (months)", ylabel=r"Between-child SD $\tau$ (logit)",
             title=f"Between-child heterogeneity — words {outcome}"),
        subject_tau,
    )

    def subject_spread(ax):
        _band(ax, SDC_td, "age_months", td_lab, COL_TD)
        _band(ax, SDC_ds, "age_months", disp_ds_lab, COL_DS)

    _save_single(
        pre + "subject_spread",
        dict(xlabel="Age (months)",
             ylabel=r"Between-child SD $\sigma_{child}$ (words)",
             title=f"Between-child spread in expected words — {outcome}"),
        subject_spread,
    )


def _plot_comprehension(ds_key, td_key, q_td_s, q_ds_s, dq_s,
                        da_td, da_ds, qa_td, qa_ds) -> None:
    """Emit the four comprehension-matched panels as standalone figures."""
    td_lab, ds_lab = C.model_label(td_key), C.model_label(ds_key)
    pre = "ds_td_comprehension_"

    def q_at_U(ax):
        _band(ax, q_td_s, "words", td_lab, COL_TD, cov=MIN_COVERAGE)
        _band(ax, q_ds_s, "words", ds_lab, COL_DS, cov=MIN_COVERAGE)

    _save_single(
        pre + "q_at_U",
        dict(ylim=(0, 1.05), xlabel="Words understood N",
             ylabel="q(U=N) = spoken / understood",
             title="Proportion spoken given understood"),
        q_at_U,
    )

    def dq(ax):
        _band(ax, dq_s, "words", "TD - DS", COL_D, cov=MIN_COVERAGE)
        ax.axhline(0, color=plot_styles.LINE_COLOUR, lw=0.6)

    _save_single(
        pre + "dq",
        dict(xlabel="Words understood N", ylabel=r"$\Delta q$ (TD - DS)",
             title="Spoken-fraction gap at matched comprehension"),
        dq,
    )

    def latency(ax):
        _band(ax, da_td, "N", td_lab, COL_TD, cov=MIN_COVERAGE)
        _band(ax, da_ds, "N", ds_lab, COL_DS, cov=MIN_COVERAGE)

    _save_single(
        pre + "latency",
        dict(xlabel="Words understood / spoken N",
             ylabel=r"$a_S(N) - a_U(N)$ (months)",
             title="Learn-to-say latency"),
        latency,
    )

    def q_at_age(ax):
        _band(ax, qa_td, "age_months", td_lab, COL_TD)
        _band(ax, qa_ds, "age_months", ds_lab, COL_DS)

    _save_single(
        pre + "q_at_age",
        dict(ylim=(0, 1.05), xlabel="Age (months)",
             ylabel="q(a) = E[S(a)] / E[U(a)]",
             title="Production ratio at matched age"),
        q_at_age,
    )


def _print_summary(outcome, ew, lr, ad, disp, het, disp_ds_lab) -> None:
    print(f"  Expected words & learning rate at key ages ({outcome}):")
    for a in KEY_AGES:
        print(f"    {a:>2} mo: TD={_at_age(ew,a,'TD_median'):6.1f}  "
              f"DS={_at_age(ew,a,'DS_median'):6.1f}  "
              f"ratio={_at_age(ew,a,'ratio_median'):4.1f}x  "
              f"|  rate TD={_at_age(lr,a,'TD_median'):5.1f}  "
              f"DS={_at_age(lr,a,'DS_median'):4.1f} w/mo  "
              f"(P(TD>DS)={_at_age(lr,a,'delta_p_gt0'):.2f})")
    print("  Attainment delay D(v) (months DS behind TD), coverage-filtered:")
    for _, r in ad[ad["coverage"] >= MIN_COVERAGE].iterrows():
        print(f"    {int(r['words']):>3} words: {r['median']:5.1f} "
              f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]")
    print(f"  Residual (observation-level) dispersion at key ages "
          f"(DS={disp_ds_lab}, study + subject REs):")
    for a in KEY_AGES:
        print(f"    {a:>2} mo: kappa TD={_at_age(disp,a,'kappa_TD_median'):4.1f} "
              f"DS={_at_age(disp,a,'kappa_DS_median'):4.1f} "
              f"(Δκ P>0={_at_age(disp,a,'dkappa_p_gt0'):.2f})  |  "
              f"σ_Y TD={_at_age(disp,a,'sdY_TD_median'):4.1f} "
              f"DS={_at_age(disp,a,'sdY_DS_median'):4.1f} "
              f"(Δσ_Y P>0={_at_age(disp,a,'dsdY_p_gt0'):.2f})  |  "
              f"φ_TD/φ_DS={_at_age(disp,a,'phi_ratio_median'):.2f}")
    print("  Between-child heterogeneity at key ages (the subject scale):")
    for a in KEY_AGES:
        print(f"    {a:>2} mo: tau TD={_at_age(het,a,'tau_TD_median'):5.2f} "
              f"DS={_at_age(het,a,'tau_DS_median'):5.2f} "
              f"(ratio={_at_age(het,a,'tau_ratio_median'):4.2f}, "
              f"P(TD>DS)={_at_age(het,a,'dtau_p_gt0'):.2f})  |  "
              f"σ_child TD={_at_age(het,a,'sdchild_TD_median'):5.1f} "
              f"DS={_at_age(het,a,'sdchild_DS_median'):5.1f} words "
              f"(P(TD>DS)={_at_age(het,a,'dsdchild_p_gt0'):.2f})")


# ----------------------------------------------------------------------------
# Comprehension-matched analysis (joint models only)
# ----------------------------------------------------------------------------
def run_comprehension_matched(ds_key: str = JOINT_DS_KEY,
                              td_key: str = JOINT_TD_KEY) -> None:
    """Contrast the production ratio q = S/U *given words understood*.

    Matching on comprehension N (rather than age) strips out the TD/DS timescale
    difference: it asks "when a child understands N words, what fraction does
    she speak, and how much sooner does TD say them?". Requires JOINT models so
    U and S are coupled per draw (VG10 DS vs VG13 TD).
    """
    print(f"\n=== COMPREHENSION-MATCHED: DS={C.model_label(ds_key)} vs "
          f"TD={C.model_label(td_key)} ===", flush=True)
    ages_ds, U_ds, S_ds = C.load_population_trajectory(
        C.trace_path(ds_key), C.n_trials(ds_key))
    ages_td, U_td, S_td = C.load_population_trajectory(
        C.trace_path(td_key), C.n_trials(td_key))
    print(f"  DS draws={U_ds.shape[0]} ages {ages_ds.min():.0f}-{ages_ds.max():.0f} | "
          f"TD draws={U_td.shape[0]} ages {ages_td.min():.0f}-{ages_td.max():.0f}",
          flush=True)
    ia, ib = C.align_draws(U_ds.shape[0], U_td.shape[0], seed=SEED)

    # q at matched comprehension U = N (the headline: spoken fraction given U).
    qU_ds = C.compute_q_at_U(ages_ds, U_ds, S_ds, N_GRID_Q)[ia]
    qU_td = C.compute_q_at_U(ages_td, U_td, S_td, N_GRID_Q)[ib]
    q_ds_s = C.summarise_draws(qU_ds, N_GRID_Q, "words")
    q_td_s = C.summarise_draws(qU_td, N_GRID_Q, "words")
    dq_s = C.summarise_draws(qU_td - qU_ds, N_GRID_Q, "words", with_p_gt0=True)

    # Learn-to-say latency a_S(N) - a_U(N): months between understanding and
    # saying N words, per population.
    da_ds, _ = C.compute_latency(ages_ds, U_ds[ia], S_ds[ia], N_GRID_Q)
    da_td, _ = C.compute_latency(ages_td, U_td[ib], S_td[ib], N_GRID_Q)

    # q at matched age, over the chronological overlap, for reference.
    lo = max(ages_ds.min(), ages_td.min())
    hi = min(ages_ds.max(), ages_td.max())
    age_grid = np.linspace(lo, hi, 41)
    qa_ds = C.summarise_draws(C.compute_q_at_age(ages_ds, U_ds[ia], S_ds[ia], age_grid), age_grid)
    qa_td = C.summarise_draws(C.compute_q_at_age(ages_td, U_td[ib], S_td[ib], age_grid), age_grid)

    os.makedirs(OUT_DIR, exist_ok=True)
    _merge(N_GRID_Q, "words", q_TD=q_td_s, q_DS=q_ds_s, dq=dq_s).to_csv(
        os.path.join(OUT_DIR, "ds_td_comprehension_q_at_U.csv"), index=False)

    _plot_comprehension(ds_key, td_key, q_td_s, q_ds_s, dq_s, da_td, da_ds, qa_td, qa_ds)

    print("  Spoken fraction given understood q(U=N) (coverage-filtered):")
    qtab = _merge(N_GRID_Q, "words", q_TD=q_td_s, q_DS=q_ds_s, dq=dq_s)
    for _, r in qtab[qtab["dq_coverage"] >= MIN_COVERAGE].iterrows():
        print(f"    U={int(r['words']):>3}: TD q={r['q_TD_median']:.2f}  "
              f"DS q={r['q_DS_median']:.2f}  "
              f"Δq(TD-DS)={r['dq_median']:+.2f} "
              f"[{r['dq_ci_lo']:+.2f}, {r['dq_ci_hi']:+.2f}]  "
              f"P(TD>DS)={r['dq_p_gt0']:.2f}")


# ----------------------------------------------------------------------------
# Self-checks
# ----------------------------------------------------------------------------
def _verify() -> None:
    from scipy.stats import betabinom
    n = 810
    for p, kap in [(0.05, 5.0), (0.30, 20.0), (0.60, 3.0)]:
        want = float(betabinom(n, p * kap, (1 - p) * kap).std())
        got = float(C.implied_sd_y(np.array(p), np.array(kap), n))
        assert abs(want - got) < 1e-6 * max(1.0, want), (p, kap, want, got)
    ages = np.linspace(8, 40, 321)
    shift = 10.0
    w_td = np.stack([20.0 * (ages - 8)] * 5)
    w_ds = np.stack([np.clip(20.0 * (ages - 8 - shift), 0, None)] * 5)
    for v in (50.0, 100.0, 200.0):
        d = C.first_crossing_age(w_ds, ages, v) - C.first_crossing_age(w_td, ages, v)
        assert np.allclose(d, shift, atol=0.2), (v, d)

    # Between-child quadrature against brute-force Monte Carlo over the subject
    # REs. The product form is the one worth checking: its tau is an induced
    # quantity, not a parameter, so an error here would be invisible downstream.
    rng = np.random.default_rng(SEED)
    f_u = np.array([[-4.0, -1.5, 0.5]])
    h = np.array([[-3.0, -1.0, 0.0]])
    tau_u, tau_q = np.array([0.9]), np.array([1.3])
    tau_g, sd_g = C.child_spread_product(f_u, h, tau_u, tau_q, n)
    z1 = rng.standard_normal((400_000, 1))
    z2 = rng.standard_normal((400_000, 1))
    p_mc = (1 / (1 + np.exp(-(f_u + tau_u[0] * z1)))) * (
        1 / (1 + np.exp(-(h + tau_q[0] * z2))))
    lg_mc = np.log(p_mc) - np.log1p(-p_mc)
    assert np.allclose(tau_g[0], lg_mc.std(axis=0), rtol=0.02), (tau_g, lg_mc.std(axis=0))
    assert np.allclose(sd_g[0], n * p_mc.std(axis=0), rtol=0.02), (sd_g, n * p_mc.std(axis=0))
    # The single-intercept form must return tau itself, exactly.
    tau_s, _ = C.child_spread_single(np.zeros((1, 3)), np.array([0.7]), n)
    assert np.allclose(tau_s, 0.7), tau_s

    print("self-check OK: implied_sd_y == scipy.betabinom.std; "
          "D(v) recovers a constant age shift; between-child quadrature "
          "matches Monte Carlo.\n")


def main() -> None:
    env.preflight_disk(2.0, OUT_DIR, label="DS/TD comparison outputs")
    argv = sys.argv[1:]
    if "--verify" in argv:
        _verify()
    td_joint = next(
        (a.split("=", 1)[1] for a in argv if a.startswith("--td-joint=")),
        JOINT_TD_KEY,
    )
    tokens = [a for a in argv if not a.startswith("-")] or ["spoken", "understood"]
    for tok in tokens:
        if tok == "comprehension":
            run_comprehension_matched(JOINT_DS_KEY, td_joint)
        elif tok in TD_KEYS:
            run_outcome(tok)
        else:
            raise SystemExit(
                f"unknown token {tok!r}; choose from "
                f"{list(TD_KEYS) + ['comprehension']}"
            )
    print(f"\nWrote CSVs + figures to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
