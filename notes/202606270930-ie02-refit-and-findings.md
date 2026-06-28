# IE_02 integration, full DS refit, and DS/TD comparison update

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

Status: 2026-06-27. Records the wiring of the new **Ireland 2 (`ie_02`)** dataset,
the reporting-quality refit of all Down syndrome (DS) models, the in-progress
typically-developing (TD) refit, and the comparison tooling changes. Numbers
below are taken from the fitted output under `output/models/<MODEL>/`
(`--config rep`) — `diagnostics.csv`, `posterior_summary_*.csv`, and the trace
`sample_stats` — not from prior prose.

## 1. Data change — ie_02 wired in

`ie_02` is a longitudinal Irish dataset (long format,
one row per timepoint `t1`/`t2`) carrying **understood, spoken _and_ signed**
counts. It is the second DS source after `uk_01` to record all three modalities.

Wiring (`scripts/prepare_data.py`, [PR #77](https://github.com/dseinternational/vocabulary-growth/pull/77)):
source entry + merge block (`study = 10`) + DuckDB table + a `vocab_combined`
UNION mapping `understood/spoken/signed`, filtered to `english_speaking = 'yes'`.

Decisions: both recruitment groups pooled as one DS study; the single
non-English-speaking child (2 rows) excluded; `sex` NULL (undocumented `gender`,
unused by the DS fits); `survey_vocab_max = 800` (matching `ie_01`, unused by the
DS loader).

Effect on the analysis set:

| Quantity | Value |
| --- | --- |
| `ie_02` rows in view | 114 (116 raw − 2 non-English) |
| `ie_02` subjects | 66 (1 non-English child dropped) |
| Total DS rows (`load_combined_data`) | 1,078 across 11 studies |
| **Signed observations (feeds VG14/VG15)** | **414 → 528 (+114, +27.5%)** |

All `ie_02` counts are ≤ 800 and satisfy `spoken ≤ understood`, `signed ≤
understood` for every row (no Beta-Binomial guard violations). `ie_02` is now the
second-largest signed source after `uk_01`.

## 2. How the models fit

**Sampling.** Reporting config (`rep`): nutpie NUTS, **6 chains × (6,000 tune +
6,000 draws)**, `target_accept = 0.95`. Per-model wall times (rep): VG01 ~8 m,
VG02 ~5 m, VG05 ~17 m, VG07 ~20 m, VG08 ~18 m, VG09 ~15 m, VG10 ~16 m, VG14
~25 m, VG15 ~17 m.

**Convergence — all 9 DS models:**

| Model | structure | max R̂ | params R̂>1.01 | min ESS (bulk) | divergences |
| ----- | --------- | ----: | ------------: | -------------: | ----------: |
| VG01 | spoken, univariate | 1.001 | 0 | 10,933 | 0 |
| VG02 | understood, univariate | 1.000 | 0 | 10,148 | 0 |
| VG05 | joint U+S | 1.001 | 0 | 9,116 | 0 |
| VG07 | + study RE | 1.001 | 0 | 6,435 | 0 |
| VG08 | + subject RE (U) | 1.004 | 0 | 1,379 | 0 |
| **VG09** | + subject RE (U,q) | **1.021** | **4** | **423** | 0 |
| VG10 | + tighter q anchors, GP anchor | 1.006 | 0 | 670 | 0 |
| VG14 | trivariate (+signed) | 1.001 | 0 | 6,628 | 0 |
| VG15 | joint four-cell sign/speech | 1.005 | 0 | 911 | 0 |

**Zero divergences across all nine models.** Convergence is clean everywhere
except **VG09**, which shows mild non-convergence (max R̂ 1.021 on 4 parameters,
min ESS ~420) — consistent with its known sampler difficulty as the model
carrying subject random intercepts on *both* the understood trajectory and the
production ratio. VG09 is **not** used downstream by the comparison suite (which
reads VG10 for population means and VG07 for dispersion), so this does not affect
the DS/TD analysis. VG10 and VG15 (the more heavily parameterised models that the
ie_02 data most affects) converged cleanly.

**Operational note — VG15 / h5py.** The first VG15 rep attempt sampled
successfully but **failed at the final `trace.to_netcdf`** with `ImportError: No
module named 'h5py'`. Root cause: a conda update had removed `h5py`/`netcdf4`
from the env; the joint model is the only one that writes its trace via xarray's
*DataTree* path, which hard-requires the `h5py` backend, whereas the other eight
models use the InferenceData path (a different, still-working writer). Fix:
reinstalled `h5py` (3.16.0), verified the DataTree round-trip, and re-ran VG15
cleanly. `environment.yml` declares `h5netcdf` (pip) but **not** `h5py` —
recommend adding `h5py` to the pip block to prevent recurrence.

## 3. Key findings

All figures are posterior medians [90% HDI], **adversarially verified against the
source `posterior_summary_*.csv`** (every headline number re-checked by an
independent agent against the CSV — all matched). "Expected words" is the latent
mean trajectory (Ey) out of the 800-item reference inventory; individual-child
predictive spread is much wider (the Y intervals routinely reach 0 at the low
end). **All fits are on the ie_02-augmented data** — ie_02 was wired in, the
DuckDB rebuilt, then every model refit.

### Comprehension and production (DS, from the headline joint model VG10)

| Quantity | 24 mo | 48 mo | 72 mo |
| --- | --- | --- | --- |
| Words understood | 87 [67, 108] | 259 [213, 307] | 441 [369, 510] |
| Words spoken | 4 [2, 6] | 131 [93, 170] | 366 [303, 431] |
| Production ratio q = S/U | 0.05 [0.03, 0.07] | 0.51 [0.39, 0.63] | 0.84 [0.72, 0.95] |

- **Comprehension leads production throughout.** Understanding reaches ~100 words
  by ~24 mo (VG02 reaches 50/100/200 understood at ~16.5/23.7/31.6 mo); the
  typical child reaches 10/50/100 *spoken* words only at ~25/39/45 mo (VG01).
- **The production ratio q crosses 0.5 around 48 months** consistently across
  specifications (VG05 0.50, VG10 0.51, VG14 0.50 at 48 mo) — a DS child speaks
  roughly half the words they understand by age 4, rising to ~0.84 by age 6.
- **Subject random effects sharply lower the _typical_-child early production
  ratio.** At 24 mo, q ≈ 0.18–0.23 in the no-/study-RE models (VG05/07/08) but
  ≈ 0.05–0.07 once subject REs are added (VG09 0.066, VG10 0.047, VG15 0.060):
  the pooled population average is inflated by a minority of early talkers; the
  modelled typical DS child says a much smaller fraction of understood words in
  the third year than the pooled mean implies. VG09 carries this with the
  mild-convergence caveat (§2); VG10 (anchored, clean) is the headline.

### Signing — VG14 / VG15, newly powered by ie_02 (+114 signed obs, +27.5%)

- **Signing is a hump, not a monotone trend.** The signed ratio r(a) (fraction of
  understood words signed) rises from ~0.12 at 12 mo to a peak **~0.40 at 24 mo**,
  then declines (0.33 at 36, 0.27 at 48, 0.14 at 60 mo) as speech takes over
  (VG14).
- **Sign → speech crossover at ~39 months.** The spoken ratio q overtakes the
  signed ratio r at ~39 mo (both ~0.30): before ~3 years a DS child signs a
  larger share of understood words than they speak; speech overtakes thereafter
  (VG14, `sign_speech_crossover.csv`).
- **Counting sign credits substantial extra early expressive vocabulary.** Total
  expressive words p_any ≈ 57 / 209 / 310 at 24/48/72 mo (VG14), versus spoken
  alone ~24 / 165 / 303 — the gap is largest in the early years where signing is
  most active.
- **Sign and speech positively co-occur within children (VG15).** The
  within-understood association **psi = 2.17 [1.46, 2.96]** (P(psi>1) ≈ 0.9999,
  HDI excludes independence): a word a DS child understands is more likely to be
  *both* signed and spoken than chance predicts — signing accompanies rather than
  displaces speech. The data-identified total expressive p_any (≈ 21 / 164 / 361
  words at 24/48/72 mo) sits just below VG14's independence upper bound,
  confirming the conditional-independence assumption modestly over-counts.

ie_02 is now the second-largest signing source after uk_01 and materially
strengthens identification of the r(a) peak/decline, the ~39-mo crossover, and
psi — the findings most specific to this round of work.

## 4. DS vs TD comparisons

All TD models refit at rep (VG03/04/06/11/12/13; all converged, R̂ ≤ 1.002, ESS
in the thousands; VG13 had 40/36,000 divergences and VG04 2 — negligible). All
contrasts below are **per-draw differences of the disjoint DS×TD posteriors**
(exact CIs, no joint model), at the **population level**, over the **empirical
age/level overlap only** (DS comparator VG10; TD VG11/VG12 univariate, VG13
joint 8–18 mo; dispersion/distribution use study-RE-only VG07 vs VG11/12). The
TD models span only ~8–30 mo, so the overlap — and several estimands — are
identified only at younger ages / lower vocabulary levels; this is flagged
throughout.

### 4.1 The gap is large and production-specific

At 24 mo, expected **comprehension** is TD 352 vs DS 87 (≈4×); expected
**production** is TD 257 vs DS 4 (≈63×) — P(TD>DS)=1.00 across the overlap. The
deficit is far larger for production than comprehension.

### 4.2 A developmental *stretch*, not a constant shift

The attainment delay D(v) = months DS reaches level v after TD **grows with the
level** on both outcomes:

| level v | understood delay | spoken delay |
| --- | --- | --- |
| 10 words | ~0 mo | 17 mo |
| 50 words | 10 mo | 22 mo |
| 100 words | 12 mo | 26 mo |
| 200 words | 21 mo | 32 mo |
| 300 words | 32 mo | 38 mo |

A flat D(v) would mean a pure time-shift; the rising curve means the gap *widens*
with development. Production lags comprehension at every level. Consistently,
**peak learning-rate age** is ~64 mo (DS) vs ~23 mo (TD) for spoken and ~71 vs
~18 mo for understood — DS peak-velocity arrives ~3.5–4.5 years later.

### 4.3 Expressive-specific delay (the "expressive delay" headline)

DS production is delayed *beyond* what its comprehension delay alone predicts,
two complementary ways:

- **Level-indexed** Δ_exp(N) = (production attainment delay) − (comprehension
  attainment delay): **16.7 mo [13.6, 19.8] at N=10, ~12.4 mo [9.6, 15.3] at
  N=50, P(>0)=1.00 throughout.** At a given vocabulary *size*, DS takes ~12–17
  extra months to *say* the words it *understands*, relative to TD. (Identified
  only at N≈10–50 — TD's 8–18 mo joint model reaches limited vocabulary.)
- **Age-indexed** extra-expressive delay (cea_U − cea_S): ~2.9 mo [P=1.00] at
  24 mo, ~1.5 mo [P=0.95] at 36 mo, ≈0 by 48 mo. Smaller, because at a fixed age
  DS comprehension is *also* heavily delayed (24 mo DS ≈ TD 12 mo receptively,
  9 mo expressively), so both map to early-TD ages and the *extra* lag compresses.

The two are consistent — they hold level vs age fixed respectively. The
level-indexed Δ_exp is the cleaner "expressive delay" statistic.

Comprehension-matched q(U=N) tells the same story: at equal comprehension DS
speaks a smaller fraction (U=50: TD q=0.08 vs DS 0.03; U=100: 0.16 vs 0.05;
Δq>0, P≈1.00 to U≈150, converging by U≈200). The low-U end is noisy (q=S(a_U)/N
is unstable when U barely exceeds N) — read the mid-range.

### 4.4 Distributional "how atypical"

Fraction of DS children below the **TD 10th centile** word count: **spoken** 0.70
(12 mo) → 0.90 (24 mo) → 0.95 (28 mo); **understood** 0.30 (12 mo) → 0.58
(16 mo) → 0.84 (24 mo). By age 2, ~90% of DS children produce fewer spoken words
than the bottom 10% of TD children — comprehension is less atypical early but
both become highly atypical as TD explodes.

### 4.5 Sign-inclusive gap (newly enabled by ie_02 → VG14 p_any)

Counting signed words credits DS extra expressive vocabulary — ~7 words (16 mo),
~17 (20 mo), ~32 (24 mo), ~57 (30 mo) — which **narrows the DS–TD expressive gap
by ~14–15%** in the 24–30 mo window (24 mo: 233→201; 30 mo: 374→317). Real and
specifically powered by the new ie_02 signing data, but modest: the gap stays
enormous because TD production is mid-explosion here, and the window is capped at
30 mo by TD support.

### 4.6 Reading

DS early vocabulary is best described as a **developmental stretch with a
production-specific deficit**: comprehension is delayed and the delay widens with
level; production is delayed *substantially more* (~12–17 months extra at matched
low vocabulary size, P(>0)=1.00), peaks ~4 years later, and leaves ~90% of
2-year-old children below the TD 10th centile for spoken words. Signing partially —
not wholly — compensates. This is exactly the regime the reserved generative
VG16 would formalise (a directly-estimated, level-indexed expressive-delay
parameter); §4.3 shows the headline is already recoverable from the disjoint
fits with exact intervals, no joint model required.

## 5. Comparison tooling changes (this round)

- Comparison figures are now emitted as **individual, linear-axis** figures
  (`compare_ds_td_re.py`) — the 2×3 / 2×2 subplot grids and all log x-scales were
  removed so each panel is usable on its own.
- New **`compare_ds_td_expressive.py`** — the non-VG16 realisation of the
  "expressive delay" question (every estimand a per-draw functional of the
  disjoint DS×TD posteriors, exact CIs, no joint model): level-indexed Δ_exp(N),
  comprehension-equivalent developmental age, the **sign-inclusive expressive
  gap** (uses the new ie_02 signing via VG14 `p_any`), and the distributional
  "fraction of DS below the TD 10th centile". New estimands live in
  `vocab_growth.comparison` with analytic self-checks.
- Deprecated `compare_ds_td_latency.py` and `compare_ds_td_q_overlap.py` to shims
  delegating to `compare_ds_td_re.run_comprehension_matched` (they duplicated its
  comprehension-matched panels).
