# Between-child heterogeneity: DS is the more variable population on the relative scale

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

## 1. The question and why the existing figures could not answer it

Asked on 2026-08-08, looking at `ds_td_spoken_re_overdispersion`: does the overdispersion panel imply less heterogeneity in the Down syndrome population? The panel shows DS above TD until roughly 24 months and below it thereafter, with TD's φ rising steeply as production explodes.

It does not, and it cannot, for a reason that predates the figure. φ = (κ + n)/(κ + 1) is a function of the Beta-Binomial concentration, and κ is an **observation-level** parameter applied to a `p_obs` that already carries the study and subject random effects. Since [#164](https://github.com/dseinternational/vocabulary-growth/pull/164) put subject REs on VG11/VG12/VG13 — and with VG10 on the DS side — persistent between-child differences are absorbed by the subject scale, and κ describes what is left. So the dispersion panels answer "how much residual noise does the likelihood carry", not "how much do children differ from one another".

The two are not merely different, they are complementary: in the TD models `tau_subject` and `kappa` are an explicit reparameterisation of one shared logit-scale scatter budget (`models.gp_utils.build_variance_partition`), so reading either alone attributes the whole budget to whichever half is in view. `overdispersion_factor`'s own docstring already warned that φ is not mean-independent, because κ is level-driven in this family; the report's prose had nonetheless described it as "mean-independent" and "pure concentration", and had drawn a between-child conclusion for comprehension from it. That prose is corrected in the same change as this analysis.

## 2. Why the obvious contrast is not available

The subject scales are not commensurable across the two sides. VG11/VG12 place a single child intercept on the logit of the outcome (`tau_subject`). VG10 places one on the logit of understood (`tau_subj_u`) and one on the logit of the production **ratio** (`tau_subj_q`), with spoken derived as $p_S = p_U \cdot q$. So VG10 has no spoken child scale to read off, and contrasting `tau_subj_q` against `tau_subject` would set the spread of a conversion ratio against the spread of a level.

What both parameterisations _do_ define is the between-child distribution of a child's own logit for the outcome in question. That is the estimand `comparison.subject_heterogeneity` computes: exactly for a single intercept, and by tensor Gauss–Hermite quadrature over the two independent subject effects for the product form. It also returns σ_child, the induced spread in expected words, which — unlike σ_Y — counts only persistent between-child variation and excludes the Beta-Binomial noise.

The quadrature was checked against 2,000,000-draw Monte Carlo (agreement to ~4 significant figures), verified converged at the shipped 21 nodes against an 81-node grid at the most extreme scales the DS fits could produce, and made stable in the far lower tail via `logit(σ(a)σ(b))` computed as a log-sum-exp rather than a clipped `log p − log(1−p)`, because DS spoken proportions at the young end are small enough that the naive form would report the clip rather than the model.

## 3. Result

DS = VG10, TD = VG11 (spoken) / VG12 (understood), reporting-quality fits, per-draw contrasts on the empirical age overlap.

|                     | τ_TD  | τ_DS      | ratio TD/DS | P(TD > DS) | σ_child TD     | σ_child DS   |
| ------------------- | ----- | --------- | ----------- | ---------- | -------------- | ------------ |
| spoken, 8 mo        | 1.038 | **1.442** | 0.72        | 0.00       | 1.8 w          | 0.7 w        |
| spoken, 18 mo       | 1.038 | **1.389** | 0.75        | 0.00       | 68.3 w         | 6.8 w        |
| spoken, 30 mo       | 1.038 | **1.273** | 0.82        | 0.00       | 172.9 w        | 35.5 w       |
| understood, 8–24 mo | 0.686 | **0.785** | 0.87        | 0.00       | 24.2 → 123.9 w | 9.0 → 84.2 w |

**Children with Down syndrome differ from one another more than typically developing children do**, on both outcomes, at every age in the overlap, with posterior probability ≈ 1.00. The gap narrows with age for spoken (ratio 0.72 → 0.82).

**And the absolute scale says the opposite.** σ_child in words is far larger for TD — 173 against 36 for spoken at 30 months — because TD sits at a much higher level, and the same relative spread around a larger mean is a larger absolute spread. Both statements describe the same posterior. This is the substantive finding: which population is "more variable" depends entirely on the scale, and the absolute reading is the one the σ_Y and φ panels invite.

## 4. Two structural checks that came out as predicted

- τ_TD is **flat in age** on both outcomes, and τ_DS is flat for comprehension — correct, because those are single intercepts on the outcome's own logit, so the SD of the child's logit is the parameter itself.
- τ_DS for **spoken declines with age** (1.442 → 1.273) even though `tau_subj_u` and `tau_subj_q` are both constants. That is a property of the product form passing through two non-linearities, not a finding about children becoming more alike, and it is precisely the reason the contrast cannot be made by reading a parameter off the trace.

## 5. What bounds the claim

- **The priors are not symmetric.** On the TD side `tau_subject` and `kappa` are sampled through the shared-budget reparameterisation with an informative prior on the split; on the DS side they are two free scales. Part of any τ difference is a difference in prior structure.
- **Identification is thin on the TD side.** The TD pool averages 1.21 observations per child, and only repeatedly-measured children identify the τ/κ split at all. `202608050900-td-hierarchical-geometry.md` §12 notes that `tau_subject`'s strikingly tight interval owes substantially to the Beta-Binomial functional form separating child variance from dispersion, not to replication in the data.
- **The intervals are the least trustworthy part.** VG11 carries 22 divergent transitions and VG12 an energy BFMI of 0.207; the same note observes that low BFMI means poor exploration of the energy tails, and tail quantiles are what interval bounds are made of. τ is an interval on a weakly-identified variance split in exactly those models. **The point estimates and the sign of the contrast are the safer part of this result; the interval widths are not.** The figures were published with `sync_report_figures.py --allow-caveats`, so the caveats travel into Appendix B with the numbers.
- **Matched on age, not on level.** At matched age the DS group sits near a floor where absolute spread is mechanically small. A full answer to "are these children more alike" also wants the comparison at matched vocabulary level, which the attainment-delay and matched-comprehension sections provide.

## 6. Artifacts

`comparison.subject_heterogeneity`, `child_spread_single`, `child_spread_product` in `src/vocab_growth/comparison.py`; wired into `scripts/compare_ds_td_re.py`, which now emits `ds_td_<outcome>_re_subject_heterogeneity.csv` plus `subject_tau` and `subject_spread` panels. Reported in the comparison book's "Between-child heterogeneity" section. Six unit tests in `tests/test_comparison.py`; a Monte-Carlo cross-check in the script's `--verify` self-check.
