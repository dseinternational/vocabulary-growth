# Challenging item exchangeability

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> Working note, 2026-07-26. The existence and direction of the finding are established; several magnitudes are provisional and are flagged as such. Nothing here has been actioned in the models. No refit is proposed.

> [!IMPORTANT]
> Revised the same day, before merge, after an independent verification pass that recomputed every checkable number from the raw CSVs and the fitted output. The §3B dispersion table is corrected (an age-standardisation error, recorded in §8 — the correction _strengthens_ the conclusion), §3B now reads `tau_subj_u` alongside `kappa`, and the IRT-feasibility conclusion in §7 is superseded by the companion note [`202607261048-incorporating-item-difficulty.md`](202607261048-incorporating-item-difficulty.md). A second round, responding to review on the pull-request thread, harmonises the §3B comparison on `kappa + 1`, scopes §3A to local independence, states what the §2 test does and does not reject, tightens the age-gradient conservativeness claim, and adds [`scripts/verify_item_difficulty_notes.py`](../scripts/verify_item_difficulty_notes.py).

> [!CAUTION]
> **The title over-claims, and the study owner was right to challenge it (2026-07-26).** "Challenging item exchangeability" reads as an attack on the likelihood. It should not. The finding is that item exchangeability is _false_ (§2, decisively) and yet _nearly costless_ for the quantity the models actually fit — and there is a clean theoretical reason for that which this note failed to state.
>
> Under a Rasch-type item model — items of arbitrary, heterogeneous difficulty but equal discrimination — the total score is a **sufficient statistic** for ability. Writing the log-likelihood as `theta * T - sum_j y_j d_j - sum_j log(1 + exp(theta - d_j))`, the Fisher–Neyman factorisation needs both halves to cooperate, and they do: the term recording _which_ items were passed, `-sum_j y_j d_j`, carries no `theta`; and the `theta`-dependent normaliser runs over the whole **administered set** irrespective of outcome, so it carries no dependence on the response pattern. Given the item difficulties, modelling sum scores therefore discards no information about ability, and heterogeneous difficulty costs nothing on that axis. Rasch is the unique family for which the _unweighted raw score_ is sufficient.
>
> That statement is loose in four ways, each of which matters somewhere in this project; §3A″ states it properly. None of the four reopens this channel. Two of them constrain what may be concluded and from which data, and one turns out to be implemented in the data pipeline already.
>
> That relocates the whole question to: **is the Beta-Binomial an adequate distribution for the total?** Which is measurable, and measured in §3A′ below: the component that item exchangeability governs carries **0.8%–5.3%** of total variance in VG10, and a 40% error in it moves the total SD by **at most 1.1%**. Well below what 613 children can resolve.
>
> So §7's conclusion — documentation only, no refit, do not rebuild the likelihood — is not a cautious compromise. It is the right answer, and this note should have said so in its title. What survives is **§3C (the estimand channel)** and **§3B (the level-scale channel, which was never an exchangeability problem)**. Those, plus the fact that the study's own hypothesis is item-level, are the case for the Route 1 work — not any defect in the aggregate likelihood. See [`202607261048-incorporating-item-difficulty.md`](202607261048-incorporating-item-difficulty.md) and [`202607261210-route1-dif-prespecification.md`](202607261210-route1-dif-prespecification.md), both re-billed accordingly.

## Summary

Every model in the family assumes that, conditional on a child's latent ability, the 810 words of the reference inventory are equally likely to be known. That assumption is false, decisively and in a consistent direction. But three quite different concerns have been travelling under the label "exchangeability", and separating them is most of the value: one is negligible, one costs a definition rather than a number, and the most consequential of the three is **not an exchangeability problem at all** — it is that the Beta-Binomial concentration on a bounded proportion scale cannot be read as latent between-child heterogeneity, which would be true even if every word were equally hard.

Nothing in this note requires a reported mean trajectory or expected word count to change. What it requires is that three passages in the technical report be rewritten, and that one interpretive claim be retired.

## 1. Where the assumption lives

Every likelihood is `y ~ BetaBinomial(810, alpha = p * kappa, beta = (1 - p) * kappa)`. Its generative story has two steps: first draw a child-specific probability `pi ~ Beta(p * kappa, (1 - p) * kappa)`, then draw `y ~ Binomial(810, pi)`.

The first step is the one the report describes: children differ, and `kappa` governs how much. The second step is the one nothing in the documentation mentions. It says that _given_ a child's `pi`, each of the 810 words is an independent Bernoulli trial with the **same** `pi` — conditional on ability, every word is equally likely to be known, and the word list could be permuted without changing anything. That is item exchangeability, and it is logically independent of the between-child homogeneity the Beta step relaxes.

[`methods-models.qmd`](../docs/report/methods-models.qmd) at `§sec-betabinomial` (line 26) conflates the two: it says a Binomial "would assume exchangeable Bernoulli trials within each observation **and** no unmodelled heterogeneity between children", then presents the Beta-Binomial as the remedy. The Beta-Binomial remedies only the second. The first survives untouched into every model of record.

## 2. The assumption is violated, on a test that fits no model

Under item exchangeability, a child who knows `T` of the 810 words has a stratum-`k` count distributed `Hypergeometric(810, n_k, T)`. This is pure combinatorics from the known checklist sizes — no model, no fitted parameter, no distributional assumption beyond the one under test. Standardising ie_01's observed follow-up-wave checklist counts against it (n = 44 children with `0 < T < 810`):

| statistic                                    | observed | exchangeability predicts |
| -------------------------------------------- | -------- | ------------------------ |
| RMS z, all strata                            | 9.61     | 1.0                      |
| mean z, Checklist 1                          | +8.40    | 0                        |
| mean z, Checklist 2                          | +3.00    | 0                        |
| mean z, Checklist 3                          | −9.01    | 0                        |
| children with a positive Checklist 1 excess  | 37 / 44  | ~22 / 44                 |
| children with a negative Checklist 3 deficit | 42 / 44  | ~22 / 44                 |
| children with `p_C1 >= p_C2 >= p_C3`         | 79.5%    | 19.2%                    |

The sign tests give p = 5.3e-06 (Checklist 1) and p = 1.1e-10 (Checklist 3). For the monotonicity rate, none of 2000 simulations drawn under exchangeability reached the observed 79.5% (simulated maximum 40.9%). Pooled proportion understood, over the 46 complete follow-up records, is 0.671 / 0.452 / 0.253 across Checklists 1 / 2 / 3, where exchangeability requires all three to equal the overall 0.398 (per-child means are slightly higher, 0.701 / 0.473 / 0.264 over the 44 tested).

Worth stating plainly: the DSE checklists were **constructed** as a difficulty ladder, so finding `C1 > C2 > C3` is partly instrument validation. The corollary is the point — the instrument's own design contradicts within-observation exchangeability, so the assumption was never plausible for this inventory.

One scope clarification, so the finding is not over-read: a likelihood on totals is silent about which words make up a count, so this test rejects the equal-item _completion_ of the aggregate model — the item-level reading invoked whenever `kappa` or `q` is interpreted as a property of individual words — not the distribution of the totals themselves. That is why §5 can leave the total-count likelihoods standing.

## 3. Three channels, which differ by two orders of magnitude

### 3A. The count-variance channel — negligible, and the intuition is backwards

For independent Bernoulli trials there is an exact identity, `sum_i p_i (1 - p_i) = N * pbar * (1 - pbar) - N * Var_items(p)`, verified here to machine precision. Item-difficulty heterogeneity therefore **reduces** within-child count variance; the natural intuition that "items differ, so there is extra noise" is the wrong way round. At the measured difficulty spread (~1.8 logits between the outer checklists) the reduction reaches 9% of within-child variance.

That does not propagate, because the between-child term carries `N^2 = 810^2`. Within-child variance is only 0.6%–4.6% of total count variance across the plausible range of ability spreads and levels, so the effect on a recovered `kappa` is about 1%. Confirmed by simulating through the project's own likelihood: `kappa` moves by under 1.1%.

This channel can be set aside — with its scope stated: the identity, and the smallness, are claims about _difficulty heterogeneity under local independence_ (items independent given ability). Item dependence beyond ability — semantic clustering, prerequisite structure — adds covariance terms that can scale as `N^2` and is a separate, unquantified channel; operationally it is part of what the observation-level `kappa` already absorbs, which reinforces §3B's residual-dispersion reading.

### 3A′. Why 3A was the only distributional channel, and a correction to its magnitude

§3A computed the right quantity without saying why it was the only one that mattered. The reason is Rasch sufficiency (see the caution at the head of this note): if the total is sufficient for ability, then — _given the item difficulties, and given the administered set_ — nothing about the item composition can affect inference on ability, and the only way heterogeneous difficulty can reach the model is through the **distribution of the total**. §3A is that channel. There is no third distributional route to look for. Both conditions in that sentence are doing work, and §3A″ discharges them.

Two things follow, and they pull in opposite directions.

**The kernel misspecification is larger than §3A said.** §3A took its difficulty spread from the ~1.8-logit gap between the outer checklists, which is a _between-stratum_ gap standing in for the _item-level_ spread — a lower bound, since it ignores difficulty variation within each checklist. Treating the three checklists as three difficulty points implies a standard deviation near 0.7–1.0 logits, which is where the quoted 9% comes from. A realistic full-inventory spread for a CDI is 1.5–2.5 logits, and the underdispersion at that spread is much larger:

| difficulty SD (logits) | Var(Poisson-binomial) / Var(Binomial), at `pbar` 0.15–0.75 |
| ---------------------- | ---------------------------------------------------------- |
| 1.0                    | 0.84–0.89                                                  |
| 1.5                    | 0.72–0.80                                                  |
| 2.0                    | 0.60–0.67                                                  |
| 2.5                    | 0.52–0.61                                                  |

So the within-child kernel may be wrong by ~40% rather than 9%. Note also that the Beta-Binomial cannot represent this at all in principle: its variance is `N p (1 - p) (N + kappa) / (kappa + 1)`, which is `>= N p (1 - p)` for every `kappa`, so the family has the Binomial as a _floor_ and underdispersion is outside it. `kappa` absorbs the difference.

**And the share is now measured rather than bounded.** §3A gave within-child variance as 0.6%–4.6% of total across a plausible range. The fitted models pin it: `posterior_kappa_*.csv` carries the variance inflation factor `(N + kappa) / (kappa + 1)`, whose reciprocal is the kernel's share.

| model / outcome     | fitted `kappa` | variance inflation | share of total variance in the kernel | worst-case effect on total SD |
| ------------------- | -------------- | ------------------ | ------------------------------------- | ----------------------------- |
| **VG10** understood | 5.3–44.0       | 19×–129×           | **0.77%–5.27%**                       | **1.06%**                     |
| **VG10** spoken     | 5.8–6.0        | 117×–120×          | 0.83%–0.86%                           | 0.17%                         |
| VG07 understood     | 3.3–11.3       | 67×–191×           | 0.52%–1.49%                           | 0.30%                         |
| VG07 spoken         | 2.3–2.5        | 234×–244×          | 0.41%–0.43%                           | 0.09%                         |

Take the figures from **VG10**, the model of record: at its youngest ages `kappa_u` reaches 44, the kernel carries 5.3% of total variance, and a 40% error in it moves the total standard deviation by 1.06%. This is the widest exposure anywhere in the family, and it is still an order of magnitude below what these data resolve. §3A's estimated range (0.6%–4.6%) was close but slightly low at the top end.

The correction to the magnitude therefore strengthens §3A's conclusion rather than weakening it: the misspecification is larger than stated, the exposure is slightly larger than stated, and the product is still negligible — now against fitted output rather than an assumed range.

> [!NOTE]
> An earlier draft of this subsection quoted VG07's figures (0.4%–1.5%, ≤0.30%) as "the fitted models". VG07 is not a model of record; its `kappa` is far lower than VG10's, which understated the exposure by a factor of three. The numbers above are checked by [`scripts/verify_item_difficulty_notes.py`](../scripts/verify_item_difficulty_notes.py) against VG10's own output so the mistake cannot recur silently.

One assumption is worth naming because it is load-bearing here and elsewhere: Rasch sufficiency requires _equal discrimination_ exactly. CDI items do not have it. That leaves the channel above untouched — the variance identity holds for _any_ set of per-item probabilities, however they arise, so the Poisson-binomial arithmetic and the ~1% budget stand whatever the discriminations are.

It does, however, open a second and distinct channel that this note does not quantify. Under a 2PL the sufficient statistic is the _weighted_ score `sum_j a_j y_j`, which a total cannot recover; so with unequal discrimination the raw count carries a genuine loss of information about ability, not merely a distributional perturbation. Its size depends on the spread of `a_j`, nothing in §3A or §3A′ bounds it, and — by §3A″ point 3 — it cannot be bounded from count data at all. The same assumption also bears on Route 1, where a 1PL fitted to data with varying discrimination can present as spurious differential item functioning; that is a threat to the DIF verdict rather than to the aggregate likelihood, and the pre-specification now carries it as a mandatory 2PL sensitivity.

### 3A″. What sufficiency does and does not give

Four qualifications on the statement at the head of this note, prompted by review on the pull-request thread. None reopens the distributional channel.

**1. The administered set is part of the condition — and this project already treats it as one.** The normaliser `sum_j log(1 + exp(theta - d_j))` runs over the items actually put to the child, so sufficiency of `T` is sufficiency _given the administered set_. Where that set varies between children and the model does not condition on it, the total alone is no longer sufficient: the pair (total, set) is. This is a live condition here rather than a technicality, because all fifteen model definitions score every count against a fixed `n_trials = 810` while the pooled Down syndrome data carry nine distinct form ceilings — 396, 408, 416, 460, 670, 675, 680, 690 and 810. It is discharged in two different ways, both already in [`data_utils`](../src/vocab_growth/data_utils.py):

- The shorter MacArthur-derived forms (Oxford 416, MB-CDI Words & Gestures 396, NZCDI 675) are treated as _nested_ instruments whose absent items are the rarer, later-acquired words an ability-matched child mostly does not know. That is checked rather than assumed: a dual-form crosswalk fitted to the `uk_02` children who took both the DSE and Oxford forms puts the fixed-810 count ratio near 1 across the range where the short forms are administered.
- The `ie_01` baseline wave (ceiling 460) omitted a whole 350-item subscale of the _same_ instrument, and there the nesting argument fails — at matched vocabulary the follow-up wave puts about 9.5% of Checklist 3 known against 0% at baseline. Those counts are masked by `mask_incomplete_administrations` rather than rescaled.

Worth noticing what both rules rest on. "The absent items are the harder ones" is a claim about the difficulty vector, and under item exchangeability it could not be written down at all: every item would be equally likely to be known, and any partial administration would rescale by simple proportion. So item heterogeneity is **load-bearing in the data pipeline while being near-costless in the likelihood** — and those two facts are not in tension. Sufficiency is what makes a count an adequate summary _once the administered set is fixed_; these two rules are what fix it.

**2. Sufficiency holds conditional on the difficulties being known.** It is sufficiency for ability _within_ the model, with `d` fixed. The response pattern is not uninformative in general: conditional on `T`, its distribution depends only on the `d_j` — the `theta` cancels exactly — which is precisely what conditional maximum likelihood exploits to estimate item parameters free of ability. The pattern is ancillary for ability and informative for the items.

**3. The converse is uncomfortable, and it is why Route 1 needs item-level data.** The property that makes counts lossless for ability is the same property that makes them carry _no_ information about the item parameters. Counts therefore cannot test whether the Rasch assumption holds, cannot detect varying discrimination, and cannot validate a difficulty ordering borrowed from Wordbank's typically-developing norms and applied to Down syndrome. Sufficiency is not free: it is conditional on a model that the data in hand provide no means of checking. This is already the stated justification for Route 1 ([pre-specification](202607261210-route1-dif-prespecification.md) §1), and it also fixes where the mandatory 2PL sensitivity can live — on the item-level Wordbank pull (§8.6b there), never on aggregate counts.

**4. Uniqueness, stated precisely.** The result is Rasch's, formalised by Birnbaum (1968) and proved under regularity conditions by Andersen (_Psychometrika_, 1977): given unidimensionality, local independence and strictly monotone continuous item characteristic curves, if the unweighted raw score is sufficient for ability then the model must be Rasch. Two qualifications on "unique". It is uniqueness for the _raw_ score — a 2PL also has a sufficient statistic, `sum_j a_j y_j`, just not one computable from a total. And the property extends to the polytomous Rasch models (partial credit, rating scale), so "family" is the right word rather than the 1PL specifically.

**And one thing sufficiency does not give at all.** It licenses the _summary_, not the _likelihood_. Given ability, `T` is a sum of independent non-identical Bernoullis — Poisson-binomial, not Binomial — and sufficiency says nothing about which distribution to place on it. That question is not closed by the argument at the head of this note. It is answered, approximately and with its magnitude bounded, by §3A′ above: the direction is known (underdispersion, which the Beta-Binomial cannot represent at all, so `kappa` absorbs it) and the consequence is at most about 1% of the total standard deviation in the model of record. An approximation with a measured cost, not an identity.

### 3B. The level-scale channel — the consequential one, and not about exchangeability

Children's abilities live on a latent logit scale, on which the proportion scale compresses near 0 and 1 and stretches near 0.5. With a **constant** latent standard deviation `sigma`, the delta method gives `Var(p) ~= [p (1 - p)]^2 * sigma^2`, while the Beta-Binomial encodes between-child variance as `p (1 - p) / (kappa + 1)`. Equating the two:

```text
kappa + 1  ~=  1 / [ p (1 - p) * sigma^2 ]
```

So `kappa` **must** fall as `p` rises toward 0.5, with no change whatever in latent spread — a 6.7-fold decline from `p = 0.05` to `p = 0.5` at `sigma = 1` by this first-order formula (exact integration of a logit-normal ability distribution gives 16.3 → 4.8, still a 3.4-fold decline) — and must rise again above 0.5. The relationship is U-shaped, minimised at `p = 0.5`. The fitted form `kappa(z) = kappa_min + exp(a_kappa - b_mag * z)` with `b_mag > 0` imposed is monotone, so it cannot represent that shape; it fits a one-way trend through it.

This effect is present under _perfect_ exchangeability. It is a bounded-inventory scale-and-link property, surfaced by the exchangeability investigation rather than caused by the thing under investigation. Item heterogeneity's own contribution to the `kappa` trend runs the other way and is small: at matched level, heterogeneity inflates `kappa` by a fairly uniform 10%–13% and _flattens_ the level-driven decline by about 2.5%–3% (exact integration at `sigma = 1`; the companion note's §1 table is the same computation).

**What this means for the report.** `§sec-kappa` (line 47) is careful to call the monotone decline an **assumption** rather than an inference, and that framing is correct and should be kept. The problem is the parenthetical defending it, which states that because the outcome is bounded the decline "allows greater heterogeneity around the mean trajectory on the latent scale". That specific reassurance is backwards: `kappa` is not a latent-scale heterogeneity parameter, and constant latent heterogeneity already implies a declining `kappa`. Two facts from VG10's own output sharpen this. First, VG10 already carries an explicit between-child latent spread — the subject random intercept scale `tau_subj_u`, posterior mean 0.754, constant by construction — _underneath_ the Beta step, so `kappa` there is residual, occasion-level dispersion, which is one more reason it cannot be read as "children fanning out". Second, reading the model's own fitted dispersion (`posterior_kappa_u.csv` medians; the parameter posteriors are `a_kappa_u = 2.837`, `b_kappa_mag_u = 0.557`, `kappa_min_u = 2.996`) over the ages where DS comprehension data exist:

| age (months) | fitted `p` | fitted `kappa` | implied residual latent SD | total, with `tau_subj_u` |
| ------------ | ---------- | -------------- | -------------------------- | ------------------------ |
| 12           | 0.028      | 39.77          | 0.956                      | 1.218                    |
| 24           | 0.146      | 29.64          | 0.512                      | 0.911                    |
| 30           | 0.227      | 25.66          | 0.462                      | 0.884                    |
| 48           | 0.379      | 16.97          | 0.486                      | 0.897                    |
| 66           | 0.526      | 11.68          | 0.562                      | 0.941                    |

On the `kappa + 1` scale the identity governs, the fitted decline over 12–66 months is 1.168 log units where a constant latent spread alone predicts 2.230 — a ratio of 0.52 (on `kappa` itself, 1.225 log units; all of this is point arithmetic on posterior summaries, with no posterior intervals propagated). The fitted parameters are therefore consistent with residual latent spread that **falls** steeply to about 30 months and then rises modestly — and with total between-child spread (subject intercept plus residual, last column) that falls to about 30 months and is nearly flat thereafter — not with spread that grows throughout. Two honest qualifications: the profile is not monotone, and the 12-month end sits at `p = 0.028`, where the delta-method reading is least reliable and the data thinnest — which is itself the point, because the direction one infers depends on where the range is taken to start. And because `b_mag > 0` is imposed, the exponential-decline family cannot express the U-shaped level effect at all, so a positive fitted `b_mag` is by itself uninformative about latent divergence; what carries information is its magnitude read against the level-effect benchmark, which is the ratio of 0.52 above.

Mitigating, and worth recording: `discussion.qmd`, `summary.qmd` and the signed-vocabulary results chapter are TODO stubs, and the one substantive results chapter — [`results-words-understood-spoken.qmd`](../docs/report/results-words-understood-spoken.qmd), filled in #182 — contains no dispersion or fan-out prose, so no published prose currently draws the "children fan out with age" inference. It appears as a recommendation in [`202607121200-statistical-model-review.md`](202607121200-statistical-model-review.md) §5, which proposes leading with `posterior_kappa` to communicate "the clinical 'children fan out' message". Acting on that recommendation would be the error.

### 3C. The estimand channel — the genuine exchangeability bite

`q(a) = P(speak | understand)` applied to a child's `U` understood words treats those words as a random draw from the inventory. They are not: they are the `U` easiest words. And production propensity is itself steeply difficulty-graded. Within the same child, the log-odds gap between the outer checklists has median +2.53 — an odds ratio of about **12.6** — positive in 27 of 30 children, exact Wilcoxon p = 4.7e-07 (a +0.5 continuity correction on the stratum counts; construction in §8).

`q` is therefore a composition-weighted average over each child's own understood set, not an item-level probability. `§sec-ratios` (line 111) defines it as the latter. Three consequences follow: `q`'s **level** is not transportable to another inventory; it is not comparable between populations matched on age; and it is acutely sensitive to how it is summarised, which no item-level probability would be.

| summary of `q_k` (n = 38 coherent records) | C1    | C2    | C3    |
| ------------------------------------------ | ----- | ----- | ----- |
| ratio of sums                              | 0.667 | 0.526 | 0.521 |
| median per child                           | 0.683 | 0.364 | 0.255 |
| mean per child                             | 0.612 | 0.414 | 0.318 |

The gradient roughly halves between the first and second rows. The ratio of sums weights children by vocabulary size, and the top quartile of children hold 53% of all understood words; those children have high `q` in every stratum, which compresses the between-stratum contrast. Both summaries are defensible and they answer different questions — which is exactly why `q` needs defining as the composition-weighted quantity it is.

**The rise in `q` survives, and the composition bias is conservative in sign.** As comprehension grows, a child's understood words shift toward the harder checklists (Checklist 1 weight 0.422 → 0.210, Checklist 3 weight 0.124 → 0.312 between the low and high halves of the sample), which drags marginal `q` _down_. A Kitagawa decomposition of the low-to-high difference gives:

| component             | change in `q` | share  |
| --------------------- | ------------- | ------ |
| total                 | +0.3399       | 100%   |
| within-stratum        | +0.3950       | 116.2% |
| composition (weights) | −0.0550       | −16.2% |

So composition explains none of the rise and works against it. One caveat that matters: this is a split on **comprehension level**, whereas the models report an **age** gradient. Across the wider set of specifications tried in the supporting investigation the composition share was negative in every one, ranging −2.8% to −35.2%, but for the age gradient specifically it was only −2.8% to −9.9% with intervals spanning zero. The defensible statement is: composition explains none of `q`'s rise; the conservative sign is established for the comprehension-level split (negative in every specification tried); for the age gradient the point estimates suggest attenuation and the intervals span zero, so no more than that should be claimed.

## 4. The matched-comprehension DS-versus-TD contrast

Because `q` is composition-dependent, matching two populations on the _number_ of words understood controls composition only if the difficulty ordering is the same in both. The headline result reproduces exactly (peak `Delta q` = +0.063 at `N` = 175, 89% interval +0.021 to +0.095), and the one quantifiable threat — instrument non-equivalence — is bounded rather than fatal. Applying the project's own dual-form crosswalk consistently to numerator and denominator moves the peak to about +0.045 with break-even around `R` = 1.25, rising to roughly 1.78 once shared short-form exposure is allowed for (only 32.4% of DS observations in the band driving the result are on the 810-item form). Two unmodelled effects push the peak up rather than down.

What binds is not measurable here. At every credibly positive level the DS children are 8–13 months older than the TD children (29.2 versus 16.8 months at `N` = 175), so any age-dependent composition difference is fully confounded with the population contrast; and whether word-difficulty ordering is the same in the two populations — the assumption matched-`N` actually needs — cannot be tested with anything in this repository, because there is **no item-level or category-level typically-developing data** here. This must be stated as a limitation rather than bounded with a fragile number.

These crosswalk figures come from the supporting investigation and were not independently reproduced for this note; the reproduced quantity is the published `Delta q` table itself.

## 5. What is not affected

Worth recording explicitly, to forestall over-correction:

- **Mean trajectories and expected word counts.** Unaffected by construction wherever data constrain the fit: `E[y] = 810 * p(z)` whatever the composition of the words a child knows, and the HSGP mean is flexible enough to absorb link misfit in-sample. Where the fit extrapolates — past ~60 months and toward the ceiling — the link's shape does real work, which is the territory of the companion note's difficulty-mixed-link proposal.
- **The DS-versus-TD dispersion ratio.** The item-difficulty contribution to the overdispersion factor is additive and small, moving the ratio by 0.2%–1.2% against a reported difference of four to five fold.
- **The count-variance accounting inside the likelihood.** Under 1.1% on `kappa` (§3A).

## 6. Evidence base, and what limits it

Every sub-inventory number here rests on two Down syndrome studies: ie_01's follow-up wave (46 children, 27–86 months, the only source recording checklist-level counts) and uk_01's 19 comprehension categories (29 children with complete category data). There is no item-level or category-level typically-developing data in the repository. The defensible grading is: **existence and direction — strong; magnitudes — provisional, single wave, single study, n < 50.**

Three data problems bear directly on the magnitudes and must be settled before any stratum number is published:

1. **The ie_01 baseline wave is defective beyond the missing Checklist 3.** Pooled Checklist 1 comprehension _falls_ from 0.855 to 0.671 between waves while the mean understood total rises from 252 to 323 words, and Checklist 1 comprehension decreases for 22 of 46 children (minimum −124). At least one wave's Checklist 1 field is unreliable. (The missing-Checklist-3 defect is handled separately, by the masking rule added in #182.)
2. **The Checklist 1 denominator is contradicted by the data.** 27 records carry Checklist 1 counts above the nominal 120 items, to a maximum of 124. Either the checklist has 124 items or those records are miscoded; the source codebook should settle it. The exchangeability test above is insensitive to this (the Checklist 3 deficit does not depend on the Checklist 1 denominator at all), but the difficulty magnitudes are not.
3. **uk_01's `understood` appears to exclude words the child also produces.** `spoken / understood` exceeds 1 for 2 of 29 children (maximum 1.95), which an inclusive comprehension field cannot do. Scope is 29 of 680 understood observations, but the bias is one-directional and sits where `q` is least identified. Verify against the source codebook before changing the pipeline.

A screening rule covering the internally inconsistent records is a **prerequisite** for publishing any stratum table, not a parallel task: the headline proportions move materially under screening (0.710 / 0.321 / 0.095 unscreened against 0.869 / 0.327 / 0.059 on a cleaned subset in the supporting investigation), as does the implied difficulty spread.

## 7. Recommended actions

No refit is proposed, and the likelihood should not be rebuilt. A stratified three-level likelihood could be supported by 46 of 680 understood observations from one wave of one study, against a blast radius of dozens of likelihood call sites plus a reporting-quality refit and recovery pass for every model of record — and the resulting model is already known to be misspecified. A full IRT reformulation is not feasible with the aggregate counts held in this repository — but this conclusion is superseded by the companion note (§5): with the item-level data reported available for roughly 218 children, a calibrate-the-instrument-then-consume-the-calibration route becomes feasible, which is a different thing from rebuilding the aggregate likelihood.

**Now, documentation only.**

1. Settle the ie_01 screening rule and the Checklist 1 item count (§6). Everything else depends on it.
2. Rewrite `§sec-betabinomial` (line 26) to separate within-observation item exchangeability from between-observation homogeneity of `p`, state that the former is false, and name what is and is not affected.
3. Rewrite the `§sec-kappa` parenthetical (line 47). Keep the "this is an assumption" framing; remove the latent-scale defence; state that the fitted decline in `kappa` is quantitatively consistent with unchanged latent between-child spread on a bounded inventory, that the monotone form cannot represent the U-shaped level effect, and that no corrected estimate of real divergence is available from these data. The same rewrite should touch the section's opening motivation ("children making more progress diverge from those making less") and the `§sec-betabinomial` gloss "(more between-child heterogeneity at a given age)" — in the RE models `kappa` is occasion-level residual dispersion above the subject intercept, so both carry the same conflation. The fan-out message that _does_ survive is on the count scale: posterior-predictive spread widens roughly five-fold from 12 to 66 months, so the `P(Y<=k)` columns and predictive intervals — not `posterior_kappa` — are the artefacts to lead with.
4. Correct the definition of `q` at `§sec-ratios` (line 111) to the composition-weighted average it is and give the measured stratum gradient. The results chapter's Limitations block already carries the composition caveat (added in #182), so the report-side gap is narrower than first drafted: state the conservativeness claim only at the strength §3C licenses (level-split attenuation; age-gradient intervals spanning zero), and give the one limitation still lacking comparable prominence — that at matched comprehension the Down syndrome children are 8–13 months older (§4).
5. Fill the [`_caveats-ds.qmd`](../docs/report/_caveats-ds.qmd) stub with the screened stratum table and the untestable-ordering statement.
6. Reword the claim at [`comparison.py:502`](../src/vocab_growth/comparison.py) that contrasting the overdispersion factor "isolates the pure concentration difference". It is true only in the narrow sense that the factor removes the explicit `p (1 - p)` mean dependence at fixed `kappa`; because `kappa` is itself level-driven (§3B), the cross-population contrast is not a _pure_ concentration difference. The reported ratio is robust (§5) — the docstring should claim the narrow thing.
7. Open issues, do not fix inline: the uk_01 exclusive-coding question, and the ie_01 baseline Checklist 1 defect.

**Next milestone.** Add a registered `dse-native-only` sensitivity for VG10 and VG15 (`survey_vocab_max == 810`: 259 of 680 understood observations, 194 of 626 children, four sources). This is the highest-value item after the documentation, because pooling 396-, 408-, 416-, 675- and 680-item forms onto an 810 denominator is where the difficulty-ordering assumption does silent load-bearing work, and it has never been tested inside the models.

**Only if data arrives** — and the companion note (§5) reports that it now has: item-level records for ie_01, a second UK study, and the Wordbank corpus. Per-item DSE response vectors for DS children **and** item-level Wordbank responses mapped onto the DSE item pool are what make IRT and a real differential-item-functioning test possible. Nothing short of both, because the question that matters is whether the ordering is the same across populations, and that needs typically-developing items. Meanwhile, add per-item response retention to any future DSE data-collection protocol: it is free at collection time and impossible to reconstruct wherever responses were not retained.

## 8. Reproducing the numbers

All figures in §2, §3 and §6 were computed directly from `data/vocab_data_ie_01.csv` and `data/vocab_data_uk_01.csv`, and from VG10's `diagnostics.csv`, `posterior_summary_u.csv` and `posterior_kappa_u.csv`. Three points of method worth recording so the numbers can be checked rather than trusted:

- The hypergeometric test uses stratum sizes 120 / 340 / 350 and the follow-up wave only, restricted to children with `0 < T < 810`; the exchangeability null for the monotonicity rate is simulated by drawing each child's actual `T` uniformly without replacement from 810 positions, 2000 replicates.
- `q_k` is reported on records that are internally coherent (`says_k <= understands_k` for all three strata), which is 38 of 46. Note that screening on this criterion is itself selection on the outcome; the within-child paired log-odds contrast is the robust statement and does not depend on the choice. That paired contrast adds 0.5 to all four cells (a Haldane-style continuity correction) and includes every coherent child with a non-empty understood set in both outer checklists, which is where n = 30 comes from; the p-value is the exact signed-rank distribution.
- §3B's `kappa` column is the `kappa_median` of VG10's `posterior_kappa_u.csv` at the tabulated ages — equivalently, the posterior-mean parameters pushed through `kappa_min + exp(a_kappa - b_mag * z)` with the **model frame's** age standardisation (mean 40.5 / SD 20.7 months). The implied residual latent SD is a first-order delta-method reading against the population `p`, not a refit; the total column adds the posterior-mean `tau_subj_u = 0.754` in quadrature; and the decline comparison is made on `kappa + 1`, the scale the identity governs. A latent-scale reparameterisation would be needed to estimate any of this properly, which is one reason `kappa` should not be asked to carry the interpretation.

A committed script, [`scripts/verify_item_difficulty_notes.py`](../scripts/verify_item_difficulty_notes.py), recomputes the checkable figures in §2, §3B, §3C and §6 — and the companion note's tables — from the raw CSVs and, where present, the fitted output. Two magnitudes are not yet covered by committed code and remain from the supporting investigation's simulations: the under-1.1% likelihood-level effect on `kappa` (§3A) and the 0.2%–1.2% movement in the dispersion ratio (§5). Treat both as provisional until regenerated.

Two corrections to earlier working claims, recorded so they are not propagated. First, uk_01's category columns **do** reconcile exactly with the recorded totals — the 19 comprehension categories sum to `understood` for all 29 complete rows. An earlier report of a 45-word discrepancy was an artefact of a column-matching pattern that split `verb14c` from `verbs14v` and missed `adject15`, `inounv` and `iverbv`. Second, the first circulated version of this note (and the pull-request description) tabulated §3B's fitted `kappa` with age standardised on the understood subset (mean 34.6 / SD 16.6) instead of the model frame's statistics (mean 40.5 / SD 20.7). That understated `kappa` at older ages by up to a quarter (8.94 against 11.68 at 66 months) and overstated the fitted log decline, so the superseded headline figures — a 1.484 log-unit decline and a ratio of 0.67 — become 1.168 and 0.52 on the `kappa + 1` scale the identity governs (1.225 log units on `kappa` itself); the correction _strengthens_ the conclusion, and the values above now come from `posterior_kappa_u.csv` directly.
