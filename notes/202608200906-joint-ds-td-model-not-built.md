# The generative joint DS–TD model: why it is not built

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Decision record, 2026-08-20. Consolidates a decision first taken on the [#122](https://github.com/dseinternational/vocabulary-growth/pull/122) thread (2026-06) and asserted in one sentence of the comparison book ever since, without its reasoning written down anywhere. Nothing here is new modelling: every figure is read from the current reporting-quality comparison tables, and the disjointness check in §3 is reproduced by the snippet in that section. §7 states what would reopen the decision.

## 1. What was proposed

[#114](https://github.com/dseinternational/vocabulary-growth/issues/114) (Q3, comprehension–production gap) opened with two deliverables. The first, a manuscript synthesis, was delivered. The second was a model:

> Implement VG14 (joint DS–TD on shared scale) per the scoping note; register + configure.

— _"the planned **VG14** joint DS–TD model (`notes/202606121500-vg14-scoping-note.md`) putting DS and TD on a shared scale for a posterior contrast of the gap."_

**That scoping note was never written.** `git log --all --diff-filter=A` over the full history returns no file matching `*scoping-note*` other than `202607031200-vg16-within-child-scoping.md`. The design was cited as though it existed, and the VG14 number went to the signing model instead. The joint model was subsequently reserved as VG16 until that number was taken by the within-child cross-lag, and it now **holds no reserved number**: `VG17`/`VG18` are the exploratory sign-group modules and `VG19` is the child-slope plan, so a build today would take VG21.

## 2. The decision

[#122](https://github.com/dseinternational/vocabulary-growth/pull/122) closed #114 having declined it, under "Option 1, per the scoping discussion": _"manuscript synthesis over existing fits — no new joint DS–TD model … a formal joint model would land in the same range at much higher cost, so it's noted as future work."_

The comparison book has carried the position since 2026-06-28 (`e83233e`), in a single parenthesis, and the results chapter defers to a Limitations entry that was never added. This note is the missing reasoning. **The decision stands, and the evidence accumulated since has strengthened rather than weakened it.**

## 3. Reason 1 — the joint posterior already factorises, exactly

The two pools share no studies and no children. Verified against the current loaders rather than asserted:

```python
from vocab_growth.data_utils import load_data, load_combined_data, TD_POOL_EXCLUDED_DATASETS
from vocab_growth.models.definitions import ENGLISH_AND_ROMANCE_LANGUAGES

ds = load_combined_data()
td = load_data("td", ["study", "subject_id", "age", "understood", "spoken"],
               languages=ENGLISH_AND_ROMANCE_LANGUAGES)
set(ds.study) & set(td.study)          # -> set()          (14 vs 9 study codes)
set(ds.subject_id) & set(td.subject_id)  # -> set()        (781 vs 5,954 children)
TD_POOL_EXCLUDED_DATASETS              # -> ('Edgin',)
```

The exclusion is structural, not incidental: the TD query requires `typically_developing = true AND health_conditions IS NULL`, and `Edgin` — the Wordbank Down syndrome cohort that supplies the DS pool's `us_01` — is named in `TD_POOL_EXCLUDED_DATASETS` on top of that. The one dataset that could have straddled both sides is excluded twice over.

With disjoint data and no shared parameters, $p(\theta_{DS}, \theta_{TD} \mid y) = p(\theta_{DS} \mid y_{DS})\, p(\theta_{TD} \mid y_{TD})$. Pairing posterior draws from the two separate fits is therefore not an approximation of a joint fit — **it is the joint posterior**, and every contrast formed that way already has an exact credible interval. A joint model that shares no parameters across populations would reproduce the same numbers at several times the sampling cost.

So the question is never "joint or separate". It is **"share what, and on what evidence"** — and each candidate for sharing is an assumption that moves the answer rather than sharpening it.

## 4. Reason 2 — the four things that could be shared, assessed

### 4a. A shared trajectory shape with a population time-shift — already rejected by the fitted data

This is the scientifically interesting form: make "delay" a parameter by asserting $\mu_{TD}(a - \Delta) = \mu_{DS}(a)$. It is the model the project's working assumption — _same order, but later_ — most naturally implies.

**A constant $\Delta$ is decisively rejected by the existing outputs**, which estimate the level-indexed delay directly as a per-draw functional. From `ds_td_{spoken,understood}_re_attainment_delay.csv` (DS `VG20` against TD `VG11` for spoken and `VG12` for understood — `compare_ds_td_re.py`'s `TD_KEYS` — `rep` fits, rows at coverage 1.0):

| words | spoken delay (mo) | understood delay (mo) |
| ----: | ----------------: | --------------------: |
|    40 |              19.1 |                   7.3 |
|   100 |              24.0 |                   9.4 |
|   200 |              30.7 |                  16.1 |
|   300 |              39.9 |                  25.8 |
|   400 |              50.3 |                     — |

A time-shift model predicts a flat column. Spoken delay grows 13.5 → 50.3 months between 10 and 400 words, understood 7.3 → 25.8 between 40 and 300 — factors of 3.7 and 3.5, with 89% intervals far too narrow to admit a constant (at 400 words, [45.7, 56.9]; at 40 words, [18.1, 20.2]). **These children are not on the typical trajectory shifted later; they are on a slower one.** A joint model would formalise a test whose answer is not in question, and a joint model that _imposed_ the shift would misreport this as a nuisance parameter.

### 4b. Shared item difficulties — untestable at the aggregate level, by construction

The other half of _same order, but later_ is measurement invariance of the item difficulty vector, and it is genuinely open. But no aggregate joint model can test it: under a Rasch-type model the total score is sufficient for ability, which is precisely the statement that a sum score carries **no** information about _which_ items a child knows ([`202607261540`](202607261540-item-difficulty-and-the-aggregate-likelihood.md) §3). That is the entire justification for Route 1 ([`202607261210`](202607261210-route1-dif-prespecification.md)), which needs item-level data and a differential-item-functioning design — not a larger growth model.

This is the strongest single point against the build. **The joint model cannot test the one part of the hypothesis still open, and the part it could test is already answered.**

### 4c. Symmetric priors — a real gap, but it does not need a joint model

[`202608081530`](202608081530-ds-td-between-child-heterogeneity.md) §5 records prior asymmetry as a live bound on a published claim: the TD side samples `tau_subject` and `kappa` through the shared-budget reparameterisation with an informative prior on the split, while the DS side carries two free scales, so _"part of any τ difference is a difference in prior structure."_

A joint model would fix this by construction. So would matching the two engines' prior blocks, which is a far smaller change and leaves the factorisation intact. **The gain is real and the joint model is not the cheapest route to it** — this is the residual worth pursuing, and it is recorded as the constructive follow-up in §7.

### 4d. Borrowing strength where one side is thin — rejected

Superficially the most attractive use: TD comprehension stops at 25 months, DS comprehension runs to 84, so let the joint model extrapolate the TD side. **The TD comprehension data above 25 months is not thin, it is absent** — Wordbank's Words & Sentences and TEDS forms do not measure comprehension at all, and their `comprehension` column is production duplicated (WS: 8 of 10,438 rows have comprehension exceeding production; TEDS: 0 of 21,919). Borrowing DS trajectory shape to populate a range where the instrument asked no comprehension question would be extrapolation presented as inference. This is the one candidate that would make the reporting worse.

## 5. Reason 3 — the coverage limits are limits of the data, not of the model

The recurring intuition is that the contrasts stop early because they are assembled from separate fits. They stop early because the posteriors stop. Coverage — the fraction of draws reaching a level within the reporting window — is read from the same tables:

| contrast                      | coverage 1.0 to | last non-zero | binding constraint                                      |
| ----------------------------- | --------------- | ------------- | ------------------------------------------------------- |
| spoken attainment delay       | 400 words       | 450 (0.0005)  | genuine: both populations reach it                      |
| understood attainment delay   | 300 words       | 350 (0.732)   | TD comprehension measurement stops at 25 months (§4d)   |
| production ratio at matched U | 200 words       | 200 (then 0)  | **VG13's 18-month window**, not the data — under repair |

The third row is the exception that proves the rule, and it is worth stating carefully because it cuts the other way. TD `q` coverage falls from 1.000 at 200 words to exactly 0.000 at 250 — a cliff, not a decay, which is the signature of a window setting rather than a data limit. [`202608171500`](202608171500-reporting-scope-audit.md) traced it to VG13's `max_age_months = 18`, which discards roughly 690 rows carrying coupled comprehension _and_ production at 19–25 months. The `window-25` and `window-22` variants are being fitted to extend it.

**That is the general shape of the remedy.** Where a contrast stops short of where it should, the cause so far has always been a scope rule, a window or an instrument — each fixable at its own site, each fixable without a joint model. A joint model would inherit every one of these limits unchanged, because they are properties of what was measured.

## 6. What the decision costs

Stated plainly, so it is not mistaken for a free win. Declining the build means:

- **The DS–TD gap is a derived functional, never a parameter.** It has no prior, no hierarchy, and no shrinkage. Where the gap is estimated from few draws reaching a level the interval is wide, and nothing regularises it.
- **No partial pooling of the dispersion or GP hyperparameters across populations.** Each side is estimated on its own data, which is conservative but not efficient.
- **Cross-population contrasts cannot be re-expressed on a latent scale** under a linking convention, which is where they would have to move if Route 1 found material DIF ([`202607261210`](202607261210-route1-dif-prespecification.md) §2).

All three are acceptable at present because the contrasts that matter are already estimated with intervals narrow enough to support the reported conclusions, and because the third is contingent on a result not yet in hand.

## 7. What would reopen it

Any one of these, and the decision should be re-taken rather than inherited:

1. **Route 1 returns material DIF.** Cross-population reporting then moves to the latent scale under an explicit linking convention, and a model carrying both populations on that scale becomes the natural home for it. This is the most likely route back.
2. **The two pools stop being disjoint.** A study contributing both DS and TD children, or a shared control group, breaks the factorisation in §3 and makes a joint model necessary rather than optional.
3. **A contrast is wanted at a level or age one side alone cannot support**, _and_ borrowing strength across populations is defensible for that quantity — which §4d says it is not for TD comprehension, but which is not a general ruling.

**The constructive follow-up, which does not wait on any of these:** symmetrise the DS and TD prior structure per §4c, and re-run `scripts/compare_ds_td_re.py` to establish how much of the between-child heterogeneity contrast survives it. That removes a stated caveat from a published claim at a fraction of the cost of the model that prompted this note.

## 8. Where this is recorded

- `docs/comparison/index.qmd` — the preamble parenthesis, which should now cite this note.
- `docs/report/results-words-understood-spoken.qmd` §sec-ds-td — states _"that model is not built, and is noted in the Limitations"_. **It was not**; the Limitations entry is added alongside this note.
- `docs/report/discussion.qmd` §sec-limitations — the new entry, under "What the models assume".
