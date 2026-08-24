# Gate 1's boundary comparisons: a valid reference distribution, and what it says about the withdrawn p-value

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Measurement record, 2026-08-24, closing the "nonregular Gate-1 chi-square likelihood-ratio calibration" item of [#233](https://github.com/dseinternational/vocabulary-growth/issues/233). Both conclusions of [202608141600](202608141600-rank-stability-tracking.md) §10.4 **survive**, and the slope conclusion strengthens. But neither the plain chi-square that was used nor the standard boundary correction that would have been the obvious fix reproduces either null. Reproduced by `scripts/experiments/rank_stability.py --bootstrap N`.

## 1. Why a chi-square was never available

`rank_stability.py` fits three within-child structures to the fitted residuals by maximum likelihood and reports `2 * delta logL` between them. It labelled those statistics "(2 df)" and "(1 df)", and §10.4 read a p-value off one of them. Both nulls sit on a boundary of the parameter space, so Wilks' theorem does not hold:

- **`tau1 = 0`** is a variance component at zero. That alone would call for a mixture of chi-squares rather than a single one — but it is worse than the usual case here, because at `tau1 = 0` the correlation `rho01` is **not identified at all**. One of the two nominal degrees of freedom does not exist under the null.
- **`rho01 = 1`** is a correlation at its own boundary.

## 2. The replacement

`bootstrap_null` keeps the statistic and replaces the reference distribution. Each child's adjusted scores are regenerated from the fitted null structure on the study's **own** design — the same children, the same ages, the same known binomial sampling variances, so the simulated data carry the study's singleton/repeater mix — and both structures are refitted to each simulation.

## 3. Results

DS spoken, repeats only: 334 children, ages centred at 36 months, 250 replicates each. Roughly 10-12 seconds a replicate, so about 40 and 48 minutes.

| comparison                  | observed | null median | null 95th | null 99th | bootstrap p |
| --------------------------- | -------: | ----------: | --------: | --------: | ----------: |
| slope vs constant intercept |    20.81 |        0.55 |      5.05 |      7.36 |  **0.0040** |
| free `rho01` vs `rho01 = 1` |     6.28 |        0.00 |      2.65 |      5.91 |  **0.0120** |

Against the chi-squares that were quoted:

| reference     | median |  95th |  99th |
| ------------- | -----: | ----: | ----: |
| chi-square(2) |  1.386 | 5.991 | 9.210 |
| chi-square(1) |  0.455 | 3.841 | 6.635 |

**Both simulated nulls are lighter than their nominal chi-square at every quantile shown.** That is the boundary asserting itself, and it is most visible in the `rho01` null, whose median is **0.00** to seven decimal places: in about half the replicates the free-`rho01` optimum sits at the constrained value of 1, so the statistic is identically zero.

### 3.1 The slope conclusion strengthens

**No replicate out of 250 reached 20.81** — the bootstrap p is at its floor of `1/251`. The observed statistic is nearly three times the null's 99th percentile of 7.36. Note what the bootstrap cannot do here: chi-square(2) would put this at `p = 3e-5`, and 250 replicates cannot resolve anything below 0.004. The bootstrap is the _valid_ reference, not the more precise one, and for a statistic this far out the honest statement is "beyond every one of 250 null replicates" rather than a p-value.

### 3.2 The `rho01` conclusion survives, and the withdrawn p-value was right by luck

The bootstrap p of 0.0120 is numerically almost the withdrawn chi-square value of 0.0122. That is a coincidence, and it is worth stating in both directions:

- **The withdrawn `p ~ 0.012` was numerically about right.** Nothing that rested on it needs revisiting on magnitude.
- **It was right by luck, and the obvious correction would have made it worse.** The reference distribution it came from is demonstrably not this null. And the standard 50:50 boundary mixture — the fix a reviewer would reach for first — gives 0.0061, half the bootstrap value. Neither off-the-shelf reference reproduces this null; only simulating it does.

**Read it with its Monte Carlo uncertainty, which is the binding limit at this replicate count.** Two of 250 replicates reached the observed 6.28. The tail probability's point estimate is 0.008 with a Monte Carlo standard error of 0.006 and an 89% interval of [0.002, 0.022]. What the run supports is "the observed statistic sits above the 99th percentile of a properly simulated null", not a p-value quoted to three decimals.

## 4. What this does and does not license

It **does** license both qualitative conclusions of §10.4. Random slopes are supported on production by within-child evidence, decisively. Rank-one covariance — children never crossing, which is the property usually quoted for Proposal A1 — is contradicted by that same evidence, at a statistic in the far tail of its own simulated null.

It does **not** restore "rejected at p = 0.012" as a form of words for the second. The honest statement is the one now in the VG19 report and the tracking note: read the magnitude, and if a p-value is wanted, take it from the bootstrap with its Monte Carlo interval attached.

It says nothing about the **all-children** column, where the slope is worth 27.09 on comprehension and the `rho01` comparison only 1.49. That column was not bootstrapped, and these nulls cannot be borrowed for it: a null distribution is simulated from a particular fitted structure on a particular design, and the all-children design is 767 children with a different singleton share. §10.4's reading of that column — that singletons say nothing about whether children cross — stands on its own argument, not on this run.

## 5. Reproducing

```bash
uv run python scripts/experiments/rank_stability.py --bootstrap 250
```

Without `--bootstrap` the statistics print with no p-value and no degrees of freedom, which is the correct default: there is no reference distribution to name. The statistics themselves are unchanged from §10.3 and reproduce exactly — 36.05, 20.81, 1.49, 6.28.
