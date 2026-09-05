> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

# us_03

## Source

Fidler, D.J., Van Deusen, K., Baumer, N.T., Bishop, S.L., & Lanfranchi, S. 
(2026) *English-language MacArthur Bates Communication Development Inventory in
Children with Down syndrome from Project CAPEabilities and Project EXPO*.

## Instrument

The **English CDI Words and Gestures** vocabulary checklist: **396 words**, each
marked by the parent as *understands* or *understands and says*. The two cells
are mutually exclusive, which is exactly the arithmetic the workbook carries —
`cdi_total = understand_tot + under_say_tot` holds for every row as received.

The *understands and says* cell is **not** speech-only. The study authors state:
"Understands and Says is inclusive of expressive language through spoken word and sign." It counts any-modality expressive vocabulary, and nothing in the share
separates the spoken part from the signed part.

`understand_tot` reaches **exactly 396** and never exceeds it, which is what
identifies the form. (Three observations nevertheless *sum* above 396; see
[Known issues](#known-issues).)

Mapping to the pipeline's canonical column names:

> [!IMPORTANT]
> **Correction, 2026-09-05.** The reading below is corrected from the version
> received with the share, which described `under_say_tot` as *understands and
> says* and stated twice that the source carries no sign or gesture modality.
> Both were inferred from the column name and the standard Words and Gestures
> wording, and both are wrong. The study authors state:
>
> > Understands and Says is inclusive of expressive language through spoken word and sign.
>
> So the cell is a **produced union** — any-modality expressive vocabulary — not
> a spoken count, and the two modalities are not separable: there is one number,
> unlike the mutually exclusive cells `nz_01` and `uk_07` carry, from which a
> spoken marginal can be built. The upstream column is still named `spoken`;
> the column was named `spoken` on that reading and has since been renamed
> to `produced`, here and upstream in `research-data-analysis`.

| Source column    | Output column     | Meaning                                        |
| ---------------- | ----------------- | ---------------------------------------------- |
| `understand_tot` | `understood_only` | Understands, does not say or sign              |
| `under_say_tot`  | `produced`        | Understands **and** says **or signs**          |
| `cdi_total`      | `understood`      | Total comprehension (the two cells summed)     |

So production is nested inside `understood` by construction, as in `mx_01`, but
it is the **any-modality** production union. This repository maps it to
`produced` and leaves both `spoken` and `signed` NULL: no spoken marginal can be
recovered from it, and a zero `signed` would assert an absence the source does
not support.

## Collection method

Parent-reported CDI checklists at up to two visits per child, roughly a year
apart (median interval **11.0 months**, range 5.5–14.6). The first visit clusters
at **17–25 months** (median 20.2) and the second at **29–35 months** (median
31.2), so this is a young, tightly age-banded longitudinal sample — younger than
most sources in the pipeline.

Only age and the CDI totals were shared; no sex, group, mental age, or item-level
responses.

## Input file

`original-data/CDI Data Share DSEI_20AUG2026.xlsx`, sheet **`in`** — 186 rows ×
9 columns, wide across the two visits:

```
ca_v1  ca_v2  understand_tot_v1  understand_tot_v2  under_say_tot_v1
under_say_tot_v2  cdi_total_v1  cdi_total_v2  id
```

Missing values are written as the literal `NA` (an R export convention), which
the pandas default missing-value tokens already cover; the script asserts that
every column parses as numeric so a change of convention is caught rather than
silently producing string columns.

Ages are in **months**. For `id` 6–186 every age is a whole hundredth of a year
multiplied by 12 (`19.2 = 1.60 × 12`, `30.599999999999898 = 2.55 × 12`); for
`id` 1–5 the ages carry full floating-point precision, i.e. they were computed
from dates. See [Known issues](#known-issues).

### Coverage

|                                | Count |
| ------------------------------ | ----- |
| Children in the workbook       | 186   |
| First visits with CDI data     | 184   |
| Second visits with an age      | 109   |
| Second visits with CDI data    | 106   |

`id` 5 and `id` 176 have a first-visit age but no CDI data at all and no second
visit; they are dropped entirely. `id` 83, 85 and 91 have a second-visit age but
no second-visit CDI block — the survey appears not to have been returned — and
lose that visit only. 77 children have no second visit recorded at all.

## Transformations

1. `id` (repeated across a child's two visits) is anonymised via
   `common.anonymise_subject(SOURCE, raw_id)` (`SOURCE = "us_03_fidler"`; prefix
   `ID_`, key from `DSE_ANONYMISATION_KEY`), guarded by `assert_injective`. The
   two visits intentionally map to the same subject id.
2. The wide frame is pivoted to one row per child per visit; `_v1`/`_v2` become
   `timepoint` `t1`/`t2` and `ca_v1`/`ca_v2` become `age_months`.
3. `age` is `age_months` rounded to the whole month, matching the rest of the
   pipeline; `age_months` keeps the source's own precision (2 dp).
4. Two properties are **asserted** on every row (both hold in the file as
   received): a visit's three CDI columns are either all present or all absent,
   and `understand_tot + under_say_tot = cdi_total`. A violation means the export
   changed or was misread.
5. Visits with no CDI data at all are dropped.

Result: **290 rows** for **184 children** (184 first visits + 106 second visits).

## Output schema

`data/vocab_data_us_03.csv`:

| Column           | Notes                                                        |
| ---------------- | ------------------------------------------------------------ |
| subject_id       | Study-namespaced HMAC of the workbook `id`, prefix `ID_`     |
| timepoint        | `t1` (first visit) / `t2` (second visit)                     |
| age              | Age in whole months at that visit                            |
| age_months       | Age in months at the source's own precision                  |
| understood       | Items understood (`understood_only + produced`)              |
| understood_only  | Items understood but not said or signed                      |
| produced         | Items understood **and** said **or signed**                  |
| survey_vocab_max | 396, constant — the Words and Gestures word count            |

> [!IMPORTANT]
> `spoken` is nested inside `understood` (`mx_01` convention), not a disjoint
> modality cell as in `nz_01`/`uk_07`. `understood_only` is the residual.
> There is no `signed`/`gestured` column: the share carries no non-vocal
> modality.

## Known issues

- **Three observations exceed the 396-word checklist.** They are carried through
  unchanged and printed as notes by the script; whether to drop or repair them is
  a downstream decision, as with the `uk_07` and `ie_02` exclusions. Downstream
  users should check `understood <= survey_vocab_max`.

  | Workbook `id` | Visit | `understood_only` | `produced` | `understood` |
  | ------------- | ----- | ----------------- | -------- | ------------ |
  | 113           | v2    | 396               | 72       | 468          |
  | 130           | v2    | 389               | 15       | 404          |
  | 3             | v1    | 56                | 376      | 432          |

  `id` 113 and 130 look like the same error: `understand_tot` recorded as *all*
  words understood (inclusive of those said) rather than the understands-only
  cell, so `cdi_total` double-counts. Read that way their comprehension is 396
  and 389 and their production 72 and 15, both plausible. `id` 3 is different —
  it belongs to the older sub-sample below, and under that reading its
  comprehension (56) would fall far below its production (376), so the
  inclusive/exclusive confusion does not explain it.

- **`id` 1–5 are a structurally distinct sub-sample.** Four of the five reach the
  output (`id` 5 has no CDI data). They are aged **62–80 months** against 17–25
  months for the rest of the first visits, have no second visit, sit near the
  form's ceiling (production 286–376 of 396), and their ages are the only ones in
  the file not recorded as a whole hundredth of a year. Everything about them
  says they came from a different file — plausibly the second of the two projects
  named in the citation. **The workbook carries no project identifier**, so
  records cannot be attributed to Project CAPEabilities or Project EXPO, and no
  project column is emitted. If the split matters downstream, it should be
  confirmed with the data providers rather than inferred from age.

  A related open question: whether those four children were administered the same
  396-word Words and Gestures form. `id` 3's total of 432 would be within range on
  a larger form (e.g. the 680-word Words and Sentences list), and children of
  5–6½ years are well past the Words and Gestures age band. `survey_vocab_max` is
  set to 396 for every row on the strength of the file-wide `understand_tot`
  ceiling, but it is the least certain value in this source for those four rows.

- **Comprehension and production fall between visits for some children.** 16 of
  106 children have lower `understood` at t2 than t1 and 15 have lower `spoken`.
  This is the ordinary parent-report noise seen in `uk_07` and `uk_01`, carried
  through unchanged.

- **Floor effects.** 30 of 290 observations have `understood = 0` and 93 have
  `spoken = 0`, concentrated at the first visit (median production there is 2
  words). Any growth model over this source has to handle a large mass at zero.

- **Longitudinal**: `subject_id` repeats across a child's visits; a row is
  identified by `(subject_id, timepoint)`.

- **No covariates**: the share carries no sex, ethnicity, maternal education,
  mental age, or intervention group, so this source cannot contribute to any
  analysis conditioning on them.
