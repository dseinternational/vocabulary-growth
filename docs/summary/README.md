# Summary report for parents and practitioners

> [!NOTE]
> Drafted with assistance from Claude Code/Fable 5.1; revised by OpenAI Codex/GPT-6.

> [!WARNING]
> These pages are an unfinished outline. One worked sentence demonstrates computed values; the remaining findings, interactive components and site export are not complete.

## What this is

The plain-language companion to the technical report, written for families and practitioners intended for the main website. Quarto is the authoring tool here, not the publishing one: each page is one `.qmd`, rendered to Markdown for import into `dsegroup/content`, and the same sources render as one combined DOCX or PDF for partners to review before publication. The interactive tools are not built here; the pages carry placeholders (`data-chart-id` blocks) for the site's chart components, which read a prediction pack exported from the models of record.

No number in the prose is typed. Each is computed in a code cell from the fitted output through the helpers in `_summary_data.qmd` and exported as `<span data-vg="…">216</span>`: the value is baked into the Markdown, DOCX and PDF, The spans provide hooks for a future site component. This repository does not implement or verify live updates.

## Layout

| File                           | Role                                                                                                                                                                                                                 |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_quarto.yml`                  | Project configuration: the pages render to Markdown (`gfm`); `all.qmd` declares its own DOCX and PDF formats. Everything lands in `output/summary/`.                                                                 |
| `_summary_data.qmd`            | The data helpers, included at the top of every page. Reads the report figure cache until the prediction pack exists; the helper vocabulary is the pack's.                                                            |
| `<page>.qmd`                   | One file per site page, in the order listed in `all.qmd`. The file stem is the page's folder name in the content repository. No front matter: the H1 is the page title, and the page must be self-contained.         |
| `all.qmd`                      | The combined review document, including every page in order, with the author block and the DOCX/PDF settings.                                                                                                        |
| `filters/callouts-to-html.lua` | Markdown export only: rewrites Quarto callouts as the raw-HTML alert blocks the site's pages use, because the content system's Markdown pipeline does not parse the GitHub alert syntax Quarto would otherwise emit. |
| `footer.tex`, `_variables.yml` | PDF page furniture (draft marker, version, date) after the technical report's.                                                                                                                                       |

## Rendering

From the repository root, in the project's `uv` environment:

```bash
uv run quarto render docs/summary --to gfm
```

renders every page to `output/summary/<page>.md` (and the combined `all.md`, which is harmless). The review document:

```bash
uv run quarto render docs/summary/all.qmd --to docx
```

or `--to pdf` (needs the report's XeLaTeX setup and fonts). A bare `uv run quarto render docs/summary` builds all of it, the PDF included, so it takes a couple of minutes. To preview one page in a browser:

```bash
uv run quarto render docs/summary/words-understood.qmd --to html
```

The cells read `docs/report/figures/` (the cache `scripts/sync_report_figures.py` fills), so the VG20 output must be synced first, and VG15's for the signing figure. Nothing is frozen: the cells are cheap, and `freeze: auto` cannot see edits to an included file.

## Conventions

- **Numbers come from helpers.** `median_at(age, outcome)`, `interval_at(age, outcome, prob)`, `lower_at` / `upper_at`, `share_below(age, outcome, k)`, `one_in_below(age, outcome, k)`, `n_obs(age, outcome)`, `cap(outcome)`, `years_months(age)` and `generated_from()`. Outcomes are `"understood"` and `"spoken"`; ages are whole months; `k` is one of the pack's bucket thresholds. Most helpers return values with an HTML binding. `years_months` and `cap` return plain text. `share_below` and `one_in_below` read inclusive probabilities, `P(Y <= k)`, so prose must say "k or fewer", not "fewer than k".
- **Component placeholders** are a pair of conditional blocks: `::: {.content-visible when-format="gfm"}` holding the `data-chart-id` div, and `::: {.content-visible unless-format="gfm"}` holding the static figure for DOCX and PDF. The `0.0.0/…` chart identifiers are placeholders until the versioning of the published assets is decided.
- **Callouts** take their title as an attribute (`title="…"`); the Markdown export renders it as a bold first paragraph of the alert block.
- **Pages are self-contained.** A cross-reference to another page resolves in the combined document but not in the page's own export.
- **The helper cell is unlabelled** on purpose: `all.qmd` includes every page, and a labelled cell included ten times is a duplicate-label error.
- Every page carries the AI-attribution callout the project requires for drafted content; retain it when publishing AI-assisted text.

## Export to the site

The export script and prediction-pack integration are not implemented here. The Markdown output is available for editorial review, but chart identifiers and bindings remain placeholders. Confirm the destination site's import and update behaviour before publication.
