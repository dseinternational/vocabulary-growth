> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5.5).

# Shared library 0.18.0 upgrade

On 7 October 2026 the project moved `dse-research-utils` from `v0.17.0` to `v0.18.0`, which takes its plot colours from the DSE design tokens ([research#121](https://github.com/dseinternational/research/pull/121), pinned to released tokens in [research#122](https://github.com/dseinternational/research/pull/122)). `[tool.uv.sources]` selects tag `v0.18.0`, release commit `cb35ffb3b6fef8f940447b6b94a3e68b218912ae`.

## The lock

`uv lock --upgrade-package dse-research-utils` moved only `dse-research-utils`, from `935bd38b` to `cb35ffb3`. The only raised minimum, `jupytext>=1.19.6` in the `notebook` extra, is already the locked version, so no other package moved. The uv release used (0.12.23) writes lock revision 5, where the previous lock had revision 3.

## What the library changed

- `CHART_COLOURS` holds the six categorical chart colours: blue, green, orange, purple, teal and pink. They are the default colour cycle, so `"C0"` to `"C5"` resolve to them. Under `tab10`, `"C1"` was orange, `"C2"` green and `"C3"` red; now `"C1"` is green, `"C2"` orange and `"C3"` purple. `"C6"` onwards wrap round to the start.
- `TEXT_COLOUR` darkens, `LINE_COLOUR` lightens to a hairline grey, and `MUTED_TEXT_COLOUR` is new for secondary text and reference lines that carry meaning.
- `categorical_palette(n)` returns hex strings and refuses more than six series. A named matplotlib palette keeps the old behaviour.
- The default colour map is a sequential blue scale, and `plot_heatmap` takes a sequential or centred diverging scale in place of `viridis`.
- The `COLOUR_*` names are deprecated and warn when read.

## Colours by role

No deprecated name remains in tracked Python. Colours are named by role at module level, and one quantity keeps one colour across the package and scripts:

| Role                                                             | Colour                                         |
| ---------------------------------------------------------------- | ---------------------------------------------- |
| Words understood                                                 | `CHART_COLOURS[0]`, blue (`"C0"`)              |
| Words spoken, and the production ratio                           | `CHART_COLOURS[1]`, green (`"C1"`)             |
| Words signed                                                     | `CHART_COLOURS[2]`, orange (`"C2"`)            |
| Down syndrome, typically developing, and their difference        | `[0]`, `[2]` and `[1]` in the comparisons      |
| Observed children over a model path                              | `CHART_COLOURS[2]`, orange                     |
| Prior draws over observed data                                   | `[2]` over `[0]`                               |
| Annotation text and reference lines (zero, equality, thresholds) | `MUTED_TEXT_COLOUR`                            |
| Composition: speech only, both, sign only, neither               | `"C1"`, `"C4"`, `"C2"` and `MUTED_TEXT_COLOUR` |

Spoken quantities were orange before the upgrade, because `"C1"` was orange. They now follow `"C1"` to green. The prior child-count figure and the VG07 study-effect forest plot named orange directly for spoken and the production ratio; they move to green with the rest. The comparison of attainment delay already drew production in green.

Two figures needed more than a renamed colour. The four-cell composition drew "neither" in `"C7"`, `tab10`'s grey, which now wraps round to the speak-only green, so it uses `MUTED_TEXT_COLOUR`. The comparison script's signing composition drew its cells in the comparison's population colours, and now uses the same colours as VG15's composition figure.

`COLOUR_RED` marked the median of the predictive count distributions and the VG05 curve in the VG05 and VG07 overlay. Neither is signed, so each takes the next chart colour not otherwise used in its figure: orange for the median and blue for VG05. The labels of the count distributions' intervals and median, previously green and red text, use `MUTED_TEXT_COLOUR`. The predictive trajectories, previously `COLOUR_DARK_BLUE`, take the median's own blue, as their comment already said.

## Figures with more than six groups

The design language allows six categorical colours. The descriptive figures and the per-study curves colour by study, and the pools hold more:

| Figure                                             | Groups coloured                  | Palette                                         |
| -------------------------------------------------- | -------------------------------- | ----------------------------------------------- |
| Descriptive observations and repeated measures, DS | 15 studies in the shared mapping | `tab20` without its red, as before              |
| Descriptive repeated measures, TD                  | 12 studies in the shared mapping | `tab20` without its red, as before              |
| Descriptive scatters by study, DS                  | 14 (9 for signing)               | `tab20` (`tab10` for signing), as before        |
| Per-study curves (`study_fans`), DS models         | 15                               | `tab20`, in place of a twelve-colour house list |
| Per-study curves (`study_fans`), TD models         | 6                                | The chart colours                               |

The descriptive helpers use the chart colours for six groups or fewer and otherwise the palette `categorical_palette` gave them before, so their study colours are unchanged. The shared mapping still skips red, because the pooled-summary overlay those figures draw is red. Its test for red had counted chart-3 orange as red, which would have refused six or more groups; it now matches only `tab10` and `tab20`'s red. The per-study curves had extended the old named hues with their dark variants, including red and yellow, and fall back to `tab20` instead. Redesigning these figures, for example as small multiples or by grouping studies, is left for later.

## Other changes

- Zero lines, equality diagonals, the 50% and 90% production-ratio thresholds and the uplift baseline move from `LINE_COLOUR` to `MUTED_TEXT_COLOUR`. The faint observed administrations behind the population path move to `MUTED_TEXT_COLOUR` at a lower alpha, so they keep their weight.
- The data-coverage heatmaps use the sequential scale in place of `viridis`. That scale darkens as counts rise, the reverse of `viridis`, so the cell labels turn white from a third of the maximum instead of dark from half of it.
- Colours that never came from the shared styles are unchanged: the red pooled-summary overlay and the violin fill in `descriptive.py`, the grey observed trajectories in `plotting.py`, the black zero-effect line and grey diagonal in the prior child checks, the grey reference lines in `common_joint_modality.py`, and the palettes of the experiment scripts.

## Figures that change on their next run

No figure, report or fit was regenerated. Every figure drawn under the shared style changes slightly, because text, ticks and grid lines take the new values. Beyond that, the next run changes series colours in the model reports' joint, trivariate and sign-and-speech figures, where spoken becomes green and signed orange; the population path, study curves, prior checks and predictive count figures; and the comparison figures, data-coverage heatmaps, prior-against-posterior panels, milestone figures and the VG07 study-effect plot.

## Signature

The executable-code signature records the installed `dse-research-utils` commit and hashes the package source, so both the new commit and the package edits change it. Resuming, strict synchronisation and publication enforce the signature, so an earlier fit needs a refit or a recorded implementation change (`resume_from_trace.py --allow-implementation-change`) before those steps.
