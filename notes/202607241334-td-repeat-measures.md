# Repeat measures in the TD data

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

There is no single "TD data" observation count: it is outcome-specific, because comprehension (`understood`) is recorded only on CDI:WG forms while production (`spoken`) exists on both WG and WS forms. Repeat measures (multiple visits per child) matter for the two TD models that carry subject-level random effects — VG11 (spoken) and VG12 (understood); the baselines VG03/VG04 pool visits as independent (`use_subject_re=False`). Counts below are reconstructed from the current prepared DuckDB via `load_data(population="td", …)` + `filter_studies_by_min_obs(200)`; the spoken figure (16,235) matches VG11's fit manifest exactly, so this is faithful to the published fit.

| metric                                    | VG11 (spoken) | VG12 (understood) |
| ----------------------------------------- | ------------- | ----------------- |
| observations (rows)                       | 16,235        | 5,997             |
| unique children                           | 12,266        | 4,764             |
| children with >1 visit                    | 1,947 (15.9%) | 1,000 (21.0%)     |
| repeat observations (rows beyond 1/child) | 3,969 (24.4%) | 1,233 (20.6%)     |
| mean visits/child                         | 1.32          | 1.26              |
| max visits (one child)                    | 11            | 6                 |
| contributing studies                      | 7             | 4                 |

Spoken studies (min 200 obs/study): ByersHeinlein, Floccia, Hoff, Kalashnikova, Marchman, Smith, Thal. Understood studies: ByersHeinlein, Floccia, Marchman, Thal.

Headline: roughly 1,900–2,000 TD children are seen more than once, and about a quarter of the spoken observations (3,969 / 16,235) are longitudinal repeats — the corresponding understood figure is about a fifth (1,233 / 5,997). The distribution is heavily single-visit (10,319 of 12,266 spoken children appear once); a distinct cluster of 708 children with exactly 4 spoken visits stands out, likely one study's longitudinal design. VG13 (young joint) is a narrower age-restricted subset (5,406 rows) of the same children.

Relevance: the subject random intercept in VG11/VG12 exists precisely to absorb this within-child correlation, and it sits alongside the #164 dataset/study intercepts whose identifiability ridges we removed in the #176 conditioning fixes — see `202607191614-full-refit-rep-hightune-run.md`. The repeat-measure structure is why these TD models are hierarchical, and why their geometry was delicate.
