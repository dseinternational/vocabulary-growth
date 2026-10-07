# Vocabulary data - UK (2)

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6.

This data was collected as part of a research project in 2010.

## Measures and preparation

The prepared CSV records DSE and Oxford CDI assessments. `form` identifies the instrument. The DSE checklist uses the 810-item reference; the Oxford form has 416 items. Preserve the form label when comparing assessments or linking repeated visits.

The current prepared fields are:

- `study`, `subject_id`, `age`, `sex` and `form`.
- `comprehension`, `spoken`, `signed` and `production`.
- `understood_only`, `signed_only`, `spoken_only` and `signed_spoken`.

The four last fields partition understood words by expression. A word spoken and signed contributes once to the union, twice to the sum of the marginal speech and signing totals. Do not add those margins to obtain distinct expressive words.

`cross_tab_sources.load_uk02_four_cell` uses a cross-tabulation only when all four counts are present, the margins reconcile and their total is positive. Its total supplies comprehension for that likelihood. Other rows can still contribute their usable marginal counts. A comprehension count below recorded speech is masked by the shared nesting rule. See the [data guide](readme.md) and the [cross-tabulation loader](../src/vocab_growth/cross_tab_sources.py).

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0). See [LICENSE](LICENSE) for details.
