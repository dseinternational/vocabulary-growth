# Vocabulary data - UK (6)

> [!NOTE]
> Column semantics drafted by an LLM-based AI tool (Claude Code/Opus 5), from the checklist completion instructions supplied by the study owner and a check against the data (2026-08-12).

A small UK Down syndrome sample: **11 children, 60–115 months**, all from one area. Its value to the pool is its age range — before `uk_07` arrived it was the only source of signing observations above 60 months, and it remains the only source above 96.

## Fields

<!-- spellchecker: disable -->

- subject_id
- area
- sex
- age
- understood
- signed
- imitated
- spoken

<!-- spellchecker: enable -->

## Instrument

The **standard DSE vocabulary checklists** — Checklists 1, 2 and 3, listing 120, 340 and 350 printed words, so **810 items** in total. Each checklist additionally admits the child's own name, family and pet names and similar proper nouns, so the maximum *achievable* counts are **127 / 349 / 353**, or 829 in total (study owner, 2026-08-25). The models score against the printed 810; the 19-word gap matters only for judging whether a high count is possible, and no observation in any source comes within 30 words of either figure. This is the same instrument as `ie_01` and `ie_02`, and it is the common reference inventory every model's likelihood scores against (`n_trials = 810`), so `survey_vocab_max = 810` is the source's native ceiling rather than a harmonisation.

The parent completes five columns per word, and the completion instructions make each of columns 2–5 **conditional on comprehension**:

| column | instruction (abridged)                                                                                              | column here |
| ------ | ------------------------------------------------------------------------------------------------------------------- | ----------- |
| 1      | tick if you are confident your child **understands** the word                                                        | `understood` |
| 2      | tick if your child **understands and signs** this word — _"tick for imitated signs as well as for spontaneous signs"_ | `signed`     |
| 3      | tick if your child **understands and imitates** the spoken word                                                      | `imitated`   |
| 4      | tick if you have heard your child **say this word spontaneously**, in a correct context, at least 3 times            | `spoken`     |
| 5      | tick when your child can say the word **clearly** enough for an unfamiliar listener                                  | _not supplied_ |

## Measurement and column semantics

### `signed` is a total sign count

Column 2 is "understands **and** signs", ticked for imitated as well as spontaneous signs. It is therefore a **total**: it counts a word the child signs whether or not they also say it. That makes uk_06 directly comparable with `uk_02`, `nz_01`, `es_01` and `uk_07`, and **unlike `uk_01`**, whose `signed` is a sign-*only* count and needs item-level re-derivation before it can be pooled (see `SIGNED_ONLY_STUDIES`).

The columns are overlapping per-word ticks rather than a mutually exclusive ladder, which the data confirms three ways:

- `signed` ≤ `understood` on **11 of 11** rows, as "understands and signs" requires.
- `spoken` ≤ `understood` on **11 of 11** rows.
- `signed + spoken` **exceeds** `understood` on **7 of 11** rows — impossible under an exclusive reading, and exactly what overlapping ticks predict when a word is both signed and said.

### History of the masking

uk_06's signing was included in the primary signing analyses from 15 June 2026, then masked from 16 July (`8ea6227`) pending confirmation that its field measured total sign use rather than uk_01's sign-only construct — the source had no field dictionary at the time. That question was tracked as [issue #211](https://github.com/dseinternational/vocabulary-growth/issues/211) and **resolved on 2026-08-12** by the completion instructions above. `UNCERTAIN_SIGN_STUDIES` is now empty and uk_06's 11 signing observations are in the primary analyses.

## Known issues

- **`imitated` is internally inconsistent, and is not used.** Column 3 is "understands and imitates", so it must be nested within `understood` — but it exceeds it on **4 of 11** rows. The `vocab_combined` view selects only `understood`, `spoken`, `signed` and `produced`, so this column reaches no model and affects no estimate. The nesting violation is a live data-quality question for the source. **The second half of this flag is withdrawn (2026-08-25):** one row reaching **822** was read as above the instrument's 810-item ceiling and so as possible evidence of a transcription or column-alignment problem touching neighbouring columns. With proper-noun slots the achievable maximum is 829, so 822 is a legitimate count and carries no such implication. It is the only value above 810 anywhere in the data.
- **Small and age-shifted.** 11 children, all 60–115 months, all from one area. Every estimate uk_06 influences is at the old end of the age range, where the pool is thinnest — which is exactly why the source matters and also why it carries weight disproportionate to its size. Note [202606151700](../notes/202606151700-vg14-signed-ratio-shape-and-p-any-bias.md) §3 records them as heavy older-age signers who raise the old-age signed ratio.
- **`produced` is `spoken`.** No signed/spoken union is derivable from these aggregates, so uk_06 is one of the signs-excluded sources for VG18's total-expressive outcome (see the `model_vg18` docstring).
- **Speech clarity not supplied.** Column 5 is collected on the instrument but is not in this file.

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.
