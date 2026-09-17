# Vocabulary data - UK (3)

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6 on 2026-09-17.

This source records comprehension and production on the 416-item Oxford CDI. The pooled view maps these to `understood` and `spoken`; it supplies no signed count or recorded sex. The source has 27 rows. The age derivation is documented below.

## Columns

- `age` — complete months of age: the months part of the source's `age_m_d` (months:days). Until 2026-09-15 it was the source's half-month `age` column rounded half-to-even, which sent half-months both ways (16.5 → 16, 35.5 → 36) and inherited that column's disagreements with `age_m_d` (`18:18` recorded as 18.0); a round-half-up derivation from `age_m_d` briefly replaced it the same day. Against the half-to-even column, 13 of the 27 ages are a month lower and none higher, and no count changed. The derivation lives in `dsegroup/research-data-analysis` (`prepare/uk_03_dsouza.py`), and every Down syndrome prepared frame moved with it.
- `age_m_d` — the source's original age string, kept for audit.

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.
