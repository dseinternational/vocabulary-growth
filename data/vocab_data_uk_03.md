# Vocabulary data - 3

[Description to follow]

## Columns

- `age` — complete months of age: the months part of the source's `age_m_d` (months:days). Until 2026-09-15 it was the source's half-month `age` column rounded half-to-even, which sent half-months both ways (16.5 → 16, 35.5 → 36) and inherited that column's disagreements with `age_m_d` (`18:18` recorded as 18.0); a round-half-up derivation from `age_m_d` briefly replaced it the same day. Against the half-to-even column, 13 of the 27 ages are a month lower and none higher, and no count changed. The derivation lives in `dsegroup/research-data-analysis` (`prepare/uk_03_dsouza.py`), and every Down syndrome prepared frame moved with it.
- `age_m_d` — the source's original age string, kept for audit.

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.
