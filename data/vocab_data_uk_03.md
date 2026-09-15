# Vocabulary data - 3

[Description to follow]

## Columns

- `age` — whole months, derived from the source's `age_m_d` (months:days) as `months + days / 30.4375` rounded half up. Until 2026-09-15 it was the source's half-month `age` column rounded half-to-even, which sent half-months both ways (16.5 → 16, 35.5 → 36) and inherited that column's disagreements with `age_m_d` (`18:18` recorded as 18.0). Eight of the 27 ages moved by one month, five up and three down; no count changed. The derivation lives in `dsegroup/research-data-analysis` (`prepare/uk_03_dsouza.py`), and every Down syndrome prepared frame moved with it.
- `age_m_d` — the source's original age string, kept for audit.

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.
