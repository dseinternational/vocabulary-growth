# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prior-sensitivity tooling for the Bayesian prior review (issue #89 §7).

`overrides` builds alternative-prior *variants* of a model definition without
mutating the committed instance; `registry` enumerates the variant matrix
covering the §7 sensitivity targets; `compare` reads the resulting fits' summary
CSVs and reports whether headline quantities stay within the baseline's interval.
"""
