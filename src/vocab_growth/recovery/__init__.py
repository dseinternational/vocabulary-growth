# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parameter-recovery tools for simulated data with known parameter values.

``spec`` declares engine likelihoods and simulation order. ``simulate`` draws
outcomes from those likelihoods. ``refit`` substitutes the synthetic frame into
the engine's pipeline. ``compare`` scores posterior estimates against truth.

Recovery tests the fitted procedure at the selected truths and study design.
It does not establish identification generally or adequacy for real observations.
See ``docs/runbooks/parameter-recovery.md`` for use and interpretation.
"""
