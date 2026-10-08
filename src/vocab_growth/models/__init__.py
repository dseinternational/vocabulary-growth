# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The vocabulary growth model family.

Each model_vgNN wrapper selects a definition and a shared engine. The catalogue
records engines and reporting hooks; MODEL_REGISTRY defines the registered set.
See docs/models/README.md for model structures and reporting roles.

Outcome suffixes distinguish latent quantities from observed counts.

u denotes comprehension, on the 810-word reference scale.
q denotes the spoken share of understood words. It has a latent logit h
but no y_q_obs or kappa_q.
s denotes spoken vocabulary, with marginal proportion p_S = p_U * q.
It labels both derived quantities and the spoken count likelihood.
sign denotes both the signed share and signed observations.

Thus tau_q scales study differences in the latent spoken share, while kappa_s
controls residual variation in spoken counts. tau_subj_u and tau_subj_q are
child scales; tau_u and tau_q are study scales.

These names form part of the trace, manifest and report interfaces. Renaming
them requires a compatibility plan for stored fits and their consumers.
subject_effects.OUTCOME_SUFFIXES records the child-effect suffixes.
"""
