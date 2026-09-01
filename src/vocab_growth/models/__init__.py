# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The vocabulary growth model family (see ``docs/models/README.md``).

Each ``model_vgNN.py`` is a thin module selecting a definition from
``definitions.py`` and dispatching to one of the shared fitting engines. Which
engine, and everything else about a model that is not part of its statistical
definition -- its analysis-frame builder, prior-predictive hook, plot hook and
report template -- is recorded once in :mod:`vocab_growth.models.catalogue`, and
every dispatch table in the package and the scripts is derived from it.

Deliberately no model list here: this docstring said "VG01-VG16" for as long as
there had been twenty registered models, which is what a hand-copied count does.
``MODEL_REGISTRY`` in ``definitions.py`` is the registered set, and
``catalogue.CATALOGUE`` covers exactly it.

Outcome suffixes
----------------

Variable names in the multi-outcome graphs carry a one- or four-letter outcome
suffix. The convention is **not** uniform, and reading it as though it were is the
likeliest way to misread a build function. The four readings:

===========  ====================================  ==================================
Suffix       Quantity                              Where it appears
===========  ====================================  ==================================
``u``        Words **understood** -- the primary    Latent (``f_u_*``) and observation
             trajectory                            (``y_u_obs``, ``kappa_u_*``)
``q``        The **conditional production ratio**   Latent only: ``h_*`` (logit),
             -- the fraction of understood words    ``q_*`` (probability), ``tau_q``,
             a child speaks                        ``delta_q``, ``subj_q``. There is
                                                   no ``y_q_obs`` and no ``kappa_q``
``s``        Words **spoken**, as a marginal on     Both the derived latent
             the 810-item scale -- ``p_S = p_U*q``  (``f_s_*``, ``p_s_*``) and the
                                                   observation (``y_s_obs``,
                                                   ``kappa_s_*``, ``obs_s_mask``)
``sign``     Words **signed** -- serves *both*      Latent (``f_sign_*``,
             roles, ratio and observation, with     ``g_sign_*``) and observation
             no ``q``/``s`` split                   (``y_sign_obs``, ``kappa_sign_*``)
===========  ====================================  ==================================

So ``kappa_s`` and ``tau_q`` in one build function are both correct: dispersion is
a property of an *observation* (``s``), and a between-study scale is a property of
the *ratio's latent* (``q``). The signed side does not follow that split because
signing was added as a single third modality rather than as a ratio plus a
marginal. The two primary latents are also asymmetric in shape: understood is
``f_u_*`` and the ratio is a bare ``h_*``.

A separate distinction rides on top of these: ``tau_u`` / ``tau_q`` /
``tau_sign`` are **study**-level scales, while ``tau_subj_u`` / ``tau_subj_q`` are
**per-child** ones. They differ by one word in the middle of the name.

**Renaming any of this is not an available fix.** These names are in every trace,
manifest and summary table on disk, and in the report's own cells; changing one
makes every existing fit of the affected models unreproducible. See
:mod:`vocab_growth.models.subject_effects`, whose ``OUTCOME_SUFFIXES`` is the
machine-readable form of the ``u``/``q``/``sign`` column above.
"""
