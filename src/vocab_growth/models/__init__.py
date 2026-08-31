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
"""
