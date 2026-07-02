# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The vocabulary growth model family (VG01-VG15; see docs/models/README.md).

Each ``model_vgNN.py`` is a thin module selecting a definition from
``definitions.py`` and dispatching to one of six shared fitting engines:
``common.py``, ``common_univariate_re.py``, ``common_bivariate.py``,
``common_bivariate_re.py``, ``common_trivariate.py`` and
``common_joint_modality.py``.
"""
