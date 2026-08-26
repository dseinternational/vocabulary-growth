# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure analysis-frame builders and the exact prepared-frame hash.

Every fit manifest records ``data.analysis_frame_hash`` — a hash of the exact
prepared analysis frame (schema, values, index and row order). Until issue
#266 nothing ever read it back: validation compared only the raw-CSV
fingerprint, so a change to the loader's *rules* (masking, exclusions,
harmonisation) silently left stale posteriors accepted as current. Closing
that gap needs the prepared frame to be recomputable outside a fit, which the
engines' ``prepare_*_data`` stage functions cannot do — they print tables and
write descriptive CSVs into a fit's output directory.

Each engine therefore exposes a pure ``build_*_analysis_frame(definition)``
function containing exactly the frame construction its prepare stage runs, and
this module maps every registered model to its engine's builder so a validator
can ask "what would this definition's frame hash be today?". The mapping is by
model key because the engine choice lives in each ``model_vgNN`` module, not
in the definition class (VG05 and VG07 share a definition class on different
engines); ``tests/test_analysis_frames.py`` pins every registered key to a
builder and pins a fitted manifest's recorded hash to the recomputed one, so
a model moved between engines cannot silently drift.
"""

from __future__ import annotations

import hashlib
import importlib
import json

import numpy as np
import pandas as pd

#: Engine frame builder for every registered model, as ``module:function``.
#: Held as strings so importing this module stays light — the engines pull in
#: PyMC. See the module docstring for why this is keyed by model rather than
#: by definition class.
FRAME_BUILDERS: dict[str, str] = {
    "vg01": "vocab_growth.models.common:build_univariate_analysis_frame",
    "vg02": "vocab_growth.models.common:build_univariate_analysis_frame",
    "vg03": "vocab_growth.models.common:build_univariate_analysis_frame",
    "vg04": "vocab_growth.models.common:build_univariate_analysis_frame",
    "vg05": "vocab_growth.models.common_bivariate:build_bivariate_analysis_frame",
    "vg07": "vocab_growth.models.common_bivariate_re:build_bivariate_re_analysis_frame",
    "vg08": "vocab_growth.models.common_bivariate_re:build_bivariate_re_analysis_frame",
    "vg09": "vocab_growth.models.common_bivariate_re:build_bivariate_re_analysis_frame",
    "vg10": "vocab_growth.models.common_bivariate_re:build_bivariate_re_analysis_frame",
    "vg11": "vocab_growth.models.common_univariate_re:build_univariate_re_analysis_frame",
    "vg12": "vocab_growth.models.common_univariate_re:build_univariate_re_analysis_frame",
    "vg13": "vocab_growth.models.common_bivariate_re:build_bivariate_re_analysis_frame",
    "vg14": "vocab_growth.models.common_trivariate:build_trivariate_analysis_frame",
    "vg15": "vocab_growth.models.common_joint_modality:build_joint_analysis_frame",
    "vg16": "vocab_growth.models.common_bivariate_re:build_bivariate_re_analysis_frame",
    "vg19": "vocab_growth.models.common_bivariate_re:build_bivariate_re_analysis_frame",
    "vg20": "vocab_growth.models.common_bivariate_re:build_bivariate_re_analysis_frame",
    "vg21": "vocab_growth.models.common_bivariate_re:build_bivariate_re_analysis_frame",
    "vg22": "vocab_growth.models.common_bivariate_re:build_bivariate_re_analysis_frame",
    "vg23": "vocab_growth.models.common_bivariate_re:build_bivariate_re_analysis_frame",
}


def analysis_frame_hash(df: pd.DataFrame) -> str:
    """Hash the exact prepared analysis frame, including schema and row order."""
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in df.dtypes.items()]
    digest.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=True, categorize=True)
    digest.update(row_hashes.to_numpy(dtype=np.uint64).tobytes())
    return f"sha256:{digest.hexdigest()}"


def build_analysis_frame(model_key: str, definition) -> tuple[pd.DataFrame, dict]:
    """Rebuild ``definition``'s prepared analysis frame outside a fit.

    Returns the frame and the engine's side information (exclusion counts and
    similar), exactly as the fit pipeline's data-preparation stage would
    construct them. Raises ``KeyError`` for a model with no registered builder
    rather than guessing an engine.
    """
    target = FRAME_BUILDERS.get(model_key.lower())
    if target is None:
        raise KeyError(
            f"No analysis-frame builder is registered for {model_key!r}. "
            "Register its engine's builder in FRAME_BUILDERS."
        )
    module_name, _, function_name = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)(definition)


def expected_analysis_frame_hash(model_key: str, definition) -> str:
    """The exact frame hash a fresh fit of ``definition`` would record today.

    This is what fitted-output validation compares against the manifest's
    recorded ``data.analysis_frame_hash``: a mismatch means the loader rules
    (or the deterministic row order) changed since the fit, even when the raw
    CSVs did not.
    """
    frame, _ = build_analysis_frame(model_key, definition)
    return analysis_frame_hash(frame)
