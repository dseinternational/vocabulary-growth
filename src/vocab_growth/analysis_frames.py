# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rebuild and hash prepared analysis frames without running a fit.

The hash records schema, values, index and row order. Validation can therefore
detect changes to masking, exclusions or harmonisation even when the raw CSVs
are unchanged. Pure engine builders produce the same frames as fitting stages
without writing descriptive outputs.

The model catalogue supplies each registered model's builder. The definition
class alone does not identify the engine: VG05 and VG07 share a definition class
but use different engines.
"""

from __future__ import annotations

import hashlib
import importlib
import json

import numpy as np
import pandas as pd

from vocab_growth.models.catalogue import CATALOGUE
from vocab_growth.models.definitions import ModelDefinition

#: Engine frame builder for every registered model, as ``module:function``.
#: String targets defer engine imports, including PyMC, until a frame is needed.
FRAME_BUILDERS: dict[str, str] = {
    key: f"{model.engine.module}:{model.engine.frame_builder}"
    for key, model in CATALOGUE.items()
}


def analysis_frame_hash(df: pd.DataFrame) -> str:
    """Hash the exact prepared analysis frame, including schema and row order."""
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in df.dtypes.items()]
    digest.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=True, categorize=True)
    digest.update(row_hashes.to_numpy(dtype=np.uint64).tobytes())
    return f"sha256:{digest.hexdigest()}"


def build_analysis_frame(
    model_key: str, definition: ModelDefinition
) -> tuple[pd.DataFrame, dict]:
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


def expected_analysis_frame_hash(
    model_key: str, definition: ModelDefinition
) -> str:
    """The exact frame hash a fresh fit of ``definition`` would record today.

    This is what fitted-output validation compares against the manifest's
    recorded ``data.analysis_frame_hash``: a mismatch means the loader rules
    (or the deterministic row order) changed since the fit, even when the raw
    CSVs did not.
    """
    frame, _ = build_analysis_frame(model_key, definition)
    return analysis_frame_hash(frame)
