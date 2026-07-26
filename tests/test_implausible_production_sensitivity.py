# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The `us01-implausible-reinstated` sensitivity must run and must bite.

``mask_implausible_production_administrations`` excludes 30 us_01 administrations
by default. The source author no longer holds the original data files, so that
exclusion can never be confirmed at source, and these two variants are the only
published check on it — what the headline joint trajectories would have been had
the judgement been wrong.

That makes two failure modes worth pinning. The variant must actually reach the
frame (a flag that stops at ``load_data`` would leave a registered check that
cannot fail — the exact fault that retired ``us01-ceiling-excluded``), and each
engine's data preparation must survive being handed the variant definition. The
second is not hypothetical: the first implementation read
``definition.max_age_months`` in the reinstated-count line, which
``JointModelDefinition`` does not define, so VG15 raised ``AttributeError`` only
once the engine was actually run.
"""

import os

import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import pytest

import vocab_growth.data_utils as vocab_data_utils
from vocab_growth.models import common_bivariate_re as cbr
from vocab_growth.models import common_joint_modality as cj
from vocab_growth.models.common import ModelFitContext
from vocab_growth.sensitivity.registry import build_variant

_VARIANT = "us01-implausible-reinstated"


def _context(definition, tmp_path):
    ctx = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=str(tmp_path),
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(ctx.reporting.output_dir, exist_ok=True)
    return ctx


def _requires_db():
    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")


@pytest.mark.parametrize(
    ("model_key", "prepare"),
    [
        ("vg10", cbr.prepare_bivariate_re_data),
        ("vg15", cj.prepare_joint_data),
    ],
)
def test_variant_preparation_runs_and_reinstates_production(
    model_key, prepare, tmp_path
):
    """Each engine must prepare the variant frame and gain the masked records."""
    _requires_db()
    from vocab_growth.models.definitions import MODEL_REGISTRY

    baseline = MODEL_REGISTRY[model_key]
    (variant,) = build_variant(model_key, _VARIANT)

    # Preparation must not raise for either — the AttributeError above was only
    # reachable by actually running the engine.
    base_ctx = _context(baseline, tmp_path)
    prepare(base_ctx, baseline)
    variant_ctx = _context(variant, tmp_path)
    prepare(variant_ctx, variant)

    # And the prepared frames must differ by exactly what the fit log reports.
    base_spoken = int(base_ctx.analysis_df["spoken"].notna().sum())
    variant_spoken = int(variant_ctx.analysis_df["spoken"].notna().sum())
    reported = vocab_data_utils.count_reinstated_implausible_production()

    assert reported > 0, "a reinstatement variant that restores nothing cannot fail"
    assert variant_spoken == base_spoken + reported


def test_baselines_do_not_carry_the_flag(tmp_path):
    """A baseline with the flag set would make the variant a silent no-op."""
    from vocab_growth.models.definitions import MODEL_REGISTRY

    for model_key in ("vg10", "vg15"):
        assert MODEL_REGISTRY[model_key].include_implausible_production is False


def test_reinstated_count_needs_no_age_bound_argument():
    """Both engines call this with no argument; JointModelDefinition has no age bound."""
    _requires_db()
    assert vocab_data_utils.count_reinstated_implausible_production() == (
        vocab_data_utils.count_reinstated_implausible_production(None)
    )
    from vocab_growth.models.definitions import VG15

    assert not hasattr(VG15, "max_age_months")
