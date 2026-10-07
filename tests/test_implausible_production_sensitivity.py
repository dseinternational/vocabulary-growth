# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check that reinstatement sensitivities reach the prepared model frames.

Both the bivariate and joint engines must accept the variant definitions.
The prepared-frame gain must match the net counts reported by the fit log.
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
_COMBINED = "us01-masked-production-reinstated"


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


@pytest.mark.parametrize("variant_name", [_VARIANT, _COMBINED])
@pytest.mark.parametrize(
    ("model_key", "prepare"),
    [
        ("vg10", cbr.prepare_bivariate_re_data),
        ("vg15", cj.prepare_joint_data),
    ],
)
def test_variant_preparation_runs_and_reinstates_production(
    model_key, prepare, variant_name, tmp_path
):
    """Each engine must prepare the variant frame and gain the masked records.

    The combined variant's two fit-log figures -- the implausible rule's catch
    with the same-day rule lifted, and the same-day rule's own catch -- must
    partition the frame's gain over the baseline (#289 task 4.3).
    """
    _requires_db()
    from vocab_growth.models.definitions import MODEL_REGISTRY

    baseline = MODEL_REGISTRY[model_key]
    (variant,) = build_variant(model_key, variant_name)

    # Preparation must not raise for either — the AttributeError above was only
    # reachable by actually running the engine.
    base_ctx = _context(baseline, tmp_path)
    prepare(base_ctx, baseline)
    variant_ctx = _context(variant, tmp_path)
    prepare(variant_ctx, variant)

    # And the prepared frames must differ by exactly what the fit log reports.
    base_spoken = int(base_ctx.analysis_df["spoken"].notna().sum())
    variant_spoken = int(variant_ctx.analysis_df["spoken"].notna().sum())
    reported = vocab_data_utils.count_reinstated_implausible_production(
        include_same_day_disagreements=variant.include_same_day_disagreements
    )
    if variant.include_same_day_disagreements:
        same_day = vocab_data_utils.count_reinstated_same_day_disagreements()
        assert same_day > 0
        assert reported == 11, "with the same-day rule lifted the rule's whole catch returns"
        reported += same_day

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
