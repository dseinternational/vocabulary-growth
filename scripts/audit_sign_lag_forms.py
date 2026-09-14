# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Audit VG25's source and target forms without fitting a model."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from vocab_growth.analysis_frames import analysis_frame_hash
from vocab_growth.environment import output_root
from vocab_growth.models.common_joint_modality import build_joint_analysis_frame
from vocab_growth.models.cross_lag import (
    prev_wave_sign_share_lag_for_frame,
    sign_cross_lag_audit_frame,
    sign_share_counts,
)
from vocab_growth.models.definitions import VG25
from vocab_growth.models.observation_arrays import prepare_joint_observations


def audit_forms(directory: Path) -> dict:
    """Write row-level provenance and counts for unrestricted and same-form lags."""
    frame, _ = build_joint_analysis_frame(VG25)
    restricted = replace(VG25, sign_lag_same_form_only=True)
    with_forms, _ = build_joint_analysis_frame(restricted)
    pd.testing.assert_frame_equal(with_forms.drop(columns="survey_vocab_max"), frame)
    sizes = with_forms["survey_vocab_max"].to_numpy()
    previous, has_lag, _ = prev_wave_sign_share_lag_for_frame(frame, VG25)
    _, has_same_form_lag, _ = prev_wave_sign_share_lag_for_frame(with_forms, restricted)
    observations = prepare_joint_observations(
        frame,
        VG25,
        n_trials=VG25.n_trials,
        use_subject_codes=True,
    )
    audit = sign_cross_lag_audit_frame(
        frame,
        previous,
        has_lag,
        spoken_indices=observations.idx_s,
        spoken_is_conditional=observations.spoken_spec.is_conditional,
        cell_indices=observations.idx_cells,
        prod_indices=observations.idx_prod,
    )
    rows = audit["row"].to_numpy(dtype=int)
    sources = audit["source_row"].to_numpy(dtype=int)
    audit["source_inventory_size"] = sizes[sources]
    audit["target_inventory_size"] = sizes[rows]
    audit["same_form_lag_retained"] = has_same_form_lag[rows].astype(bool)
    transitions = (
        audit.groupby(
            ["study", "source_inventory_size", "target_inventory_size"],
            dropna=False,
        )
        .size()
        .reset_index(name="lagged_observations")
    )

    signed, understood = sign_share_counts(frame)
    usable = np.isfinite(signed) & (understood > 0) & np.isfinite(sizes)
    shares = with_forms.loc[
        usable, ["study", "subject_code", "age", "survey_vocab_max"]
    ]
    paired = shares.groupby(["study", "subject_code", "age"])[
        "survey_vocab_max"
    ].nunique()
    dropped = (has_lag > 0) & (has_same_form_lag == 0)
    summary = {
        "model": VG25.model_id,
        "analysis_frame_hash": analysis_frame_hash(frame),
        "frame_with_inventory_hash": analysis_frame_hash(with_forms),
        "observations": len(frame),
        "unrestricted_lags": int(has_lag.sum()),
        "same_form_lags": int(has_same_form_lag.sum()),
        "dropped_lags_by_study": frame.loc[dropped].groupby("study").size().to_dict(),
        "unknown_inventory_rows": int(np.isnan(sizes).sum()),
        "unknown_lag_source_inventories": int(np.isnan(sizes[sources]).sum()),
        "unknown_lag_target_inventories": int(np.isnan(sizes[rows]).sum()),
        "waves_with_usable_signed_shares_on_different_forms": int((paired > 1).sum()),
    }
    directory.mkdir(parents=True, exist_ok=True)
    audit.to_csv(directory / "lag_forms.csv", index=False)
    transitions.to_csv(directory / "form_transitions.csv", index=False)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(output_root()) / "audits" / "sign-lag-forms",
    )
    args = parser.parse_args()
    print(json.dumps(audit_forms(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
