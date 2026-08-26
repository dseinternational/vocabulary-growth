# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check whether fitted output is safe to resume from or publish."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import dse_research_utils.statistics.models.sampling as sampling

from vocab_growth import environment as env
from vocab_growth.analysis_frames import expected_analysis_frame_hash
from vocab_growth.fit_artifacts import (
    fit_validation_kwargs,
    git_metadata,
    source_data_hash,
    validate_fit_output,
)
from vocab_growth.models.definitions import MODEL_REGISTRY


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="+",
        help="Registered model ids, or 'all'.",
    )
    parser.add_argument(
        "--config",
        default="rep",
        help="Expected sampling configuration (default: rep).",
    )
    parser.add_argument(
        "--purpose",
        choices=("resume", "publish"),
        default="resume",
        help="Apply compatibility checks for resuming or stricter publication checks.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root override; follows the same rules as fit_model.py.",
    )
    args = parser.parse_args()

    requested = [model.lower() for model in args.models]
    if requested == ["all"]:
        keys = list(MODEL_REGISTRY)
    else:
        unknown = [key for key in requested if key not in MODEL_REGISTRY]
        if unknown:
            parser.error(f"Unknown model(s): {', '.join(unknown)}")
        keys = requested
    expected_sampling = sampling.get_sampling_configuration(args.config)

    env.set_output_root(args.output_dir)
    current_source_hash = source_data_hash(env.DATA_DIR)
    current_git = git_metadata(env.ROOT_DIR) if args.purpose == "resume" else None
    invalid = False
    for key in keys:
        definition = MODEL_REGISTRY[key]
        model_label = f"{definition.model_id}-{definition.config_name}"
        output_dir = os.path.join(env.models_output_dir(), model_label)
        errors = validate_fit_output(
            output_dir,
            **fit_validation_kwargs(
                args.purpose,
                expected_definition=definition,
                expected_sampling_config_name=args.config,
                expected_sampling_parameters=asdict(expected_sampling),
                current_git=current_git,
                current_source_data_hash=current_source_hash,
                # Both purposes this script offers carry the data checks, so the
                # prepared-frame hash is always computed (issue #266 finding 1).
                current_analysis_frame_hash=expected_analysis_frame_hash(
                    key, definition
                ),
            ),
        )
        if errors:
            invalid = True
            print(f"[invalid] {key}: {output_dir}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[valid] {key}: {output_dir} ({args.purpose}, config={args.config})")
    return int(invalid)


if __name__ == "__main__":
    raise SystemExit(main())
