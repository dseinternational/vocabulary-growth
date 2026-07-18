# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check whether fitted output is safe to resume from or publish."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import dse_research_utils.statistics.models.sampling as sampling

from vocab_growth import environment as env
from vocab_growth.fit_artifacts import (
    git_metadata,
    source_data_hash,
    validate_fit_output,
)
from vocab_growth.models.definitions import MODEL_REGISTRY


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Registered model id, for example vg01.")
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

    key = args.model.lower()
    if key not in MODEL_REGISTRY:
        parser.error(f"Unknown model: {args.model}")
    definition = MODEL_REGISTRY[key]
    expected_sampling = sampling.get_sampling_configuration(args.config)

    env.set_output_root(args.output_dir)
    model_label = f"{definition.model_id}-{definition.config_name}"
    output_dir = os.path.join(env.models_output_dir(), model_label)
    errors = validate_fit_output(
        output_dir,
        expected_definition=definition,
        expected_sampling_config_name=args.config,
        expected_sampling_parameters=asdict(expected_sampling),
        expected_git=git_metadata(env.ROOT_DIR),
        expected_source_data_hash=source_data_hash(env.DATA_DIR),
        require_reporting_quality=args.purpose == "publish",
        require_rendered_report=args.purpose == "publish",
        require_clean_fit=args.purpose == "publish",
    )
    if errors:
        print(f"[invalid] {key}: {output_dir}")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"[valid] {key}: {output_dir} ({args.purpose}, config={args.config})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
