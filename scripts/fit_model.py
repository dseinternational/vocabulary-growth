# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Fits the specified model to the latest data. Saves plots and data, and report to output directory.
"""

import argparse
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup
from rich import print

from vocab_growth.models import model_vg01, model_vg02, model_vg03, model_vg04

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default=None, help='Model id or all.')
    parser.add_argument('--config', type=str, default='dev', help='Sampling configuration to use (e.g., dev[elopment], test, rep[orting])')

    freeze_support()

    setup.init_script()

    args = parser.parse_args()

    if args.model == "vg01" or args.model == "all":
        model_vg01.fit(args.config)
    elif args.model == "vg02" or args.model == "all":
        model_vg02.fit(args.config)
    elif args.model == "vg03" or args.model == "all":
        model_vg03.fit(args.config)
    elif args.model == "vg04" or args.model == "all":
        model_vg04.fit(args.config)
    else:
        print(f"Unknown model: {args.model}")
        exit(1)
