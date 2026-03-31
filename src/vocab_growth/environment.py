# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_MODULE_DIR)

ROOT_DIR = os.path.dirname(_SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

MODELS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models")

DOCS_DIR = os.path.join(ROOT_DIR, "docs")

REPORT_DIR = os.path.join(DOCS_DIR, "report")
REPORT_FIGS_DIR = os.path.join(REPORT_DIR, "figures")
