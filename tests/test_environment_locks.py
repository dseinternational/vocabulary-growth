# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Static checks for the two-layer reproducible environment locks."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_conda_lock_covers_supported_reporting_platforms():
    payload = yaml.safe_load((ROOT / "conda-lock.yml").read_text(encoding="utf-8"))

    assert set(payload["metadata"]["platforms"]) == {"linux-64", "osx-arm64"}
    assert payload["metadata"]["sources"] == ["environment.yml"]
    assert {package["platform"] for package in payload["package"]} == {
        "linux-64",
        "osx-arm64",
    }


def test_pip_lock_has_only_portable_exact_requirements():
    lines = [
        line
        for line in (ROOT / "requirements-pip.lock")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    ]

    assert lines[0].startswith(
        "dse-research-utils[viz,notebook,io] @ git+https://"
    )
    revision = lines[0].split(".git@", maxsplit=1)[1].split("#", maxsplit=1)[0]
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)
    assert all("file://" not in line for line in lines)
    assert all("==" in line for line in lines[1:])


def test_agent_instruction_copies_remain_identical():
    agents = (ROOT / "AGENTS.md").read_bytes()

    assert agents == (ROOT / "CLAUDE.md").read_bytes()
    assert agents == (ROOT / ".github/copilot-instructions.md").read_bytes()
