# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Static checks for the locked, reproducible software environment."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]

SUPPORTED_PLATFORMS = {
    ("linux", "x86_64"),
    ("linux", "aarch64"),
    ("darwin", "arm64"),
    ("win32", "AMD64"),
}


def _platform_pairs(markers: list[str]) -> set[tuple[str, str]]:
    """Reduce environment markers to ``(sys_platform, platform_machine)`` pairs.

    uv normalises marker expressions when it writes the lockfile, so the clause
    order there differs from the order declared in ``pyproject.toml``. Compare
    the values rather than the strings.
    """
    pairs = set()
    for marker in markers:
        platform = re.search(r"sys_platform == '([^']+)'", marker)
        machine = re.search(r"platform_machine == '([^']+)'", marker)
        assert platform is not None and machine is not None, marker
        pairs.add((platform.group(1), machine.group(1)))
    return pairs


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _uv_lock() -> dict:
    return tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))


def test_uv_lock_covers_the_supported_platforms():
    declared = _platform_pairs(_pyproject()["tool"]["uv"]["environments"])
    locked = _platform_pairs(_uv_lock()["supported-markers"])

    assert declared == SUPPORTED_PLATFORMS
    assert locked == SUPPORTED_PLATFORMS


def test_uv_lock_pins_the_shared_library_to_an_exact_revision():
    tag = _pyproject()["tool"]["uv"]["sources"]["dse-research-utils"]["tag"]
    locked = [
        package
        for package in _uv_lock()["package"]
        if package["name"] == "dse-research-utils"
    ]

    assert len(locked) == 1
    source = locked[0]["source"]["git"]
    assert source.startswith("https://")
    assert f"tag={tag}" in source
    revision = source.rsplit("#", maxsplit=1)[1]
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)


def test_agent_instruction_copies_remain_identical():
    agents = (ROOT / "AGENTS.md").read_bytes()

    assert agents == (ROOT / "CLAUDE.md").read_bytes()
    assert agents == (ROOT / ".github/copilot-instructions.md").read_bytes()
