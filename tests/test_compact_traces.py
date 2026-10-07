# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check whether a trace-promotion staging entry may still be active.

A parsed PID uses the portable ``psutil.pid_exists`` probe. A name that
cannot be parsed, or a probe that raises, is treated as live so compaction
cannot rewrite a potentially active fit. ``/proc`` is unavailable on native
Windows, and ``os.kill(pid, 0)`` is not a portable existence check.
"""

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "compact_traces.py"
_SPEC = importlib.util.spec_from_file_location("compact_traces_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_is_live = _MODULE._is_live

# A well-formed staging name: `<config_name>-<timestamp>-<pid>-<hash>`.
_STAGING_ENTRY = "VG03-age-spoken-td-20260814T120000-4242-deadbeef"


def test_parseable_name_with_existing_pid_is_live(monkeypatch):
    seen = []

    def pid_exists(pid):
        seen.append(pid)
        return True

    monkeypatch.setattr(_MODULE.psutil, "pid_exists", pid_exists)
    assert _is_live(_STAGING_ENTRY) is True
    # The PID probed is the one embedded in the name, parsed as an integer.
    assert seen == [4242]


def test_parseable_name_with_dead_pid_is_not_live(monkeypatch):
    monkeypatch.setattr(_MODULE.psutil, "pid_exists", lambda pid: False)
    assert _is_live(_STAGING_ENTRY) is False


def test_unparseable_name_is_live(monkeypatch):
    # The probe must not even be consulted for a name that cannot be parsed:
    # the conservative direction is to refuse.
    def pid_exists(pid):  # pragma: no cover - reaching this is the failure
        raise AssertionError("pid_exists consulted for an unparseable name")

    monkeypatch.setattr(_MODULE.psutil, "pid_exists", pid_exists)
    assert _is_live("not-a-staging-name") is True
    assert _is_live("VG03-age-spoken-td-20260814T120000-notapid-deadbeef") is True


def test_probe_raising_is_live(monkeypatch):
    def pid_exists(pid):
        raise OSError("probe failed")

    monkeypatch.setattr(_MODULE.psutil, "pid_exists", pid_exists)
    assert _is_live(_STAGING_ENTRY) is True


def test_own_pid_reads_as_live_without_monkeypatching():
    # An end-to-end probe against the real psutil: this process certainly
    # exists, so a staging entry carrying its PID must read as live.
    import os

    assert _is_live(f"VG03-age-spoken-td-20260814T120000-{os.getpid()}-deadbeef") is True
