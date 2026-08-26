# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the staging-liveness check in ``scripts/compact_traces.py``.

``_is_live`` guards the mid-promotion refusal: a staging entry whose embedded
PID belongs to a running process must read as live so the script refuses to
rewrite that model's trace under the fit. The original implementation probed
``/proc/<pid>``, which on native Windows — this repository's primary
development platform — is always absent, so every entry read as dead and the
guard never fired. The check now uses ``psutil.pid_exists``, which is portable
(``os.kill(pid, 0)`` is not: CPython's ``os.kill`` on Windows calls
``TerminateProcess``). These tests pin the portable probe and the conservative
fallbacks: an unparseable name, or a probe that raises, is treated as live.
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
