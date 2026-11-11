# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check that ``--dist loadgroup`` keeps each ``xdist_group`` on one worker.

Slow modules mark the tests that share an expensive module fixture with
``xdist_group``, so the slow CI job sets each fixture up once. A broken group
mark fails nothing: every worker that receives one of those tests repeats the
fits, and runs only get slower. pytest 9.2.0 with pytest-xdist 3.8.0 broke the
marks this way (pytest-dev/pytest-xdist#1386). The installed pair runs a small
grouped suite here, and the test checks where each test ran. It checks the
outcome, not a mechanism, because the upstream fix changes how groups reach
the scheduler.
"""

import json
import os
import subprocess
import sys
import textwrap

CONFTEST = """
import json
import os
import pathlib

import pytest


@pytest.fixture(autouse=True)
def record_worker(request):
    worker = getattr(request.config, "workerinput", {}).get("workerid", "none")
    path = pathlib.Path(os.environ["XDIST_GROUP_PROBE_DIR"], request.node.name)
    path.write_text(json.dumps(worker))
"""

# One group spans two modules and uses both a module mark and a function mark,
# as the slow modules do. Ungrouped tests give the second worker work.
GROUPED_MODULE = """
import pytest

pytestmark = pytest.mark.xdist_group("shared")


def test_grouped_first():
    pass


def test_grouped_second():
    pass


@pytest.mark.parametrize("value", ["a", "b"])
def test_grouped_param(value):
    pass
"""

MIXED_MODULE = """
import pytest


@pytest.mark.xdist_group("shared")
def test_grouped_elsewhere():
    pass


def test_free_first():
    pass


def test_free_second():
    pass


def test_free_third():
    pass
"""

GROUPED = {
    "test_grouped_first",
    "test_grouped_second",
    "test_grouped_param[a]",
    "test_grouped_param[b]",
    "test_grouped_elsewhere",
}
FREE = {"test_free_first", "test_free_second", "test_free_third"}


def test_loadgroup_runs_each_group_on_one_worker(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    probe = tmp_path / "probe"
    probe.mkdir()
    # An ini file stops the run from finding this repository's settings.
    (suite / "pytest.ini").write_text("[pytest]\n")
    (suite / "conftest.py").write_text(textwrap.dedent(CONFTEST))
    (suite / "test_grouped.py").write_text(textwrap.dedent(GROUPED_MODULE))
    (suite / "test_mixed.py").write_text(textwrap.dedent(MIXED_MODULE))

    # Drop the outer session's PYTEST_ADDOPTS, xdist worker identity and
    # plugin settings.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["XDIST_GROUP_PROBE_DIR"] = str(probe)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "2",
            "--dist",
            "loadgroup",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=suite,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output

    workers = {path.name: json.loads(path.read_text()) for path in probe.iterdir()}
    assert set(workers) == GROUPED | FREE, output
    assert len(set(workers.values())) == 2, (
        f"the probe suite did not run on two workers: {workers}"
    )
    grouped_on = {name: workers[name] for name in sorted(GROUPED)}
    assert len(set(grouped_on.values())) == 1, (
        "--dist loadgroup split one xdist_group across workers, so slow "
        "fixtures shared under a group mark would be rebuilt on each worker. "
        "Check the installed pytest and pytest-xdist against "
        f"pytest-dev/pytest-xdist#1386. Workers: {grouped_on}"
    )
