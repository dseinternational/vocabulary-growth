# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The VM campaign's lists stay in step with the registries.

``scripts/vm/campaign.sh`` names models, sensitivity arms and recovery targets
in shell arrays. A renamed or newly registered arm would otherwise surface as a
failed step, or a silently skipped one, a day or two into a campaign on a VM.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vocab_growth.models.catalogue import CATALOGUE, publication_models
from vocab_growth.recovery.spec import supported_models
from vocab_growth.sensitivity.registry import VARIANTS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "vm"
CAMPAIGN = (SCRIPTS / "campaign.sh").read_text(encoding="utf-8")


def _array(name: str) -> list[tuple[str, str]]:
    match = re.search(rf"^{name}=\((.*?)\)$", CAMPAIGN, flags=re.S | re.M)
    assert match, f"{name} is not defined in campaign.sh"
    return re.findall(r'"(vg\d\d) ([a-z0-9-]+)"', match.group(1))


ARRAYS = ("C_DS_ARMS", "C_TD_ARMS", "D_DS_ARMS", "D_TD_ARMS")


@pytest.mark.parametrize("name", ARRAYS)
def test_every_arm_named_is_registered(name):
    unknown = [pair for pair in _array(name) if pair not in VARIANTS]
    assert not unknown, f"{name} names unregistered arms: {unknown}"


def test_the_arm_lists_cover_every_registered_arm_once():
    named = [pair for name in ARRAYS for pair in _array(name)]
    assert len(named) == len(set(named)), "an arm is scheduled twice"
    assert set(named) == set(VARIANTS), sorted(set(VARIANTS) - set(named))


def test_typically_developing_arms_run_serially_and_ds_arms_pool():
    """Keep the campaign's TD fits serial to limit concurrent memory demand."""
    td = {key for key, model in CATALOGUE.items() if "td" in model.definition.config_name.split("-")}
    for name in ("C_TD_ARMS", "D_TD_ARMS"):
        assert {model for model, _ in _array(name)} <= td
    for name in ("C_DS_ARMS", "D_DS_ARMS"):
        assert not {model for model, _ in _array(name)} & td
    assert re.search(r'arms C 1 "\$\{C_TD_ARMS\[@\]\}"', CAMPAIGN)
    assert re.search(r'arms D 1 "\$\{D_TD_ARMS\[@\]\}"', CAMPAIGN)


def _driver_models(step: str) -> list[str]:
    match = re.search(rf'"{step}::[^"]*-Models ([a-z0-9,]+)"', CAMPAIGN)
    assert match, f"no driver step {step}"
    return match.group(1).split(",")


def test_stage_a_is_the_publication_scope():
    refitted = _driver_models("A-fit-ds") + _driver_models("A-fit-td")
    assert sorted(refitted) == sorted(publication_models())


def test_stages_a_and_b_cover_the_registry_without_overlap():
    steps = ("A-fit-ds", "A-fit-td", "B-fit-ds", "B-fit-td")
    refitted = [key for step in steps for key in _driver_models(step)]
    assert len(refitted) == len(set(refitted))
    assert set(refitted) == set(CATALOGUE)


def test_every_recovery_target_is_supported_and_its_variant_registered():
    targets = re.findall(r"\$\{REC\[\*\]\} (vg\d\d)( --variant ([a-z0-9-]+))?", CAMPAIGN)
    assert targets
    for model, _, variant in targets:
        assert model in supported_models(), model
        if variant:
            assert (model, variant) in VARIANTS


def test_every_refit_replaces_the_model_of_record_deliberately():
    assert "-ReplaceModelOfRecord" in CAMPAIGN


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="needs a POSIX bash; on Windows `bash` may resolve to the WSL stub",
)
@pytest.mark.parametrize("script", ["campaign.sh", "bootstrap.sh"])
def test_the_scripts_parse(script):
    result = subprocess.run(
        ["bash", "-n", str(SCRIPTS / script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
