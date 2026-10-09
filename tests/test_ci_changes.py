# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run these classification checks without installing the project environment."""

import copy
import importlib.util
import json
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "ci_changes", ROOT / "scripts/ci_changes.py"
)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class ChangeClassificationTests(unittest.TestCase):
    def setUp(self):
        self.lock = {
            "version": 1,
            "package": [
                {
                    "name": "dse-vocab-growth",
                    "dependencies": [{"name": "numpy"}],
                    "dev-dependencies": {"dev": [{"name": "ruff"}, {"name": "pytest"}]},
                },
                {"name": "ruff", "version": "1"},
                {"name": "numpy", "version": "1"},
                {"name": "pytest", "version": "1"},
            ],
        }

    def test_only_exclusive_tool_dependencies_are_removed(self):
        changed = copy.deepcopy(self.lock)
        changed["package"][1]["version"] = "2"
        self.assertEqual(
            ci.lock_without_tools(self.lock), ci.lock_without_tools(changed)
        )
        for name in ("numpy", "pytest"):
            changed = copy.deepcopy(self.lock)
            next(p for p in changed["package"] if p["name"] == name)["version"] = "2"
            self.assertNotEqual(
                ci.lock_without_tools(self.lock), ci.lock_without_tools(changed)
            )

    def test_a_tool_dependency_shared_with_the_runtime_is_kept(self):
        self.lock["package"][1]["dependencies"] = [{"name": "numpy"}]
        self.assertIn(
            "numpy", {p["name"] for p in ci.lock_without_tools(self.lock)["package"]}
        )

    def test_unknown_paths_and_sensitive_documentation_run_everything(self):
        for path in (
            "AGENTS.md",
            "docs/models/VG01.qmd",
            "pnpm-lock.yaml",
            "pnpm-workspace.yaml",
            "new-file.xyz",
            "src/new.py",
        ):
            self.assertTrue(ci.needs_full_checks([path], lambda _: "", lambda _: ""))
        self.assertFalse(
            ci.needs_full_checks(
                ["notes/note.md", "docs/report/results.qmd"], lambda _: "", lambda _: ""
            )
        )

    def test_missing_or_malformed_locks_run_everything(self):
        for source in ("not TOML", "version = 1"):
            self.assertTrue(
                ci.needs_full_checks(
                    ["uv.lock"],
                    lambda _, source=source: source,
                    lambda _, source=source: source,
                )
            )

    def test_real_ruff_update_does_not_change_the_runtime_lock(self):
        lock = tomllib.loads((ROOT / "uv.lock").read_text())
        updated = copy.deepcopy(lock)
        next(p for p in updated["package"] if p["name"] == "ruff")["version"] = "next"
        self.assertEqual(ci.lock_without_tools(lock), ci.lock_without_tools(updated))

    def test_node_tool_update_skips_models_but_script_changes_do_not(self):
        old = {"devDependencies": {"cspell": "1"}, "scripts": {"spellcheck": "cspell"}}
        new = copy.deepcopy(old)
        new["devDependencies"]["cspell"] = "2"
        self.assertFalse(
            ci.needs_full_checks(
                ["package.json"], lambda _: json.dumps(old), lambda _: json.dumps(new)
            )
        )
        new["scripts"]["spellcheck"] = "different"
        self.assertTrue(
            ci.needs_full_checks(
                ["package.json"], lambda _: json.dumps(old), lambda _: json.dumps(new)
            )
        )


if __name__ == "__main__":
    unittest.main()
