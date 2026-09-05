# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage, render and publish the comparison book as one unit.

The comparison book is the one published artefact with no publishing path of its
own. ``upload.py`` knows only about model output directories, whose figures live
inside them and therefore travel with them; the comparison book instead reads
**bare filenames from its own directory** -- both its CSV inputs at render time
and its PNG figures at view time. Assembling that by hand is what this script
exists to stop. On 2026-09-03 a hand-assembled upload carried ``index.html`` and
``index_files/`` but none of the 24 figures, and the page published with every
image broken; the ``index.html`` returning 200 was taken as success.

Three failure modes are designed out rather than documented:

**Stale staged inputs.** ``docs/comparison/`` is gitignored and persists between
runs, and the runbook's staging step is a ``cp`` that overwrites but never
removes. A file deleted from ``output/comparisons/`` survives there and the book
renders against it -- which happened the same day, publishing a recovery table
that included a quarantined replicate. ``stage`` clears the staged inputs first.

**Incomplete uploads.** ``collect`` walks the rendered HTML and gathers every
local asset it references, so what is uploaded is derived from the page rather
than from someone's memory of what a page needs.

**Unverified uploads.** ``verify`` requests every referenced asset over HTTP
after publishing and fails on any that does not return 200. A page whose images
are all missing is indistinguishable from a healthy one if only ``index.html``
is checked.

Usage::

    uv run python scripts/publish_comparison.py            # stage, render, publish, verify
    uv run python scripts/publish_comparison.py --no-render  # publish what is rendered
    uv run python scripts/publish_comparison.py --dry-run    # stage and render only
    uv run python scripts/publish_comparison.py --run-id <id>  # republish in place

Pass ``--run-id`` with an existing upload's identifier to replace that
publication rather than create a new URL -- which is how a broken publication is
repaired without invalidating a link already circulated.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from vocab_growth import environment as env
from vocab_growth.publication_checks import referenced_assets, verify_published

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK_DIR = os.path.join(REPO_ROOT, "docs", "comparison")
BOOK_SOURCE = os.path.join(BOOK_DIR, "index.qmd")
BOOK_HTML = os.path.join(BOOK_DIR, "index.html")
LABEL = "comparison-book"
PROJECT = "vocabulary-growth"

#: Files in the staged directory that are inputs, not outputs, and must survive
#: the clear. Everything else there is copied in from the comparisons directory.
KEEP = ("index.qmd",)


def stage_inputs(comparisons_dir: str) -> int:
    """Clear the staged copy, then copy the comparison artefacts in fresh.

    Clearing first is the point: without it a file removed from the source
    directory persists here and the book renders against an artefact that no
    longer exists upstream.
    """
    removed = 0
    for name in os.listdir(BOOK_DIR):
        path = os.path.join(BOOK_DIR, name)
        if name in KEEP or name.startswith("_") or not os.path.isfile(path):
            continue
        os.remove(path)
        removed += 1
    copied = 0
    for source_dir in (comparisons_dir, os.path.join(comparisons_dir, "recovery")):
        if not os.path.isdir(source_dir):
            continue
        for name in sorted(os.listdir(source_dir)):
            path = os.path.join(source_dir, name)
            if os.path.isfile(path):
                shutil.copy2(path, os.path.join(BOOK_DIR, name))
                copied += 1
    print(
        f"[stage] cleared {removed} stale files, copied {copied} from {comparisons_dir}"
    )
    return copied


def render() -> None:
    """Render the book with Quarto's Python pinned to this environment.

    A bare ``quarto render`` resolved an interpreter without ``yaml`` on
    2026-09-03 and the book died in its first cell.
    """
    environ = dict(os.environ, QUARTO_PYTHON=sys.executable)
    print(f"[render] quarto render {BOOK_SOURCE} (QUARTO_PYTHON={sys.executable})")
    subprocess.run(
        ["quarto", "render", BOOK_SOURCE], check=True, env=environ, cwd=REPO_ROOT
    )


def collect(destination: str) -> list[str]:
    """Assemble exactly the page and the assets it references."""
    if os.path.isdir(destination):
        shutil.rmtree(destination)
    os.makedirs(destination)
    shutil.copy2(BOOK_HTML, os.path.join(destination, "index.html"))
    assets = referenced_assets(BOOK_HTML)
    for relative in assets:
        target = os.path.join(destination, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(os.path.join(BOOK_DIR, relative), target)
    print(f"[collect] index.html plus {len(assets)} referenced assets")
    return assets


def verify(base_url: str, assets: list[str], timeout: float = 30.0) -> list[str]:
    """Request every published asset; return the ones that did not return 200.

    ``base_url`` is the published page's own URL; the assets sit beside it.
    Shared with the model-report upload through ``publication_checks`` since
    #289 task 4.10, so the two paths cannot drift.
    """
    return verify_published(
        base_url.rsplit("/", 1)[0], ["index.html", *assets], timeout=timeout
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="output root (see fit_model.py)")
    parser.add_argument(
        "--run-id", default=None, help="republish into an existing upload id"
    )
    parser.add_argument(
        "--no-stage", action="store_true", help="use the staged inputs as they are"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="publish the rendered page as it is"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="stage and render, do not publish"
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="where to assemble the upload (default: alongside the output root)",
    )
    args = parser.parse_args()

    env.set_output_root(args.out)
    comparisons_dir = env.comparisons_output_dir()

    if not args.no_stage:
        stage_inputs(comparisons_dir)
    if not args.no_render:
        render()
    if not os.path.isfile(BOOK_HTML):
        raise SystemExit(f"{BOOK_HTML} does not exist; render the book first")

    work_dir = args.work_dir or os.path.join(env.output_root(), "publish", LABEL)
    assets = collect(work_dir)
    if args.dry_run:
        print(f"[dry-run] assembled {work_dir}; not published")
        return

    from dse_research_utils.storage.azure import upload_directory_to_blob_storage

    result = upload_directory_to_blob_storage(
        work_dir, LABEL, project=PROJECT, run_id=args.run_id
    )
    print(f"[published] {result.report_url}")

    failures = verify(result.report_url, assets)
    if failures:
        print(
            f"\n[FAILED] {len(failures)} of {len(assets) + 1} published files did not return 200:"
        )
        for failure in failures[:20]:
            print(f"   {failure}")
        raise SystemExit(1)
    print(f"[verified] index.html and all {len(assets)} referenced assets return 200")


if __name__ == "__main__":
    main()
