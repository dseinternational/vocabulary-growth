# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive ``data/vocab_data_us_01.csv`` from the Edgin item-level source files.

Why this exists rather than reading the by-child export
-------------------------------------------------------
``data/wordbank_administration_data.csv`` is produced by Wordbank's By-Child
Summary Data page, which calls ``wordbankr::get_administration_data()`` **without**
``filter_age = FALSE``.  That default filters every administration to its
instrument's registered age window (English (American) WG 8-18 months, WS 16-30),
*before* the page's own age slider is built -- so no setting in the web UI can
recover the rest.  For the Edgin Down syndrome subset that silently truncates 347
administrations to 196.

Two things follow that the by-child export cannot give us:

1. **The out-of-window administrations.**  They are in the Wordbank database
   (``instruments/import_dataset.py`` applies no age filter), just not reachable
   through the download page.  Most of them turn out to be defective -- see
   ``notes/202608031500-edgin-out-of-window-administrations.md`` -- which is a
   reason to hold them, documented, rather than a reason not to have them.
2. **Identification of empty administrations.**  Three WG and one WS source rows
   have *every* word item blank.  Wordbank scores them as zero, so in the by-child
   export they are indistinguishable from a genuine "understands no words" record:
   two are Down syndrome rows inside the age window, and at 12 months the export
   holds two ``(0, 0)`` rows of which only one is the empty form.  Only the
   item-level file separates them.

Provenance
----------
Source files are the contributor files committed to the public ``langcog/wordbank``
repository, which is what populates the Wordbank database.  ``--manifest`` records
the source URLs, retrieval date, SHA-256 of each downloaded file and the derived row
counts, per the provenance requirement adopted for Wordbank-derived data on the
PR #183 thread.

Scoring follows the instruments' own value maps (``*_values.csv``):

- WG (396 word items): ``1`` = understands, ``2`` = understands and says.  So
  comprehension counts ``{1, 2}`` and production counts ``{2}``.
- WS (680 word items): ``1`` = produces.  The form has no comprehension section;
  Wordbank reports comprehension equal to production by data convention, and this
  script reproduces that so the downstream bivariate-form guard
  (``WORDBANK_BIVARIATE_FORMS``) stays the single place the proxy is discarded.

Verification gate
-----------------
``--verify`` re-derives the in-window Down syndrome administrations and checks them
against ``wordbank_administration_data.csv`` as a multiset of
``(age, comprehension, production)``.  This reproduces 87 WG and 109 WS rows
exactly; the reconstruction is only trustworthy for the out-of-window rows because
it is exact on the in-window ones.

Usage::

    python scripts/build_us01_source.py                 # fetch, derive, write CSV
    python scripts/build_us01_source.py --verify         # also run the gate
    python scripts/build_us01_source.py --cache-dir DIR  # reuse downloaded files
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import os
import sys
import urllib.request

REPO_RAW = "https://raw.githubusercontent.com/langcog/wordbank/master/raw_data"

# (form, directory, file stem, word-item count, norming age window)
FORMS = (
    ("WG", "English_American_WG", "EnglishWG_Edgin", 396, (8, 18)),
    ("WS", "English_American_WS", "EnglishWS_Edgin", 680, (16, 30)),
)

# Source column names differ between the two contributor files.
SUBJECT_COLUMNS = ("SubID", "ID")
STATUS_COLUMNS = ("DevStatus", "DevelopmentalDiagnosis")
AGE_COLUMN = "CDIAge"
SEX_COLUMN = "Gender"

# ``*_values.csv`` maps the study's condition codes. Code 0 maps to an *empty*
# condition name, which Wordbank's importer nonetheless links -- flipping those
# children to ``typically_developing = false`` with a blank label. They are a
# separate comparison group, not Down syndrome children with a missing code: no
# child carries both codes.
DEV_STATUS = {
    "0": "comparison",
    "1": "down_syndrome",
    "2": "pre_term",
    "3": "autism_spectrum",
    "": "unrecorded",
}

OUTPUT_COLUMNS = (
    "subject_id",
    "form",
    "age",
    "sex",
    "dev_status",
    "comprehension",
    "production",
    "survey_vocab_max",
    "in_norming_window",
)


def _fetch(url: str, cache_dir: str | None) -> bytes:
    if cache_dir:
        cached = os.path.join(cache_dir, os.path.basename(url))
        if os.path.exists(cached):
            with open(cached, "rb") as handle:
                return handle.read()
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, os.path.basename(url)), "wb") as handle:
            handle.write(payload)
    return payload


def _rows(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def _first_present(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in row:
            return name
    raise KeyError(f"none of {candidates} present; have {sorted(row)[:12]}")


def derive(cache_dir: str | None) -> tuple[list[dict], dict]:
    records: list[dict] = []
    manifest: dict = {
        "retrieved": datetime.date.today().isoformat(),
        "source_repository": "https://github.com/langcog/wordbank",
        "files": [],
        "excluded_empty_administrations": [],
    }

    for form, directory, stem, n_items, (age_min, age_max) in FORMS:
        files = {}
        for kind in ("data", "fields", "values"):
            url = f"{REPO_RAW}/{directory}/{stem}_{kind}.csv"
            payload = _fetch(url, cache_dir)
            files[kind] = payload
            manifest["files"].append(
                {
                    "url": url,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

        fields = _rows(files["fields"])
        word_columns = [
            row["column"] for row in fields if (row.get("type") or "").strip() == "word"
        ]
        if len(word_columns) != n_items:
            raise SystemExit(
                f"{form}: expected {n_items} word items, found {len(word_columns)}. "
                "The instrument definition has changed; re-check the item counts "
                "before regenerating."
            )

        data = _rows(files["data"])
        subject_column = _first_present(data[0], SUBJECT_COLUMNS)
        status_column = _first_present(data[0], STATUS_COLUMNS)

        empty = 0
        for row in data:
            responses = [(row.get(column) or "").strip() for column in word_columns]

            # An administration with every word item blank is not a measurement.
            # Wordbank scores it as zero, which is indistinguishable from a genuine
            # zero once the item responses are gone. Drop it here, at the only point
            # where it is still identifiable, and record it in the manifest.
            if all(value == "" for value in responses):
                empty += 1
                manifest["excluded_empty_administrations"].append(
                    {
                        "form": form,
                        "subject_id": (row.get(subject_column) or "").strip(),
                        "age": (row.get(AGE_COLUMN) or "").strip(),
                        "dev_status": DEV_STATUS.get(
                            (row.get(status_column) or "").strip(), "unrecorded"
                        ),
                    }
                )
                continue

            raw_age = (row.get(AGE_COLUMN) or "").strip()
            if not raw_age:
                continue
            age = float(raw_age)

            if form == "WG":
                comprehension = sum(1 for value in responses if value in ("1", "2"))
                production = sum(1 for value in responses if value == "2")
            else:
                production = sum(1 for value in responses if value == "1")
                comprehension = production

            sex = (row.get(SEX_COLUMN) or "").strip().upper()
            records.append(
                {
                    "subject_id": (row.get(subject_column) or "").strip(),
                    "form": form,
                    "age": int(age) if age.is_integer() else age,
                    "sex": sex if sex in ("M", "F") else "",
                    "dev_status": DEV_STATUS.get(
                        (row.get(status_column) or "").strip(), "unrecorded"
                    ),
                    "comprehension": comprehension,
                    "production": production,
                    "survey_vocab_max": n_items,
                    "in_norming_window": age_min <= age <= age_max,
                }
            )
        manifest.setdefault("per_form", {})[form] = {
            "source_rows": len(data),
            "empty_administrations_excluded": empty,
            "word_items": n_items,
            "norming_window": [age_min, age_max],
        }

    by_status = collections.Counter(record["dev_status"] for record in records)
    manifest["derived_rows"] = len(records)
    manifest["derived_rows_by_dev_status"] = dict(sorted(by_status.items()))
    manifest["derived_down_syndrome_in_window"] = sum(
        1
        for record in records
        if record["dev_status"] == "down_syndrome" and record["in_norming_window"]
    )
    manifest["derived_down_syndrome_total"] = by_status["down_syndrome"]
    return records, manifest


def verify(records: list[dict], export_path: str) -> bool:
    """Check in-window DS records against the by-child export, as a multiset."""
    try:
        import duckdb
    except ImportError:
        print("duckdb unavailable; skipping verification", file=sys.stderr)
        return True

    ok = True
    for form in ("WG", "WS"):
        export = duckdb.sql(
            f"""
            SELECT age, comprehension, production
            FROM read_csv_auto('{export_path}')
            WHERE dataset_name = 'Edgin' AND form = '{form}'
              AND lower(health_conditions) = 'down syndrome'
            """
        ).df()
        expected = collections.Counter(
            (int(row.age), int(row.comprehension), int(row.production))
            for row in export.itertuples()
        )
        derived = collections.Counter(
            (int(r["age"]), r["comprehension"], r["production"])
            for r in records
            if r["form"] == form
            and r["dev_status"] == "down_syndrome"
            and r["in_norming_window"]
        )
        # The two empty administrations are dropped here but present in the export
        # as fabricated zeros, so the derived multiset is the export minus those.
        residual = expected - derived
        surplus = derived - expected
        empty_zeros = all(count == 0 for _, _, count in residual)
        status = "OK" if not surplus and empty_zeros else "MISMATCH"
        if surplus or not empty_zeros:
            ok = False
        print(
            f"  {form}: export={sum(expected.values())} derived={sum(derived.values())} "
            f"export-only={sorted(residual.elements())} derived-only="
            f"{sorted(surplus.elements())} -> {status}"
        )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="./data/vocab_data_us_01.csv")
    parser.add_argument("--manifest", default="./data/vocab_data_us_01_manifest.json")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--export",
        default="./data/wordbank_administration_data.csv",
        help="by-child export used by --verify",
    )
    args = parser.parse_args()

    records, manifest = derive(args.cache_dir)
    records.sort(key=lambda r: (r["subject_id"], r["form"], r["age"]))

    with open(args.output, "w", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    with open(args.manifest, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {len(records):,} rows to {args.output}")
    print(f"  by developmental status: {manifest['derived_rows_by_dev_status']}")
    print(
        f"  Down syndrome: {manifest['derived_down_syndrome_total']} total, "
        f"{manifest['derived_down_syndrome_in_window']} inside the norming window"
    )
    for form, summary in manifest["per_form"].items():
        print(
            f"  {form}: {summary['source_rows']} source rows, "
            f"{summary['empty_administrations_excluded']} empty administrations excluded"
        )

    if args.verify:
        print("verifying in-window Down syndrome rows against the by-child export:")
        if not verify(records, args.export):
            print("VERIFICATION FAILED", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
