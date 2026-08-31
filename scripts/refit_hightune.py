# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refit a model — or one of its registered sensitivity variants — at reporting
quality with overridden sampling parameters, without editing source.
Monkey-patches the shared sampling-config factory so the 'rep' tier returns the
requested tune/draws/target_accept/chains; the fit is still treated as
reporting-quality (convergence gate enforced). The refit overwrites the
canonical output dir for whatever it fits (back it up first if you need it).

Note that there is no ``rep-hightune`` sampling configuration: the registered
tiers are ``dev``, ``test``, ``rep-lite`` and ``rep``, and a high-tune run is
this script overriding ``rep``. Such a fit records itself as ``rep`` in its
manifest, with the raised tune/draws in ``sampling.parameters`` — which is why
VG12's and VG13's manifests read ``rep`` at tune=12000, draws=8000.

Usage:
    python scripts/refit_hightune.py <model_key> \
        --tune 12000 --draws 8000 --target-accept 0.99 [--chains 6]

    python scripts/refit_hightune.py vg11 --variant anchor-broad \
        --tune 12000 --draws 8000 --target-accept 0.99

``--variant`` builds the named variant from ``vocab_growth.sensitivity.registry``
and runs it through the same engine ``fit_sensitivity.py`` uses, so the only
difference from that script is the sampling override. It exists because a
variant that misses the convergence gate marginally has no other way to be
retried with heavier tuning — ``fit_sensitivity.py`` takes a registered tier and
the tiers stop at ``rep``. Output lands in the variant's own directory and never
touches the model of record.

Be aware of what a high-tuned variant means for ``compare_sensitivity.py``: it
scores the variant against a baseline that may have been fitted at plain ``rep``,
so a difference between them is no longer attributable to the prior change
alone. Record the mismatch wherever the comparison is reported.
"""
import argparse
import importlib
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup
import dse_research_utils.statistics.models.sampling as S
from dse_research_utils.statistics.models.sampling import SamplingConfiguration

from vocab_growth import environment as env


# Which models have sensitivity variants, and the engine each is fitted through.
# Both come from the catalogue and the variant registry rather than from a table
# maintained here: the copy this replaced was missing VG16, VG21 and VG23, whose
# registered variants were therefore unreachable from both this script and
# scripts/fit_sensitivity.py (issue #273). Importing the variant registry is
# cheap -- it holds no PyMC -- and the engine module is still resolved lazily, so
# a plain model refit does not pay for the engine import.
def _models_with_variants() -> list[str]:
    from vocab_growth.sensitivity.registry import VARIANTS

    return sorted({model_key for model_key, _ in VARIANTS})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument(
        "--variant",
        default=None,
        help=(
            "Registered sensitivity variant to fit instead of the model of "
            "record (e.g. anchor-broad). Output goes to the variant's own dir."
        ),
    )
    p.add_argument("--tune", type=int, required=True)
    p.add_argument("--draws", type=int, required=True)
    p.add_argument("--target-accept", type=float, required=True)
    p.add_argument("--chains", type=int, default=6)
    p.add_argument("--output-dir", default=None)
    freeze_support()
    a = p.parse_args()

    if a.variant and a.model not in _models_with_variants():
        p.error(
            f"No sensitivity variants for model {a.model!r} "
            f"(available: {', '.join(_models_with_variants())})."
        )

    _orig = S.get_sampling_configuration

    def _patched(config: str = "dev", random_seed: int = 47):
        if config in ("rep", "reporting", "report"):
            return SamplingConfiguration(
                draws=a.draws,
                tune=a.tune,
                chains=a.chains,
                cores=a.chains,
                target_accept=a.target_accept,
                random_seed=random_seed,
            )
        return _orig(config, random_seed)

    S.get_sampling_configuration = _patched

    env.set_output_root(a.output_dir)
    setup.init_script()
    target = f"{a.model} [variant: {a.variant}]" if a.variant else a.model
    print(
        f"[refit_hightune] {target}: rep-tier overridden -> tune={a.tune} "
        f"draws={a.draws} target_accept={a.target_accept} chains={a.chains}"
    )

    if a.variant:
        from vocab_growth.models.catalogue import engine_for
        from vocab_growth.sensitivity.registry import build_variant

        runner = engine_for(a.model).resolve("fit")
        # build_variant raises KeyError on an unregistered name, which is the
        # right failure: better than fitting the model of record by accident.
        (vdef,) = build_variant(a.model, a.variant)
        runner("rep", vdef)
    else:
        m = importlib.import_module(f"vocab_growth.models.model_{a.model}")
        m.fit("rep")
