# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refit a model at reporting quality with overridden sampling parameters,
without editing source. Monkey-patches the shared sampling-config factory so the
'rep' tier returns the requested tune/draws/target_accept/chains; the fit is
still treated as reporting-quality (convergence gate enforced). The refit
overwrites the canonical model output dir (back it up first if you need it).

Usage:
    python scripts/refit_hightune.py <model_key> \
        --tune 12000 --draws 8000 --target-accept 0.99 [--chains 6]
"""
import argparse
import importlib
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup
import dse_research_utils.statistics.models.sampling as S
from dse_research_utils.statistics.models.sampling import SamplingConfiguration

from vocab_growth import environment as env

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--tune", type=int, required=True)
    p.add_argument("--draws", type=int, required=True)
    p.add_argument("--target-accept", type=float, required=True)
    p.add_argument("--chains", type=int, default=6)
    p.add_argument("--output-dir", default=None)
    freeze_support()
    a = p.parse_args()

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
    print(
        f"[refit_hightune] {a.model}: rep-tier overridden -> tune={a.tune} "
        f"draws={a.draws} target_accept={a.target_accept} chains={a.chains}"
    )
    m = importlib.import_module(f"vocab_growth.models.model_{a.model}")
    m.fit("rep")
