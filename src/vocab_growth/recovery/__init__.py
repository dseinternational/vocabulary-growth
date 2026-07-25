# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parameter-recovery tooling for the statistical-validity review (issue #163).

A recovery check regenerates a dataset from a model at a known parameter draw,
refits the model to it, and asks whether the truth is found again. It is the
counterpart to the posterior-predictive calibration every fit already writes:
calibration asks whether the model describes the observed data, recovery asks
whether the sampler can identify the model's own parameters from data of this
size, shape and missingness.

`spec` declares what each engine needs in order to be simulated; `simulate`
forward-simulates a dataset from the model's own likelihood nodes at a fixed
draw; `refit` reruns the engine's pipeline against that dataset with only the
data-preparation stage substituted; `compare` scores the recovered posterior
against the truth.

See `docs/runbooks/parameter-recovery.md` for how to run a study and how to read
the result — in particular, what a handful of replicates can and cannot show.
"""
