# Reproducing the software environment

> [!NOTE]
> Drafted by an LLM-based AI tool (OpenAI Codex/GPT-5).

`environment.yml` is the readable environment specification: it states the supported ranges and explains why important constraints exist. The generated `conda-lock.yml` resolves the compiled Conda layer to exact package builds for `linux-64` and `osx-arm64`. `requirements-pip.lock` records the exact pip-installed packages from a verified `dse-vocab-growth` environment, including the immutable Git commit of `dse-research-utils`. The project itself remains editable and is installed from the checked-out Git revision.

## Create an environment from the lock

Install the lock tool in a small separate environment or virtual environment; it is tooling, not a runtime dependency of the statistical model.

```bash
python -m pip install -r requirements-lock-tool.txt
conda-lock install --name dse-vocab-growth conda-lock.yml
conda run --name dse-vocab-growth python -m pip install -r requirements-pip.lock
conda run --name dse-vocab-growth python -m pip install -e ./
conda run --name dse-vocab-growth dse-check-env environment.yml
```

The lock intentionally covers CPU installations. GPU drivers and CUDA are host-specific and remain an opt-in overlay rather than part of the reporting baseline.

## Refresh locks after an intentional dependency change

First update and verify the normal environment. Then run the generator using the separate Python environment in which `conda-lock` is installed:

```bash
python scripts/lock_environment.py --environment dse-vocab-growth
git diff -- conda-lock.yml requirements-pip.lock
```

Commit the readable specification and both generated lock files together. Do not refresh locks as an unrelated formatting change: a lock diff is part of the scientific computing change and should be reviewed for unexpected solver upgrades.

## Relationship to fit manifests

The lock reconstructs a known software environment prospectively. Every completed model fit also writes the versions it actually used into `fit_manifest.json`. The two records answer different questions: the lock says what a clean environment should contain, while the manifest says what a particular fit did contain. A reporting run should agree with both.
