> [!NOTE]
> Drafted and revised with assistance from OpenAI Codex/GPT-6.

# Shared library 0.17.0 upgrade

On 4 October 2026 the project selected `dse-research-utils` from the published `v0.17.0` tag. The tag resolves to release commit `935bd38bdd09da05cd9895fb3ee73d38c27b3e7c`. The [library upgrade guide](https://github.com/dseinternational/research/blob/v0.17.0/docs/migrating-to-0.17.md) describes public sampling diagnostics, reductions of existing diagnostic tables, optional file-permission controls and the fix for nullable missing diagnostics.

The library's Python requirement, dependency minimums and extras are unchanged from `v0.16.2`. This project retains its existing extras, model specifications, sampling thresholds and publication rules. That upgrade retained unrelated package versions; later lock refreshes have their own commits.

At that revision, `uv sync --locked` installed distribution and module version `0.17.0`. For the current environment use [the environment runbook](../docs/runbooks/environment-locks.md), `pyproject.toml` and `uv.lock`. Apply the project's executable-code signature and fit-compatibility checks before resuming or publishing stored results. A dependency upgrade does not itself approve an older fit. Keep historical manifests and recorded sampling environments intact.

The atomic-file adapter now requests ordinary new-file permissions through `mode="default"`, after the writer returns. The marginal-arm experiment uses the shared energy diagnostic in named chain and draw order and still stops when energy is unavailable. These source changes alter the executable-code signature; the existing saved-fit checks still apply.

The legacy sensitivity reader consumes a rounded CSV. Its reduction remains local because the shared table reducer requires unrounded input.
