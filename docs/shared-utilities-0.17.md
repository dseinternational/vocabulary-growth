> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Shared utilities 0.17.0

The project now selects `dse-research-utils` from the published `v0.17.0` tag. The tag resolves to release commit `935bd38bdd09da05cd9895fb3ee73d38c27b3e7c`. The [library upgrade guide](https://github.com/dseinternational/research/blob/v0.17.0/docs/migrating-to-0.17.md) describes public sampling diagnostics, reductions of existing diagnostic tables, optional file-permission controls and the fix for nullable missing diagnostics.

The library's Python requirement, dependency minimums and extras are unchanged from `v0.16.2`. This project retains its existing extras, model specifications, sampling thresholds and publication rules. The lock refresh selects the new library tag while retaining unrelated package versions.

Install the updated environment with `uv sync --locked`. Both the installed distribution and `dse_research_utils.__version__` must report `0.17.0`. Apply the project's executable-code signature and fit-compatibility checks before resuming or publishing stored results. A dependency upgrade does not itself approve an older fit. Keep historical manifests and recorded sampling environments intact.

The atomic-file adapter now requests ordinary new-file permissions through `mode="default"`, after the writer returns. The legacy sensitivity CSV reader uses the shared diagnostic reductions and retains its inclusive cutoffs and fallback caveat. The marginal-arm experiment uses the shared energy diagnostic in named chain and draw order and still stops when energy is unavailable. These source changes alter the executable-code signature; the existing saved-fit checks still apply.
