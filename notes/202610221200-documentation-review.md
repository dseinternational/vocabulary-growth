# Repository documentation review, 22 October 2026

> [!NOTE]
> Drafted with assistance from OpenAI Codex/GPT-6.

The review began at `c245e65`; the final changes are based on `6c33d29`. It covered the tracked documentation and source-comment inventory: maintained guides, source records, Quarto templates and includes, dated notes, decision records, experiment instructions and code comments and docstrings. Cross-checks used the model registry, likelihood and GP helpers, data mappings, workflow scripts and subsequent decision records. This was an editorial and consistency review, not an independent review of every cited paper or a reproduction of fitted results.

## Corrections

The model-report glossary now distinguishes raw checklist counts on the common denominator from proportional rescaling; concentration from total count variation; a zero-effect reference from an average over children; and an age anchor from the full fitted curve. Its GP, HSGP, predictive-score and sampling definitions no longer promise accuracy or identification that those devices alone cannot establish. The technical-report glossary carries the corresponding reference-inventory and reference-curve explanations.

VG10 and VG16 now describe GP projection followed by point anchoring, which can restore a constant component, and the smooth levelling of the spoken-share trend. Neither operation forces a full curve to be flat or monotone. VG08–VG10 distinguish fixed latent-logit ordering from observed count ranks; multiplying comprehension and spoken-share proportions can also change the ordering of spoken expectations. VG14 and VG15 distinguish the middle signing knot from the ratio maximum and the signed-count maximum. Source and experiment comments no longer claim that current VG10/VG19/VG20 comparisons isolate child structure, or that excluding child effects directly from compositions makes their fitted parameters independent of the association.

The Spanish source record no longer claims to be the repository's only comparison group. The UK 07 record names the current cross-tabulation loader. The summary's comments describe a planned export, and its worked example allows for discrete intervals containing more than their nominal probability. The page about comprehension and speech no longer implies a temporal mechanism in its title. Missing tools, obsolete file references and duplicated explanations were removed where found.

## Historical records

The VG20/VG22 decision retains all four original promotion criteria and links to their outcome and the later role assignments. Its notice distinguishes those recorded criteria from a general validation standard. The interrupted July Mac run retains its environment, measured fits, LOO failure and fix, and recorded artefact locations; empty result sections and the superseded restart recipe were removed.

The August model-retirement summary duplicated the September role decision and current inventory, so it was removed with its index row. Completed promotion and drafting work lists now point to their successors. Analysis evidence, source decisions, pre-specifications and their deviation logs remain. Numerical tables in retained notes still describe their original revisions.

## Compatibility and verification

The only executable Python difference is the glossary's display text. Statistical definitions, likelihoods and algorithms are unchanged; executable Python cells in the Quarto files are unchanged. The three agent-instruction copies remain identical. Glossary text is nevertheless inside the conservatively hashed package, so it changes the executable signature. A retained fit needs the runbook's reviewed, recorded implementation-change procedure before strict reuse under this revision; matching statistical definitions alone does not bypass that check. No production fit or published output was regenerated.

The branch was rebased onto `6c33d29`, its locked Python and documentation environments synchronised, and the data prepared again. Ruff and mypy passed. The complete fast suite passed 2,749 tests with 63 skips; the slow suite passed all 392 tests. The final report, glossary, link and instruction checks passed another 229 tests. Warnings concerned compiler optimisation, Numba object-mode fallback and the low-importance-weight checks exercised by synthetic tests. They do not establish calibration of production results.

Spelling and Markdown formatting passed, and all 158 dated notes have unique index rows. Checks across 248 prose files found no missing targets among 1,023 local source links and Markdown fragments. Quarto inspection passed for the report, summary and paper projects. Pandoc rendered all 45 model-glossary definitions to HTML. AST comparisons confirmed that only the glossary display text changes executable Python; report cells and inline Python are unchanged. The ignore patterns and three instruction copies also match the base. Full model reports and books were not rendered as part of this review.
