# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Figure text must not use symbols that the plot font cannot draw.

Since ``dse-research-utils`` 0.16.0 the shared plot style sets text in Noto
Sans. Noto Sans has almost nothing in the arrow, mathematical-operator and
other symbol blocks (U+2190-U+27FF): of those code points it draws only the
minus sign, U+2212, and the dotted circle that displays combining marks.
matplotlib does not fall back to another font for plain
text under the style's generic ``sans-serif`` family, so a literal ``≤`` or
``→`` in a label is drawn as a missing-glyph box, with only a warning in the
log. Written as mathtext (``$\\leq$``, ``$\\rightarrow$``) the same symbol comes
from Noto Sans Math and renders.

This scans the figure-producing code for string literals passed to
matplotlib's text functions, or as a ``label``, and fails on any such symbol
outside a ``$...$`` span. The font files are not read, so the check runs where
the fonts are not installed, as in CI.
"""

import ast
import re
from pathlib import Path

REPO = Path(__file__).parents[1]
SOURCES = sorted((REPO / "src" / "vocab_growth").rglob("*.py")) + sorted((REPO / "scripts").glob("*.py"))

#: matplotlib calls whose positional string arguments are drawn as text.
TEXT_CALLS = {
    "annotate", "figtext", "set_text", "set_title", "set_xlabel", "set_ylabel",
    "supxlabel", "supylabel", "suptitle", "text", "title", "xlabel", "ylabel",
}
#: Keyword arguments drawn as text on any call (legend entries).
TEXT_KEYWORDS = {"label"}
#: The symbol blocks Noto Sans leaves out, and the one code point in them it draws.
SYMBOL_BLOCKS = range(0x2190, 0x2800)
DRAWN = {"−"}
MATHTEXT = re.compile(r"(?<!\\)\$.*?(?<!\\)\$")


def _strings(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [part.value for part in node.values if isinstance(part, ast.Constant)]
    return []


def _undrawable(text: str) -> list[str]:
    plain = MATHTEXT.sub("", text)
    return sorted({ch for ch in plain if ord(ch) in SYMBOL_BLOCKS and ch not in DRAWN})


def test_figure_text_uses_only_symbols_the_plot_font_draws():
    problems = []
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            candidates = list(node.args) if name in TEXT_CALLS else []
            candidates += [kw.value for kw in node.keywords if kw.arg in TEXT_KEYWORDS]
            for candidate in candidates:
                for text in _strings(candidate):
                    if bad := _undrawable(text):
                        rel = path.relative_to(REPO).as_posix()
                        problems.append(f"{rel}:{node.lineno}: {' '.join(bad)} in {text!r}")
    assert not problems, "Write these symbols as mathtext:\n" + "\n".join(problems)


def test_the_check_catches_a_literal_symbol_and_allows_mathtext():
    assert _undrawable("P(Y ≤ k)") == ["≤"]
    assert _undrawable(r"$P(Y \leq k)$ and sign $\rightarrow$ speech") == []
    assert _undrawable("a − b") == []
    assert _undrawable(r"cost \$5 → more") == ["→"]
