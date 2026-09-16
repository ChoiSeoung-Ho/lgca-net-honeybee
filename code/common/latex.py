"""Helpers for emitting booktabs ``tabular`` *bodies* (no table/caption wrapper).

Every fragment written by this package is meant to be pulled into the
manuscript with ``\\input{tables/xyz.tex}`` from inside the author's own
``\\begin{table}...\\end{table}``, so the fragments contain exactly one
``tabular`` environment and nothing else.

Number formatting convention: 3 decimals everywhere, except IoU which uses 4.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Optional, Sequence

FLOAT_FMT = "{:.3f}"
IOU_FMT = "{:.4f}"


def esc(text) -> str:
    """Escape the LaTeX special characters that can appear in model names."""
    s = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def fmt(value, decimals: int = 3, dash: str = "--") -> str:
    """Format a float to ``decimals`` places; NaN/None become an en-dash."""
    if value is None:
        return dash
    try:
        v = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if not math.isfinite(v):
        return dash
    return f"{v:.{decimals}f}"


def fmt_ci(mean, lo, hi, decimals: int = 3) -> str:
    """``0.928 (0.873--0.983)``."""
    return f"{fmt(mean, decimals)} ({fmt(lo, decimals)}--{fmt(hi, decimals)})"


def fmt_pm(mean, sd, decimals: int = 3) -> str:
    """``0.928 $\\pm$ 0.021``."""
    return f"{fmt(mean, decimals)} $\\pm$ {fmt(sd, decimals)}"


def fmt_p(p, decimals: int = 3) -> str:
    """p-value with a ``<0.001`` floor."""
    if p is None:
        return "--"
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(v):
        return "--"
    if v < 0.001:
        return "$<$0.001"
    return f"{v:.{decimals}f}"


def tagged(name: str, run_tag: Optional[str] = None) -> str:
    """Append a run tag to a file stem: ``('cv_metrics', 'original')`` -> ``cv_metrics_original``.

    An empty or ``None`` tag returns the stem unchanged, so the untagged
    (default) run keeps the original file names.
    """
    tag = (run_tag or "").strip()
    return f"{name}_{tag}" if tag else name


def tabular(
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    align: Optional[str] = None,
    midrules_after: Sequence[int] = (),
    notes: Optional[Sequence[str]] = None,
    group_header: Optional[Sequence[Tuple[str, int]]] = None,
) -> str:
    """Build a booktabs ``tabular`` body.

    Parameters
    ----------
    header:
        Column headers, already LaTeX-ready (not escaped again).
    rows:
        Iterable of row cell sequences, already LaTeX-ready.
    align:
        Column spec, default ``l`` for the first column and ``c`` for the rest.
    midrules_after:
        Row indices (0-based) after which to insert a ``\\midrule``.
    notes:
        Optional lines emitted as ``%`` comments above the tabular (units,
        provenance) so the fragment stays a pure tabular.
    group_header:
        Optional spanning header row as ``(label, span)`` pairs, emitted with
        ``\\multicolumn`` plus ``\\cmidrule`` under every label wider than one
        column. The spans must sum to the number of columns. Requires
        ``booktabs`` (already needed for the rules).
    """
    ncol = len(header)
    if align is None:
        align = "l" + "c" * (ncol - 1)
    out = []
    for n in notes or []:
        out.append(f"% {n}")
    out.append(r"\begin{tabular}{" + align + "}")
    out.append(r"\toprule")
    if group_header:
        total = sum(int(n) for _, n in group_header)
        if total != ncol:
            raise ValueError(
                f"group_header spans sum to {total} but the table has {ncol} columns")
        cells, rules, col = [], [], 1
        for label, span in group_header:
            span = int(span)
            if span == 1:
                cells.append(label)
            else:
                cells.append(r"\multicolumn{%d}{c}{%s}" % (span, label))
                if label.strip():
                    rules.append(r"\cmidrule(lr){%d-%d}" % (col, col + span - 1))
            col += span
        out.append(" & ".join(cells) + r" \\")
        if rules:
            out.append(" ".join(rules))
    out.append(" & ".join(header) + r" \\")
    out.append(r"\midrule")
    rows = list(rows)
    for i, r in enumerate(rows):
        out.append(" & ".join(str(c) for c in r) + r" \\")
        if i in set(midrules_after) and i != len(rows) - 1:
            out.append(r"\midrule")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out) + "\n"


def write_fragment(path: str, content: str) -> str:
    """Write a LaTeX fragment, creating parent directories."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def placeholder(script: str, what: str = "") -> str:
    """A compile-safe stand-in fragment for a table whose inputs are missing."""
    label = f"{what}: " if what else ""
    body = (
        r"\begin{tabular}{l}" "\n"
        r"\toprule" "\n"
        r"Result pending \\" "\n"
        r"\midrule" "\n"
        rf"\textcolor{{red}}{{[pending: run {esc(script)}]}} \\" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
    )
    return f"% {label}auto-generated placeholder; requires {script}\n" + body


def bold(s: str) -> str:
    """Bold a cell."""
    return r"\textbf{" + s + "}"
