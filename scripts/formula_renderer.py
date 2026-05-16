#!/usr/bin/env python3
r"""
Render LaTeX math formulas to PNG images using matplotlib.
Handles standard mathtext and \begin{pmatrix}...\end{pmatrix} environments.
"""

import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})


def _parse_matrix(content: str):
    """Parse a pmatrix content string into a 2D list of cell strings."""
    rows = content.split(r"\\")
    matrix = []
    for row in rows:
        cells = [c.strip() for c in row.split("&")]
        if cells and cells != [""]:
            matrix.append(cells)
    return matrix


def _render_mathtext(formula: str, fontsize: int = 13, dpi: int = 150, color="#222222"):
    """Render a simple LaTeX formula using matplotlib mathtext. Returns a PIL Image."""
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.text(
        0, 0, f"${formula}$",
        fontsize=fontsize,
        color=color,
        verticalalignment="baseline",
        horizontalalignment="left",
    )
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight",
                pad_inches=0.04, transparent=True)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf)


def _render_matrix_cells(matrix, fontsize=13, dpi=150, color="#222222"):
    """Render a matrix as a table image with parentheses."""
    nrows = len(matrix)
    ncols = max(len(row) for row in matrix) if matrix else 0

    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis("off")

    cell_texts = []
    for row in matrix:
        cell_texts.extend(row)

    tbl = ax.table(
        cellText=matrix,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.4)

    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor("none")
        cell.set_facecolor("none")
        cell.get_text().set_color(color)

    ax.set_xlim(-1.2, ncols + 0.2)
    ax.set_ylim(-0.2, nrows + 1.2)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight",
                pad_inches=0.05, transparent=True)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf)


def _combine_images_horizontal(images, spacing=10):
    """Combine a list of PIL Images horizontally with spacing."""
    from PIL import Image
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    total_w = sum(widths) + spacing * (len(images) - 1)
    max_h = max(heights)

    combined = Image.new("RGBA", (total_w, max_h), (255, 255, 255, 0))
    x = 0
    for im in images:
        y = (max_h - im.height) // 2
        combined.paste(im, (x, y), im)
        x += im.width + spacing
    return combined


def render_formula(formula: str, fontsize: int = 13, dpi: int = 150, color="#222222"):
    r"""Render a LaTeX formula to a PNG file or buffer.
    
    Handles:
    - Standard mathtext
    - \begin{pmatrix}...\end{pmatrix}
    - Mixed: text + matrix + text (split on \quad / ,)
    """
    if r"\begin{pmatrix}" not in formula:
        return _render_mathtext(formula, fontsize, dpi, color)

    # Handle formulas containing pmatrix by splitting and composing
    # Split pattern: pmatrix blocks vs surrounding text
    pattern = r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}"
    parts = []
    last_end = 0
    matrices = []

    for m in re.finditer(pattern, formula, re.DOTALL):
        # Text before this matrix
        text_before = formula[last_end : m.start()]
        if text_before.strip():
            parts.append(("text", text_before.strip().rstrip(",").strip()))
        parts.append(("matrix", m.group(1)))
        last_end = m.end()

    text_after = formula[last_end:].strip()
    if text_after.strip().lstrip(",").strip():
        parts.append(("text", text_after.strip().lstrip(",").strip()))

    images = []
    for ptype, pcontent in parts:
        if ptype == "text":
            img = _render_mathtext(pcontent, fontsize, dpi, color)
            images.append(img)
        else:
            matrix = _parse_matrix(pcontent)
            if matrix:
                img = _render_matrix_cells(matrix, fontsize, dpi, color)
                images.append(img)

    if not images:
        return _render_mathtext(formula, fontsize, dpi, color)

    return _combine_images_horizontal(images, spacing=12)


def render_formula_to_file(formula: str, filepath: str | Path,
                           fontsize: int = 13, dpi: int = 150, color="#222222"):
    """Render formula and save to file."""
    img = render_formula(formula, fontsize, dpi, color)
    img.save(str(filepath), format="PNG")
    return filepath
