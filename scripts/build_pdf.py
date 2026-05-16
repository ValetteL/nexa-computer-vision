#!/usr/bin/env python3
"""
Build NEXA Computer Vision Course PDF from Markdown sources.

Pipeline:
  Markdown → Pre-process formulas (→ PNG) → Pandoc (HTML5) → WeasyPrint (PDF)
"""

import re
import subprocess
import sys
import shutil
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PDF = COURSE_DIR / "NEXA_Computer_Vision_Complet.pdf"
FORMULA_DIR = COURSE_DIR / "outputs" / "pdf_formulas"
PANDOC = Path("/tmp/pandoc-3.9.0.2/bin/pandoc")
CSS_FILE = COURSE_DIR / "scripts" / "pdf_style.css"
TEMPLATE_FILE = COURSE_DIR / "scripts" / "pdf_template.html"
SOURCE_FILES = [
    "JOUR-01.md",
    "JOUR-02.md",
    "JOUR-03.md",
    "PROJET-RECONNAISSANCE-FACIALE.md",
]

COVER_MD = """---
title: "Computer Vision — Des descripteurs classiques à la détection d'objets"
subtitle: "Module de formation — 3 jours"
author: "NEXA School"
date: "Mai 2026"
---

"""


def check_tools():
    if not PANDOC.exists():
        print(f"Error: Pandoc not found at {PANDOC}")
        sys.exit(1)
    try:
        import weasyprint
    except ImportError:
        print("Error: weasyprint not installed. Run: pip install weasyprint")
        sys.exit(1)
    print(f"[OK] Pandoc: {PANDOC}")
    print(f"[OK] WeasyPrint: {weasyprint.__version__}")


def _fix_list_breaks(markdown: str) -> str:
    """Ensure blank lines before list items so Pandoc parses them as proper lists."""
    lines = markdown.split("\n")
    result = []
    for i, line in enumerate(lines):
        is_list = line.strip().startswith("- ") or line.strip().startswith("* ")
        if is_list and i > 0:
            prev = lines[i - 1]
            prev_is_list = prev.strip().startswith("- ") or prev.strip().startswith("* ")
            prev_is_blank = prev.strip() == ""
            if not prev_is_list and not prev_is_blank:
                result.append("")
        result.append(line)
    return "\n".join(result)


def preprocess_formulas(markdown: str) -> str:
    """Replace display ($$..$$) and inline ($..$) math with rendered PNG images."""
    sys.path.insert(0, str(COURSE_DIR / "scripts"))
    from formula_renderer import render_formula_to_file

    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    counter = [0]
    inline_counter = [0]

    def _render_display(match):
        counter[0] += 1
        formula = match.group(1).strip()
        fname = FORMULA_DIR / f"display_{counter[0]:03d}.png"
        render_formula_to_file(formula, str(fname), fontsize=12, dpi=150)
        rel = fname.relative_to(COURSE_DIR)
        return f"![]({rel})\n"

    def _render_inline(match):
        inline_counter[0] += 1
        formula = match.group(1).strip()
        fname = FORMULA_DIR / f"inline_{inline_counter[0]:03d}.png"
        render_formula_to_file(formula, str(fname), fontsize=11, dpi=150)
        rel = fname.relative_to(COURSE_DIR)
        return f'<img src="{rel}" class="formula-inline" alt="{formula}" />'

    # Replace display math first ($$..$$)
    result = re.sub(r"\$\$(.*?)\$\$", _render_display, markdown, flags=re.DOTALL)
    # Replace inline math ($...$) - must not match $$ cases
    result = re.sub(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", _render_inline, result)

    total = counter[0] + inline_counter[0]
    print(f"[*] Rendered {total} formulas ({counter[0]} display, {inline_counter[0]} inline)")
    return result


def combine_sources():
    print("[*] Combining and preprocessing Markdown sources...")
    parts = [COVER_MD]
    for fname in SOURCE_FILES:
        path = COURSE_DIR / fname
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        parts.append(f"\n\n")
        parts.append(content)
    combined = "".join(parts)
    combined = preprocess_formulas(combined)
    combined = _fix_list_breaks(combined)
    tmp_md = COURSE_DIR / ".build" / "course_combined.md"
    tmp_md.parent.mkdir(parents=True, exist_ok=True)
    tmp_md.write_text(combined, encoding="utf-8")
    print(f"[OK] Combined source: {tmp_md}")
    return tmp_md


def build_html(md_path):
    print("[*] Converting Markdown to HTML with Pandoc...")
    html_path = md_path.with_suffix(".html")
    cmd = [
        str(PANDOC),
        str(md_path),
        "--from", "markdown",
        "--to", "html5",
        "--standalone",
        "--template", str(TEMPLATE_FILE),
        "--css", str(CSS_FILE),
        "--syntax-highlighting", "pygments",
        "--toc", "--toc-depth", "3",
        "-o", str(html_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=COURSE_DIR)
    if result.returncode != 0:
        print(f"Pandoc error:\n{result.stderr}")
        sys.exit(1)
    if result.stderr:
        print(f"  Stderr: {result.stderr[:300]}")
    print(f"[OK] HTML generated: {html_path}")
    return html_path


def build_pdf(html_path):
    print("[*] Generating PDF with WeasyPrint...")
    from weasyprint import HTML
    HTML(str(html_path), base_url=str(COURSE_DIR)).write_pdf(str(OUTPUT_PDF))
    size_mb = OUTPUT_PDF.stat().st_size / (1024 * 1024)
    print(f"[OK] PDF generated: {OUTPUT_PDF.name} ({size_mb:.1f} MB)")


def clean():
    build_dir = COURSE_DIR / ".build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("[OK] Cleaned .build directory")


def main():
    print("=" * 60)
    print("NEXA Computer Vision — PDF Builder")
    print("=" * 60)
    check_tools()
    clean()
    md_path = combine_sources()
    html_path = build_html(md_path)
    build_pdf(html_path)
    clean()
    print("Done!")


if __name__ == "__main__":
    main()
