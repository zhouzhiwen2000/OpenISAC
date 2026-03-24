#!/usr/bin/env python3
"""Sync docs/index*.html content from README*.md.

Managed range:
- English: from "## Hardware Setup"
- Chinese: from "## 硬件准备"
"""

from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]


SYNC_JOBS = (
    {
        "readme": ROOT / "README.md",
        "start_heading": "## Hardware Setup",
        "html": ROOT / "docs" / "index.html",
        "section_id": "hardware",
        "range_start": "<!-- README_SYNC_START -->",
        "range_end": "<!-- README_SYNC_END -->",
    },
    {
        "readme": ROOT / "README_zh.md",
        "start_heading": "## 硬件准备",
        "html": ROOT / "docs" / "index_zh.html",
        "section_id": "hardware",
        "range_start": "<!-- README_SYNC_START -->",
        "range_end": "<!-- README_SYNC_END -->",
    },
)


def _extract_markdown_section(md_text: str, start_heading: str) -> str:
    lines = md_text.splitlines()
    start_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == start_heading:
            start_idx = i
            break
    if start_idx < 0:
        raise ValueError(f"Heading not found: {start_heading}")

    section = "\n".join(lines[start_idx:]).strip() + "\n"
    return section


def _render_markdown_to_html(md_text: str) -> str:
    md = MarkdownIt("commonmark", {"html": False})
    md.enable("table")
    return md.render(md_text).strip()


def _replace_managed_range(
    html_text: str,
    section_id: str,
    range_start: str,
    range_end: str,
    rendered_html: str,
) -> str:
    managed_block = (
        f"{range_start}\n"
        f'<section id="{section_id}">\n'
        "<!-- AUTO-GENERATED FROM README: DO NOT EDIT THIS SECTION DIRECTLY -->\n"
        f"{rendered_html}\n"
        "</section>\n"
        f"{range_end}"
    )

    start_idx = html_text.find(range_start)
    end_idx = html_text.find(range_end)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        end_idx = end_idx + len(range_end)
        return html_text[:start_idx] + managed_block + html_text[end_idx:]

    # Fallback for first migration: replace from <section id="hardware"> to </main>
    section_marker = f'<section id="{section_id}">'
    s_idx = html_text.find(section_marker)
    if s_idx == -1:
        raise ValueError(f'Section id="{section_id}" not found in HTML')
    main_end = html_text.find("</main>", s_idx)
    if main_end == -1:
        raise ValueError("Closing </main> not found in HTML")
    return html_text[:s_idx] + managed_block + "\n\n    " + html_text[main_end:]


def sync_once(
    readme_path: Path,
    start_heading: str,
    html_path: Path,
    section_id: str,
    range_start: str,
    range_end: str,
) -> None:
    md_text = readme_path.read_text(encoding="utf-8")
    section_md = _extract_markdown_section(md_text, start_heading)
    rendered = _render_markdown_to_html(section_md)

    html_text = html_path.read_text(encoding="utf-8")
    updated_html = _replace_managed_range(
        html_text=html_text,
        section_id=section_id,
        range_start=range_start,
        range_end=range_end,
        rendered_html=rendered,
    )
    html_path.write_text(updated_html, encoding="utf-8")


def main() -> None:
    for job in SYNC_JOBS:
        sync_once(
            readme_path=job["readme"],
            start_heading=job["start_heading"],
            html_path=job["html"],
            section_id=job["section_id"],
            range_start=job["range_start"],
            range_end=job["range_end"],
        )
    print("Synced docs index pages from README sections starting at Hardware Setup.")


if __name__ == "__main__":
    main()
