"""The text a model sees for each release.

The input is the chosen press release (see secbench.releases). If it has no income
statement, the first other EX-99.x exhibit that has one is appended under a heading:
ExxonMobil's Q2 2026 release moved its statements from EX-99.1 to EX-99.2. Which
exhibits count is secbench.link.input_exhibits, the rule link's earnings check reads
too, so a release never links through an exhibit that isn't in its input.

Library only: build_input(record_path), release_records(releases_dir).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from secbench.link import input_exhibits


@dataclass(frozen=True)
class ReleaseInput:
    text: str
    exhibits: tuple[str, ...]  # exhibit types used, in order ("EX-99.1", "EX-99.2")


def exhibit_heading(exhibit: dict) -> str:
    description = (exhibit.get("description") or "").strip()
    return f"# {exhibit['type']}" + (f": {description}" if description else "")


def build_input(record_path: Path) -> ReleaseInput:
    """The input for the release saved as record_path (<accession>.json): the chosen
    exhibit's markdown unchanged, plus, if it has no income statement, the first
    other exhibit (in filing-index order) that has one, under its heading (see
    secbench.link.input_exhibits)."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    [(release, text), *others] = input_exhibits(record, record_path.parent)
    for exhibit, markdown in others:
        text = f"{text.rstrip()}\n\n{exhibit_heading(exhibit)}\n\n{markdown}"
    return ReleaseInput(
        text, (release["type"], *(exhibit["type"] for exhibit, _ in others))
    )


def release_records(releases_dir: Path) -> dict[str, Path]:
    """Release accession -> its saved record (<cik>/<accession>.json)."""
    return {path.stem: path for path in sorted(releases_dir.glob("*/*.json"))}
