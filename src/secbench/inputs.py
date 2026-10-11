"""The text a model sees for each release.

The input is the chosen press release (see secbench.releases). If it has no income
statement, the first other EX-99.x exhibit that has one is appended under a heading:
ExxonMobil's Q2 2026 release moved its statements from EX-99.1 to EX-99.2.

Library only: build_input(record_path), release_records(releases_dir).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from secbench.link import (
    is_net_income_row,
    is_per_share_row,
    markdown_tables,
    row_labels,
)
from secbench.releases import is_amount

# "Total net sales", "Total revenues and other income", "Sales and other operating
# revenue", "Total sales and revenues"; not "Cost of sales" or "Cost of revenue".
REVENUE_LABEL = re.compile(
    r"(?<!cost of )\b(revenues?|net sales|sales)\b", re.IGNORECASE
)


def is_revenue_row(row: list[str]) -> bool:
    return any(is_amount(c) for c in row) and any(
        REVENUE_LABEL.search(label) for label in row_labels(row)
    )


def has_income_statement(markdown: str) -> bool:
    """A table whose body (below its header row) has a revenue row, a net income row
    and a per-share row. Stricter than link.check_earnings, which ExxonMobil's Q2
    2026 EX-99.1 passes with a summary table of earnings and per-share rows."""
    for table in markdown_tables(markdown):
        body = table[1:]
        if (
            any(map(is_revenue_row, body))
            and any(map(is_net_income_row, body))
            and any(map(is_per_share_row, body))
        ):
            return True
    return False


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
    other exhibit (in filing-index order) that has one."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    chosen = [e for e in record["exhibits"] if e["chosen"]]
    if len(chosen) != 1 or not chosen[0]["file"]:
        raise ValueError(f"{record_path}: no single chosen exhibit with a file")
    release = chosen[0]
    text = (record_path.parent / release["file"]).read_text(encoding="utf-8")
    if has_income_statement(text):
        return ReleaseInput(text, (release["type"],))
    for exhibit in record["exhibits"]:
        if exhibit is release or not exhibit["file"]:
            continue
        markdown = (record_path.parent / exhibit["file"]).read_text(encoding="utf-8")
        if has_income_statement(markdown):
            text = f"{text.rstrip()}\n\n{exhibit_heading(exhibit)}\n\n{markdown}"
            return ReleaseInput(text, (release["type"], exhibit["type"]))
    return ReleaseInput(text, (release["type"],))


def release_records(releases_dir: Path) -> dict[str, Path]:
    """Release accession -> its saved record (<cik>/<accession>.json)."""
    return {path.stem: path for path in sorted(releases_dir.glob("*/*.json"))}
