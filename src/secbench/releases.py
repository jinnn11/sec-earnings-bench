"""Earnings press releases: Item 2.02 8-Ks and their EX-99 exhibits, as markdown.

For each 8-K (not 8-K/A) whose items include 2.02, read the filing index, choose the
press release (EX-99.1; else EX-99.01-style or plain EX-99) and save every EX-99.x
exhibit as markdown under data/releases/<cik>/, with a JSON record per filing.

Usage: uv run python -m secbench.releases AAPL [MSFT ...] --since 2025-04-01
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup, Tag
from bs4.element import PreformattedString
from markdownify import MarkdownConverter

from secbench.edgar import (
    EdgarClient,
    EdgarError,
    default_client,
    format_cik,
    normalize_ticker,
)

log = logging.getLogger(__name__)

ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/"
DEFAULT_OUT_DIR = Path("data/releases")
INDEX_COLUMNS = ["Seq", "Description", "Document", "Type", "Size"]
HTML_SUFFIXES = (".htm", ".html")


# Selecting filings


def has_item(items: str, item: str) -> bool:
    """Whether SEC's comma-separated items string ("2.02,9.01") lists item."""
    return item in {part.strip() for part in items.split(",")}


def select_releases(
    filings: pd.DataFrame, since: date, until: date | None = None
) -> pd.DataFrame:
    """8-Ks (not 8-K/A) listing Item 2.02, filed from since to until inclusive."""
    keep = (filings["form"] == "8-K") & filings["items"].map(
        lambda items: has_item(items, "2.02")
    )
    keep &= filings["filing_date"] >= pd.Timestamp(since)
    if until is not None:
        keep &= filings["filing_date"] <= pd.Timestamp(until)
    return filings[keep].reset_index(drop=True)


# Filing index


@dataclass(frozen=True)
class Document:
    """One row of a filing's "Document Format Files" table."""

    seq: str
    description: str
    filename: str
    type: str
    url: str

    @property
    def is_html(self) -> bool:
        return self.filename.lower().endswith(HTML_SUFFIXES)


def filing_folder_url(cik: int | str, accession: str) -> str:
    return f"{ARCHIVES_URL}{int(cik)}/{accession.replace('-', '')}/"


def filing_index_url(cik: int | str, accession: str) -> str:
    return f"{filing_folder_url(cik, accession)}{accession}-index.htm"


def _text(tag: Tag) -> str:
    return " ".join(tag.get_text(" ").split())


def in_filing(url: str, folder_url: str) -> bool:
    """Whether url is a document of the filing whose folder is folder_url: anything
    in that folder, or a file directly in the same accession's folder under another
    CIK. A filing made jointly with another registrant (a utility and its
    subsidiaries) may list its documents under a co-registrant's CIK."""
    if url.startswith(folder_url):
        return True
    accession = folder_url.rstrip("/").rsplit("/", 1)[-1]
    pattern = rf"{re.escape(ARCHIVES_URL)}\d+/{re.escape(accession)}/[^/?#]+"
    return re.fullmatch(pattern, url) is not None


def parse_filing_index(html: bytes, folder_url: str) -> list[Document]:
    """Documents listed in a filing's -index.htm page.

    Links to the inline XBRL viewer ("/ix?doc=/Archives/...") are unwrapped. A link
    outside the filing (see in_filing) raises EdgarError.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", summary="Document Format Files")
    if table is None:
        raise EdgarError("filing index has no 'Document Format Files' table")
    rows = table.find_all("tr")
    columns = [_text(th) for th in rows[0].find_all("th")] if rows else []
    if columns != INDEX_COLUMNS:
        raise EdgarError(f"filing index has unexpected columns {columns}")
    documents = []
    for row in rows[1:]:
        cells = row.find_all("td")
        link = cells[2].find("a") if len(cells) == len(INDEX_COLUMNS) else None
        if link is None:
            raise EdgarError(f"filing index row without a document link: {_text(row)}")
        href = link["href"].removeprefix("/ix?doc=")
        url = urljoin("https://www.sec.gov/", href)
        if not in_filing(url, folder_url):
            raise EdgarError(f"filing index links outside the filing: {url}")
        documents.append(
            Document(
                seq=_text(cells[0]),
                description=_text(cells[1]),
                filename=url.rsplit("/", 1)[-1],
                type=_text(cells[3]),
                url=url,
            )
        )
    return documents


# Choosing exhibits

EX99 = re.compile(r"EX-99(\..+)?")


def ex99_exhibits(documents: list[Document]) -> list[Document]:
    """EX-99, EX-99.1, EX-99.01, EX-99.2, ... in index order."""
    return [doc for doc in documents if EX99.fullmatch(doc.type)]


def release_rank(doc_type: str) -> int | None:
    """Preference for the press release: EX-99.1, then EX-99.01 and the like, then
    plain EX-99. None for any other type."""
    if doc_type == "EX-99.1":
        return 0
    if re.fullmatch(r"EX-99\.0+1", doc_type):
        return 1
    if doc_type == "EX-99":
        return 2
    return None


def choose_release(documents: list[Document]) -> Document | None:
    ranked = [
        (rank, i)
        for i, doc in enumerate(documents)
        if (rank := release_rank(doc.type)) is not None
    ]
    return documents[min(ranked)[1]] if ranked else None


def exhibit_file_names(
    accession: str, exhibits: list[Document], release: Document
) -> list[str]:
    """File names for each exhibit: <accession>.md for the press release and
    <accession>_<type>.md for the others (plus _<seq> if a type repeats)."""
    counts = Counter(doc.type for doc in exhibits)
    names = []
    for doc in exhibits:
        if doc == release:
            names.append(f"{accession}.md")
            continue
        stem = re.sub(r"[^A-Za-z0-9.-]", "-", doc.type)
        if counts[doc.type] > 1:
            stem += f"_{doc.seq}"
        names.append(f"{accession}_{stem}.md")
    return names


# HTML to markdown

BLOCK_TAGS = frozenset(
    {"br", "div", "p", "li", "tr", "td", "th", "table", "h1", "h2", "h3", "h4", "h5"}
)
HIDDEN = re.compile(r"display\s*:\s*none", re.IGNORECASE)
# Superscripts set by CSS: "vertical-align:super", or Workiva's raised text
# ("position:relative;top:-3.15pt").
SUPERSCRIPT = re.compile(
    r"vertical-align\s*:\s*super"
    r"|^(?=.*position\s*:\s*relative)(?=.*(?<![\w-])top\s*:\s*-)",
    re.IGNORECASE,
)
SUP_MARKUP = re.compile(r"<sup>.*?</sup>")
# Cells that hold only part of a number; they join the cell beside them.
PREFIXES = frozenset({"$", "(", "$(", "€", "£", "¥"})
SUFFIXES = frozenset({")", "%", ")%", "%)"})
AMOUNT = re.compile(r"[$€£¥(+\-–—]*\s*\d[\d,]*(\.\d+)?\s*[)%]*")
YEAR = re.compile(r"(19|20)\d\d")
DASHES = re.compile(r"[—–-]+")


def strip_sgml(html: bytes) -> bytes:
    """Drop the <DOCUMENT>...<TEXT> wrapper EDGAR leaves around some documents."""
    if not html.lstrip()[:10].upper().startswith(b"<DOCUMENT>"):
        return html
    match = re.search(rb"<TEXT>(.*)</TEXT>", html, re.DOTALL)
    return match.group(1) if match else html


def mark_superscripts(soup: BeautifulSoup) -> None:
    """Turn CSS superscripts into <sup> tags, so a footnote mark is not read as
    another digit ("$2.11" with a raised "1" is not "$2.111")."""
    for tag in soup.find_all(style=SUPERSCRIPT):
        tag.name = "sup"


def _inline_text(node: Tag) -> str:
    parts = []
    for child in node.children:
        if isinstance(child, Tag):
            inner = _inline_text(child)
            if child.name == "sup":
                inner = f"<sup>{inner.strip()}</sup>" if inner.strip() else ""
            elif child.name in BLOCK_TAGS:
                inner = f" {inner} "
            parts.append(inner)
        elif not isinstance(child, PreformattedString):  # skip comments and the like
            parts.append(str(child))
    return "".join(parts)


def cell_text(cell: Tag) -> str:
    """Visible text of a table cell; <br> and block boundaries become spaces,
    whitespace (including nbsp) collapses, superscripts are written <sup>1</sup>,
    and everything else is kept as is."""
    return " ".join(_inline_text(cell).split())


@dataclass
class Cell:
    text: str
    start: int  # first grid column
    span: int


def is_amount(text: str) -> bool:
    """A figure such as $1,234, (321), 1.85, 12% or a dash, ignoring footnote
    marks; a bare year is not."""
    text = SUP_MARKUP.sub("", text).strip()
    if DASHES.fullmatch(text):
        return True
    return bool(AMOUNT.fullmatch(text)) and not YEAR.fullmatch(text)


def glue_fragments(row: list[Cell]) -> None:
    """Join "$" and "(" to the cell after, and ")" and "%" to the cell before."""
    for left, right in itertools.pairwise(row):
        if left.text in PREFIXES and right.text:
            right.text = left.text + right.text
            left.text = ""
    for left, right in itertools.pairwise(row):
        if right.text in SUFFIXES and left.text:
            left.text += right.text
            right.text = ""


def layout_table(
    rows: list[list[tuple[str, int, int]]],
) -> tuple[list[list[str]], list[list[str]]]:
    """Turn rows of (text, colspan, rowspan) into a rectangular grid of strings.

    A cell spanning rows takes its columns in the rows below, where it counts as
    empty. Number fragments are joined, empty rows dropped, and only columns where
    some one-column cell has text are kept. Returns the grid twice: with a cell
    that spans several kept columns written once, in the first of them, and with
    it repeated in each (for headers). A cell covering the first column, such as
    a row label or title, is never repeated.
    """
    taken: dict[int, set[int]] = {}  # row -> columns held by cells from above
    grid_rows = []
    for r, row in enumerate(rows):
        cells, start = [], 0
        for text, colspan, rowspan in row:
            while start in taken.get(r, ()):
                start += 1
            cells.append(Cell(text, start, colspan))
            for below in range(r + 1, r + rowspan):
                taken.setdefault(below, set()).update(range(start, start + colspan))
            start += colspan
        glue_fragments(cells)
        if any(cell.text for cell in cells):
            grid_rows.append(cells)

    filled = [cell for cells in grid_rows for cell in cells if cell.text]
    keep = {cell.start for cell in filled if cell.span == 1}
    for cell in sorted(filled, key=lambda c: c.span):
        if not any(cell.start <= k < cell.start + cell.span for k in keep):
            keep.add(cell.start)
    columns = sorted(keep)

    once, repeated = [], []
    for cells in grid_rows:
        row_once, row_repeated = [""] * len(columns), [""] * len(columns)
        for cell in cells:
            if not cell.text:
                continue
            covered = [
                j
                for j, k in enumerate(columns)
                if cell.start <= k < cell.start + cell.span
            ]
            row_once[covered[0]] = cell.text
            for j in covered[:1] if covered[0] == 0 else covered:
                row_repeated[j] = cell.text
        once.append(row_once)
        repeated.append(row_repeated)
    return once, repeated


def split_header(
    once: list[list[str]], repeated: list[list[str]]
) -> tuple[list[str], list[list[str]]]:
    """Merge the stacked header rows above a table's figures into one row.

    Header rows are the unbroken run of rows with text outside the first column
    that ends just above the first row with figures (rows with text only in the
    first column, such as "Net sales:", may sit between). A title row with text
    only in the first column ends the run, so page titles above it stay in the
    body. A table without figures gets a blank header. The header repeats
    spanning cells; the body writes them once.
    """
    width = len(once[0]) if once else 0
    first_figures = next(
        (i for i, row in enumerate(once) if any(is_amount(v) for v in row[1:])), 0
    )
    header_rows: set[int] = set()
    for i in reversed(range(first_figures)):
        if any(once[i][1:]):
            header_rows.add(i)
        elif header_rows:
            break
    header = [
        " ".join(repeated[i][j] for i in sorted(header_rows) if repeated[i][j])
        for j in range(width)
    ]
    body = [row for i, row in enumerate(once) if i not in header_rows]
    return header, body


def render_table(header: list[str], body: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", r"\|") for c in cells) + " |"

    separator = "| " + " | ".join("---" for _ in header) + " |"
    return "\n".join([line(header), separator, *(line(row) for row in body)])


def _span(cell: Tag, attribute: str) -> int:
    value = str(cell.get(attribute, "1")).strip()
    return int(value) if value.isdigit() and value != "0" else 1


def table_rows(table: Tag) -> list[list[tuple[str, int, int]]]:
    """(text, colspan, rowspan) for each cell of each row of table (not nested
    tables)."""
    rows = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        rows.append(
            [
                (cell_text(td), _span(td, "colspan"), _span(td, "rowspan"))
                for td in tr.find_all(["td", "th"], recursive=False)
            ]
        )
    return rows


def table_to_markdown(table: Tag) -> str:
    once, repeated = layout_table(table_rows(table))
    if not once:
        return ""
    if len(once[0]) == 1:  # a layout table: one paragraph per row
        return "\n\n".join(row[0] for row in once)
    return render_table(*split_header(once, repeated))


class ReleaseConverter(MarkdownConverter):
    def convert_table(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return "\n\n" + table_to_markdown(el) + "\n\n"


def html_to_markdown(html: bytes) -> str:
    """Markdown for an exhibit; text is kept as written, tables as tables."""
    soup = BeautifulSoup(strip_sgml(html), "lxml")
    hidden = soup.find_all(["script", "style", "head", "img", "ix:header"])
    hidden += soup.find_all(style=HIDDEN)
    for tag in hidden:
        if not tag.decomposed:
            tag.decompose()
    mark_superscripts(soup)
    converter = ReleaseConverter(
        escape_asterisks=False,
        escape_underscores=False,
        escape_misc=False,
        sup_symbol="<sup>",
    )
    markdown = converter.convert_soup(soup.body or soup).replace("\xa0", " ")
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip() + "\n"


# Saving


@dataclass
class Outcome:
    accession_number: str
    filing_date: date
    skip_reason: str | None = None
    release_type: str | None = None
    other_types: list[str] = field(default_factory=list)
    release_has_table: bool = False


def write_text(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".part")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def save_filing(
    client: EdgarClient,
    ticker: str,
    cik: int,
    accession: str,
    filing_date: date,
    out_dir: Path,
) -> Outcome:
    """Save one filing's EX-99 exhibits as markdown plus a JSON record."""
    outcome = Outcome(accession, filing_date)
    try:
        documents = parse_filing_index(
            client.fetch(filing_index_url(cik, accession)),
            filing_folder_url(cik, accession),
        )
    except EdgarError as exc:
        outcome.skip_reason = f"filing index: {exc}"
        return outcome
    release = choose_release(documents)
    if release is None:
        exhibit_types = [doc.type for doc in documents if doc.type.startswith("EX-")]
        listed = ", ".join(exhibit_types) or "none"
        outcome.skip_reason = f"no EX-99.1 exhibit (exhibits: {listed})"
        return outcome
    if not release.is_html:
        outcome.skip_reason = f"{release.type} is {release.filename}, not HTML"
        return outcome

    exhibits = ex99_exhibits(documents)
    markdown: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for doc, name in zip(
        exhibits, exhibit_file_names(accession, exhibits, release), strict=True
    ):
        if doc.is_html:
            try:
                markdown[name] = html_to_markdown(client.fetch(doc.url))
            except EdgarError as exc:
                outcome.skip_reason = f"{doc.type}: {exc}"
                return outcome
        else:
            log.warning("%s %s: %s is not HTML; not saved", ticker, accession, doc.url)
        records.append(
            {
                "type": doc.type,
                "description": doc.description,
                "url": doc.url,
                "file": name if doc.is_html else None,
                "chosen": doc == release,
            }
        )

    folder = out_dir / format_cik(cik)
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in markdown.items():
        write_text(folder / name, text)
    record = {
        "ticker": ticker,
        "cik": cik,
        "filing_date": filing_date.isoformat(),
        "accession_number": accession,
        "exhibit": release.type,
        "exhibit_url": release.url,
        "exhibits": records,
    }
    write_text(folder / f"{accession}.json", json.dumps(record, indent=2) + "\n")
    outcome.release_type = release.type
    outcome.other_types = [doc.type for doc in exhibits if doc != release]
    outcome.release_has_table = "\n| --- |" in markdown[f"{accession}.md"]
    return outcome


def collect_releases(
    client: EdgarClient,
    ticker: str,
    since: date,
    until: date | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> tuple[list[int], list[Outcome]]:
    """Save every Item 2.02 release for ticker in the date range, from each of its
    CIKs (see edgar.TICKER_CIKS). A filing listed under several CIKs is saved once,
    under the first. Returns the CIKs and one outcome per filing, oldest first."""
    ticker = normalize_ticker(ticker)
    ciks = client.ticker_to_ciks(ticker)
    filings = pd.concat(
        [
            select_releases(client.get_filings(cik), since, until).assign(cik=cik)
            for cik in ciks
        ],
        ignore_index=True,
    ).drop_duplicates("accession_number")
    outcomes = []
    for filing in filings.sort_values("filing_date", kind="stable").itertuples():
        outcome = save_filing(
            client,
            ticker,
            int(filing.cik),
            filing.accession_number,
            filing.filing_date.date(),
            out_dir,
        )
        if outcome.skip_reason:
            log.warning(
                "skip %s %s (%s): %s",
                ticker,
                outcome.accession_number,
                outcome.filing_date,
                outcome.skip_reason,
            )
        outcomes.append(outcome)
    return ciks, outcomes


def format_report(ticker: str, ciks: list[int], outcomes: list[Outcome]) -> str:
    saved = [o for o in outcomes if not o.skip_reason]
    skipped = [o for o in outcomes if o.skip_reason]
    cik_list = ", ".join(format_cik(cik) for cik in ciks)
    lines = [
        (
            f"{normalize_ticker(ticker)} (CIK {cik_list}): {len(outcomes)} "
            f"Item 2.02 8-Ks, {len(saved)} saved, {len(skipped)} skipped"
        )
    ]
    if saved:
        releases = Counter(o.release_type for o in saved)
        others = Counter(t for o in saved for t in o.other_types)
        lines.append(f"  press release: {_counts(releases)}")
        lines.append(f"  other EX-99:   {_counts(others) or 'none'}")
    for o in saved:
        if not o.release_has_table:
            lines.append(f"  no table in {o.accession_number} ({o.filing_date})")
    for o in skipped:
        lines.append(
            f"  skipped {o.accession_number} ({o.filing_date}): {o.skip_reason}"
        )
    return "\n".join(lines)


def _counts(counter: Counter[str]) -> str:
    return ", ".join(f"{name} x{n}" for name, n in sorted(counter.items()))


def main(argv: list[str] | None = None, client: EdgarClient | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.releases",
        description="Save earnings press releases (Item 2.02 8-K exhibits) as markdown.",
    )
    parser.add_argument("tickers", nargs="+", metavar="ticker")
    parser.add_argument("--since", type=date.fromisoformat, required=True)
    parser.add_argument("--until", type=date.fromisoformat)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    status = 0
    for ticker in args.tickers:
        try:
            client = client or default_client()
            ciks, outcomes = collect_releases(
                client, ticker, args.since, args.until, args.out
            )
        except EdgarError as exc:
            print(f"error: {ticker}: {exc}", file=sys.stderr)
            status = 1
            continue
        print(format_report(ticker, ciks, outcomes))
    return status


if __name__ == "__main__":
    sys.exit(main())
