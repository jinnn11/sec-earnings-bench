"""Link each saved earnings release to the 10-Q or 10-K that reports its period.

A saved release (see secbench.releases) counts only if one of its saved EX-99.x
exhibits (the press release, then the others, as secbench.inputs.build_input reads
them) has a table with a net income row and a per-share row, and it isn't in the
hand-kept EXCLUDED_RELEASES; others are excluded. Each remaining
release links to the first 10-Q or 10-K filed on or after it whose period ended 0 to
100 days before it, searched across every CIK of the ticker. If several releases link
to one filing, the one filed closest to it keeps the link; the others are conflicts.

Usage: uv run python -m secbench.link [--releases DIR] [--out data/links.csv] [--refresh]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields, replace
from datetime import date
from pathlib import Path

import pandas as pd

from secbench.edgar import EdgarClient, EdgarError, default_client
from secbench.releases import DEFAULT_OUT_DIR as DEFAULT_RELEASES_DIR
from secbench.releases import SUP_MARKUP, is_amount, write_text

log = logging.getLogger(__name__)

DEFAULT_OUT = Path("data/links.csv")
# Form -> period type of the figures it reports.
LINK_FORMS = {"10-Q": "quarter", "10-K": "full year"}
MAX_PERIOD_AGE = pd.Timedelta(days=100)  # release date minus period end, at most

LINKED = "linked"
CONFLICT = "conflict"
EXCLUDED = "excluded"
UNLINKED = "unlinked"
STATUSES = (LINKED, CONFLICT, EXCLUDED, UNLINKED)

# Earnings-release filter

# A bottom-line label: "Net income", "Consolidated net income attributable to
# Walmart", "Net (loss) income", "NET EARNINGS"; or one that starts with "Profit" or
# "Earnings" (Caterpillar's "Profit", ExxonMobil's "Earnings/(Loss) (U.S. GAAP)"), but
# not "Profit per share", "Earnings before income taxes", "Profit margin" or "Net
# income tax benefit" (Constellation Brands' outlook updates).
NET_INCOME = re.compile(
    r"\bnet\s+\(?(income|earnings|loss)\b(?!\s+tax(es)?\b)"
    r"|^(total\s+)?(profit|earnings)\b(?!.*\b(per|before|margin)\b)",
    re.IGNORECASE,
)
# "Earnings per share:", "Diluted net income per common share", "Profit per common
# share — diluted"; or the word EPS, in capitals ("Diluted EPS", Rockwell's "Diluted
# EPS from continuing operations", AEP's "EPS ($):").
PER_SHARE = re.compile(r"\bper\s+([\w-]+\s+){0,3}shares?\b", re.IGNORECASE)
EPS = re.compile(r"\bEPS\b")
CELL_SPLIT = re.compile(r"(?<!\\)\|")
# Item 2.02 8-Ks that pass the earnings check but aren't earnings releases, kept by
# hand: accession -> why. link_releases excludes them before linking, so they never
# claim a filing.
EXCLUDED_RELEASES = {
    "0000007084-24-000031": (
        "ADM's preliminary third-quarter 2024 results, released before it restated "
        "its earlier filings"
    ),
    "0001659166-25-000104": (
        "Fortive's announcement of the Ralliant spin-off; its EX-99.2 is pro forma "
        "statements"
    ),
    "0000773840-26-000084": (
        "Honeywell's announcement of the Aerospace spin-off; its EX-99.3 is pro forma "
        "statements"
    ),
}
LABELS_IN_REASON = 5
LABEL_WIDTH = 40


@dataclass(frozen=True)
class EarningsCheck:
    """Whether a press release looks like an earnings release, and if not, why."""

    ok: bool
    reason: str = ""
    exhibit: str = ""  # the exhibit type that passed (set by load_release)


def markdown_tables(markdown: str) -> list[list[list[str]]]:
    """The tables in markdown written by secbench.releases, each a list of rows of
    cells; the first row is the header and the separator row is dropped."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] | None = None
    for line in markdown.splitlines():
        if not line.startswith("|"):
            current = None
            continue
        cells = [c.strip().replace(r"\|", "|") for c in CELL_SPLIT.split(line)[1:-1]]
        if current is None:
            current = []
            tables.append(current)
        if current and all(c == "---" for c in cells):
            continue
        current.append(cells)
    return tables


def clean_label(text: str) -> str:
    """A cell's text without footnote marks and bold markers."""
    return " ".join(SUP_MARKUP.sub("", text).replace("**", "").split())


def row_labels(row: list[str]) -> list[str]:
    """The text cells of a row, in any column (ExxonMobil puts some labels between
    figures)."""
    return [label for c in row if c and not is_amount(c) if (label := clean_label(c))]


def is_net_income_row(row: list[str]) -> bool:
    return any(is_amount(c) for c in row) and any(
        NET_INCOME.search(label) for label in row_labels(row)
    )


def is_per_share_row(row: list[str]) -> bool:
    # A label alone counts: Apple's "Earnings per share:" heads its Basic and
    # Diluted rows.
    return any(
        PER_SHARE.search(label) or EPS.search(label) for label in row_labels(row)
    )


def check_earnings(markdown: str) -> EarningsCheck:
    """An earnings release has a table whose body (below its header row) has both a
    net income row with a figure and a per-share row."""
    tables = markdown_tables(markdown)
    for table in tables:
        body = table[1:]
        if any(map(is_net_income_row, body)) and any(map(is_per_share_row, body)):
            return EarningsCheck(True)
    if not tables:
        return EarningsCheck(False, "not an earnings release: no tables")
    labels = [labels[0] for t in tables for row in t[1:] if (labels := row_labels(row))]
    shown = ", ".join(f'"{shorten(label)}"' for label in labels[:LABELS_IN_REASON])
    return EarningsCheck(
        False,
        f"not an earnings release: no table has both a net income row and a "
        f"per-share row ({len(tables)} tables, {len(labels)} labeled rows; "
        f"first labels: {shown or 'none'})",
    )


def shorten(text: str, width: int = LABEL_WIDTH) -> str:
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


# Saved releases


@dataclass(frozen=True)
class Release:
    """A release saved by secbench.releases, with the filter's verdict."""

    ticker: str
    cik: int
    accession_number: str
    filing_date: date
    earnings: EarningsCheck


def load_release(record_path: Path) -> Release:
    """Read <accession>.json and check its exhibits' markdown: the chosen exhibit,
    then every other saved EX-99.x in filing-index order (the exhibits
    secbench.inputs.build_input reads; some companies put their statements in
    EX-99.2). The first that passes makes the release an earnings release; if none
    does, the reason is the chosen exhibit's."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    chosen = [e for e in record["exhibits"] if e["chosen"]]
    if len(chosen) != 1 or not chosen[0]["file"]:
        raise ValueError(f"{record_path}: no single chosen exhibit with a file")
    others = [e for e in record["exhibits"] if not e["chosen"] and e["file"]]
    failed = []
    for exhibit in [chosen[0], *others]:
        markdown = (record_path.parent / exhibit["file"]).read_text(encoding="utf-8")
        check = check_earnings(markdown)
        if check.ok:
            earnings = replace(check, exhibit=exhibit["type"])
            break
        failed.append(check)
    else:
        earnings = failed[0]
        if others:
            checked = ", ".join(e["type"] for e in others)
            earnings = replace(
                earnings, reason=f"{earnings.reason}; also checked: {checked}"
            )
    return Release(
        ticker=record["ticker"],
        cik=int(record["cik"]),
        accession_number=record["accession_number"],
        filing_date=date.fromisoformat(record["filing_date"]),
        earnings=earnings,
    )


def load_releases(releases_dir: Path) -> list[Release]:
    """Every release saved under releases_dir (<cik>/<accession>.json)."""
    return [load_release(path) for path in sorted(releases_dir.glob("*/*.json"))]


# Linking


def combine_filings(filings_by_cik: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """One filings table, with a cik column, from several CIKs in priority order.

    A filing listed under several CIKs (ExxonMobil's joint 10-Q for Q2 2026) is
    kept once, with the first of them.
    """
    frames = [filings.assign(cik=cik) for cik, filings in filings_by_cik.items()]
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates("accession_number").reset_index(drop=True)


def ticker_filings(
    client: EdgarClient, ticker: str, *, refresh: bool = False
) -> pd.DataFrame:
    """Filings under every CIK of ticker (edgar.TICKER_CIKS), current CIK first;
    refresh re-downloads the submissions JSON."""
    return combine_filings(
        {
            cik: client.get_filings(cik, refresh=refresh)
            for cik in client.ticker_to_ciks(ticker)
        }
    )


def find_filing(filings: pd.DataFrame, release_date: date) -> pd.Series | None:
    """The first 10-Q or 10-K filed on or after release_date whose period ended
    0 to 100 days before it, or None. Same-day filings go in accession order."""
    released = pd.Timestamp(release_date)
    candidates = filings[
        filings["form"].isin(list(LINK_FORMS))
        & (filings["filing_date"] >= released)
        & (filings["report_date"] <= released)
        & (filings["report_date"] >= released - MAX_PERIOD_AGE)
    ]
    return None if candidates.empty else first_filed(candidates)


def first_filed(filings: pd.DataFrame) -> pd.Series:
    """The earliest-filed row; same-day filings go in accession order."""
    return filings.sort_values(["filing_date", "accession_number"], kind="stable").iloc[
        0
    ]


def unlinked_reason(filings: pd.DataFrame, release_date: date) -> str:
    later = filings[
        filings["form"].isin(list(LINK_FORMS))
        & (filings["filing_date"] >= pd.Timestamp(release_date))
    ]
    if later.empty:
        return "no 10-Q or 10-K filed on or after the release yet"
    first = first_filed(later)
    period = "no period" if pd.isna(first.report_date) else first.report_date.date()
    return (
        "no 10-Q or 10-K filed on or after the release has a period ending 0-100 "
        f"days before it (first after: {first.form} {first.accession_number}, "
        f"filed {first.filing_date.date()}, period {period})"
    )


@dataclass
class Link:
    """One row of links.csv."""

    ticker: str
    release_cik: int
    release_accession: str
    release_date: date
    linked_form: str = ""
    linked_cik: int | None = None
    linked_accession: str = ""
    period_end: date | None = None
    period_type: str = ""
    status: str = ""
    reason: str = ""


COLUMNS = [f.name for f in fields(Link)]


def link_releases(releases: list[Release], filings: pd.DataFrame) -> list[Link]:
    """Link one ticker's releases to its filings (see combine_filings), oldest
    release first."""
    links = []
    claims: dict[str, list[tuple[Link, pd.Timestamp]]] = defaultdict(list)
    for release in sorted(releases, key=lambda r: (r.filing_date, r.accession_number)):
        link = Link(
            release.ticker, release.cik, release.accession_number, release.filing_date
        )
        links.append(link)
        listed = EXCLUDED_RELEASES.get(release.accession_number)
        if listed is not None:
            link.status = EXCLUDED
            link.reason = f"not an earnings release (link.EXCLUDED_RELEASES): {listed}"
            continue
        if not release.earnings.ok:
            link.status, link.reason = EXCLUDED, release.earnings.reason
            continue
        filing = find_filing(filings, release.filing_date)
        if filing is None:
            link.status = UNLINKED
            link.reason = unlinked_reason(filings, release.filing_date)
            continue
        link.linked_form = filing.form
        link.linked_cik = int(filing.cik)
        link.linked_accession = filing.accession_number
        link.period_end = filing.report_date.date()
        link.period_type = LINK_FORMS[filing.form]
        link.status = LINKED
        claims[filing.accession_number].append((link, filing.filing_date))
    for claimants in claims.values():
        if len(claimants) > 1:
            mark_conflicts(claimants)
    return links


def mark_conflicts(claimants: list[tuple[Link, pd.Timestamp]]) -> None:
    """Of releases linked to one filing, keep the one filed closest to it; mark the
    others as conflicts. If several are equally close, all are conflicts."""
    links = [link for link, _ in claimants]
    gaps = [(filed - pd.Timestamp(link.release_date)).days for link, filed in claimants]
    closest = [link for link, gap in zip(links, gaps, strict=True) if gap == min(gaps)]
    if len(closest) == 1:
        kept = closest[0]
        why = f"{kept.release_accession} was filed closer to it"
    else:
        kept = None
        names = " and ".join(link.release_accession for link in closest)
        why = f"{names} were filed equally close to it, so none is kept"
    for link in links:
        if link is kept:
            continue
        others = ", ".join(o.release_accession for o in links if o is not link)
        link.status = CONFLICT
        link.reason = (
            f"{link.linked_form} {link.linked_accession} is also linked to {others}; "
            f"{why}"
        )


# Output


def links_csv(links: list[Link]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for link in links:
        row = asdict(link)
        writer.writerow(
            {
                key: value.isoformat()
                if isinstance(value, date)
                else ("" if value is None else value)
                for key, value in row.items()
            }
        )
    return buffer.getvalue()


def format_counts(links: list[Link]) -> str:
    """Counts by status per ticker, a total row, and each release not linked."""
    counts = Counter((link.ticker, link.status) for link in links)
    tickers = sorted({link.ticker for link in links})

    def line(name: str, values: list[int]) -> str:
        return f"{name:<8}" + "".join(f"{v:>10}" for v in values)

    lines = [f"{'ticker':<8}" + "".join(f"{s:>10}" for s in (*STATUSES, "total"))]
    for ticker in tickers:
        values = [counts[ticker, status] for status in STATUSES]
        lines.append(line(ticker, [*values, sum(values)]))
    totals = [sum(counts[t, status] for t in tickers) for status in STATUSES]
    lines.append(line("total", [*totals, sum(totals)]))
    for link in links:
        if link.status != LINKED:
            lines.append(
                f"  {link.ticker} {link.release_accession} ({link.release_date}) "
                f"{link.status}: {link.reason}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None, client: EdgarClient | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.link",
        description="Link saved earnings releases to the 10-Q or 10-K for their period.",
    )
    parser.add_argument("--releases", type=Path, default=DEFAULT_RELEASES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download the submissions JSON (to see 10-Qs filed since the cache)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    releases = load_releases(args.releases)
    if not releases:
        print(f"error: no saved releases in {args.releases}", file=sys.stderr)
        return 1
    by_ticker: dict[str, list[Release]] = defaultdict(list)
    for release in releases:
        by_ticker[release.ticker].append(release)

    status = 0
    links: list[Link] = []
    for ticker in sorted(by_ticker):
        try:
            client = client or default_client()
            filings = ticker_filings(client, ticker, refresh=args.refresh)
        except EdgarError as exc:
            print(
                f"error: {ticker}: {exc}; its releases are not linked", file=sys.stderr
            )
            status = 1
            continue
        links.extend(link_releases(by_ticker[ticker], filings))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text(args.out, links_csv(links))
    print(format_counts(links))
    print(f"wrote {len(links)} rows to {args.out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
