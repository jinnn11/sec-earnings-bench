"""The stage 2 company list and split (PLAN.md Decisions 2026-10-27).

The universe is the S&P 500 minus GICS Financials, from Wikipedia's "List of S&P 500
companies" at one revision. Its permalink (index.php?title=...&oldid=N) is downloaded
once and saved under data/raw/en.wikipedia.org/, beside a JSON note of the URL, the
revision's id and timestamp and when it was fetched; with no --revision, the latest
revision is looked up once through the MediaWiki API (also cached). Requests send a
descriptive User-Agent with the contact in SEC_USER_AGENT, as Wikimedia's policy asks.

Share classes (GOOGL and GOOG) are merged by CIK, keeping the snapshot's first
ticker. Each CIK is checked against SEC's ticker map (the cached
company_tickers.json); a mismatch (a company that moved to a new CIK, as ExxonMobil
did) is reported, not fixed.

Split, with seed SEED: validation = the 10 stage 1 companies; company_test =
COMPANY_TEST_SIZE companies drawn at random from the rest, sorted by CIK first; train
= everyone else. Written to data/companies.csv (ticker, cik, name, sector, split) and
data/companies.json (the snapshot, seed, counts, merged share classes and CIK
mismatches). An existing companies.csv is never changed: the split is fixed before any
SEC download for the new companies.

Usage: uv run python -m secbench.companies [--revision OLDID] [--out data/companies.csv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from bs4 import BeautifulSoup

from secbench.edgar import (
    DEFAULT_CACHE_DIR,
    TICKER_CIKS,
    TICKERS_URL,
    EdgarClient,
    EdgarError,
    default_client,
    load_user_agent,
    normalize_ticker,
    parse_company_tickers,
)
from secbench.releases import write_text

WIKI_HOST = "en.wikipedia.org"
PAGE_TITLE = "List of S&P 500 companies"
API_URL = (
    f"https://{WIKI_HOST}/w/api.php?action=query&format=json&formatversion=2"
    f"&prop=revisions&rvprop=ids%7Ctimestamp&titles={quote(PAGE_TITLE.replace(' ', '_'))}"
)
TABLE_ID = "constituents"
COLUMNS = {
    "Symbol": "ticker",
    "Security": "name",
    "GICS Sector": "sector",
    "CIK": "cik",
}
FINANCIALS = "Financials"
CLASS_SUFFIX = re.compile(r"\s*\(Class [A-Z]\)$")
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

DEFAULT_OUT = Path("data/companies.csv")
SEED = 20261027  # the date of the decision (PLAN.md Decisions 2026-10-27)
VALIDATION = "validation"
COMPANY_TEST = "company_test"
TRAIN = "train"
SPLITS = (VALIDATION, COMPANY_TEST, TRAIN)
# The stage 1 companies, whose 60 releases are the validation set.
VALIDATION_TICKERS = (
    "AAPL",
    "CAT",
    "HD",
    "MSFT",
    "NFLX",
    "NKE",
    "NVDA",
    "ORCL",
    "WMT",
    "XOM",
)
COMPANY_TEST_SIZE = 40
CSV_COLUMNS = ["ticker", "cik", "name", "sector", "split"]


class CompaniesError(Exception):
    """The company list can't be built or written; the message says why."""


# The snapshot


def permalink(revision: int) -> str:
    title = quote(PAGE_TITLE.replace(" ", "_"))
    return f"https://{WIKI_HOST}/w/index.php?title={title}&oldid={revision}"


def wiki_cache_path(url: str, cache_dir: Path) -> Path:
    """<cache_dir>/en.wikipedia.org/<path>, with a hash of the query string, as
    edgar.cache_path lays out SEC URLs."""
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != WIKI_HOST:
        raise ValueError(f"not an {WIKI_HOST} URL: {url}")
    segments = [s for s in parts.path.split("/") if s]
    if not segments or any(s in (".", "..") for s in segments):
        raise ValueError(f"bad URL path: {url}")
    if parts.query:
        digest = hashlib.sha256(parts.query.encode()).hexdigest()[:12]
        segments[-1] = f"{segments[-1]}__{digest}"
    return cache_dir.joinpath(WIKI_HOST, *segments)


def user_agent() -> str:
    """sec-earnings-bench with the contact in SEC_USER_AGENT (never printed)."""
    return f"sec-earnings-bench/0.1 ({load_user_agent()}) httpx/{httpx.__version__}"


class WikiClient:
    """Fetches en.wikipedia.org URLs once each, caching every body under
    cache_dir; a cached URL is never fetched again."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        *,
        transport: httpx.BaseTransport | None = None,
        agent: str | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self._transport = transport
        self._agent = agent

    def fetch(self, url: str) -> bytes:
        path = wiki_cache_path(url, self.cache_dir)
        if path.is_file():
            return path.read_bytes()
        headers = {"User-Agent": self._agent or user_agent()}
        with httpx.Client(
            headers=headers, timeout=TIMEOUT, transport=self._transport
        ) as client:
            response = client.get(url)
        if response.status_code != 200:
            raise CompaniesError(f"HTTP {response.status_code} for {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return response.content


@dataclass(frozen=True)
class Snapshot:
    url: str  # the permalink
    revision_id: int
    revision_timestamp: str
    fetched_at: str
    html: bytes


def latest_revision(wiki: WikiClient) -> tuple[int, str]:
    """The page's latest revision id and timestamp (cached after the first call)."""
    data = json.loads(wiki.fetch(API_URL))
    [page] = data["query"]["pages"]
    [revision] = page["revisions"]
    return int(revision["revid"]), str(revision["timestamp"])


def load_snapshot(wiki: WikiClient, revision: int | None = None) -> Snapshot:
    """The page at revision (default: the latest), saved with a note beside it."""
    if revision is None:
        revision, timestamp = latest_revision(wiki)
    else:
        timestamp = ""
    url = permalink(revision)
    cached = wiki_cache_path(url, wiki.cache_dir)
    note_path = cached.with_name(cached.name + ".source.json")
    if note_path.is_file():
        note = json.loads(note_path.read_text(encoding="utf-8"))
        return Snapshot(
            url,
            revision,
            note["revision_timestamp"],
            note["fetched_at"],
            wiki.fetch(url),
        )
    html = wiki.fetch(url)
    note = {
        "url": url,
        "revision_id": revision,
        "revision_timestamp": timestamp,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_text(note_path, json.dumps(note, indent=2) + "\n")
    return Snapshot(url, revision, timestamp, note["fetched_at"], html)


# Parsing


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: int
    name: str
    sector: str
    split: str = ""


def parse_constituents(html: bytes) -> list[Company]:
    """The constituents table, one row per listing (share classes not merged)."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=TABLE_ID)
    if table is None:
        raise CompaniesError(f"no table with id {TABLE_ID!r} in the snapshot")
    rows = table.find_all("tr")
    header = [cell.get_text(" ", strip=True) for cell in rows[0].find_all(["th", "td"])]
    missing = [name for name in COLUMNS if name not in header]
    if missing:
        raise CompaniesError(f"the constituents table has no {', '.join(missing)}")
    index = {name: header.index(name) for name in COLUMNS}
    companies = []
    for row in rows[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if len(cells) < len(header):
            continue
        cik = cells[index["CIK"]]
        if not cik.isdigit():
            raise CompaniesError(f"bad CIK {cik!r} for {cells[index['Symbol']]}")
        companies.append(
            Company(
                ticker=normalize_ticker(cells[index["Symbol"]]),
                cik=int(cik),
                name=CLASS_SUFFIX.sub("", cells[index["Security"]]),
                sector=cells[index["GICS Sector"]],
            )
        )
    return companies


def merge_share_classes(
    companies: Iterable[Company],
) -> tuple[list[Company], list[list[str]]]:
    """One company per CIK, the first listing's ticker kept; and the tickers of each
    CIK listed more than once."""
    by_cik: dict[int, list[Company]] = {}
    for company in companies:
        by_cik.setdefault(company.cik, []).append(company)
    merged = [listings[0] for listings in by_cik.values()]
    classes = [[c.ticker for c in ls] for ls in by_cik.values() if len(ls) > 1]
    return merged, classes


@dataclass(frozen=True)
class Mismatch:
    ticker: str
    cik: int  # the snapshot's
    sec_cik: int | None  # SEC's ticker map's; None if the ticker isn't in it


def cik_mismatches(
    companies: Iterable[Company], sec_ciks: dict[str, int]
) -> list[Mismatch]:
    """Companies whose snapshot CIK isn't SEC's for the ticker. A ticker in
    edgar.TICKER_CIKS matches if both CIKs are among its listed ones."""
    mismatches = []
    for company in companies:
        sec_cik = sec_ciks.get(company.ticker)
        known = TICKER_CIKS.get(company.ticker, [])
        if sec_cik == company.cik or (sec_cik in known and company.cik in known):
            continue
        mismatches.append(Mismatch(company.ticker, company.cik, sec_cik))
    return mismatches


# The split


def assign_splits(
    companies: Sequence[Company],
    seed: int = SEED,
    validation: Sequence[str] = VALIDATION_TICKERS,
    test_size: int = COMPANY_TEST_SIZE,
) -> list[Company]:
    """Every company with its split, sorted by CIK: the validation tickers, then
    test_size of the others drawn with random.Random(seed) from their CIK order,
    then train."""
    tickers = {c.ticker for c in companies}
    absent = [t for t in validation if t not in tickers]
    if absent:
        raise CompaniesError(
            f"validation companies not in the list: {', '.join(absent)}"
        )
    if len({c.cik for c in companies}) != len(companies):
        raise CompaniesError("several companies share a CIK; merge share classes first")
    ordered = sorted(companies, key=lambda c: c.cik)
    others = [c.cik for c in ordered if c.ticker not in validation]
    if len(others) < test_size:
        raise CompaniesError(f"only {len(others)} companies outside validation")
    test = set(random.Random(seed).sample(others, test_size))

    def split(company: Company) -> str:
        if company.ticker in validation:
            return VALIDATION
        return COMPANY_TEST if company.cik in test else TRAIN

    return [Company(**{**asdict(c), "split": split(c)}) for c in ordered]


# Output


def companies_csv(companies: Iterable[Company]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for company in companies:
        writer.writerow(asdict(company))
    return buffer.getvalue()


def load_companies(path: Path = DEFAULT_OUT) -> list[Company]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise CompaniesError(
                f"{path} has columns {reader.fieldnames}, not {CSV_COLUMNS}"
            )
        return [Company(**{**row, "cik": int(row["cik"])}) for row in reader]


def write_companies(path: Path, companies: list[Company], note: dict[str, Any]) -> bool:
    """Write companies.csv and its note (companies.json beside it). An existing
    companies.csv must be identical: returns False if it already was, and raises if
    it differs, since the split must never change once written."""
    text = companies_csv(companies)
    if path.is_file():
        if path.read_text(encoding="utf-8") == text:
            return False
        raise CompaniesError(
            f"{path} exists with a different list or split; the split is fixed once "
            "written (delete the file only to change it deliberately)"
        )
    write_text(path, text)
    write_text(path.with_suffix(".json"), json.dumps(note, indent=2) + "\n")
    return True


def build(
    snapshot: Snapshot,
    sec_ciks: dict[str, int],
    seed: int = SEED,
    test_size: int = COMPANY_TEST_SIZE,
) -> tuple[list[Company], dict[str, Any]]:
    """The split company list and its note."""
    listings = parse_constituents(snapshot.html)
    kept = [c for c in listings if c.sector != FINANCIALS]
    merged, classes = merge_share_classes(kept)
    companies = assign_splits(merged, seed, test_size=test_size)
    mismatches = cik_mismatches(companies, sec_ciks)
    note = {
        "source_url": snapshot.url,
        "revision_id": snapshot.revision_id,
        "revision_timestamp": snapshot.revision_timestamp,
        "fetched_at": snapshot.fetched_at,
        "listings": len(listings),
        "financials_dropped": len(listings) - len(kept),
        "share_classes_merged": classes,
        "companies": len(companies),
        "seed": seed,
        "splits": {s: sum(c.split == s for c in companies) for s in SPLITS},
        "cik_mismatches": [asdict(m) for m in mismatches],
    }
    return companies, note


def format_report(note: dict[str, Any], out: Path, written: bool) -> str:
    groups = "; ".join("/".join(g) for g in note["share_classes_merged"]) or "none"
    splits = ", ".join(f"{s} {n}" for s, n in note["splits"].items())
    lines = [
        (
            f"snapshot: {note['source_url']} (revision of "
            f"{note['revision_timestamp']}, fetched {note['fetched_at']})"
        ),
        (
            f"{note['listings']} listings, {note['financials_dropped']} in "
            f"{FINANCIALS} dropped, share classes merged: {groups}"
        ),
        f"{note['companies']} companies: {splits} (seed {note['seed']})",
    ]
    mismatches = note["cik_mismatches"]
    lines.append(f"CIKs that differ from SEC's ticker map: {len(mismatches)}")
    for m in mismatches:
        sec = m["sec_cik"] if m["sec_cik"] is not None else "not in the map"
        lines.append(f"  {m['ticker']}: snapshot {m['cik']}, SEC {sec}")
    done = "wrote" if written else "unchanged:"
    lines.append(f"{done} {out} and {out.with_suffix('.json')}")
    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
    *,
    wiki: WikiClient | None = None,
    client: EdgarClient | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.companies",
        description="Build the stage 2 company list and split from an S&P 500 snapshot.",
    )
    parser.add_argument("--revision", type=int, help="Wikipedia revision id (oldid)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)
    try:
        wiki = wiki or WikiClient(args.cache_dir)
        snapshot = load_snapshot(wiki, args.revision)
        client = client or default_client()
        sec_ciks = parse_company_tickers(client.fetch_json(TICKERS_URL))
        companies, note = build(snapshot, sec_ciks, test_size=COMPANY_TEST_SIZE)
        written = write_companies(args.out, companies, note)
    except (CompaniesError, EdgarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_report(note, args.out, written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
