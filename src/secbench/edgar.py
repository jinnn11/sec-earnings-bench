"""Small EDGAR client: SEC User-Agent, on-disk cache, rate limit and retries.

Usage: uv run python -m secbench.edgar AAPL [--refresh]
"""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import json
import logging
import os
import random
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

import httpx
import pandas as pd
from dotenv import find_dotenv, load_dotenv

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/"
DEFAULT_CACHE_DIR = Path("data/raw")
PLACEHOLDER_USER_AGENT = "Your Name your@email.com"  # the value in .env.example
REQUESTS_PER_SECOND = 8.0  # SEC's limit is 10
MAX_RETRIES = 5
MAX_RETRY_WAIT = 120.0  # seconds; asked to wait longer than this, we give up
# File name for the body of a URL whose cache path is also a folder (".../0001" when
# ".../0001/index.json" is cached too). Raw URL path segments never contain "#".
FOLDER_BODY = "#body"


class EdgarError(Exception):
    """A request to EDGAR failed or the client is misconfigured."""


def load_user_agent() -> str:
    """Read SEC_USER_AGENT from the environment or .env."""
    load_dotenv(find_dotenv(usecwd=True))
    value = os.environ.get("SEC_USER_AGENT", "").strip()
    if not value or value == PLACEHOLDER_USER_AGENT:
        raise EdgarError(
            "SEC_USER_AGENT is not set. SEC requires a name and email in the "
            'User-Agent of every request: add SEC_USER_AGENT="Your Name you@example.com" '
            "to .env (see .env.example)."
        )
    return value


def check_url(url: str) -> None:
    """Allow only https URLs on sec.gov, so the User-Agent goes nowhere else."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.scheme != "https" or not (host == "sec.gov" or host.endswith(".sec.gov")):
        raise ValueError(f"not an EDGAR URL: {url}")


def cache_path(url: str, cache_dir: Path) -> Path:
    """Map a URL to its cache file: <cache_dir>/<host>/<path>.

    A URL ending in "/" maps to <path>/index; a query string adds a hash suffix.
    If another URL needs this path as a folder, the body is kept in it as FOLDER_BODY
    (see EdgarClient.fetch).
    """
    check_url(url)
    parts = urlsplit(url)
    segments = [s for s in parts.path.split("/") if s]
    if any(s in (".", "..") for s in segments):
        raise ValueError(f"URL path has dot segments: {url}")
    if not segments or parts.path.endswith("/"):
        segments.append("index")
    if parts.query:
        digest = hashlib.sha256(parts.query.encode()).hexdigest()[:12]
        segments[-1] = f"{segments[-1]}__{digest}"
    return cache_dir.joinpath(parts.hostname or "", *segments)


def make_cache_folder(cache_dir: Path, folder: Path) -> None:
    """Create folder under cache_dir, moving any cached file in the way inside it.

    A file at a path the folder needs becomes <that path>/FOLDER_BODY.
    """
    current = cache_dir
    for part in folder.relative_to(cache_dir).parts:
        current = current / part
        if current.is_file():
            moving = current.with_name(current.name + ".moving")
            os.replace(current, moving)
            current.mkdir()
            os.replace(moving, current / FOLDER_BODY)
    folder.mkdir(parents=True, exist_ok=True)


def parse_retry_after(value: str | None, now: float) -> float | None:
    """Seconds to wait from a Retry-After header (delta-seconds or HTTP date)."""
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, when.timestamp() - now)


def format_cik(cik: int | str) -> str:
    """Zero-pad a CIK to the 10 digits EDGAR uses in file names."""
    return f"{int(cik):010d}"


def normalize_ticker(ticker: str) -> str:
    """Upper-case, with class shares written as SEC does ("BRK.B" -> "BRK-B")."""
    return ticker.strip().upper().replace(".", "-")


# Tickers whose filings are under more than one CIK, current registrant first.
# SEC's ticker map (TICKERS_URL) knows only the current one.
TICKER_CIKS: dict[str, list[int]] = {
    # Exxon Mobil Corp (34088) came under a new holding company, ExxonMobil
    # Holdings Corp (2115436), on 2026-07-01 (8-K12B 0001193125-26-291990).
    "XOM": [2115436, 34088],
}


def parse_company_tickers(data: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Map ticker to CIK from company_tickers.json ({"0": {"cik_str", "ticker", ...}})."""
    return {entry["ticker"].upper(): int(entry["cik_str"]) for entry in data.values()}


# Submissions JSON field -> column in the filings table.
FILING_COLUMNS = {
    "accessionNumber": "accession_number",
    "form": "form",
    "filingDate": "filing_date",
    "reportDate": "report_date",
    "items": "items",
    "primaryDocument": "primary_document",
}


def parse_submissions(blocks: list[dict[str, list[Any]]]) -> pd.DataFrame:
    """Combine columnar filing blocks into one table, newest first.

    A block is filings.recent from the submissions JSON or one of the older-filings
    files it lists; both hold one list per field. Blank report dates become NaT;
    items stay as SEC's comma-separated string ("2.02,9.01", or "" for none).
    """
    frames = []
    for block in blocks:
        missing = FILING_COLUMNS.keys() - block.keys()
        if missing:
            raise EdgarError(f"submissions block lacks fields {sorted(missing)}")
        if len({len(block[field]) for field in FILING_COLUMNS}) > 1:
            raise EdgarError("submissions block has fields of different lengths")
        frames.append(
            pd.DataFrame({col: block[field] for field, col in FILING_COLUMNS.items()})
        )
    filings = pd.concat(frames, ignore_index=True)
    filings = filings.drop_duplicates("accession_number")
    filings["filing_date"] = pd.to_datetime(filings["filing_date"], format="%Y-%m-%d")
    filings["report_date"] = pd.to_datetime(filings["report_date"], format="%Y-%m-%d")
    filings["items"] = filings["items"].fillna("")
    filings["primary_document"] = filings["primary_document"].fillna("")
    filings = filings.sort_values("filing_date", ascending=False, kind="stable")
    return filings.reset_index(drop=True)


class RateLimiter:
    """Keep calls at least 1/rate seconds apart."""

    def __init__(
        self,
        rate: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._next: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._next is not None and now < self._next:
                self._sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


# One limiter per process, shared by every client that doesn't bring its own.
_SHARED_LIMITER = RateLimiter(REQUESTS_PER_SECOND)


class EdgarClient:
    """HTTP client for EDGAR that caches every response under cache_dir. An
    offline client only reads the cache: a URL not in it (or a refresh) raises
    EdgarError instead of making a request."""

    def __init__(
        self,
        user_agent: str | None = None,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        *,
        transport: httpx.BaseTransport | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
        max_retries: int = MAX_RETRIES,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self.offline = offline
        self._limiter = limiter or _SHARED_LIMITER
        self._sleep = sleep
        self._wall_clock = wall_clock
        self._http = httpx.Client(
            headers={"User-Agent": user_agent or load_user_agent()},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
            event_hooks={"request": [self._before_request]},
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def fetch(self, url: str, *, refresh: bool = False) -> bytes:
        """Return the body of url, from the cache unless refresh is set."""
        path = cache_path(url, self.cache_dir)
        if path.is_dir():
            path = path / FOLDER_BODY
        if not refresh and path.is_file():
            log.info("cache hit %s", url)
            return path.read_bytes()
        log.info("cache miss%s %s", " (refresh)" if refresh else "", url)
        if self.offline:
            raise EdgarError(
                f"{'refresh' if refresh else 'not in the cache'} (offline): {url}"
            )
        body = self._download(url)
        make_cache_folder(self.cache_dir, path.parent)
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(body)
        os.replace(partial, path)
        return body

    def fetch_json(self, url: str, *, refresh: bool = False) -> Any:
        return json.loads(self.fetch(url, refresh=refresh))

    def ticker_to_cik(self, ticker: str, *, refresh: bool = False) -> int:
        """Look up a ticker's CIK; refresh re-downloads company_tickers.json."""
        ciks = parse_company_tickers(self.fetch_json(TICKERS_URL, refresh=refresh))
        try:
            return ciks[normalize_ticker(ticker)]
        except KeyError:
            raise EdgarError(f"ticker {ticker!r} is not in {TICKERS_URL}") from None

    def ticker_to_ciks(self, ticker: str, *, refresh: bool = False) -> list[int]:
        """Every CIK a ticker's filings are under: TICKER_CIKS if it lists the
        ticker, else the one CIK from SEC's map."""
        cik = self.ticker_to_cik(ticker, refresh=refresh)
        ciks = TICKER_CIKS.get(normalize_ticker(ticker), [cik])
        if cik not in ciks:
            raise EdgarError(
                f"{TICKERS_URL} maps {ticker!r} to CIK {cik}, which TICKER_CIKS "
                f"does not list ({ciks}); update TICKER_CIKS"
            )
        return list(ciks)

    def get_filings(self, cik: int | str, *, refresh: bool = False) -> pd.DataFrame:
        """All filings for a CIK, including the older-filings files.

        refresh re-downloads the submissions JSON and every older file it lists.
        """
        url = f"{SUBMISSIONS_URL}CIK{format_cik(cik)}.json"
        submissions = self.fetch_json(url, refresh=refresh)
        blocks = [submissions["filings"]["recent"]]
        for older in submissions["filings"].get("files", []):
            blocks.append(
                self.fetch_json(SUBMISSIONS_URL + older["name"], refresh=refresh)
            )
        return parse_submissions(blocks)

    def _before_request(self, request: httpx.Request) -> None:
        """Runs before every HTTP request, including each redirect hop."""
        try:
            check_url(str(request.url))
        except ValueError:
            raise EdgarError(
                f"refusing to request {request.url}: not an https sec.gov URL "
                "(redirected?)"
            ) from None
        self._limiter.wait()

    def _download(self, url: str) -> bytes:
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            retry_after = None
            try:
                response = self._http.get(url)
            except httpx.TransportError as exc:
                problem = f"{type(exc).__name__}: {exc}"
            else:
                if response.is_success:
                    return response.content
                status = response.status_code
                if status == 403:
                    raise EdgarError(
                        f"GET {url} returned HTTP 403. SEC blocks requests without a "
                        "name and email in the User-Agent, and clients that exceed 10 "
                        "requests per second; check SEC_USER_AGENT, and if you were "
                        "rate limited wait 10 minutes."
                    )
                if status != 429 and status < 500:
                    raise EdgarError(f"GET {url} returned HTTP {status}")
                problem = f"HTTP {status}"
                retry_after = parse_retry_after(
                    response.headers.get("Retry-After"), self._wall_clock()
                )
            if attempt + 1 == attempts:
                raise EdgarError(
                    f"GET {url} failed after {attempts} attempts ({problem})"
                )
            delay = retry_after if retry_after is not None else self._backoff(attempt)
            if delay > MAX_RETRY_WAIT:
                raise EdgarError(
                    f"GET {url}: {problem}, and the server asked to wait {delay:.0f}s; "
                    "try again later"
                )
            log.warning(
                "GET %s: %s; retry %d/%d in %.1fs",
                url,
                problem,
                attempt + 1,
                self.max_retries,
                delay,
            )
            self._sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _backoff(attempt: int) -> float:
        base = 2.0**attempt  # 1, 2, 4, 8, 16 s
        return base + random.uniform(0, 0.25 * base)


_default_client: EdgarClient | None = None


def default_client() -> EdgarClient:
    """A process-wide client, created on first use."""
    global _default_client
    if _default_client is None:
        _default_client = EdgarClient()
    return _default_client


def fetch(url: str, *, refresh: bool = False) -> bytes:
    return default_client().fetch(url, refresh=refresh)


def ticker_to_cik(ticker: str, *, refresh: bool = False) -> int:
    return default_client().ticker_to_cik(ticker, refresh=refresh)


def get_filings(cik: int | str, *, refresh: bool = False) -> pd.DataFrame:
    return default_client().get_filings(cik, refresh=refresh)


def main(argv: list[str] | None = None, client: EdgarClient | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.edgar",
        description="Print a company's EDGAR filing counts by form.",
    )
    parser.add_argument("ticker")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download company_tickers.json and the submissions JSON",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        client = client or default_client()
        cik = client.ticker_to_cik(args.ticker, refresh=args.refresh)
        filings = client.get_filings(cik, refresh=args.refresh)
    except EdgarError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    counts = filings.groupby("form").size().sort_values(ascending=False, kind="stable")
    print(
        f"{normalize_ticker(args.ticker)} (CIK {format_cik(cik)}): {len(filings)} filings"
    )
    for form, count in counts.items():
        print(f"{form:<20}{count:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
