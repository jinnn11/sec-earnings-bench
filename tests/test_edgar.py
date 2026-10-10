import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from secbench import edgar
from secbench.edgar import (
    EdgarClient,
    EdgarError,
    RateLimiter,
    cache_path,
    format_cik,
    parse_company_tickers,
    parse_retry_after,
    parse_submissions,
)

FIXTURES = Path(__file__).parent / "fixtures"
UA = "Test Runner test@example.com"
MAIN_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
OLDER_URL = "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json"
ROUTES = {
    edgar.TICKERS_URL: "company_tickers.json",
    MAIN_URL: "CIK0000320193.json",
    OLDER_URL: "CIK0000320193-submissions-001.json",
}
DOC_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019326000018/aapl-20260730.htm"

Handler = Callable[[httpx.Request], httpx.Response]


class FakeClock:
    """Stands in for both clocks and sleep; sleeping advances time."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(
    tmp_path: Path, handler: Handler, clock: FakeClock | None = None, **kwargs
) -> EdgarClient:
    clock = clock or FakeClock()
    return EdgarClient(
        UA,
        tmp_path / "raw",
        transport=httpx.MockTransport(handler),
        limiter=RateLimiter(8.0, clock=clock, sleep=clock.sleep),
        sleep=clock.sleep,
        wall_clock=clock,
        **kwargs,
    )


def fixture_server(seen: list[str]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        name = ROUTES.get(str(request.url))
        if name is None:
            return httpx.Response(404)
        return httpx.Response(200, content=(FIXTURES / name).read_bytes())

    return handler


def scripted(responses: list[httpx.Response | Exception], seen: list[str]) -> Handler:
    """Serve the given responses in order (an exception is raised instead)."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return handler


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# User-Agent


def test_missing_user_agent_raises_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(edgar, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(EdgarError, match="SEC_USER_AGENT is not set"):
        EdgarClient()


def test_placeholder_user_agent_raises(monkeypatch) -> None:
    monkeypatch.setattr(edgar, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("SEC_USER_AGENT", "Your Name your@email.com")
    with pytest.raises(EdgarError, match="SEC_USER_AGENT is not set"):
        edgar.load_user_agent()


def test_user_agent_read_from_environment(monkeypatch) -> None:
    monkeypatch.setattr(edgar, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    assert edgar.load_user_agent() == UA


def test_every_request_sends_user_agent(tmp_path) -> None:
    agents = []

    def handler(request: httpx.Request) -> httpx.Response:
        agents.append(request.headers["User-Agent"])
        return httpx.Response(200, content=b"x")

    client = make_client(tmp_path, handler)
    client.fetch(DOC_URL)
    client.fetch(edgar.TICKERS_URL)
    assert agents == [UA, UA]


# Cache


def test_cache_path_mirrors_url(tmp_path) -> None:
    assert cache_path(edgar.TICKERS_URL, tmp_path) == (
        tmp_path / "www.sec.gov" / "files" / "company_tickers.json"
    )
    assert cache_path(DOC_URL, tmp_path) == tmp_path.joinpath(
        "www.sec.gov/Archives/edgar/data/320193/000032019326000018/aapl-20260730.htm"
    )


def test_cache_path_directory_and_query(tmp_path) -> None:
    folder = "https://www.sec.gov/Archives/edgar/data/320193/000032019326000018/"
    assert cache_path(folder, tmp_path) == tmp_path.joinpath(
        "www.sec.gov/Archives/edgar/data/320193/000032019326000018/index"
    )
    a = cache_path("https://efts.sec.gov/LATEST/search-index?q=a", tmp_path)
    b = cache_path("https://efts.sec.gov/LATEST/search-index?q=b", tmp_path)
    assert a.parent == tmp_path / "efts.sec.gov" / "LATEST"
    assert a.name.startswith("search-index__")
    assert a != b


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/files/company_tickers.json",
        "http://www.sec.gov/files/company_tickers.json",
        "https://notsec.gov/x",
        "https://www.sec.gov/files/../../etc/passwd",
    ],
)
def test_cache_path_rejects_non_edgar_urls(tmp_path, url) -> None:
    with pytest.raises(ValueError):
        cache_path(url, tmp_path)


def test_fetch_caches_and_logs_hit_and_miss(tmp_path, caplog) -> None:
    seen: list[str] = []
    client = make_client(tmp_path, scripted([httpx.Response(200, content=b"v1")], seen))
    with caplog.at_level(logging.INFO, logger="secbench.edgar"):
        assert client.fetch(DOC_URL) == b"v1"
        assert client.fetch(DOC_URL) == b"v1"
    assert seen == [DOC_URL]
    assert cache_path(DOC_URL, tmp_path / "raw").read_bytes() == b"v1"
    assert [r.getMessage() for r in caplog.records] == [
        f"cache miss {DOC_URL}",
        f"cache hit {DOC_URL}",
    ]


def test_fetch_refresh_redownloads(tmp_path) -> None:
    seen: list[str] = []
    responses = [httpx.Response(200, content=b"v1"), httpx.Response(200, content=b"v2")]
    client = make_client(tmp_path, scripted(responses, seen))
    client.fetch(edgar.TICKERS_URL)
    assert client.fetch(edgar.TICKERS_URL, refresh=True) == b"v2"
    assert client.fetch(edgar.TICKERS_URL) == b"v2"
    assert len(seen) == 2


FOLDER_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123"
NESTED_URL = FOLDER_URL + "/index.json"


@pytest.mark.parametrize("order", [(FOLDER_URL, NESTED_URL), (NESTED_URL, FOLDER_URL)])
def test_url_and_url_under_it_can_both_be_cached(tmp_path, order) -> None:
    seen: list[str] = []

    def echo(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=str(request.url).encode())

    client = make_client(tmp_path, echo)
    for url in order:
        assert client.fetch(url) == url.encode()
    for url in order:
        assert client.fetch(url) == url.encode()
    assert seen == list(order)

    folder = cache_path(FOLDER_URL, tmp_path / "raw")
    assert sorted(p.name for p in folder.iterdir()) == ["#body", "index.json"]
    assert (folder / "#body").read_bytes() == FOLDER_URL.encode()

    assert client.fetch(FOLDER_URL, refresh=True) == FOLDER_URL.encode()
    assert seen == [*order, FOLDER_URL]


def test_error_responses_are_not_cached(tmp_path) -> None:
    seen: list[str] = []
    client = make_client(tmp_path, scripted([httpx.Response(404)], seen))
    with pytest.raises(EdgarError, match="HTTP 404"):
        client.fetch(DOC_URL)
    assert seen == [DOC_URL]  # 404 is not retried
    assert not (tmp_path / "raw").exists()


# Retries


def test_retries_5xx_with_backoff(tmp_path) -> None:
    seen: list[str] = []
    clock = FakeClock()
    responses = [httpx.Response(503), httpx.Response(500), httpx.Response(200)]
    client = make_client(tmp_path, scripted(responses, seen), clock)
    assert client.fetch(DOC_URL) == b""
    assert len(seen) == 3
    first, second = clock.sleeps
    assert 1.0 <= first <= 1.25
    assert 2.0 <= second <= 2.5


def test_retries_transport_errors(tmp_path) -> None:
    seen: list[str] = []
    responses = [httpx.ConnectError("boom"), httpx.Response(200, content=b"ok")]
    client = make_client(tmp_path, scripted(responses, seen))
    assert client.fetch(DOC_URL) == b"ok"
    assert len(seen) == 2


def test_429_honors_retry_after_seconds(tmp_path) -> None:
    seen: list[str] = []
    clock = FakeClock()
    responses = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(200, content=b"ok"),
    ]
    client = make_client(tmp_path, scripted(responses, seen), clock)
    assert client.fetch(DOC_URL) == b"ok"
    assert clock.sleeps == [7.0]


def test_429_honors_retry_after_date(tmp_path) -> None:
    seen: list[str] = []
    clock = FakeClock(datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC).timestamp())
    responses = [
        httpx.Response(429, headers={"Retry-After": "Sat, 10 Oct 2026 12:00:30 GMT"}),
        httpx.Response(200, content=b"ok"),
    ]
    client = make_client(tmp_path, scripted(responses, seen), clock)
    assert client.fetch(DOC_URL) == b"ok"
    assert clock.sleeps == [30.0]


def test_parse_retry_after() -> None:
    now = datetime(2015, 10, 21, 7, 28, 0, tzinfo=UTC).timestamp()
    assert parse_retry_after(None, now) is None
    assert parse_retry_after("120", now) == 120.0
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:45 GMT", now) == 45.0
    assert parse_retry_after("Wed, 21 Oct 2015 07:27:00 GMT", now) == 0.0
    assert parse_retry_after("soon", now) is None


def test_retry_after_longer_than_limit_raises(tmp_path) -> None:
    seen: list[str] = []
    clock = FakeClock()
    responses = [httpx.Response(429, headers={"Retry-After": "600"})]
    client = make_client(tmp_path, scripted(responses, seen), clock)
    with pytest.raises(EdgarError, match="asked to wait 600s"):
        client.fetch(DOC_URL)
    assert clock.sleeps == []


def test_gives_up_after_max_retries(tmp_path) -> None:
    seen: list[str] = []
    responses = [httpx.Response(500)] * 3
    client = make_client(tmp_path, scripted(responses, seen), max_retries=2)
    with pytest.raises(EdgarError, match=r"failed after 3 attempts \(HTTP 500\)"):
        client.fetch(DOC_URL)
    assert len(seen) == 3
    assert not (tmp_path / "raw").exists()


def test_403_is_not_retried(tmp_path) -> None:
    seen: list[str] = []
    client = make_client(tmp_path, scripted([httpx.Response(403)], seen))
    with pytest.raises(EdgarError, match="User-Agent"):
        client.fetch(DOC_URL)
    assert len(seen) == 1


# Rate limit


def test_requests_spaced_at_8_per_second(tmp_path) -> None:
    clock = FakeClock()
    client = make_client(tmp_path, lambda r: httpx.Response(200), clock)
    for n in range(3):
        client.fetch(f"{DOC_URL}.{n}")
    assert clock.sleeps == pytest.approx([0.125, 0.125])
    client.fetch(f"{DOC_URL}.0")  # cache hit: no wait
    assert len(clock.sleeps) == 2
    clock.now += 1.0  # idle long enough: no wait
    client.fetch(f"{DOC_URL}.3")
    assert len(clock.sleeps) == 2


def test_redirect_hops_are_rate_limited(tmp_path) -> None:
    seen: list[str] = []
    clock = FakeClock()
    responses = [
        httpx.Response(301, headers={"Location": FOLDER_URL + "/"}),
        httpx.Response(200, content=b"listing"),
    ]
    client = make_client(tmp_path, scripted(responses, seen), clock)
    assert client.fetch(FOLDER_URL) == b"listing"
    assert seen == [FOLDER_URL, FOLDER_URL + "/"]
    assert clock.sleeps == pytest.approx([0.125])
    assert cache_path(FOLDER_URL, tmp_path / "raw").read_bytes() == b"listing"


@pytest.mark.parametrize(
    "location",
    [
        "https://example.com/collect",
        "https://sec.gov.example.com/x",
        "http://www.sec.gov/Archives/x.htm",
    ],
)
def test_redirect_off_sec_gov_is_refused(tmp_path, location) -> None:
    seen: list[str] = []
    agents: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        agents.append(request.headers["User-Agent"])
        return httpx.Response(302, headers={"Location": location})

    client = make_client(tmp_path, handler)
    with pytest.raises(EdgarError, match="not an https sec.gov URL"):
        client.fetch(DOC_URL)
    assert seen == [DOC_URL]
    assert agents == [UA]
    assert not (tmp_path / "raw").exists()


# Tickers


def test_format_cik() -> None:
    assert format_cik(320193) == "0000320193"
    assert format_cik("0000320193") == "0000320193"


def test_parse_company_tickers() -> None:
    assert parse_company_tickers(load("company_tickers.json")) == {
        "NVDA": 1045810,
        "AAPL": 320193,
        "GOOGL": 1652044,
        "BRK-B": 1067983,
        "XOM": 2115436,
        "GOOG": 1652044,
        "BRK-A": 1067983,
    }


def test_ticker_to_cik(tmp_path) -> None:
    seen: list[str] = []
    client = make_client(tmp_path, fixture_server(seen))
    assert client.ticker_to_cik("AAPL") == 320193
    assert client.ticker_to_cik(" aapl ") == 320193
    assert client.ticker_to_cik("BRK.B") == 1067983
    assert client.ticker_to_cik("GOOG") == client.ticker_to_cik("GOOGL") == 1652044
    with pytest.raises(EdgarError, match="'ZZZZ' is not in"):
        client.ticker_to_cik("ZZZZ")
    assert seen == [edgar.TICKERS_URL]
    client.ticker_to_cik("AAPL", refresh=True)
    assert seen == [edgar.TICKERS_URL] * 2


def test_ticker_to_ciks_adds_listed_predecessors(tmp_path) -> None:
    client = make_client(tmp_path, fixture_server([]))
    assert client.ticker_to_ciks("XOM") == [2115436, 34088]  # holding company first
    assert client.ticker_to_ciks("xom") == [2115436, 34088]
    assert client.ticker_to_ciks("AAPL") == [320193]  # not listed: SEC's CIK only


def test_ticker_to_ciks_rejects_a_stale_list(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(edgar.TICKER_CIKS, "AAPL", [111, 222])
    client = make_client(tmp_path, fixture_server([]))
    with pytest.raises(EdgarError, match="maps 'AAPL' to CIK 320193"):
        client.ticker_to_ciks("AAPL")


def test_ticker_ciks_data_is_well_formed() -> None:
    for ticker, ciks in edgar.TICKER_CIKS.items():
        assert ticker == edgar.normalize_ticker(ticker)
        assert len(ciks) >= 2  # a single CIK needs no entry
        assert len(set(ciks)) == len(ciks)
        assert all(isinstance(cik, int) and cik > 0 for cik in ciks)


# Filings


def test_parse_submissions_columns_and_values() -> None:
    recent = load("CIK0000320193.json")["filings"]["recent"]
    older = load("CIK0000320193-submissions-001.json")
    filings = parse_submissions([recent, older])

    assert list(filings.columns) == [
        "accession_number",
        "form",
        "filing_date",
        "report_date",
        "items",
        "primary_document",
    ]
    assert len(filings) == 11 + 6
    assert filings["filing_date"].is_monotonic_decreasing
    assert filings["accession_number"].iloc[0] == "0001140361-26-039158"
    assert filings["accession_number"].iloc[-1] == "0000320193-99-000010"

    by_acc = filings.set_index("accession_number")
    q3_release = by_acc.loc["0000320193-26-000018"]
    assert q3_release["form"] == "8-K"
    assert q3_release["filing_date"] == pd.Timestamp("2026-07-30")
    assert q3_release["report_date"] == pd.Timestamp("2026-07-30")
    assert q3_release["items"] == "2.02,9.01"
    assert q3_release["primary_document"] == "aapl-20260730.htm"

    ten_k = by_acc.loc["0000320193-25-000079"]
    assert ten_k["form"] == "10-K"
    assert ten_k["report_date"] == pd.Timestamp("2025-09-27")
    assert ten_k["items"] == ""

    assert pd.isna(by_acc.loc["0001958244-26-000636", "report_date"])  # Form 144
    assert by_acc.loc["0000912057-00-024086", "primary_document"] == ""  # S-8, 2000
    assert by_acc.loc["0000320193-99-000010", "items"] == "2"  # pre-2004 numbering


def test_parse_submissions_sorts_and_dedupes() -> None:
    recent = load("CIK0000320193.json")["filings"]["recent"]
    older = load("CIK0000320193-submissions-001.json")
    filings = parse_submissions([older, recent, recent])
    assert len(filings) == 17
    assert filings["filing_date"].is_monotonic_decreasing
    assert filings["accession_number"].iloc[0] == "0001140361-26-039158"


def test_parse_submissions_rejects_bad_blocks() -> None:
    recent = load("CIK0000320193.json")["filings"]["recent"]
    ragged = {**recent, "form": recent["form"][:-1]}
    with pytest.raises(EdgarError, match="different lengths"):
        parse_submissions([ragged])
    missing = {k: v for k, v in recent.items() if k != "items"}
    with pytest.raises(EdgarError, match="lacks fields \\['items'\\]"):
        parse_submissions([missing])


def test_get_filings_reads_older_files(tmp_path) -> None:
    seen: list[str] = []
    client = make_client(tmp_path, fixture_server(seen))
    filings = client.get_filings(320193)
    assert len(filings) == 17
    assert seen == [MAIN_URL, OLDER_URL]
    assert client.get_filings("0000320193").equals(filings)
    assert seen == [MAIN_URL, OLDER_URL]  # second call served from cache
    client.get_filings(320193, refresh=True)
    assert seen == [MAIN_URL, OLDER_URL] * 2


# Command line


def test_cli_prints_counts_by_form(tmp_path, capsys) -> None:
    client = make_client(tmp_path, fixture_server([]))
    assert edgar.main(["aapl"], client=client) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "AAPL (CIK 0000320193): 17 filings"
    counts = [line.rsplit(maxsplit=1) for line in lines[1:]]
    assert [(form.strip(), int(n)) for form, n in counts] == [
        ("8-K", 6),
        ("10-K", 2),
        ("10-Q", 2),
        ("4", 2),
        ("144", 1),
        ("424B2", 1),
        ("8-K/A", 1),
        ("DEF 14A", 1),
        ("S-8", 1),
    ]


def test_cli_reports_errors(tmp_path, capsys) -> None:
    client = make_client(tmp_path, fixture_server([]))
    assert edgar.main(["ZZZZ"], client=client) == 1
    assert "is not in" in capsys.readouterr().err
