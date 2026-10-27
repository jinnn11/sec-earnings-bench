import json
import random
from pathlib import Path

import httpx
import pytest

from secbench import companies
from secbench.companies import (
    API_URL,
    COMPANY_TEST,
    TRAIN,
    VALIDATION,
    VALIDATION_TICKERS,
    CompaniesError,
    Company,
    Mismatch,
    WikiClient,
    assign_splits,
    build,
    cik_mismatches,
    companies_csv,
    load_companies,
    load_snapshot,
    merge_share_classes,
    parse_constituents,
    permalink,
    wiki_cache_path,
    write_companies,
)

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT = FIXTURES / "sp500_snapshot.html"
COMMITTED = Path(__file__).parent.parent / "data" / "companies.csv"
REVISION = 1382097112
CONTACT = "Test Runner test@example.com"
AGENT = f"sec-earnings-bench/0.1 ({CONTACT}) httpx/{httpx.__version__}"
API_BODY = {
    "batchcomplete": True,
    "query": {
        "pages": [
            {
                "pageid": 2676045,
                "ns": 0,
                "title": "List of S&P 500 companies",
                "revisions": [
                    {
                        "revid": REVISION,
                        "parentid": 1,
                        "timestamp": "2026-10-26T16:05:24Z",
                    }
                ],
            }
        ]
    },
}
# SEC's ticker map for the fixture's tickers (XOM at its new CIK).
SEC_CIKS = {
    "ABT": 1800,
    "AMD": 2488,
    "APD": 2969,
    "GOOGL": 1652044,
    "GOOG": 1652044,
    "AAPL": 320193,
    "BRK-B": 1067983,
    "CAT": 18230,
    "XOM": 2115436,
    "FOXA": 1754301,
    "FOX": 1754301,
    "HD": 354950,
    "JPM": 19617,
    "LIN": 1707925,
    "MSFT": 789019,
    "NFLX": 1065280,
    "NWSA": 1564708,
    "NWS": 1564708,
    "NKE": 320187,
    "NVDA": 1045810,
    "ORCL": 1341439,
    "V": 1403161,
    "WMT": 104169,
}


# Parsing the snapshot


def test_parse_constituents() -> None:
    listings = parse_constituents(SNAPSHOT.read_bytes())
    assert len(listings) == 23
    by_ticker = {c.ticker: c for c in listings}
    # Tickers as SEC writes them; names without the share class.
    assert by_ticker["BRK-B"] == Company(
        "BRK-B", 1067983, "Berkshire Hathaway", "Financials"
    )
    assert by_ticker["GOOGL"] == Company(
        "GOOGL", 1652044, "Alphabet Inc.", "Communication Services"
    )
    assert by_ticker["GOOG"].cik == 1652044
    assert by_ticker["XOM"] == Company("XOM", 2115436, "ExxonMobil", "Energy")
    assert {c.ticker: c.cik for c in listings} == SEC_CIKS


def test_a_snapshot_without_the_table_is_refused() -> None:
    with pytest.raises(CompaniesError, match="no table with id 'constituents'"):
        parse_constituents(b"<html><body><table id='other'></table></body></html>")


def test_merge_share_classes_keeps_the_first_listing() -> None:
    listings = parse_constituents(SNAPSHOT.read_bytes())
    merged, classes = merge_share_classes(listings)
    assert classes == [["GOOGL", "GOOG"], ["FOXA", "FOX"], ["NWSA", "NWS"]]
    tickers = [c.ticker for c in merged]
    assert len(tickers) == 20 and len({c.cik for c in merged}) == 20
    assert "GOOGL" in tickers and "GOOG" not in tickers


def test_cik_mismatches() -> None:
    listed = [
        Company("AAPL", 320193, "Apple", "IT"),
        Company("NEWCO", 9999999, "Spun off last week", "Industrials"),
        Company("OLDCO", 1111, "Moved to a new CIK", "Energy"),
        # A ticker in edgar.TICKER_CIKS matches under either of its CIKs.
        Company("XOM", 34088, "Exxon Mobil Corp", "Energy"),
    ]
    sec = {"AAPL": 320193, "OLDCO": 2222, "XOM": 2115436}
    assert cik_mismatches(listed, sec) == [
        Mismatch("NEWCO", 9999999, None),
        Mismatch("OLDCO", 1111, 2222),
    ]


# The split


def synthetic(count: int = 100) -> list[Company]:
    """The 10 validation tickers and count others, CIKs 1 to count + 10."""
    names = [*VALIDATION_TICKERS, *(f"T{n:03d}" for n in range(count))]
    return [Company(t, cik, t, "Industrials") for cik, t in enumerate(names, 1)]


def test_assign_splits() -> None:
    result = assign_splits(synthetic())
    splits = {c.ticker: c.split for c in result}
    assert [c.cik for c in result] == sorted(c.cik for c in result)
    assert sorted(t for t, s in splits.items() if s == VALIDATION) == sorted(
        VALIDATION_TICKERS
    )
    test = [c.cik for c in result if c.split == COMPANY_TEST]
    assert len(test) == 40
    assert sum(s == TRAIN for s in splits.values()) == 60
    # Pinned: random.Random(20261027).sample over the others' CIKs in order.
    others = [c.cik for c in sorted(synthetic(), key=lambda c: c.cik)][10:]
    assert sorted(test) == sorted(random.Random(20261027).sample(others, 40))


def test_split_ignores_input_order_and_changes_with_the_seed() -> None:
    companies_list = synthetic()
    shuffled = companies_list[::-1]
    assert assign_splits(shuffled) == assign_splits(companies_list)
    assert assign_splits(companies_list, seed=1) != assign_splits(companies_list)


def test_split_needs_every_validation_company_and_merged_classes() -> None:
    without_xom = [c for c in synthetic() if c.ticker != "XOM"]
    with pytest.raises(
        CompaniesError, match="validation companies not in the list: XOM"
    ):
        assign_splits(without_xom)
    duplicate = [*synthetic(), Company("DUP", 15, "Second class", "Industrials")]
    with pytest.raises(CompaniesError, match="merge share classes first"):
        assign_splits(duplicate)
    with pytest.raises(CompaniesError, match="only 30 companies outside validation"):
        assign_splits(synthetic(30))


def test_the_committed_split_reproduces() -> None:
    # data/companies.csv is tracked: the same companies and seed give the same split.
    committed = load_companies(COMMITTED)
    unsplit = [Company(c.ticker, c.cik, c.name, c.sector) for c in committed]
    assert assign_splits(unsplit) == committed
    counts = {s: sum(c.split == s for c in committed) for s in companies.SPLITS}
    assert counts == {VALIDATION: 10, COMPANY_TEST: 40, TRAIN: 374}
    assert all(c.sector != "Financials" for c in committed)


# Writing


def test_write_companies_never_changes_a_written_split(tmp_path) -> None:
    out = tmp_path / "companies.csv"
    result = assign_splits(synthetic())
    note = {"seed": 20261027}
    assert write_companies(out, result, note) is True
    assert out.read_text() == companies_csv(result)
    assert json.loads(out.with_suffix(".json").read_text()) == note
    assert load_companies(out) == result
    # The same list again: nothing to do.
    assert write_companies(out, result, {"seed": 0}) is False
    assert json.loads(out.with_suffix(".json").read_text()) == note
    # Another split: refused, and the file is untouched.
    with pytest.raises(CompaniesError, match="exists with a different list or split"):
        write_companies(out, assign_splits(synthetic(), seed=1), note)
    assert out.read_text() == companies_csv(result)


# Fetching


class FakeWikipedia:
    """Serves the API's latest revision and the fixture as the permalink."""

    def __init__(self, status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.status = status

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status)
        if str(request.url) == API_URL:
            return httpx.Response(200, json=API_BODY)
        if str(request.url) == permalink(REVISION):
            return httpx.Response(200, content=SNAPSHOT.read_bytes())
        return httpx.Response(404)

    def client(self, cache_dir: Path) -> WikiClient:
        return WikiClient(cache_dir, transport=httpx.MockTransport(self.handler))


def test_load_snapshot_fetches_once_with_a_descriptive_user_agent(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", CONTACT)
    wiki = FakeWikipedia()
    snapshot = load_snapshot(wiki.client(tmp_path))
    assert [str(r.url) for r in wiki.requests] == [API_URL, permalink(REVISION)]
    for request in wiki.requests:
        assert request.headers["User-Agent"] == AGENT
    assert snapshot.url == (
        "https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies"
        f"&oldid={REVISION}"
    )
    assert (snapshot.revision_id, snapshot.revision_timestamp) == (
        REVISION,
        "2026-10-26T16:05:24Z",
    )
    assert snapshot.html == SNAPSHOT.read_bytes()
    # Saved under the cache with a note of its source.
    cached = wiki_cache_path(snapshot.url, tmp_path)
    assert cached.read_bytes() == SNAPSHOT.read_bytes()
    assert cached.parent == tmp_path / "en.wikipedia.org" / "w"
    note = json.loads(cached.with_name(cached.name + ".source.json").read_text())
    assert note == {
        "url": snapshot.url,
        "revision_id": REVISION,
        "revision_timestamp": "2026-10-26T16:05:24Z",
        "fetched_at": snapshot.fetched_at,
    }
    # Again: from the cache, with the same note.
    again = load_snapshot(wiki.client(tmp_path))
    assert len(wiki.requests) == 2
    assert again == snapshot


def test_wiki_client_refuses_other_hosts_and_errors(tmp_path) -> None:
    client = FakeWikipedia().client(tmp_path)
    for url in ("https://www.sec.gov/files/x.json", "http://en.wikipedia.org/w/x"):
        with pytest.raises(ValueError, match="not an en.wikipedia.org URL"):
            client.fetch(url)
    failing = FakeWikipedia(status=503)
    with pytest.raises(CompaniesError, match="HTTP 503"):
        WikiClient(
            tmp_path, transport=httpx.MockTransport(failing.handler), agent="x"
        ).fetch(API_URL)
    assert not wiki_cache_path(API_URL, tmp_path).exists()  # errors aren't cached


def test_user_agent_needs_sec_user_agent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "")
    monkeypatch.chdir(tmp_path)  # no .env here
    with pytest.raises(Exception, match="SEC_USER_AGENT is not set"):
        companies.user_agent()


# End to end


class FakeEdgar:
    """Answers company_tickers.json from SEC_CIKS."""

    def fetch_json(self, url: str):
        assert url == companies.TICKERS_URL
        return {
            str(i): {"cik_str": cik, "ticker": ticker, "title": ticker}
            for i, (ticker, cik) in enumerate(SEC_CIKS.items())
        }


def test_build_on_the_fixture(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", CONTACT)
    snapshot = load_snapshot(FakeWikipedia().client(tmp_path))
    result, note = build(snapshot, SEC_CIKS, test_size=3)
    assert note["listings"] == 23
    assert note["financials_dropped"] == 3  # BRK.B, JPM, V
    assert note["share_classes_merged"] == [
        ["GOOGL", "GOOG"],
        ["FOXA", "FOX"],
        ["NWSA", "NWS"],
    ]
    assert note["companies"] == 17
    assert note["splits"] == {VALIDATION: 10, COMPANY_TEST: 3, TRAIN: 4}
    assert note["cik_mismatches"] == []
    assert note["source_url"] == permalink(REVISION)
    assert {c.ticker for c in result} == {
        *VALIDATION_TICKERS,
        "ABT",
        "AMD",
        "APD",
        "GOOGL",
        "FOXA",
        "LIN",
        "NWSA",
    }


def test_main_writes_the_list_once(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", CONTACT)
    monkeypatch.setattr(companies, "COMPANY_TEST_SIZE", 3)  # the fixture has 7 others
    out = tmp_path / "companies.csv"
    wiki = FakeWikipedia()
    argv = ["--out", str(out), "--cache-dir", str(tmp_path / "raw")]
    assert (
        companies.main(argv, wiki=wiki.client(tmp_path / "raw"), client=FakeEdgar())
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        f"snapshot: {permalink(REVISION)} (revision of 2026-10-26T16:05:24Z, fetched "
        f"{json.loads(out.with_suffix('.json').read_text())['fetched_at']})"
    )
    assert lines[1:] == [
        (
            "23 listings, 3 in Financials dropped, share classes merged: GOOGL/GOOG; "
            "FOXA/FOX; NWSA/NWS"
        ),
        "17 companies: validation 10, company_test 3, train 4 (seed 20261027)",
        "CIKs that differ from SEC's ticker map: 0",
        f"wrote {out} and {out.with_suffix('.json')}",
    ]
    assert CONTACT not in "\n".join(lines)  # the contact is sent, never printed
    # A second run: no requests, nothing written.
    assert (
        companies.main(argv, wiki=wiki.client(tmp_path / "raw"), client=FakeEdgar())
        == 0
    )
    assert capsys.readouterr().out.splitlines()[-1].startswith("unchanged: ")
    assert len(wiki.requests) == 2
