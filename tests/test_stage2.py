import contextlib
import csv
import io
import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from secbench import answers, coverage, stage2
from secbench.companies import COMPANY_TEST, TRAIN, VALIDATION, Company, companies_csv
from secbench.edgar import EdgarClient, RateLimiter
from secbench.stage2 import (
    DONE,
    ERROR,
    FAILED,
    NO_ITEM_202,
    NO_LINKED_FILING,
    NO_PRESS_RELEASE,
    NO_XBRL_ANSWER,
    TIME_TEST,
    UNUSED,
)

FIXTURES = Path(__file__).parent / "fixtures"
UA = "Test Runner test@example.com"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{:010d}.json"
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{:010d}.json"
TICKERS = "https://www.sec.gov/files/company_tickers.json"

INCOME_STATEMENT = """<html><body><p>Results for the quarter</p><table>
<tr><td></td><td>Three Months Ended</td></tr>
<tr><td>Total revenues</td><td>$1,000</td></tr>
<tr><td>Net income</td><td>$100</td></tr>
<tr><td>Diluted earnings per share</td><td>$1.00</td></tr>
</table></body></html>"""
NO_TABLES = "<html><body><p>Our chief financial officer will retire.</p></body></html>"


def accession(cik: int, n: int) -> str:
    return f"{cik:010d}-26-{n:06d}"


class FakeSec:
    """Six companies, one per outcome, served from routes built here."""

    def __init__(self) -> None:
        self.routes: dict[str, bytes] = {}
        self.requests: list[str] = []
        self.tickers: dict[str, int] = {}
        self.facts: dict[int, dict] = {}

    # Building the routes

    def company(self, ticker: str, cik: int, filings: list[dict]) -> None:
        """filings: dicts of accession, form, filed, report, items, and for an 8-K
        its exhibit ("release", "no_tables" or None for no EX-99)."""
        self.tickers[ticker] = cik
        recent = {
            "accessionNumber": [f["accession"] for f in filings],
            "form": [f["form"] for f in filings],
            "filingDate": [f["filed"] for f in filings],
            "reportDate": [f.get("report", "") for f in filings],
            "items": [f.get("items", "") for f in filings],
            "primaryDocument": ["form.htm" for _ in filings],
        }
        self.routes[SUBMISSIONS.format(cik)] = json.dumps(
            {"filings": {"recent": recent, "files": []}}
        ).encode()
        for f in filings:
            if f["form"] == "8-K" and "2.02" in f.get("items", ""):
                self.eight_k(cik, f["accession"], f.get("exhibit"))

    def eight_k(self, cik: int, accn: str, exhibit: str | None) -> None:
        folder = f"/Archives/edgar/data/{cik}/{accn.replace('-', '')}/"
        rows = [("1", "8-K", "form.htm", "8-K")]
        if exhibit:
            rows.append(("2", "Press release", "ex991.htm", "EX-99.1"))
            body = INCOME_STATEMENT if exhibit == "release" else NO_TABLES
            self.routes[f"https://www.sec.gov{folder}ex991.htm"] = body.encode()
        else:
            rows.append(("2", "Agreement", "ex101.htm", "EX-10.1"))
        cells = "".join(
            f'<tr><td scope="row">{seq}</td><td scope="row">{desc}</td>'
            f'<td scope="row"><a href="{folder}{name}">{name}</a></td>'
            f'<td scope="row">{kind}</td><td scope="row">100</td></tr>'
            for seq, desc, name, kind in rows
        )
        self.routes[f"https://www.sec.gov{folder}{accn}-index.htm"] = (
            '<html><body><table class="tableFile" summary="Document Format Files"><tr>'
            '<th scope="col">Seq</th><th scope="col">Description</th>'
            '<th scope="col">Document</th><th scope="col">Type</th>'
            f'<th scope="col">Size</th></tr>{cells}</table></body></html>'
        ).encode()

    def quarter_facts(
        self, cik: int, accn: str, end: str, revenue: int, eps: bool = True
    ) -> None:
        """Quarter revenue, net income and (unless eps is False) diluted EPS from the
        10-Q accn."""
        start = (date.fromisoformat(end) - timedelta(days=90)).isoformat()
        tags = self.facts.setdefault(cik, {})

        def add(tag: str, unit: str, value) -> None:
            entry = {"start": start, "end": end, "val": value, "accn": accn}
            tags.setdefault(tag, {"units": {unit: []}})["units"][unit].append(entry)

        add("Revenues", "USD", revenue)
        add("NetIncomeLoss", "USD", revenue // 10)
        if eps:
            add("EarningsPerShareDiluted", "USD/shares", 1.0)

    def finish(self) -> None:
        self.routes[TICKERS] = json.dumps(
            {
                str(i): {"cik_str": cik, "ticker": ticker, "title": ticker}
                for i, (ticker, cik) in enumerate(self.tickers.items())
            }
        ).encode()
        for cik, tags in self.facts.items():
            self.routes[COMPANYFACTS.format(cik)] = json.dumps(
                {"cik": cik, "facts": {"us-gaap": tags}}
            ).encode()

    # Serving

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        body = self.routes.get(url)
        return (
            httpx.Response(404) if body is None else httpx.Response(200, content=body)
        )

    def client(self, cache_dir: Path) -> EdgarClient:
        return EdgarClient(
            UA,
            cache_dir,
            transport=httpx.MockTransport(self.handler),
            limiter=RateLimiter(8.0, clock=lambda: 0.0, sleep=lambda s: None),
            sleep=lambda s: None,
        )


def eight_k(cik: int, n: int, filed: str, exhibit: str | None = "release") -> dict:
    return {
        "accession": accession(cik, n),
        "form": "8-K",
        "filed": filed,
        "items": "2.02,9.01",
        "exhibit": exhibit,
    }


def ten_q(cik: int, n: int, filed: str, report: str) -> dict:
    return {
        "accession": accession(cik, n),
        "form": "10-Q",
        "filed": filed,
        "report": report,
    }


OK, NOREL, NOPR, UNLNK, NOXBRL, BAD = 101, 102, 103, 104, 105, 106


def fake_sec() -> FakeSec:
    sec = FakeSec()
    # OKCO (train): one release before 2026-07-01 (train), one after (time_test),
    # one before the window and one after it; its second 10-Q has no EPS.
    sec.company(
        "OKCO",
        OK,
        [
            eight_k(OK, 1, "2023-08-01"),  # before SINCE
            eight_k(OK, 2, "2026-04-30"),
            ten_q(OK, 3, "2026-05-05", "2026-03-31"),
            eight_k(OK, 4, "2026-07-30"),
            ten_q(OK, 5, "2026-08-04", "2026-06-30"),
            eight_k(OK, 6, "2026-10-20"),  # after UNTIL
        ],
    )
    sec.quarter_facts(OK, accession(OK, 3), "2026-03-31", 1000)
    sec.quarter_facts(OK, accession(OK, 5), "2026-06-30", 1200, eps=False)
    # NOREL (train): an 8-K, but not Item 2.02.
    sec.company(
        "NOREL",
        NOREL,
        [
            {
                "accession": accession(NOREL, 1),
                "form": "8-K",
                "filed": "2026-02-01",
                "items": "5.02",
            }
        ],
    )
    # NOPR (train): an Item 2.02 8-K without an EX-99, and one that isn't an
    # earnings release.
    sec.company(
        "NOPR",
        NOPR,
        [
            eight_k(NOPR, 1, "2026-02-01", exhibit=None),
            eight_k(NOPR, 2, "2026-03-01", exhibit="no_tables"),
        ],
    )
    # UNLNK (train): a release whose 10-Q never came.
    sec.company("UNLNK", UNLNK, [eight_k(UNLNK, 1, "2026-05-01")])
    # NOXBRL (company_test): five linked releases, no company facts.
    filings = []
    for q, (filed, report) in enumerate(
        [
            ("2025-04-30", "2025-03-31"),
            ("2025-07-30", "2025-06-30"),
            ("2025-10-30", "2025-09-30"),
            ("2026-04-30", "2026-03-31"),
            ("2026-07-30", "2026-06-30"),
        ]
    ):
        filings.append(eight_k(NOXBRL, 2 * q + 1, filed))
        q_filed = (date.fromisoformat(filed) + timedelta(days=5)).isoformat()
        filings.append(ten_q(NOXBRL, 2 * q + 2, q_filed, report))
    sec.company("NOXBRL", NOXBRL, filings)
    sec.finish()
    return sec


COMPANIES = [
    Company("OKCO", OK, "OK Co", "Industrials", TRAIN),
    Company("NOREL", NOREL, "No Release Co", "Industrials", TRAIN),
    Company("NOPR", NOPR, "No Press Release Co", "Industrials", TRAIN),
    Company("UNLNK", UNLNK, "Unlinked Co", "Industrials", TRAIN),
    Company("NOXBRL", NOXBRL, "No XBRL Co", "Utilities", COMPANY_TEST),
    # Not in SEC's ticker map (reported) and nothing at its CIK: an error, logged.
    Company("BADCO", BAD, "Bad Co", "Energy", TRAIN),
]


def run(
    tmp_path: Path, sec: FakeSec, *args: str, companies=COMPANIES
) -> tuple[int, str]:
    listing = tmp_path / "companies.csv"
    listing.write_text(companies_csv(companies))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = stage2.main(
            ["--companies", str(listing), "--out", str(tmp_path / "stage2"), *args],
            client=sec.client(tmp_path / "raw"),
        )
    return status, out.getvalue()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_build_logs_every_failure_and_goes_on(tmp_path) -> None:
    sec = fake_sec()
    status, out = run(tmp_path, sec)
    assert status == 1  # BADCO's error
    lines = out.splitlines()
    assert lines[0] == (
        "6 companies selected, 0 already done, 6 to run (releases filed 2023-10-01 to "
        "2026-09-30)"
    )
    assert lines[1:3] == [
        (
            "SEC's ticker map disagrees with companies.csv for 1 of 6 companies "
            f"(in {tmp_path / 'stage2' / 'ticker_map.csv'}; not followed)"
        ),
        "  BADCO: companies.csv 106, SEC not in the map",
    ]
    assert lines[3].startswith(
        "[1/6] OKCO (train): 2 8-Ks, 2 linked, 5 answers; no_xbrl_answer 1; done "
    )
    assert lines[8].startswith(
        "[6/6] BADCO (train): 0 8-Ks, 0 linked, 0 answers; error 1; failed"
    )
    folder = tmp_path / "stage2"
    failures = [
        (f["ticker"], f["release_accession"], f["step"], f["kind"])
        for f in read(folder / "failures.csv")
    ]
    assert failures == [
        ("OKCO", accession(OK, 4), "answers", NO_XBRL_ANSWER),  # no EPS
        ("NOREL", "", "releases", NO_ITEM_202),
        ("NOPR", accession(NOPR, 1), "releases", NO_PRESS_RELEASE),
        ("NOPR", accession(NOPR, 2), "link", NO_PRESS_RELEASE),
        ("UNLNK", accession(UNLNK, 1), "link", NO_LINKED_FILING),
        ("NOXBRL", "", "answers", NO_XBRL_ANSWER),
        ("BADCO", "", "build", ERROR),
    ]
    reasons = {
        (f["ticker"], f["release_accession"]): f["reason"]
        for f in read(folder / "failures.csv")
    }
    assert reasons["NOREL", ""] == "no Item 2.02 8-K filed 2023-10-01 to 2026-09-30"
    assert (
        reasons["NOPR", accession(NOPR, 1)] == "no EX-99.1 exhibit (exhibits: EX-10.1)"
    )
    assert reasons["NOPR", accession(NOPR, 2)] == ("not an earnings release: no tables")
    assert reasons["UNLNK", accession(UNLNK, 1)].startswith("unlinked: no 10-Q or 10-K")
    assert reasons["OKCO", accession(OK, 4)].startswith(
        "eps_diluted quarter: no fact for that accession"
    )
    assert reasons["NOXBRL", ""].startswith("company facts: ")
    assert reasons["BADCO", ""] == (
        f"EdgarError: GET {SUBMISSIONS.format(BAD)} returned HTTP 404"
    )
    assert read(folder / "ticker_map.csv") == [
        {"ticker": "BADCO", "cik": "106", "sec_cik": ""}
    ]

    releases = read(folder / "releases.csv")
    assert [
        (r["ticker"], r["filing_date"], r["link_status"], r["split"]) for r in releases
    ] == [
        ("OKCO", "2026-04-30", "linked", TRAIN),
        ("OKCO", "2026-07-30", "linked", TIME_TEST),  # from 2026-07-01
        ("NOPR", "2026-02-01", "", TRAIN),  # not saved
        ("NOPR", "2026-03-01", "excluded", TRAIN),
        ("UNLNK", "2026-05-01", "unlinked", TRAIN),
        # The latest four linked releases; the oldest is unused.
        ("NOXBRL", "2025-04-30", "linked", UNUSED),
        ("NOXBRL", "2025-07-30", "linked", COMPANY_TEST),
        ("NOXBRL", "2025-10-30", "linked", COMPANY_TEST),
        ("NOXBRL", "2026-04-30", "linked", COMPANY_TEST),
        ("NOXBRL", "2026-07-30", "linked", COMPANY_TEST),
    ]
    okco = [r for r in releases if r["ticker"] == "OKCO"]
    assert [(r["answers"], r["missing"]) for r in okco] == [("3", "0"), ("2", "1")]
    assert [a["value"] for a in read(folder / "answers.csv")] == [
        "1000",
        "100",
        "1.0",
        "1200",
        "120",
    ]
    assert len(read(folder / "links.csv")) == 9  # every saved release
    statuses = {
        c.ticker: json.loads(
            (folder / "companies" / f"{c.cik:010d}" / "status.json").read_text()
        )["status"]
        for c in COMPANIES
    }
    assert statuses == {
        "OKCO": DONE,
        "NOREL": DONE,
        "NOPR": DONE,
        "UNLNK": DONE,
        "NOXBRL": DONE,
        "BADCO": FAILED,
    }
    # Releases outside the window were never fetched.
    assert not any(accession(OK, 1).replace("-", "") in url for url in sec.requests)
    assert not any(accession(OK, 6).replace("-", "") in url for url in sec.requests)


def test_a_second_run_resumes_from_the_cache(tmp_path) -> None:
    sec = fake_sec()
    run(tmp_path, sec)
    first = {name: (tmp_path / "stage2" / name).read_text() for name in stage2.FILES}
    sec.requests.clear()
    status, out = run(tmp_path, sec)
    # Only the failed company runs again; nothing else is fetched (an error
    # response is never cached).
    assert out.splitlines()[0] == (
        "6 companies selected, 5 already done, 1 to run (releases filed 2023-10-01 to "
        "2026-09-30)"
    )
    assert "BADCO" in out.splitlines()[3]
    assert status == 1 and sec.requests == [SUBMISSIONS.format(BAD)]
    assert {n: (tmp_path / "stage2" / n).read_text() for n in stage2.FILES} == first


def test_an_interrupted_company_runs_again(tmp_path, monkeypatch) -> None:
    sec = fake_sec()
    calls = {"n": 0}
    real = stage2.answer_ticker

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("connection reset mid-company")
        return real(*args, **kwargs)

    monkeypatch.setattr(stage2, "answer_ticker", flaky)
    status, out = run(tmp_path, sec, "--tickers", "OKCO")
    assert status == 1
    assert "OKCO (train): 0 8-Ks, 0 linked, 0 answers; error 1; failed" in out
    [failure] = read(tmp_path / "stage2" / "failures.csv")
    assert failure["reason"] == "RuntimeError: connection reset mid-company"
    assert read(tmp_path / "stage2" / "releases.csv") == []  # not merged until done
    sec.requests.clear()
    status, out = run(tmp_path, sec, "--tickers", "OKCO")
    # From the cache, but for the company facts the first run never got to.
    assert status == 0 and sec.requests == [COMPANYFACTS.format(OK)]
    assert "OKCO (train): 2 8-Ks, 2 linked, 5 answers; no_xbrl_answer 1; done" in out
    assert len(read(tmp_path / "stage2" / "releases.csv")) == 2


def test_redo_runs_finished_companies_again(tmp_path) -> None:
    sec = fake_sec()
    run(tmp_path, sec, "--tickers", "OKCO", "UNLNK")
    status, out = run(tmp_path, sec, "--tickers", "OKCO")
    assert "0 to run" in out.splitlines()[0]
    status, out = run(tmp_path, sec, "--tickers", "OKCO", "--redo")
    assert "1 to run" in out.splitlines()[0] and status == 0


def test_unknown_tickers_are_refused(tmp_path, capsys) -> None:
    listing = tmp_path / "companies.csv"
    listing.write_text(companies_csv(COMPANIES))
    status = stage2.main(
        ["--companies", str(listing), "--tickers", "NOPE"], client=None
    )
    assert status == 1
    assert "not in" in capsys.readouterr().err


# Validation: the stage 1 links, answered again from the cached company facts


def fixture_client(cache_dir: Path) -> EdgarClient:
    """Serves link_submissions.json and answers_companyfacts.json (by URL)."""
    bodies = json.loads((FIXTURES / "link_submissions.json").read_bytes()) | json.loads(
        (FIXTURES / "answers_companyfacts.json").read_bytes()
    )
    routes = {url: json.dumps(body).encode() for url, body in bodies.items()}

    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(str(request.url))
        return (
            httpx.Response(404) if body is None else httpx.Response(200, content=body)
        )

    return EdgarClient(
        UA,
        cache_dir,
        transport=httpx.MockTransport(handler),
        limiter=RateLimiter(8.0, clock=lambda: 0.0, sleep=lambda s: None),
    )


@pytest.fixture
def stage1(tmp_path) -> tuple[Path, Path]:
    """The stage 1 links fixture and an answers.csv built from the fixtures."""
    links = FIXTURES / "answers_links.csv"
    key = tmp_path / "answers.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        assert (
            answers.main(
                ["--links", str(links), "--out", str(key)],
                client=fixture_client(tmp_path / "raw"),
            )
            == 0
        )
    return links, key


def run_validation(tmp_path: Path, links: Path, key: Path) -> tuple[int, str]:
    listing = tmp_path / "companies.csv"
    listing.write_text(
        companies_csv(
            [
                Company("AAPL", 320193, "Apple", "IT", VALIDATION),
                Company("ORCL", 1341439, "Oracle", "IT", VALIDATION),
            ]
        )
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = stage2.main(
            [
                "--companies",
                str(listing),
                "--out",
                str(tmp_path / "stage2"),
                "--links",
                str(links),
                "--answers",
                str(key),
            ],
            client=fixture_client(tmp_path / "raw"),
        )
    return status, out.getvalue()


def test_validation_is_the_stage1_data(tmp_path, stage1) -> None:
    links, key = stage1
    status, out = run_validation(tmp_path, links, key)
    assert status == 0
    assert "[1/2] AAPL (validation): 6 8-Ks, 6 linked, 32 answers; no failures" in out
    folder = tmp_path / "stage2"
    assert read(folder / "failures.csv") == []
    releases = read(folder / "releases.csv")
    assert {r["split"] for r in releases} == {VALIDATION}
    assert len(releases) == 12 and all(r["link_status"] == "linked" for r in releases)
    stage1_rows = [r for r in read(key) if r["ticker"] in ("AAPL", "ORCL")]
    assert read(folder / "answers.csv") == stage1_rows
    # Oracle's three releases with preferred dividends (Findings 2026-10-14).
    dividends = {
        r["release_accession"] for r in read(folder / "preferred_dividends.csv")
    }
    assert dividends == {
        "0001193125-26-100148",
        "0001193125-26-265848",
        "0001193125-26-387905",
    }


def test_validation_answers_must_match_stage1(tmp_path, stage1) -> None:
    links, key = stage1
    rows = read(key)
    rows[0]["value"] = "1"  # an AAPL answer that no longer matches
    with key.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    status, _ = run_validation(tmp_path, links, key)
    assert status == 1
    [failure] = read(tmp_path / "stage2" / "failures.csv")
    assert (failure["ticker"], failure["kind"]) == ("AAPL", ERROR)
    assert failure["reason"] == f"answers differ from the stage 1 {key}"


# Coverage


def test_coverage_report(tmp_path) -> None:
    sec = fake_sec()
    run(tmp_path, sec)
    listing = tmp_path / "companies.csv"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = coverage.main(
            ["--stage2", str(tmp_path / "stage2"), "--companies", str(listing)]
        )
    assert status == 0
    text = (tmp_path / "stage2" / "coverage.md").read_text()
    assert out.getvalue() == text + f"wrote {tmp_path / 'stage2' / 'coverage.md'}\n"
    lines = text.splitlines()
    assert lines[:3] == [
        "# Stage 2 coverage",
        "",
        (
            "Companies in the list: validation 0, company_test 1, train 5; with "
            "releases in the data: 4."
        ),
    ]
    start = lines.index("## By release split")
    assert lines[start + 2 : start + 8] == [
        (
            "| split | companies | 8-Ks | no press release | not earnings | linked | not "
            "linked | figures answered | figures missing | answered | preferred dividends |"
        ),
        "|" + "|".join([" --- "] * 11) + "|",
        "| company_test | 1 | 4 | 0 | 0 | 4 | 0 | 0 | 0 | – | 0 |",
        "| train | 3 | 4 | 1 | 1 | 1 | 1 | 3 | 0 | 100.0% | 0 |",
        "| time_test | 1 | 1 | 0 | 0 | 1 | 0 | 2 | 1 | 66.7% | 0 |",
        "| unused | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | – | 0 |",
    ]
    # NOXBRL has five linked releases.
    start = lines.index("## company_test companies with fewer than 4 linked releases")
    assert lines[start + 2] == "None."
    start = lines.index("## By quarter of filing")
    assert [line.split(" | ")[0] for line in lines[start + 4 : start + 9]] == [
        "| 2025Q2",
        "| 2025Q3",
        "| 2025Q4",
        "| 2026Q1",
        "| 2026Q2",
    ]
    start = lines.index("## Answer tags by release split")
    assert lines[start + 2 : start + 7] == [
        "| field | tag | train | time_test |",
        "|" + "|".join([" --- "] * 4) + "|",
        "| eps_diluted | EarningsPerShareDiluted | 1 | 0 |",
        "| net_income | NetIncomeLoss | 1 | 1 |",
        "| revenue | Revenues | 1 | 1 |",
    ]
    start = lines.index("## Failures by kind and company split")
    assert lines[start + 2 : start + 9] == [
        "| kind | validation | company_test | train |",
        "|" + "|".join([" --- "] * 4) + "|",
        "| no_item_202 | 0 | 0 | 1 |",
        "| no_press_release | 0 | 0 | 2 |",
        "| no_linked_filing | 0 | 0 | 1 |",
        "| no_xbrl_answer | 0 | 1 | 1 |",
        "| error | 0 | 0 | 1 |",
    ]
    assert "- BADCO (train): build: EdgarError: GET " in text
    start = lines.index("## CIKs that differ from SEC's ticker map")
    assert lines[start + 2 : start + 7] == [
        "Reported, not followed: stage 2 fetches by companies.csv's CIK.",
        "",
        "| ticker | companies.csv | SEC |",
        "|" + "|".join([" --- "] * 3) + "|",
        "| BADCO | 106 | not in the map |",
    ]
    start = lines.index(
        "## Revenue answers between 1/5 and 1/2 of a larger revenue-like fact"
    )
    assert lines[start + 4] == "None."


def test_coverage_counts_preferred_dividends_by_split(tmp_path, stage1) -> None:
    links, key = stage1
    run_validation(tmp_path, links, key)
    text = coverage.format_coverage(
        read(tmp_path / "stage2" / "releases.csv"),
        read(tmp_path / "stage2" / "answers.csv"),
        read(tmp_path / "stage2" / "failures.csv"),
        {"AAPL": VALIDATION, "ORCL": VALIDATION},
    )
    [row] = [line for line in text.splitlines() if line.startswith("| validation |")]
    cells = row.strip("|").split(" | ")
    assert cells[0].strip() == "validation" and cells[-1].strip() == "3"


def test_coverage_lists_company_test_companies_with_fewer_than_four(tmp_path) -> None:
    def release(ticker: str, n: int, status: str) -> dict[str, str]:
        return {
            "ticker": ticker,
            "release_accession": f"{ticker}-{n}",
            "filing_date": "2026-04-30",
            "saved": "True",
            "link_status": status,
            "split": COMPANY_TEST if status == "linked" else UNUSED,
            "answers": "0",
            "missing": "0",
            "preferred_dividends": "False",
        }

    releases = (
        [release("FULL", n, "linked") for n in range(4)]
        + [release("SHORT", n, "linked") for n in range(3)]
        + [release("SHORT", 3, "excluded"), release("SHORT", 4, "unlinked")]
        + [release("TRAINCO", 1, "linked")]
    )
    splits = {
        "FULL": COMPANY_TEST,
        "SHORT": COMPANY_TEST,
        "NONE": COMPANY_TEST,  # no releases in the data
        "TRAINCO": TRAIN,
    }
    lines = coverage.format_coverage(releases, [], [], splits).splitlines()
    start = lines.index("## company_test companies with fewer than 4 linked releases")
    assert lines[start + 2 : start + 7] == [
        "| ticker | 8-Ks | linked |",
        "|" + "|".join([" --- "] * 3) + "|",
        "| NONE | 0 | 0 |",
        "| SHORT | 5 | 3 |",
        "",
    ]


# Revenue answers near a component (Decisions 2026-10-28)


def test_revenue_checks_are_written_and_merged(tmp_path) -> None:
    sec = fake_sec()
    # OKCO's first 10-Q: lease income 3x its revenue (1000), so the revenue answer is
    # listed; 5x would drop it.
    entry = {
        "start": "2025-12-31",
        "end": "2026-03-31",
        "val": 3000,
        "accn": accession(OK, 3),
    }
    sec.facts[OK]["OperatingLeaseLeaseIncome"] = {"units": {"USD": [entry]}}
    sec.finish()
    run(tmp_path, sec)
    folder = tmp_path / "stage2"
    [check] = read(folder / "revenue_checks.csv")
    assert check == {
        "ticker": "OKCO",
        "release_accession": accession(OK, 2),
        "period_type": "quarter",
        "derived": "False",
        "answer": "1000",
        "accession": accession(OK, 3),
        "tag": "Revenues",
        "start": "2025-12-31",
        "end": "2026-03-31",
        "value": "1000",
        "larger_tag": "OperatingLeaseLeaseIncome",
        "larger_value": "3000",
    }
    assert read(folder / "companies" / f"{OK:010d}" / "revenue_checks.csv") == [check]


def test_merge_refuses_a_finished_company_without_a_file(tmp_path) -> None:
    sec = fake_sec()
    run(tmp_path, sec)
    (tmp_path / "stage2" / "companies" / f"{OK:010d}" / "revenue_checks.csv").unlink()
    with pytest.raises(ValueError, match="revenue_checks.csv: missing"):
        stage2.merge(tmp_path / "stage2", COMPANIES)


AAPL_Q2_FY25 = FIXTURES / "input_releases" / "0000320193" / "0000320193-25-000055.json"


def test_printed_in_names_the_row_and_figure() -> None:
    assert coverage.printed_in(Decimal(95359000000), AAPL_Q2_FY25) == (
        "Total net sales: 95,359"
    )
    assert coverage.printed_in(Decimal(95358000000), AAPL_Q2_FY25) == "not printed"
    assert coverage.printed_in(Decimal(1), None) == "release not saved"


def test_coverage_lists_revenue_checks() -> None:
    releases = [
        {
            "ticker": "CPT",
            "release_accession": "CPT-1",
            "filing_date": "2026-04-30",
            "saved": "True",
            "link_status": "linked",
            "split": TRAIN,
            "answers": "1",
            "missing": "0",
            "preferred_dividends": "False",
        }
    ]
    check = {
        "ticker": "CPT",
        "release_accession": "CPT-1",
        "period_type": "quarter",
        "derived": "True",
        "answer": "320",
        "accession": "CPT-10Q",
        "tag": "Revenues",
        "start": "2025-01-01",
        "end": "2025-09-30",
        "value": "880",
        "larger_tag": "LeaseIncome",
        "larger_value": "2200",
    }
    text = coverage.format_coverage(
        releases, [], [], {"CPT": TRAIN}, [check], lambda c: f"row for {c['answer']}"
    )
    lines = text.splitlines()
    start = lines.index(
        "## Revenue answers between 1/5 and 1/2 of a larger revenue-like fact"
    )
    assert lines[start + 4 : start + 7] == [
        (
            "| ticker | release | split | period | answer | tag | larger | share | "
            "printed in |"
        ),
        "|" + "|".join([" --- "] * 9) + "|",
        (
            "| CPT | CPT-1 | train | quarter, derived | 320 | Revenues "
            "2025-01-01..2025-09-30 880 | LeaseIncome 2200 | 40% | row for 320 |"
        ),
    ]


# CIKs from companies.csv, never a ticker lookup


def test_company_ciks_come_from_companies_csv() -> None:
    assert stage2.company_ciks(COMPANIES[0]) == [OK]
    xom = Company("XOM", 2115436, "Exxon Mobil", "Energy", VALIDATION)
    assert stage2.company_ciks(xom) == [2115436, 34088]  # its TICKER_CIKS group
    old = Company("XOM", 34088, "Exxon Mobil", "Energy", VALIDATION)
    assert stage2.company_ciks(old) == [34088, 2115436]


def test_a_moved_ticker_is_reported_not_followed(tmp_path) -> None:
    sec = fake_sec()
    # SEC's ticker map now points OKCO at UNLNK's CIK.
    sec.tickers["OKCO"] = UNLNK
    sec.finish()
    status, out = run(tmp_path, sec, "--tickers", "OKCO")
    assert status == 0
    assert "  OKCO: companies.csv 101, SEC 104" in out.splitlines()
    assert read(tmp_path / "stage2" / "ticker_map.csv") == [
        {"ticker": "OKCO", "cik": "101", "sec_cik": "104"}
    ]
    # OKCO's own filings and facts, under CIK 101; nothing under 104.
    assert SUBMISSIONS.format(OK) in sec.requests
    assert COMPANYFACTS.format(OK) in sec.requests
    assert not any(
        f"{UNLNK:010d}" in url or f"/{UNLNK}/" in url for url in sec.requests
    )
    releases = read(tmp_path / "stage2" / "releases.csv")
    assert [(r["ticker"], r["cik"], r["link_status"]) for r in releases] == [
        ("OKCO", "101", "linked"),
        ("OKCO", "101", "linked"),
    ]


def test_coverage_without_a_ticker_map_check() -> None:
    lines = coverage.format_coverage([], [], [], {}).splitlines()
    start = lines.index("## CIKs that differ from SEC's ticker map")
    assert lines[start + 2] == "Not checked (no ticker_map.csv)."
    lines = coverage.format_coverage([], [], [], {}, ticker_map=[]).splitlines()
    assert lines[start + 2] == "None."
