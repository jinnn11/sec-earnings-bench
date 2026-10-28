import contextlib
import csv
import io
import json
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from secbench import edgar, link
from secbench.edgar import EdgarClient, RateLimiter, parse_submissions
from secbench.link import (
    COLUMNS,
    EarningsCheck,
    Release,
    check_earnings,
    combine_filings,
    find_filing,
    link_releases,
    load_release,
    load_releases,
    markdown_tables,
    ticker_filings,
)
from secbench.releases import html_to_markdown

FIXTURES = Path(__file__).parent / "fixtures"
UA = "Test Runner test@example.com"
HANDCHECKED = FIXTURES / "links_handchecked.csv"
HANDCHECKED_COLUMNS = [
    "ticker",
    "release_accession",
    "expected_status",
    "expected_linked_form",
    "expected_linked_accession",
    "expected_period_end",
]
NKE_CFO_NEWS = "0000320187-26-000070"  # Item 2.02 8-K announcing a CFO change


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def release_fixtures() -> list[dict]:
    return json.loads(fixture("link_releases.json"))


def make_client(cache_dir: Path) -> EdgarClient:
    """A client served from link_submissions.json (URL -> JSON); other URLs 404."""
    routes = json.loads(fixture("link_submissions.json"))

    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(str(request.url))
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, content=json.dumps(body).encode())

    return EdgarClient(
        UA,
        cache_dir,
        transport=httpx.MockTransport(handler),
        limiter=RateLimiter(8.0, clock=lambda: 0.0, sleep=lambda s: None),
    )


def write_releases(entries: list[dict], folder: Path) -> None:
    """Lay out release fixtures as secbench.releases saves them."""
    for entry in entries:
        record = entry["record"]
        cik_dir = folder / edgar.format_cik(record["cik"])
        cik_dir.mkdir(parents=True, exist_ok=True)
        (cik_dir / f"{record['accession_number']}.json").write_text(json.dumps(record))
        chosen = next(e for e in record["exhibits"] if e["chosen"])
        (cik_dir / chosen["file"]).write_text(entry["markdown"])


def table(header: list[str], *rows: list[str]) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    separator = line(["---"] * len(header))
    return "\n".join([line(header), separator, *map(line, rows)]) + "\n"


HEADER = ["", "Three Months Ended June 30, 2026", "Three Months Ended June 30, 2025"]


# Earnings-release filter


def test_markdown_tables_splits_tables_and_drops_separators() -> None:
    markdown = (
        "Intro text\n\n"
        + table(["", "2026"], ["Net income", "$1"], [r"A \| B", "2"])
        + "\nMore text\n\n"
        + table(["", "2026"], ["Revenue", "3"])
    )
    assert markdown_tables(markdown) == [
        [["", "2026"], ["Net income", "$1"], ["A | B", "2"]],
        [["", "2026"], ["Revenue", "3"]],
    ]


@pytest.mark.parametrize(
    ("net_income", "per_share"),
    [
        ("Net income", "Diluted earnings per share"),
        ("NET EARNINGS", "EARNINGS PER SHARE:"),
        ("Net (loss) income", "Net loss per share — basic and diluted"),
        ("Net income/(loss) attributable to ExxonMobil", "Earnings per common share"),
        (
            "Consolidated net income attributable to Walmart",
            "Diluted net income per common share attributable to Walmart",
        ),
        ("Profit <sup>1</sup>", "Profit per common share — diluted"),
        ("Earnings/(Loss) (U.S. GAAP)", "Earnings Per Common Share (U.S. GAAP) ¹"),
        ("Total Earnings/(Loss) (U.S. GAAP)", "Per share data"),
        ("**Net income**", "Earnings per Class A common share"),
    ],
)
def test_filter_accepts_bottom_line_and_per_share_labels(net_income, per_share) -> None:
    markdown = table(
        HEADER,
        ["Revenue", "$100", "$90"],
        [net_income, "$10", "$(9)"],
        [per_share, "0.10", "0.09"],
    )
    assert check_earnings(markdown) == EarningsCheck(True)


def test_filter_accepts_per_share_heading_without_figures() -> None:
    # Apple's layout: "Earnings per share:" heads the Basic and Diluted rows.
    markdown = table(
        HEADER,
        ["Net income", "$23,434", "$21,448"],
        ["Earnings per share:", "", ""],
        ["Basic", "$1.58", "$1.40"],
        ["Diluted", "$1.57", "$1.40"],
    )
    assert check_earnings(markdown).ok


def test_filter_reads_labels_between_figures() -> None:
    # ExxonMobil's layout: current quarters, the label, then year-to-date.
    markdown = table(
        ["2Q26", "1Q26", "Dollars in millions", "YTD 2026"],
        ["14,525", "4,183", "Earnings (U.S. GAAP)", "18,708"],
        ["3.48", "1.00", "Earnings Per Common Share (U.S. GAAP)", "4.47"],
    )
    assert check_earnings(markdown).ok


@pytest.mark.parametrize(
    "markdown",
    [
        pytest.param(
            table(
                ["(In millions, except per share amounts)", "2026", "2025"],
                ["Net income", "$10", "$9"],
            ),
            id="per-share-only-in-header",
        ),
        pytest.param(
            table(HEADER, ["Net income", "$10", "$9"])
            + "\n"
            + table(HEADER, ["Earnings per share", "$0.10", "$0.09"]),
            id="rows-in-different-tables",
        ),
        pytest.param(
            table(
                HEADER,
                ["Net income:", "", ""],
                ["Diluted earnings per share", "1", "2"],
            ),
            id="net-income-without-figures",
        ),
        pytest.param(
            table(
                HEADER,
                ["Profit (Loss) by Segment", "", ""],
                ["Profit margin", "12%", "11%"],
                ["Earnings before income taxes", "$12", "$11"],
                ["Earnings per share", "$0.10", "$0.09"],
            ),
            id="profit-and-earnings-that-are-not-the-bottom-line",
        ),
        pytest.param(
            table(
                HEADER, ["Net sales", "$100", "$90"], ["Operating income", "$12", "$11"]
            ),
            id="neither",
        ),
    ],
)
def test_filter_rejects_tables_without_both_rows(markdown) -> None:
    check = check_earnings(markdown)
    assert not check.ok
    assert check.reason.startswith(
        "not an earnings release: no table has both a net income row and a "
        "per-share row ("
    )


def test_exclusion_reason_lists_first_labels_checked() -> None:
    long_label = "Revenue from contracts with customers, excluding assessed tax"
    markdown = (
        table(
            HEADER,
            [long_label, "$100", "$90"],
            ["Cost of sales", "60", "55"],
            ["", "40", "35"],  # no label
            ["Gross margin", "40", "35"],
        )
        + "\n"
        + table(["", "2026"], ["Net income", "$10"], ["Taxes", "2"], ["Other", "1"])
    )
    assert check_earnings(markdown).reason == (
        "not an earnings release: no table has both a net income row and a per-share "
        "row (2 tables, 6 labeled rows; first labels: "
        '"Revenue from contracts with customers,…", "Cost of sales", "Gross margin", '
        '"Net income", "Taxes")'
    )


def test_exclusion_reason_for_release_without_tables() -> None:
    assert check_earnings("Nike names a new CFO.\n") == EarningsCheck(
        False, "not an earnings release: no tables"
    )


@pytest.mark.parametrize(
    "name", ["a8-kex991q3202606272026.htm", "a8-kex991q4202509272025.htm"]
)
def test_filter_accepts_real_apple_releases(name) -> None:
    assert check_earnings(html_to_markdown(fixture(name))).ok


def test_filter_rejects_real_cfo_announcement() -> None:
    markdown = html_to_markdown(fixture("pressrelease062326.htm"))
    assert "Planned CFO Transition" in markdown
    assert check_earnings(markdown) == EarningsCheck(
        False, "not an earnings release: no tables"
    )


def test_filter_on_saved_release_fixtures() -> None:
    # Each excerpt is the first table of a saved release that the filter accepted;
    # the one release it rejected is kept whole.
    entries = release_fixtures()
    for entry in entries:
        check = check_earnings(entry["markdown"])
        assert check.ok == entry["markdown_is_excerpt"], entry["record"][
            "accession_number"
        ]
    rejected = [
        e["record"]["accession_number"] for e in entries if not e["markdown_is_excerpt"]
    ]
    assert rejected == [NKE_CFO_NEWS]


# Saved releases


EARNINGS = table(
    HEADER, ["Net income", "$10", "$9"], ["Diluted earnings per share", "1", "2"]
)
NO_TABLES = "not an earnings release: no tables"


def save_record(folder: Path, exhibits: list[tuple[str, str | None, bool]]) -> Path:
    """A saved release whose exhibits are (type, markdown or None if not HTML,
    chosen), in filing-index order."""
    accession = "0000320193-26-000018"
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for doc_type, markdown, chosen in exhibits:
        name = f"{accession}.md" if chosen else f"{accession}_{doc_type}.md"
        if markdown is not None:
            (folder / name).write_text(markdown)
        entries.append(
            {
                "type": doc_type,
                "file": name if markdown is not None else None,
                "chosen": chosen,
            }
        )
    record = {
        "ticker": "AAPL",
        "cik": 320193,
        "filing_date": "2026-07-30",
        "accession_number": accession,
        "exhibits": entries,
    }
    path = folder / f"{accession}.json"
    path.write_text(json.dumps(record))
    return path


def test_load_releases_checks_the_chosen_exhibit(tmp_path) -> None:
    save_record(
        tmp_path / "0000320193",
        [("EX-99.2", EARNINGS, False), ("EX-99.1", EARNINGS, True)],
    )
    assert load_releases(tmp_path) == [
        Release(
            "AAPL",
            320193,
            "0000320193-26-000018",
            date(2026, 7, 30),
            EarningsCheck(True, exhibit="EX-99.1"),  # chosen first, though listed 2nd
        )
    ]


@pytest.mark.parametrize(
    ("exhibits", "expected"),
    [
        # Statements only in EX-99.2 (ConocoPhillips, Philip Morris).
        (
            [("EX-99.1", "Results.\n", True), ("EX-99.2", EARNINGS, False)],
            EarningsCheck(True, exhibit="EX-99.2"),
        ),
        # The first that passes, in filing-index order.
        (
            [
                ("EX-99.1", "Results.\n", True),
                ("EX-99.3", EARNINGS, False),
                ("EX-99.2", EARNINGS, False),
            ],
            EarningsCheck(True, exhibit="EX-99.3"),
        ),
        # A non-HTML exhibit has no markdown and is skipped.
        (
            [
                ("EX-99.1", "Results.\n", True),
                ("EX-99.2", None, False),
                ("EX-99.3", EARNINGS, False),
            ],
            EarningsCheck(True, exhibit="EX-99.3"),
        ),
        # None passes: the chosen exhibit's reason, naming the others checked.
        (
            [
                ("EX-99.1", "Results.\n", True),
                ("EX-99.2", "Slides.\n", False),
                ("EX-99.3", None, False),
            ],
            EarningsCheck(False, f"{NO_TABLES}; also checked: EX-99.2"),
        ),
        # No other exhibit: the reason is unchanged (Nike's CFO news, stage 1).
        (
            [("EX-99.1", "Nike names a new CFO.\n", True)],
            EarningsCheck(False, NO_TABLES),
        ),
    ],
    ids=["ex-99.2", "index-order", "non-html-skipped", "none-pass", "chosen-only"],
)
def test_load_release_reads_every_saved_exhibit(tmp_path, exhibits, expected) -> None:
    path = save_record(tmp_path / "0000320193", exhibits)
    assert load_release(path).earnings == expected


def test_load_release_needs_one_chosen_exhibit(tmp_path) -> None:
    save_record(tmp_path / "0000320193", [("EX-99.1", EARNINGS, False)])
    with pytest.raises(ValueError, match="no single chosen exhibit"):
        load_releases(tmp_path)


# Linking


def filings_table(
    *rows: tuple[str, str, str, str | None], cik: int = 1
) -> pd.DataFrame:
    """(accession, form, filing date, report date) rows, shaped like get_filings."""
    return parse_submissions(
        [
            {
                "accessionNumber": [r[0] for r in rows],
                "form": [r[1] for r in rows],
                "filingDate": [r[2] for r in rows],
                "reportDate": [r[3] or "" for r in rows],
                "items": ["" for _ in rows],
                "primaryDocument": ["doc.htm" for _ in rows],
            }
        ]
    ).assign(cik=cik)


def release(
    accession: str, filed: date, ok: bool = True, ticker: str = "ACME"
) -> Release:
    reason = "" if ok else "not an earnings release: no tables"
    return Release(ticker, 1, accession, filed, EarningsCheck(ok, reason))


RELEASED = date(2026, 7, 30)


def test_find_filing_takes_first_qualifying_10q_or_10k() -> None:
    filings = filings_table(
        ("before", "10-Q", "2026-07-29", "2026-06-27"),  # filed before the release
        ("amended", "10-Q/A", "2026-07-30", "2026-06-27"),
        ("late-notice", "NT 10-Q", "2026-07-30", "2026-06-27"),
        ("transition", "10-KT", "2026-07-30", "2026-06-27"),
        ("old-period", "10-Q", "2026-07-30", "2026-03-28"),  # 124 days before
        ("no-period", "10-Q", "2026-07-30", None),
        ("future-period", "10-Q", "2026-07-30", "2026-09-26"),
        ("later", "10-Q", "2026-08-05", "2026-06-27"),
        ("first", "10-Q", "2026-07-31", "2026-06-27"),
    )
    assert find_filing(filings, RELEASED).accession_number == "first"


@pytest.mark.parametrize(
    ("period_end", "found"),
    [
        ("2026-07-30", True),  # same day
        ("2026-04-21", True),  # 100 days before
        ("2026-04-20", False),  # 101 days before
    ],
)
def test_find_filing_period_window(period_end, found) -> None:
    filings = filings_table(("q", "10-Q", "2026-07-31", period_end))
    assert (find_filing(filings, RELEASED) is not None) == found


def test_find_filing_breaks_same_day_ties_by_accession() -> None:
    filings = filings_table(
        ("0000000001-26-000002", "10-Q", "2026-07-31", "2026-06-27"),
        ("0000000001-26-000001", "10-K", "2026-07-31", "2026-06-27"),
    )
    assert find_filing(filings, RELEASED).accession_number == "0000000001-26-000001"


def test_link_fields_and_period_types() -> None:
    filings = filings_table(
        ("k", "10-K", "2025-10-31", "2025-09-27"),
        ("q", "10-Q", "2026-01-30", "2025-12-27"),
        cik=320193,
    )
    links = link_releases(
        [release("r2", date(2026, 1, 29)), release("r1", date(2025, 10, 30))], filings
    )
    assert links == [
        link.Link(
            "ACME",
            1,
            "r1",
            date(2025, 10, 30),
            "10-K",
            320193,
            "k",
            date(2025, 9, 27),
            "full year",
            "linked",
            "",
        ),
        link.Link(
            "ACME",
            1,
            "r2",
            date(2026, 1, 29),
            "10-Q",
            320193,
            "q",
            date(2025, 12, 27),
            "quarter",
            "linked",
            "",
        ),
    ]


def test_unlinked_reasons() -> None:
    filings = filings_table(("q", "10-Q", "2026-08-01", "2026-03-28"))
    early, late = link_releases(
        [release("early", RELEASED), release("late", date(2026, 9, 1))], filings
    )
    assert (early.status, early.linked_accession) == ("unlinked", "")
    assert early.reason == (
        "no 10-Q or 10-K filed on or after the release has a period ending 0-100 days "
        "before it (first after: 10-Q q, filed 2026-08-01, period 2026-03-28)"
    )
    assert (late.status, late.reason) == (
        "unlinked",
        "no 10-Q or 10-K filed on or after the release yet",
    )


def test_conflict_keeps_release_filed_closest() -> None:
    filings = filings_table(("k", "10-K", "2026-07-15", "2026-05-31"))
    first, second = link_releases(
        [release("r-0623", date(2026, 6, 23)), release("r-0630", date(2026, 6, 30))],
        filings,
    )
    assert (second.status, second.linked_accession) == ("linked", "k")
    assert (first.status, first.linked_accession, first.period_type) == (
        "conflict",
        "k",
        "full year",
    )
    assert (
        first.reason == "10-K k is also linked to r-0630; r-0630 was filed closer to it"
    )


def test_conflict_among_three_releases() -> None:
    filings = filings_table(("q", "10-Q", "2026-08-05", "2026-06-27"))
    links = link_releases(
        [
            release("a", date(2026, 7, 1)),
            release("b", date(2026, 7, 20)),
            release("c", date(2026, 8, 1)),
        ],
        filings,
    )
    assert [(lk.release_accession, lk.status) for lk in links] == [
        ("a", "conflict"),
        ("b", "conflict"),
        ("c", "linked"),
    ]
    assert links[0].reason == "10-Q q is also linked to b, c; c was filed closer to it"


def test_conflict_tie_keeps_neither() -> None:
    filings = filings_table(("q", "10-Q", "2026-08-05", "2026-06-27"))
    links = link_releases([release("b", RELEASED), release("a", RELEASED)], filings)
    assert [(lk.release_accession, lk.status) for lk in links] == [
        ("a", "conflict"),
        ("b", "conflict"),
    ]
    assert links[0].reason == (
        "10-Q q is also linked to b; a and b were filed equally close to it, so none "
        "is kept"
    )


def test_excluded_release_does_not_conflict() -> None:
    # Nike's CFO announcement (2026-06-23) and its Q4 release (2026-06-30) would
    # both link to the same 10-K.
    filings = filings_table(("k", "10-K", "2026-07-15", "2026-05-31"))
    news, earnings = link_releases(
        [
            release("news", date(2026, 6, 23), ok=False),
            release("q4", date(2026, 6, 30)),
        ],
        filings,
    )
    assert (news.status, news.reason, news.linked_accession) == (
        "excluded",
        "not an earnings release: no tables",
        "",
    )
    assert (earnings.status, earnings.linked_accession) == ("linked", "k")


# Filings under several CIKs


def test_combine_filings_keeps_shared_filing_once_with_first_cik() -> None:
    joint = ("0000034088-26-000093", "10-Q", "2026-08-03", "2026-06-30")
    combined = combine_filings(
        {
            2115436: filings_table(joint).drop(columns="cik"),
            34088: filings_table(
                joint, ("0000034088-26-000067", "10-Q", "2026-05-04", "2026-03-31")
            ).drop(columns="cik"),
        }
    )
    assert list(zip(combined["accession_number"], combined["cik"], strict=True)) == [
        ("0000034088-26-000093", 2115436),
        ("0000034088-26-000067", 34088),
    ]


def test_ticker_filings_searches_every_cik(tmp_path) -> None:
    filings = ticker_filings(make_client(tmp_path / "raw"), "XOM")
    periodic = filings[filings["form"].isin(["10-Q", "10-K"])]
    assert periodic["accession_number"].is_unique
    joint = periodic[periodic["accession_number"] == "0000034088-26-000093"]
    assert list(joint["cik"]) == [2115436]
    assert set(periodic["cik"]) == {2115436, 34088}
    assert len(periodic) == 7  # 10-K filed 2025-02 through the joint Q2 2026 10-Q


# Command line


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory) -> tuple[int, str, list[dict[str, str]], Path]:
    """link.main over every release fixture: exit code, stdout, links.csv rows and
    links.csv path."""
    tmp = tmp_path_factory.mktemp("link")
    write_releases(release_fixtures(), tmp / "releases")
    out = tmp / "links.csv"
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        status = link.main(
            ["--releases", str(tmp / "releases"), "--out", str(out)],
            client=make_client(tmp / "raw"),
        )
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return status, stdout.getvalue(), rows, out


def test_cli_on_saved_release_fixtures(fixture_run) -> None:
    status, stdout, rows, out = fixture_run
    assert status == 0
    lines = stdout.splitlines()
    assert lines[0].split() == [
        "ticker",
        "linked",
        "conflict",
        "excluded",
        "unlinked",
        "total",
    ]
    assert lines[6].split() == ["NKE", "6", "0", "1", "0", "7"]
    assert lines[11].split() == ["total", "60", "0", "1", "0", "61"]
    assert lines[12:] == [
        f"  NKE {NKE_CFO_NEWS} (2026-06-23) excluded: not an earnings release: no tables",
        f"wrote 61 rows to {out}",
    ]
    assert list(rows[0]) == COLUMNS
    by_accession = {row["release_accession"]: row for row in rows}
    assert by_accession["0002115436-26-000006"] == {
        "ticker": "XOM",
        "release_cik": "2115436",
        "release_accession": "0002115436-26-000006",
        "release_date": "2026-07-31",
        "linked_form": "10-Q",
        "linked_cik": "2115436",
        "linked_accession": "0000034088-26-000093",
        "period_end": "2026-06-30",
        "period_type": "quarter",
        "status": "linked",
        "reason": "",
    }
    assert by_accession[NKE_CFO_NEWS]["linked_cik"] == ""


def test_cli_continues_after_edgar_error(tmp_path, capsys) -> None:
    entries = [e for e in release_fixtures() if e["record"]["ticker"] == "AAPL"][:2]
    unknown = json.loads(json.dumps(entries[0]))
    unknown["record"]["ticker"] = "ZZZZ"
    unknown["record"]["accession_number"] = "0000320193-26-999999"
    for exhibit in unknown["record"]["exhibits"]:
        if exhibit["chosen"]:
            exhibit["file"] = "0000320193-26-999999.md"
    write_releases([*entries, unknown], tmp_path / "releases")
    out = tmp_path / "links.csv"

    argv = ["--releases", str(tmp_path / "releases"), "--out", str(out)]
    assert link.main(argv, client=make_client(tmp_path / "raw")) == 1
    captured = capsys.readouterr()
    assert "error: ZZZZ: ticker 'ZZZZ' is not in" in captured.err
    assert captured.out.splitlines()[-2].split() == ["total", "2", "0", "0", "0", "2"]
    with out.open(newline="") as f:
        assert [row["ticker"] for row in csv.DictReader(f)] == ["AAPL", "AAPL"]


def test_cli_without_releases(tmp_path, capsys) -> None:
    argv = ["--releases", str(tmp_path / "none"), "--out", str(tmp_path / "links.csv")]
    assert link.main(argv, client=make_client(tmp_path / "raw")) == 1
    assert "error: no saved releases in" in capsys.readouterr().err
    assert not (tmp_path / "links.csv").exists()


# Hand-checked links (tests/fixtures/links_handchecked.csv)


def handchecked_rows() -> list:
    if not HANDCHECKED.exists():
        return [pytest.param(None, id="file-missing")]
    with HANDCHECKED.open(newline="", encoding="utf-8") as f:
        return [
            pytest.param(row, id=f"{row['ticker']}-{row['release_accession']}")
            for row in csv.DictReader(f)
        ]


def test_handchecked_file_is_well_formed() -> None:
    assert HANDCHECKED.exists(), f"{HANDCHECKED} is missing"
    with HANDCHECKED.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == HANDCHECKED_COLUMNS
    assert rows
    for row in rows:
        assert row["expected_status"] in {"linked", "excluded"}, row
        if row["expected_status"] == "excluded":
            linked = [row[c] for c in HANDCHECKED_COLUMNS[3:]]
            assert linked == ["", "", ""], row


@pytest.mark.parametrize("expected", handchecked_rows())
def test_handchecked_link(expected, fixture_run) -> None:
    assert expected is not None, f"{HANDCHECKED} is missing"
    _, _, rows, _ = fixture_run
    accession = expected["release_accession"].strip()
    got = next((row for row in rows if row["release_accession"] == accession), None)
    assert got is not None, f"{accession} is not in tests/fixtures/link_releases.json"
    assert {
        "ticker": got["ticker"],
        "status": got["status"],
        "linked_form": got["linked_form"],
        "linked_accession": got["linked_accession"],
        "period_end": got["period_end"],
    } == {
        "ticker": expected["ticker"].strip(),
        "status": expected["expected_status"].strip(),
        "linked_form": expected["expected_linked_form"].strip(),
        "linked_accession": expected["expected_linked_accession"].strip(),
        "period_end": expected["expected_period_end"].strip(),
    }, got["reason"]
