import json
import logging
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest
from bs4 import BeautifulSoup

from secbench import edgar, releases
from secbench.edgar import EdgarClient, EdgarError, RateLimiter, parse_submissions
from secbench.releases import (
    Document,
    choose_release,
    collect_releases,
    ex99_exhibits,
    exhibit_file_names,
    has_item,
    html_to_markdown,
    parse_filing_index,
    save_filing,
    select_releases,
)

FIXTURES = Path(__file__).parent / "fixtures"
UA = "Test Runner test@example.com"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/320193/"
Q3_ACCESSION = "0000320193-26-000018"
Q3_FOLDER = ARCHIVE + "000032019326000018/"
Q3_INDEX_URL = Q3_FOLDER + "0000320193-26-000018-index.htm"
Q3_EXHIBIT_URL = Q3_FOLDER + "a8-kex991q3202606272026.htm"
Q2_INDEX_URL = ARCHIVE + "000032019326000011/0000320193-26-000011-index.htm"
FIXTURE_ROUTES = {
    edgar.TICKERS_URL: "company_tickers.json",
    "https://data.sec.gov/submissions/CIK0000320193.json": "CIK0000320193.json",
    "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json": (
        "CIK0000320193-submissions-001.json"
    ),
    Q3_INDEX_URL: "0000320193-26-000018-index.htm",
    Q3_EXHIBIT_URL: "a8-kex991q3202606272026.htm",
}
FAKE_ACCESSION = "0000320193-26-000099"
FAKE_FOLDER = ARCHIVE + "000032019326000099/"
FAKE_INDEX_URL = FAKE_FOLDER + "0000320193-26-000099-index.htm"

Handler = Callable[[httpx.Request], httpx.Response]


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def make_client(tmp_path: Path, routes: dict[str, bytes | str], seen: list[str]):
    """A client served from routes (URL -> fixture name or body); other URLs 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        body = routes.get(str(request.url))
        if body is None:
            return httpx.Response(404)
        return httpx.Response(
            200, content=fixture(body) if isinstance(body, str) else body
        )

    return EdgarClient(
        UA,
        tmp_path / "raw",
        transport=httpx.MockTransport(handler),
        limiter=RateLimiter(8.0, clock=lambda: 0.0, sleep=lambda s: None),
    )


def index_html(
    rows: list[tuple[str, str, str, str]], folder: str = FAKE_FOLDER
) -> bytes:
    """A filing index shaped like the real one: (seq, description, file, type) rows."""
    path = folder.removeprefix("https://www.sec.gov")
    body = "".join(
        f'<tr><td scope="row">{seq}</td><td scope="row">{desc}</td>'
        f'<td scope="row"><a href="{path}{name}">{name}</a></td>'
        f'<td scope="row">{doc_type}</td><td scope="row">100</td></tr>'
        for seq, desc, name, doc_type in rows
    )
    return (
        '<html><body><table class="tableFile" summary="Document Format Files"><tr>'
        '<th scope="col"><acronym title="Sequence Number">Seq</acronym></th>'
        '<th scope="col">Description</th><th scope="col">Document</th>'
        f'<th scope="col">Type</th><th scope="col">Size</th></tr>{body}</table>'
        "</body></html>"
    ).encode()


def doc(doc_type: str, name: str = "x.htm", seq: str = "2") -> Document:
    return Document(seq, doc_type, name, doc_type, FAKE_FOLDER + name)


def table_lines(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("|")]


# Selecting filings


def test_has_item() -> None:
    assert has_item("2.02,9.01", "2.02")
    assert has_item("9.01, 2.02", "2.02")
    assert not has_item("2", "2.02")  # pre-2004 numbering
    assert not has_item("5.02,9.01", "2.02")
    assert not has_item("", "2.02")


def test_select_releases_on_submissions_fixture() -> None:
    filings = parse_submissions(
        [
            json.loads(fixture("CIK0000320193.json"))["filings"]["recent"],
            json.loads(fixture("CIK0000320193-submissions-001.json")),
        ]
    )
    selected = select_releases(filings, date(2025, 4, 1))
    assert list(selected["accession_number"]) == [
        "0000320193-26-000018",
        "0000320193-26-000011",
    ]
    older = select_releases(filings, date(1999, 1, 1), until=date(2026, 4, 30))
    assert list(older["accession_number"]) == [
        "0000320193-26-000011",  # until is inclusive
        "0001193125-15-258464",
    ]  # the 1999 8-K lists item "2", not "2.02"


def test_select_releases_takes_only_8k() -> None:
    filings = pd.DataFrame(
        {
            "accession_number": ["a", "b", "c", "d", "e"],
            "form": ["8-K", "8-K/A", "8-K", "10-Q", "8-K"],
            "filing_date": pd.to_datetime(
                ["2025-04-01", "2025-05-01", "2025-05-01", "2025-05-01", "2025-03-31"]
            ),
            "items": ["2.02,9.01", "2.02", "7.01,9.01", "", "2.02"],
        }
    )
    assert list(select_releases(filings, date(2025, 4, 1))["accession_number"]) == ["a"]


# Filing index


def test_parse_real_filing_index() -> None:
    documents = parse_filing_index(fixture("0000320193-26-000018-index.htm"), Q3_FOLDER)
    assert documents == [
        Document(
            "1", "8-K", "aapl-20260730.htm", "8-K", Q3_FOLDER + "aapl-20260730.htm"
        ),
        Document(
            "2", "EX-99.1", "a8-kex991q3202606272026.htm", "EX-99.1", Q3_EXHIBIT_URL
        ),
        Document(
            "7",
            "",
            "aapl-20260730_g1.jpg",
            "GRAPHIC",
            Q3_FOLDER + "aapl-20260730_g1.jpg",
        ),
        Document(
            "",
            "Complete submission text file",
            "0000320193-26-000018.txt",
            "",
            Q3_FOLDER + "0000320193-26-000018.txt",
        ),
    ]


def test_parse_filing_index_rejects_links_outside_the_filing() -> None:
    html = index_html(
        [("2", "EX-99.1", "ex991.htm", "EX-99.1")], ARCHIVE + "000000000/"
    )
    with pytest.raises(EdgarError, match="outside the filing"):
        parse_filing_index(html, FAKE_FOLDER)


def test_parse_filing_index_needs_the_document_table() -> None:
    with pytest.raises(EdgarError, match="no 'Document Format Files' table"):
        parse_filing_index(b"<html><body>Not found</body></html>", FAKE_FOLDER)


# Choosing exhibits


@pytest.mark.parametrize(
    ("types", "chosen"),
    [
        (["8-K", "EX-99.1", "EX-99.2"], "EX-99.1"),
        (["8-K", "EX-99", "EX-99.1"], "EX-99.1"),
        (["8-K", "EX-99.2", "EX-99.01"], "EX-99.01"),
        (["8-K", "EX-99"], "EX-99"),
        (["8-K", "EX-99.10", "EX-99.2"], None),
        (["8-K", "EX-104"], None),
    ],
)
def test_choose_release(types, chosen) -> None:
    release = choose_release([doc(t, f"{i}.htm") for i, t in enumerate(types)])
    assert (release.type if release else None) == chosen


def test_ex99_exhibits() -> None:
    types = ["8-K", "EX-99.1", "EX-99", "EX-99.01", "EX-990", "EX-101.INS", "EX-99.2"]
    found = ex99_exhibits([doc(t) for t in types])
    assert [d.type for d in found] == ["EX-99.1", "EX-99", "EX-99.01", "EX-99.2"]


def test_exhibit_file_names() -> None:
    exhibits = [
        doc("EX-99.01", seq="2"),
        doc("EX-99.2", seq="3"),
        doc("EX-99.3", "a.htm", seq="4"),
        doc("EX-99.3", "b.htm", seq="5"),
        doc("EX-99.(a)", seq="6"),
    ]
    assert exhibit_file_names("acc", exhibits, exhibits[0]) == [
        "acc.md",
        "acc_EX-99.2.md",
        "acc_EX-99.3_4.md",
        "acc_EX-99.3_5.md",
        "acc_EX-99.-a-.md",
    ]


# HTML to markdown: table rules

SPLIT_TABLE = """
<table>
<tr><td></td><td colspan="7">Unaudited</td></tr>
<tr><td></td><td colspan="3">Three Months Ended</td><td></td><td colspan="3">Year Ended</td></tr>
<tr><td></td><td colspan="3">June 30,<br/>2026</td><td></td><td colspan="3">June 30, 2026</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td colspan="8">Segment results:</td></tr>
<tr><td>Revenue</td><td>$</td><td>1,234</td><td></td><td></td><td>$</td><td>5,678</td><td></td></tr>
<tr><td>Net loss</td><td colspan="2">(12</td><td>)</td><td></td><td colspan="2">(345</td><td>)</td></tr>
<tr><td>Margin</td><td colspan="2">12.5</td><td>%</td><td></td><td colspan="2">&#8212;</td><td></td></tr>
</table>
"""


def test_table_joins_split_numbers_and_merges_headers() -> None:
    assert html_to_markdown(SPLIT_TABLE.encode()) == (
        "|  | Unaudited Three Months Ended June 30, 2026 | Unaudited Year Ended June 30, 2026 |\n"
        "| --- | --- | --- |\n"
        "| Segment results: |  |  |\n"
        "| Revenue | $1,234 | $5,678 |\n"
        "| Net loss | (12) | (345) |\n"
        "| Margin | 12.5% | — |\n"
    )


def test_table_with_year_headers_and_text_kept_as_written() -> None:
    html = """
    <table>
    <tr><th>(In millions)</th><th>2026</th><th>2025</th></tr>
    <tr><td>Revenue*</td><td>1,234&nbsp;</td><td>(56)</td></tr>
    <tr><td>Other <!-- note -->A|B</td><td>&#8211;</td><td>n/m</td></tr>
    </table>
    """
    assert html_to_markdown(html.encode()) == (
        "| (In millions) | 2026 | 2025 |\n"
        "| --- | --- | --- |\n"
        "| Revenue* | 1,234 | (56) |\n"
        "| Other A\\|B | – | n/m |\n"
    )


def test_table_title_row_above_header_stays_in_body() -> None:
    # Shaped like NVDA's summary tables: a "GAAP" title row, then the header row.
    html = """
    <table>
    <tr><td>GAAP</td><td></td><td></td></tr>
    <tr><td>($ in millions)</td><td>Q2 FY27</td><td>Q1 FY27</td></tr>
    <tr><td>Revenue</td><td>$96,221</td><td>$81,615</td></tr>
    </table>
    """
    assert html_to_markdown(html.encode()) == (
        "| ($ in millions) | Q2 FY27 | Q1 FY27 |\n"
        "| --- | --- | --- |\n"
        "| GAAP |  |  |\n"
        "| Revenue | $96,221 | $81,615 |\n"
    )


def test_rowspan_label_and_page_titles_above_header() -> None:
    # Shaped like XOM's EX-99.2: page titles inside the table, then a label cell
    # spanning both header rows.
    html = """
    <table>
    <tr><td></td><td colspan="2">ATTACHMENT I-a</td><td colspan="2">EXHIBIT 99.2</td></tr>
    <tr><td colspan="5">CONDENSED CONSOLIDATED STATEMENT OF INCOME</td></tr>
    <tr><td rowspan="2">Dollars in millions</td><td colspan="2">Three Months Ended</td>
        <td colspan="2">Six Months Ended</td></tr>
    <tr><td>2026</td><td>2025</td><td>2026</td><td>2025</td></tr>
    <tr><td>Revenues and other income</td><td></td><td></td><td></td><td></td></tr>
    <tr><td>Sales</td><td>81,511</td><td>81,058</td><td>164,629</td><td>164,349</td></tr>
    </table>
    """
    assert html_to_markdown(html.encode()) == (
        "| Dollars in millions | Three Months Ended 2026 | Three Months Ended 2025 "
        "| Six Months Ended 2026 | Six Months Ended 2025 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | ATTACHMENT I-a |  | EXHIBIT 99.2 |  |\n"
        "| CONDENSED CONSOLIDATED STATEMENT OF INCOME |  |  |  |  |\n"
        "| Revenues and other income |  |  |  |  |\n"
        "| Sales | 81,511 | 81,058 | 164,629 | 164,349 |\n"
    )


def test_figure_spanning_columns_is_written_once() -> None:
    # Shaped like XOM's segment table: one worldwide figure spans U.S. and Non-U.S.
    html = """
    <table>
    <tr><td></td><td colspan="2">Upstream</td><td colspan="2">Energy</td><td>Total</td></tr>
    <tr><td>Dollars in millions</td><td>U.S.</td><td>Non-U.S.</td><td>U.S.</td>
        <td>Non-U.S.</td><td></td></tr>
    <tr><td>Earnings</td><td>1,920</td><td>6,007</td><td>2,987</td><td>2,478</td>
        <td>13,392</td></tr>
    <tr><td>Timing effects</td><td colspan="2">(63)</td><td colspan="2">2,546</td>
        <td>2,483</td></tr>
    <tr><td>Outlook</td><td colspan="4">Increase 3.5% to 4.5%</td><td></td></tr>
    </table>
    """
    assert html_to_markdown(html.encode()) == (
        "| Dollars in millions | Upstream U.S. | Upstream Non-U.S. | Energy U.S. "
        "| Energy Non-U.S. | Total |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Earnings | 1,920 | 6,007 | 2,987 | 2,478 | 13,392 |\n"
        "| Timing effects | (63) |  | 2,546 |  | 2,483 |\n"
        "| Outlook | Increase 3.5% to 4.5% |  |  |  |  |\n"
    )


def test_footnote_superscripts_are_marked() -> None:
    # ORCL raises footnote marks with vertical-align:super; AAPL's Workiva HTML
    # with a negative top offset. Unmarked, "$2.11" + "1" would read "$2.111".
    html = (
        '<div>EPS of $2.11<font style="vertical-align:super;font-size:7pt">1</font>.'
        "</div>"
        '<div>adjusted basis.<font style="font-size:5.85pt;position:relative;'
        'top:-3.15pt;vertical-align:baseline">2</font></div>'
        '<div>Not raised: <font style="position:relative;margin-top:-2pt">3</font></div>'
        "<table><tr><td></td><td>Q4</td></tr><tr><td>EPS<sup>(a)</sup></td>"
        '<td>$2.11<font style="vertical-align: super">1</font></td></tr></table>'
    )
    assert html_to_markdown(html.encode()) == (
        "EPS of $2.11<sup>1</sup>.\n\n"
        "adjusted basis.<sup>2</sup>\n\n"
        "Not raised: 3\n\n"
        "|  | Q4 |\n"
        "| --- | --- |\n"
        "| EPS<sup>(a)</sup> | $2.11<sup>1</sup> |\n"
    )


def test_table_without_header_rows_gets_blank_header() -> None:
    html = "<table><tr><td>Revenue</td><td>$1,234</td></tr><tr><td>Cost</td><td>99</td></tr></table>"
    assert html_to_markdown(html.encode()) == (
        "|  |  |\n| --- | --- |\n| Revenue | $1,234 |\n| Cost | 99 |\n"
    )


def test_one_column_table_becomes_paragraphs() -> None:
    html = (
        "<table><tr><td>Contacts:</td><td></td></tr><tr><td>Jane Doe</td></tr></table>"
    )
    assert html_to_markdown(html.encode()) == "Contacts:\n\nJane Doe\n"


def test_hidden_content_and_sgml_wrapper_removed() -> None:
    html = (
        b"<DOCUMENT>\n<TYPE>EX-99.1\n<SEQUENCE>2\n<FILENAME>ex991.htm\n<TEXT>\n"
        b"<html><head><title>Document</title></head><body>"
        b'<div style="display: none"><ix:header>hidden facts</ix:header></div>'
        b'<img src="logo.jpg"><div>Revenue was $1.2 billion.</div>'
        b"<script>var x = 1;</script></body></html>\n</TEXT>\n</DOCUMENT>\n"
    )
    assert html_to_markdown(html) == "Revenue was $1.2 billion.\n"


# HTML to markdown: real AAPL releases

NUMBER = re.compile(r"\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?\d+\.\d+\)?")


@pytest.mark.parametrize(
    "name", ["a8-kex991q3202606272026.htm", "a8-kex991q4202509272025.htm"]
)
def test_real_release_keeps_every_number_in_order(name) -> None:
    html = fixture(name)
    soup = BeautifulSoup(releases.strip_sgml(html), "lxml")
    for tag in soup.find_all(["script", "style", "head"]):
        tag.decompose()
    source = NUMBER.findall(soup.get_text(" "))
    assert len(source) > 200
    assert NUMBER.findall(html_to_markdown(html)) == source


def test_q4_release_three_month_and_full_year_columns() -> None:
    markdown = html_to_markdown(fixture("a8-kex991q4202509272025.htm"))
    lines = table_lines(markdown)
    header = (
        "|  | Three Months Ended September 27, 2025 | Three Months Ended September 28, "
        "2024 | Twelve Months Ended September 27, 2025 | Twelve Months Ended "
        "September 28, 2024 |"
    )
    assert lines[0] == header
    assert lines[1] == "| --- | --- | --- | --- | --- |"
    for row in [
        "| Products | $73,716 | $69,958 | $307,003 | $294,866 |",
        "| Total net sales <sup>(1)</sup> | 102,466 | 94,930 | 416,161 | 391,035 |",
        "| Other income/(expense), net | 377 | 19 | (321) | 269 |",
        "| Net income | $27,466 | $14,736 | $112,010 | $93,736 |",
        "| Diluted | $1.85 | $0.97 | $7.46 | $6.08 |",
    ]:
        assert row in lines
    units = "(In millions, except number of shares, which are reflected in thousands, and per-share amounts)"
    assert f"{units}\n\n{header}\n" in markdown

    assert "|  | September 27, 2025 | September 28, 2024 |" in lines  # balance sheet
    assert "| Accumulated deficit | (14,264) | (19,154) |" in lines
    assert "| Proceeds from issuance of term debt, net | 4,481 | — |" in lines


def test_q3_release_three_and_nine_month_columns() -> None:
    lines = table_lines(html_to_markdown(fixture("a8-kex991q3202606272026.htm")))
    assert lines[0] == (
        "|  | Three Months Ended June 27, 2026 | Three Months Ended June 28, 2025 | "
        "Nine Months Ended June 27, 2026 | Nine Months Ended June 28, 2025 |"
    )
    assert "| Other income/(expense), net | 572 | (171) | 670 | (698) |" in lines
    assert "| Diluted | $2.02 | $1.57 | $6.88 | $5.62 |" in lines


# Saving


def test_collect_releases_saves_release_and_logs_skips(tmp_path, caplog) -> None:
    seen: list[str] = []
    client = make_client(tmp_path, FIXTURE_ROUTES, seen)
    out = tmp_path / "releases"
    with caplog.at_level(logging.WARNING, logger="secbench.releases"):
        ciks, outcomes = collect_releases(client, "aapl", date(2025, 4, 1), out_dir=out)
    assert ciks == [320193]

    reason = f"filing index: GET {Q2_INDEX_URL} returned HTTP 404"
    assert [(o.accession_number, o.skip_reason) for o in outcomes] == [
        ("0000320193-26-000011", reason),
        (Q3_ACCESSION, None),
    ]
    assert outcomes[1].release_type == "EX-99.1"
    assert outcomes[1].release_has_table
    assert [r.getMessage() for r in caplog.records] == [
        f"skip AAPL 0000320193-26-000011 (2026-04-30): {reason}"
    ]

    folder = out / "0000320193"
    assert sorted(p.name for p in folder.iterdir()) == [
        f"{Q3_ACCESSION}.json",
        f"{Q3_ACCESSION}.md",
    ]
    assert (folder / f"{Q3_ACCESSION}.md").read_text() == html_to_markdown(
        fixture("a8-kex991q3202606272026.htm")
    )
    assert json.loads((folder / f"{Q3_ACCESSION}.json").read_text()) == {
        "ticker": "AAPL",
        "cik": 320193,
        "filing_date": "2026-07-30",
        "accession_number": Q3_ACCESSION,
        "exhibit": "EX-99.1",
        "exhibit_url": Q3_EXHIBIT_URL,
        "exhibits": [
            {
                "type": "EX-99.1",
                "description": "EX-99.1",
                "url": Q3_EXHIBIT_URL,
                "file": f"{Q3_ACCESSION}.md",
                "chosen": True,
            }
        ],
    }

    first_run = len(seen)
    collect_releases(client, "AAPL", date(2025, 4, 1), out_dir=out)
    assert seen[first_run:] == [Q2_INDEX_URL]  # only the failed request is retried


def submissions_json(rows: list[tuple[str, str, str, str]]) -> bytes:
    """A submissions JSON shaped like SEC's: (accession, form, date, items) rows."""
    recent = {
        "accessionNumber": [row[0] for row in rows],
        "form": [row[1] for row in rows],
        "filingDate": [row[2] for row in rows],
        "reportDate": [row[2] for row in rows],
        "items": [row[3] for row in rows],
        "primaryDocument": ["form.htm" for _ in rows],
    }
    return json.dumps({"filings": {"recent": recent, "files": []}}).encode()


def test_collect_releases_reads_every_cik_of_a_ticker(tmp_path) -> None:
    # Made-up filings under XOM's two CIKs (edgar.TICKER_CIKS): one release under
    # each, and one listed under both, which is saved once, under the first CIK.
    new, old, joint = (
        "0002115436-26-900001",
        "0000034088-26-900002",
        "0002115436-26-900003",
    )
    holding = "https://www.sec.gov/Archives/edgar/data/2115436/"
    predecessor = "https://www.sec.gov/Archives/edgar/data/34088/"
    folders = {
        new: holding + "000211543626900001/",
        old: predecessor + "000003408826900002/",
        joint: holding + "000211543626900003/",
    }
    routes: dict[str, bytes | str] = {
        edgar.TICKERS_URL: "company_tickers.json",
        "https://data.sec.gov/submissions/CIK0002115436.json": submissions_json(
            [
                (joint, "8-K", "2026-08-20", "2.02"),
                (new, "8-K", "2026-07-31", "2.02,7.01"),
                (old, "10-Q", "2026-05-02", ""),  # not a release
            ]
        ),
        "https://data.sec.gov/submissions/CIK0000034088.json": submissions_json(
            [
                (joint, "8-K", "2026-08-20", "2.02"),
                (old, "8-K", "2026-05-01", "2.02,7.01"),
            ]
        ),
    }
    for accession, folder in folders.items():
        index = index_html([("1", "EX-99.1", "ex991.htm", "EX-99.1")], folder)
        routes[f"{folder}{accession}-index.htm"] = index
        routes[folder + "ex991.htm"] = f"<p>Release {accession}</p>".encode()
    seen: list[str] = []
    client = make_client(tmp_path, routes, seen)
    out = tmp_path / "releases"

    ciks, outcomes = collect_releases(client, "XOM", date(2025, 4, 1), out_dir=out)
    assert ciks == [2115436, 34088]
    assert [(o.accession_number, o.skip_reason) for o in outcomes] == [
        (old, None),
        (new, None),
        (joint, None),
    ]
    assert not any("/34088/000211543626900003/" in url for url in seen)

    for accession, cik_folder, cik in [
        (old, "0000034088", 34088),
        (new, "0002115436", 2115436),
        (joint, "0002115436", 2115436),
    ]:
        record = json.loads((out / cik_folder / f"{accession}.json").read_text())
        assert (record["ticker"], record["cik"]) == ("XOM", cik)
        text = (out / cik_folder / f"{accession}.md").read_text()
        assert text == f"Release {accession}\n"
    assert releases.format_report("XOM", ciks, outcomes).splitlines()[0] == (
        "XOM (CIK 0002115436, 0000034088): 3 Item 2.02 8-Ks, 3 saved, 0 skipped"
    )


def test_save_filing_saves_every_ex99_exhibit(tmp_path) -> None:
    index = index_html(
        [
            ("1", "8-K", "form8k.htm", "8-K"),
            ("2", "PRESS RELEASE", "ex9901.htm", "EX-99.01"),
            ("3", "CFO COMMENTARY", "ex9902.htm", "EX-99.2"),
            ("4", "SLIDES", "ex9903.pdf", "EX-99.3"),
            ("5", "", "logo.jpg", "GRAPHIC"),
        ]
    )
    routes = {
        FAKE_INDEX_URL: index,
        FAKE_FOLDER + "ex9901.htm": b"<p>Revenue was $1.2 billion.</p>",
        FAKE_FOLDER + "ex9902.htm": b"<p>Gross margin was 40%.</p>",
    }
    seen: list[str] = []
    client = make_client(tmp_path, routes, seen)
    out = tmp_path / "releases"
    outcome = save_filing(client, "AAPL", 320193, FAKE_ACCESSION, date(2026, 8, 1), out)
    assert outcome.skip_reason is None
    assert outcome.release_type == "EX-99.01"
    assert outcome.other_types == ["EX-99.2", "EX-99.3"]
    assert not outcome.release_has_table
    assert seen == [
        FAKE_INDEX_URL,
        FAKE_FOLDER + "ex9901.htm",
        FAKE_FOLDER + "ex9902.htm",
    ]

    folder = out / "0000320193"
    assert (
        folder / f"{FAKE_ACCESSION}.md"
    ).read_text() == "Revenue was $1.2 billion.\n"
    assert (
        folder / f"{FAKE_ACCESSION}_EX-99.2.md"
    ).read_text() == "Gross margin was 40%.\n"
    assert sorted(p.name for p in folder.iterdir()) == [
        f"{FAKE_ACCESSION}.json",
        f"{FAKE_ACCESSION}.md",
        f"{FAKE_ACCESSION}_EX-99.2.md",
    ]
    record = json.loads((folder / f"{FAKE_ACCESSION}.json").read_text())
    assert record["exhibit"] == "EX-99.01"
    assert record["exhibit_url"] == FAKE_FOLDER + "ex9901.htm"
    assert record["exhibits"] == [
        {
            "type": "EX-99.01",
            "description": "PRESS RELEASE",
            "url": FAKE_FOLDER + "ex9901.htm",
            "file": f"{FAKE_ACCESSION}.md",
            "chosen": True,
        },
        {
            "type": "EX-99.2",
            "description": "CFO COMMENTARY",
            "url": FAKE_FOLDER + "ex9902.htm",
            "file": f"{FAKE_ACCESSION}_EX-99.2.md",
            "chosen": False,
        },
        {
            "type": "EX-99.3",
            "description": "SLIDES",
            "url": FAKE_FOLDER + "ex9903.pdf",
            "file": None,
            "chosen": False,
        },
    ]


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        (
            [
                ("1", "8-K", "f.htm", "8-K"),
                ("2", "", "e2.htm", "EX-99.2"),
                ("3", "", "c.htm", "EX-104"),
            ],
            "no EX-99.1 exhibit (exhibits: EX-99.2, EX-104)",
        ),
        ([("1", "8-K", "f.htm", "8-K")], "no EX-99.1 exhibit (exhibits: none)"),
        (
            [("1", "8-K", "f.htm", "8-K"), ("2", "", "ex991.pdf", "EX-99.1")],
            "EX-99.1 is ex991.pdf, not HTML",
        ),
        (
            [("1", "8-K", "f.htm", "8-K"), ("2", "", "gone.htm", "EX-99.1")],
            f"EX-99.1: GET {FAKE_FOLDER}gone.htm returned HTTP 404",
        ),
    ],
)
def test_save_filing_skip_reasons(tmp_path, rows, reason) -> None:
    client = make_client(tmp_path, {FAKE_INDEX_URL: index_html(rows)}, [])
    out = tmp_path / "releases"
    outcome = save_filing(client, "AAPL", 320193, FAKE_ACCESSION, date(2026, 8, 1), out)
    assert outcome.skip_reason == reason
    assert not out.exists()


# Command line


def test_cli_reports_saved_and_skipped(tmp_path, capsys) -> None:
    client = make_client(tmp_path, FIXTURE_ROUTES, [])
    out = tmp_path / "releases"
    argv = ["aapl", "--since", "2025-04-01", "--out", str(out)]
    assert releases.main(argv, client=client) == 0
    assert capsys.readouterr().out.splitlines() == [
        "AAPL (CIK 0000320193): 2 Item 2.02 8-Ks, 1 saved, 1 skipped",
        "  press release: EX-99.1 x1",
        "  other EX-99:   none",
        (
            "  skipped 0000320193-26-000011 (2026-04-30): filing index: "
            f"GET {Q2_INDEX_URL} returned HTTP 404"
        ),
    ]
    assert (out / "0000320193" / f"{Q3_ACCESSION}.md").exists()


def test_cli_continues_after_unknown_ticker(tmp_path, capsys) -> None:
    client = make_client(tmp_path, FIXTURE_ROUTES, [])
    argv = ["ZZZZ", "AAPL", "--since", "2026-07-01", "--out", str(tmp_path / "out")]
    assert releases.main(argv, client=client) == 1
    captured = capsys.readouterr()
    assert "error: ZZZZ: ticker 'ZZZZ' is not in" in captured.err
    assert captured.out.splitlines()[0] == (
        "AAPL (CIK 0000320193): 1 Item 2.02 8-Ks, 1 saved, 0 skipped"
    )
