"""Tests for secbench.labels: stage 2 training labels from the XBRL answers.

Expected labels were read by hand from each release's markdown (the row and column
are noted with each); the fixture releases are copies of real ones (see
tests/fixtures/README.md).
"""

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from secbench import extract, grade, labels
from secbench.answers import Answer
from secbench.edgar import EdgarClient, RateLimiter
from secbench.inputs import build_input

FIXTURES = Path(__file__).parent / "fixtures"
LABEL_RELEASES = FIXTURES / "label_releases"
INPUT_RELEASES = FIXTURES / "input_releases"
LABEL_ANSWERS = FIXTURES / "label_answers.csv"
LINKS = FIXTURES / "answers_links.csv"
COMPANYFACTS = FIXTURES / "answers_companyfacts.json"
SUBMISSIONS = FIXTURES / "link_submissions.json"
PROMPTS = Path(__file__).parents[1] / "prompts"
UA = "Test Runner test@example.com"
D = Decimal

AAPL = "0000320193-25-000055"  # Q2 FY25 (stage 1)
MSFT = "0000950170-25-100226"  # FY25, a 10-K (stage 1)
WMT = "0000104169-25-000069"  # Q1 FY26 (stage 1)
CAT = "0000018230-25-000037"  # Q2 2025 (stage 1)
XOM = "0002115436-26-000006"  # Q2 2026 (stage 1)
DOV = "0000029905-25-000003"  # Q4 2024: EPS rows under "Diluted earnings per share:"
VLO = "0001035002-25-000002"  # Q4 2024: basic and diluted EPS equal for the year
COP = "0001163165-24-000004"  # Q4 2023: "2023 4th Qtr" columns, in EX-99.2
FSLR = "0001274494-25-000009"  # Q4 2024: three three-month columns
BALL = "0000009389-26-000039"  # Q4 2025: continuing 0.74, total 0.75
ABT = "0001628280-24-002009"  # Q4 2023: quarter and full-year EPS in separate tables
SWK = "0000093556-23-000028"  # Q3 2023: millions to one decimal, in EX-99.2
AMCR = "0001748790-23-000034"  # Q1 FY24: EPS to three decimals
EXHIBIT = {"type": "EX-99.1", "url": "https://www.sec.gov/test.htm"}


@pytest.fixture(scope="module")
def key() -> dict[str, dict[tuple[str, str], Answer]]:
    """label_answers.csv (rows of data/stage2/answers.csv) by release."""
    by_release: dict[str, dict[tuple[str, str], Answer]] = defaultdict(dict)
    for (accession, field, period), a in grade.load_answers(LABEL_ANSWERS).items():
        by_release[accession][field, period] = a
    return by_release


def record_of(accession: str) -> Path:
    found = [
        *LABEL_RELEASES.glob(f"*/{accession}.json"),
        *INPUT_RELEASES.glob(f"*/{accession}.json"),
    ]
    assert len(found) == 1, accession
    return found[0]


def located_of(accession: str) -> list[labels.Located]:
    record = record_of(accession)
    text = build_input(record).text
    return labels.locate(text, labels.exhibit_starts(record, text))


def train_release(
    accession: str, form: str, period_end: date, ticker: str = "TEST"
) -> labels.TrainRelease:
    return labels.TrainRelease(
        ticker, accession, date(2025, 1, 30), form, "linked", period_end, "train",
        "train", False,
    )  # fmt: skip


def label_fixture(
    key, accession: str, form: str, period_end: date, year_ends=frozenset()
) -> labels.Result:
    return labels.label_release(
        train_release(accession, form, period_end),
        key[accession],
        located_of(accession),
        (),
        lambda: set(year_ends),
    )


def summary(result: labels.Result) -> list[tuple[str, ...]]:
    """(group, field, value, unit, source, row, column) of each label."""
    return [
        (
            label.group,
            label.field,
            str(label.value),
            label.unit,
            label.source,
            label.cell.row.label,
            label.cell.column_header,
        )
        for label in result.labels
    ]


# Stage 1 fixture releases


def test_second_quarter_labels(key) -> None:
    # Apple's Q2 FY25 condensed consolidated statements of operations, in millions.
    result = label_fixture(key, AAPL, "10-Q", date(2025, 3, 29))
    q, ytd = "Three Months Ended March 29, 2025", "Six Months Ended March 29, 2025"
    assert result.kind == labels.MIDDLE
    assert summary(result) == [
        ("quarter", "revenue", "95359", "millions", "cell", "Total net sales", q),
        ("quarter", "net_income", "24780", "millions", "cell", "Net income", q),
        ("quarter", "eps_diluted", "1.65", "", "cell", "Diluted", q),
        ("year_to_date", "revenue", "219659", "millions", "cell", "Total net sales", ytd),
        ("year_to_date", "net_income", "61110", "millions", "cell", "Net income", ytd),
        ("year_to_date", "eps_diluted", "4.05", "", "cell", "Diluted", ytd),
    ]  # fmt: skip
    assert labels.reply(date(2025, 3, 29), result.labels) == (
        '{"period_end": "2025-03-29", "quarter": {"revenue": {"value": 95359, "unit": '
        '"millions"}, "net_income": {"value": 24780, "unit": "millions"}, '
        '"eps_diluted": 1.65}, "year_to_date": {"revenue": {"value": 219659, "unit": '
        '"millions"}, "net_income": {"value": 61110, "unit": "millions"}, '
        '"eps_diluted": 4.05}}'
    )


def test_fourth_quarter_labels(key) -> None:
    # Microsoft's FY25 income statements: Diluted $3.65 beside $13.64 for the year,
    # in the column where net income is $27,233 (the derived fourth quarter).
    result = label_fixture(key, MSFT, "10-K", date(2025, 6, 30))
    q, fy = "Three Months Ended June 30, 2025", "Twelve Months Ended June 30, 2025"
    assert result.kind == labels.FOURTH
    assert summary(result) == [
        ("quarter", "revenue", "76441", "millions", "cell", "Total revenue", q),
        ("quarter", "net_income", "27233", "millions", "cell", "Net income", q),
        ("quarter", "eps_diluted", "3.65", "", "fourth_quarter_eps", "Diluted", q),
        ("year_to_date", "revenue", "281724", "millions", "cell", "Total revenue", fy),
        ("year_to_date", "net_income", "101832", "millions", "cell", "Net income", fy),
        ("year_to_date", "eps_diluted", "13.64", "", "cell", "Diluted", fy),
    ]  # fmt: skip
    eps = result.labels[2]
    assert eps.answer is None
    assert eps.anchor == (
        f"full year: 'Diluted' row, {fy!r} column: $13.64; three-month column "
        f"confirmed by 'Net income' row, {q!r} column: $27,233"
    )


def test_first_quarter_year_to_date_is_the_quarter(key) -> None:
    # Walmart's Q1 FY26 income statement; FY25 ended 2025-01-31, the day before.
    result = label_fixture(key, WMT, "10-Q", date(2025, 4, 30), {date(2025, 1, 31)})
    q = "Three Months Ended April 30, 2025"
    net_income = "Consolidated net income attributable to Walmart"
    eps = "Diluted net income per common share attributable to Walmart"
    assert result.kind == labels.FIRST
    assert summary(result) == [
        ("quarter", "revenue", "165609", "millions", "cell", "Total revenues", q),
        ("quarter", "net_income", "4487", "millions", "cell", net_income, q),
        ("quarter", "eps_diluted", "0.56", "", "cell", eps, q),
        ("year_to_date", "revenue", "165609", "millions", "first_quarter", "Total revenues", q),
        ("year_to_date", "net_income", "4487", "millions", "first_quarter", net_income, q),
        ("year_to_date", "eps_diluted", "0.56", "", "first_quarter", eps, q),
    ]  # fmt: skip


def test_first_quarter_needs_a_fiscal_year_end_the_day_before(key) -> None:
    result = label_fixture(key, WMT, "10-Q", date(2025, 4, 30), {date(2024, 1, 31)})
    assert not result.labelled
    assert result.problems == (
        labels.Problem(
            labels.NOT_FIRST_QUARTER,
            "year_to_date",
            "the quarter starts 2025-02-01, and no 10-K's period ends 2025-01-31",
        ),
    )


def test_missing_answer_left_out_with_its_reason(key) -> None:
    answers = dict(key[AAPL])
    del answers["net_income", "year_to_date"]
    result = labels.label_release(
        train_release(AAPL, "10-Q", date(2025, 3, 29)),
        answers,
        located_of(AAPL),
        ["net_income year_to_date: no fact for that accession (NetIncomeLoss: none)"],
    )
    assert result.labels == ()
    assert result.problems == (
        labels.Problem(
            labels.NO_XBRL_ANSWER,
            "net_income year_to_date",
            "no fact for that accession (NetIncomeLoss: none)",
        ),
    )
    assert result.first_reason() == labels.NO_XBRL_ANSWER


def test_missing_year_to_date_is_not_a_first_quarter(key) -> None:
    # Stage 2 expected a year to date (it logged why there is none), so the quarter
    # isn't copied into it.
    answers = {k: a for k, a in key[AAPL].items() if k[1] == "quarter"}
    missing = [f"{f} year_to_date: no matching period (...)" for f in grade.FIELD_NAMES]
    result = labels.label_release(
        train_release(AAPL, "10-Q", date(2025, 3, 29)),
        answers,
        located_of(AAPL),
        missing,
        # A year ending the day before the quarter: only the logged reasons keep
        # the release from counting as a first quarter.
        lambda: {date(2024, 12, 28)},
    )
    assert result.kind == labels.MIDDLE
    assert [(p.reason, p.figure) for p in result.problems] == [
        (labels.NO_XBRL_ANSWER, f"{f} year_to_date") for f in grade.FIELD_NAMES
    ]


def test_answer_not_printed_left_out(key) -> None:
    answers = dict(key[AAPL])
    answers["revenue", "quarter"] = Answer(
        **{**answers["revenue", "quarter"].__dict__, "value": D("95358000000")}
    )
    result = labels.label_release(
        train_release(AAPL, "10-Q", date(2025, 3, 29)), answers, located_of(AAPL)
    )
    assert result.problems == (
        labels.Problem(
            labels.NOT_PRINTED,
            "revenue quarter",
            "answer 95358000000; searched 95,358, 95,358.0, 95,358,000",
        ),
    )


# Train fixture releases


@pytest.mark.parametrize(
    ("accession", "period_end", "eps", "row", "section", "column", "exhibit"),
    [
        # Dover: "Net earnings" under "Diluted earnings per share:", $10.38 beside
        # $19.45 for the year (a gain on discontinued operations in the quarter).
        (
            DOV,
            date(2024, 12, 31),
            "10.38",
            "Net earnings",
            "Diluted earnings per share:",
            "Three Months Ended December 31, 2024",
            "EX-99.1",
        ),
        # Valero: basic and diluted are both $8.58 for the year; the quarter's are
        # $0.89 basic and $0.88 assuming dilution.
        (
            VLO,
            date(2024, 12, 31),
            "0.88",
            "Earnings per common share – assuming dilution",
            "Cost of sales:",  # the last label-only row above it in its table
            "Three Months Ended December 31, 2024",
            "EX-99.01",  # Valero numbers its press release so
        ),
        # ConocoPhillips: quarterly columns "2023 1st Qtr" ... "2023 YTD" in EX-99.2.
        (
            COP,
            date(2023, 12, 31),
            "2.52",
            "Diluted",
            "Net Income Per Share of Common Stock (dollars)",
            "2023 4th Qtr",
            "EX-99.2",
        ),
        # First Solar: Q4 2024, Q3 2024 and Q4 2023 columns; net income $393,116
        # thousand marks the first.
        (
            FSLR,
            date(2024, 12, 31),
            "3.65",
            "Diluted",
            "Net income per share:",
            "Three Months Ended December 31, 2024",
            "EX-99.1",
        ),
    ],
)
def test_fourth_quarter_eps_cell(
    key, accession, period_end, eps, row, section, column, exhibit
) -> None:
    result = label_fixture(key, accession, "10-K", period_end)
    assert result.labelled, result.problems
    label = result.labels[2]
    assert (label.field, label.group, label.source) == (
        "eps_diluted",
        "quarter",
        labels.FOURTH_QUARTER_EPS,
    )
    assert str(label.value) == eps
    assert label.cell.row.label == row
    assert label.cell.row.section == section
    assert label.cell.column_header == column
    assert label.cell.located.place.exhibit == exhibit


def test_dover_labels_in_thousands(key) -> None:
    result = label_fixture(key, DOV, "10-K", date(2024, 12, 31))
    assert labels.reply(date(2024, 12, 31), result.labels) == (
        '{"period_end": "2024-12-31", "quarter": {"revenue": {"value": 1929866, '
        '"unit": "thousands"}, "net_income": {"value": 1435983, "unit": "thousands"}, '
        '"eps_diluted": 10.38}, "year_to_date": {"revenue": {"value": 7745909, "unit": '
        '"thousands"}, "net_income": {"value": 2697126, "unit": "thousands"}, '
        '"eps_diluted": 19.45}}'
    )


def test_valero_full_year_eps_from_the_diluted_row(key) -> None:
    result = label_fixture(key, VLO, "10-K", date(2024, 12, 31))
    label = result.labels[5]
    assert (label.group, str(label.value)) == ("year_to_date", "8.58")
    assert label.cell.row.label == "Earnings per common share – assuming dilution"


def test_rows_that_disagree_are_left_out(key) -> None:
    # Ball: "Diluted - continuing operations" and "Total diluted earnings per share"
    # both show $3.30 for 2025, but $0.74 and $0.75 for the quarter.
    result = label_fixture(key, BALL, "10-K", date(2025, 12, 31))
    assert [p.reason for p in result.problems] == [labels.ROWS_DISAGREE]
    detail = result.problems[0].detail
    assert detail.startswith("0.74 in 'Diluted - continuing operations")
    assert "; 0.75 in 'Total diluted earnings per share" in detail


def test_eps_without_a_confirmed_column_is_left_out(key) -> None:
    # Abbott prints the quarter's and the year's results in separate tables, so the
    # $3.26 row has no three-month column (the quarter's is $0.91 elsewhere).
    result = label_fixture(key, ABT, "10-K", date(2023, 12, 31))
    assert [p.reason for p in result.problems] == [labels.NO_CONFIRMED_COLUMN]
    assert "'Diluted Earnings per Common Share' row, '12M23' column: $3.26" in (
        result.problems[0].detail
    )


def test_millions_to_one_decimal(key) -> None:
    # Stanley Black & Decker's Q3 2023 statements of operations (EX-99.2),
    # "Millions of Dollars Except Per Share Amounts".
    result = label_fixture(key, SWK, "10-Q", date(2023, 9, 30))
    q, ytd = "THIRD QUARTER 2023", "YEAR-TO-DATE 2023"
    continuing = "NET EARNINGS (LOSS) FROM CONTINUING OPERATIONS"
    diluted = "NET EARNINGS (LOSS) ATTRIBUTABLE TO COMMON SHAREOWNERS - DILUTED"
    assert summary(result) == [
        ("quarter", "revenue", "3953.9", "millions", "cell", "NET SALES", q),
        ("quarter", "net_income", "4.7", "millions", "cell", continuing, q),
        ("quarter", "eps_diluted", "0.03", "", "cell", "Continuing operations", q),
        ("year_to_date", "revenue", "12044.6", "millions", "cell", "NET SALES", ytd),
        ("year_to_date", "net_income", "-6.1", "millions", "cell", diluted, ytd),
        ("year_to_date", "eps_diluted", "-0.04", "", "cell", "Continuing operations", ytd),
    ]  # fmt: skip
    assert result.labels[2].cell.row.section == (
        "DILUTED EARNINGS (LOSS) PER SHARE OF COMMON STOCK"
    )
    assert {label.cell.located.place.exhibit for label in result.labels} == {"EX-99.2"}
    text = labels.reply(date(2023, 9, 30), result.labels)
    assert text == (
        '{"period_end": "2023-09-30", "quarter": {"revenue": {"value": 3953.9, "unit": '
        '"millions"}, "net_income": {"value": 4.7, "unit": "millions"}, '
        '"eps_diluted": 0.03}, "year_to_date": {"revenue": {"value": 12044.6, "unit": '
        '"millions"}, "net_income": {"value": -6.1, "unit": "millions"}, '
        '"eps_diluted": -0.04}}'
    )
    parsed = extract.parse_reply(text, SWK, extract.WITH_UNITS)
    assert parsed.prediction.quarter == grade.Figures(
        revenue=D("3953900000"), net_income=D("4700000"), eps_diluted=D("0.03")
    )
    assert parsed.prediction.year_to_date == grade.Figures(
        revenue=D("12044600000"), net_income=D("-6100000"), eps_diluted=D("-0.04")
    )


def test_eps_to_three_decimals_from_the_diluted_row(key) -> None:
    # Amcor's Q1 FY24 statement of income: basic and diluted EPS both 0.105.
    result = label_fixture(key, AMCR, "10-Q", date(2023, 9, 30), {date(2023, 6, 30)})
    assert result.labelled
    eps = result.labels[2]
    assert str(eps.value) == "0.105"
    assert eps.cell.row.label == "Diluted earnings per share attributable to Amcor"
    assert eps.cell.column_header == "Three Months Ended September 30, 2023"
    assert [str(label.value) for label in result.labels] == [
        "3443", "152", "0.105", "3443", "152", "0.105",
    ]  # fmt: skip


# Cells


def locate_text(text: str) -> list[labels.Located]:
    return labels.locate(text, [(EXHIBIT, 0)])


TABLE = """| ($ in millions) | Q3 2025 |
| --- | --- |
| Net sales | $3,953.9 |
| Net sales (thousands) | 3,953,900 |
| Net income | (1,234) |
| Net loss | 1,234 |
| Net earnings | -1,234 |
| Other income | 1,234 |
| Adjusted net income (non-GAAP) | (1,234) |
| Segment sales | 13,953.9 |
"""


def test_answer_cells_by_form_sign_and_context() -> None:
    located = locate_text(TABLE)
    found = labels.answer_cells(D("3953900000"), "revenue", located)
    assert [(c.figure.text, unit) for c, unit in found] == [
        ("$3,953.9", 10**6),
        ("3,953,900", 10**3),
    ]
    found = labels.answer_cells(D("-1234000000"), "net_income", located)
    assert [c.row.label for c, _ in found] == ["Net income", "Net loss", "Net earnings"]
    found = labels.answer_cells(D("1234000000"), "net_income", located)
    assert [c.row.label for c, _ in found] == ["Net loss", "Other income"]


def test_cell_label_keeps_the_printed_decimals() -> None:
    located = locate_text(TABLE)
    answer = synthetic("revenue", "quarter", "3953900000")
    cell, unit = labels.answer_cells(answer.value, "revenue", located)[0]
    label = labels.cell_label(answer, "quarter", cell, unit)
    assert (str(label.value), label.unit, label.raw_value()) == (
        "3953.9",
        "millions",
        D("3953900000"),
    )
    answer = synthetic("net_income", "quarter", "-1234000000")
    cell, unit = labels.answer_cells(answer.value, "net_income", located)[1]
    assert str(labels.cell_label(answer, "quarter", cell, unit).value) == "-1234"


def test_choose_prefers_the_income_statement() -> None:
    text = (
        "| Highlights | Q4 2025 |\n| --- | --- |\n| Revenue ($ millions) | 1,000 |\n\n"
        "| (in thousands) | Three Months Ended December 31, 2025 |\n| --- | --- |\n"
        "| Segment total | 1,000,000 |\n| Total revenues | 1,000,000 |\n"
        "| Net income | 100,000 |\n| Diluted earnings per share | 1.00 |\n"
    )
    located = locate_text(text)
    assert [x.place.income_statement for x in located] == [
        False, True, True, True, True,
    ]  # fmt: skip
    cell, unit = labels.choose(
        labels.answer_cells(D("1000000000"), "revenue", located), "revenue"
    )
    assert (cell.row.label, unit) == ("Total revenues", 10**3)


def test_quoted_drops_zero_width_spaces() -> None:
    # Ball's row labels, as its markdown has them.
    label = "Diluted - continuing operations \u200b \u200b \u200b \u200b"
    assert labels.quoted(label, 60) == "'Diluted - continuing operations'"
    assert labels.quoted("Net income attributable to Ball Corporation", 20) == (
        "'Net income attribut…'"
    )


def test_locate_tables_and_exhibits() -> None:
    # Stanley Black & Decker's input: EX-99.1, then its statements in EX-99.2.
    record = record_of(SWK)
    text = build_input(record).text
    starts = labels.exhibit_starts(record, text)
    assert [(e["type"], offset > 0) for e, offset in starts] == [
        ("EX-99.1", False),
        ("EX-99.2", True),
    ]
    assert text[starts[1][1] :].startswith("# EX-99.2")
    located = labels.locate(text, starts)
    assert len(located) == len(grade.input_rows(text))
    sales = [x for x in located if x.row.label == "NET SALES" and x.row.figures]
    assert sales[0].place.exhibit == "EX-99.2"
    assert sales[0].place.url.endswith(".htm")
    assert sales[0].place.income_statement


def test_exhibit_starts_must_rebuild_the_input() -> None:
    record = record_of(SWK)
    with pytest.raises(labels.LabelError, match="isn't its exhibits joined"):
        labels.exhibit_starts(record, build_input(record).text + "\n")


# Fourth-quarter EPS on small tables


def synthetic(field: str, period: str, value: str) -> Answer:
    unit = "USD/shares" if field == "eps_diluted" else "USD"
    start = date(2025, 10, 1) if period == "quarter" else date(2025, 1, 1)
    return Answer(
        "TEST", "0000000001-26-000001", "0000000001-26-000002", field, period,
        start, date(2025, 12, 31), D(value), unit, "Tag",
    )  # fmt: skip


HEADER = (
    "| (in millions, except per share) | Three Months Ended December 31, 2025 | "
    "Three Months Ended December 31, 2024 | Year Ended December 31, 2025 | "
    "Year Ended December 31, 2024 |\n| --- | --- | --- | --- | --- |\n"
    "| Total revenues | 1,000 | 900 | 4,000 | 3,600 |\n"
)


def q4(
    eps_rows: str = "| Diluted earnings per share | 1.00 | 0.90 | 4.00 | 3.60 |",
    net_income_row: str = "| Net income | 100 | 90 | 400 | 360 |",
    net_income: tuple[str, str] = ("100000000", "400000000"),
    eps: str = "4.00",
) -> tuple[labels.Label | None, list[labels.Problem]]:
    answers = {
        ("revenue", "quarter"): synthetic("revenue", "quarter", "1000000000"),
        ("revenue", "full_year"): synthetic("revenue", "full_year", "4000000000"),
        ("net_income", "quarter"): synthetic("net_income", "quarter", net_income[0]),
        ("net_income", "full_year"): synthetic("net_income", "full_year", net_income[1]),
        ("eps_diluted", "full_year"): synthetic("eps_diluted", "full_year", eps),
    }  # fmt: skip
    located = locate_text(HEADER + net_income_row + "\n" + eps_rows + "\n")
    return labels.fourth_quarter_eps(located, answers)


def reasons(found: tuple[labels.Label | None, list[labels.Problem]]) -> list[str]:
    return [p.reason for p in found[1]]


def test_fourth_quarter_eps_three_month_column() -> None:
    label, problems = q4()
    assert problems == []
    assert label is not None
    assert (str(label.value), label.cell.column_header) == (
        "1.00",
        "Three Months Ended December 31, 2025",
    )


@pytest.mark.parametrize(
    ("printed", "verdict"),
    [
        ("1.20", None),  # 20% above 100 x 4.00 / 400 = 1.00: kept
        ("1.21", labels.FAR_FROM_RATIO),
        ("0.80", None),
        ("0.79", labels.FAR_FROM_RATIO),
    ],
)
def test_fourth_quarter_eps_ratio_check(printed, verdict) -> None:
    found = q4(f"| Diluted earnings per share | {printed} | 0.90 | 4.00 | 3.60 |")
    assert reasons(found) == ([verdict] if verdict else [])


def test_fourth_quarter_eps_sign_check() -> None:
    found = q4("| Diluted earnings per share | (1.00) | 0.90 | 4.00 | 3.60 |")
    assert reasons(found) == [labels.SIGN_DIFFERS]
    assert found[1][0].detail.endswith("the quarter's net income is 100000000")


def test_fourth_quarter_eps_no_ratio() -> None:
    found = q4("| Diluted earnings per share | 1.00 | 0.90 | 0.00 | 3.60 |", eps="0")
    assert reasons(found) == [labels.NO_RATIO]


def test_fourth_quarter_loss_unsigned_on_a_loss_row() -> None:
    label, problems = q4(
        "| Net loss per diluted share | 1.00 | 0.90 | 4.00 | 3.60 |",
        "| Net loss | 100 | 90 | 400 | 360 |",
        ("-100000000", "-400000000"),
        "-4.00",
    )
    assert problems == []
    assert label is not None and str(label.value) == "-1.00"


@pytest.mark.parametrize("cell", ["—", "1.0"])  # no figure; not printed as EPS is
def test_fourth_quarter_eps_without_a_figure_in_the_column(cell) -> None:
    found = q4(f"| Diluted earnings per share | {cell} | 0.90 | 4.00 | 3.60 |")
    assert reasons(found) == [labels.NO_THREE_MONTH_FIGURE]


def test_fourth_quarter_eps_never_the_full_year_column() -> None:
    # Nine months of nothing: the quarter's net income (400) is the year's, so the
    # year's own column prints it too; only the three-month column counts, and its
    # 1.00 is then far from 400 x 4.00 / 400.
    found = q4(net_income_row="| Net income | 400 | 90 | 400 | 360 |",
               net_income=("400000000", "400000000"))  # fmt: skip
    assert reasons(found) == [labels.FAR_FROM_RATIO]


def test_fourth_quarter_eps_from_basic_rows_only() -> None:
    found = q4("| Basic earnings per share | 1.00 | 0.90 | 4.00 | 3.60 |")
    assert reasons(found) == [labels.NO_DILUTED_ROW]
    # A per-share row that says neither basic nor diluted counts when none says
    # diluted.
    label, problems = q4("| Earnings per share | 1.00 | 0.90 | 4.00 | 3.60 |")
    assert problems == [] and label is not None and str(label.value) == "1.00"


def test_fourth_quarter_eps_prefers_diluted_rows() -> None:
    rows = (
        "| Earnings per share | 1.01 | 0.90 | 4.00 | 3.60 |\n"
        "| Earnings per share - assuming dilution | 1.00 | 0.90 | 4.00 | 3.60 |"
    )
    label, problems = q4(rows)
    assert problems == [] and label is not None and str(label.value) == "1.00"


def test_fourth_quarter_eps_rows_disagree() -> None:
    rows = (
        "| Diluted - continuing operations | 0.99 | 0.90 | 4.00 | 3.60 |\n"
        "| Total diluted earnings per share | 1.00 | 0.90 | 4.00 | 3.60 |"
    )
    assert reasons(q4(rows)) == [labels.ROWS_DISAGREE]


def test_fourth_quarter_eps_in_another_table() -> None:
    rows = (
        "\n| | Year Ended December 31, 2025 |\n| --- | --- |\n"
        "| Diluted earnings per share | 4.00 |"
    )
    assert reasons(q4(rows)) == [labels.NO_CONFIRMED_COLUMN]


def test_fourth_quarter_eps_not_printed_has_no_reason_of_its_own() -> None:
    # The full-year EPS (4.00) isn't printed: the release is left out for that.
    assert q4("| Diluted earnings per share | 1.00 | 0.90 | 4.01 | 3.60 |") == (
        None,
        [],
    )


# Training examples


def test_example_messages(key) -> None:
    result = label_fixture(key, AAPL, "10-Q", date(2025, 3, 29))
    prompt = extract.load_prompt("extract_v2", PROMPTS).text
    text = build_input(record_of(AAPL)).text
    found = labels.example(result.release, result.labels, prompt, text)
    assert found["release_accession"] == AAPL
    assert [m["role"] for m in found["messages"]] == ["system", "user", "assistant"]
    assert found["messages"][0]["content"] == (PROMPTS / "extract_v2.txt").read_text()
    assert found["messages"][1]["content"] == text
    parsed = extract.parse_reply(found["messages"][2]["content"], AAPL, "with_units")
    assert parsed.prediction.quarter == grade.Figures(
        revenue=D("95359000000"), net_income=D("24780000000"), eps_diluted=D("1.65")
    )
    assert parsed.units == ("millions",) * 4


ROWS = {
    "a": {"ticker": "AAA", "split": "train", "filing_date": "2025-05-01"},
    "b": {"ticker": "AAA", "split": "time_test", "filing_date": "2026-07-31"},
    "c": {"ticker": "VVV", "split": "validation", "filing_date": "2025-05-01"},
    "d": {"ticker": "CCC", "split": "company_test", "filing_date": "2025-05-01"},
    "e": {"ticker": "CCC", "split": "unused", "filing_date": "2024-05-01"},
    "f": {"ticker": "AAA", "split": "train", "filing_date": "2026-07-01"},
    "g": {"ticker": "VVV", "split": "train", "filing_date": "2025-05-01"},
}
SPLITS = {"AAA": "train", "VVV": "validation", "CCC": "company_test"}


def test_train_only_problems() -> None:
    assert labels.train_only_problems(["a"], ROWS, SPLITS) == []
    assert labels.train_only_problems(list("bcdefg") + ["a", "a", "z"], ROWS, SPLITS) == [
        "a: 2 examples",
        "b: split time_test in releases.csv",
        "b: filed 2026-07-31",
        "c: split validation in releases.csv",
        "c: VVV is validation in companies.csv",
        "d: split company_test in releases.csv",
        "d: CCC is company_test in companies.csv",
        "e: split unused in releases.csv",
        "e: CCC is company_test in companies.csv",
        "f: filed 2026-07-01",
        "g: VVV is validation in companies.csv",
        "z: not in releases.csv",
    ]  # fmt: skip


def test_examples_file_problems(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    labels.write_examples(path, [{"release_accession": "a"}])
    assert labels.examples_file_problems(path, ROWS, SPLITS) == []
    labels.write_examples(
        path, [{"release_accession": "a"}, {"release_accession": "d"}]
    )
    assert labels.examples_file_problems(path, ROWS, SPLITS) == [
        "d: split company_test in releases.csv",
        "d: CCC is company_test in companies.csv",
    ]


# Tokens


def test_tokenizer_path_mirrors_the_url() -> None:
    assert labels.tokenizer_path(Path("data/raw")) == Path(
        "data/raw/huggingface.co/Qwen/Qwen3-8B/resolve/"
        "b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json"
    )


def test_fetch_pinned_downloads_once(tmp_path) -> None:
    body = b'{"model": {}}'
    digest = hashlib.sha256(body).hexdigest()
    url = "https://huggingface.co/org/model/resolve/abc/tokenizer.json"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=body)

    path = tmp_path / "huggingface.co" / "tokenizer.json"
    assert labels.fetch_pinned(url, path, digest, httpx.MockTransport(handler)) == path
    assert path.read_bytes() == body
    assert len(requests) == 1
    assert "@" not in requests[0].headers["user-agent"]  # not the SEC User-Agent

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("fetched twice")

    assert labels.fetch_pinned(url, path, digest, httpx.MockTransport(refuse)) == path


def test_fetch_pinned_refuses_another_file(tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"other"))
    with pytest.raises(labels.LabelError, match="pinned sha256"):
        labels.fetch_pinned("https://huggingface.co/t.json", path, "0" * 64, transport)
    assert not path.exists()
    path.write_bytes(b"changed")
    with pytest.raises(labels.LabelError, match="pinned sha256"):
        labels.fetch_pinned("https://huggingface.co/t.json", path, "0" * 64, transport)


def test_chat_text_is_qwen3s_layout() -> None:
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": '{"a": 1}'},
    ]
    assert labels.chat_text(messages) == (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
        '<|im_start|>assistant\n<think>\n\n</think>\n\n{"a": 1}<|im_end|>\n'
    )


class FakeTokenizer:
    """One token per whitespace-separated word."""

    def __init__(self) -> None:
        self.special: list[bool] = []

    def encode_batch(self, texts: list[str], add_special_tokens: bool = True):
        self.special.append(add_special_tokens)
        return [SimpleNamespace(ids=text.split()) for text in texts]


def test_token_counts() -> None:
    examples = [
        {
            "messages": [
                {"role": "system", "content": "two words"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "a b c"},
            ]
        }
    ]
    tokenizer = FakeTokenizer()
    assert labels.token_counts(examples, tokenizer) == {
        "system prompt": [2],
        "user (release input)": [1],
        "assistant (reply)": [3],
        "whole example": [len(labels.chat_text(examples[0]["messages"]).split())],
    }
    assert tokenizer.special == [False] * 4


# The command, end to end on the stage 1 fixture releases


def make_client(cache_dir: Path) -> EdgarClient:
    """A client serving link_submissions.json and answers_companyfacts.json."""
    bodies = json.loads(SUBMISSIONS.read_bytes()) | json.loads(
        COMPANYFACTS.read_bytes()
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


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


RELEASES = [
    # ticker, cik, company split, accession, filed, form, linked, period end, split
    ("AAPL", "320193", "train", AAPL, "2025-05-01", "10-Q", "0000320193-25-000057", "2025-03-29", "train"),
    ("MSFT", "789019", "train", MSFT, "2025-07-30", "10-K", "0000950170-25-100235", "2025-06-30", "train"),
    ("WMT", "104169", "train", WMT, "2025-05-15", "10-Q", "0000104169-25-000090", "2025-04-30", "train"),
    ("CAT", "18230", "validation", CAT, "2025-08-05", "10-Q", "0000018230-25-000040", "2025-06-30", "validation"),
    ("XOM", "2115436", "train", XOM, "2026-07-31", "10-Q", "0000034088-26-000093", "2026-06-30", "time_test"),
    ("ZZZ", "1", "company_test", "0000000001-25-000001", "2025-05-01", "10-Q", "0000000001-25-000003", "2025-03-31", "company_test"),
    ("ZZZ", "1", "company_test", "0000000001-25-000002", "2025-02-01", "10-K", "0000000001-25-000004", "2024-12-31", "unused"),
]  # fmt: skip


@pytest.fixture
def stage2(tmp_path) -> Path:
    """A stage 2 folder with one release of each split: AAPL, MSFT and WMT train."""
    out = tmp_path / "stage2"
    shutil.copytree(INPUT_RELEASES, out / "releases")
    columns = [
        "ticker",
        "cik",
        "company_split",
        "release_accession",
        "filing_date",
        "linked_form",
        "linked_accession",
        "period_end",
        "split",
    ]
    rows = []
    for values in RELEASES:
        row = dict(zip(columns, values, strict=True))
        rows.append(
            {
                **{k: row[k] for k in columns[:5]},
                "saved": "True",
                "link_status": "linked",
                **{k: row[k] for k in columns[5:]},
                "answers": "6",
                "missing": "0",
                "preferred_dividends": "False",
            }
        )
    write_csv(out / "releases.csv", rows)
    shutil.copy(LINKS, out / "links.csv")
    lines = LABEL_ANSWERS.read_text().splitlines()
    kept = [lines[0]] + [x for x in lines[1:] if x.split(",")[1] in {AAPL, MSFT, WMT}]
    (out / "answers.csv").write_text("\n".join(kept) + "\n")
    (out / "failures.csv").write_text(
        "ticker,cik,split,release_accession,step,kind,reason\n"
    )
    (out / "merged.json").write_text('{"commit": "stage2-commit"}\n')
    companies = [
        {"ticker": t, "cik": c, "name": t, "sector": "", "split": s}
        for t, c, s in dict.fromkeys((r[0], r[1], r[2]) for r in RELEASES)
    ]
    write_csv(tmp_path / "companies.csv", companies)
    return out


def run(stage2: Path, tmp_path: Path, tokenizer=None) -> tuple[int, Path]:
    out = tmp_path / "labels"
    code = labels.main(
        [
            "--stage2", str(stage2),
            "--companies", str(tmp_path / "companies.csv"),
            "--out", str(out),
            "--prompts-dir", str(PROMPTS),
        ],
        client=make_client(tmp_path / "raw"),
        commit="test-commit",
        tokenizer=tokenizer or FakeTokenizer(),
    )  # fmt: skip
    return code, out


def test_main_writes_train_releases_only(stage2, tmp_path, capsys) -> None:
    code, out = run(stage2, tmp_path)
    assert code == 0, capsys.readouterr().err
    examples = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    assert [e["release_accession"] for e in examples] == [AAPL, MSFT, WMT]
    replies = {e["release_accession"]: e["messages"][2]["content"] for e in examples}
    assert replies[MSFT] == (
        '{"period_end": "2025-06-30", "quarter": {"revenue": {"value": 76441, "unit": '
        '"millions"}, "net_income": {"value": 27233, "unit": "millions"}, '
        '"eps_diluted": 3.65}, "year_to_date": {"revenue": {"value": 281724, "unit": '
        '"millions"}, "net_income": {"value": 101832, "unit": "millions"}, '
        '"eps_diluted": 13.64}}'
    )
    assert replies[WMT] == (
        '{"period_end": "2025-04-30", "quarter": {"revenue": {"value": 165609, "unit": '
        '"millions"}, "net_income": {"value": 4487, "unit": "millions"}, '
        '"eps_diluted": 0.56}, "year_to_date": {"revenue": {"value": 165609, "unit": '
        '"millions"}, "net_income": {"value": 4487, "unit": "millions"}, '
        '"eps_diluted": 0.56}}'
    )
    records = {AAPL: record_of(AAPL), MSFT: record_of(MSFT), WMT: record_of(WMT)}
    for e in examples:
        assert (
            e["messages"][1]["content"]
            == build_input(records[e["release_accession"]]).text
        )
    # Self-check: every graded figure correct; fourth-quarter EPS and the first
    # quarter's year to date have no answer.
    with (out / "selfcheck" / "graded.csv").open(newline="") as f:
        graded = list(csv.DictReader(f))
    assert Counter(r["verdict"] for r in graded) == {"correct": 14, "no_answer": 4}
    with (out / "examples.csv").open(newline="") as f:
        index = list(csv.DictReader(f))
    assert [(r["release_accession"], r["ticker"], r["kind"]) for r in index] == [
        (AAPL, "AAPL", "second or third quarter"),
        (MSFT, "MSFT", "fourth quarter"),
        (WMT, "WMT", "first quarter"),
    ]
    assert [r["linked_form"] for r in index] == ["10-Q", "10-K", "10-Q"]
    assert (out / "left_out.csv").read_text().count("\n") == 1  # the header
    with (out / "figures.csv").open(newline="") as f:
        figures = list(csv.DictReader(f))
    assert len(figures) == 18
    assert Counter(r["source"] for r in figures) == {
        "cell": 14,
        "first_quarter": 3,
        "fourth_quarter_eps": 1,
    }
    report = (out / "report.md").read_text()
    assert "Linked train releases: 3; labelled 3, left out 0." in report
    assert "Examples by split (releases.csv): train 3;" in report
    assert "Problems: none; every graded figure is correct." in report
    assert "| whole example |" in report and "| median | p90 | p95 | p99 |" in report
    longest = next(x for x in report.splitlines() if x.startswith("Longest whole"))
    assert {AAPL, MSFT, WMT} <= set(longest.replace(";", " ").split())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["commit"] == "test-commit"
    assert manifest["stage2"]["commit"] == "stage2-commit"
    assert manifest["releases"] == {"linked_train": 3, "labelled": 3, "left_out": 0}
    handcheck = (out / "handcheck.md").read_text()
    assert "## 20 labelled figures" in handcheck
    assert "## 10 fourth-quarter EPS cells" in handcheck
    assert handcheck.count("[ ]") == 15 + 1  # all 15 read from a release, 1 EPS
    assert (
        "printed `$3.65` in row 'Diluted', column 'Three Months Ended June 30, 2025'"
        in handcheck
    )


def test_handcheck_draw_is_fixed_by_the_seed(stage2, tmp_path) -> None:
    built = labels.build(
        stage2,
        tmp_path / "companies.csv",
        "prompt",
        make_client(tmp_path / "raw"),
    )
    first = labels.format_handcheck(built.results)
    assert first == labels.format_handcheck(built.results)
    assert "first_quarter" not in first and "(filed 2025-05-15)" in first


def test_self_check_catches_a_wrong_reply(stage2, tmp_path, capsys) -> None:
    code, out = run(stage2, tmp_path)
    assert code == 0
    client = make_client(tmp_path / "raw")
    built = labels.build(stage2, tmp_path / "companies.csv", "prompt", client)
    path = out / "train.jsonl"
    examples = [json.loads(x) for x in path.read_text().splitlines()]
    reply = examples[0]["messages"][2]
    reply["content"] = reply["content"].replace('"value": 95359,', '"value": 95358,')
    labels.write_examples(path, examples)
    check = labels.self_check(out, built.results, stage2, client)
    assert check.problems == [
        f"{AAPL} revenue quarter: parsed 95358000000, labelled 95359000000",
        (
            f"{AAPL} revenue quarter: other (no evidence), answer 95359000000, "
            "label 95358000000"
        ),
    ]
    assert check.verdicts == {"correct": 13, "other": 1, "no_answer": 4}
