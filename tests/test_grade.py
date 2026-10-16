import contextlib
import csv
import dataclasses
import io
import json
import shutil
from datetime import date
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from secbench import answers, grade
from secbench.answers import PERIOD_TYPES, Answer, Fact, LinkedRelease, load_links
from secbench.edgar import EdgarClient, RateLimiter
from secbench.grade import (
    ADJUSTED,
    CORRECT,
    GAAP_LABEL,
    GRADED_COLUMNS,
    MISSING,
    NO_ANSWER,
    NONCONTROLLING_INTEREST,
    NOT_IN_INPUT,
    OTHER,
    ROUNDED,
    SCALE,
    SIGN,
    VERDICTS,
    WRONG_LINE,
    WRONG_PERIOD,
    Figure,
    Graded,
    GradeError,
    Prediction,
    PredictionsError,
    RunNotes,
    cell_figure,
    check_key,
    find_in_input,
    format_summary,
    grade_prediction,
    grade_value,
    input_forms,
    input_rows,
    load_answers,
    load_filing_facts,
    load_predictions,
    other_period,
    precision,
    prose_figures,
)
from secbench.inputs import build_input, release_records

FIXTURES = Path(__file__).parent / "fixtures"
UA = "Test Runner test@example.com"
LINKS = FIXTURES / "answers_links.csv"
COMPANYFACTS = FIXTURES / "answers_companyfacts.json"
RELEASES = FIXTURES / "input_releases"
# Maintained by hand.
HANDCHECKED = FIXTURES / "grade_handchecked.csv"
HANDCHECKED_COLUMNS = [
    "ticker",
    "linked_accession",
    "field",
    "period_type",
    "predicted_value",
    "expected_verdict",
]
FIELD_NAMES = [field.name for field in answers.FIELDS]
D = Decimal

# Synthetic cases: a second-quarter 10-Q.
Q2 = (date(2026, 4, 1), date(2026, 6, 30))
H1 = (date(2026, 1, 1), date(2026, 6, 30))
Q2_PRIOR = (date(2025, 4, 1), date(2025, 6, 30))
RELEASE = "0000000001-26-000009"
FILING = "0000000001-26-000010"
TEXT = """Second-quarter net income was $14,525 million.

| | Three Months Ended June 30, 2026 | Three Months Ended June 30, 2025 |
| --- | --- | --- |
| Total revenues | 116,017 | 81,506 |
| Net income | 14,525 | 7,082 |
| Adjusted net income (non-GAAP) | 14,680 | 7,100 |
| Operating cash flow | 20,000 | 12,000 |
| Diluted earnings per share | 3.48 | 1.64 |
| Adjusted diluted EPS (non-GAAP) | 3.52 | 1.66 |
"""


def answer(
    value: str = "14525000000",
    field: str = "net_income",
    tag: str = "NetIncomeLoss",
    period: tuple[date, date] = Q2,
    period_type: str = "quarter",
) -> Answer:
    unit = "USD/shares" if field == "eps_diluted" else "USD"
    return Answer(
        "TEST",
        RELEASE,
        FILING,
        field,
        period_type,
        period[0],
        period[1],
        D(value),
        unit,
        tag,
    )


def fact(tag: str, period: tuple[date, date], value: str, unit: str = "USD") -> Fact:
    return Fact(tag, unit, period[0], period[1], D(value))


def in_table(text: str) -> str:
    """A table row put under a header row (the first row of a table is its header);
    prose is left alone."""
    return f"| | 2026 |\n| --- | --- |\n{text}" if text.startswith("|") else text


def verdict(
    predicted: str | None,
    expected: Answer | None = None,
    text: str = TEXT,
    facts: tuple[Fact, ...] = (),
) -> grade.Verdict:
    return grade_value(
        expected or answer(),
        None if predicted is None else D(predicted),
        text,
        input_rows(text),
        set(facts),
    )


# Reading the input


@pytest.mark.parametrize(
    ("cell", "value"),
    [
        ("1,234", "1234"),
        ("$1,234", "1234"),
        ("$ 1,234", "1234"),
        ("$(1,234)", "-1234"),
        ("($1,234)", "-1234"),
        ("(1,234)", "-1234"),
        ("-1,234", "-1234"),
        ("–1,234", "-1234"),
        ("−1,234", "-1234"),
        ("2.11<sup>1</sup>", "2.11"),
        ("**3,123**", "3123"),
        ("0.80", "0.80"),
        ("12.5 %", None),
        ("12.5%", None),
        ("—", None),
        ("n/a", None),
        ("Q2", None),
        ("", None),
    ],
)
def test_cell_figure(cell, value) -> None:
    figure = cell_figure(cell)
    assert (figure and str(figure.value)) == value


def test_prose_figures() -> None:
    line = (
        "Revenue of $95.4 billion, up 5% from 2025; EPS of $(0.25) in Q2; "
        "margin 3 percent; cash of $1,234 million"
    )
    assert prose_figures(line) == (
        Figure("$95.4 billion", D("95.4"), 10**9),
        Figure("2025", D("2025")),
        Figure("(0.25)", D("-0.25")),
        Figure("$1,234 million", D("1234"), 10**6),
    )


def test_input_rows() -> None:
    rows = input_rows(TEXT)
    assert [row.label for row in rows] == [
        "Second-quarter net income was $14,525 million.",
        "Total revenues",
        "Net income",
        "Adjusted net income (non-GAAP)",
        "Operating cash flow",
        "Diluted earnings per share",
        "Adjusted diluted EPS (non-GAAP)",
    ]
    assert [f.value for f in rows[2].figures] == [D(14525), D(7082)]
    assert [f.column for f in rows[2].figures] == [1, 2]
    assert rows[2].header == (
        "",
        "Three Months Ended June 30, 2026",
        "Three Months Ended June 30, 2025",
    )
    assert (
        rows[2].column_header(rows[2].figures[1]) == "Three Months Ended June 30, 2025"
    )
    # The sentence above the table is not its heading.
    assert (rows[2].section, rows[2].heading) == ("", "")
    assert rows[0].figures == (Figure("$14,525 million", D(14525), 10**6),)
    assert rows[0].header == ()


def test_input_rows_skip_each_tables_header() -> None:
    text = "| | 2026 |\n| --- | --- |\n| Revenue | 5 |\n\n| | 2025 |\n| --- | --- |\n| Profit | 7 |"
    assert [row.label for row in input_rows(text)] == ["Revenue", "Profit"]


# Answerable check


@pytest.mark.parametrize(
    ("value", "field", "forms"),
    [
        ("94036000000", "revenue", ["94,036", "94,036,000"]),
        ("10542801000", "revenue", ["10,542,801"]),  # Netflix reports in thousands
        ("1234567", "net_income", []),
        ("-2500000000", "net_income", ["2,500", "2,500,000"]),
        ("1.57", "eps_diluted", ["1.57"]),
        ("0.8", "eps_diluted", ["0.80"]),
        ("1", "eps_diluted", ["1.00"]),
        ("-0.25", "eps_diluted", ["0.25"]),
        ("1.565", "eps_diluted", []),
    ],
)
def test_input_forms(value, field, forms) -> None:
    assert input_forms(D(value), field) == forms


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("Revenue was 94,036.", "94,036"),
        ("| Revenue | $94,036 |", "94,036"),
        ("| Revenue | 94,036<sup>1</sup> |", "94,036"),
        ("| Revenue (in thousands) | 94,036,000 |", "94,036,000"),
        ("| Revenue | 194,036 |", None),
        ("| Revenue | 94,036.5 |", None),
        ("| Revenue | 94,0361 |", None),
        ("Revenue grew 94,036%", None),
        ("Revenue grew 94,036 %", None),
    ],
)
def test_find_positive_answer(text, found) -> None:
    text = in_table(text)
    assert find_in_input(D("94036000000"), "revenue", text, input_rows(text)) == found


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("| Net income | (2,500) |", "(2,500)"),
        ("| Net income | $(2,500) |", "(2,500)"),
        ("| Net income | ($2,500) |", "(2,500)"),
        ("| Net income | ( 2,500 ) |", "(2,500)"),
        ("Net income was -2,500.", "-2,500"),
        ("Net income was –2,500.", "-2,500"),
        ("Net income was −2,500.", "-2,500"),
        ("Net income was -$2,500.", "-2,500"),
        ("| Net loss | 2,500 |", "2,500 on a loss row (Net loss)"),
        ("| Net income (loss) | 2,500 |", "2,500 on a loss row (Net income (loss))"),
        ("| NET LOSSES | $2,500 |", "2,500 on a loss row (NET LOSSES)"),
        (
            "Net loss of $2,500 million.",
            "2,500 on a loss row (Net loss of $2,500 million.)",
        ),
        ("| Net loss | 2,500,000 |", "2,500,000 on a loss row (Net loss)"),
        ("| Net income | 2,500 |", None),  # unsigned, not a loss row
        ("| Net loss | (12,500) |", None),
        ("| Net loss | 12,500 |", None),
    ],
)
def test_find_negative_answer(text, found) -> None:
    text = in_table(text)
    assert (
        find_in_input(D("-2500000000"), "net_income", text, input_rows(text)) == found
    )


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("| Diluted EPS | $(0.25) |", "(0.25)"),
        (
            "| Diluted loss per share | 0.25 |",
            "0.25 on a loss row (Diluted loss per share)",
        ),
        ("| Diluted EPS | 0.25 |", None),
        ("| Diluted EPS | (0.255) |", None),
    ],
)
def test_find_negative_eps(text, found) -> None:
    text = in_table(text)
    assert find_in_input(D("-0.25"), "eps_diluted", text, input_rows(text)) == found


def test_not_in_input_comes_first() -> None:
    # Even a correct prediction is not graded when the answer isn't in the input.
    text = TEXT.replace("14,525", "14,526")
    assert verdict("14525000000", text=text) == grade.Verdict(
        NOT_IN_INPUT, "searched 14,525, 14,525,000"
    )
    assert verdict(None, text=text).verdict == NOT_IN_INPUT


def test_not_in_input_without_an_input_form() -> None:
    expected = answer("1.565", "eps_diluted", "EarningsPerShareDiluted")
    assert verdict("1.565", expected) == grade.Verdict(
        NOT_IN_INPUT, "no input form for this value"
    )


# Correct


@pytest.mark.parametrize(
    ("value", "field", "unit"),
    [
        ("14525000000", "net_income", "1000000"),
        ("10542801000", "revenue", "1000"),
        ("10542801500", "revenue", "100"),
        ("1234567", "revenue", "1"),
        ("95000000000", "revenue", "1000000"),  # at most whole millions
        ("0", "net_income", "1000000"),
        ("-2500000000", "net_income", "1000000"),
        ("1.57", "eps_diluted", "0.01"),
        ("2", "eps_diluted", "0.01"),
    ],
)
def test_precision(value, field, unit) -> None:
    assert precision(D(value), field) == D(unit)


@pytest.mark.parametrize(
    ("predicted", "correct"),
    [
        ("14525000000", True),
        ("14525499999", True),
        ("14524500001", True),
        ("14525500000", False),  # half a unit away rounds to the next unit
        ("14524500000", False),
    ],
)
def test_usd_tolerance_is_half_the_answers_precision(predicted, correct) -> None:
    assert (verdict(predicted).verdict == CORRECT) == correct


@pytest.mark.parametrize(
    ("predicted", "correct"),
    [("10542801000", True), ("10542801499", True), ("10542801500", False)],
)
def test_usd_tolerance_in_thousands(predicted, correct) -> None:
    expected = answer("10542801000", "revenue", "Revenues")
    text = "| Revenues | $ 10,542,801 |"
    assert (verdict(predicted, expected, text).verdict == CORRECT) == correct


@pytest.mark.parametrize(
    ("predicted", "correct"),
    [
        ("3.48", True),
        ("3.484", True),
        ("3.4849", True),
        ("3.476", True),
        ("3.485", False),
        ("3.475", False),
    ],
)
def test_eps_tolerance_is_half_a_cent(predicted, correct) -> None:
    expected = answer("3.48", "eps_diluted", "EarningsPerShareDiluted")
    assert (verdict(predicted, expected).verdict == CORRECT) == correct


# Errors, in the order they are checked


def test_missing() -> None:
    assert verdict(None) == grade.Verdict(MISSING)


def test_sign() -> None:
    assert verdict("-14525000000") == grade.Verdict(SIGN)
    eps = answer("3.48", "eps_diluted", "EarningsPerShareDiluted")
    assert verdict("-3.48", eps) == grade.Verdict(SIGN)


@pytest.mark.parametrize(
    ("predicted", "evidence"),
    [
        ("14525", "prediction x1,000,000 is the answer"),
        ("14525000", "prediction x1,000 is the answer"),
        ("14525000000000", "prediction /1,000 is the answer"),
        ("14525000000000000", "prediction /1,000,000 is the answer"),
    ],
)
def test_scale(predicted, evidence) -> None:
    assert verdict(predicted) == grade.Verdict(SCALE, evidence)


FACTS = (
    fact("NetIncomeLoss", Q2, "14525000000"),
    fact("NetIncomeLoss", H1, "18708000000"),
    fact("NetIncomeLoss", Q2_PRIOR, "7082000000"),
    fact("Revenues", H1, "201155000000"),
    fact("ProfitLoss", Q2, "14881000000"),
    fact("ProfitLoss", H1, "19353000000"),
)


@pytest.mark.parametrize(
    ("predicted", "evidence"),
    [
        ("7082000000", "NetIncomeLoss 2025-04-01..2025-06-30: 7082000000"),
        ("18708000000", "NetIncomeLoss 2026-01-01..2026-06-30: 18708000000"),
        ("18708400000", "NetIncomeLoss 2026-01-01..2026-06-30: 18708000000"),
    ],
)
def test_wrong_period(predicted, evidence) -> None:
    assert verdict(predicted, facts=FACTS) == grade.Verdict(WRONG_PERIOD, evidence)


def test_wrong_period_needs_the_same_tag() -> None:
    assert verdict("201155000000", facts=FACTS) == grade.Verdict(OTHER)


def test_noncontrolling_interest() -> None:
    assert verdict("14881000000", facts=FACTS) == grade.Verdict(
        NONCONTROLLING_INTEREST, "ProfitLoss 2026-04-01..2026-06-30: 14881000000"
    )


def test_noncontrolling_interest_needs_the_same_period() -> None:
    assert verdict("19353000000", facts=FACTS) == grade.Verdict(OTHER)


def test_noncontrolling_interest_is_for_net_income_only() -> None:
    expected = answer("116017000000", "revenue", "Revenues")
    assert verdict("14881000000", expected, facts=FACTS) == grade.Verdict(OTHER)


@pytest.mark.parametrize(
    ("predicted", "unit"),
    [
        ("95360000000", "10,000,000"),
        ("95400000000", "100,000,000"),  # "$95.4 billion"
        ("95000000000", "1,000,000,000"),
        ("100000000000", "10,000,000,000"),
    ],
)
def test_rounded(predicted, unit) -> None:
    expected = answer("95359000000", "revenue", "Revenues")
    text = "| Total net sales | 95,359 |"
    assert verdict(predicted, expected, text) == grade.Verdict(
        ROUNDED, f"answer rounded to {unit}"
    )


def test_truncated_is_not_rounded() -> None:
    expected = answer("95359000000", "revenue", "Revenues")
    assert (
        verdict("95300000000", expected, "| Total net sales | 95,359 |").verdict
        == OTHER
    )


@pytest.mark.parametrize(("predicted", "unit"), [("1.6", "0.10"), ("2", "1.00")])
def test_rounded_eps(predicted, unit) -> None:
    expected = answer("1.57", "eps_diluted", "EarningsPerShareDiluted")
    assert verdict(predicted, expected, "| Diluted EPS | 1.57 |") == grade.Verdict(
        ROUNDED, f"answer rounded to {unit}"
    )


def test_not_rounded_to_zero() -> None:
    expected = answer("0.4", "eps_diluted", "EarningsPerShareDiluted")
    assert verdict("0", expected, "| Diluted EPS | 0.40 |").verdict == OTHER


@pytest.mark.parametrize(
    ("predicted", "evidence"),
    [
        ("14680000000", "'Adjusted net income (non-GAAP)': 14,680 (x1,000,000)"),
        ("7100000000", "'Adjusted net income (non-GAAP)': 7,100 (x1,000,000)"),
    ],
)
def test_adjusted_row(predicted, evidence) -> None:
    assert verdict(predicted) == grade.Verdict(ADJUSTED, evidence)


def test_adjusted_eps() -> None:
    expected = answer("3.48", "eps_diluted", "EarningsPerShareDiluted")
    assert verdict("3.52", expected) == grade.Verdict(
        ADJUSTED, "'Adjusted diluted EPS (non-GAAP)': 3.52"
    )


@pytest.mark.parametrize(
    "line",
    [
        "Excluding the charge, net income was $15.1 billion.",
        "Non-GAAP net income was $15.1 billion.",
        "Non GAAP net income was $15.1 billion.",
    ],
)
def test_adjusted_prose(line) -> None:
    assert verdict("15100000000", text=TEXT + "\n" + line + "\n") == grade.Verdict(
        ADJUSTED, f"{line!r}: $15.1 billion"
    )


@pytest.mark.parametrize(
    ("predicted", "evidence"),
    [
        ("20000000000", "'Operating cash flow': 20,000 (x1,000,000)"),
        ("116017000000", "'Total revenues': 116,017 (x1,000,000)"),
    ],
)
def test_wrong_line(predicted, evidence) -> None:
    assert verdict(predicted) == grade.Verdict(WRONG_LINE, evidence)


def test_wrong_line_in_prose() -> None:
    text = TEXT + "\nFree cash flow was $9.8 billion.\n"
    assert verdict("9800000000", text=text) == grade.Verdict(
        WRONG_LINE, "'Free cash flow was $9.8 billion.': $9.8 billion"
    )


def test_eps_figures_are_not_scaled() -> None:
    expected = answer("3.48", "eps_diluted", "EarningsPerShareDiluted")
    assert verdict("3.52", expected).verdict == ADJUSTED
    assert verdict("0.00352", expected).verdict == OTHER


def test_percentages_are_not_figures() -> None:
    expected = answer("3.48", "eps_diluted", "EarningsPerShareDiluted")
    text = TEXT + "\nOperating margin was 2.5%.\n| Margin | 2.5% |\n"
    assert verdict("2.5", expected, text).verdict == OTHER


def test_other() -> None:
    assert verdict("123") == grade.Verdict(OTHER)


def test_wrong_period_before_adjusted() -> None:
    facts = (fact("NetIncomeLoss", Q2_PRIOR, "14680000000"),)
    assert verdict("14680000000", facts=facts).verdict == WRONG_PERIOD


def test_rounded_before_adjusted() -> None:
    text = TEXT + "| Adjusted net income, rounded | 14,500 | |\n"
    assert verdict("14500000000", text=text).verdict == ROUNDED


def test_scale_before_wrong_line() -> None:
    assert (
        verdict("14525", text=TEXT + "\nWe have 14,525 employees.\n").verdict == SCALE
    )


# Another period column of the answer's row

SUMMARY = """| 2Q26 | 1Q26 | Change vs 1Q26 | Dollars in millions | YTD 2026 | YTD 2025 |
| --- | --- | --- | --- | --- | --- |
| Results Summary |  |  |  |  |  |
| 14,525 | 4,183 | 10,342 | Earnings (U.S. GAAP) | 18,708 | 14,795 |
| 14,680 | 8,772 | 5,908 | Adjusted earnings (non-GAAP) | 23,452 | 14,555 |
"""


@pytest.mark.parametrize(
    ("predicted", "evidence"),
    [
        ("4183000000", "'1Q26' column of 'Earnings (U.S. GAAP)': 4,183 (x1,000,000)"),
        (
            "18708000000",
            "'YTD 2026' column of 'Earnings (U.S. GAAP)': 18,708 (x1,000,000)",
        ),
        (
            "14795000000",
            "'YTD 2025' column of 'Earnings (U.S. GAAP)': 14,795 (x1,000,000)",
        ),
    ],
)
def test_other_period_column(predicted, evidence) -> None:
    assert verdict(predicted, text=SUMMARY) == grade.Verdict(WRONG_PERIOD, evidence)


def test_other_period_column_of_the_answers_row_in_text() -> None:
    expected = answer("3.48", "eps_diluted", "EarningsPerShareDiluted")
    assert verdict("1.64", expected) == grade.Verdict(
        WRONG_PERIOD,
        "'Three Months Ended June 30, 2025' column of 'Diluted earnings per share': "
        "1.64",
    )
    assert verdict("7082000000") == grade.Verdict(
        WRONG_PERIOD,
        "'Three Months Ended June 30, 2025' column of 'Net income': 7,082 (x1,000,000)",
    )


def test_xbrl_period_comes_before_a_column() -> None:
    facts = (fact("NetIncomeLoss", H1, "18708000000"),)
    assert verdict("18708000000", text=SUMMARY, facts=facts) == grade.Verdict(
        WRONG_PERIOD, "NetIncomeLoss 2026-01-01..2026-06-30: 18708000000"
    )


def test_change_column_is_not_a_period() -> None:
    assert verdict("10342000000", text=SUMMARY) == grade.Verdict(
        WRONG_LINE, "'Earnings (U.S. GAAP)': 10,342 (x1,000,000)"
    )


def test_adjusted_row_is_not_the_answers_row() -> None:
    # Adjusted earnings equal GAAP this quarter; its other column is still adjusted.
    text = (
        "| 2Q26 | 1Q26 | Dollars in millions |\n| --- | --- | --- |\n"
        "| 14,525 | 8,772 | Adjusted earnings (non-GAAP) |\n"
    )
    assert verdict("8772000000", text=text) == grade.Verdict(
        ADJUSTED, "'Adjusted earnings (non-GAAP)': 8,772 (x1,000,000)"
    )


def test_measure_columns_are_not_periods() -> None:
    # Microsoft's reconciliation: one period, a column per measure.
    revenue = answer(
        "77673000000", "revenue", "RevenueFromContractWithCustomerExcludingAssessedTax"
    )
    text = (
        "| | Three Months Ended September 30, Revenue "
        "| Three Months Ended September 30, Operating Income |\n| --- | --- | --- |\n"
        "| 2025 As Reported (GAAP) | $77,673 | $37,961 |\n"
    )
    assert verdict("37961000000", revenue, text) == grade.Verdict(
        WRONG_LINE, "'2025 As Reported (GAAP)': $37,961 (x1,000,000)"
    )
    # Caterpillar's: no period in the column headers at all.
    profit = answer("2179000000")
    text = (
        "| | Operating Profit | Profit Before Taxes | Profit |\n| --- | --- | --- | --- |\n"
        "| Three Months Ended June 30, 2025 - U.S. GAAP | $2,860 | $2,818 | $2,179 |\n"
    )
    assert verdict("2818000000", profit, text).verdict == WRONG_LINE


def test_basis_columns_are_not_periods() -> None:
    text = "| | 2026 GAAP | 2026 Adjusted |\n| --- | --- | --- |\n| Net income | 14,525 | 14,680 |"
    assert verdict("14680000000", text=text) == grade.Verdict(
        ADJUSTED, "'Net income' (column '2026 Adjusted'): 14,680 (x1,000,000)"
    )


def test_other_period_on_a_loss_row() -> None:
    loss = answer("-2500000000")
    text = "| | 2026 | 2025 |\n| --- | --- | --- |\n| Net loss | 2,500 | 1,200 |"
    assert verdict("-1200000000", loss, text) == grade.Verdict(
        WRONG_PERIOD, "'2025' column of 'Net loss': 1,200 (x1,000,000)"
    )


def test_prose_has_no_columns() -> None:
    line = "Net income was $14,525 million versus $7,082 million."
    assert verdict("7082000000", text=line) == grade.Verdict(
        WRONG_LINE, f"{line!r}: $7,082 million"
    )


@pytest.mark.parametrize(
    ("header", "answer_header", "other"),
    [
        ("1Q26", "2Q26", True),
        ("YTD 2026", "2Q26", True),
        ("Six Months Ended March 30, 2024", "Three Months Ended March 29, 2025", True),
        ("Q3 FY25", "Q4 FY25", True),
        ("Fiscal 2025 Q4", "Fiscal 2026 Q3", True),
        ("Year Ended May 31, 2025", "Year Ended May 31, 2026", True),
        ("Fourth Quarter 2024", "Full Year 2025", True),
        (
            "Three Months Ended September 30, Operating Income",
            "Three Months Ended September 30, Revenue",
            False,
        ),
        ("Change vs 1Q26", "2Q26", False),
        ("2026 Adjusted", "2026 GAAP", False),
        ("Walmart International 2025", "Walmart U.S. 2025", False),
        ("2Q26", "2Q26", False),
    ],
)
def test_other_period(header, answer_header, other) -> None:
    assert other_period(header, answer_header) == other


# Adjusted by context: section row, column header, heading above the table

SECTIONS = """| | 2Q26 | 1Q26 |
| --- | --- | --- |
| Earnings (U.S. GAAP) |  |  |
| Upstream | 7,927 | 5,737 |
| Total | 14,525 | 4,183 |
| Adjusted earnings (non-GAAP) |  |  |
| Upstream | 9,189 | 6,265 |
| Total | 14,680 | 8,772 |
| Volumes |  |  |
| Production | 4,514 | 4,594 |
| Please see the reconciliation on page 12. |  |  |
| Shares | 4,300 | 4,310 |
"""


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [
        (
            "9189000000",
            grade.Verdict(
                ADJUSTED,
                "'Upstream' (section 'Adjusted earnings (non-GAAP)'): 9,189 (x1,000,000)",
            ),
        ),
        (
            "8772000000",
            grade.Verdict(
                ADJUSTED,
                "'Total' (section 'Adjusted earnings (non-GAAP)'): 8,772 (x1,000,000)",
            ),
        ),
        # Under the GAAP section.
        ("7927000000", grade.Verdict(WRONG_LINE, "'Upstream': 7,927 (x1,000,000)")),
        # A new section ends the adjusted one.
        ("4514000000", grade.Verdict(WRONG_LINE, "'Production': 4,514 (x1,000,000)")),
        # A sentence is not a section.
        ("4300000000", grade.Verdict(WRONG_LINE, "'Shares': 4,300 (x1,000,000)")),
    ],
)
def test_adjusted_by_section_row(predicted, expected) -> None:
    assert verdict(predicted, text=SECTIONS) == expected


def test_input_rows_context() -> None:
    text = "Adjusted results\n\n7\n\n" + SECTIONS
    rows = {
        (r.label, r.figures[0].text if r.figures else ""): r for r in input_rows(text)
    }
    upstream = rows["Upstream", "9,189"]
    assert upstream.header == ("", "2Q26", "1Q26")
    assert upstream.section == "Adjusted earnings (non-GAAP)"
    assert upstream.heading == "Adjusted results"
    assert rows["Upstream", "7,927"].section == "Earnings (U.S. GAAP)"
    assert rows["Shares", "4,300"].section == "Volumes"


def under_heading(lines_above: str, body: str = "| Total | 14,680 |") -> str:
    return (
        f"Net income was $14,525 million.\n\n{lines_above}\n\n"
        f"| | 2026 |\n| --- | --- |\n{body}\n"
    )


@pytest.mark.parametrize(
    ("lines_above", "adjusted"),
    [
        ("Adjusted earnings", True),
        ("RECONCILIATION OF ADJUSTED DILUTED EARNINGS PER SHARE", True),
        ("Reconciliations of adjusted results to GAAP are as follows:", True),
        ("Non-GAAP Free Cash Flow", True),
        ("# EX-99.2: Earnings excluding identified items", True),
        # Past a page number, a footnote and "(more)".
        ("Adjusted earnings\n\n7\n\n<sup>1</sup> Restated.\n\n(more)", True),
        # A table split across pages continues under the same heading.
        ("Adjusted earnings\n\n| | 2025 |\n| --- | --- |\n| Total | 1 |", True),
        ("Please see a reconciliation of GAAP to non-GAAP measures on page 12.", False),
        ("Adjusted " + "results " * 20, False),  # a paragraph
        ("Results summary", False),
        ("Adjusted earnings\n\nResults summary", False),  # the nearest line only
    ],
)
def test_adjusted_by_heading(lines_above, adjusted) -> None:
    got = verdict("14680000000", text=under_heading(lines_above))
    assert got.verdict == (ADJUSTED if adjusted else WRONG_LINE), got.evidence


def test_heading_evidence() -> None:
    assert verdict("14680000000", text=under_heading("Adjusted earnings")) == (
        grade.Verdict(
            ADJUSTED, "'Total' (heading 'Adjusted earnings'): 14,680 (x1,000,000)"
        )
    )


def test_gaap_label_keeps_a_row_gaap() -> None:
    # Caterpillar's reconciliation under an adjusted-results heading.
    body = (
        "| Three Months Ended June 30, 2025 - U.S. GAAP | 2,818 |\n"
        "| Three Months Ended June 30, 2025 - Adjusted | 2,874 |"
    )
    text = under_heading("Reconciliations of adjusted results are as follows:", body)
    assert verdict("2818000000", text=text).verdict == WRONG_LINE
    assert verdict("2874000000", text=text) == grade.Verdict(
        ADJUSTED, "'Three Months Ended June 30, 2025 - Adjusted': 2,874 (x1,000,000)"
    )


def test_gaap_column_keeps_a_row_gaap() -> None:
    text = (
        "Net income was $14,525 million.\n\nAdjusted results\n\n"
        "| | 2026 GAAP |\n| --- | --- |\n| Total | 3,000 |\n"
    )
    assert verdict("3000000000", text=text).verdict == WRONG_LINE


@pytest.mark.parametrize(
    ("text", "gaap"),
    [
        ("Earnings (U.S. GAAP)", True),
        ("GAAP net income", True),
        ("As Reported (GAAP)", True),
        ("Non-GAAP net income", False),
        ("non GAAP", False),
        ("NON-GAAP", False),
    ],
)
def test_gaap_label(text, gaap) -> None:
    assert bool(GAAP_LABEL.search(text)) == gaap


# Predictions


def prediction_line(accession: str = RELEASE, **groups: dict) -> str:
    body = {
        "release_accession": accession,
        "period_end": "2026-06-30",
        "quarter": {},
        "year_to_date": {},
    } | groups
    return json.dumps(body)


def test_load_predictions(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        prediction_line(quarter={"revenue": 116017000000, "eps_diluted": 0.1})
        + "\n\n"
        + prediction_line("0000000001-26-000011", year_to_date={"net_income": None})
        + "\n"
    )
    first, second = load_predictions(path)
    assert first.quarter.revenue == D("116017000000")
    assert str(first.quarter.eps_diluted) == "0.1"  # exact, not a float
    assert first.quarter.net_income is None
    assert first.period_end == date(2026, 6, 30)
    assert second.release_accession == "0000000001-26-000011"


def test_load_predictions_reports_every_bad_line(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    lines = [
        prediction_line(quarter={"revenu": 1}),
        json.dumps(
            {"release_accession": RELEASE, "period_end": "2026-06-30", "quarter": {}}
        ),
        prediction_line("0000000001-26-000012", quarter={"eps_diluted": "NaN"}),
        "{not json",
        prediction_line("0000000001-26-000013"),
        prediction_line("0000000001-26-000013"),
    ]
    path.write_text("\n".join(lines))
    with pytest.raises(PredictionsError) as raised:
        load_predictions(path)
    message = str(raised.value)
    assert "line 1: quarter.revenu: Extra inputs are not permitted" in message
    assert "line 2: year_to_date: Field required" in message
    assert "line 3: quarter.eps_diluted:" in message
    assert "line 4: Invalid JSON" in message
    assert "line 6: 0000000001-26-000013 is also on line 5" in message


# Grading a prediction


def linked_release(form: str = "10-Q") -> LinkedRelease:
    end = date(2026, 6, 30) if form == "10-Q" else date(2025, 12, 31)
    return LinkedRelease("TEST", RELEASE, form, FILING, end)


def test_grade_prediction_on_a_10q() -> None:
    key = {
        (RELEASE, "net_income", "quarter"): answer(),
        (RELEASE, "net_income", "year_to_date"): answer(
            "18708000000", period=H1, period_type="year_to_date"
        ),
    }
    text = TEXT + "\n| Net income, six months | 18,708 |\n"
    prediction = Prediction.model_validate(
        {
            "release_accession": RELEASE,
            "period_end": "2026-06-30",
            "quarter": {"revenue": 116017000000, "net_income": 14525000000},
            "year_to_date": {"net_income": 14525000000},
        }
    )
    graded = grade_prediction(prediction, linked_release(), key, set(FACTS), text)
    assert [(g.field, g.period_type, g.verdict, g.evidence) for g in graded] == [
        ("revenue", "quarter", NO_ANSWER, "no quarter answer for this release"),
        ("net_income", "quarter", CORRECT, ""),
        (
            "net_income",
            "year_to_date",
            WRONG_PERIOD,
            "NetIncomeLoss 2026-04-01..2026-06-30: 14525000000",
        ),
    ]
    assert graded[1] == Graded(
        "TEST",
        RELEASE,
        FILING,
        "net_income",
        "quarter",
        D("14525000000"),
        "NetIncomeLoss",
        D("14525000000"),
        CORRECT,
        "",
    )
    assert graded[0].answer is None and graded[0].answer_tag == ""


def test_grade_prediction_on_a_10k() -> None:
    # year_to_date is the full year; a quarter figure has no answer.
    full_year = (date(2025, 1, 1), date(2025, 12, 31))
    key = {
        (RELEASE, "eps_diluted", "full_year"): answer(
            "6.7", "eps_diluted", "EarningsPerShareDiluted", full_year, "full_year"
        )
    }
    prediction = Prediction.model_validate(
        {
            "release_accession": RELEASE,
            "period_end": "2025-12-31",
            "quarter": {"eps_diluted": 1.8},
            "year_to_date": {"eps_diluted": 6.7, "revenue": None},
        }
    )
    graded = grade_prediction(
        prediction, linked_release("10-K"), key, set(), "| Diluted EPS | 1.80 | 6.70 |"
    )
    assert [(g.field, g.period_type, g.verdict) for g in graded] == [
        ("eps_diluted", "quarter", NO_ANSWER),
        ("eps_diluted", "full_year", CORRECT),
    ]


def test_answers_without_a_prediction_are_missing() -> None:
    key = {(RELEASE, "net_income", "quarter"): answer()}
    prediction = Prediction.model_validate(
        {
            "release_accession": RELEASE,
            "period_end": "2026-06-30",
            "quarter": {},
            "year_to_date": {},
        }
    )
    graded = grade_prediction(prediction, linked_release(), key, set(), TEXT)
    assert [(g.field, g.verdict) for g in graded] == [("net_income", MISSING)]


def test_load_answers_reads_what_answers_writes(tmp_path) -> None:
    derived = Answer(
        "TEST",
        RELEASE,
        FILING,
        "revenue",
        "quarter",
        date(2025, 10, 1),
        date(2025, 12, 31),
        D("320"),
        "USD",
        "Revenues",
        derived=True,
        full_year_accession=FILING,
        nine_months_accession="0000000001-25-000030",
        fiscal_year_start=date(2025, 1, 1),
    )
    path = tmp_path / "answers.csv"
    path.write_text(answers.answers_csv([answer(), derived]))
    assert path.read_text().splitlines()[1:] == [
        (
            f"TEST,{RELEASE},{FILING},net_income,quarter,2026-04-01,2026-06-30,"
            "14525000000,USD,NetIncomeLoss,False,,,"
        ),
        (
            f"TEST,{RELEASE},{FILING},revenue,quarter,2025-10-01,2025-12-31,320,USD,"
            f"Revenues,True,{FILING},0000000001-25-000030,2025-01-01"
        ),
    ]
    assert load_answers(path) == {
        (RELEASE, "net_income", "quarter"): answer(),
        (RELEASE, "revenue", "quarter"): derived,
    }


def test_load_answers_rejects_other_columns(tmp_path) -> None:
    path = tmp_path / "answers.csv"
    lines = answers.answers_csv([answer()]).splitlines()
    # answers.csv from before the derived columns
    path.write_text("\n".join(line.rsplit(",", 4)[0] for line in lines) + "\n")
    with pytest.raises(GradeError, match="rerun secbench.answers"):
        load_answers(path)
    path.write_text("\n".join([lines[0], lines[1].replace(",False,", ",yes,")]))
    with pytest.raises(GradeError, match="line 2: derived is 'yes', not True or"):
        load_answers(path)


def test_check_key() -> None:
    facts = {FILING: set(FACTS)}
    assert check_key([answer()], facts) == []
    assert check_key([answer("14526000000")], facts) == [
        (
            f"{RELEASE} net_income quarter: answers.csv has 14526000000, company facts "
            "14525000000"
        )
    ]
    assert check_key([answer(tag="ProfitLossX")], facts) == [
        (
            f"{RELEASE} net_income quarter: answers.csv has 14525000000, company facts "
            "no fact"
        )
    ]


# A fourth quarter derived from a 10-K (FILING) and its third-quarter 10-Q.
NINE_MONTHS_FILING = "0000000001-25-000030"
FULL_YEAR_25 = (date(2025, 1, 1), date(2025, 12, 31))
NINE_MONTHS_25 = (date(2025, 1, 1), date(2025, 9, 30))
Q4_25 = (date(2025, 10, 1), date(2025, 12, 31))


def derived_answer(value: str = "320", **changes) -> Answer:
    return dataclasses.replace(
        Answer(
            "TEST",
            RELEASE,
            FILING,
            "revenue",
            "quarter",
            *Q4_25,
            D(value),
            "USD",
            "Revenues",
            derived=True,
            full_year_accession=FILING,
            nine_months_accession=NINE_MONTHS_FILING,
            fiscal_year_start=FULL_YEAR_25[0],
        ),
        **changes,
    )


def test_check_key_recomputes_derived_answers() -> None:
    facts = {
        # The 10-K's own fourth-quarter fact (321) is not what a derived row checks.
        FILING: {
            fact("Revenues", FULL_YEAR_25, "1200"),
            fact("Revenues", Q4_25, "321"),
        },
        NINE_MONTHS_FILING: {fact("Revenues", NINE_MONTHS_25, "880")},
    }
    assert check_key([derived_answer()], facts) == []
    has = f"{RELEASE} revenue quarter: answers.csv has"
    other_10q = "0000000001-25-000031"
    assert check_key(
        [
            derived_answer("321"),
            derived_answer(nine_months_accession=other_10q),
            derived_answer(fiscal_year_start=date(2025, 1, 2)),
            derived_answer(start=date(2025, 9, 1)),  # nine months end 2025-08-31
            derived_answer(full_year_accession=NINE_MONTHS_FILING),
            derived_answer(fiscal_year_start=None),
            derived_answer(tag="RevenueFromContractWithCustomerExcludingAssessedTax"),
        ],
        facts,
    ) == [
        (
            f"{has} 321, company facts full year 1200 ({FILING}) minus nine months "
            f"880 ({NINE_MONTHS_FILING}) = 320"
        ),
        (
            f"{has} 320, company facts full year 1200 ({FILING}) minus nine months "
            f"no fact ({other_10q})"
        ),
        (
            f"{has} 320, company facts full year no fact ({FILING}) minus nine months "
            f"no fact ({NINE_MONTHS_FILING})"
        ),
        (
            f"{has} 320, company facts full year 1200 ({FILING}) minus nine months "
            f"no fact ({NINE_MONTHS_FILING})"
        ),
        f"{has} 320, derived from {NINE_MONTHS_FILING}, not the linked one",
        f"{has} 320, derived without a fiscal_year_start",
        (
            f"{has} 320, company facts full year no fact ({FILING}) minus nine months "
            f"no fact ({NINE_MONTHS_FILING})"
        ),
    ]
    facts[NINE_MONTHS_FILING].add(fact("Revenues", NINE_MONTHS_25, "881"))
    assert check_key([derived_answer()], facts) == [
        (
            f"{has} 320, company facts full year 1200 ({FILING}) minus nine months "
            f"880, 881 ({NINE_MONTHS_FILING})"
        )
    ]


def graded_row(
    verdict: str, ticker: str = "AAPL", field: str = "revenue", period="quarter"
):
    return Graded(
        ticker, RELEASE, FILING, field, period, D(1), "Revenues", D(1), verdict, "e"
    )


def test_format_summary() -> None:
    graded = [
        graded_row(CORRECT),
        graded_row(CORRECT, "XOM", "net_income", "year_to_date"),
        graded_row(WRONG_PERIOD, "XOM", "net_income"),
        graded_row(NOT_IN_INPUT, "XOM", "eps_diluted"),
        graded_row(NO_ANSWER, "MSFT", "eps_diluted"),
    ]
    notes = RunNotes(3, 57, ["0000320187-26-000070"], ["AAPL x: predicted y"])
    lines = format_summary(graded, notes).splitlines()
    verdict_lines = [
        f"  {v:<24}{n:>6}"
        for v, n in zip(VERDICTS, [2, 0, 0, 0, 1] + [0] * 5 + [1, 1], strict=True)
    ]
    assert lines == [
        "graded 3 releases, 5 figures (1 not in input, 1 without an answer)",
        "accuracy: 2/3 (66.7%)",
        "by field:",
        "  revenue       1/1 (100.0%)",
        "  net_income    1/2 (50.0%)",
        "  eps_diluted   0/0",
        "by period type:",
        "  quarter       1/2 (50.0%)",
        "  year_to_date  1/1 (100.0%)",
        "by ticker:",
        "  AAPL          1/1 (100.0%)",
        "  MSFT          0/0",
        "  XOM           1/2 (50.0%)",
        "verdicts:",
        *verdict_lines,
        "not in input: 1",
        f"  XOM {RELEASE} eps_diluted quarter (answer 1): e",
        "period_end differs from the linked filing: 1",
        "  AAPL x: predicted y",
        "predictions for releases not linked (not graded): 1",
        "  0000320187-26-000070",
        "linked releases without a prediction: 57",
    ]


# Command line, on the fixtures


def make_client(cache_dir: Path, routes: dict[str, bytes] | None = None) -> EdgarClient:
    """A client serving answers_companyfacts.json (URL -> JSON); other URLs 404."""
    if routes is None:
        bodies = json.loads(COMPANYFACTS.read_bytes())
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


@pytest.fixture(scope="module")
def key_dir(tmp_path_factory) -> Path:
    """A folder with answers.csv built from the answers fixtures."""
    tmp = tmp_path_factory.mktemp("key")
    with contextlib.redirect_stdout(io.StringIO()):
        status = answers.main(
            ["--links", str(LINKS), "--out", str(tmp / "answers.csv")],
            client=make_client(tmp / "raw"),
        )
    assert status == 0
    return tmp


XOM = "0002115436-26-000006"  # Q2 2026, statements in EX-99.2
AAPL = "0000320193-25-000055"  # Q2 FY25
CAT = "0000018230-25-000037"  # Q2 2025
MSFT = "0000950170-25-100226"  # FY25, a 10-K
NKE_CFO_NEWS = "0000320187-26-000070"  # excluded from links


def figures(revenue=None, net_income=None, eps_diluted=None) -> dict:
    return {"revenue": revenue, "net_income": net_income, "eps_diluted": eps_diluted}


RUN_PREDICTIONS = [
    {
        "release_accession": XOM,
        "period_end": "2026-06-30",
        "quarter": figures(116017000000, 14680000000, 3.48),
        "year_to_date": figures(201155000000, 18708000000, 4.47),
    },
    {
        "release_accession": AAPL,
        "period_end": "2025-03-31",  # the filing's period ends 2025-03-29
        "quarter": figures(95400000000, 24780000000, None),
        "year_to_date": figures(),
    },
    {
        "release_accession": CAT,
        "period_end": "2025-06-30",
        "quarter": figures(30818000000, 2179000000, 4.62),
        "year_to_date": figures(30818000000, 4182000000, 8.82),
    },
    {
        "release_accession": MSFT,
        "period_end": "2025-06-30",
        "quarter": figures(76441000000),
        "year_to_date": figures(281724000000, 101832000000, 13.64),
    },
    {
        "release_accession": NKE_CFO_NEWS,
        "period_end": "2026-05-31",
        "quarter": figures(),
        "year_to_date": figures(),
    },
]


def write_run(folder: Path, predictions: list[dict]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "predictions.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in predictions)
    )
    return folder


def grade_args(run: Path, key_dir: Path, *extra: str) -> list[str]:
    return [
        str(run),
        "--links",
        str(LINKS),
        "--answers",
        str(key_dir / "answers.csv"),
        "--releases",
        str(RELEASES),
        *extra,
    ]


@pytest.fixture(scope="module")
def graded_run(key_dir, tmp_path_factory) -> tuple[int, str, Path]:
    run = write_run(tmp_path_factory.mktemp("run"), RUN_PREDICTIONS)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        status = grade.main(
            grade_args(run, key_dir), client=make_client(key_dir / "raw")
        )
    return status, stdout.getvalue(), run


def test_cli_summary(graded_run) -> None:
    status, stdout, run = graded_run
    assert status == 0
    lines = stdout.splitlines()
    counts = {
        CORRECT: 14,
        MISSING: 4,
        WRONG_PERIOD: 1,
        ROUNDED: 1,
        ADJUSTED: 1,
        NO_ANSWER: 1,
    }
    assert lines == [
        "graded 4 releases, 22 figures (0 not in input, 1 without an answer)",
        "accuracy: 14/21 (66.7%)",
        "by field:",
        "  revenue       4/7 (57.1%)",
        "  net_income    5/7 (71.4%)",
        "  eps_diluted   5/7 (71.4%)",
        "by period type:",
        "  quarter       5/9 (55.6%)",
        "  year_to_date  6/9 (66.7%)",
        "  full_year     3/3 (100.0%)",
        "by ticker:",
        "  AAPL          1/6 (16.7%)",
        "  CAT           5/6 (83.3%)",
        "  MSFT          3/3 (100.0%)",
        "  XOM           5/6 (83.3%)",
        "verdicts:",
        *(f"  {v:<24}{counts.get(v, 0):>6}" for v in VERDICTS),
        "not in input: 0",
        "period_end differs from the linked filing: 1",
        (
            f"  AAPL {AAPL}: predicted 2025-03-31, 10-Q 0000320193-25-000057 ends "
            "2025-03-29"
        ),
        "predictions for releases not linked (not graded): 1",
        f"  {NKE_CFO_NEWS}",
        "linked releases without a prediction: 56",
        f"wrote 22 rows to {run / 'graded.csv'}",
    ]
    assert (run / "summary.txt").read_text() == "\n".join(lines[:-1]) + "\n"


def test_cli_graded_csv(graded_run) -> None:
    _, _, run = graded_run
    with (run / "graded.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == GRADED_COLUMNS
    assert len(rows) == 22
    by_key = {(r["release_accession"], r["field"], r["period_type"]): r for r in rows}
    assert by_key[XOM, "net_income", "quarter"] == {
        "ticker": "XOM",
        "release_accession": XOM,
        "linked_accession": "0000034088-26-000093",
        "field": "net_income",
        "period_type": "quarter",
        "answer": "14525000000",
        "answer_tag": "NetIncomeLoss",
        "predicted": "14680000000",
        "verdict": ADJUSTED,
        "evidence": "'Adjusted Earnings/(Loss) (non-GAAP)': 14,680 (x1,000,000)",
    }
    assert by_key[XOM, "revenue", "quarter"]["verdict"] == CORRECT  # from EX-99.2
    assert by_key[AAPL, "revenue", "quarter"]["evidence"] == (
        "answer rounded to 100,000,000"
    )
    assert by_key[AAPL, "eps_diluted", "year_to_date"]["predicted"] == ""
    assert by_key[CAT, "revenue", "quarter"]["evidence"] == (
        "Revenues 2025-01-01..2025-06-30: 30818000000"
    )
    assert by_key[MSFT, "revenue", "quarter"]["verdict"] == NO_ANSWER
    assert by_key[MSFT, "net_income", "full_year"]["verdict"] == CORRECT


def test_cli_errors(key_dir, tmp_path, capsys) -> None:
    client = make_client(key_dir / "raw")

    def run_error(run: Path, *extra: str) -> str:
        assert grade.main(grade_args(run, key_dir, *extra), client=client) == 1
        return capsys.readouterr().err

    empty = tmp_path / "empty"
    empty.mkdir()
    assert "predictions.jsonl not found" in run_error(empty)

    bad = write_run(tmp_path / "bad", [])
    (bad / "predictions.jsonl").write_text('{"release_accession": "x"}\n')
    assert "line 1: period_end: Field required" in run_error(bad)

    run = write_run(tmp_path / "run", RUN_PREDICTIONS[:1])
    stale = tmp_path / "stale.csv"
    stale.write_text(
        (key_dir / "answers.csv").read_text().replace("116017000000", "116018000000")
    )
    err = run_error(run, "--answers", str(stale))
    assert "doesn't match the company facts; rerun secbench.answers" in err
    assert f"{XOM} revenue quarter: answers.csv has 116018000000" in err

    assert f"no saved release {XOM} in" in run_error(run, "--releases", str(empty))
    assert not (run / "graded.csv").exists()


MSFT_10K = "0000950170-25-100235"
MSFT_Q3_10Q = "0000950170-25-061046"  # its fiscal year's third-quarter 10-Q
RFC = "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_cli_checks_a_derived_answer(key_dir, tmp_path, capsys) -> None:
    # Microsoft's fourth-quarter FY25 revenue: 281,724 for the year minus 205,283
    # for nine months. The nine months come from a filing not linked to this run.
    def key_with(value: str) -> Path:
        lines = [
            line
            for line in (key_dir / "answers.csv").read_text().splitlines()
            if not line.startswith(f"MSFT,{MSFT},{MSFT_10K},revenue,quarter,")
        ]
        lines.append(
            f"MSFT,{MSFT},{MSFT_10K},revenue,quarter,2025-04-01,2025-06-30,{value},"
            f"USD,{RFC},True,{MSFT_10K},{MSFT_Q3_10Q},2024-07-01"
        )
        path = tmp_path / f"answers_{value}.csv"
        path.write_text("\n".join(lines) + "\n")
        return path

    client = make_client(key_dir / "raw")
    run = write_run(tmp_path / "run", [RUN_PREDICTIONS[3]])  # MSFT only
    argv = grade_args(run, key_dir, "--answers", str(key_with("76441000000")))
    with contextlib.redirect_stdout(io.StringIO()):
        assert grade.main(argv, client=client) == 0
    with (run / "graded.csv").open(newline="") as f:
        rows = {(r["field"], r["period_type"]): r for r in csv.DictReader(f)}
    assert rows["revenue", "quarter"]["verdict"] == CORRECT
    assert rows["revenue", "quarter"]["answer_tag"] == RFC

    argv = grade_args(run, key_dir, "--answers", str(key_with("76442000000")))
    assert grade.main(argv, client=client) == 1
    assert (
        f"{MSFT} revenue quarter: answers.csv has 76442000000, company facts full "
        f"year 281724000000 ({MSFT_10K}) minus nine months 205283000000 "
        f"({MSFT_Q3_10Q}) = 76441000000"
    ) in capsys.readouterr().err


def test_cli_company_facts_unavailable(key_dir, tmp_path, capsys) -> None:
    run = write_run(tmp_path / "run", RUN_PREDICTIONS[:1])
    client = make_client(tmp_path / "raw", routes={})
    assert grade.main(grade_args(run, key_dir), client=client) == 1
    assert "error: company facts:" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["runs/x", "--check-inputs"]])
def test_cli_needs_a_run_or_check_inputs(argv) -> None:
    with pytest.raises(SystemExit) as raised:
        grade.main(argv)
    assert raised.value.code == 2


# --check-inputs


def fixture_answers(key_dir: Path, folder: Path) -> Path:
    """answers.csv cut to the releases in input_releases."""
    lines = (key_dir / "answers.csv").read_text().splitlines()
    kept = [lines[0]] + [
        line for line in lines[1:] if line.split(",")[1] in {XOM, AAPL, CAT, MSFT}
    ]
    path = folder / "answers.csv"
    path.write_text("\n".join(kept) + "\n")
    return path


def test_check_inputs(key_dir, tmp_path, capsys) -> None:
    argv = ["--check-inputs", "--answers", str(fixture_answers(key_dir, tmp_path))]
    assert grade.main([*argv, "--releases", str(RELEASES)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "answers in input: 21 of 21"
    assert lines[-1] == "not in input: 0"


def test_check_inputs_lists_answers_not_in_input(key_dir, tmp_path, capsys) -> None:
    # Without EX-99.2, ExxonMobil's Q2 2026 input has no revenue.
    releases = tmp_path / "releases"
    shutil.copytree(RELEASES, releases)
    record = releases / "0002115436" / f"{XOM}.json"
    saved = json.loads(record.read_text())
    for exhibit in saved["exhibits"]:
        if exhibit["type"] == "EX-99.2":
            exhibit["file"] = None
    record.write_text(json.dumps(saved))
    argv = ["--check-inputs", "--answers", str(fixture_answers(key_dir, tmp_path))]
    assert grade.main([*argv, "--releases", str(releases)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "answers in input: 19 of 21"
    assert lines[-3:] == [
        "not in input: 2",
        (
            f"  XOM {XOM} revenue quarter (answer 116017000000): searched 116,017, "
            "116,017,000"
        ),
        (
            f"  XOM {XOM} revenue year_to_date (answer 201155000000): searched 201,155, "
            "201,155,000"
        ),
    ]


# Hand-checked grades (tests/fixtures/grade_handchecked.csv)


@pytest.fixture(scope="module")
def fixture_context(key_dir):
    """Answer key, filing facts and input texts for the releases in input_releases."""
    key = load_answers(key_dir / "answers.csv")
    records = release_records(RELEASES)
    releases = {
        r.linked_accession: r
        for r in load_links(LINKS)[0]
        if r.release_accession in records
    }
    facts = load_filing_facts(make_client(key_dir / "raw"), releases.values())
    texts = {
        r.release_accession: build_input(records[r.release_accession]).text
        for r in releases.values()
    }
    return key, releases, facts, texts


WMT = "0000104169-25-000069"  # Q1 FY26


def grade_real(context, accession: str, field: str, period: str, predicted: str):
    key, releases, facts, texts = context
    release = next(r for r in releases.values() if r.release_accession == accession)
    text = texts[accession]
    return grade_value(
        key[accession, field, period],
        D(predicted),
        text,
        input_rows(text),
        facts.get(release.linked_accession, set()),
    )


def test_real_noncontrolling_interest_wmt(fixture_context) -> None:
    # Walmart's "Consolidated net income" (4,639) includes $152 million for
    # noncontrolling interests; the answer is the line attributable to Walmart.
    assert "| Consolidated net income | 4,639 |" in fixture_context[3][WMT]
    assert grade_real(fixture_context, WMT, "net_income", "quarter", "4639000000") == (
        grade.Verdict(
            NONCONTROLLING_INTEREST, "ProfitLoss 2025-02-01..2025-04-30: 4639000000"
        )
    )


def test_real_other_period_column_xom(fixture_context) -> None:
    # The filing tags Q2 and the half years, not the first quarter.
    assert grade_real(fixture_context, XOM, "net_income", "quarter", "4183000000") == (
        grade.Verdict(
            WRONG_PERIOD, "'1Q26' column of 'Earnings (U.S. GAAP)': 4,183 (x1,000,000)"
        )
    )


def test_real_adjusted_section_xom(fixture_context) -> None:
    assert grade_real(fixture_context, XOM, "net_income", "quarter", "9189000000") == (
        grade.Verdict(
            ADJUSTED,
            "'Upstream' (section 'Adjusted Earnings/(Loss) (non-GAAP)'): "
            "9,189 (x1,000,000)",
        )
    )


def handchecked_rows() -> list:
    if not HANDCHECKED.exists():
        return [pytest.param(None, id="file-missing")]
    with HANDCHECKED.open(newline="", encoding="utf-8") as f:
        return [
            pytest.param(
                row,
                id=f"{i}-{row['ticker']}-{row['field']}-{row['period_type']}-"
                f"{row['expected_verdict']}".replace(" ", ""),
            )
            for i, row in enumerate(csv.DictReader(f), 2)
        ]


def test_handchecked_file_is_well_formed() -> None:
    assert HANDCHECKED.exists(), f"{HANDCHECKED} is missing"
    with HANDCHECKED.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == HANDCHECKED_COLUMNS
    assert rows
    for row in rows:
        assert row["field"].strip() in FIELD_NAMES, row
        assert row["period_type"].strip() in PERIOD_TYPES, row
        assert row["expected_verdict"].strip() in VERDICTS, row
        if row["predicted_value"].strip():
            D(row["predicted_value"].strip())


@pytest.mark.parametrize("expected", handchecked_rows())
def test_handchecked_grade(expected, fixture_context) -> None:
    assert expected is not None, f"{HANDCHECKED} is missing"
    key, releases, facts, texts = fixture_context
    ticker, accession, field, period = (
        expected[c].strip() for c in HANDCHECKED_COLUMNS[:4]
    )
    release = releases.get(accession)
    assert release is not None, f"{accession} is not a release in {RELEASES.name}"
    assert release.ticker == ticker
    raw = expected["predicted_value"].strip()
    group = "quarter" if period == "quarter" else "year_to_date"
    prediction = Prediction.model_validate(
        {
            "release_accession": release.release_accession,
            "period_end": release.period_end,
            "quarter": {},
            "year_to_date": {},
            group: {field: raw or None},
        }
    )
    graded = grade_prediction(
        prediction,
        release,
        key,
        facts.get(accession, set()),
        texts[release.release_accession],
    )
    got = next((g for g in graded if (g.field, g.period_type) == (field, period)), None)
    assert got is not None, f"no {field} {period} answer for {accession}"
    assert got.verdict == expected["expected_verdict"].strip(), (
        f"answer {got.answer} ({got.answer_tag}); evidence: {got.evidence}"
    )
