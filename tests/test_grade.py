import contextlib
import csv
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
    assert rows[0].figures == (Figure("$14,525 million", D(14525), 10**6),)


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
        ("7082000000", "'Net income': 7,082 (x1,000,000)"),  # a column without XBRL
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


def test_wrong_line_eps_figures_are_not_scaled() -> None:
    expected = answer("3.48", "eps_diluted", "EarningsPerShareDiluted")
    assert verdict("1.64", expected) == grade.Verdict(
        WRONG_LINE, "'Diluted earnings per share': 1.64"
    )
    assert verdict("0.00164", expected).verdict == OTHER


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
def handchecked_context(key_dir):
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
def test_handchecked_grade(expected, handchecked_context) -> None:
    assert expected is not None, f"{HANDCHECKED} is missing"
    key, releases, facts, texts = handchecked_context
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
