"""Grade model predictions against the answer key.

A prediction gives, for one release, revenue, net income and diluted EPS for the
quarter and for the year to date (the full year in a fourth-quarter release), in raw
dollars or null. Each figure is graded against data/answers.csv (see
secbench.answers) only if the answer appears in the release's input text (see
secbench.inputs); otherwise it is not_in_input and left out of accuracy. A wrong
figure is classified using the linked filing's other XBRL facts and the input's rows:
a figure in another period column of the answer's own row ("1Q26") is a wrong
period, and a row counts as adjusted by its label, the label-only row above it in its
table, its column header or the heading above its table. A figure off by a power of
ten from 10 to 1,000,000 is a scale error (1,000 or 1,000,000) or miscounted zeros
(the others); after every exact check, the sign, period, noncontrolling-interest,
rounded, adjusted and line checks are tried again on the prediction times each such
factor. graded.csv's factor column holds it: the prediction times the factor is what
was matched (0.01 for "prediction /100 is the answer").

Usage: uv run python -m secbench.grade runs/<run>/ [--links] [--answers] [--releases]
       uv run python -m secbench.grade --check-inputs [--answers] [--releases]

runs/<run>/predictions.jsonl holds one Prediction per line; grading writes
runs/<run>/graded.csv and runs/<run>/summary.txt. --check-inputs runs the answerable
check on every answer, without predictions.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from dataclasses import field as dataclass_field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import cached_property
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from secbench.answers import COLUMNS as ANSWER_COLUMNS
from secbench.answers import DEFAULT_OUT as DEFAULT_ANSWERS
from secbench.answers import (
    FIELDS,
    FULL_YEAR,
    QUARTER,
    YEAR_TO_DATE,
    Answer,
    Fact,
    LinkedRelease,
    index_facts,
    load_company_facts,
    load_links,
    plural,
)
from secbench.edgar import EdgarClient, EdgarError, default_client
from secbench.inputs import build_input, release_records
from secbench.link import CELL_SPLIT, clean_label, row_labels, shorten
from secbench.link import DEFAULT_OUT as DEFAULT_LINKS
from secbench.releases import DEFAULT_OUT_DIR as DEFAULT_RELEASES
from secbench.releases import write_text

log = logging.getLogger(__name__)

PREDICTIONS = "predictions.jsonl"
GRADED = "graded.csv"
SUMMARY = "summary.txt"

FIELD_NAMES = [field.name for field in FIELDS]
NET_INCOME = "net_income"
EPS = "eps_diluted"
PROFIT_LOSS = "ProfitLoss"  # net income including noncontrolling interests
EPS_PRECISION = Decimal("0.01")
MAX_USD_PRECISION = 10**6  # an answer's precision is at most whole millions
# Powers of ten a prediction may be off by, smallest first, multiplying before
# dividing. The answer off by 1,000 or 1,000,000 is a scale error (a unit misread);
# off by another, zeros miscounted ("16,569" in millions written 1656900000000).
FACTORS = tuple(Decimal(10) ** (k * s) for k in range(1, 7) for s in (1, -1))
SCALES = {Decimal(10) ** e for e in (3, -3, 6, -6)}
# A table figure's unit isn't parsed from its heading, so each is tried.
TABLE_UNITS = (1, 10**3, 10**6)
SCALE_WORDS = {"thousand": 10**3, "million": 10**6, "billion": 10**9}
PERIOD_ORDER = (QUARTER, YEAR_TO_DATE, FULL_YEAR)
# Prediction groups; year_to_date is graded as full_year for a 10-K.
GROUPS = ("quarter", "year_to_date")

CORRECT = "correct"
MISSING = "missing"
SIGN = "sign"
SCALE = "scale"
ZEROS = "zeros"
WRONG_PERIOD = "wrong_period"
NONCONTROLLING_INTEREST = "noncontrolling_interest"
ROUNDED = "rounded"
ADJUSTED = "adjusted"
WRONG_LINE = "wrong_line"
OTHER = "other"
NOT_IN_INPUT = "not_in_input"  # the answer isn't in the input: not graded
NO_ANSWER = "no_answer"  # a figure for a period without an answer: not graded
VERDICTS = (
    CORRECT,
    MISSING,
    SIGN,
    SCALE,
    ZEROS,
    WRONG_PERIOD,
    NONCONTROLLING_INTEREST,
    ROUNDED,
    ADJUSTED,
    WRONG_LINE,
    OTHER,
    NOT_IN_INPUT,
    NO_ANSWER,
)
UNGRADED = (NOT_IN_INPUT, NO_ANSWER)

ADJUSTED_LABEL = re.compile(
    r"\badjusted\b|\bnon[-\s]?gaap\b|\bexcluding\b", re.IGNORECASE
)
# Says GAAP (not non-GAAP): nearer than an "adjusted" heading, it keeps a row GAAP.
GAAP_LABEL = re.compile(r"(?<!non-)(?<!non )\bgaap\b", re.IGNORECASE)
LOSS_LABEL = re.compile(r"\bloss(es)?\b", re.IGNORECASE)
# A column header naming a period: "1Q26", "YTD 2026", "Three Months Ended June 30".
PERIOD_HEADER = re.compile(
    r"\b(19|20)\d\d\b|\b[1-4]Q\d\d\b|\bQ[1-4]\b"
    r"|\b(quarters?|months?|weeks?|years?|ytd|fiscal|ended)\b",
    re.IGNORECASE,
)
# A word of a period header. Two columns are different periods when their headers
# differ only in such words ("2Q26"/"1Q26", "Three Months Ended March 29, 2025"/"Six
# Months Ended March 30, 2024"), not in a measure ("..., Revenue"/"..., Operating
# Income"), a change ("Change vs 1Q26") or a basis ("GAAP"/"Adjusted").
PERIOD_WORD = re.compile(
    r"(19|20)\d\d|\d{1,2}|[1-4]q\d\d|q[1-4]|fy\d{2,4}|ytd|year-to-date|fiscal|full"
    r"|three|six|nine|twelve|thirteen|fourteen|twenty-six|twenty-seven|thirty-nine"
    r"|forty|fifty-two|fifty-three|months?|weeks?|ended|ending|quarters?|years?"
    r"|first|second|third|fourth|jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may"
    r"|june?|july?|aug(ust)?|sept?(ember)?|oct(ober)?|nov(ember)?|dec(ember)?"
)
HEADING_WIDTH = 120  # a longer line above a table is a paragraph, not its heading
CONTINUED = re.compile(r"\((more|continued)\)", re.IGNORECASE)
MINUS = "-–−"  # hyphen, en dash, minus sign
MINUS_CLASS = re.escape(MINUS)  # for use inside [...]
NUMBER_START = r"(?<![\d,.])"
NUMBER_END = r"(?!\d|,\d|\.\d|\s*%)"
CELL_FIGURE = re.compile(
    rf"([$(\s{MINUS_CLASS}]*)(\d{{1,3}}(?:,\d{{3}})+|\d+)(\.\d+)?([)\s]*)"
)
PROSE_FIGURE = re.compile(
    rf"(?<![\w.,])(?P<sign>\(|[{MINUS_CLASS}])?\s*\$?\s*"
    r"(?P<number>\d{1,3}(?:,\d{3})+|\d+)(?P<decimals>\.\d+)?\)?"
    r"(?:\s*(?P<word>thousand|million|billion)\b)?(?P<percent>\s*(?:%|percent\b))?",
    re.IGNORECASE,
)


# Predictions

Value = Annotated[Decimal, Field(allow_inf_nan=False)] | None


class Figures(BaseModel):
    """Raw dollars (EPS in dollars per share), or null."""

    model_config = ConfigDict(extra="forbid")

    revenue: Value = None
    net_income: Value = None
    eps_diluted: Value = None


class Prediction(BaseModel):
    """A model's figures for one release. year_to_date is the full year in a
    fourth-quarter release."""

    model_config = ConfigDict(extra="forbid")

    release_accession: str
    period_end: date
    quarter: Figures
    year_to_date: Figures


assert list(Figures.model_fields) == FIELD_NAMES


class PredictionsError(ValueError):
    """predictions.jsonl can't be read."""


def load_predictions(path: Path) -> list[Prediction]:
    """One Prediction per non-blank line; every bad or repeated line is reported."""
    predictions: list[Prediction] = []
    problems = []
    seen: dict[str, int] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            prediction = Prediction.model_validate_json(line)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(map(str, first["loc"]))
            problems.append(
                f"line {number}: " + (f"{where}: " if where else "") + first["msg"]
            )
            continue
        accession = prediction.release_accession
        if accession in seen:
            problems.append(
                f"line {number}: {accession} is also on line {seen[accession]}"
            )
            continue
        seen[accession] = number
        predictions.append(prediction)
    if problems:
        raise PredictionsError(f"{path}: " + "; ".join(problems))
    return predictions


def period_type(group: str, form: str) -> str:
    if group == "quarter":
        return QUARTER
    return FULL_YEAR if form == "10-K" else YEAR_TO_DATE


# Answer key


def load_answers(path: Path) -> dict[tuple[str, str, str], Answer]:
    """answers.csv by (release accession, field, period type)."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if reader.fieldnames != ANSWER_COLUMNS:
        raise GradeError(
            f"{path} has columns {reader.fieldnames}, not {ANSWER_COLUMNS}; rerun "
            "secbench.answers"
        )
    key = {}
    for number, row in enumerate(rows, 2):
        if row["derived"] not in ("True", "False"):
            raise GradeError(
                f"{path} line {number}: derived is {row['derived']!r}, not True or "
                "False"
            )
        start = row["fiscal_year_start"]
        answer = Answer(
            ticker=row["ticker"],
            release_accession=row["release_accession"],
            linked_accession=row["linked_accession"],
            field=row["field"],
            period_type=row["period_type"],
            start=date.fromisoformat(row["start"]),
            end=date.fromisoformat(row["end"]),
            value=Decimal(row["value"]),
            unit=row["unit"],
            tag=row["tag"],
            derived=row["derived"] == "True",
            full_year_accession=row["full_year_accession"],
            nine_months_accession=row["nine_months_accession"],
            fiscal_year_start=date.fromisoformat(start) if start else None,
        )
        key[answer.release_accession, answer.field, answer.period_type] = answer
    return key


def load_filing_facts(
    client: EdgarClient,
    releases: Iterable[LinkedRelease],
    answers: Iterable[Answer] = (),
) -> dict[str, set[Fact]]:
    """Accession -> the us-gaap facts of that filing, from the cached company facts
    of every CIK of each ticker: each release's linked filing, and the nine-month
    10-Q of each derived answer."""
    by_ticker: dict[str, set[str]] = defaultdict(set)
    for release in releases:
        by_ticker[release.ticker].add(release.linked_accession)
    for answer in answers:
        if answer.derived:
            by_ticker[answer.ticker].add(answer.nine_months_accession)
    facts: dict[str, set[Fact]] = {}
    for ticker, accessions in sorted(by_ticker.items()):
        documents = [
            load_company_facts(client, cik) for cik in client.ticker_to_ciks(ticker)
        ]
        facts |= index_facts(documents, accessions).by_accession
    return facts


def check_key(answers: Iterable[Answer], facts: dict[str, set[Fact]]) -> list[str]:
    """Answers that don't match the company facts (answers.csv older than them). An
    answer from one fact must be the linked filing's fact for its tag and period; a
    derived answer is recomputed (see derived_mismatch)."""
    problems = []
    for a in answers:
        if a.derived:
            mismatch = derived_mismatch(a, facts)
        else:
            values = tagged_values(facts, a.linked_accession, a, a.start, a.end)
            mismatch = None if values == [a.value] else f"company facts {show(values)}"
        if mismatch:
            problems.append(
                f"{a.release_accession} {a.field} {a.period_type}: answers.csv has "
                f"{a.value}, {mismatch}"
            )
    return problems


def derived_mismatch(a: Answer, facts: dict[str, set[Fact]]) -> str | None:
    """How a derived answer differs from the linked filing's full year minus the
    nine months in its nine_months_accession (both of its tag, from its
    fiscal_year_start, the nine months ending the day before its start), or None."""
    if a.full_year_accession != a.linked_accession:
        return (
            f"derived from {a.full_year_accession or 'no filing'}, not the linked one"
        )
    if a.fiscal_year_start is None:
        return "derived without a fiscal_year_start"
    full_year = tagged_values(
        facts, a.full_year_accession, a, a.fiscal_year_start, a.end
    )
    nine_months = tagged_values(
        facts,
        a.nine_months_accession,
        a,
        a.fiscal_year_start,
        a.start - timedelta(days=1),
    )
    text = (
        f"company facts full year {show(full_year)} ({a.full_year_accession}) minus "
        f"nine months {show(nine_months)} ({a.nine_months_accession or 'no filing'})"
    )
    if len(full_year) == 1 and len(nine_months) == 1:
        difference = full_year[0] - nine_months[0]
        if difference == a.value:
            return None
        text += f" = {difference}"
    return text


def tagged_values(
    facts: dict[str, set[Fact]], accession: str, a: Answer, start: date, end: date
) -> list[Decimal]:
    """The values of a's tag and unit for start..end in a filing."""
    return sorted(
        {
            f.value
            for f in facts.get(accession, ())
            if (f.tag, f.unit, f.start, f.end) == (a.tag, a.unit, start, end)
        }
    )


def show(values: list[Decimal]) -> str:
    return ", ".join(map(str, values)) or "no fact"


# Input rows


@dataclass(frozen=True)
class Figure:
    text: str  # as shown: "$(1,234)", "95.4 billion"
    value: Decimal  # signed, in the unit shown
    scale: int | None = None  # from a word ("billion"); None if the unit isn't given
    column: int | None = None  # a table cell's column; None in prose


@dataclass(frozen=True)
class InputRow:
    """A table body row (labelled by its text cells) or a line of prose (labelled by
    itself). A table row also has its table's header row, the nearest label-only
    row above it in the table (XOM's "Adjusted Earnings/(Loss) (non-GAAP)") and the
    heading above the table."""

    label: str
    line: str
    figures: tuple[Figure, ...]
    header: tuple[str, ...] = ()  # header cells, by column
    section: str = ""
    heading: str = ""

    def column_header(self, figure: Figure) -> str:
        if figure.column is None or figure.column >= len(self.header):
            return ""
        return self.header[figure.column]


def cell_figure(cell: str, column: int | None = None) -> Figure | None:
    text = clean_label(cell)
    if "%" in text:
        return None
    match = CELL_FIGURE.fullmatch(text)
    if match is None:
        return None
    prefix, number, decimals = match[1], match[2], match[3] or ""
    value = Decimal(number.replace(",", "") + decimals)
    negative = "(" in prefix or any(c in prefix for c in MINUS)
    return Figure(text, -value if negative else value, None, column)


def prose_figures(line: str) -> tuple[Figure, ...]:
    figures = []
    for m in PROSE_FIGURE.finditer(line):
        if m["percent"]:
            continue
        value = Decimal(m["number"].replace(",", "") + (m["decimals"] or ""))
        word = (m["word"] or "").lower()
        figures.append(
            Figure(
                m[0].strip(),
                -value if m["sign"] else value,
                SCALE_WORDS[word] if word else None,
            )
        )
    return tuple(figures)


def is_heading(text: str) -> bool:
    """Short, and not a sentence: "RECONCILIATION OF ADJUSTED DILUTED EARNINGS PER
    SHARE" or "...are as follows:", not "Please see a reconciliation ... pages 12 and
    13." or a paragraph."""
    return 0 < len(text) <= HEADING_WIDTH and not text.endswith(".")


def table_heading(lines: list[str], start: int) -> str:
    """The heading above the table whose header row is lines[start]: the first line
    above it past blank lines, lines without letters (page numbers), footnotes,
    "(more)" and tables directly above (a table split across pages), if is_heading."""
    for line in reversed(lines[:start]):
        text = clean_label(line).lstrip("# ")
        if (
            line.startswith("|")
            or line.lstrip().startswith("<sup>")
            or not re.search(r"[A-Za-z]", text)
            or CONTINUED.fullmatch(text)
        ):
            continue
        return text if is_heading(text) else ""
    return ""


def input_rows(text: str) -> list[InputRow]:
    """The input's table body rows and prose lines, in order. Header rows (the first
    row of each table) and separator rows are not rows themselves."""
    rows = []
    lines = text.splitlines()
    header: tuple[str, ...] | None = None  # None outside a table
    section = heading = ""
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            header = None
            if label := clean_label(line):
                rows.append(InputRow(label, line, prose_figures(line)))
            continue
        cells = [c.strip().replace(r"\|", "|") for c in CELL_SPLIT.split(line)[1:-1]]
        if header is None:
            header = tuple(clean_label(c) for c in cells)
            section, heading = "", table_heading(lines, i)
            continue
        if all(c == "---" for c in cells):
            continue
        figures = tuple(f for j, c in enumerate(cells) if (f := cell_figure(c, j)))
        label = " ".join(row_labels(cells))
        rows.append(InputRow(label, line, figures, header, section, heading))
        if not figures and is_heading(label):
            section = label
    return rows


# Answerable check


def input_forms(value: Decimal, field: str) -> list[str]:
    """The answer as an input could show it, unsigned: EPS with two decimals; USD in
    every unit it is whole in, millions and thousands ("94,036", "94,036,000")."""
    magnitude = abs(value)
    if field == EPS:
        cents = magnitude.quantize(EPS_PRECISION)
        return [f"{cents}"] if cents == magnitude else []
    forms = [
        f"{int(magnitude) // unit:,}"
        for unit in (10**6, 10**3)
        if magnitude % unit == 0
    ]
    return list(dict.fromkeys(forms))


def find_in_input(
    value: Decimal, field: str, text: str, rows: Sequence[InputRow]
) -> str | None:
    """How the answer appears in the input, or None. A negative answer appears in
    parentheses, after a minus sign (-, –, −), or unsigned on a row whose label says
    loss ("Net loss 1,234")."""
    for form in input_forms(value, field):
        number = re.escape(form)
        unsigned = NUMBER_START + number + NUMBER_END
        if value >= 0:
            if re.search(unsigned, text):
                return form
            continue
        if re.search(rf"\(\s*\$?\s*{number}\s*\)", text):
            return f"({form})"
        if re.search(rf"[{MINUS_CLASS}]\s*\$?\s*{number}{NUMBER_END}", text):
            return f"-{form}"
        for row in rows:
            if LOSS_LABEL.search(row.label) and re.search(unsigned, row.line):
                return f"{form} on a loss row ({shorten(row.label)})"
    return None


# Grading one figure


def precision(value: Decimal, field: str) -> Decimal:
    """A value's own precision: a cent for EPS; for USD, the largest power of ten
    (up to a million) that divides it."""
    if field == EPS:
        return EPS_PRECISION
    if value != value.to_integral_value():
        return Decimal(1)
    whole, unit = abs(int(value)), 1
    while unit < MAX_USD_PRECISION and whole % (unit * 10) == 0:
        unit *= 10
    return Decimal(unit)


def close(predicted: Decimal, reference: Decimal, field: str) -> bool:
    """Within half the reference's precision (a value exactly half a unit away rounds
    to the next unit, so it is not close)."""
    return abs(predicted - reference) < precision(reference, field) / 2


def rounding_unit(answer: Decimal, field: str, predicted: Decimal) -> Decimal | None:
    """The coarser unit (no bigger than the answer) that the answer, rounded half up
    to it, equals predicted at, if any: 95,359,000,000 -> 95,400,000,000 at 10^8."""
    unit = precision(answer, field) * 10
    while unit <= abs(answer):
        rounded = (answer / unit).quantize(Decimal(1), rounding=ROUND_HALF_UP) * unit
        if rounded == predicted:
            return unit
        unit *= 10
    return None


def describe_fact(fact: Fact) -> str:
    period = str(fact.end) if fact.start is None else f"{fact.start}..{fact.end}"
    return f"{fact.tag} {period}: {fact.value}"


def figure_units(figure: Figure, field: str) -> tuple[int, ...]:
    """The units a figure may be in: EPS as shown; USD by its scale word, else
    dollars, thousands or millions (a table's unit isn't parsed from its heading)."""
    if field == EPS:
        return (1,) if figure.scale is None else ()
    return (figure.scale,) if figure.scale else TABLE_UNITS


def match_figure(
    predicted: Decimal, field: str, candidates: Iterable[tuple[InputRow, Figure]]
) -> tuple[InputRow, Figure, int] | None:
    """The first (row, figure) that equals predicted in one of its units."""
    for row, figure in candidates:
        for unit in figure_units(figure, field):
            if figure.value * unit == predicted:
                return row, figure, unit
    return None


def is_period_column(header: str) -> bool:
    return bool(PERIOD_HEADER.search(header))


def header_words(header: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", header.lower()))


def other_period(header: str, answer_header: str) -> bool:
    """Whether header names another period than answer_header: they differ, and only
    in period words."""
    differ = header_words(header) ^ header_words(answer_header)
    return bool(differ) and all(PERIOD_WORD.fullmatch(w) for w in differ)


def other_period_column(
    answer: Decimal, predicted: Decimal, field: str, rows: Iterable[InputRow]
) -> tuple[InputRow, Figure, int] | None:
    """predicted in another period column of a table row that shows the answer: a
    period the filing may not tag (XOM's 1Q26 beside 2Q26). The answer's cell must
    not be adjusted, and the two column headers must differ only in period words, so
    columns that are measures (Microsoft's "..., Revenue" and "..., Operating
    Income") don't count. On a loss row showing the answer unsigned, predicted is
    read the same way."""
    for row in rows:
        periods = [f for f in row.figures if is_period_column(row.column_header(f))]
        if len(periods) < 2:
            continue
        flips = (1, -1) if answer < 0 and LOSS_LABEL.search(row.label) else (1,)
        for unit in figure_units(periods[0], field):
            for flip in flips:
                shown = [
                    row.column_header(f)
                    for f in periods
                    if f.value * unit * flip == answer
                    and adjusted_context(row, f) is None
                ]
                for f in periods:
                    if f.value * unit * flip == predicted and any(
                        other_period(row.column_header(f), h) for h in shown
                    ):
                        return row, f, unit
    return None


def adjusted_context(row: InputRow, figure: Figure) -> tuple[str, str] | None:
    """What marks a figure as adjusted, as (where, text): the first of the row's
    label, its section row, its column header and its table's heading that says
    adjusted, non-GAAP or excluding. None if none does, or if a nearer one says GAAP
    (Caterpillar's "... - U.S. GAAP" rows under an adjusted-results heading)."""
    for where, text in (
        ("label", row.label),
        ("section", row.section),
        ("column", row.column_header(figure)),
        ("heading", row.heading),
    ):
        if ADJUSTED_LABEL.search(text):
            return where, text
        if GAAP_LABEL.search(text):
            return None
    return None


def times(figure: Figure, unit: int) -> str:
    return f" (x{unit:,})" if unit != 1 and figure.scale is None else ""


def describe_match(row: InputRow, figure: Figure, unit: int) -> str:
    context = adjusted_context(row, figure)
    where = ""
    if context and context[0] != "label":
        where = f" ({context[0]} {shorten(context[1], 60)!r})"
    return f"{shorten(row.label, 60)!r}{where}: {figure.text}{times(figure, unit)}"


def describe_column(row: InputRow, figure: Figure, unit: int) -> str:
    return (
        f"{shorten(row.column_header(figure), 60)!r} column of "
        f"{shorten(row.label, 60)!r}: {figure.text}{times(figure, unit)}"
    )


@dataclass(frozen=True)
class Verdict:
    verdict: str
    evidence: str = ""
    factor: Decimal | None = None  # the prediction times it matched; None if as given


def factor_text(factor: Decimal) -> str:
    """What the prediction is multiplied or divided by: "x1,000" or "/1,000"."""
    return f"x{int(factor):,}" if factor > 1 else f"/{int(1 / factor):,}"


def with_factor(found: Verdict, factor: Decimal) -> Verdict:
    """found, matched by the prediction times factor: "prediction /10 is <what it
    matched>"."""
    what = {SIGN: "minus the answer", ROUNDED: f"the {found.evidence}"}.get(
        found.verdict, found.evidence
    )
    return Verdict(found.verdict, f"prediction {factor_text(factor)} is {what}", factor)


@dataclass
class ErrorChecks:
    """The checks for a wrong figure of one answer. Each takes a prediction, as given
    or times a factor, and returns the error it matches, or None."""

    answer: Answer
    rows: Sequence[InputRow]
    filing_facts: Collection[Fact]

    @cached_property
    def other_periods(self) -> list[Fact]:
        """The filing's facts of the answer's tag and unit for other periods, latest
        first."""
        a = self.answer
        return sorted(
            (
                f
                for f in self.filing_facts
                if f.tag == a.tag
                and f.unit == a.unit
                and (f.start, f.end) != (a.start, a.end)
            ),
            key=lambda f: (f.end, f.start or f.end),
            reverse=True,
        )

    @cached_property
    def figures(self) -> list[tuple[InputRow, Figure]]:
        return [(row, figure) for row in self.rows for figure in row.figures]

    @cached_property
    def adjusted_figures(self) -> list[tuple[InputRow, Figure]]:
        return [(r, f) for r, f in self.figures if adjusted_context(r, f)]

    def sign(self, predicted: Decimal) -> Verdict | None:
        value, field = self.answer.value, self.answer.field
        if value != 0 and close(-predicted, value, field):
            return Verdict(SIGN)
        return None

    def wrong_period(self, predicted: Decimal) -> Verdict | None:
        field = self.answer.field
        for fact in self.other_periods:
            if close(predicted, fact.value, field):
                return Verdict(WRONG_PERIOD, describe_fact(fact))
        if found := other_period_column(self.answer.value, predicted, field, self.rows):
            return Verdict(WRONG_PERIOD, describe_column(*found))
        return None

    def noncontrolling_interest(self, predicted: Decimal) -> Verdict | None:
        a = self.answer
        if a.field != NET_INCOME:
            return None
        for fact in self.filing_facts:
            if (fact.tag, fact.unit, fact.start, fact.end) == (
                PROFIT_LOSS,
                a.unit,
                a.start,
                a.end,
            ) and close(predicted, fact.value, a.field):
                return Verdict(NONCONTROLLING_INTEREST, describe_fact(fact))
        return None

    def rounded(self, predicted: Decimal) -> Verdict | None:
        unit = rounding_unit(self.answer.value, self.answer.field, predicted)
        if unit is None:
            return None
        return Verdict(ROUNDED, f"answer rounded to {unit:,}")

    def adjusted(self, predicted: Decimal) -> Verdict | None:
        found = match_figure(predicted, self.answer.field, self.adjusted_figures)
        return Verdict(ADJUSTED, describe_match(*found)) if found else None

    def wrong_line(self, predicted: Decimal) -> Verdict | None:
        found = match_figure(predicted, self.answer.field, self.figures)
        return Verdict(WRONG_LINE, describe_match(*found)) if found else None


def grade_value(
    answer: Answer,
    predicted: Decimal | None,
    text: str,
    rows: Sequence[InputRow],
    filing_facts: Collection[Fact],
) -> Verdict:
    """Grade one predicted figure against its answer. After correct: sign; the
    answer times a factor (scale or zeros); another period, noncontrolling interest,
    rounded, adjusted and another line, as given; then sign and each of those again,
    in that order, on the prediction times each factor (smallest first)."""
    field, value = answer.field, answer.value
    if find_in_input(value, field, text, rows) is None:
        forms = input_forms(value, field)
        return Verdict(
            NOT_IN_INPUT,
            f"searched {', '.join(forms)}" if forms else "no input form for this value",
        )
    if predicted is None:
        return Verdict(MISSING)
    if close(predicted, value, field):
        return Verdict(CORRECT)
    checks = ErrorChecks(answer, rows, filing_facts)
    if found := checks.sign(predicted):
        return found
    for factor in FACTORS:
        if close(predicted * factor, value, field):
            return Verdict(
                SCALE if factor in SCALES else ZEROS,
                f"prediction {factor_text(factor)} is the answer",
                factor,
            )
    exact = (
        checks.wrong_period,
        checks.noncontrolling_interest,
        checks.rounded,
        checks.adjusted,
        checks.wrong_line,
    )
    for check in exact:
        if found := check(predicted):
            return found
    for check in (checks.sign, *exact):
        for factor in FACTORS:
            if found := check(predicted * factor):
                return with_factor(found, factor)
    return Verdict(OTHER)


# Grading a prediction


@dataclass(frozen=True)
class Graded:
    """One row of graded.csv."""

    ticker: str
    release_accession: str
    linked_accession: str
    field: str
    period_type: str
    answer: Decimal | None
    answer_tag: str
    predicted: Decimal | None
    verdict: str
    evidence: str
    factor: Decimal | None = None  # see Verdict.factor


GRADED_COLUMNS = [f.name for f in fields(Graded)]


def grade_prediction(
    prediction: Prediction,
    release: LinkedRelease,
    key: dict[tuple[str, str, str], Answer],
    filing_facts: Collection[Fact],
    text: str,
) -> list[Graded]:
    """A graded row for each answer of the release, and a no_answer row for each
    figure predicted for a period without one."""
    rows = input_rows(text)
    graded = []
    for field in FIELD_NAMES:
        for group in GROUPS:
            period = period_type(group, release.linked_form)
            predicted = getattr(getattr(prediction, group), field)
            answer = key.get((release.release_accession, field, period))
            if answer is None:
                if predicted is not None:
                    graded.append(
                        Graded(
                            release.ticker,
                            release.release_accession,
                            release.linked_accession,
                            field,
                            period,
                            None,
                            "",
                            predicted,
                            NO_ANSWER,
                            f"no {period} answer for this release",
                        )
                    )
                continue
            verdict = grade_value(answer, predicted, text, rows, filing_facts)
            graded.append(
                Graded(
                    release.ticker,
                    release.release_accession,
                    release.linked_accession,
                    field,
                    period,
                    answer.value,
                    answer.tag,
                    predicted,
                    verdict.verdict,
                    verdict.evidence,
                    verdict.factor,
                )
            )
    return graded


def graded_csv(graded: list[Graded]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=GRADED_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in graded:
        writer.writerow(
            {k: "" if v is None else str(v) for k, v in asdict(row).items()}
        )
    return buffer.getvalue()


def accuracy(rows: list[Graded]) -> str:
    counted = [r for r in rows if r.verdict not in UNGRADED]
    correct = sum(r.verdict == CORRECT for r in counted)
    if not counted:
        return "0/0"
    return f"{correct}/{len(counted)} ({100 * correct / len(counted):.1f}%)"


@dataclass
class RunNotes:
    """What the summary reports besides the graded rows."""

    graded_releases: int = 0
    without_prediction: int = 0  # linked releases with no prediction in the run
    not_linked: list[str] = dataclass_field(default_factory=list)
    period_mismatches: list[str] = dataclass_field(default_factory=list)


def format_summary(graded: list[Graded], notes: RunNotes) -> str:
    """Accuracy overall and by field, period type and ticker (not_in_input and
    no_answer left out), counts by verdict, and every not_in_input answer."""
    ungraded = Counter(r.verdict for r in graded if r.verdict in UNGRADED)
    lines = [
        (
            f"graded {plural(notes.graded_releases, 'release')}, "
            f"{plural(len(graded), 'figure')} ({ungraded[NOT_IN_INPUT]} not in input, "
            f"{ungraded[NO_ANSWER]} without an answer)"
        ),
        f"accuracy: {accuracy(graded)}",
    ]
    tickers = sorted({row.ticker for row in graded})
    for title, attribute, order in (
        ("field", "field", FIELD_NAMES),
        ("period type", "period_type", PERIOD_ORDER),
        ("ticker", "ticker", tickers),
    ):
        groups: dict[str, list[Graded]] = defaultdict(list)
        for row in graded:
            groups[getattr(row, attribute)].append(row)
        lines.append(f"by {title}:")
        lines.extend(
            f"  {name:<14}{accuracy(groups[name])}" for name in order if name in groups
        )
    counts = Counter(r.verdict for r in graded)
    lines.append("verdicts:")
    lines.extend(f"  {verdict:<24}{counts[verdict]:>6}" for verdict in VERDICTS)
    missing = [r for r in graded if r.verdict == NOT_IN_INPUT]
    lines.append(f"not in input: {len(missing)}")
    lines.extend(
        f"  {r.ticker} {r.release_accession} {r.field} {r.period_type} "
        f"(answer {r.answer}): {r.evidence}"
        for r in missing
    )
    lines.append(
        f"period_end differs from the linked filing: {len(notes.period_mismatches)}"
    )
    lines.extend(f"  {m}" for m in notes.period_mismatches)
    lines.append(
        f"predictions for releases not linked (not graded): {len(notes.not_linked)}"
    )
    lines.extend(f"  {a}" for a in notes.not_linked)
    lines.append(f"linked releases without a prediction: {notes.without_prediction}")
    return "\n".join(lines)


# Commands


class GradeError(Exception):
    """A run can't be graded; the message says why."""


def release_texts(accessions: Iterable[str], releases_dir: Path) -> dict[str, str]:
    """Release accession -> input text, for each release."""
    records = release_records(releases_dir)
    texts = {}
    for accession in accessions:
        if accession not in records:
            raise GradeError(f"no saved release {accession} in {releases_dir}")
        texts[accession] = build_input(records[accession]).text
    return texts


def grade_run(
    run_dir: Path,
    links_path: Path,
    answers_path: Path,
    releases_dir: Path,
    client: EdgarClient | None,
) -> tuple[list[Graded], RunNotes]:
    predictions_path = run_dir / PREDICTIONS
    if not predictions_path.is_file():
        raise GradeError(f"{predictions_path} not found")
    for path in (links_path, answers_path):
        if not path.is_file():
            raise GradeError(f"{path} not found")
    predictions = load_predictions(predictions_path)
    linked = {r.release_accession: r for r in load_links(links_path)[0]}
    key = load_answers(answers_path)

    notes = RunNotes()
    graded_predictions = []
    for prediction in predictions:
        release = linked.get(prediction.release_accession)
        if release is None:
            notes.not_linked.append(prediction.release_accession)
            continue
        graded_predictions.append((prediction, release))
        if prediction.period_end != release.period_end:
            notes.period_mismatches.append(
                f"{release.ticker} {release.release_accession}: predicted "
                f"{prediction.period_end}, {release.linked_form} "
                f"{release.linked_accession} ends {release.period_end}"
            )
    notes.graded_releases = len(graded_predictions)
    notes.without_prediction = len(linked) - notes.graded_releases

    releases = [release for _, release in graded_predictions]
    accessions = {r.release_accession for r in releases}
    answers = [a for a in key.values() if a.release_accession in accessions]
    try:
        facts = load_filing_facts(client or default_client(), releases, answers)
    except EdgarError as exc:
        raise GradeError(f"company facts: {exc}") from None
    problems = check_key(answers, facts)
    if problems:
        raise GradeError(
            f"{answers_path} doesn't match the company facts; rerun secbench.answers: "
            + "; ".join(problems)
        )
    texts = release_texts(sorted(accessions), releases_dir)
    graded = []
    for prediction, release in graded_predictions:
        graded.extend(
            grade_prediction(
                prediction,
                release,
                key,
                facts.get(release.linked_accession, set()),
                texts[release.release_accession],
            )
        )
    return graded, notes


def check_inputs(answers_path: Path, releases_dir: Path) -> tuple[str, bool]:
    """Run the answerable check on every answer: a report, and whether all passed."""
    if not answers_path.is_file():
        raise GradeError(f"{answers_path} not found")
    key = load_answers(answers_path)
    texts = release_texts(
        sorted({a.release_accession for a in key.values()}), releases_dir
    )
    rows = {accession: input_rows(text) for accession, text in texts.items()}
    found: Counter[tuple[str, str]] = Counter()
    total: Counter[tuple[str, str]] = Counter()
    missing = []
    for a in key.values():
        total[a.field, a.period_type] += 1
        text = texts[a.release_accession]
        if find_in_input(a.value, a.field, text, rows[a.release_accession]):
            found[a.field, a.period_type] += 1
        else:
            forms = input_forms(a.value, a.field)
            missing.append(
                f"  {a.ticker} {a.release_accession} {a.field} {a.period_type} "
                f"(answer {a.value}): searched {', '.join(forms) or 'nothing'}"
            )
    lines = [f"answers in input: {sum(found.values())} of {sum(total.values())}"]
    for field in FIELD_NAMES:
        for period in PERIOD_ORDER:
            if total[field, period]:
                lines.append(
                    f"  {field:<12} {period:<13}"
                    f"{found[field, period]:>4} of {total[field, period]}"
                )
    lines.append(f"not in input: {len(missing)}")
    lines.extend(missing)
    return "\n".join(lines), not missing


def main(argv: list[str] | None = None, client: EdgarClient | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.grade",
        description="Grade a run's predictions against the answer key, or check that "
        "every answer appears in its release's input.",
    )
    parser.add_argument("run", nargs="?", type=Path, help="runs/<run>/")
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="run the answerable check on every answer, without predictions",
    )
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--releases", type=Path, default=DEFAULT_RELEASES)
    args = parser.parse_args(argv)
    if args.check_inputs == (args.run is not None):
        parser.error("give a run folder or --check-inputs (not both)")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        if args.check_inputs:
            report, ok = check_inputs(args.answers, args.releases)
            print(report)
            return 0 if ok else 1
        graded, notes = grade_run(
            args.run, args.links, args.answers, args.releases, client
        )
    except (GradeError, PredictionsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = format_summary(graded, notes)
    write_text(args.run / GRADED, graded_csv(graded))
    write_text(args.run / SUMMARY, summary + "\n")
    print(summary)
    print(f"wrote {len(graded)} rows to {args.run / GRADED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
