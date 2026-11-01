"""Stage 2 training labels (PLAN.md Decisions 2026-10-27 and 2026-10-30): one chat
example per fully labelled train release, in extract_v2's reply format, from the XBRL
answer key only (never a model's output).

Each linked train release of <stage2>/releases.csv becomes system =
prompts/extract_v2.txt verbatim, user = its model input (secbench.inputs) and
assistant = the reply a perfect model would give:

- revenue and net income as the release prints the XBRL answer, with that table's
  unit ({"value": 3953.9, "unit": "millions"}); EPS as printed. The figure is a table
  cell, not marked adjusted (grade.adjusted_context), whose number is one of the
  answer's printed forms (grade.printed_forms, the forms the grader's input check
  searches for), with the answer's sign (parentheses or a minus sign, or unsigned on a
  loss row). Of several such cells, the first in an income-statement table (revenue,
  net income and per-share rows), then on a row whose label or section row names the
  figure, then in document order. A figure printed only in prose has no table unit,
  so it isn't labelled;
- in a first-quarter release the year to date is the quarter: a 10-Q with no
  year-to-date answer (secbench.answers expects none) whose quarter starts the day
  after a 10-K's period end (the cached submissions);
- in a fourth-quarter release (linked to a 10-K) the quarter's revenue and net income
  are the derived answers (full year minus nine months), and its EPS is the cell in
  the same table row as the 10-K's full-year EPS, in the three-month column, which
  the derived quarter's net income or revenue pins down (see fourth_quarter_eps);
- period_end is the linked filing's.

A release with any figure that can't be labelled this way is left out, with every
reason (left_out.csv). The labels are checked by the code that reads model replies:
each reply goes through extract.parse_reply (extract_v2) and the grader
(selfcheck/), and every graded figure must be correct, or the build fails.

Usage: uv run python -m secbench.labels [--stage2 data/stage2]
       [--companies data/companies.csv] [--out data/stage2/labels] [--raw data/raw]

Reads only the EDGAR cache, and refuses to start while tracked files have uncommitted
changes (manifest.json records the commit). Writes to <out>: train.jsonl (the
examples), figures.csv (each labelled figure and the cell it came from), left_out.csv
(each reason a release was left out), selfcheck/ (predictions.jsonl, graded.csv,
summary.txt), report.md (also printed), handcheck.md (figures drawn for checking by
hand), examples.csv (each example's release and quarter kind, in train.jsonl's
order; secbench.train draws its subsets by kind) and manifest.json. Token counts use Qwen3-8B's tokenizer at a pinned revision,
downloaded once into <raw>.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from secbench import extract, grade
from secbench.answers import FULL_YEAR, QUARTER, YEAR_TO_DATE, Answer
from secbench.companies import DEFAULT_OUT as DEFAULT_COMPANIES
from secbench.companies import TRAIN, Company, load_companies
from secbench.edgar import EdgarClient, EdgarError
from secbench.grade import (
    CELL_FIGURE,
    CORRECT,
    EPS,
    FIELD_NAMES,
    GROUPS,
    LOSS_LABEL,
    NET_INCOME,
    NO_ANSWER,
    Figure,
    InputRow,
    adjusted_context,
    input_rows,
    printed_forms,
)
from secbench.inputs import build_input, exhibit_heading, release_records
from secbench.link import (
    CELL_SPLIT,
    LINKED,
    PER_SHARE,
    REVENUE_LABEL,
    input_exhibits,
    is_net_income_row,
    is_per_share_row,
    is_revenue_row,
    shorten,
    ticker_filings,
)
from secbench.link import EPS as EPS_LABEL
from secbench.link import NET_INCOME as NET_INCOME_LABEL
from secbench.releases import write_text
from secbench.stage2 import DEFAULT_OUT as DEFAULT_STAGE2
from secbench.stage2 import MERGED, TIME_TEST_START, company_ciks, current_commit
from secbench.stage2 import NO_XBRL_ANSWER as NO_XBRL_ANSWER_FAILURE

log = logging.getLogger(__name__)

PROMPT = "extract_v2"
DEFAULT_OUT = DEFAULT_STAGE2 / "labels"
DEFAULT_RAW = Path("data/raw")
EXAMPLES = "train.jsonl"
EXAMPLE_INDEX = "examples.csv"
FIGURES = "figures.csv"
LEFT_OUT = "left_out.csv"
SELF_CHECK = "selfcheck"
REPORT = "report.md"
HANDCHECK = "handcheck.md"
MANIFEST = "manifest.json"
SEED = 20261028
HANDCHECK_FIGURES = 20
HANDCHECK_FOURTH_QUARTER_EPS = 10
# A fourth-quarter EPS cell more than this far from the quarter's net income over the
# full year's diluted shares (its net income over its EPS) is left out.
MAX_RATIO_DEVIATION = Decimal("0.2")
REVENUE = "revenue"
UNIT_NAMES = {multiplier: name for name, multiplier in extract.UNITS.items()}
DILUTED = re.compile(r"\bdilut", re.IGNORECASE)  # "Diluted", "assuming dilution"
BASIC = re.compile(r"\bbasic\b", re.IGNORECASE)
# What names each figure on a row's label or section row (see fits).
FIELD_LABELS = {
    REVENUE: (REVENUE_LABEL,),
    NET_INCOME: (NET_INCOME_LABEL,),
    EPS: (DILUTED, PER_SHARE, EPS_LABEL),
}

# Qwen3-8B's tokenizer, pinned: the revision of main on 2026-10-30, and the file's
# sha256.
QWEN3 = "Qwen/Qwen3-8B"
QWEN3_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_URL = (
    f"https://huggingface.co/{QWEN3}/resolve/{QWEN3_REVISION}/tokenizer.json"
)
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKEN_LIMITS = (8192, 16384, 24576, 32768)
LONGEST = 5  # longest examples the report names

# Where a label comes from.
CELL = "cell"  # a table cell printing the XBRL answer
FIRST_QUARTER = "first_quarter"  # a first quarter's year to date: the quarter's label
FOURTH_QUARTER_EPS = "fourth_quarter_eps"  # see fourth_quarter_eps

# Kinds of release, by the quarter it reports.
FIRST = "first quarter"
MIDDLE = "second or third quarter"
FOURTH = "fourth quarter"
KINDS = (FIRST, MIDDLE, FOURTH)

# Why a figure can't be labelled, in the order the report counts releases by (each
# under its first).
NO_XBRL_ANSWER = "no XBRL answer"
NOT_PRINTED = "not printed in a table"
NOT_FIRST_QUARTER = "no year-to-date answer, but not a first quarter"
NO_DILUTED_ROW = "fourth-quarter EPS: no diluted EPS row shows the full-year EPS"
NO_CONFIRMED_COLUMN = "fourth-quarter EPS: no three-month column confirmed"
NO_THREE_MONTH_FIGURE = "fourth-quarter EPS: no EPS figure in the three-month column"
ROWS_DISAGREE = "fourth-quarter EPS: rows disagree"
SIGN_DIFFERS = "fourth-quarter EPS: sign differs from the quarter's net income"
NO_RATIO = "fourth-quarter EPS: no ratio to check (full-year net income or EPS is 0)"
FAR_FROM_RATIO = (
    "fourth-quarter EPS: more than 20% from the quarter's net income x full-year EPS "
    "/ full-year net income"
)
REASONS = (
    NO_XBRL_ANSWER,
    NOT_PRINTED,
    NOT_FIRST_QUARTER,
    NO_DILUTED_ROW,
    NO_CONFIRMED_COLUMN,
    NO_THREE_MONTH_FIGURE,
    ROWS_DISAGREE,
    SIGN_DIFFERS,
    NO_RATIO,
    FAR_FROM_RATIO,
)

FIGURE_COLUMNS = [
    "ticker",
    "release_accession",
    "linked_form",
    "field",
    "group",
    "period_type",
    "source",
    "value",
    "unit",
    "printed",
    "answer",
    "exhibit",
    "url",
    "table_heading",
    "section",
    "row",
    "column",
    "anchor",
]
EXAMPLE_COLUMNS = ["release_accession", "ticker", "kind", "linked_form", "filing_date"]
LEFT_OUT_COLUMNS = [
    "ticker",
    "release_accession",
    "linked_form",
    "filing_date",
    "kind",
    "reason",
    "figure",
    "detail",
]


class LabelError(Exception):
    """The labels can't be built or don't check out; the message says why."""


# Where each row of an input is


@dataclass(frozen=True)
class Place:
    """The exhibit an input row comes from, and its table."""

    exhibit: str  # "EX-99.1"
    url: str  # the exhibit on sec.gov
    table: int | None  # the table's number in the input, from 0; None for prose
    income_statement: bool = False  # the table has revenue, net income, per-share rows


@dataclass(frozen=True)
class Located:
    row: InputRow
    place: Place
    position: int  # the row's index in grade.input_rows of the input


@dataclass(frozen=True)
class Cell:
    """A figure in a table cell of the input."""

    located: Located
    figure: Figure

    @property
    def row(self) -> InputRow:
        return self.located.row

    @property
    def column_header(self) -> str:
        return self.row.column_header(self.figure)

    def describe(self) -> str:
        """'Net income' row, 'Three Months Ended June 30, 2025' column: $27,233"""
        column = self.column_header
        return (
            f"{quoted(self.row.label, 60)} row"
            + (f", {quoted(column, 60)} column" if column else "")
            + f": {self.figure.text}"
        )


def quoted(text: str, width: int) -> str:
    """text for a reader, quoted: without zero-width spaces (Ecolab's and Ball's
    cells are full of them), spaces collapsed, at most width characters."""
    return repr(shorten(" ".join(text.replace("\u200b", " ").split()), width))


def exhibit_starts(record_path: Path, text: str) -> list[tuple[dict, int]]:
    """Each exhibit of a release's input with the offset in text where it starts (an
    appended exhibit at its heading), joined as secbench.inputs.build_input joins
    them; text must be that input."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    [(release, joined), *others] = input_exhibits(record, record_path.parent)
    starts = [(release, 0)]
    for exhibit, markdown in others:
        joined = f"{joined.rstrip()}\n\n"
        starts.append((exhibit, len(joined)))
        joined += f"{exhibit_heading(exhibit)}\n\n{markdown}"
    if joined != text:
        raise LabelError(
            f"{record_path}: the input isn't its exhibits joined as build_input joins "
            "them"
        )
    return starts


def row_cells(line: str) -> list[str]:
    """A table line's cells, split as secbench.link and secbench.grade split them."""
    return [c.strip().replace(r"\|", "|") for c in CELL_SPLIT.split(line)[1:-1]]


def is_income_statement(lines: Sequence[str]) -> bool:
    """A table body with a revenue row, a net income row and a row that says "per ...
    share", as secbench.link.has_income_statement reads one."""
    rows = [row_cells(line) for line in lines]
    return (
        any(map(is_revenue_row, rows))
        and any(map(is_net_income_row, rows))
        and any(is_per_share_row(row, eps=False) for row in rows)
    )


def locate(text: str, starts: Sequence[tuple[dict, int]]) -> list[Located]:
    """grade.input_rows of text, each with its exhibit (the last of starts beginning
    at or before its line) and table: consecutive table rows under one header row."""
    rows = input_rows(text)
    lines = text.splitlines()
    offsets = list(
        itertools.accumulate(
            (len(x) for x in text.splitlines(keepends=True)), initial=0
        )
    )
    placed: list[tuple[InputRow, dict, int | None]] = []
    line = 0
    table = -1
    header: tuple[str, ...] | None = None
    for row in rows:
        while lines[line] != row.line:
            line += 1
        exhibit = max((s for s in starts if s[1] <= offsets[line]), key=lambda s: s[1])[
            0
        ]
        if row.line.startswith("|"):
            if row.header is not header:
                table += 1
                header = row.header
            placed.append((row, exhibit, table))
        else:
            header = None
            placed.append((row, exhibit, None))
        line += 1
    bodies: dict[int, list[str]] = defaultdict(list)
    for row, _, number in placed:
        if number is not None:
            bodies[number].append(row.line)
    statements = {n for n, body in bodies.items() if is_income_statement(body)}
    return [
        Located(
            row,
            Place(
                exhibit["type"],
                exhibit.get("url") or "",
                number,
                number in statements,
            ),
            position,
        )
        for position, (row, exhibit, number) in enumerate(placed)
    ]


# Finding a figure in the input


def printed_number(figure: Figure) -> str:
    """A cell figure's number as printed, without sign or currency: "3,953.9"."""
    match = CELL_FIGURE.fullmatch(figure.text)
    if match is None:
        raise ValueError(f"not a cell figure: {figure.text!r}")
    return match[2] + (match[3] or "")


def signs_agree(value: Decimal, figure: Figure, row: InputRow) -> bool:
    """Whether a cell can show value's sign: a positive one unsigned; a negative one
    in parentheses or after a minus sign, or unsigned on a loss row ("Net loss")."""
    if value > 0:
        return figure.value > 0
    if value < 0:
        return figure.value < 0 or (
            figure.value > 0 and bool(LOSS_LABEL.search(row.label))
        )
    return figure.value == 0


def answer_cells(
    value: Decimal, field: str, located: Iterable[Located]
) -> list[tuple[Cell, int]]:
    """The table cells that print value (in one of its printed forms, with its sign,
    not marked adjusted), each with the unit of its form, in document order."""
    forms = {f.text: f.unit for f in printed_forms(value, field)}
    found = []
    for item in located:
        if item.place.table is None:
            continue
        for figure in item.row.figures:
            unit = forms.get(printed_number(figure))
            if (
                unit is None
                or not signs_agree(value, figure, item.row)
                or adjusted_context(item.row, figure) is not None
            ):
                continue
            found.append((Cell(item, figure), unit))
    return found


def says(row: InputRow, pattern: re.Pattern[str]) -> bool:
    """Whether a row's label or its section row matches pattern."""
    return bool(pattern.search(row.label) or pattern.search(row.section))


def fits(field: str, row: InputRow) -> bool:
    """Whether a row's label or section row names the figure: revenue or sales; net
    income, earnings or loss; diluted, per share or EPS."""
    return any(says(row, pattern) for pattern in FIELD_LABELS[field])


def choose(found: Sequence[tuple[Cell, int]], field: str) -> tuple[Cell, int]:
    """The cell to label from: the first in an income-statement table, then on a row
    that names the figure (for EPS, one that says diluted before another per-share
    row: Amcor's basic and diluted EPS can be equal), then in document order."""

    def fit(row: InputRow) -> int:
        if field == EPS and says(row, DILUTED):
            return 0
        return 1 if fits(field, row) else 2

    return min(
        found,
        key=lambda f: (
            not f[0].located.place.income_statement,
            fit(f[0].row),
            f[0].located.position,
            f[0].figure.column or 0,
        ),
    )


# Labels


@dataclass(frozen=True)
class Label:
    """One figure of a training reply."""

    field: str
    group: str  # quarter or year_to_date (the full year in a fourth-quarter release)
    value: Decimal  # as printed, with the answer's sign (3953.9; EPS in dollars)
    unit: str  # extract_v2's unit ("millions", "thousands"); "" for EPS
    source: str  # CELL, FIRST_QUARTER or FOURTH_QUARTER_EPS
    cell: Cell  # where it is printed
    answer: Decimal | None  # the XBRL answer it prints; None for fourth-quarter EPS
    anchor: str = ""  # fourth-quarter EPS: the full-year cell and confirming row

    def reply_value(self) -> Any:
        if self.field == EPS:
            return self.value
        return {"value": self.value, "unit": self.unit}

    def raw_value(self) -> Decimal:
        """In raw dollars (EPS in dollars per share), as extract converts it."""
        return self.value * extract.UNITS[self.unit] if self.unit else self.value


@dataclass(frozen=True)
class Problem:
    """Why a figure can't be labelled."""

    reason: str  # one of REASONS
    figure: str  # "revenue quarter"
    detail: str = ""


def cell_label(answer: Answer, group: str, cell: Cell, unit: int) -> Label:
    """The label of a cell printing answer: its number with the answer's sign."""
    magnitude = abs(cell.figure.value)
    value = -magnitude if answer.value < 0 else magnitude
    if value * unit != answer.value:
        raise ValueError(f"{cell.figure.text} in {unit:,}s is not {answer.value}")
    unit_name = "" if answer.field == EPS else UNIT_NAMES[unit]
    return Label(answer.field, group, value, unit_name, CELL, cell, answer.value)


@dataclass(frozen=True)
class Candidate:
    """A fourth-quarter EPS cell: the three-month cell of a row showing the full-year
    EPS (the anchor), in the column where the quarter's net income or revenue is
    printed (confirming)."""

    value: Decimal
    cell: Cell
    anchor: Cell
    confirming: Cell


def eps_anchors(cells: Sequence[Cell]) -> list[Cell]:
    """The cells showing the full-year EPS on a row whose label or section row says
    diluted ("Diluted", "assuming dilution", "Diluted earnings per share:" above "Net
    earnings"); if none does, on one that says per share or EPS but not basic."""
    diluted = [c for c in cells if says(c.row, DILUTED)]
    if diluted:
        return diluted
    return [
        c
        for c in cells
        if (says(c.row, PER_SHARE) or says(c.row, EPS_LABEL)) and not says(c.row, BASIC)
    ]


def confirmed_columns(
    anchor: Cell, pairs: Sequence[tuple[Sequence[Cell], Sequence[Cell]]]
) -> dict[int, Cell]:
    """Columns of the anchor's table, other than its own, where a quarter's figure is
    printed on a row that prints the full year's in the anchor's column: column ->
    that quarter cell. pairs are (quarter cells, full-year cells) of net income and
    revenue."""
    columns: dict[int, Cell] = {}
    column = anchor.figure.column
    for quarter_cells, full_year_cells in pairs:
        rows = {
            c.located.position
            for c in full_year_cells
            if c.located.place.table == anchor.located.place.table
            and c.figure.column == column
        }
        for c in quarter_cells:
            if c.located.position in rows and c.figure.column != column:
                assert c.figure.column is not None
                columns.setdefault(c.figure.column, c)
    return columns


def is_eps_figure(figure: Figure) -> bool:
    """Printed as EPS is: with two or three decimals."""
    return printed_number(figure) in {f.text for f in printed_forms(figure.value, EPS)}


def signed_eps(figure: Figure, anchor: Cell, full_year: Decimal) -> Decimal:
    """A three-month EPS cell's value as printed; on a row that prints the full
    year's loss unsigned (a loss row), an unsigned figure is a loss too."""
    if full_year < 0 < anchor.figure.value and figure.value > 0:
        return -figure.value
    return figure.value


def implied_eps(
    quarter_net_income: Decimal, full_year_net_income: Decimal, full_year_eps: Decimal
) -> Decimal | None:
    """The quarter's net income over the full year's diluted shares (its net income
    over its EPS); None when either is 0."""
    if full_year_net_income == 0 or full_year_eps == 0:
        return None
    return quarter_net_income * full_year_eps / full_year_net_income


def fourth_quarter_eps(
    located: Sequence[Located], answers: Mapping[tuple[str, str], Answer]
) -> tuple[Label | None, list[Problem]]:
    """A fourth-quarter release's EPS: the cell in the same table row as the 10-K's
    full-year EPS (eps_anchors), in the three-month column, which is where the
    derived quarter's net income or revenue is printed in that table, on the row that
    prints the full year's in the anchor's column (confirmed_columns). Every anchor
    with such a column must give the same value. Then two checks against the XBRL
    answers: the sign must be the quarter's net income's, and the value within 20%
    of the quarter's net income x full-year EPS / full-year net income. Without one
    of those answers, or if the full-year EPS or both the net income and revenue are
    not printed, there are no problems of its own: the release is left out for those."""
    name = f"{EPS} {QUARTER}"
    keys = (
        (EPS, FULL_YEAR),
        (NET_INCOME, QUARTER),
        (NET_INCOME, FULL_YEAR),
        (REVENUE, QUARTER),
        (REVENUE, FULL_YEAR),
    )
    if any(key not in answers for key in keys):
        return None, []

    def cells(key: tuple[str, str]) -> list[Cell]:
        return [c for c, _ in answer_cells(answers[key].value, key[0], located)]

    full_year_eps = answers[EPS, FULL_YEAR].value
    full_year_cells = cells((EPS, FULL_YEAR))
    pairs = [
        (cells((NET_INCOME, QUARTER)), cells((NET_INCOME, FULL_YEAR))),
        (cells((REVENUE, QUARTER)), cells((REVENUE, FULL_YEAR))),
    ]
    if not full_year_cells or not any(q and y for q, y in pairs):
        return None, []  # the figures it needs are reported not printed
    anchors = eps_anchors(full_year_cells)
    if not anchors:
        shown = "; ".join(c.describe() for c in full_year_cells[:3])
        return None, [Problem(NO_DILUTED_ROW, name, f"full-year EPS in {shown}")]
    candidates: list[Candidate] = []
    empty: list[str] = []
    for anchor in anchors:
        for column, confirming in confirmed_columns(anchor, pairs).items():
            figure = next((g for g in anchor.row.figures if g.column == column), None)
            if figure is None or not is_eps_figure(figure):
                empty.append(f"{anchor.describe()}, column {column}")
                continue
            candidates.append(
                Candidate(
                    signed_eps(figure, anchor, full_year_eps),
                    Cell(anchor.located, figure),
                    anchor,
                    confirming,
                )
            )
    if not candidates:
        if empty:
            return None, [Problem(NO_THREE_MONTH_FIGURE, name, "; ".join(empty))]
        shown = "; ".join(a.describe() for a in anchors[:3])
        return None, [Problem(NO_CONFIRMED_COLUMN, name, f"full-year cells: {shown}")]
    values = sorted({c.value for c in candidates})
    if len(values) > 1:
        shown = "; ".join(
            f"{c.value} in {c.cell.describe()}"
            for c in sorted(candidates, key=lambda c: c.value)
        )
        return None, [Problem(ROWS_DISAGREE, name, shown)]
    chosen = min(
        candidates,
        key=lambda c: (
            not c.anchor.located.place.income_statement,
            c.anchor.located.position,
        ),
    )
    value = chosen.value
    quarter_net_income = answers[NET_INCOME, QUARTER].value
    full_year_net_income = answers[NET_INCOME, FULL_YEAR].value
    if value * quarter_net_income < 0:
        return None, [
            Problem(
                SIGN_DIFFERS,
                name,
                f"{value} in {chosen.cell.describe()}; the quarter's net income is "
                f"{quarter_net_income}",
            )
        ]
    implied = implied_eps(quarter_net_income, full_year_net_income, full_year_eps)
    if implied is None:
        return None, [
            Problem(
                NO_RATIO,
                name,
                f"full-year net income {full_year_net_income}, EPS {full_year_eps}",
            )
        ]
    if abs(value - implied) > MAX_RATIO_DEVIATION * abs(implied):
        return None, [
            Problem(
                FAR_FROM_RATIO,
                name,
                f"{value} in {chosen.cell.describe()}; {quarter_net_income} x "
                f"{full_year_eps} / {full_year_net_income} = {implied:.4f}",
            )
        ]
    return Label(
        EPS,
        "quarter",
        value,
        "",
        FOURTH_QUARTER_EPS,
        chosen.cell,
        None,
        f"full year: {chosen.anchor.describe()}; three-month column confirmed by "
        f"{chosen.confirming.describe()}",
    ), []


# Releases


@dataclass(frozen=True)
class TrainRelease:
    """A linked train release (a row of releases.csv)."""

    ticker: str
    accession: str
    filing_date: date
    linked_form: str
    linked_accession: str
    period_end: date
    split: str
    company_split: str
    preferred_dividends: bool


@dataclass(frozen=True)
class Result:
    """A release's labels in reply order (quarter, then year to date: revenue, net
    income, EPS), or, if any figure can't be labelled, the problems and no labels."""

    release: TrainRelease
    kind: str  # one of KINDS
    labels: tuple[Label, ...] = ()
    problems: tuple[Problem, ...] = ()

    @property
    def labelled(self) -> bool:
        return not self.problems

    def first_reason(self) -> str:
        return min((p.reason for p in self.problems), key=REASONS.index)


def missing_detail(name: str, reasons: Sequence[str]) -> str:
    """The stage 2 failure reason for a figure ("revenue quarter: no fact ..."),
    without the figure's name."""
    for reason in reasons:
        if reason.startswith(f"{name}: "):
            return reason[len(name) + 2 :]
    return "not in answers.csv"


def label_release(
    release: TrainRelease,
    answers: Mapping[tuple[str, str], Answer],
    located: Sequence[Located],
    missing: Sequence[str] = (),
    fiscal_year_ends: Callable[[], set[date]] = set,
) -> Result:
    """A release's labels, or every reason it can't be labelled. answers are its
    answers by (field, period type); missing, the stage 2 reasons for answers it
    lacks ("revenue year_to_date: ..."); fiscal_year_ends, the company's 10-K period
    ends, asked for only to confirm a first quarter."""
    form = release.linked_form
    year_to_date_expected = form == "10-K" or any(
        period == YEAR_TO_DATE for _, period in answers
    )
    year_to_date_expected |= any(
        reason.startswith(f"{field} {YEAR_TO_DATE}: ")
        for field in FIELD_NAMES
        for reason in missing
    )
    kind = FOURTH if form == "10-K" else MIDDLE if year_to_date_expected else FIRST
    labels: dict[tuple[str, str], Label] = {}
    problems: list[Problem] = []
    for group in GROUPS:
        if group == "year_to_date" and not year_to_date_expected:
            continue
        period = grade.period_type(group, form)
        for field in FIELD_NAMES:
            if kind == FOURTH and group == "quarter" and field == EPS:
                continue  # see fourth_quarter_eps
            name = f"{field} {period}"
            answer = answers.get((field, period))
            if answer is None:
                problems.append(
                    Problem(NO_XBRL_ANSWER, name, missing_detail(name, missing))
                )
                continue
            found = answer_cells(answer.value, field, located)
            if not found:
                forms = ", ".join(f.text for f in printed_forms(answer.value, field))
                problems.append(
                    Problem(
                        NOT_PRINTED,
                        name,
                        f"answer {answer.value}; searched {forms or 'no form'}",
                    )
                )
                continue
            labels[group, field] = cell_label(answer, group, *choose(found, field))
    if kind == FOURTH:
        label, eps_problems = fourth_quarter_eps(located, answers)
        problems.extend(eps_problems)
        if label is not None:
            labels["quarter", EPS] = label
    elif kind == FIRST:
        starts = sorted({a.start for (_, p), a in answers.items() if p == QUARTER})
        if starts:
            day_before = starts[0] - timedelta(days=1)
            if day_before not in fiscal_year_ends():
                problems.append(
                    Problem(
                        NOT_FIRST_QUARTER,
                        "year_to_date",
                        f"the quarter starts {starts[0]}, and no 10-K's period ends "
                        f"{day_before}",
                    )
                )
            for field in FIELD_NAMES:
                if ("quarter", field) in labels:
                    labels["year_to_date", field] = replace(
                        labels["quarter", field],
                        group="year_to_date",
                        source=FIRST_QUARTER,
                    )
    if problems:
        return Result(release, kind, (), tuple(problems))
    return Result(
        release, kind, tuple(labels[g, f] for g in GROUPS for f in FIELD_NAMES)
    )


def reply(period_end: date, labels: Sequence[Label]) -> str:
    """The reply a perfect model gives under extract_v2: one line of JSON in the
    prompt's key order, numbers as printed (no commas or currency signs)."""
    figures: dict[str, dict[str, Any]] = {group: {} for group in GROUPS}
    for label in labels:
        figures[label.group][label.field] = label.reply_value()
    return extract.as_json(
        {
            "period_end": period_end.isoformat(),
            **{g: {f: figures[g][f] for f in FIELD_NAMES} for g in GROUPS},
        }
    )


def example(
    release: TrainRelease, labels: Sequence[Label], prompt: str, input_text: str
) -> dict[str, Any]:
    """One training example: the release's accession and its chat messages."""
    return {
        "release_accession": release.accession,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": reply(release.period_end, labels)},
        ],
    }


# Train releases only


def train_only_problems(
    accessions: Sequence[str],
    releases: Mapping[str, Mapping[str, str]],
    company_splits: Mapping[str, str],
) -> list[str]:
    """Why any of accessions isn't a train release: not in releases.csv, another split
    there, a company companies.csv doesn't put in train, filed on or after
    TIME_TEST_START, or given twice."""
    problems = []
    for accession, count in Counter(accessions).items():
        if count > 1:
            problems.append(f"{accession}: {count} examples")
    for accession in dict.fromkeys(accessions):
        row = releases.get(accession)
        if row is None:
            problems.append(f"{accession}: not in releases.csv")
            continue
        if row["split"] != TRAIN:
            problems.append(f"{accession}: split {row['split']} in releases.csv")
        company = company_splits.get(row["ticker"])
        if company != TRAIN:
            problems.append(
                f"{accession}: {row['ticker']} is {company or 'not'} in companies.csv"
            )
        if date.fromisoformat(row["filing_date"]) >= TIME_TEST_START:
            problems.append(f"{accession}: filed {row['filing_date']}")
    return problems


def examples_file_problems(
    path: Path,
    releases: Mapping[str, Mapping[str, str]],
    company_splits: Mapping[str, str],
) -> list[str]:
    """train_only_problems for the examples written to path."""
    accessions = [
        json.loads(line)["release_accession"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return train_only_problems(accessions, releases, company_splits)


# Building


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_csv(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: "" if row[k] is None else row[k] for k in columns})
    return buffer.getvalue()


def train_releases(rows: Iterable[Mapping[str, str]]) -> list[TrainRelease]:
    """The linked train rows of releases.csv, in file order."""
    return [
        TrainRelease(
            row["ticker"],
            row["release_accession"],
            date.fromisoformat(row["filing_date"]),
            row["linked_form"],
            row["linked_accession"],
            date.fromisoformat(row["period_end"]),
            row["split"],
            row["company_split"],
            row["preferred_dividends"] == "True",
        )
        for row in rows
        if row["split"] == TRAIN and row["link_status"] == LINKED
    ]


@dataclass
class Build:
    """Everything the labels command writes, before the self-check."""

    results: list[Result]
    examples: list[dict[str, Any]]
    releases: dict[str, dict[str, str]]  # releases.csv rows by accession
    company_splits: dict[str, str]  # companies.csv: ticker -> split


def fiscal_year_ends(client: EdgarClient, company: Company) -> set[date]:
    """The period ends of a company's 10-Ks (and 10-K/As, 10-KTs), from the cached
    submissions of its CIKs."""
    filings = ticker_filings(client, company.ticker, ciks=company_ciks(company))
    kept = filings[
        filings["form"].str.startswith("10-K") & filings["report_date"].notna()
    ]
    return {stamp.date() for stamp in kept["report_date"]}


def build(
    stage2: Path, companies_path: Path, prompt: str, client: EdgarClient
) -> Build:
    """Label every linked train release of stage2's merged data."""
    release_rows = read_csv(stage2 / "releases.csv")
    companies = {c.ticker: c for c in load_companies(companies_path)}
    key = grade.load_answers(stage2 / "answers.csv")
    by_release: dict[str, dict[tuple[str, str], Answer]] = defaultdict(dict)
    for (accession, field, period), answer in key.items():
        by_release[accession][field, period] = answer
    missing: dict[str, list[str]] = defaultdict(list)
    for failure in read_csv(stage2 / "failures.csv"):
        if failure["kind"] == NO_XBRL_ANSWER_FAILURE and failure["release_accession"]:
            missing[failure["release_accession"]].append(failure["reason"])
    records = release_records(stage2 / "releases")
    year_ends: dict[str, set[date]] = {}

    def year_ends_of(ticker: str) -> Callable[[], set[date]]:
        def ends() -> set[date]:
            if ticker not in year_ends:
                year_ends[ticker] = fiscal_year_ends(client, companies[ticker])
            return year_ends[ticker]

        return ends

    results, examples = [], []
    for release in train_releases(release_rows):
        record = records.get(release.accession)
        if record is None:
            raise LabelError(f"no saved release {release.accession} in {stage2}")
        text = build_input(record).text
        located = locate(text, exhibit_starts(record, text))
        result = label_release(
            release,
            by_release.get(release.accession, {}),
            located,
            missing.get(release.accession, []),
            year_ends_of(release.ticker),
        )
        results.append(result)
        if result.labelled:
            examples.append(example(release, result.labels, prompt, text))
    return Build(
        results,
        examples,
        {r["release_accession"]: r for r in release_rows},
        {ticker: c.split for ticker, c in companies.items()},
    )


def figure_rows(results: Iterable[Result]) -> list[dict[str, Any]]:
    """figures.csv: every labelled figure, in example and reply order."""
    rows = []
    for result in results:
        r = result.release
        for label in result.labels:
            cell = label.cell
            rows.append(
                {
                    "ticker": r.ticker,
                    "release_accession": r.accession,
                    "linked_form": r.linked_form,
                    "field": label.field,
                    "group": label.group,
                    "period_type": grade.period_type(label.group, r.linked_form),
                    "source": label.source,
                    "value": label.value,
                    "unit": label.unit,
                    "printed": cell.figure.text,
                    "answer": label.answer,
                    "exhibit": cell.located.place.exhibit,
                    "url": cell.located.place.url,
                    "table_heading": cell.row.heading,
                    "section": cell.row.section,
                    "row": cell.row.label,
                    "column": cell.column_header,
                    "anchor": label.anchor,
                }
            )
    return rows


def example_rows(results: Iterable[Result]) -> list[dict[str, Any]]:
    """examples.csv: each labelled release, in train.jsonl's order, with its kind."""
    return [
        {
            "release_accession": result.release.accession,
            "ticker": result.release.ticker,
            "kind": result.kind,
            "linked_form": result.release.linked_form,
            "filing_date": result.release.filing_date.isoformat(),
        }
        for result in results
        if result.labelled
    ]


def left_out_rows(results: Iterable[Result]) -> list[dict[str, Any]]:
    """left_out.csv: every reason of every release left out."""
    return [
        {
            "ticker": result.release.ticker,
            "release_accession": result.release.accession,
            "linked_form": result.release.linked_form,
            "filing_date": result.release.filing_date.isoformat(),
            "kind": result.kind,
            "reason": problem.reason,
            "figure": problem.figure,
            "detail": problem.detail,
        }
        for result in results
        for problem in result.problems
    ]


def write_examples(path: Path, examples: Iterable[Mapping[str, Any]]) -> None:
    write_text(
        path,
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples),
    )


# Self-check


@dataclass
class SelfCheck:
    problems: list[str]
    verdicts: Counter[str]
    not_graded: Counter[str]  # no_answer figures by label source


def self_check(
    out: Path,
    results: Sequence[Result],
    stage2: Path,
    client: EdgarClient,
) -> SelfCheck:
    """Read every reply in out/train.jsonl back with extract.parse_reply (extract_v2),
    check it against its labels, and grade the predictions (secbench.grade, written
    to out/selfcheck/). Every graded figure must be correct; the only ungraded ones
    (no answer) must be fourth-quarter EPS and first quarters' year to date."""
    labelled = {r.release.accession: r for r in results if r.labelled}
    folder = out / SELF_CHECK
    folder.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    predictions = []
    for line in (out / EXAMPLES).read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        accession = data["release_accession"]
        content = data["messages"][-1]["content"]
        try:
            prediction = extract.parse_reply(
                content, accession, extract.WITH_UNITS
            ).prediction
        except extract.ReplyError as exc:
            problems.append(f"{accession}: reply doesn't parse: {exc}")
            continue
        predictions.append(prediction)
        result = labelled[accession]
        for label in result.labels:
            got = getattr(getattr(prediction, label.group), label.field)
            if got != label.raw_value():
                problems.append(
                    f"{accession} {label.field} {label.group}: parsed {got}, labelled "
                    f"{label.raw_value()}"
                )
        if prediction.period_end != result.release.period_end:
            problems.append(f"{accession}: period_end {prediction.period_end}")
    write_text(
        folder / grade.PREDICTIONS,
        "".join(p.model_dump_json() + "\n" for p in predictions),
    )
    graded, notes = grade.grade_run(
        folder,
        stage2 / "links.csv",
        stage2 / "answers.csv",
        stage2 / "releases",
        client,
    )
    write_text(folder / grade.GRADED, grade.graded_csv(graded))
    write_text(folder / grade.SUMMARY, grade.format_summary(graded, notes) + "\n")
    sources = {
        (r.release.accession, label.field, label.group): label.source
        for r in labelled.values()
        for label in r.labels
    }
    not_graded: Counter[str] = Counter()
    for row in graded:
        group = "quarter" if row.period_type == QUARTER else "year_to_date"
        source = sources.get((row.release_accession, row.field, group))
        if row.verdict == NO_ANSWER:
            not_graded[source or "unknown"] += 1
            if source not in (FOURTH_QUARTER_EPS, FIRST_QUARTER):
                problems.append(
                    f"{row.release_accession} {row.field} {row.period_type}: no answer "
                    f"for a {source} label"
                )
        elif row.verdict != CORRECT:
            problems.append(
                f"{row.release_accession} {row.field} {row.period_type}: "
                f"{row.verdict} ({row.evidence or 'no evidence'}), answer {row.answer}, "
                f"label {row.predicted}"
            )
    problems.extend(notes.period_mismatches)
    problems.extend(f"{a}: not linked" for a in notes.not_linked)
    if notes.graded_releases != len(labelled):
        problems.append(
            f"graded {notes.graded_releases} releases, labelled {len(labelled)}"
        )
    expected = sum(
        label.source in (FOURTH_QUARTER_EPS, FIRST_QUARTER)
        for r in labelled.values()
        for label in r.labels
    )
    if sum(not_graded.values()) != expected:
        problems.append(
            f"{sum(not_graded.values())} figures without an answer, expected {expected}"
        )
    return SelfCheck(problems, Counter(row.verdict for row in graded), not_graded)


# Token counts


def tokenizer_path(raw: Path) -> Path:
    """Where the pinned tokenizer is cached: <raw>/<host>/<path of its URL>."""
    parts = urlsplit(TOKENIZER_URL)
    return raw.joinpath(parts.hostname or "", *parts.path.strip("/").split("/"))


def fetch_pinned(
    url: str, path: Path, sha256: str, transport: httpx.BaseTransport | None = None
) -> Path:
    """path, downloading url to it first if it isn't there; its sha256 must be the
    pinned one. Sends no SEC User-Agent (huggingface.co is not SEC)."""
    if not path.is_file():
        with httpx.Client(
            follow_redirects=True, timeout=60.0, transport=transport
        ) as client:
            response = client.get(url)
            response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != sha256:
            raise LabelError(f"{url} doesn't have the pinned sha256 {sha256}")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(response.content)
        partial.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise LabelError(f"{path} doesn't have the pinned sha256 {sha256}")
    return path


def load_tokenizer(raw: Path) -> Any:
    """Qwen3-8B's tokenizer at QWEN3_REVISION (the tokenizers package, a dev
    dependency)."""
    from tokenizers import Tokenizer

    path = fetch_pinned(TOKENIZER_URL, tokenizer_path(raw), TOKENIZER_SHA256)
    return Tokenizer.from_file(str(path))


def chat_text(messages: Sequence[Mapping[str, str]]) -> str:
    """A system, user and assistant message as Qwen3-8B's chat template
    (tokenizer_config.json at QWEN3_REVISION) lays out a final assistant message
    without tools or reasoning: its turn opens with an empty think block, as with
    thinking off."""
    system, user, assistant = (m["content"] for m in messages)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant}<|im_end|>\n"
    )


def token_counts(
    examples: Sequence[Mapping[str, Any]], tokenizer: Any
) -> dict[str, list[int]]:
    """Tokens of each example's system prompt, user message, reply and whole chat."""

    def count(texts: list[str]) -> list[int]:
        encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
        return [len(e.ids) for e in encoded]

    def part(index: int) -> list[str]:
        return [e["messages"][index]["content"] for e in examples]

    return {
        "system prompt": count(part(0)),
        "user (release input)": count(part(1)),
        "assistant (reply)": count(part(2)),
        "whole example": count([chat_text(e["messages"]) for e in examples]),
    }


# Report


def percentile(values: Sequence[int], p: float) -> int:
    found = extract.percentile(values, p)
    return 0 if found is None else int(found)


def table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    def line(cells: Sequence[Any]) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    return [line(header), "|" + "|".join(" --- " for _ in header) + "|"] + [
        line(row) for row in rows
    ]


def unit_kind(label: Label) -> str:
    if label.field == EPS:
        decimals = len(printed_number(label.cell.figure).partition(".")[2])
        return f"dollars, {decimals} decimals"
    one_decimal = "." in printed_number(label.cell.figure)
    return label.unit + (", one decimal" if one_decimal else "")


def format_report(
    results: Sequence[Result],
    examples: Sequence[Mapping[str, Any]],
    check: SelfCheck,
    tokens: Mapping[str, Sequence[int]],
    splits: Counter[str],
    manifest: Mapping[str, Any],
) -> str:
    labelled = [r for r in results if r.labelled]
    left_out = [r for r in results if not r.labelled]
    labels = [label for r in labelled for label in r.labels]
    tickers = {r.release.accession: r.release.ticker for r in results}
    lines = [
        "# Stage 2 training labels",
        "",
        (
            f"Built by commit {manifest['commit']} from {manifest['stage2']['dir']} "
            f"(merged by {manifest['stage2']['commit']}); prompt {PROMPT} (sha256 "
            f"{manifest['prompt']['sha256'][:12]}); tokens by {QWEN3} at revision "
            f"{QWEN3_REVISION[:12]}."
        ),
        "",
        "## Releases",
        "",
        (
            f"Linked train releases: {len(results)}; labelled {len(labelled)}, left "
            f"out {len(left_out)}."
        ),
        "",
        *table(
            ["kind", "linked", "labelled", "left out"],
            (
                [
                    kind,
                    sum(r.kind == kind for r in results),
                    sum(r.kind == kind for r in labelled),
                    sum(r.kind == kind for r in left_out),
                ]
                for kind in KINDS
            ),
        ),
        "",
        "## Left out",
        "",
        "Each release under its first reason, in this order:",
        "",
    ]
    first = Counter(r.first_reason() for r in left_out)
    every = Counter(p.reason for r in left_out for p in r.problems)
    releases_with = Counter(
        reason for r in left_out for reason in {p.reason for p in r.problems}
    )
    lines.extend(
        table(
            ["reason", "releases (first reason)", "releases (any)", "figures"],
            (
                [reason, first[reason], releases_with[reason], every[reason]]
                for reason in REASONS
                if every[reason]
            ),
        )
        if left_out
        else ["None."]
    )
    by_unit = Counter((label.field, unit_kind(label)) for label in labels)
    sources = Counter(label.source for label in labels)
    cells = [label for label in labels if label.source == CELL]
    lines += [
        "",
        "## Figures in the replies",
        "",
        *table(
            ["field", "unit as printed", "figures"],
            (
                [field, kind, by_unit[field, kind]]
                for field in FIELD_NAMES
                for kind in sorted(k for f, k in by_unit if f == field)
            ),
        ),
        "",
        (
            f"By source: {sources[CELL]} table cells printing the XBRL answer, "
            f"{sources[FIRST_QUARTER]} first-quarter year-to-date copies of the "
            f"quarter, {sources[FOURTH_QUARTER_EPS]} fourth-quarter EPS cells."
        ),
        (
            f"Cells outside an income-statement table: "
            f"{sum(not c.cell.located.place.income_statement for c in cells)}; on a "
            f"row whose label or section doesn't name the figure: "
            f"{sum(not fits(c.field, c.cell.row) for c in cells)}."
        ),
        (
            f"Labelled releases whose linked filing reports preferred dividends: "
            f"{sum(r.release.preferred_dividends for r in labelled)} (their net income "
            f"is the answer key's, before preferred dividends)."
        ),
        "",
        "## Self-check",
        "",
        (
            "Every reply read back by extract.parse_reply (extract_v2) and graded "
            f"(selfcheck/): {sum(check.verdicts.values())} figures, "
            + ", ".join(f"{v} {n}" for v, n in sorted(check.verdicts.items()))
            + "."
        ),
        (
            "Not graded (no answer): "
            + (
                ", ".join(f"{s} {n}" for s, n in sorted(check.not_graded.items()))
                or "none"
            )
            + "."
        ),
        (
            f"Problems: {len(check.problems)}."
            if check.problems
            else "Problems: none; every graded figure is correct."
        ),
        *(f"- {p}" for p in check.problems[:50]),
        "",
        f"## Tokens ({QWEN3})",
        "",
        *table(
            ["", "median", "p90", "p95", "p99", "max", "total"],
            (
                [
                    name,
                    *(f"{percentile(v, p):,}" for p in (50, 90, 95, 99)),
                    f"{max(v, default=0):,}",
                    f"{sum(v):,}",
                ]
                for name, v in tokens.items()
            ),
        ),
        "",
        "Whole examples longer than "
        + "; ".join(
            f"{limit:,} tokens: {sum(n > limit for n in tokens['whole example'])}"
            for limit in TOKEN_LIMITS
        )
        + f" (of {len(examples)}).",
        "",
        "Longest whole examples: "
        + "; ".join(
            f"{tickers[e['release_accession']]} {e['release_accession']} {n:,}"
            for n, e in sorted(
                zip(tokens["whole example"], examples, strict=True),
                key=lambda pair: pair[0],
                reverse=True,
            )[:LONGEST]
        )
        + ".",
        "",
        "## Splits",
        "",
        (
            "Examples by split (releases.csv): "
            + ", ".join(f"{s} {n}" for s, n in sorted(splits.items()))
            + "; each checked against releases.csv and companies.csv (train company, "
            f"filed before {TIME_TEST_START})."
        ),
    ]
    return "\n".join(lines) + "\n"


def describe_label(number: int, result: Result, label: Label) -> list[str]:
    """One hand-check entry."""
    r, cell = result.release, label.cell
    answer = (
        f"XBRL answer {label.answer}"
        if label.answer is not None
        else "no XBRL answer (fourth-quarter EPS)"
    )
    lines = [
        (
            f"{number}. [ ] {r.ticker} {r.accession} (filed {r.filing_date}), "
            f"{label.field} {grade.period_type(label.group, r.linked_form)}"
        ),
        (
            f"   - printed `{cell.figure.text}` in row "
            f"{quoted(cell.row.label, 100)}, column "
            f"{quoted(cell.column_header, 100)}"
        ),
        f"   - table heading {quoted(cell.row.heading, 100)}"
        + (f", section {quoted(cell.row.section, 60)}" if cell.row.section else ""),
        f"   - label {extract.as_json(label.reply_value())}; {answer}",
    ]
    if label.anchor:
        lines.append(f"   - {label.anchor}")
    lines.append(f"   - {cell.located.place.exhibit}: {cell.located.place.url}")
    return lines


def format_handcheck(results: Sequence[Result]) -> str:
    """HANDCHECK_FIGURES labelled figures (first-quarter copies left out: they are
    the quarter's cell), then HANDCHECK_FOURTH_QUARTER_EPS fourth-quarter EPS cells,
    each drawn with SEED."""
    figures = [
        (r, label)
        for r in results
        for label in r.labels
        if label.source != FIRST_QUARTER
    ]
    eps = [(r, label) for r, label in figures if label.source == FOURTH_QUARTER_EPS]
    lines = [
        "# Hand check: training labels",
        "",
        (
            f"Drawn with seed {SEED} (random.Random(seed).sample, in figures.csv "
            f"order) from {len(figures)} labelled figures read from a release "
            "(first-quarter year-to-date copies left out) and from its "
            f"{len(eps)} fourth-quarter EPS cells. Check each against the release at "
            "its link: the row, the column and the figure as printed."
        ),
    ]
    for title, population, size in (
        (f"{HANDCHECK_FIGURES} labelled figures", figures, HANDCHECK_FIGURES),
        (
            f"{HANDCHECK_FOURTH_QUARTER_EPS} fourth-quarter EPS cells",
            eps,
            HANDCHECK_FOURTH_QUARTER_EPS,
        ),
    ):
        drawn = random.Random(SEED).sample(population, min(size, len(population)))
        lines += ["", f"## {title}", ""]
        for number, (result, label) in enumerate(drawn, 1):
            lines.extend(describe_label(number, result, label))
    return "\n".join(lines) + "\n"


# Command


def main(
    argv: list[str] | None = None,
    client: EdgarClient | None = None,
    commit: str | None = None,
    tokenizer: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.labels",
        description="Build the stage 2 training examples from the XBRL answers.",
    )
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--companies", type=Path, default=DEFAULT_COMPANIES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--raw", type=Path, default=DEFAULT_RAW, help="where the tokenizer is cached"
    )
    parser.add_argument("--prompts-dir", type=Path, default=extract.PROMPTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if commit is None:
        commit, dirty = current_commit()
        if dirty:
            print(
                "error: tracked files have uncommitted changes; commit them first "
                "(manifest.json records the commit that built the labels)",
                file=sys.stderr,
            )
            return 1
    try:
        prompt = extract.load_prompt(PROMPT, args.prompts_dir)
        if tokenizer is None:
            tokenizer = load_tokenizer(args.raw)
        if client is None:
            client = EdgarClient(offline=True)
        built = build(args.stage2, args.companies, prompt.text, client)
        problems = train_only_problems(
            [e["release_accession"] for e in built.examples],
            built.releases,
            built.company_splits,
        )
        if problems:
            raise LabelError("not train releases: " + "; ".join(problems))
        args.out.mkdir(parents=True, exist_ok=True)
        write_examples(args.out / EXAMPLES, built.examples)
        problems = examples_file_problems(
            args.out / EXAMPLES, built.releases, built.company_splits
        )
        if problems:
            raise LabelError(f"{args.out / EXAMPLES}: " + "; ".join(problems))
        index = example_rows(built.results)
        if [row["release_accession"] for row in index] != [
            e["release_accession"] for e in built.examples
        ]:
            raise LabelError("examples.csv and train.jsonl list different releases")
        write_text(args.out / EXAMPLE_INDEX, rows_csv(index, EXAMPLE_COLUMNS))
        write_text(
            args.out / FIGURES, rows_csv(figure_rows(built.results), FIGURE_COLUMNS)
        )
        write_text(
            args.out / LEFT_OUT,
            rows_csv(left_out_rows(built.results), LEFT_OUT_COLUMNS),
        )
        check = self_check(args.out, built.results, args.stage2, client)
    except (LabelError, EdgarError, grade.GradeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    tokens = token_counts(built.examples, tokenizer)
    merged = args.stage2 / MERGED
    labelled = sum(r.labelled for r in built.results)
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "stage2": {
            "dir": str(args.stage2),
            "commit": json.loads(merged.read_text())["commit"]
            if merged.is_file()
            else None,
        },
        "prompt": {"version": PROMPT, "sha256": prompt.sha256},
        "tokenizer": {
            "model": QWEN3,
            "revision": QWEN3_REVISION,
            "sha256": TOKENIZER_SHA256,
        },
        "seed": SEED,
        "releases": {
            "linked_train": len(built.results),
            "labelled": labelled,
            "left_out": len(built.results) - labelled,
        },
        "self_check_problems": len(check.problems),
    }
    splits = Counter(
        built.releases[e["release_accession"]]["split"] for e in built.examples
    )
    report = format_report(
        built.results, built.examples, check, tokens, splits, manifest
    )
    write_text(args.out / REPORT, report)
    write_text(args.out / HANDCHECK, format_handcheck(built.results))
    write_text(args.out / MANIFEST, json.dumps(manifest, indent=2) + "\n")
    print(report, end="")
    print(
        f"wrote {len(built.examples)} examples to {args.out / EXAMPLES}; "
        f"report, hand check and manifest in {args.out}"
    )
    if check.problems:
        print(
            f"error: the self-check found {len(check.problems)} problems",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
