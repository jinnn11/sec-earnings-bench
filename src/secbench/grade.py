"""Grade model predictions against the answer key.

A prediction gives, for one release, revenue, net income and diluted EPS for the
quarter and for the year to date (the full year in a fourth-quarter release), in raw
dollars or null. Each figure is graded against data/answers.csv (see
secbench.answers) only if the answer appears in the release's input text (see
secbench.inputs); otherwise it is not_in_input and left out of accuracy. A wrong
figure is classified using the linked filing's other XBRL facts and the input's rows.

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
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
SCALES = (10**3, 10**6)
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
LOSS_LABEL = re.compile(r"\bloss(es)?\b", re.IGNORECASE)
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
        rows = list(csv.DictReader(f))
    key = {}
    for row in rows:
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
        )
        key[answer.release_accession, answer.field, answer.period_type] = answer
    return key


def load_filing_facts(
    client: EdgarClient, releases: Iterable[LinkedRelease]
) -> dict[str, set[Fact]]:
    """Linked accession -> the us-gaap facts of that filing, from the cached
    company facts of every CIK of each ticker."""
    by_ticker: dict[str, set[str]] = defaultdict(set)
    for release in releases:
        by_ticker[release.ticker].add(release.linked_accession)
    facts: dict[str, set[Fact]] = {}
    for ticker, accessions in sorted(by_ticker.items()):
        documents = [
            load_company_facts(client, cik) for cik in client.ticker_to_ciks(ticker)
        ]
        facts |= index_facts(documents, accessions).by_accession
    return facts


def check_key(answers: Iterable[Answer], facts: dict[str, set[Fact]]) -> list[str]:
    """Answers whose value isn't the linked filing's fact for its tag and period
    (answers.csv older than the company facts)."""
    problems = []
    for a in answers:
        values = {
            f.value
            for f in facts.get(a.linked_accession, ())
            if (f.tag, f.unit, f.start, f.end) == (a.tag, a.unit, a.start, a.end)
        }
        if values != {a.value}:
            got = ", ".join(map(str, sorted(values))) or "no fact"
            problems.append(
                f"{a.release_accession} {a.field} {a.period_type}: answers.csv has "
                f"{a.value}, company facts {got}"
            )
    return problems


# Input rows


@dataclass(frozen=True)
class Figure:
    text: str  # as shown: "$(1,234)", "95.4 billion"
    value: Decimal  # signed, in the unit shown
    scale: int | None = None  # from a word ("billion"); None if the unit isn't given


@dataclass(frozen=True)
class InputRow:
    """A table body row (labelled by its text cells) or a line of prose (labelled by
    itself)."""

    label: str
    line: str
    figures: tuple[Figure, ...]


def cell_figure(cell: str) -> Figure | None:
    text = clean_label(cell)
    if "%" in text:
        return None
    match = CELL_FIGURE.fullmatch(text)
    if match is None:
        return None
    prefix, number, decimals = match[1], match[2], match[3] or ""
    value = Decimal(number.replace(",", "") + decimals)
    negative = "(" in prefix or any(c in prefix for c in MINUS)
    return Figure(text, -value if negative else value)


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


def input_rows(text: str) -> list[InputRow]:
    """The input's table body rows and prose lines, in order. Header rows (the first
    row of each table) and separator rows are skipped."""
    rows = []
    in_table = False
    for line in text.splitlines():
        if not line.startswith("|"):
            in_table = False
            if label := clean_label(line):
                rows.append(InputRow(label, line, prose_figures(line)))
            continue
        cells = [c.strip().replace(r"\|", "|") for c in CELL_SPLIT.split(line)[1:-1]]
        if not in_table:
            in_table = True  # the header row
            continue
        if all(c == "---" for c in cells):
            continue
        figures = tuple(f for c in cells if (f := cell_figure(c)))
        rows.append(InputRow(" ".join(row_labels(cells)), line, figures))
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


def match_row(
    predicted: Decimal, field: str, rows: Iterable[InputRow]
) -> tuple[InputRow, Figure, int] | None:
    """The first row figure that equals predicted, in the unit shown (USD figures
    without a scale word are tried in dollars, thousands and millions)."""
    for row in rows:
        for figure in row.figures:
            if field == EPS:
                units = (1,) if figure.scale is None else ()
            else:
                units = (figure.scale,) if figure.scale else TABLE_UNITS
            for unit in units:
                if figure.value * unit == predicted:
                    return row, figure, unit
    return None


def describe_match(row: InputRow, figure: Figure, unit: int) -> str:
    times = f" (x{unit:,})" if unit != 1 and figure.scale is None else ""
    return f"{shorten(row.label, 60)!r}: {figure.text}{times}"


@dataclass(frozen=True)
class Verdict:
    verdict: str
    evidence: str = ""


def grade_value(
    answer: Answer,
    predicted: Decimal | None,
    text: str,
    rows: Sequence[InputRow],
    filing_facts: Collection[Fact],
) -> Verdict:
    """Grade one predicted figure against its answer."""
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
    if value != 0 and close(-predicted, value, field):
        return Verdict(SIGN)
    for factor in SCALES:
        if close(predicted * factor, value, field):
            return Verdict(SCALE, f"prediction x{factor:,} is the answer")
        if close(predicted / factor, value, field):
            return Verdict(SCALE, f"prediction /{factor:,} is the answer")
    others = sorted(
        (
            f
            for f in filing_facts
            if f.tag == answer.tag
            and f.unit == answer.unit
            and (f.start, f.end) != (answer.start, answer.end)
        ),
        key=lambda f: (f.end, f.start or f.end),
        reverse=True,
    )
    for fact in others:
        if close(predicted, fact.value, field):
            return Verdict(WRONG_PERIOD, describe_fact(fact))
    if field == NET_INCOME:
        for fact in filing_facts:
            if (fact.tag, fact.unit, fact.start, fact.end) == (
                PROFIT_LOSS,
                answer.unit,
                answer.start,
                answer.end,
            ) and close(predicted, fact.value, field):
                return Verdict(NONCONTROLLING_INTEREST, describe_fact(fact))
    unit = rounding_unit(value, field, predicted)
    if unit is not None:
        return Verdict(ROUNDED, f"answer rounded to {unit:,}")
    adjusted = [row for row in rows if ADJUSTED_LABEL.search(row.label)]
    if found := match_row(predicted, field, adjusted):
        return Verdict(ADJUSTED, describe_match(*found))
    if found := match_row(predicted, field, rows):
        return Verdict(WRONG_LINE, describe_match(*found))
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
    try:
        facts = load_filing_facts(client or default_client(), releases)
    except EdgarError as exc:
        raise GradeError(f"company facts: {exc}") from None
    accessions = {r.release_accession for r in releases}
    problems = check_key(
        (a for a in key.values() if a.release_accession in accessions), facts
    )
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
