"""The dataset repo of the Hugging Face export (<user>/sec-earnings-bench), built by
secbench.hfexport into dist/hf/dataset/: four configs of JSONL files, the prompts the
rows refer to, and the dataset card (README.md).

- releases: every release of the three scored splits (validation, company_test,
  time_test) as hfcommon.ReleaseRow, with its model input in `text`. The releases
  left out by the context limit stay, flagged (kept false, left_out_reason).
- answers: each split's XBRL answer key as hfcommon.AnswerRow, with where the grader
  finds each answer: spans, [start, end] character offsets into the release's text.
- train: the labelled training examples as hfcommon.TrainRow (chat messages), with
  whether the qwen3-8b-2k run trained on each and whether step-00025 had seen it.
- predictions: the exported runs' graded.csv rows as hfcommon.PredictionRow (each
  figure with its verdict, not_in_input and no_answer included):
  results/comparison.md's main table on validation, results/test_results.json's
  runs on the test splits. Figures and verdicts only, never a model's reply text.

Format: JSONL, one JSON object per line (hfcommon.write_jsonl). Why: the build needs
no new dependency, so it runs and is tested in CI; plain text is byte-for-byte
deterministic and readable by the privacy scan; the Hub's viewer converts JSONL to
Parquet itself. The card's YAML declares each config's files and its features,
derived from the row dataclasses' annotations, so every split loads with one schema
(a column null in every row of one split still loads as a string).

Every number in the card is counted from the rows written, read from a results file
(results/test_results.json, the training run's config.json, data/companies.csv) or
taken from the code's own constants. The build checks what the card states: the
training run's selection, the splits' filing windows, and that no validation or test
release, and no validation or company-test company, is in the training data.

Library only: build_dataset(ctx, out).
"""

from __future__ import annotations

import json
import re
import types
import typing
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from secbench import (
    answers,
    chat,
    companies,
    compare,
    grade,
    hfcommon,
    labels,
    link,
    splits,
    stage2,
    testreport,
)
from secbench.hfcommon import (
    COMPANY_TEST,
    SPLIT_NAMES,
    TIME_TEST,
    VALIDATION,
    Context,
    ExportError,
    Written,
)

RELEASES, ANSWERS, TRAIN, PREDICTIONS = "releases", "answers", "train", "predictions"
CONFIGS = (RELEASES, ANSWERS, TRAIN, PREDICTIONS)
TRAIN_SPLIT = "train"
ROW_TYPES: dict[str, type] = {
    RELEASES: hfcommon.ReleaseRow,
    ANSWERS: hfcommon.AnswerRow,
    TRAIN: hfcommon.TrainRow,
    PREDICTIONS: hfcommon.PredictionRow,
}
README = "README.md"
PROMPTS_DIR = "prompts"
TRAINING_CONFIG = "config.json"  # training/qwen3-8b-2k/config.json
# BibTeX styles lowercase a title's words; braces keep the acronyms in capitals.
CITATION_TITLE = (
    "extracting {GAAP} figures from {SEC} earnings releases, graded against {XBRL}"
)
# The planning notes some recorded texts cite (left_out_reason); not published.
PLANNING_NOTES = "PLAN.md"

# The card's metadata.
LICENSE = "cc-by-4.0"
LANGUAGES = ("en",)
PRETTY_NAME = "SEC earnings releases graded against XBRL"
TASK_CATEGORIES = ("text-generation", "table-question-answering")
TAGS = (
    "finance",
    "sec",
    "edgar",
    "xbrl",
    "earnings",
    "information-extraction",
    "benchmark",
)
# The Hub's size categories: the first whose bound is above the row count.
SIZE_CATEGORIES = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
)

# datasets' feature types (Value dtypes) for the rows' scalar annotations, and the
# keys of the columns that hold dicts (TrainRow.messages), in order.
SCALAR_TYPES = {str: "string", int: "int64", bool: "bool"}
STRUCT_KEYS = {"messages": ("role", "content")}

# The companies.csv split of each split's companies: the time test is the training
# companies' later releases.
COMPANY_SPLITS = {
    VALIDATION: companies.VALIDATION,
    COMPANY_TEST: companies.COMPANY_TEST,
    TIME_TEST: companies.TRAIN,
    TRAIN_SPLIT: companies.TRAIN,
}
LAST_TRAIN_DAY = stage2.TIME_TEST_START - timedelta(days=1)

CONFIG_ROWS = {
    RELEASES: (
        "one row per release: the company, the 8-K and the linked 10-Q or 10-K with "
        "their EDGAR links, and `text`, the model input"
    ),
    ANSWERS: (
        "one row per answer: one figure (revenue, net income or diluted EPS) for one "
        "period of one release, from XBRL, with where the release prints it"
    ),
    TRAIN: (
        "one row per training example: chat messages whose assistant reply is the "
        "label, written by code from the answer key"
    ),
    PREDICTIONS: (
        "one row per figure of one run: the model's figure or null, the answer and "
        "the grader's verdict"
    ),
}

COLUMNS: dict[str, dict[str, str]] = {
    RELEASES: {
        "split": "validation, company_test or time_test",
        "ticker": "the company's ticker",
        "company": "the company's name, from the S&P 500 list",
        "sector": "its GICS sector",
        "cik": "the CIK the 8-K is filed under",
        "release_accession": "the 8-K's accession number, which the configs join on",
        "filing_date": "the 8-K's filing date",
        "exhibits": (
            "the exhibits in `text`: the press release, then the exhibit appended "
            "when it has no income statement"
        ),
        "release_url": "the press release document on sec.gov",
        "filing_url": "the 8-K's filing index on sec.gov",
        "linked_form": "10-Q or 10-K: the filing whose XBRL gives the answers",
        "linked_cik": "the CIK the linked filing is filed under",
        "linked_accession": "the linked filing's accession number",
        "linked_url": "the linked filing's index on sec.gov",
        "period_end": "the linked filing's fiscal period end",
        "period_type": '"quarter" for a 10-Q, "full year" for a 10-K',
        "preferred_dividends": (
            "whether the linked filing's XBRL tags nonzero preferred dividends for "
            "the release's periods (see Answer key)"
        ),
        "prompt_tokens": (
            "the extract_v2 prompt with this release, in Qwen3-8B tokens (its chat "
            "template, thinking off)"
        ),
        "kept": (
            "false when the prompt is over the context limit: left out for every model"
        ),
        "left_out_reason": (
            "why the release was left out, as recorded when the split was frozen; "
            "null when kept"
        ),
        "input_sha256": (
            "the sha256 of `text` (UTF-8), as data/frozen_test_releases.csv pins it"
        ),
        "text": "the model input: the press release as Markdown",
    },
    ANSWERS: {
        "split": "validation, company_test or time_test",
        "ticker": "the company's ticker",
        "release_accession": "the release (`releases`' release_accession)",
        "linked_accession": "the 10-Q or 10-K whose XBRL gives the answer",
        "field": "revenue, net_income or eps_diluted",
        "period_type": "quarter, year_to_date or full_year",
        "start": (
            "the XBRL period's start; for a derived quarter, the day after the nine "
            "months end"
        ),
        "end": "the XBRL period's end",
        "value": (
            "the answer, exactly as SEC wrote it (for a derived quarter, the 10-K's "
            "full year minus the nine months): US dollars, EPS in dollars per share"
        ),
        "unit": "USD or USD/shares",
        "tag": "the us-gaap tag it comes from",
        "derived": (
            "true for a fourth quarter derived as the 10-K's full year minus the "
            "third-quarter 10-Q's nine months"
        ),
        "full_year_accession": "a derived quarter's 10-K; null otherwise",
        "nine_months_accession": (
            "a derived quarter's third-quarter 10-Q; null otherwise"
        ),
        "fiscal_year_start": "a derived quarter's fiscal year start; null otherwise",
        "in_input": (
            "whether the release prints the answer in a form the grader searches: "
            "only these answers are graded"
        ),
        "printed_as": (
            "the form the grader found: the figure as printed, in parentheses or "
            'after a minus sign for a negative answer, or "... on a loss row '
            '(label)" when it is unsigned on a row whose label says loss; null when '
            "not printed"
        ),
        "spans": (
            "every [start, end] character offset (end exclusive, in Unicode code "
            "points) of that form in the release's `text`"
        ),
    },
    TRAIN: {
        "release_accession": "the 8-K's accession number",
        "ticker": "the company's ticker",
        "kind": "first quarter, second or third quarter, or fourth quarter",
        "linked_form": "10-Q or 10-K",
        "filing_date": "the 8-K's filing date",
        "messages": (
            "system: prompts/extract_v2.txt; user: the release as Markdown, built as "
            "`releases`' text; assistant: the label, a line of JSON in extract_v2's "
            "reply format"
        ),
        "trained": "whether the qwen3-8b-2k run trained on it",
        "position": "its place in that run's training order, from 0; null when not trained",
        "seen_by_checkpoint": "whether step-00025, the published checkpoint, had seen it",
    },
    PREDICTIONS: {
        "split": "validation, company_test or time_test",
        "run": (
            "the run's name, as results/comparison.md and results/test_results.md "
            "list it"
        ),
        "model": "the model's name in the results tables",
        "model_id": (
            "the model in the request: an API model, a Hugging Face model or the "
            "adapter's served name"
        ),
        "source": "where the model ran: Anthropic API, OpenAI API or vLLM (self-hosted)",
        "prompt": "the system prompt: extract_v1 or extract_v2 (prompts/)",
        "ticker": "the company's ticker",
        "release_accession": "the release",
        "field": "revenue, net_income or eps_diluted",
        "period_type": "quarter, year_to_date or full_year",
        "answer": "the XBRL answer; null in a no_answer row",
        "answer_tag": "the answer's us-gaap tag; null in a no_answer row",
        "predicted": (
            "the model's figure in US dollars (EPS in dollars per share), as graded; "
            "null when it gave none"
        ),
        "verdict": "the grader's verdict (see Grading)",
        "factor": (
            "the power of ten a unit slip was matched with: the prediction times "
            "`factor` is what was matched, so it is below 1 for a prediction too "
            "large; null otherwise"
        ),
        "evidence": (
            "the grader's evidence: the release row, column or XBRL fact that "
            "explains a wrong figure, or the forms searched for a not_in_input answer"
        ),
    },
}

VERDICT_MEANINGS = {
    grade.CORRECT: (
        "equals the answer after rounding to the precision printed; EPS exact to "
        "the cent"
    ),
    grade.MISSING: "no figure (null) where the answer is graded",
    grade.SIGN: "the answer with the opposite sign",
    grade.SCALE: "the answer times or divided by {scales}: a unit misread",
    grade.ZEROS: "the answer times or divided by {zeros}: zeros miscounted",
    grade.WRONG_PERIOD: (
        "another period's figure: the answer's tag for another period of the linked "
        "filing, or another period column of the answer's table row"
    ),
    grade.NONCONTROLLING_INTEREST: (
        "net income including noncontrolling interests (the filing's ProfitLoss for "
        "the period)"
    ),
    grade.ROUNDED: "the answer rounded to a coarser power of ten",
    grade.ADJUSTED: (
        "a figure on a row or line of the release labelled adjusted, non-GAAP or "
        "excluding"
    ),
    grade.WRONG_LINE: "a figure from another row of the release",
    grade.OTHER: "none of the above",
    grade.NOT_IN_INPUT: (
        "the release doesn't print the answer in any form the grader searches"
    ),
    grade.NO_ANSWER: (
        "a figure for a period without an XBRL answer, such as a fourth quarter's EPS "
        "or a first quarter's year to date"
    ),
}

# Conditions on the answer key's later tags (secbench.answers), by the code's sets.
TAG_CONDITIONS = (
    (
        answers.WITHOUT_CURRENT_REVENUE,
        "only when the filing has no fact under "
        + " or ".join(f"`{tag}`" for tag in answers.CURRENT_REVENUE_TAGS),
    ),
    (
        answers.AFTER_PREFERRED_DIVIDENDS,
        "never when the filing reports nonzero preferred dividends for the period",
    ),
    (
        set(answers.ZERO_REQUIRED),
        "only when the filing tags noncontrolling interest zero for the period",
    ),
    (
        answers.CONTINUING_OPERATIONS,
        "only when the filing reports no discontinued operations for the period",
    ),
)


# The configs and their features


def config_splits(config: str) -> tuple[str, ...]:
    return (TRAIN_SPLIT,) if config == TRAIN else tuple(SPLIT_NAMES)


def data_file(config: str, split: str) -> str:
    """A config's file for one split, relative to the dataset repo's root."""
    return f"data/{config}/{split}.jsonl"


def column_annotations(config: str) -> dict[str, Any]:
    """The config's columns, in the row dataclass's order (the JSON keys' order),
    with their resolved annotations."""
    row_type = ROW_TYPES[config]
    hints = typing.get_type_hints(row_type)
    return {f.name: hints[f.name] for f in fields(row_type)}


def without_none(annotation: Any) -> tuple[Any, bool]:
    """X for X | None, and whether None was allowed."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) != 1:
            raise ExportError(f"no datasets feature for the union {annotation}")
        return args[0], True
    return annotation, False


def feature(annotation: Any, column: str) -> Any:
    """A value's datasets feature in the card's YAML form (as datasets'
    Features._to_yaml_list writes it): a dtype ("string"), {"list": item} for a
    tuple, or a list of {"name", "dtype"} for a dict whose keys STRUCT_KEYS names."""
    annotation, _ = without_none(annotation)
    if annotation in SCALAR_TYPES:
        return SCALAR_TYPES[annotation]
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is tuple:
        items = {a for a in args if a is not Ellipsis}
        if len(items) == 1:
            return {"list": feature(items.pop(), column)}
    if origin is dict and column in STRUCT_KEYS:
        value = feature(args[1], column)
        if isinstance(value, str):
            return [{"name": key, "dtype": value} for key in STRUCT_KEYS[column]]
    raise ExportError(f"no datasets feature for {column}: {annotation}")


def column_feature(column: str, annotation: Any) -> dict[str, Any]:
    value = feature(annotation, column)
    if isinstance(value, str):
        return {"name": column, "dtype": value}
    if isinstance(value, dict):
        return {"name": column, **value}
    return {"name": column, "struct": value}


def features(config: str) -> list[dict[str, Any]]:
    """The config's features, as the card's dataset_info lists them."""
    return [column_feature(c, a) for c, a in column_annotations(config).items()]


def type_text(annotation: Any, column: str) -> str:
    """A column's type for the card's tables: "string", "list of string", "int64 or
    null"."""
    inner, nullable = without_none(annotation)
    text = describe_feature(feature(inner, column))
    return f"{text} or null" if nullable else text


def describe_feature(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return f"list of {describe_feature(value['list'])}"
    return "{" + ", ".join(f"{f['name']}: {f['dtype']}" for f in value) + "}"


def size_category(rows: int) -> str:
    for bound, name in SIZE_CATEGORIES:
        if rows < bound:
            return name
    raise ExportError(f"no size category for {rows:,} rows")


# YAML (the card's front matter), written by hand: no YAML library is a dependency


PLAIN = re.compile(r"[A-Za-z][A-Za-z0-9_./-]*")
# Plain words YAML 1.1 (PyYAML, which the Hub and datasets read cards with) would
# turn into booleans or null.
YAML_WORDS = re.compile(r"y|n|yes|no|true|false|on|off|null", re.IGNORECASE)


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        if PLAIN.fullmatch(value) and not YAML_WORDS.fullmatch(value):
            return value
        return json.dumps(value, ensure_ascii=False)
    raise ExportError(f"no YAML scalar for {value!r}")


def yaml_lines(value: Mapping[str, Any] | Sequence[Any], indent: int = 0) -> list[str]:
    """Block-style YAML as PyYAML writes it: two-space indents, a mapping's list
    items at the mapping's own indent."""
    pad = " " * indent
    lines = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, Mapping) and item:
                lines += [f"{pad}{yaml_scalar(key)}:", *yaml_lines(item, indent + 2)]
            elif isinstance(item, list) and item:
                lines += [f"{pad}{yaml_scalar(key)}:", *yaml_lines(item, indent)]
            else:
                lines.append(f"{pad}{yaml_scalar(key)}: {yaml_inline(item)}")
        return lines
    for item in value:
        if isinstance(item, Mapping) and item:
            first, *rest = yaml_lines(item, indent + 2)
            lines += [f"{pad}- {first.lstrip()}", *rest]
        elif isinstance(item, list) and item:
            raise ExportError("a list directly inside a list isn't written")
        else:
            lines.append(f"{pad}- {yaml_inline(item)}")
    return lines


def yaml_inline(value: Any) -> str:
    if isinstance(value, Mapping):
        return "{}"
    if isinstance(value, list):
        return "[]"
    return yaml_scalar(value)


def metadata(built: Built) -> dict[str, Any]:
    largest = max(
        sum(count for (name, _), count in built.rows.items() if name == config)
        for config in CONFIGS
    )
    return {
        "license": LICENSE,
        "language": list(LANGUAGES),
        "pretty_name": PRETTY_NAME,
        "size_categories": [size_category(largest)],
        "task_categories": list(TASK_CATEGORIES),
        "tags": list(TAGS),
        "configs": [
            {
                "config_name": config,
                "data_files": [
                    {"split": split, "path": data_file(config, split)}
                    for split in config_splits(config)
                ],
            }
            for config in CONFIGS
        ],
        "dataset_info": [
            {"config_name": config, "features": features(config)} for config in CONFIGS
        ],
    }


# Building


@dataclass
class TrainTally:
    """What the card says about the training examples, counted as they're written,
    and checked: each example's system message is the training prompt."""

    prompt: str
    rows: int = 0
    trained: int = 0
    seen: int = 0
    kinds: Counter[str] = field(default_factory=Counter)
    trained_kinds: Counter[str] = field(default_factory=Counter)
    tickers: set[str] = field(default_factory=set)
    accessions: set[str] = field(default_factory=set)
    first_filed: str = ""
    last_filed: str = ""

    def count(self, rows: Iterable[hfcommon.TrainRow]) -> Iterator[hfcommon.TrainRow]:
        for row in rows:
            system = row.messages[0]
            if system["role"] != "system" or system["content"] != self.prompt:
                raise ExportError(
                    f"training example {row.release_accession}: its system message "
                    f"isn't prompts/{labels.PROMPT}.txt"
                )
            self.rows += 1
            self.trained += row.trained
            self.seen += row.seen_by_checkpoint
            self.kinds[row.kind] += 1
            if row.trained:
                self.trained_kinds[row.kind] += 1
            self.tickers.add(row.ticker)
            self.accessions.add(row.release_accession)
            self.first_filed = min(self.first_filed or row.filing_date, row.filing_date)
            self.last_filed = max(self.last_filed, row.filing_date)
            yield row


@dataclass
class Built:
    """What was written, for the card."""

    rows: dict[tuple[str, str], int]  # (config, split) -> rows
    train: TrainTally
    training: dict[str, Any]  # training/qwen3-8b-2k/config.json
    prompts: list[str]  # the prompts copied, by version
    companies: list[dict[str, str]]  # data/companies.csv


def build_dataset(ctx: Context, out: Path) -> Written:
    """Writes the dataset repo into out: each config's JSONL files, the prompts the
    rows refer to and README.md, the card. Stops with ExportError when a check
    fails (see the module docstring)."""
    sources = ctx.sources
    company_rows = hfcommon.read_csv(sources.companies)
    check_companies(ctx, company_rows)
    check_windows(ctx)
    training = hfcommon.read_json(sources.training / TRAINING_CONFIG)
    prompt = (sources.prompts / f"{labels.PROMPT}.txt").read_text(encoding="utf-8")
    tally = TrainTally(prompt)
    written = Written()
    rows: dict[tuple[str, str], int] = {}
    for config in CONFIGS:
        for split in config_splits(config):
            path = out / data_file(config, split)
            rows[config, split] = hfcommon.write_jsonl(
                path, config_rows(ctx, config, split, tally)
            )
            written.files.append(path)
    check_training(ctx, tally, training, company_rows)
    names = prompt_names(ctx)
    for name in names:
        target = out / PROMPTS_DIR / f"{name}.txt"
        hfcommon.copy_file(sources.prompts / f"{name}.txt", target)
        written.files.append(target)
    readme = out / README
    built = Built(rows, tally, training, names, company_rows)
    hfcommon.write_text(readme, card(ctx, built))
    written.files.append(readme)
    return written


def config_rows(
    ctx: Context, config: str, split: str, tally: TrainTally
) -> Iterable[Any]:
    if config == TRAIN:
        return tally.count(hfcommon.train_rows(ctx.sources))
    if config == RELEASES:
        rows: Iterable[Any] = ctx.releases[split]
    elif config == ANSWERS:
        rows = ctx.answers[split]
    else:
        rows = hfcommon.prediction_rows(ctx.runs(split))
    return in_split(rows, split)


def in_split(rows: Iterable[Any], split: str) -> Iterator[Any]:
    for row in rows:
        if row.split != split:
            raise ExportError(f"a {split} file would hold a {row.split} row")
        yield row


def prompt_names(ctx: Context) -> list[str]:
    """The prompts the rows refer to: the runs' and the training examples'."""
    used = {run.prompt for split in SPLIT_NAMES for run in ctx.runs(split)}
    return sorted(used | {labels.PROMPT})


def check_companies(ctx: Context, company_rows: Sequence[Mapping[str, str]]) -> None:
    """Each split's releases are of companies of its companies.csv split."""
    split_of = {row["ticker"]: row["split"] for row in company_rows}
    for split, rows in ctx.releases.items():
        wrong = sorted(
            {r.ticker for r in rows if split_of.get(r.ticker) != COMPANY_SPLITS[split]}
        )
        if wrong:
            raise ExportError(
                f"{split} has releases of companies outside its companies.csv split "
                f"({COMPANY_SPLITS[split]}): {', '.join(wrong[:5])}"
            )


def check_windows(ctx: Context) -> None:
    """The time test's releases were filed in its window (test_results.json)."""
    first, last = time_test_window(ctx)
    outside = [
        r.release_accession
        for r in ctx.releases[TIME_TEST]
        if not first <= r.filing_date <= last
    ]
    if outside:
        raise ExportError(
            f"time_test releases filed outside {first} to {last}: "
            + ", ".join(outside[:5])
        )


def time_test_window(ctx: Context) -> tuple[str, str]:
    window = ctx.test_results["time_test"]
    return window["first_filed"], window["last_filed"]


def check_training(
    ctx: Context,
    tally: TrainTally,
    training: Mapping[str, Any],
    company_rows: Sequence[Mapping[str, str]],
) -> None:
    """The training examples are the run's (its config.json's selection), filed in
    the training window, of training companies, and none is a scored release."""
    selection = training["selection"]
    expected = (
        selection["examples"],
        selection["selected"],
        dict(selection["selected_by_kind"]),
        hfcommon.examples_seen(ctx.sources),
    )
    found = (tally.rows, tally.trained, dict(tally.trained_kinds), tally.seen)
    if found != expected:
        raise ExportError(
            f"training examples (rows, trained, trained by kind, seen) {found} differ "
            f"from {ctx.sources.training / TRAINING_CONFIG}'s selection {expected}"
        )
    first, last = str(stage2.SINCE), str(LAST_TRAIN_DAY)
    if tally.rows and not first <= tally.first_filed <= tally.last_filed <= last:
        raise ExportError(
            f"training examples filed {tally.first_filed} to {tally.last_filed}, "
            f"outside {first} to {last}"
        )
    scored = {r.release_accession for rows in ctx.releases.values() for r in rows}
    if leaked := sorted(tally.accessions & scored):
        raise ExportError(f"scored releases among the training examples: {leaked[:5]}")
    train_companies = {
        r["ticker"] for r in company_rows if r["split"] == COMPANY_SPLITS[TRAIN_SPLIT]
    }
    if outside := sorted(tally.tickers - train_companies):
        raise ExportError(f"training examples of held-out companies: {outside[:5]}")


# The card


def card(ctx: Context, built: Built) -> str:
    sections = [
        intro(ctx, built),
        configs_section(ctx, built),
        companies_section(ctx, built),
        releases_section(ctx),
        answers_section(ctx),
        training_section(built),
        context_section(ctx),
        grading_section(),
        columns_section(ctx),
        predictions_section(ctx, built),
        loading_section(ctx),
        limitations_section(ctx),
        license_section(),
        citation_section(ctx),
    ]
    front = ["---", *yaml_lines(metadata(built)), "---", ""]
    body = [line for section in sections for line in [*section, ""]]
    return "\n".join(front + body).rstrip("\n") + "\n"


def n(count: int) -> str:
    return f"{count:,}"


def plural(count: int, word: str, words: str | None = None) -> str:
    return f"{count:,} {word if count == 1 else words or word + 's'}"


def were(count: int) -> str:
    return "was" if count == 1 else "were"


def and_list(items: Sequence[str], conjunction: str = "and") -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def or_list(items: Sequence[str]) -> str:
    return and_list(items, "or")


def code(text: str) -> str:
    return f"`{text}`"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return compare.markdown_table(header, rows)


def kept(rows: Sequence[hfcommon.ReleaseRow]) -> list[hfcommon.ReleaseRow]:
    return [r for r in rows if r.kept]


def left_out(rows: Sequence[hfcommon.ReleaseRow]) -> list[hfcommon.ReleaseRow]:
    return [r for r in rows if not r.kept]


def tickers(rows: Iterable[Any]) -> set[str]:
    return {r.ticker for r in rows}


def filed(rows: Sequence[hfcommon.ReleaseRow]) -> str:
    dates = [r.filing_date for r in rows]
    return f"{min(dates)} to {max(dates)}"


def all_releases(ctx: Context) -> list[hfcommon.ReleaseRow]:
    return [r for split in SPLIT_NAMES for r in ctx.releases[split]]


def all_answers(ctx: Context) -> list[hfcommon.AnswerRow]:
    return [a for split in SPLIT_NAMES for a in ctx.answers[split]]


def intro(ctx: Context, built: Built) -> list[str]:
    runs = sum(len(ctx.runs(split)) for split in SPLIT_NAMES)
    predictions = sum(built.rows[PREDICTIONS, split] for split in SPLIT_NAMES)
    releases, key, tally = all_releases(ctx), all_answers(ctx), built.train
    out = left_out(releases)
    return [
        f"# {hfcommon.REPO_NAMES[hfcommon.DATASET]}",
        "",
        (
            f"{hfcommon.REPO_NAMES[hfcommon.DATASET]} measures how accurately "
            "language models extract GAAP "
            "figures from US earnings press releases: revenue, net income and diluted "
            "EPS, for the quarter and the fiscal year to date (the full year in a "
            "fourth-quarter release). Each release is the press-release exhibit of "
            "an 8-K filed on SEC EDGAR, and each figure is graded automatically "
            "against XBRL in the company's 10-Q or 10-K for the period."
        ),
        "",
        (
            f"This dataset holds the {n(len(releases))} releases of the validation "
            "and test splits with their model inputs"
            + (
                f" ({n(len(out))} {were(len(out))} left out by the context limit "
                "and sent to no model)"
                if out
                else ""
            )
            + f"; their XBRL answer key ({n(len(key))} answers; the "
            f"{n(sum(a.in_input for a in key))} found in their release are located "
            f"in its text); and {n(predictions)} prediction rows from "
            f"{plural(runs, 'run')}, each with the model's figure (null when it "
            "gave none) and the grader's verdict. It also holds the "
            f"{n(tally.rows)} training examples labelled from XBRL: "
            f"{n(tally.trained)} {were(tally.trained)} drawn for the "
            f"{hfcommon.TRAINING_RUN} training run, and the published checkpoint, "
            f"{hfcommon.CHECKPOINT}, had seen the first {n(tally.seen)}."
        ),
        "",
        (
            f"- Code, method and results: [GitHub]({hfcommon.REPO_URL}) "
            f"([test results]({hfcommon.REPO_URL}/blob/main/results/test_results.md))"
        ),
        (
            f"- The fine-tuned model, {hfcommon.CHECKPOINT} (Qwen3-8B with a LoRA "
            f"adapter): [{ctx.repo_id(hfcommon.MODEL)}]({ctx.hub_url(hfcommon.MODEL)})"
        ),
        (
            "- Results explorer, with every graded figure beside its release: "
            f"[{ctx.repo_id(hfcommon.SPACE)}]({ctx.hub_url(hfcommon.SPACE)})"
        ),
    ]


def configs_section(ctx: Context, built: Built) -> list[str]:
    splits_shown = [*SPLIT_NAMES, TRAIN_SPLIT]
    rows = [
        [code(config)]
        + [
            n(built.rows[config, split]) if (config, split) in built.rows else "–"
            for split in splits_shown
        ]
        for config in CONFIGS
    ]
    time_test = ctx.releases[TIME_TEST]
    return [
        "## Configs",
        "",
        *[f"- {code(config)}: {CONFIG_ROWS[config]}." for config in CONFIGS],
        "",
        "Rows per config and split:",
        "",
        *table(["config", *splits_shown], rows),
        "",
        (
            f"The time test's {n(len(time_test))} releases include "
            f"{n(len(left_out(time_test)))} left out by the context limit "
            "(Context limit)."
            if left_out(time_test)
            else "No release is left out by the context limit."
        ),
    ]


def companies_section(ctx: Context, built: Built) -> list[str]:
    split_counts = Counter(r["split"] for r in built.companies)
    drawn = sorted(
        r["ticker"] for r in built.companies if r["split"] == companies.COMPANY_TEST
    )
    without = sorted(set(drawn) - tickers(ctx.releases[COMPANY_TEST]))
    validation = ctx.releases[VALIDATION]
    company_test = ctx.releases[COMPANY_TEST]
    time_test = ctx.releases[TIME_TEST]
    first, last = time_test_window(ctx)
    return [
        "## Companies and splits",
        "",
        (
            "The companies are the S&P 500's members outside the "
            f"{companies.FINANCIALS} sector, from one snapshot of Wikipedia's list of its "
            f"constituents: {n(len(built.companies))} companies, share classes "
            "merged. They were split by company with a fixed seed "
            f"({companies.SEED}) before the filings of any company outside "
            f"validation were downloaded: {n(split_counts[companies.VALIDATION])} "
            "for validation (the companies the benchmark began with), "
            f"{n(split_counts[companies.COMPANY_TEST])} drawn for the company test "
            f"and the other {n(split_counts[companies.TRAIN])} for training."
        ),
        "",
        (
            f"- {code(VALIDATION)}: {plural(len(validation), 'release')} of "
            f"{plural(len(tickers(validation)), 'company', 'companies')} "
            f"({', '.join(sorted(tickers(validation)))}), filed {filed(validation)}. "
            "Prompts and models were compared on it, and the fine-tuned checkpoint "
            "was chosen on it."
        ),
        (
            f"- {code(COMPANY_TEST)}: {plural(len(company_test), 'release')} of "
            f"{plural(len(tickers(company_test)), 'company', 'companies')} never "
            f"trained on, each company's latest {stage2.COMPANY_TEST_RELEASES} "
            f"linked releases, filed {filed(company_test)}. "
            + (
                f"{n(len(drawn) - len(without))} of the {n(len(drawn))} drawn "
                f"companies have a linked release ({and_list(without)} "
                f"{'has' if len(without) == 1 else 'have'} none)."
                if without
                else "Every drawn company has a linked release."
            )
        ),
        (
            f"- {code(TIME_TEST)}: {plural(len(time_test), 'release')} of "
            f"{plural(len(tickers(time_test)), 'training company', 'training companies')}"
            f", filed {first} to {last}, after every training release: "
            f"{n(len(kept(time_test)))} kept and {n(len(left_out(time_test)))} left "
            "out by the context limit."
        ),
        (
            f"- {code(TRAIN_SPLIT)}, the {code(TRAIN)} config's only split: "
            f"{plural(built.train.rows, 'labelled release')} of "
            f"{plural(len(built.train.tickers), 'training company', 'training companies')}"
            f", filed {stage2.SINCE} to {LAST_TRAIN_DAY}."
        ),
        "",
        (
            "No validation or test release, and no validation or company-test "
            "company, is in the training data; the build checks it."
        ),
    ]


def releases_section(ctx: Context) -> list[str]:
    releases = all_releases(ctx)
    appended = sum(len(r.exhibits) > 1 for r in releases)
    forms = Counter(r.linked_form for r in releases)
    return [
        "## Releases and links",
        "",
        (
            "A release is an 8-K that reports Item 2.02 (results of operations). Its "
            "press release (EX-99.1, or another EX-99 exhibit) is converted from HTML "
            "to Markdown with its tables intact. If it has no income statement, the "
            "first other EX-99 exhibit that has one is appended under its own "
            f"heading ({plural(appended, 'release')} here; `exhibits` lists them). "
            "Filings with no income statement at all, such as an executive change, "
            "are left out. `text` is the model input: exactly the user message each "
            "model was sent after the system prompt (the releases left out by the "
            "context limit were sent to none)."
        ),
        "",
        (
            "Each release is linked to the first 10-Q or 10-K the company filed on "
            "or after the release whose fiscal period ended at most "
            f"{link.MAX_PERIOD_AGE.days} days before it. A 10-Q gives the quarter "
            "and the fiscal year to date, a 10-K the full year. Here, "
            + and_list(
                [f"{n(forms[form])} link to a {form}" for form in link.LINK_FORMS]
            )
            + "."
        ),
    ]


def tag_condition(tag: str) -> str:
    return "; ".join(text for tags, text in TAG_CONDITIONS if tag in tags)


def tag_order(fld: answers.Field) -> str:
    """The field's tags in order, each run of tags with the same condition followed
    by it once."""
    groups: list[tuple[list[str], str]] = []
    for tag in fld.tags:
        condition = tag_condition(tag)
        if groups and groups[-1][1] == condition and condition:
            groups[-1][0].append(tag)
        else:
            groups.append(([tag], condition))
    parts = []
    for tags, condition in groups:
        text = ", then ".join(code(tag) for tag in tags)
        if condition:
            text += f" ({'each ' if len(tags) > 1 else ''}{condition})"
        parts.append(text)
    return ", then ".join(parts)


def printed_example(ctx: Context) -> str:
    """How the grader searches two of the answers: the first printed money answer
    and the first printed EPS answer."""
    shown = []
    for is_eps in (False, True):
        answer = next(
            (
                a
                for a in all_answers(ctx)
                if a.in_input and (a.field == grade.EPS) == is_eps
            ),
            None,
        )
        if answer is None:
            raise ExportError("no printed answer to show the grader's forms with")
        value = Decimal(answer.value)
        forms = grade.printed_forms(value, answer.field)
        name = "EPS" if is_eps else answer.field.replace("_", " ")
        shown.append(
            f"{answer.ticker}'s {answer.period_type.replace('_', ' ')} {name}, "
            f"{answer.value if is_eps else f'{value:,}'}, is searched as "
            + and_list([f'"{f.text}"' for f in forms])
        )
    return "; ".join(shown)


def unflagged_preferred(results: Mapping[str, Any]) -> dict[str, list[str]]:
    """By test split, the companies whose releases print preferred-stock lines that
    preferred_dividends missed, as results/test_results.md names them: in its miss
    summary, step-00025's misses on the common shareholders' line in releases
    outside preferred.csv with a row naming preferred (secbench.testreport)."""
    found = {}
    for split in hfcommon.TEST_SPLITS:
        outside = results["splits"][split]["miss_summary"]["outside_rows"]
        if named := [ticker for ticker, rows in outside.items() if "preferred" in rows]:
            found[split] = named
    return found


def named_by_split(named: Mapping[str, Sequence[str]]) -> str:
    """The companies by split, as "HPE, KDP and LHX in the time test"."""
    return and_list(
        [
            f"{and_list(t)} in the {split.replace('_', ' ')}"
            for split, t in named.items()
        ]
    )


def answers_section(ctx: Context) -> list[str]:
    by_split = {split: ctx.answers[split] for split in SPLIT_NAMES}
    count_rows = [
        [
            code(split),
            n(len(rows)),
            n(sum(a.in_input for a in rows)),
            n(sum(not a.in_input for a in rows)),
            n(sum(a.derived for a in rows)),
        ]
        for split, rows in by_split.items()
    ]
    tag_counts = {
        split: Counter(a.tag for a in rows) for split, rows in by_split.items()
    }
    tag_rows = [
        [code(tag), *[n(tag_counts[split][tag]) for split in SPLIT_NAMES]]
        for fld in answers.FIELDS
        for tag in fld.tags
        if any(tag_counts[split][tag] for split in SPLIT_NAMES)
    ]
    preferred = sum(r.preferred_dividends for r in all_releases(ctx))
    missed = unflagged_preferred(ctx.test_results)
    return [
        "## Answer key",
        "",
        (
            "The answers come from SEC's XBRL company facts, using only facts "
            "reported in the linked filing (matched by accession number) for its "
            "period. The first tag with a value wins:"
        ),
        "",
        *[f"- {fld.name}: {tag_order(fld)}." for fld in answers.FIELDS],
        "",
        (
            "Fourth-quarter revenue and net income are derived as the 10-K's full "
            "year minus the third-quarter 10-Q's nine months (`derived`). "
            "Fourth-quarter EPS has no answer: XBRL has no fourth-quarter value, and "
            "EPS can't be derived by subtraction. A revenue answer is dropped when "
            "another revenue-like fact for the period is at least "
            f"{answers.COMPONENT_RATIO} times larger (a component, not the total), "
            f"and an EPS answer of {n(answers.MAX_EPS)} or more is dropped "
            "(mis-scaled XBRL)."
        ),
        "",
        (
            "Net income is the company's, before preferred dividends: where there "
            "are preferred dividends, a release's net income available to common "
            "shareholders is another figure. `preferred_dividends` flags the "
            "releases whose linked filing tags nonzero preferred dividends for their "
            "periods in XBRL, under "
            + or_list([code(tag) for tag in answers.PREFERRED_DIVIDEND_TAGS])
            + f" ({plural(preferred, 'release')} here). The flag reads XBRL only"
            + (
                ", and it missed some releases that print preferred-stock lines: "
                f"results/test_results.md names {named_by_split(missed)}."
                if missed
                else "."
            )
        ),
        "",
        (
            "An answer is graded only if the release prints it in one of the "
            f"grader's forms: {printed_example(ctx)}. A negative answer may be in "
            "parentheses, after a minus sign, or unsigned on a row whose label says "
            "loss. `in_input`, `printed_as` and `spans` say where it was found."
        ),
        "",
        *table(
            [
                "split",
                "answers",
                "in the input (graded)",
                "not in the input",
                "derived",
            ],
            count_rows,
        ),
        "",
        "Tags used:",
        "",
        *table(["tag", *SPLIT_NAMES], tag_rows),
    ]


def training_section(built: Built) -> list[str]:
    tally, training = built.train, built.training
    data, selection = training["settings"]["data"], training["selection"]
    under_cap = selection["examples"] - selection["over_max_tokens"]
    kinds = [f"{kind} {n(tally.kinds[kind])}" for kind in labels.KINDS]
    return [
        "## Training labels",
        "",
        (
            "Each training example's reply is written by code from the XBRL answer "
            "key, never from a model's output: each figure as the release prints it "
            "in a table, with that table's unit, in extract_v2's reply format. A "
            "first quarter's year to date is the quarter; a fourth quarter's EPS is "
            "the cell in the three-month column of the row that shows the 10-K's "
            "full-year EPS. A release with any figure that can't be labelled this way "
            "is left out, so no label is null."
        ),
        "",
        (
            f"{plural(tally.rows, 'training release')} were labelled "
            f"({', '.join(kinds)}). {n(under_cap)} fit the training cap of "
            f"{n(data['max_tokens'])} tokens, and {n(tally.trained)} of those were "
            f"drawn, stratified by quarter kind with seed {data['seed']}, for the "
            f"{hfcommon.TRAINING_RUN} training run (`trained`, and `position` in its "
            f"order). The published checkpoint, {hfcommon.CHECKPOINT}, had seen the "
            f"first {n(tally.seen)} (`seen_by_checkpoint`)."
        ),
    ]


def context_section(ctx: Context) -> list[str]:
    out = left_out(all_releases(ctx))
    names = [f"{r.ticker} ({n(r.prompt_tokens)} tokens)" for r in out]
    return [
        "## Context limit",
        "",
        (
            "The fine-tuned model is served with a context of "
            f"{n(chat.MAX_MODEL_LEN)} tokens, and each reply is allowed "
            f"{n(chat.MAX_TOKENS)}, so a release whose prompt is over "
            f"{n(splits.MAX_PROMPT_TOKENS)} Qwen3-8B tokens is left out for every "
            "model, the API models included. "
            + (
                f"That leaves out {plural(len(out), 'release')}: {and_list(names)}. "
                + (
                    "They stay in `releases`, with `kept` false and "
                    "`left_out_reason`; they have no answers or predictions. "
                    if len(out) > 1
                    else "It stays in `releases`, with `kept` false and "
                    "`left_out_reason`; it has no answers or predictions. "
                )
                if out
                else "No release here is over it. "
            )
            + "`prompt_tokens` gives every release's prompt length."
        ),
    ]


def factor_list(factors: Iterable[Decimal]) -> str:
    return or_list([n(int(f)) for f in sorted(factors)])


def grading_section() -> list[str]:
    up = sorted(f for f in grade.FACTORS if f > 1)
    down = sorted(f for f in grade.FACTORS if f < 1)
    powers = [Decimal(10) ** k for k in range(1, len(up) + 1)]
    if up != powers or down != sorted(1 / f for f in up):
        raise ExportError(
            "grade.FACTORS isn't each power of ten from 10 up, multiplied and "
            "divided, as the card's unit slips say"
        )
    scales = [f for f in up if f in grade.SCALES]
    zeros = [f for f in up if f not in grade.SCALES]
    example = max(f for f in down if f in grade.SCALES)  # 0.001
    meanings = {
        verdict: text.format(scales=factor_list(scales), zeros=factor_list(zeros))
        for verdict, text in VERDICT_MEANINGS.items()
    }
    if list(meanings) != list(grade.VERDICTS):
        raise ExportError("the card's verdicts aren't grade.VERDICTS")
    rows = [
        [code(v), "no" if v in grade.UNGRADED else "yes", meanings[v]]
        for v in grade.VERDICTS
    ]
    ungraded = and_list([code(v) for v in grade.UNGRADED])
    return [
        "## Grading",
        "",
        (
            "The repository's grader (secbench.grade) gives each predicted figure "
            "one verdict. A figure whose answer the release doesn't print is "
            "not_in_input, and one for a period without an answer is no_answer: "
            "neither is graded. A graded figure is correct, missing, or wrong, and a "
            "wrong figure gets the first verdict that explains it, in the table's "
            "order:"
        ),
        "",
        *table(["verdict", "graded", "meaning"], rows),
        "",
        (
            f"Accuracy is correct figures over graded figures: {ungraded} rows are "
            "left out of it."
        ),
        "",
        (
            "Unit slips: after the exact checks, sign, wrong_period, "
            "noncontrolling_interest, rounded, adjusted and wrong_line are tried "
            "again on the prediction multiplied or divided by each power of ten "
            f"from {n(int(up[0]))} to {n(int(up[-1]))}. `factor` records what a "
            "match needed: the prediction times `factor` is what was matched, so a "
            f"division is a factor below 1 ({example}: the prediction divided by "
            f"{n(int(1 / example))}). Scale and zeros rows carry theirs too. A row "
            "with a factor is a unit slip."
        ),
    ]


def several_ciks(rows: Iterable[hfcommon.ReleaseRow]) -> dict[str, list[int]]:
    """The companies whose releases are filed under more than one CIK."""
    ciks: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        ciks[row.ticker].add(row.cik)
    return {t: sorted(c) for t, c in sorted(ciks.items()) if len(c) > 1}


def column_notes(ctx: Context) -> dict[tuple[str, str], str]:
    """What the rows and the test results show about some columns, added to their
    descriptions (COLUMNS) by (config, column)."""
    releases = all_releases(ctx)
    notes = {}
    if several := several_ciks(releases):
        shown = "; ".join(
            f"{ticker}: {and_list([str(c) for c in ciks])}"
            for ticker, ciks in several.items()
        )
        notes[RELEASES, "cik"] = (
            f"; a company's releases can be under more than one ({shown}), so group "
            "by `ticker`"
        )
    if missed := unflagged_preferred(ctx.test_results):
        notes[RELEASES, "preferred_dividends"] = (
            "; the flag missed some releases that print preferred-stock lines "
            f"({named_by_split(missed)})"
        )
    if any(PLANNING_NOTES in (r.left_out_reason or "") for r in releases):
        notes[RELEASES, "left_out_reason"] = (
            f". It cites {PLANNING_NOTES}, the project's planning notes, which "
            "aren't published; the Context limit section states the rule"
        )
    return notes


def columns_section(ctx: Context) -> list[str]:
    notes = column_notes(ctx)
    lines = ["## Columns"]
    for config in CONFIGS:
        annotations = column_annotations(config)
        if list(COLUMNS[config]) != list(annotations):
            raise ExportError(f"the card's {config} columns aren't its rows' fields")
        rows = [
            [
                code(column),
                type_text(annotation, column),
                COLUMNS[config][column] + notes.get((config, column), ""),
            ]
            for column, annotation in annotations.items()
        ]
        lines += [
            "",
            f"### {config}",
            "",
            *table(["column", "type", "description"], rows),
        ]
    return lines


def predictions_section(ctx: Context, built: Built) -> list[str]:
    lines = [
        "## Predictions by model",
        "",
        (
            "Each run has a row for every answer of each release it was given, and "
            "one for every figure it gave for a period without an answer "
            "(no_answer); a null figure for such a period has no row. Prompt "
            "extract_v1 asks for raw dollars; extract_v2 asks for revenue and net "
            "income as printed with their unit, which code converts, so `predicted` "
            "is in dollars either way. The prompts are in "
            + and_list([code(f"{PROMPTS_DIR}/{name}.txt") for name in built.prompts])
            + "."
        ),
    ]
    for split in SPLIT_NAMES:
        runs = ctx.runs(split)
        where = (
            "results/comparison.md's main table"
            if split == VALIDATION
            else "results/test_results.md"
        )
        rows = [
            [run.model, run.source, run.prompt, code(run.name), n(len(run.graded))]
            for run in runs
        ]
        lines += [
            "",
            f"### {split}",
            "",
            (
                f"{plural(len(runs), 'run')} ({n(built.rows[PREDICTIONS, split])} "
                f"rows), as {where} lists them."
            ),
            "",
            *table(["model", "source", "prompt", "run", "rows"], rows),
        ]
    checkpoints = sorted(
        {
            run.model
            for split in SPLIT_NAMES
            for run in ctx.runs(split)
            if run.name.startswith(f"{hfcommon.TRAINING_RUN}--step-")
            and run.model != hfcommon.CHECKPOINT
        }
    )
    commercial = [
        run
        for split in SPLIT_NAMES
        for run in ctx.runs(split)
        if run.source in hfcommon.COMMERCIAL_SOURCES
    ]
    sources = or_list([f'"{s}"' for s in hfcommon.COMMERCIAL_SOURCES])
    lines += [
        "",
        (
            f"{hfcommon.CHECKPOINT} is Qwen3-8B with the LoRA adapter published as "
            f"[{ctx.repo_id(hfcommon.MODEL)}]({ctx.hub_url(hfcommon.MODEL)}), and "
            "Qwen3-8B base is the same model without it, served the same way"
            + (
                f"; {and_list(checkpoints)} are later checkpoints of the same "
                "training run, scored on validation only."
                if checkpoints
                else "."
            )
        ),
        "",
        (
            f"Rows whose `source` is {sources} "
            f"({n(sum(len(r.graded) for r in commercial))} rows from "
            f"{plural(len(commercial), 'run')}) are outputs of commercial APIs, and "
            "their providers' terms apply to them. Only the predicted figures and the "
            "grader's verdicts are included, never the reply text."
        ),
    ]
    return lines


def loading_snippet(ctx: Context) -> list[str]:
    ungraded = json.dumps(list(grade.UNGRADED))
    return [
        "from datasets import load_dataset",
        "",
        f'repo = "{ctx.repo_id(hfcommon.DATASET)}"',
        f'releases = load_dataset(repo, "{RELEASES}", split="{COMPANY_TEST}")',
        f'answers = load_dataset(repo, "{ANSWERS}", split="{COMPANY_TEST}")',
        "",
        "# Where the grader found an answer in its release's text",
        'text = {r["release_accession"]: r["text"] for r in releases}',
        'answer = next(a for a in answers if a["in_input"])',
        'start, end = answer["spans"][0]',
        'print(answer["field"], answer["value"], text[answer["release_accession"]][start:end])',
        "",
        "# Accuracy by model: correct over graded figures",
        f'predictions = load_dataset(repo, "{PREDICTIONS}", split="{TIME_TEST}").to_pandas()',
        f'graded = predictions[~predictions["verdict"].isin({ungraded})]',
        f'print(graded.groupby("model")["verdict"].apply(lambda v: (v == "{grade.CORRECT}").mean()))',
    ]


def loading_section(ctx: Context) -> list[str]:
    return [
        "## Loading",
        "",
        "```python",
        *loading_snippet(ctx),
        "```",
    ]


def limitations_section(ctx: Context) -> list[str]:
    results = ctx.test_results
    gaps = []
    for split in hfcommon.TEST_SPLITS:
        coverage = results["splits"][split]["coverage"]
        share = testreport.num(100 * coverage["missing"] / coverage["expected"], 1)
        gaps.append(
            f"{n(coverage['missing'])} of {n(coverage['expected'])} on the "
            f"{split.replace('_', ' ')} ({share}%)"
        )
    sectors = results["splits"][TIME_TEST]["coverage"]["no_answer_by_sector"]
    worst, entry = max(sectors.items(), key=lambda item: item[1]["no_answer"])
    without = sum(v["no_answer"] for v in sectors.values())
    never_null = [
        label
        for model, label in results["models"].items()
        if all(
            results["splits"][split]["abstention"][model]["null"] == 0
            for split in hfcommon.TEST_SPLITS
        )
    ]
    notes = [
        f"{note['ticker']} {note['release_accession']} is {note['note']}; "
        + (
            f"none of its {note['answers']} answers is in its input, so it adds no "
            "graded figure"
            if note["not_in_input"] == note["answers"]
            else f"{note['not_in_input']} of its {note['answers']} answers aren't in "
            "its input"
        )
        for note in results["splits"][COMPANY_TEST]["notes"]
    ]
    _, last = time_test_window(ctx)
    return [
        "## Limitations",
        "",
        (
            "- **Survivorship:** the companies are the index's members at one "
            "snapshot, so companies that left the S&P 500 during the period covered "
            "are absent."
        ),
        (
            "- **Validation was used for design:** prompt extract_v2 was designed, "
            f"and the fine-tuned checkpoint chosen, on its {n(len(ctx.releases[VALIDATION]))} "
            "releases, so scores on it may overstate; the test splits give the "
            "estimates."
        ),
        (
            "- **Answer-key gaps:** some figures the kept releases should report have "
            "no XBRL answer, partly because SEC's company facts lacked some filings "
            f"when they were downloaded: {and_list(gaps)}. "
            f"{plural(without, 'time-test release')} have no answer at all, "
            f"{n(entry['no_answer'])} of them among the time test's {n(entry['kept'])} "
            f"{worst} releases."
        ),
        (
            "- **No nulls in training:** a release is labelled only when every figure "
            "it should report is printed, so no training example shows the prompt's "
            '"null when the release doesn\'t report a figure" applied.'
        ),
        (
            "- **Abstention isn't scored:** not_in_input figures are left out of "
            "accuracy, so answering null there is neither rewarded nor penalized. "
            "Many such releases print the figure in a form the grader doesn't search "
            "or as another value on the line asked for, and answering is then right. "
            + (
                f"{and_list(never_null)} never answered null on the test sets."
                if never_null
                else ""
            )
        ).rstrip(),
        (
            "- **Fourth-quarter EPS isn't graded:** XBRL has no fourth-quarter value, "
            "and EPS can't be derived by subtraction; predictions of it are no_answer "
            "rows."
        ),
        *[
            f"- **A non-earnings release in the company test:** {note}."
            for note in notes
        ],
        (
            f"- **The time test stops at {last}:** the next quarter's releases were "
            "not added, because their 10-Qs, which give the answer key, were not all "
            "filed when it was frozen. The protocol had planned to add them: a "
            "deviation, recorded in results/test_results.md."
        ),
        (
            "- **No repeat runs:** runs sent without temperature 0 vary from run to "
            "run, and no repeat runs measure that variation."
        ),
        (
            "- **SEC data can be amended:** a fresh download may differ. The frozen "
            "test releases are pinned by the sha256 of their links, answers and input "
            "text (data/frozen_test_releases.csv in the repository; `input_sha256` "
            "here)."
        ),
    ]


def license_section() -> list[str]:
    return [
        "## License",
        "",
        (
            "The annotations (the links between releases and filings, the answer "
            "key, the training labels, and the predictions' figures and verdicts) "
            "are released under CC BY 4.0. The release texts are the companies' press "
            "releases as filed with the SEC on EDGAR, converted to Markdown. The "
            "prompts are part of the code, released under the MIT license in the "
            f"[GitHub repository]({hfcommon.REPO_URL}). Predictions whose `source` is "
            "a commercial API also fall under its provider's terms."
        ),
    ]


def citation_section(ctx: Context) -> list[str]:
    year, name = hfcommon.author(ctx.sources)
    surname = re.sub(r"[^a-z]", "", name.split()[-1].lower())
    title = hfcommon.REPO_NAMES[hfcommon.DATASET]
    return [
        "## Citation",
        "",
        "```bibtex",
        "@misc{" + surname + year + title.replace("-", "") + ",",
        "  author = {" + name + "},",
        "  title = {{" + title + "}: " + CITATION_TITLE + "},",
        "  year = {" + year + "},",
        "  url = {" + hfcommon.REPO_URL + "}",
        "}",
        "```",
    ]
