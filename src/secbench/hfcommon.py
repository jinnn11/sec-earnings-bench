"""What the Hugging Face export (secbench.hfexport) reads, as one data model: the
frozen splits' releases and answer keys, the exported runs' graded figures, the
training examples and the validation table, plus the shared helpers (names, links,
writers). The model card, the dataset and the results explorer are all built from
these, so they show the same rows.

Every number comes from a results file: the splits from data/splits/ and
data/frozen_test_releases.csv (rebuilt with secbench.splits.build_all, which stops on
any change to a frozen release's links, answers or input text), the runs' run.json
and graded.csv, results/test_results.json and the training run's own files. Nothing
is read from runs/*/raw/ (API responses can carry account details): a run's figures
come from its graded.csv alone.

Library only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from secbench import answers, compare, extract, grade, inputs, link, splits
from secbench.releases import filing_index_url
from secbench.releases import write_text as _write_text
from secbench.stage2 import current_commit

REPO_URL = "https://github.com/jinnn11/sec-earnings-bench"
PLACEHOLDER_USER = "HF_USER"  # for local builds; hf_publish.sh refuses it
MODEL, INT4, DATASET, SPACE = "model", "int4-awq", "dataset", "space"
REPO_NAMES = {
    MODEL: "qwen3-8b-sec-earnings-lora",
    INT4: "qwen3-8b-sec-earnings-int4-awq",
    DATASET: "sec-earnings-bench",
    SPACE: "sec-earnings-explorer",
}
REPO_TYPES = {MODEL: "model", INT4: "model", DATASET: "dataset", SPACE: "space"}
# A Hub name: letters, digits, "-", "_" and "."; not starting or ending with "-" or
# ".", no "--" or "..", at most 96 characters.
HUB_NAME = re.compile(r"(?!.*(--|\.\.))[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?")

TRAINING_RUN = "qwen3-8b-2k"
CHECKPOINT = "step-00025"
VALIDATION, COMPANY_TEST, TIME_TEST = splits.SPLITS
SPLIT_NAMES = splits.SPLITS
TEST_SPLITS = (COMPANY_TEST, TIME_TEST)
FIELD_NAMES = grade.FIELD_NAMES  # revenue, net_income, eps_diluted
PERIOD_TYPES = grade.PERIOD_ORDER  # quarter, year_to_date, full_year

ANTHROPIC_API, OPENAI_API, SELF_HOSTED = (
    "Anthropic API",
    "OpenAI API",
    "vLLM (self-hosted)",
)
COMMERCIAL_SOURCES = (ANTHROPIC_API, OPENAI_API)
# Display names by run.json's model, as scripts/plot_results.py names them.
MODEL_NAMES = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "gpt-6.1-sol": "GPT-6.1 Sol",
    "gpt-6-luna": "GPT-6 Luna",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "google/gemma-3-12b-it": "Gemma-3-12B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}
GPU_HOURLY = 1.0  # results/comparison.md prices GPU runs at $1 per GPU-hour


class ExportError(Exception):
    """The export can't be built; the message says why."""


# Inputs


@dataclass(frozen=True)
class Sources:
    """Where everything is, under the repository root (tests use a small copy)."""

    root: Path = Path(".")

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def results(self) -> Path:
        return self.path("results")

    @property
    def test_results(self) -> Path:
        return self.results / "test_results.json"

    @property
    def comparison(self) -> Path:
        return self.results / "comparison.md"

    @property
    def serving_results(self) -> Path:
        return self.results / "serving_results.json"

    @property
    def validation_chart(self) -> Path:
        return self.results / "accuracy_vs_cost.png"

    @property
    def test_chart(self) -> Path:
        return self.results / "test_accuracy_vs_cost.png"

    @property
    def serving_chart(self) -> Path:
        return self.results / "serving_throughput_latency.png"

    @property
    def protocol(self) -> Path:
        return self.results / "test_protocol.md"

    @property
    def runs(self) -> Path:
        return self.path("runs")

    @property
    def training(self) -> Path:
        return self.path("training", TRAINING_RUN)

    @property
    def checkpoint(self) -> Path:
        return self.training / "checkpoints" / CHECKPOINT

    @property
    def test_config(self) -> Path:
        return self.path("configs", "test", f"{TRAINING_RUN}.toml")

    @property
    def serving_config(self) -> Path:
        return self.path("configs", "serving", f"{TRAINING_RUN}.toml")

    @property
    def splits_dir(self) -> Path:
        return self.path("data", "splits")

    @property
    def frozen(self) -> Path:
        return self.path("data", "frozen_test_releases.csv")

    @property
    def stage1(self) -> splits.Sources:
        return splits.Sources(
            self.path("data", "links.csv"),
            self.path("data", "answers.csv"),
            self.path("data", "releases"),
        )

    @property
    def stage2(self) -> Path:
        return self.path("data", "stage2")

    @property
    def labels(self) -> Path:
        return self.stage2 / "labels"

    @property
    def companies(self) -> Path:
        return self.path("data", "companies.csv")

    @property
    def prompts(self) -> Path:
        return self.path("prompts")

    @property
    def license(self) -> Path:
        return self.path("LICENSE")

    @property
    def calibration(self) -> Path:
        return self.path("data", "serving", TRAINING_RUN, "calibration.json")


def check_user(user: str) -> str:
    """A Hugging Face user or organization name, as the Hub allows it."""
    if not HUB_NAME.fullmatch(user):
        raise ExportError(
            f"--hf-user {user!r} isn't a Hub name (letters, digits, '-', '_' and '.', "
            "at most 96, not starting or ending with '-' or '.')"
        )
    return user


def repo_id(user: str, kind: str) -> str:
    return f"{user}/{REPO_NAMES[kind]}"


def hub_url(user: str, kind: str) -> str:
    prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[REPO_TYPES[kind]]
    return f"https://huggingface.co/{prefix}{repo_id(user, kind)}"


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise ExportError(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ExportError(f"{path} not found")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_results(sources: Sources) -> dict[str, Any]:
    return read_json(sources.test_results)


def author(sources: Sources) -> tuple[str, str]:
    """(year, name) from LICENSE's copyright line, for the citations."""
    text = sources.license.read_text(encoding="utf-8")
    match = re.search(r"Copyright \(c\) (\d{4}) (.+)", text)
    if match is None:
        raise ExportError(f"{sources.license} has no 'Copyright (c) YEAR NAME' line")
    return match[1], match[2].strip()


def build_info() -> dict[str, Any]:
    """The code's commit, and whether tracked files had uncommitted changes."""
    commit, dirty = current_commit()
    return {"commit": commit, "dirty": dirty}


# Releases


@dataclass(frozen=True)
class ReleaseRow:
    """One release of a scored split: its links, its model input and where it is on
    EDGAR. Left-out releases (prompts over the context limit) are kept, flagged."""

    split: str
    ticker: str
    company: str
    sector: str
    cik: int
    release_accession: str
    filing_date: str
    exhibits: tuple[str, ...]  # the exhibits in the input: ("EX-99.1",) or more
    release_url: str  # the press release document on sec.gov
    filing_url: str  # the 8-K's filing index
    linked_form: str  # 10-Q or 10-K
    linked_cik: int
    linked_accession: str
    linked_url: str  # the 10-Q's or 10-K's filing index
    period_end: str
    period_type: str  # links.csv's: "quarter" or "full year"
    # The linked filing tags nonzero preferred dividends (answers.py's XBRL check on
    # PREFERRED_DIVIDEND_TAGS; it misses a release that only prints them).
    preferred_dividends: bool
    prompt_tokens: int  # extract_v2's prompt with this input, Qwen3-8B tokens
    kept: bool  # False: left out for every model (over the context limit)
    left_out_reason: str | None
    input_sha256: str
    text: str  # the model input (secbench.inputs)


def load_releases(sources: Sources) -> dict[str, list[ReleaseRow]]:
    """Each scored split's releases in the frozen list's order, rebuilt from their
    sources and checked against the frozen list and the split's files (tokens.csv,
    links.csv, answers.csv, left_out.csv)."""
    try:
        frozen_rows = splits.read_table(sources.frozen, splits.FROZEN_COLUMNS)
        built = splits.build_all(sources.stage1, sources.stage2, frozen_rows, [])
    except (splits.SplitError, OSError, ValueError) as exc:
        raise ExportError(f"the frozen splits don't rebuild: {exc}") from None
    companies = {row["ticker"]: row for row in read_csv(sources.companies)}
    found: dict[str, list[ReleaseRow]] = {}
    for split in built:
        if split.status != splits.FROZEN:
            raise ExportError(f"{split.name} isn't frozen in {sources.frozen}")
        found[split.name] = split_rows(sources, split, companies)
    return found


def split_rows(
    sources: Sources, split: splits.Split, companies: Mapping[str, Mapping[str, str]]
) -> list[ReleaseRow]:
    folder = sources.splits_dir / split.name
    try:
        tokens = {
            r["release_accession"]: r
            for r in splits.read_table(folder / splits.TOKENS, splits.TOKEN_COLUMNS)
        }
        left_out = {
            r["release_accession"]: r
            for r in splits.read_table(
                folder / splits.LEFT_OUT_FILE, splits.LEFT_OUT_COLUMNS
            )
        }
        kept_links = {
            r["release_accession"]: r
            for r in splits.read_table(folder / splits.LINKS, link.COLUMNS)
        }
        answer_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in splits.read_table(folder / splits.ANSWERS, answers.COLUMNS):
            answer_rows[row["release_accession"]].append(row)
    except splits.SplitError as exc:
        raise ExportError(str(exc)) from None
    records = inputs.release_records(split.sources.releases)
    rows = []
    for release in split.releases:
        where = f"{split.name} {release.accession}"
        token = tokens.get(release.accession)
        if token is None:
            raise ExportError(f"{where}: not in {folder / splits.TOKENS}")
        if token["input_sha256"] != release.input_sha256:
            raise ExportError(f"{where}: input differs from {folder / splits.TOKENS}")
        kept = token["status"] == splits.KEPT
        if kept != (release.accession in kept_links) or kept == (
            release.accession in left_out
        ):
            raise ExportError(
                f"{where}: status {token['status']} disagrees with {folder}'s "
                f"{splits.LINKS} and {splits.LEFT_OUT_FILE}"
            )
        if kept and kept_links[release.accession] != release.links_row:
            raise ExportError(f"{where}: links row differs from {folder}")
        mine = answer_rows.pop(release.accession, [])
        if kept and sorted_lines(mine) != sorted_lines(release.answer_rows):
            raise ExportError(f"{where}: answers differ from {folder}")
        company = companies.get(release.ticker)
        if company is None:
            raise ExportError(f"{where}: {release.ticker} not in {sources.companies}")
        record_path = records[release.accession]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        built = inputs.build_input(record_path)
        if built.text != release.text:
            raise ExportError(f"{where}: input text differs between two builds")
        row = release.links_row
        rows.append(
            ReleaseRow(
                split=split.name,
                ticker=release.ticker,
                company=company["name"],
                sector=company["sector"],
                cik=int(record["cik"]),
                release_accession=release.accession,
                filing_date=record["filing_date"],
                exhibits=built.exhibits,
                release_url=record["exhibit_url"],
                filing_url=filing_index_url(record["cik"], release.accession),
                linked_form=row["linked_form"],
                linked_cik=int(row["linked_cik"]),
                linked_accession=row["linked_accession"],
                linked_url=filing_index_url(row["linked_cik"], row["linked_accession"]),
                period_end=row["period_end"],
                period_type=row["period_type"],
                preferred_dividends=release.preferred,
                prompt_tokens=int(token["prompt_tokens"]),
                kept=kept,
                left_out_reason=None if kept else left_out[release.accession]["reason"],
                input_sha256=release.input_sha256,
                text=release.text,
            )
        )
    if answer_rows:
        raise ExportError(
            f"{folder / splits.ANSWERS} has answers for releases outside the split: "
            + ", ".join(sorted(answer_rows)[:5])
        )
    return rows


def sorted_lines(rows: Iterable[Mapping[str, str]]) -> list[str]:
    return sorted(splits.csv_line(row, answers.COLUMNS) for row in rows)


# Answers


@dataclass(frozen=True)
class AnswerRow:
    """One answer of a split's key (data/splits/<split>/answers.csv), with where the
    grader finds it in the release's input: printed_as is grade.find_in_input's
    result (None: not_in_input, so never graded), spans every [start, end) of the
    input that the same search matches."""

    split: str
    ticker: str
    release_accession: str
    linked_accession: str
    field: str
    period_type: str
    start: str
    end: str
    value: str  # exact, as SEC wrote it (a derived quarter's: full year - nine months)
    unit: str
    tag: str
    derived: bool
    full_year_accession: str | None
    nine_months_accession: str | None
    fiscal_year_start: str | None
    in_input: bool
    printed_as: str | None
    spans: tuple[tuple[int, int], ...]


def answer_spans(
    value: Decimal,
    field: str,
    text: str,
    rows: Sequence[grade.InputRow] | None = None,
) -> tuple[str | None, tuple[tuple[int, int], ...]]:
    """What grade.find_in_input returns for the answer, and every span of text that
    its search matches for that form: the first printed form found (find_in_input's
    order); for a negative answer, the first way of signing that matches
    (parentheses, a minus sign, then unsigned on a row whose label says loss). The
    patterns are find_in_input's; load_answers checks the form against it for every
    answer."""
    if rows is None:
        rows = grade.input_rows(text)
    for printed in grade.printed_forms(value, field):
        form = printed.text
        number = re.escape(form)
        unsigned = grade.NUMBER_START + number + grade.NUMBER_END
        if value >= 0:
            if spans := find_all(unsigned, text):
                return form, spans
            continue
        if spans := find_all(rf"\(\s*\$?\s*{number}\s*\)", text):
            return f"({form})", spans
        minus = rf"[{grade.MINUS_CLASS}]\s*\$?\s*{number}{grade.NUMBER_END}"
        if spans := find_all(minus, text):
            return f"-{form}", spans
        loss_rows = [
            row
            for row in rows
            if grade.LOSS_LABEL.search(row.label) and re.search(unsigned, row.line)
        ]
        if loss_rows:
            lines = {row.line for row in loss_rows}
            spans = []
            offset = 0
            for line in text.splitlines(keepends=True):
                content = line.splitlines()[0] if line.splitlines() else ""
                if content in lines:
                    spans += [
                        (offset + a, offset + b) for a, b in find_all(unsigned, content)
                    ]
                offset += len(line)
            label = link.shorten(loss_rows[0].label)
            return f"{form} on a loss row ({label})", tuple(spans)
    return None, ()


def find_all(pattern: str, text: str) -> tuple[tuple[int, int], ...]:
    return tuple(m.span() for m in re.finditer(pattern, text))


def load_answers(
    sources: Sources, releases: Mapping[str, Sequence[ReleaseRow]]
) -> dict[str, list[AnswerRow]]:
    """Each split's answer key in answers.csv's order, with where each answer is
    printed. Stops if answer_spans and grade.find_in_input ever disagree."""
    found: dict[str, list[AnswerRow]] = {}
    for split, rows in releases.items():
        kept = {r.release_accession: r for r in rows if r.kept}
        path = sources.splits_dir / split / splits.ANSWERS
        try:
            table = splits.read_table(path, answers.COLUMNS)
        except splits.SplitError as exc:
            raise ExportError(str(exc)) from None
        cache: dict[str, list[grade.InputRow]] = {}
        out = []
        for row in table:
            release = kept.get(row["release_accession"])
            if release is None:
                raise ExportError(f"{path}: {row['release_accession']} isn't kept")
            text = release.text
            input_rows = cache.setdefault(
                release.release_accession, grade.input_rows(text)
            )
            value = Decimal(row["value"])
            graded_form = grade.find_in_input(value, row["field"], text, input_rows)
            form, spans = answer_spans(value, row["field"], text, input_rows)
            if form != graded_form or (form is None) != (not spans):
                raise ExportError(
                    f"{path}: {row['release_accession']} {row['field']} "
                    f"{row['period_type']}: answer_spans found {form!r}, "
                    f"grade.find_in_input {graded_form!r}"
                )
            out.append(
                AnswerRow(
                    split=split,
                    ticker=row["ticker"],
                    release_accession=row["release_accession"],
                    linked_accession=row["linked_accession"],
                    field=row["field"],
                    period_type=row["period_type"],
                    start=row["start"],
                    end=row["end"],
                    value=row["value"],
                    unit=row["unit"],
                    tag=row["tag"],
                    derived=row["derived"] == "True",
                    full_year_accession=row["full_year_accession"] or None,
                    nine_months_accession=row["nine_months_accession"] or None,
                    fiscal_year_start=row["fiscal_year_start"] or None,
                    in_input=form is not None,
                    printed_as=form,
                    spans=spans,
                )
            )
        found[split] = out
    return found


# Runs


@dataclass
class ExportRun:
    """One exported run: a run of comparison.md's main table (validation) or a test
    split's reported run (results/test_results.json)."""

    split: str
    key: str  # test runs: test_results.json's model key; validation: the run name
    run: compare.Run
    model: str  # display name
    source: str  # ANTHROPIC_API, OPENAI_API or SELF_HOSTED

    @property
    def name(self) -> str:
        return self.run.name

    @property
    def info(self) -> dict[str, Any]:
        return self.run.info

    @property
    def model_id(self) -> str:
        return self.info["model"]

    @property
    def prompt(self) -> str:
        return self.info["prompt_version"]

    @property
    def settings(self) -> str:
        return compare.settings_sent(self.info)

    @property
    def graded(self) -> list[dict[str, str]]:
        assert self.run.graded is not None
        return self.run.graded


def source(info: Mapping[str, Any]) -> str:
    """Where the model ran: a commercial API, or vLLM on a rented GPU."""
    if info["provider"] == extract.ANTHROPIC:
        return ANTHROPIC_API
    if compare.cost_basis(dict(info)) == extract.GPU_HOURS:
        return SELF_HOSTED
    return OPENAI_API


def display_name(run_name: str, info: Mapping[str, Any]) -> str:
    """The model's name in tables: stage 2's runs by checkpoint, Qwen3-8B with its
    thinking setting, the rest by MODEL_NAMES."""
    prefix = f"{TRAINING_RUN}--"
    if run_name.startswith(prefix + "step-"):
        return run_name.removeprefix(prefix)
    if run_name == prefix + "base":
        return "Qwen3-8B base"
    name = MODEL_NAMES.get(info["model"], info["model"])
    if info["model"] == "Qwen/Qwen3-8B":
        kwargs = info.get("request_parameters", {}).get("chat_template_kwargs", {})
        name += (
            " (no thinking)"
            if kwargs.get("enable_thinking") is False
            else (" (thinking)")
        )
    return name


def validation_runs(sources: Sources) -> list[ExportRun]:
    """The runs of results/comparison.md's main table: graded, not stopped, covering
    every linked release of data/links.csv (compare.format_comparison's rule), in
    compare's order. Stops on a stale one."""
    stage1 = sources.stage1
    key = grade.load_answers(stage1.answers)
    linked = {r.release_accession for r in answers.load_links(stage1.links)[0]}
    runs, _ = compare.load_runs(
        sources.runs, key, links_sha256=sha256_file(stage1.links)
    )
    full = [
        r
        for r in runs
        if r.info.get("status") != extract.ABORTED
        and r.graded is not None
        and linked <= r.covered
    ]
    stale = [r.name for r in full if r.stale]
    if stale:
        raise ExportError(f"stale runs, re-grade them first: {', '.join(stale)}")
    return [
        ExportRun(VALIDATION, r.name, r, display_name(r.name, r.info), source(r.info))
        for r in full
    ]


def test_runs(
    sources: Sources, results: Mapping[str, Any]
) -> dict[str, list[ExportRun]]:
    """Each test split's reported runs, in test_results.json's model order, checked:
    on the split's links file, covering exactly its kept releases, graded.csv
    current."""
    found = {}
    for split in TEST_SPLITS:
        folder = sources.splits_dir / split
        links_path = folder / splits.LINKS
        key = grade.load_answers(folder / splits.ANSWERS)
        kept = {r["release_accession"] for r in read_csv(links_path)}
        links_sha256 = sha256_file(links_path)
        found[split] = []
        for model, label in results["models"].items():
            name = results["splits"][split]["runs"][model]["run"]
            run = load_run(sources.runs / name)
            where = f"runs/{name}"
            if run.info.get("links_sha256") != links_sha256:
                raise ExportError(f"{where} isn't on {links_path}")
            if run.covered != kept:
                raise ExportError(f"{where} doesn't cover exactly {split}'s releases")
            if (
                run.graded is None
                or run.stale
                or compare.is_stale(run.graded, run.covered, key)
            ):
                raise ExportError(f"{where}: graded.csv isn't current; re-grade it")
            found[split].append(ExportRun(split, model, run, label, source(run.info)))
    return found


def load_run(folder: Path) -> compare.Run:
    info = read_json(folder / extract.RUN_INFO)
    path = folder / grade.GRADED
    graded, columns = None, []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            graded = list(reader)
            columns = list(reader.fieldnames or [])
    run = compare.Run(folder.name, info, graded, columns)
    run.stale = graded is not None and columns != grade.GRADED_COLUMNS
    return run


@dataclass(frozen=True)
class PredictionRow:
    """One graded figure of one run: graded.csv's row, with the model's name and
    where it ran. Figures only: never a model's reply text."""

    split: str
    run: str
    model: str
    model_id: str
    source: str
    prompt: str
    ticker: str
    release_accession: str
    field: str
    period_type: str
    answer: str | None
    answer_tag: str | None
    predicted: str | None  # raw dollars (EPS in dollars per share), as graded
    verdict: str
    factor: str | None  # the power of ten a unit slip was matched with
    evidence: str | None  # the grader's, from the release and XBRL


def prediction_rows(runs: Iterable[ExportRun]) -> list[PredictionRow]:
    rows = []
    for run in runs:
        for g in run.graded:
            rows.append(
                PredictionRow(
                    split=run.split,
                    run=run.name,
                    model=run.model,
                    model_id=run.model_id,
                    source=run.source,
                    prompt=run.prompt,
                    ticker=g["ticker"],
                    release_accession=g["release_accession"],
                    field=g["field"],
                    period_type=g["period_type"],
                    answer=g["answer"] or None,
                    answer_tag=g["answer_tag"] or None,
                    predicted=g["predicted"] or None,
                    verdict=g["verdict"],
                    factor=g["factor"] or None,
                    evidence=g["evidence"] or None,
                )
            )
    return rows


def validation_table(runs: Iterable[ExportRun]) -> list[dict[str, Any]]:
    """comparison.md's main-table cells for each validation run, by compare's own
    cell functions at $1 per GPU-hour, plus the counts behind its accuracy."""
    rows = []
    for r in runs:
        info, verdicts = r.info, r.run.verdicts()
        graded = [v for v in verdicts if v not in grade.UNGRADED]
        rows.append(
            {
                "run": r.name,
                "model": r.model,
                "model_id": r.model_id,
                "source": r.source,
                "prompt": r.prompt,
                "settings": r.settings,
                "concurrency": str(info.get("max_parallel", "")),
                "releases": str(info["releases"]),
                "accuracy": compare.accuracy(verdicts),
                **{f: compare.accuracy(r.run.verdicts(f)) for f in FIELD_NAMES},
                "unit_slips": r.run.unit_slips(),
                "format_retries": compare.format_retries(info),
                "input_tokens_per_release": compare.input_tokens_per_release(info),
                "cost": compare.cost(info, GPU_HOURLY),
                "cost_basis": compare.describe_basis(info, GPU_HOURLY),
                "cost_per_1000": compare.cost_per_thousand(info, GPU_HOURLY),
                "latency_p50": compare.latency(info, "p50"),
                "latency_p95": compare.latency(info, "p95"),
                "correct": graded.count(grade.CORRECT),
                "graded": len(graded),
            }
        )
    return rows


# Training examples


@dataclass(frozen=True)
class TrainRow:
    """One labelled training example (data/stage2/labels/train.jsonl, in its order),
    as chat messages, with whether the qwen3-8b-2k run trained on it: its position
    in that run's one epoch, and whether step-00025 had seen it."""

    release_accession: str
    ticker: str
    kind: str  # first quarter, second or third quarter, fourth quarter
    linked_form: str
    filing_date: str
    messages: tuple[dict[str, str], ...]  # system, user, assistant
    trained: bool
    position: int | None
    seen_by_checkpoint: bool


def examples_seen(sources: Sources) -> int:
    return int(read_json(sources.checkpoint / "state.json")["examples_seen"])


def training_positions(sources: Sources) -> dict[str, int]:
    """Each trained example's position in the run's one epoch."""
    rows = read_csv(sources.training / "examples.csv")
    if any(r["epoch"] != "1" for r in rows):
        raise ExportError(f"{sources.training / 'examples.csv'}: more than one epoch")
    positions = {r["release_accession"]: int(r["position"]) for r in rows}
    if len(positions) != len(rows) or sorted(positions.values()) != list(
        range(len(rows))
    ):
        raise ExportError(f"{sources.training / 'examples.csv'}: positions not 0..n-1")
    return positions


def train_rows(sources: Sources) -> Iterator[TrainRow]:
    """Streams the examples (train.jsonl is about 144 MB) with examples.csv's
    metadata, which must be in the same order."""
    index = read_csv(sources.labels / "examples.csv")
    positions = training_positions(sources)
    seen = examples_seen(sources)
    path = sources.labels / "train.jsonl"
    if not path.is_file():
        raise ExportError(f"{path} not found")
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if count >= len(index):
                raise ExportError(f"{path} has more lines than examples.csv")
            meta, record = index[count], json.loads(line)
            if record["release_accession"] != meta["release_accession"]:
                raise ExportError(f"{path} line {count + 1}: not examples.csv's order")
            roles = [m["role"] for m in record["messages"]]
            if roles != ["system", "user", "assistant"]:
                raise ExportError(f"{path} line {count + 1}: roles {roles}")
            position = positions.get(meta["release_accession"])
            yield TrainRow(
                release_accession=meta["release_accession"],
                ticker=meta["ticker"],
                kind=meta["kind"],
                linked_form=meta["linked_form"],
                filing_date=meta["filing_date"],
                messages=tuple(
                    {"role": m["role"], "content": m["content"]}
                    for m in record["messages"]
                ),
                trained=position is not None,
                position=position,
                seen_by_checkpoint=position is not None and position < seen,
            )
            count += 1
    if count != len(index):
        raise ExportError(f"{path} has {count} lines, examples.csv {len(index)}")
    missing = set(positions) - {r["release_accession"] for r in index}
    if missing:
        raise ExportError(
            f"trained examples not in examples.csv: {', '.join(sorted(missing)[:5])}"
        )


# Everything at once


@dataclass
class Context:
    """What every part of the export reads, loaded and checked once: the scored
    splits' releases and answer keys, the exported runs and the test results.
    Training examples are streamed (train_rows), not held here."""

    sources: Sources
    hf_user: str
    test_results: dict[str, Any]
    releases: dict[str, list[ReleaseRow]]
    answers: dict[str, list[AnswerRow]]
    validation_runs: list[ExportRun]
    test_runs: dict[str, list[ExportRun]]

    def runs(self, split: str) -> list[ExportRun]:
        return self.validation_runs if split == VALIDATION else self.test_runs[split]

    def repo_id(self, kind: str) -> str:
        return repo_id(self.hf_user, kind)

    def hub_url(self, kind: str) -> str:
        return hub_url(self.hf_user, kind)


def load_context(sources: Sources, hf_user: str) -> Context:
    results = load_test_results(sources)
    releases = load_releases(sources)
    return Context(
        sources=sources,
        hf_user=check_user(hf_user),
        test_results=results,
        releases=releases,
        answers=load_answers(sources, releases),
        validation_runs=validation_runs(sources),
        test_runs=test_runs(sources, results),
    )


# Writing


def as_json(value: Any) -> Any:
    """A row (dataclass) or value as plain JSON types; tuples become lists."""
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {k: as_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [as_json(v) for v in value]
    if isinstance(value, Decimal):
        return str(value)
    return value


def json_text(data: Any, *, indent: int | None = 2) -> str:
    """Deterministic JSON: keys in insertion order, UTF-8 as is, one final newline;
    compact without an indent."""
    if indent is None:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    return json.dumps(data, ensure_ascii=False, indent=indent) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(path, text)


def write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    write_text(path, json_text(as_json(data), indent=indent))


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    """One compact JSON object per line, written to a .part file and moved into
    place; returns the number of lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    count = 0
    with partial.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json_text(as_json(row), indent=None))
            count += 1
    partial.replace(path)
    return count


def copy_file(source_path: Path, target: Path) -> None:
    """The bytes only (no timestamps or permissions), so two builds are identical."""
    if not source_path.is_file():
        raise ExportError(f"{source_path} not found")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target)


@dataclass
class Written:
    """What a build step wrote and what it skipped, for the build's summary."""

    files: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
