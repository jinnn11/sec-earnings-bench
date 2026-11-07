"""Stage 2 test report: results/test_protocol.md's analysis of both test sets.

Reads the reported test runs (five models on the company test and on the time
test), each model's validation reference run, the frozen split files and
configs/report/stage2.toml, which holds only the facts the run files don't: which
runs are reported, the GPU price billed, the instance freeze, the attempts and their
causes in words, and notes. Writes results/test_results.md and
results/test_results.json (the unrounded values the md rounds); both are
deterministic. No network, GPU or API call: runs/, data/ and training/ are only
read, and graded.csv is re-graded in memory, never rewritten.

Checks come first; any failure stops the report with a message saying which:
- frozen items (protocol, "Frozen from this commit"): the protocol's sha256, every
  data/splits/<split>/*.csv by the protocol's table (validation, company_test) or
  PLAN.md Decisions 2026-11-07 (time_test), data/frozen_test_releases.csv (its
  validation and company_test rows as at cf77acb), the prompt, the test config, each
  test record's config sha256, adapter and layout check, and no change to the frozen
  code, prompt or test config since cf77acb (git diff);
- each reported run: complete, with no selection, its predictions and calls covering
  exactly the split's kept releases, no stop and no API error but refusals (the
  model's answer, graded missing; anything else is an infrastructure error, whose run
  the protocol moves aside and reruns), the frozen links and answers, the split's
  commit, clean;
- graded.csv is current: not stale (compare.is_stale), its predicted values are
  predictions.jsonl's, and re-grading in memory (grade.grade_run with an offline
  client) gives graded.csv byte for byte;
- settings equal to the model's validation reference run (provider, model,
  request_parameters, prompt_sha256, price_per_million_tokens); other differences
  (concurrency, settings file, GPU price) are shown as notes;
- every run on a split has the same graded keys and the same not_in_input keys.

The analysis is the protocol's: per split, each model's accuracy (correct over
graded figures, graded meaning a verdict outside grade.UNGRADED) overall, by field
and by period type; step-00025 minus the base model (the primary result on the
company test) and minus each API model, with the discordant figures and the
companies they come from; the same without the releases in preferred.csv, beside
each model's net income on those releases alone (counts only); cost (GPU runs at the
price billed and at $1/h; API runs at their token prices) and latency per run.

Bootstrap (details fixed on 2026-11-07, after the runs were graded): the split's
companies with at least one graded figure, sorted by ticker;
rng = numpy.random.default_rng(20261105); draws = rng.integers(0, n, size=(10000,
n)); each draw's accuracy = sum(times drawn x correct) / sum(times drawn x graded),
in points; intervals = numpy.percentile(values, [2.5, 97.5]). One set of draws per
split, shared by every model, subset and difference; in a subset, a company with no
figures left adds 0 to both sums, and a draw with no graded figures stops the report
(draws are never dropped). numpy comes with pandas.

Beyond the protocol, labelled descriptive (not pre-registered): the bootstrap's
robustness to company order, step-00025's misses grouped by the rule in
group_misses, abstention (answers given where the release doesn't print the
answer), and the company test's instance freeze.

Usage: uv run python -m secbench.testreport [--config configs/report/stage2.toml]
       [--out-dir results]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import re
import statistics
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from secbench import compare, grade, splits
from secbench.answers import (
    FIELDS,
    QUARTER,
    YEAR_TO_DATE,
    Answer,
    LinkedRelease,
    load_links,
)
from secbench.companies import COMPANY_TEST, VALIDATION
from secbench.companies import CSV_COLUMNS as COMPANY_COLUMNS
from secbench.companies import DEFAULT_OUT as COMPANIES_CSV
from secbench.edgar import EdgarClient
from secbench.extract import ABORTED, API_ERROR, COMPLETE, GPU_HOURS, RUN_INFO
from secbench.grade import (
    CORRECT,
    GRADED,
    GRADED_COLUMNS,
    NET_INCOME,
    NOT_IN_INPUT,
    PERIOD_ORDER,
    PREDICTIONS,
    UNGRADED,
    VERDICTS,
    InputRow,
    Prediction,
)
from secbench.inputs import build_input, release_records
from secbench.link import shorten
from secbench.releases import write_text
from secbench.stage2 import DEFAULT_OUT as STAGE2_DIR
from secbench.stage2 import RELEASE_COLUMNS, REPO, TIME_TEST, TIME_TEST_START, UNTIL

DEFAULT_CONFIG = Path("configs/report/stage2.toml")
DEFAULT_OUT_DIR = Path("results")
RESULTS_MD = "test_results.md"
RESULTS_JSON = "test_results.json"
TEST_SPLITS = (COMPANY_TEST, TIME_TEST)
SPLIT_TITLES = {COMPANY_TEST: "Company test", TIME_TEST: "Time test"}
FIELD_NAMES = [f.name for f in FIELDS]
SUBSETS = ("all", *FIELD_NAMES, *PERIOD_ORDER)
WITHOUT_PREFERRED = "without_preferred"

# What the protocol froze (results/test_protocol.md, "Frozen from this commit"), and
# the time test's split files (PLAN.md Decisions 2026-11-07, commit e352c14).
PROTOCOL = Path("results/test_protocol.md")
PROTOCOL_SHA256 = "99767156e46cc99cf80e62688339aeef4d119419745413c625e97a356dd9c186"
PROTOCOL_COMMIT = "cf77acb9e502769aff36b31d35e288a8fbb05c0e"
TIME_TEST_COMMIT = "e352c14a4ba4a5fa522a33b892d7b0cd0cecd1b0"
RUN_COMMITS = {COMPANY_TEST: PROTOCOL_COMMIT, TIME_TEST: TIME_TEST_COMMIT}
SPLIT_FILES = {
    VALIDATION: {
        "links.csv": "9613ce78eaec205cbeba07b8faf9252acd92e868aea4cf4451d83bc4a24bff75",
        "answers.csv": (
            "d56762363f6abdd224ca35ed0acb261411ed279f03f3e59e75c8f556b489de15"
        ),
        "tokens.csv": "e1aaa95fa1a8c0704dde37cafe2c2fb62a1cf4483010e017cf522323cc79cf43",
        "left_out.csv": (
            "0f3dc7f1f52c6e1a5ab00038cbffbf7683995b674ba492bd889bbed7070d1ab1"
        ),
        "preferred.csv": (
            "ee97a30a04bc7d07b5d89f886e9caa3ec1e5f6c2737ed624ef2b4b42a92ba832"
        ),
    },
    COMPANY_TEST: {
        "links.csv": "382791e2ab011a8669a53268eb474bcf26f60107070ce110191f36391f893af0",
        "answers.csv": (
            "07001d58c10057f8e0168f8102f4a8badec247e8f802851c70146c6b42488cab"
        ),
        "tokens.csv": "eba3ab4b4873b572b8a67a8a4e58b1c8ee99c78dab36366019035d14f4fb0513",
        "left_out.csv": (
            "0f3dc7f1f52c6e1a5ab00038cbffbf7683995b674ba492bd889bbed7070d1ab1"
        ),
        "preferred.csv": (
            "31b53b1d4f6a137107d3949229a22a31350c5892d3baa925a82b4c0a517a8e71"
        ),
    },
    TIME_TEST: {
        "links.csv": "abca9a8fd1e675c975e29bf82997347cef1a5cb422d725edbab4f4a54c776c92",
        "answers.csv": (
            "850f99c37347e439bdade1bcd8778d0d54d306bc105f7ee2db2e8315a9d2dd61"
        ),
        "tokens.csv": "fafa2f0dea41f50a7a3d682592ed06aea39b03bd041c3e7b16db4655fd3f37ad",
        "left_out.csv": (
            "e91e49a8c3b79661fd2fd17b2806710439a6b3900ef78f95d57202219ccc8b17"
        ),
        "preferred.csv": (
            "023ed02e6966301cb49c2fddf0e8379e8d4435f05c5dd91ba1c646e7afb4c6df"
        ),
    },
}
FROZEN_LIST_SHA256 = "46c185e27f643970f909c03e29eda1a49de26795fc7b9bf734e3679b5a28ae7b"
# The list as at cf77acb: its validation and company_test rows.
FROZEN_LIST_AT_PROTOCOL_SHA256 = (
    "6cba5eb290e21e6a0c34c08115b72ece94e8901ceae5e73c23eb91ae5e048101"
)
PROTOCOL_SPLITS = (VALIDATION, COMPANY_TEST)
PROMPT = Path("prompts/extract_v2.txt")
PROMPT_SHA256 = "2e9e989384563ebf94db29e78604c0d78187767bcb688645fbafa3777c9ece29"
TEST_CONFIG = Path("configs/test/qwen3-8b-2k.toml")
TEST_CONFIG_SHA256 = "e211fe0da1e99fc02483d4c961cb96d23366d29c0f0b0756f40bc03df6b30155"
ADAPTER_SHA256 = "b93376d91af812159f4bd3a0fb80be090808e6d3539b251dc53ed26b9d27041b"
TRAINING_RUN = Path("training/qwen3-8b-2k")
FROZEN_CODE = (
    "src/secbench/inputs.py",
    "src/secbench/link.py",
    "src/secbench/answers.py",
    "src/secbench/extract.py",
    "src/secbench/grade.py",
    "src/secbench/splits.py",
    str(PROMPT),
    str(TEST_CONFIG),
)

SEED = 20261105
DRAWS = 10_000
PERCENTILES = (2.5, 97.5)
NO_SELECTION = {"tickers": None, "accessions": None, "limit": None}
REFUSAL = "refusal"  # extract's reason for an API error that is a refusal
# Compared with the model's validation reference run (protocol, References), and
# shown when they differ.
SETTINGS = (
    "provider",
    "model",
    "request_parameters",
    "prompt_sha256",
    "price_per_million_tokens",
)
SETTING_NOTES = ("max_parallel", "env_file", "gpu_hourly_usd")
CHOSEN_ON_VALIDATION = (
    "chosen on validation (the checkpoint was picked for that score, so it "
    "overstates; the test sets give the estimate)"
)

# step-00025's misses (descriptive; see group_misses): (1) net income printed on a
# row naming common shareholders, stockholders, shares or stock; (2) the quarter
# copied into year to date; (3) the rest, by verdict.
COMMON_LINE = re.compile(
    r"\bcommon\s+(?:shareholders?|stockholders?|shares?|stock)\b", re.IGNORECASE
)
COMMON = "common shareholders' line"
QUARTER_AS_YTD = "quarter copied into year to date"
PREFERRED_ROW = re.compile(r"\bpreferred\b", re.IGNORECASE)
PARTICIPATING_ROW = re.compile(r"\bparticipating\b", re.IGNORECASE)

TIMESTAMP = re.compile(r"\b(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\b")
THROUGHPUT = "Avg prompt throughput"

Key = tuple[str, str, str]  # (release_accession, field, period_type)


class ReportError(Exception):
    """The report can't be written; the message says why."""


# Config


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    validation: Path
    runs: Mapping[str, Path]  # split -> reported run


@dataclass(frozen=True)
class Attempt:
    split: str
    model: str
    run: Path
    cause: str  # in words; the recorded error text is never copied
    rerun_env_file: str = ""
    rerun_settings: str = ""


@dataclass(frozen=True)
class Freeze:
    split: str
    model: str
    start: datetime
    end: datetime

    @property
    def seconds(self) -> float:
        return (self.end - self.start).total_seconds()


@dataclass(frozen=True)
class Note:
    split: str
    release: str
    note: str


@dataclass(frozen=True)
class Config:
    path: Path
    under_test: str
    base: str
    bootstrap_details_fixed: date
    gpu_host: str
    models: tuple[Model, ...]
    billed: Mapping[str, float]  # split -> $/h billed for its GPU session
    validation_gpu: str
    recorded_price_note: Mapping[str, str]  # split -> why run.json's price differs
    logs: Mapping[str, Mapping[str, Path]]  # split -> {"server", "setup"}
    freeze: Freeze | None
    attempts: tuple[Attempt, ...]
    notes: tuple[Note, ...]

    def model(self, key: str) -> Model:
        for model in self.models:
            if model.key == key:
                return model
        raise ReportError(f"{self.path}: no model {key!r}")

    @property
    def references(self) -> list[str]:
        """The models step-00025 is compared with: the base model, then the APIs."""
        return [m.key for m in self.models if m.key != self.under_test]


def load_config(path: Path) -> Config:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ReportError(f"{path}: {exc}") from None
    try:
        models = tuple(
            Model(
                m["key"],
                m["label"],
                Path(m["validation"]),
                {split: Path(m[split]) for split in TEST_SPLITS},
            )
            for m in raw["models"]
        )
        freeze = raw.get("freeze")
        config = Config(
            path=path,
            under_test=raw["under_test"],
            base=raw["base"],
            bootstrap_details_fixed=raw["bootstrap_details_fixed"],
            gpu_host=raw["gpu_host"],
            models=models,
            billed={
                split: float(raw["gpu"][split]["billed_usd_per_hour"])
                for split in (VALIDATION, *TEST_SPLITS)
            },
            validation_gpu=raw["gpu"][VALIDATION]["gpu"],
            recorded_price_note={
                split: raw["gpu"][split]["recorded_price_note"]
                for split in TEST_SPLITS
                if "recorded_price_note" in raw["gpu"][split]
            },
            logs={
                split: {kind: Path(p) for kind, p in raw["logs"][split].items()}
                for split in TEST_SPLITS
            },
            freeze=Freeze(
                freeze["split"], freeze["model"], freeze["start"], freeze["end"]
            )
            if freeze
            else None,
            attempts=tuple(
                Attempt(
                    a["split"],
                    a["model"],
                    Path(a["run"]),
                    a["cause"],
                    a.get("rerun_env_file", ""),
                    a.get("rerun_settings", ""),
                )
                for a in raw.get("attempts", [])
            ),
            notes=tuple(
                Note(n["split"], n["release"], n["note"]) for n in raw.get("notes", [])
            ),
        )
    except (KeyError, TypeError) as exc:
        raise ReportError(f"{path}: missing or bad setting {exc}") from None
    keys = [m.key for m in models]
    for key in (config.under_test, config.base):
        if key not in keys:
            raise ReportError(f"{path}: {key!r} is not a model")
    if keys[:2] != [config.under_test, config.base]:
        raise ReportError(f"{path}: list the model under test, then its base, first")
    return config


# Runs


@dataclass
class Run:
    model: str
    path: Path
    info: dict[str, Any]
    graded_text: str | None
    predictions: dict[str, Prediction]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def covered(self) -> set[str]:
        return {call["release_accession"] for call in self.info.get("calls", [])}

    @cached_property
    def rows(self) -> list[dict[str, str]]:
        if self.graded_text is None:
            raise ReportError(f"{self.path / GRADED} not found")
        return list(csv_rows(self.graded_text))

    @cached_property
    def by_key(self) -> dict[Key, dict[str, str]]:
        return {row_key(row): row for row in self.rows}

    @cached_property
    def graded(self) -> dict[Key, dict[str, str]]:
        """Rows of graded figures: verdict not in grade.UNGRADED."""
        return {k: r for k, r in self.by_key.items() if r["verdict"] not in UNGRADED}

    def keys(self, verdict: str) -> set[Key]:
        return {k for k, r in self.by_key.items() if r["verdict"] == verdict}

    @property
    def api_errors(self) -> list[dict[str, Any]]:
        return [f for f in self.info.get("fallbacks", []) if f["outcome"] == API_ERROR]

    @property
    def refusals(self) -> int:
        return sum(f["reason"].startswith(REFUSAL) for f in self.api_errors)

    @property
    def gpu(self) -> bool:
        return self.info.get("cost_basis") == GPU_HOURS


def csv_rows(text: str) -> Iterable[dict[str, str]]:
    return csv.DictReader(io.StringIO(text))


def row_key(row: Mapping[str, str]) -> Key:
    return (row["release_accession"], row["field"], row["period_type"])


def load_run(model: str, path: Path, *, graded: bool = True) -> Run:
    info_path = path / RUN_INFO
    if not info_path.is_file():
        raise ReportError(f"{info_path} not found")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    predictions_path = path / PREDICTIONS
    predictions = {}
    if predictions_path.is_file():
        try:
            predictions = {
                p.release_accession: p for p in grade.load_predictions(predictions_path)
            }
        except grade.PredictionsError as exc:
            raise ReportError(str(exc)) from None
    graded_path = path / GRADED
    text = (
        graded_path.read_text(encoding="utf-8")
        if graded and graded_path.is_file()
        else None
    )
    return Run(model, path, info, text, predictions)


def predicted_value(prediction: Prediction, key: Key) -> Decimal | None:
    """A prediction's figure for a graded key: quarter is the quarter group;
    year_to_date and full_year are the year_to_date group."""
    _, name, period = key
    group = prediction.quarter if period == QUARTER else prediction.year_to_date
    return getattr(group, name)


# Splits


@dataclass
class Split:
    name: str
    folder: Path
    releases_dir: Path
    key: dict[Key, Answer]
    links: list[LinkedRelease]
    preferred: set[str]  # kept releases whose linked filing reports preferred dividends
    left_out: list[dict[str, str]]
    manifest: dict[str, Any]

    @property
    def kept(self) -> set[str]:
        return {r.release_accession for r in self.links}

    @property
    def ticker(self) -> dict[str, str]:
        return {r.release_accession: r.ticker for r in self.links}


def load_split(name: str, splits_dir: Path) -> Split:
    folder = splits_dir / name
    manifest_path = folder / splits.MANIFEST
    if not manifest_path.is_file():
        raise ReportError(f"{manifest_path} not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        preferred = splits.read_table(
            folder / splits.PREFERRED, splits.PREFERRED_COLUMNS
        )
        left_out = splits.read_table(
            folder / splits.LEFT_OUT_FILE, splits.LEFT_OUT_COLUMNS
        )
    except splits.SplitError as exc:
        raise ReportError(str(exc)) from None
    return Split(
        name,
        folder,
        Path(manifest["releases_dir"]),
        grade.load_answers(folder / splits.ANSWERS),
        load_links(folder / splits.LINKS)[0],
        {r["release_accession"] for r in preferred if r["status"] == splits.KEPT},
        left_out,
        manifest,
    )


# Checks


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ReportError(f"{path} not found")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Frozen:
    """What the protocol froze, by path (under root) and sha256."""

    root: Path = Path(".")
    protocol: Path = PROTOCOL
    protocol_sha256: str = PROTOCOL_SHA256
    splits_dir: Path = splits.DEFAULT_OUT
    split_files: Mapping[str, Mapping[str, str]] = field(
        default_factory=lambda: SPLIT_FILES
    )
    frozen_list: Path = splits.FROZEN_LIST
    frozen_list_sha256: str = FROZEN_LIST_SHA256
    frozen_list_at_protocol_sha256: str = FROZEN_LIST_AT_PROTOCOL_SHA256
    prompt: Path = PROMPT
    prompt_sha256: str = PROMPT_SHA256
    test_config: Path = TEST_CONFIG
    test_config_sha256: str = TEST_CONFIG_SHA256
    adapter_sha256: str = ADAPTER_SHA256
    test_records: Path = TRAINING_RUN / "test"
    commit: str = PROTOCOL_COMMIT
    code: tuple[str, ...] = FROZEN_CODE


def check_frozen(frozen: Frozen) -> list[str]:
    """Every frozen item unchanged; what was checked, one line each."""
    root = frozen.root
    problems = []

    def expect(path: Path, sha256: str, what: str) -> None:
        got = sha256_file(root / path)
        if got != sha256:
            problems.append(f"{what} {path}: sha256 {got[:8]}…, not {sha256[:8]}…")

    expect(frozen.protocol, frozen.protocol_sha256, "protocol")
    files = 0
    for split, expected in frozen.split_files.items():
        folder = root / frozen.splits_dir / split
        found = sorted(p.name for p in folder.glob("*.csv"))
        if found != sorted(expected):
            problems.append(f"{folder}: CSV files {found}, not {sorted(expected)}")
        for name, sha256 in expected.items():
            expect(frozen.splits_dir / split / name, sha256, "split file")
            files += 1
    expect(frozen.frozen_list, frozen.frozen_list_sha256, "frozen release list")
    lines = (root / frozen.frozen_list).read_text(encoding="utf-8").splitlines(True)
    at_protocol = lines[:1] + [
        line for line in lines[1:] if line.split(",", 1)[0] in PROTOCOL_SPLITS
    ]
    got = hashlib.sha256("".join(at_protocol).encode("utf-8")).hexdigest()
    if got != frozen.frozen_list_at_protocol_sha256:
        problems.append(
            f"{frozen.frozen_list}: its {' and '.join(PROTOCOL_SPLITS)} rows differ "
            f"from the protocol commit's file (sha256 {got[:8]}…)"
        )
    expect(frozen.prompt, frozen.prompt_sha256, "prompt")
    expect(frozen.test_config, frozen.test_config_sha256, "test config")
    for split in TEST_SPLITS:
        path = root / frozen.test_records / f"{split}.json"
        if not path.is_file():
            problems.append(f"{path} not found")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("test_config", {}).get("sha256") != frozen.test_config_sha256:
            problems.append(f"{path}: another test config")
        if record.get("checkpoint", {}).get("adapter_sha256") != frozen.adapter_sha256:
            problems.append(f"{path}: another adapter")
        if (
            record.get("layout_check", {}).get("identical_to_training_layout")
            is not True
        ):
            problems.append(f"{path}: prompts not identical to training's layout")
    changed = git_changed(root, frozen.commit, frozen.code)
    if changed:
        problems.append(f"changed since {frozen.commit[:7]}: {', '.join(changed)}")
    if problems:
        raise ReportError("frozen items changed: " + "; ".join(problems))
    return [
        f"protocol {frozen.protocol} sha256 {frozen.protocol_sha256[:8]}…",
        (
            f"{files} split files under {frozen.splits_dir}/ by sha256 (the protocol's "
            "table; the time test's from PLAN.md Decisions 2026-11-07)"
        ),
        (
            f"{frozen.frozen_list} sha256 {frozen.frozen_list_sha256[:8]}…, its "
            f"{' and '.join(PROTOCOL_SPLITS)} rows as at {frozen.commit[:7]} "
            f"({frozen.frozen_list_at_protocol_sha256[:8]}…)"
        ),
        (
            f"{frozen.prompt} {frozen.prompt_sha256[:8]}…, {frozen.test_config} "
            f"{frozen.test_config_sha256[:8]}…"
        ),
        (
            f"{frozen.test_records}/<split>.json: that config, adapter "
            f"{frozen.adapter_sha256[:8]}…, prompts identical to training's layout"
        ),
        (
            f"no change since {frozen.commit[:7]} (git diff, committed and working tree) "
            f"to {', '.join(frozen.code)}"
        ),
    ]


def git_changed(root: Path, commit: str, paths: Sequence[str]) -> list[str]:
    """paths that differ between commit and HEAD, or between HEAD and the working
    tree."""
    changed: list[str] = []
    for revisions in ([commit, "HEAD"], ["HEAD"]):
        args = ["git", "-C", str(root), "diff", "--quiet", *revisions, "--", *paths]
        result = subprocess.run(args, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            continue
        if result.returncode != 1:
            raise ReportError(f"{' '.join(args)}: {result.stderr.strip()}")
        names = subprocess.run(
            [*args[:4], "--name-only", *args[5:]],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        changed.extend(n for n in names if n not in changed)
    return changed


def check_run(run: Run, split: Split, commit: str) -> None:
    """A reported run is complete, unselected and on exactly the split's kept
    releases, with no stop and no infrastructure API error (refusals are the
    model's answers), on the frozen split files at the split's commit."""
    info, problems = run.info, []
    if info.get("status") != COMPLETE:
        problems.append(f"status {info.get('status')!r}, not {COMPLETE!r}")
    if info.get("selection") != NO_SELECTION:
        problems.append(f"selection {info.get('selection')}, not none")
    kept = split.kept
    for what, got in (
        ("run.json's calls", run.covered),
        (PREDICTIONS, set(run.predictions)),
    ):
        if got != kept:
            problems.append(
                f"{what} cover {len(got)} releases ({len(got - kept)} not kept, "
                f"{len(kept - got)} kept ones missing), not the {len(kept)} kept"
            )
    outcomes = info.get("outcomes", {})
    if outcomes.get(ABORTED):
        problems.append(f"{outcomes[ABORTED]} aborted calls")
    if outcomes.get(API_ERROR, 0) != len(run.api_errors):
        problems.append("outcomes and fallbacks disagree on API errors")
    for error in run.api_errors:
        if not error["reason"].startswith(REFUSAL):
            problems.append(
                f"{error['ticker']} {error['release_accession']}: an API error that "
                "isn't a refusal (an infrastructure error: the protocol moves the run "
                "to <name>--attempt-N and reruns it)"
            )
    for name, file in (
        ("links_sha256", "links.csv"),
        ("answers_sha256", "answers.csv"),
    ):
        if info.get(name) != SPLIT_FILES[split.name][file]:
            problems.append(f"{name} isn't the frozen {split.name} {file}'s")
    if info.get("commit") != commit:
        problems.append(f"commit {str(info.get('commit'))[:7]}, not {commit[:7]}")
    if info.get("clean") is not True:
        problems.append("not run from a clean working tree")
    if problems:
        raise ReportError(f"{run.name}: " + "; ".join(problems))


Regrade = Callable[[Run, Split], str]


def regrade_offline(run: Run, split: Split) -> str:
    """graded.csv as grade.grade_run writes it now, from the cache only."""
    client = EdgarClient(user_agent="offline check", offline=True)
    try:
        graded, _ = grade.grade_run(
            run.path,
            split.folder / splits.LINKS,
            split.folder / splits.ANSWERS,
            Path(run.info["releases_dir"]),
            client,
        )
    except (grade.GradeError, grade.PredictionsError) as exc:
        raise ReportError(f"{run.name}: re-grading failed: {exc}") from None
    return grade.graded_csv(graded)


def check_graded(run: Run, split: Split, regrade: Regrade) -> None:
    """graded.csv is current: the grader's columns, not stale, its predicted values
    equal predictions.jsonl's (every non-null figure has a row), and re-grading in
    memory gives the same bytes."""
    problems = []
    header = (run.graded_text or "").split("\n", 1)[0].split(",")
    if header != GRADED_COLUMNS:
        problems.append(f"{GRADED} has columns {header}, not the grader's")
    elif compare.is_stale(run.rows, run.covered, split.key):
        problems.append(f"{GRADED} is stale: its answers aren't the split's")
    else:
        seen: set[tuple[str, str, bool]] = set()  # (release, field, is quarter)
        for key, row in run.by_key.items():
            prediction = run.predictions.get(key[0])
            if prediction is None:
                problems.append(f"{key[0]}: graded but not predicted")
                continue
            got = Decimal(row["predicted"]) if row["predicted"] else None
            if got != predicted_value(prediction, key):
                problems.append(
                    f"{' '.join(key)}: predicted {got}, not the prediction's"
                )
            seen.add((key[0], key[1], key[2] == QUARTER))
        for accession, prediction in run.predictions.items():
            for period in (QUARTER, YEAR_TO_DATE):
                for name in FIELD_NAMES:
                    value = predicted_value(prediction, (accession, name, period))
                    if value is not None and (
                        (accession, name, period == QUARTER) not in seen
                    ):
                        problems.append(
                            f"{accession} {period} {name}: predicted, with no row"
                        )
    if not problems and regrade(run, split) != run.graded_text:
        problems.append(f"re-grading in memory doesn't give {GRADED} byte for byte")
    if problems:
        shown = problems[:5] + (
            [f"{len(problems) - 5} more"] if len(problems) > 5 else []
        )
        raise ReportError(f"{run.name}: " + "; ".join(shown))


def check_settings(run: Run, reference: Run) -> list[str]:
    """SETTINGS must equal the reference run's; SETTING_NOTES' differences, shown."""
    different = [
        name for name in SETTINGS if run.info.get(name) != reference.info.get(name)
    ]
    if different:
        raise ReportError(
            f"{run.name}: {', '.join(different)} differ from its validation reference "
            f"{reference.name}"
        )
    return [
        f"{name} {run.info.get(name)} (reference {reference.info.get(name)})"
        for name in SETTING_NOTES
        if run.info.get(name) != reference.info.get(name)
    ]


def check_same_keys(runs: Sequence[Run]) -> tuple[int, int]:
    """Every run has the same graded keys and not_in_input keys: their counts."""
    first = runs[0]
    for run in runs[1:]:
        for what, mine, theirs in (
            ("graded", set(run.graded), set(first.graded)),
            (NOT_IN_INPUT, run.keys(NOT_IN_INPUT), first.keys(NOT_IN_INPUT)),
        ):
            if mine != theirs:
                differ = sorted(mine ^ theirs)
                raise ReportError(
                    f"{run.name} and {first.name} have different {what} keys: "
                    + ", ".join(" ".join(k) for k in differ[:5])
                    + (f" and {len(differ) - 5} more" if len(differ) > 5 else "")
                )
    return len(first.graded), len(first.keys(NOT_IN_INPUT))


# Bootstrap


def times_drawn(picks: np.ndarray, n: int) -> np.ndarray:
    """(draws, n) counts: how many times each company is in each draw."""
    draws = picks.shape[0]
    flat = picks + n * np.arange(draws)[:, None]
    return np.bincount(flat.ravel(), minlength=draws * n).reshape(draws, n)


class Bootstrap:
    """One set of draws of companies with replacement, shared by every model,
    subset and difference of a split."""

    def __init__(self, companies: Sequence[str], seed: int = SEED, draws: int = DRAWS):
        self.companies = list(companies)
        self.index = {c: i for i, c in enumerate(self.companies)}
        n = len(self.companies)
        rng = np.random.default_rng(seed)
        self.counts = times_drawn(rng.integers(0, n, size=(draws, n)), n)

    def per_company(self, tickers: Iterable[str]) -> np.ndarray:
        vector = np.zeros(len(self.companies), dtype=np.int64)
        for ticker in tickers:
            vector[self.index[ticker]] += 1
        return vector

    def accuracies(self, figures: Sequence[tuple[str, bool]], what: str) -> np.ndarray:
        """Each draw's accuracy in points over figures (ticker, correct). A draw with
        no graded figure stops the report: draws are never dropped."""
        graded = self.counts @ self.per_company(t for t, _ in figures)
        empty = int((graded == 0).sum())
        if empty:
            raise ReportError(
                f"{what}: {empty} of {len(graded):,} draws have no graded figure, so "
                "the interval is undefined (draws are not dropped)"
            )
        correct = self.counts @ self.per_company(t for t, ok in figures if ok)
        return 100 * correct / graded

    def without(self, tickers: Iterable[str]) -> float:
        """The share of draws that contain none of these companies."""
        columns = [self.index[t] for t in set(tickers)]
        return float((self.counts[:, columns].sum(axis=1) == 0).mean())


def interval(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(values, PERCENTILES)]


# Analysis


@dataclass
class Table:
    """A split's graded figures, the same keys in every run, in graded.csv order."""

    keys: list[Key]
    ticker: dict[Key, str]
    correct: dict[str, dict[Key, bool]]  # model -> key -> correct

    @classmethod
    def from_runs(cls, runs: Mapping[str, Run]) -> Table:
        first = next(iter(runs.values()))
        keys = list(first.graded)
        return cls(
            keys,
            {k: first.graded[k]["ticker"] for k in keys},
            {
                model: {k: run.graded[k]["verdict"] == CORRECT for k in keys}
                for model, run in runs.items()
            },
        )

    def figures(self, model: str, keys: Iterable[Key]) -> list[tuple[str, bool]]:
        return [(self.ticker[k], self.correct[model][k]) for k in keys]


def subset_keys(table: Table, subset: str, drop: set[str] = frozenset()) -> list[Key]:
    """The keys of a subset (all, a field or a period type), without the releases in
    drop."""
    return [
        k
        for k in table.keys
        if k[0] not in drop and (subset == "all" or subset in (k[1], k[2]))
    ]


def accuracy_entry(
    table: Table, model: str, keys: list[Key], boot: Bootstrap, what: str
) -> dict[str, Any]:
    figures = table.figures(model, keys)
    correct = sum(ok for _, ok in figures)
    return {
        "correct": correct,
        "graded": len(figures),
        "accuracy": 100 * correct / len(figures),
        "interval": interval(boot.accuracies(figures, what)),
    }


def difference_entry(
    table: Table, a: str, b: str, keys: list[Key], boot: Bootstrap, what: str
) -> dict[str, Any]:
    first, second = table.figures(a, keys), table.figures(b, keys)
    values = boot.accuracies(first, what) - boot.accuracies(second, what)
    a_only = [k for k in keys if table.correct[a][k] and not table.correct[b][k]]
    b_only = [k for k in keys if table.correct[b][k] and not table.correct[a][k]]
    return {
        "difference": 100
        * (sum(ok for _, ok in first) - sum(ok for _, ok in second))
        / len(keys),
        "interval": interval(values),
        "discordant": len(a_only) + len(b_only),
        "first_right": len(a_only),
        "second_right": len(b_only),
        "companies": len({table.ticker[k] for k in a_only + b_only}),
    }


def analyse_accuracy(
    config: Config, split: Split, table: Table, boot: Bootstrap
) -> dict[str, Any]:
    models = [m.key for m in config.models]
    without = subset_keys(table, "all", split.preferred)
    # Subsets with figures: a split may have no full-year figure at all.
    subsets = [s for s in SUBSETS if subset_keys(table, s)]
    accuracy = {
        model: {
            subset: accuracy_entry(
                table, model, subset_keys(table, subset), boot, f"{model} {subset}"
            )
            for subset in subsets
        }
        | {
            WITHOUT_PREFERRED: accuracy_entry(
                table, model, without, boot, f"{model} {WITHOUT_PREFERRED}"
            )
        }
        for model in models
    }
    differences = {
        other: {
            "all": difference_entry(
                table, config.under_test, other, table.keys, boot, f"minus {other}"
            ),
            WITHOUT_PREFERRED: difference_entry(
                table,
                config.under_test,
                other,
                without,
                boot,
                f"minus {other} {WITHOUT_PREFERRED}",
            ),
        }
        for other in config.references
    }
    on_preferred = [
        k for k in table.keys if k[0] in split.preferred and k[1] == NET_INCOME
    ]
    preferred_companies = {
        table.ticker[k] for k in table.keys if k[0] in split.preferred
    }
    return {
        "accuracy": accuracy,
        "differences": differences,
        "preferred": {
            "releases": len(split.preferred),
            "figures": len(table.keys) - len(without),
            "companies": sorted(preferred_companies),
            "draws_without_them": boot.without(preferred_companies),
            "net_income": {
                model: {
                    "correct": sum(table.correct[model][k] for k in on_preferred),
                    "graded": len(on_preferred),
                }
                for model in models
            },
        },
    }


def robustness(
    table: Table, under_test: str, base: str, orders: Mapping[str, Sequence[str]]
) -> dict[str, list[float]]:
    """The primary interval with the companies in other orders (descriptive)."""
    result = {}
    for name, companies in orders.items():
        boot = Bootstrap(companies)
        values = boot.accuracies(
            table.figures(under_test, table.keys), name
        ) - boot.accuracies(table.figures(base, table.keys), name)
        result[name] = interval(values)
    return result


def company_orders(
    split: Split, companies: Sequence[str], ciks: Mapping[str, int]
) -> dict[str, list[str]]:
    """The bootstrap's companies by ticker, in links.csv order (first release) and
    by CIK (data/companies.csv)."""
    wanted = set(companies)
    links_order = list(
        dict.fromkeys(r.ticker for r in split.links if r.ticker in wanted)
    )
    return {
        "ticker": sorted(wanted),
        "links_order": links_order,
        "cik": sorted(wanted, key=lambda t: (ciks[t], t)),
    }


def read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        return splits.read_table(path, columns)
    except splits.SplitError as exc:
        raise ReportError(str(exc)) from None


def coverage(
    split: Split,
    table: Table,
    stage2_releases: Path,
    sectors: Mapping[str, str],
) -> dict[str, Any]:
    """The answer key over the kept releases (data/stage2/releases.csv's answers and
    missing), and which releases and companies have a graded figure."""
    rows = {
        r["release_accession"]: r for r in read_csv(stage2_releases, RELEASE_COLUMNS)
    }
    answers_per_release = Counter(a.release_accession for a in split.key.values())
    kept = list(split.links)
    by_sector: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    answered = missing = 0
    for release in kept:
        row = rows.get(release.release_accession)
        if row is None:
            raise ReportError(f"{release.release_accession}: not in {stage2_releases}")
        if int(row["answers"]) != answers_per_release[release.release_accession]:
            raise ReportError(
                f"{release.release_accession}: {stage2_releases} counts "
                f"{row['answers']} answers, {split.folder}/answers.csv "
                f"{answers_per_release[release.release_accession]}"
            )
        answered += int(row["answers"])
        missing += int(row["missing"])
        sector = sectors[release.ticker]
        by_sector[sector][1] += 1
        by_sector[sector][0] += int(row["answers"]) == 0
    graded_releases = {k[0] for k in table.keys}
    without_graded = [
        {
            "ticker": r.ticker,
            "release_accession": r.release_accession,
            "answers": answers_per_release[r.release_accession],
            "sector": sectors[r.ticker],
        }
        for r in kept
        if r.release_accession not in graded_releases
    ]
    counts = split.manifest["counts"]
    if answered != counts["answers"] or len(kept) != counts["kept"]:
        raise ReportError(f"{split.folder}: manifest counts disagree with its files")
    return {
        "expected": answered + missing,
        "answered": answered,
        "missing": missing,
        "in_input": counts["answers_in_input"],
        "graded": len(table.keys),
        "kept_releases": len(kept),
        "kept_companies": len({r.ticker for r in kept}),
        "releases_graded": len(graded_releases),
        "companies_graded": len({table.ticker[k] for k in table.keys}),
        "no_answer_by_sector": {
            sector: {"no_answer": n, "kept": total}
            for sector, (n, total) in sorted(by_sector.items())
        },
        "releases_without_graded": without_graded,
    }


def verdict_counts(run: Run) -> dict[str, int]:
    counts = Counter(row["verdict"] for row in run.rows)
    return {v: counts[v] for v in VERDICTS}


def unit_slips(run: Run) -> int:
    return sum(bool(row["factor"]) for row in run.rows)


def abstention(run: Run) -> dict[str, int]:
    """not_in_input figures the model still answered, and its null figures among all
    it predicted."""
    absent = [run.by_key[k] for k in run.keys(NOT_IN_INPUT)]
    values = [
        getattr(getattr(p, group), name)
        for p in run.predictions.values()
        for group in ("quarter", "year_to_date")
        for name in FIELD_NAMES
    ]
    return {
        "not_in_input": len(absent),
        "answered": sum(bool(r["predicted"]) for r in absent),
        "figures": len(values),
        "null": sum(v is None for v in values),
    }


# Costs


def cost_entry(run: Run, billed: float | None) -> dict[str, Any]:
    """A GPU run at the price billed and at $1/h (price x wall time); an API run at
    its token prices (run.json's estimate). Per run and per 1,000 releases."""
    info, releases = run.info, run.info["releases"]

    def per_thousand(value: float) -> float:
        return value / releases * 1000

    if run.gpu:
        if billed is None:
            raise ReportError(f"{run.name}: a GPU run without a billed price")
        hours = info["wall_time_s"] / 3600
        return {
            "basis": "gpu",
            "wall_time_s": info["wall_time_s"],
            "recorded_usd_per_hour": info["gpu_hourly_usd"],
            "billed_usd_per_hour": billed,
            "cost": billed * hours,
            "per_1000": per_thousand(billed * hours),
            "cost_at_1": hours,
            "per_1000_at_1": per_thousand(hours),
            "cost_at_recorded": info["gpu_hourly_usd"] * hours,
            "per_1000_at_recorded": per_thousand(info["gpu_hourly_usd"] * hours),
            "releases": releases,
        }
    value = compare.cost_value(info)
    if value is None:
        raise ReportError(f"{run.name}: no estimated cost")
    return {
        "basis": "tokens",
        "wall_time_s": info["wall_time_s"],
        "prices": info["price_per_million_tokens"],
        "cost": value,
        "per_1000": per_thousand(value),
        "releases": releases,
    }


def freeze_entry(run: Run, base: Run, freeze: Freeze, log: Path, billed: float) -> dict:
    """The instance freeze in a GPU run (descriptive): the server log's gap, the
    run's time by its own (monotonic) clock, and a freeze-excluded estimate."""
    started = datetime.fromisoformat(run.info["started"])
    finished = datetime.fromisoformat(run.info["finished"])
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    stamped = [
        (t, THROUGHPUT in line)
        for line in lines
        if (t := line_time(line, started.year)) and started <= t <= finished
    ]
    times = [t for t, _ in stamped]
    gaps = [(b - a, a, b) for a, b in itertools.pairwise(times)]
    longest, gap_start, gap_end = max(gaps)
    if not (gap_start <= freeze.start and freeze.end <= gap_end):
        raise ReportError(
            f"{log}: no gap covering the configured freeze {freeze.start}-{freeze.end}"
        )
    throughput = [t for t, is_throughput in stamped if is_throughput]
    spacing = statistics.median(
        (b - a).total_seconds()
        for a, b in itertools.pairwise(throughput)
        if b - a < longest
    )

    def latencies(r: Run) -> list[float]:
        return sorted(c["latency_s"] for c in r.info["calls"])

    mine, theirs = latencies(run), latencies(base)
    width, base_width = run.info["max_parallel"], base.info["max_parallel"]
    monotonic = sum(mine) / width
    base_overhead = base.info["wall_time_s"] - sum(theirs) / base_width
    frozen_calls, others = mine[-width:], mine[:-width]
    excluded = run.info["wall_time_s"] - freeze.seconds
    hours = excluded / 3600
    return {
        "run": run.name,
        "start": freeze.start.isoformat(),
        "end": freeze.end.isoformat(),
        "window_s": freeze.seconds,
        "log": str(log),
        "log_gap": [gap_start.isoformat(), gap_end.isoformat()],
        "log_gap_s": longest.total_seconds(),
        "throughput_spacing_s": spacing,
        "wall_time_s": run.info["wall_time_s"],
        "concurrency": width,
        "monotonic_s": monotonic,
        "base_overhead_s": base_overhead,
        "wall_clock_gain_s": run.info["wall_time_s"] - monotonic - base_overhead,
        "frozen_calls": len(frozen_calls),
        "other_calls": len(others),
        "frozen_latency_s": [frozen_calls[0], frozen_calls[-1]],
        "next_longest_s": others[-1],
        "p95_and_max_frozen": run.info["latency_s"]["p95"] >= frozen_calls[0]
        and run.info["latency_s"]["max"] == frozen_calls[-1],
        "excluded_s": excluded,
        "cross_check_s": (sum(others) + width * statistics.median(others)) / width
        + base_overhead,
        "base_wall_time_s": base.info["wall_time_s"],
        "excluded_cost": billed * hours,
        "excluded_per_1000": billed * hours / run.info["releases"] * 1000,
        "excluded_cost_at_1": hours,
        "excluded_per_1000_at_1": hours / run.info["releases"] * 1000,
    }


def line_time(line: str, year: int) -> datetime | None:
    match = TIMESTAMP.search(line)
    if match is None:
        return None
    month, day, hour, minute, second = map(int, match.groups())
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


# Misses (descriptive)


def common_line(predicted: Decimal, rows: Sequence[InputRow]) -> InputRow | None:
    """The first input row whose label names common shareholders, stockholders,
    shares or stock and that prints predicted, in any table unit."""
    candidates = [
        (row, figure)
        for row in rows
        if COMMON_LINE.search(row.label)
        for figure in row.figures
    ]
    found = grade.match_figure(predicted, NET_INCOME, candidates)
    return found[0] if found else None


def printed_at(answer: Answer, rows: Sequence[InputRow]) -> str:
    """Where the input prints an answer: the first matching row's heading and
    label."""
    candidates = [(row, figure) for row in rows for figure in row.figures]
    found = grade.match_figure(answer.value, answer.field, candidates)
    if found is None:
        return ""
    row = found[0]
    return " › ".join(part for part in (row.heading, row.section, row.label) if part)


def group_misses(
    run: Run,
    base: Run,
    split: Split,
    texts: Mapping[str, str],
) -> list[dict[str, Any]]:
    """step-00025's misses, in groups: (1) net income whose predicted value is
    printed on an input row whose label names common shareholders, stockholders,
    shares or stock, whatever its verdict; (2) a 10-Q release's year-to-date
    prediction equal to its quarter prediction; (3) the rest, by verdict. A miss
    meeting (1) and (2) goes to (2): its value is the quarter's, whichever line
    prints it (TWLO's only bottom line names common stockholders). Labels are read
    from the input (grade.input_rows), not graded.csv's shortened evidence."""
    misses = []
    rows_cache: dict[str, list[InputRow]] = {}
    for key, row in run.graded.items():
        if row["verdict"] == CORRECT:
            continue
        accession, name, period = key
        rows = rows_cache.setdefault(accession, grade.input_rows(texts[accession]))
        predicted = Decimal(row["predicted"]) if row["predicted"] else None
        prediction = run.predictions[accession]
        entry: dict[str, Any] = {
            "ticker": row["ticker"],
            "release_accession": accession,
            "field": name,
            "period_type": period,
            "answer": row["answer"],
            "predicted": row["predicted"],
            "verdict": row["verdict"],
            "base_verdict": base.graded[key]["verdict"],
            "preferred": accession in split.preferred,
        }
        found = (
            common_line(predicted, rows)
            if name == NET_INCOME and predicted is not None
            else None
        )
        if (
            period == YEAR_TO_DATE
            and predicted is not None
            and predicted == getattr(prediction.quarter, name)
        ):
            entry |= {
                "group": QUARTER_AS_YTD,
                "row": printed_at(split.key[key], rows),
            }
        elif found is not None:
            entry |= {"group": COMMON, "row": found.label}
            entry["release_rows"] = [
                kind
                for kind, pattern in (
                    ("preferred", PREFERRED_ROW),
                    ("participating", PARTICIPATING_ROW),
                )
                if any(pattern.search(r.label) for r in rows if r.figures)
            ]
        else:
            entry |= {"group": row["verdict"], "row": row["evidence"]}
        misses.append(entry)
    order = {COMMON: 0, QUARTER_AS_YTD: 1}
    misses.sort(key=lambda m: (order.get(m["group"], 2), m["group"], m["ticker"]))
    return misses


def miss_summary(misses: Sequence[Mapping[str, Any]], broken: int) -> dict[str, Any]:
    common = [m for m in misses if m["group"] == COMMON]
    outside = [m for m in common if not m["preferred"]]
    outside_rows: dict[str, set[str]] = defaultdict(set)
    for m in outside:
        outside_rows[m["ticker"]].update(m["release_rows"])
    return {
        "misses": len(misses),
        "groups": dict(Counter(m["group"] for m in misses)),
        "common_companies": sorted({m["ticker"] for m in common}),
        "common_verdicts": dict(Counter(m["verdict"] for m in common)),
        "common_in_preferred": len(common) - len(outside),
        "common_outside_preferred": len(outside),
        "outside_rows": {t: sorted(r) for t, r in sorted(outside_rows.items())},
        "broken": broken,
        "broken_common": sum(m["base_verdict"] == CORRECT for m in common),
        "quarter_as_ytd_companies": sorted(
            {m["ticker"] for m in misses if m["group"] == QUARTER_AS_YTD}
        ),
    }


# The attempts


def same_figures(a: Prediction, b: Prediction) -> bool:
    return (a.quarter, a.year_to_date) == (b.quarter, b.year_to_date)


def attempt_entry(attempt: Attempt, rerun: Run | None) -> dict[str, Any]:
    run = load_run(attempt.model, attempt.run, graded=False)
    errors = run.api_errors
    calls = {
        (c["release_accession"], c["call"]): c
        for c in run.info["calls"]
        if c.get("status") != 200
    }
    statuses = sorted({c["status"] for c in calls.values()})
    entry: dict[str, Any] = {
        "run": str(attempt.run),
        "folder": attempt.run.name,
        "run_name": run.info["run_name"],
        "status": run.info["status"],
        "started": run.info["started"],
        "finished": run.info["finished"],
        "cause": attempt.cause,
        "http_statuses": statuses,
        "http_attempts": sorted({c["http_attempts"] for c in calls.values()}),
        "api_errors": [
            {"ticker": e["ticker"], "release_accession": e["release_accession"]}
            for e in errors
        ],
        "max_parallel": run.info["max_parallel"],
        "replies": sum(c.get("status") == 200 for c in run.info["calls"]),
        "cost": compare.cost_value(run.info) or 0.0,
    }
    if rerun is not None:
        failed = {e["release_accession"] for e in errors}
        tickers = {c["release_accession"]: c["ticker"] for c in run.info["calls"]}
        retried = {c["release_accession"] for c in run.info["calls"] if c["call"] > 1}
        both = [
            a for a in run.predictions if a not in failed and a in rerun.predictions
        ]
        entry |= {
            "rerun": rerun.name,
            "rerun_env_file": attempt.rerun_env_file,
            "rerun_settings": attempt.rerun_settings,
            "rerun_max_parallel": rerun.info["max_parallel"],
            "same_settings": [
                name
                for name in (*SETTINGS, "reply_format")
                if run.info.get(name) == rerun.info.get(name)
            ],
            "both_answered": len(both),
            "same_figures": sum(
                same_figures(run.predictions[a], rerun.predictions[a]) for a in both
            ),
            "period_end_differs": [
                {
                    "ticker": tickers[a],
                    "release_accession": a,
                    "retried_in_attempt": a in retried,
                    "attempt": run.predictions[a].period_end.isoformat(),
                    "rerun": rerun.predictions[a].period_end.isoformat(),
                }
                for a in both
                if run.predictions[a].period_end != rerun.predictions[a].period_end
            ],
        }
    return entry


# The report


def period_end_warnings(run: Run, split: Split) -> list[str]:
    ends = {r.release_accession: r.period_end for r in split.links}
    return sorted(a for a, p in run.predictions.items() if p.period_end != ends[a])


def release_texts(split: Split, accessions: Iterable[str]) -> dict[str, str]:
    records = release_records(split.releases_dir)
    return {a: build_input(records[a]).text for a in sorted(set(accessions))}


def git_commit_time(commit: str, repo: Path = REPO) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "show", "-s", "--format=%cI", commit],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return datetime.fromisoformat(out).astimezone(UTC).isoformat()


def model_under_test(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    config = json.loads((TRAINING_RUN / "config.json").read_text(encoding="utf-8"))
    settings, selection = config["settings"], config["selection"]
    checkpoints = {
        json.dumps(r["checkpoint"], sort_keys=True) for r in records.values()
    }
    if len(checkpoints) != 1:
        raise ReportError("the test records name different checkpoints")
    checkpoint = next(iter(records.values()))["checkpoint"]
    return {
        "base_model": settings["model"]["name"],
        "revision": settings["model"]["revision"],
        "lora": settings["lora"],
        "steps": config["schedule"]["steps"],
        "examples_per_step": config["schedule"]["examples_per_step"],
        "selected": selection["selected"],
        "labelled": selection["examples"],
        "within_max_tokens": sum(selection["within_max_tokens"].values()),
        "max_tokens": settings["data"]["max_tokens"],
        "checkpoint": checkpoint,
        "server": {split: r["server"] for split, r in records.items()},
        "gpus": {split: r["gpus"] for split, r in records.items()},
    }


def validation_counts(config: Config, splits_dir: Path) -> dict[str, Any]:
    split = load_split(VALIDATION, splits_dir)
    counts = {}
    for model in config.models:
        run = load_run(model.key, model.validation)
        graded = run.graded
        without = [k for k in graded if k[0] not in split.preferred]
        counts[model.key] = {
            "run": model.validation.name,
            "all": [sum(r["verdict"] == CORRECT for r in graded.values()), len(graded)],
            WITHOUT_PREFERRED: [
                sum(graded[k]["verdict"] == CORRECT for k in without),
                len(without),
            ],
            "recorded_usd_per_hour": run.info.get("gpu_hourly_usd"),
        }
    tickers = sorted({split.ticker[a] for a in split.preferred})
    return {
        "counts": counts,
        "preferred_releases": len(split.preferred),
        "preferred_tickers": tickers,
        "billed_usd_per_hour": config.billed[VALIDATION],
        "gpu": config.validation_gpu,
    }


def analyse_split(
    config: Config,
    name: str,
    splits_dir: Path,
    regrade: Regrade,
    stage2_releases: Path,
    companies_csv: Path,
) -> tuple[dict[str, Any], list[str]]:
    split = load_split(name, splits_dir)
    runs = {m.key: load_run(m.key, m.runs[name]) for m in config.models}
    references = {
        m.key: load_run(m.key, m.validation, graded=False) for m in config.models
    }
    checks = []
    for run in runs.values():
        check_run(run, split, RUN_COMMITS[name])
    checks.append(
        f"runs: all {len(runs)} complete with no selection, covering the "
        f"{len(split.kept)} kept releases, no stop and no API error but refusals "
        f"({sum(r.refusals for r in runs.values())} refusals), on the frozen links "
        f"and answers, at commit {RUN_COMMITS[name][:7]}, clean"
    )
    for run in runs.values():
        check_graded(run, split, regrade)
    checks.append(
        f"{GRADED}: current in all {len(runs)} (not stale; predicted values are "
        f"{PREDICTIONS}'s; re-graded in memory, byte for byte)"
    )
    notes = {
        model: check_settings(run, references[model]) for model, run in runs.items()
    }
    checks.append(
        "settings: provider, model, request_parameters, prompt_sha256 and "
        "price_per_million_tokens equal to each model's validation reference"
    )
    graded_count, absent_count = check_same_keys(list(runs.values()))
    checks.append(
        f"keys: the same {graded_count:,} graded and {absent_count} not_in_input "
        f"figures in all {len(runs)} runs"
    )

    table = Table.from_runs(runs)
    companies = sorted({table.ticker[k] for k in table.keys})
    boot = Bootstrap(companies)
    analysis = analyse_accuracy(config, split, table, boot)
    company_rows = read_csv(companies_csv, COMPANY_COLUMNS)
    orders = company_orders(
        split, companies, {r["ticker"]: int(r["cik"]) for r in company_rows}
    )
    analysis["robustness"] = robustness(table, config.under_test, config.base, orders)

    under, base = runs[config.under_test], runs[config.base]
    texts = release_texts(split, (k[0] for k in under.graded))
    misses = group_misses(under, base, split, texts)
    broken = analysis["differences"][config.base]["all"]["second_right"]
    attempts = [
        attempt_entry(a, runs[a.model]) for a in config.attempts if a.split == name
    ]
    # Every call on the split: the reported runs' and the attempts'.
    started = min(
        [r.info["started"] for r in runs.values()] + [a["started"] for a in attempts]
    )
    finished = max(
        [r.info["finished"] for r in runs.values()] + [a["finished"] for a in attempts]
    )
    costs = {
        model: cost_entry(run, config.billed[name] if run.gpu else None)
        for model, run in runs.items()
    }
    result: dict[str, Any] = {
        "folder": str(split.folder),
        "commit": RUN_COMMITS[name],
        "runs_started": started,
        "runs_finished": finished,
        "runs": {
            model: {
                "run": run.name,
                "model": run.info["model"],
                "provider": run.info["provider"],
                "settings": compare.settings_sent(run.info),
                "concurrency": run.info["max_parallel"],
                "releases": run.info["releases"],
                "outcomes": run.info["outcomes"],
                "refusals": run.refusals,
                "reference": references[model].name,
                "setting_notes": notes[model],
                "period_end_warnings": period_end_warnings(run, split),
            }
            for model, run in runs.items()
        },
        "coverage": coverage(
            split,
            table,
            stage2_releases,
            {r["ticker"]: r["sector"] for r in company_rows},
        ),
        "bootstrap_companies": len(companies),
        "left_out": [
            {k: r[k] for k in ("ticker", "release_accession", "prompt_tokens")}
            for r in split.left_out
        ],
        "attempts": attempts,
        **analysis,
        "verdicts": {model: verdict_counts(run) for model, run in runs.items()},
        "unit_slips": {model: unit_slips(run) for model, run in runs.items()},
        "cost": costs,
        "latency": {model: run.info["latency_s"] for model, run in runs.items()},
        "misses": misses,
        "miss_summary": miss_summary(misses, broken),
        "abstention": {model: abstention(run) for model, run in runs.items()},
        "billed_usd_per_hour": config.billed[name],
        "recorded_price_note": config.recorded_price_note.get(name, ""),
        "logs": {kind: str(p) for kind, p in config.logs[name].items()},
        "notes": [
            note_entry(n, split, runs, under) for n in config.notes if n.split == name
        ],
    }
    if config.freeze and config.freeze.split == name:
        result["freeze"] = freeze_entry(
            runs[config.freeze.model],
            base,
            config.freeze,
            config.logs[name]["server"],
            config.billed[name],
        )
    for kind, path in config.logs[name].items():
        if not path.is_file():
            raise ReportError(f"{kind} log {path} not found")
    return result, checks


def note_entry(note: Note, split: Split, runs: Mapping[str, Run], under: Run) -> dict:
    answers = [k for k in split.key if k[0] == note.release]
    absent = [k for k in answers if under.by_key[k]["verdict"] == NOT_IN_INPUT]
    warned = [
        m for m, r in runs.items() if note.release in period_end_warnings(r, split)
    ]
    return {
        "ticker": split.ticker[note.release],
        "release_accession": note.release,
        "note": note.note,
        "answers": len(answers),
        "not_in_input": len(absent),
        "period_end_warnings": warned,
    }


def build_report(
    config: Config,
    *,
    frozen: Frozen | None = None,
    regrade: Regrade = regrade_offline,
    splits_dir: Path = splits.DEFAULT_OUT,
    stage2_releases: Path = STAGE2_DIR / "releases.csv",
    companies_csv: Path = COMPANIES_CSV,
) -> dict[str, Any]:
    frozen = frozen or Frozen()
    checks = {"frozen": check_frozen(frozen)}
    print("no frozen item changed")
    records = {
        split: json.loads(
            (frozen.root / frozen.test_records / f"{split}.json").read_text("utf-8")
        )
        for split in TEST_SPLITS
    }
    report: dict[str, Any] = {
        "protocol": {
            "path": str(PROTOCOL),
            "sha256": PROTOCOL_SHA256,
            "commit": PROTOCOL_COMMIT,
        },
        "config": str(config.path),
        "models": {m.key: m.label for m in config.models},
        "under_test": config.under_test,
        "base": config.base,
        "checks": checks,
        "model_under_test": model_under_test(records),
        "bootstrap": {
            "seed": SEED,
            "draws": DRAWS,
            "percentiles": list(PERCENTILES),
            "details_fixed": config.bootstrap_details_fixed.isoformat(),
        },
        "gpu_host": config.gpu_host,
        "time_test": {
            "first_filed": TIME_TEST_START.isoformat(),
            "last_filed": UNTIL.isoformat(),
            "frozen_commit": TIME_TEST_COMMIT,
            "frozen_at": git_commit_time(TIME_TEST_COMMIT),
        },
        "splits": {},
    }
    for name in TEST_SPLITS:
        print(f"{name}: checking and analysing the runs (re-grading takes a minute)")
        result, split_checks = analyse_split(
            config, name, splits_dir, regrade, stage2_releases, companies_csv
        )
        report["splits"][name] = result
        checks[name] = split_checks
    report["validation"] = validation_counts(config, splits_dir)
    report["validation"]["attempts"] = [
        attempt_entry(a, None) for a in config.attempts if a.split == VALIDATION
    ]
    return report


# Markdown


MINUS = "−"


def num(value: float, places: int = 2, signed: bool = False) -> str:
    text = f"{abs(value):,.{places}f}"
    if float(text.replace(",", "")) == 0:
        return text
    if value < 0:
        return MINUS + text
    return ("+" if signed else "") + text


def bounds(values: Sequence[float], signed: bool = False) -> str:
    return f"[{num(values[0], signed=signed)}, {num(values[1], signed=signed)}]"


def frac(correct: int, graded: int) -> str:
    return f"{correct:,}/{graded:,}"


def acc_cell(entry: Mapping[str, Any], with_interval: bool = True) -> str:
    text = f"{frac(entry['correct'], entry['graded'])} ({num(entry['accuracy'])}%)"
    return f"{text} {bounds(entry['interval'])}" if with_interval else text


def company_count(count: int) -> str:
    return f"{count} {'company' if count == 1 else 'companies'}"


def money(value: float) -> str:
    return f"${value:,.2f}"


def table_md(header: list[str], rows: list[list[str]]) -> list[str]:
    return compare.markdown_table(header, rows)


def utc_time(stamp: str) -> str:
    """HH:MM UTC, to the nearest minute."""
    moment = datetime.fromisoformat(stamp).astimezone(UTC)
    return (moment + timedelta(seconds=30)).strftime("%H:%M")


def utc_clock(stamp: str) -> str:
    """HH:MM:SS UTC, as recorded (not rounded)."""
    return datetime.fromisoformat(stamp).astimezone(UTC).strftime("%H:%M:%S")


def render(report: Mapping[str, Any]) -> str:
    labels = report["models"]
    under, base = report["under_test"], report["base"]
    company, time_test = report["splits"][COMPANY_TEST], report["splits"][TIME_TEST]
    primary = company["differences"][base]["all"]
    validation = report["validation"]
    step_validation = validation["counts"][under]["all"]
    lines = [
        "# Stage 2 test results",
        "",
        (
            "From `uv run python -m secbench.testreport` with "
            f"{report['config']}; the analysis is fixed by {report['protocol']['path']} "
            f"(sha256 {report['protocol']['sha256'][:8]}…, commit "
            f"{report['protocol']['commit'][:7]}). Anything beyond the protocol is "
            "marked descriptive (not pre-registered). Figures are graded figures: "
            "answers printed in the input (not_in_input and no_answer are left out). "
            "Intervals are 95% bootstrap intervals over companies (Bootstrap below)."
        ),
        "",
        "## Primary result",
        "",
        (
            f"On the company test, {labels[under]} minus {labels[base]}: "
            f"**{num(primary['difference'], signed=True)} points "
            f"{bounds(primary['interval'], signed=True)}**, from "
            f"{num(company['accuracy'][base]['all']['accuracy'])}% to "
            f"{num(company['accuracy'][under]['all']['accuracy'])}% of "
            f"{company['coverage']['graded']:,} figures; {primary['discordant']} "
            f"discordant figures ({primary['first_right']} fixed by training, "
            f"{primary['second_right']} broken) from {primary['companies']} companies. "
            "Everything else is descriptive under the protocol."
        ),
        "",
        (
            f"{labels[under]}'s validation score, {num(100 * step_validation[0] / step_validation[1], 1)}% "
            f"({step_validation[0]}/{step_validation[1]}), is {CHOSEN_ON_VALIDATION}."
        ),
        "",
        "## Checks",
        "",
        "Each would stop the report. No frozen item changed:",
        "",
        *(f"- {line}" for line in report["checks"]["frozen"]),
        "",
    ]
    for name in TEST_SPLITS:
        lines.append(f"{SPLIT_TITLES[name]}:")
        lines.append("")
        lines.extend(f"- {line}" for line in report["checks"][name])
        lines.append("")
    lines.extend(render_model(report))
    for name in TEST_SPLITS:
        lines.extend(render_split(report, name))
    lines.extend(render_validation(report))
    lines.extend(render_bootstrap(report))
    lines.extend(render_deviations(report))
    lines.extend(render_notes(report, company, time_test))
    return "\n".join(lines).rstrip() + "\n"


def render_model(report: Mapping[str, Any]) -> list[str]:
    m = report["model_under_test"]
    lora = m["lora"]
    checkpoint = m["checkpoint"]
    gpus = sorted({g.split(",")[0] for gs in m["gpus"].values() for g in gs})
    return [
        "## Model under test",
        "",
        (
            f"- {m['base_model']} at revision {m['revision'][:8]}, with LoRA r "
            f"{lora['r']} (alpha {lora['alpha']}) on all {len(lora['target_modules'])} "
            f"projections ({', '.join(t.removesuffix('_proj') for t in lora['target_modules'])}), "
            "thinking off."
        ),
        (
            f"- {m['steps']} steps of {m['examples_per_step']} examples over "
            f"{m['selected']:,} of the {m['labelled']:,} labelled train releases (drawn "
            f"from the {m['within_max_tokens']:,} under the {m['max_tokens']:,}-token "
            f"cap). The chosen checkpoint, {checkpoint['label']}, had seen "
            f"{checkpoint['examples_seen']} examples; adapter sha256 "
            f"{checkpoint['adapter_sha256'][:8]}…{checkpoint['adapter_sha256'][-4:]}."
        ),
        (
            f"- Served with {', '.join(sorted(set(m['server'].values())))} on one "
            f"{report['gpu_host']} {', '.join(gpus)} per test split, as "
            f"{TEST_CONFIG} sets (temperature 0, max_tokens 1024, 16 requests at a "
            "time); the base model in the same session."
        ),
        "",
    ]


def render_split(report: Mapping[str, Any], name: str) -> list[str]:
    s = report["splits"][name]
    labels = report["models"]
    under, base = report["under_test"], report["base"]
    cov = s["coverage"]
    time_note = (
        f" On the time test {cov['releases_graded']} releases with graded figures come "
        f"from {cov['companies_graded']} companies, so the bootstrap is in effect over "
        "releases (protocol, Uncertainty)."
        if name == TIME_TEST
        else ""
    )
    lines = [
        f"## {SPLIT_TITLES[name]}",
        "",
        (
            f"{cov['kept_releases']} kept releases of {cov['kept_companies']} companies "
            f"({s['folder']}/), runs made on 2026-11-07 from "
            f"{utc_time(s['runs_started'])} to {utc_time(s['runs_finished'])} UTC at "
            f"commit {s['commit'][:7]}."
        ),
        "",
        "### Runs",
        "",
        *table_md(
            [
                "model",
                "run",
                "model ID",
                "provider",
                "settings sent",
                "concurrency",
                "outcomes",
                "validation reference",
            ],
            [
                [
                    labels[m],
                    r["run"],
                    r["model"],
                    r["provider"],
                    r["settings"],
                    str(r["concurrency"]),
                    outcome_text(r),
                    r["reference"],
                ]
                for m, r in s["runs"].items()
            ],
        ),
        "",
    ]
    notes = [(m, n) for m, r in s["runs"].items() for n in r["setting_notes"]]
    if notes:
        lines.append(
            "Settings differing from the validation reference, besides those checked "
            "equal (notes):"
        )
        lines.append("")
        lines.extend(f"- {labels[m]}: {n}" for m, n in notes)
        lines.append("")
    lines.extend(render_coverage(s, name, labels))
    lines.extend(
        [
            "### Accuracy",
            "",
            f"Correct / graded figures (%), 95% interval.{time_note}",
            "",
            *table_md(
                ["model", "all", *FIELD_NAMES],
                [
                    [
                        labels[m],
                        *(
                            acc_cell(a[sub]) if sub in a else "–"
                            for sub in ("all", *FIELD_NAMES)
                        ),
                    ]
                    for m, a in s["accuracy"].items()
                ],
            ),
            "",
            *table_md(
                ["model", *PERIOD_ORDER],
                [
                    [
                        labels[m],
                        *(acc_cell(a[p]) if p in a else "–" for p in PERIOD_ORDER),
                    ]
                    for m, a in s["accuracy"].items()
                ],
            ),
            "",
            "### Differences",
            "",
            (
                f"{labels[under]} minus each reference, in points, with the discordant "
                f"figures (one model correct, the other not) and the companies they "
                f"come from.{time_note}"
                + (
                    " The first row is the primary result."
                    if name == COMPANY_TEST
                    else ""
                )
            ),
            "",
            *difference_table(s, "all", labels, under, base),
            "",
            "### Sensitivity: without the preferred-dividend releases",
            "",
            (
                f"Without the {s['preferred']['releases']} kept releases in "
                f"preferred.csv (whole releases dropped: {s['preferred']['figures']} "
                f"graded figures; {cov['graded'] - s['preferred']['figures']:,} left)."
                f"{time_note}"
            ),
            "",
            *table_md(
                ["model", "accuracy"],
                [
                    [labels[m], acc_cell(a[WITHOUT_PREFERRED])]
                    for m, a in s["accuracy"].items()
                ],
            ),
            "",
            *difference_table(s, WITHOUT_PREFERRED, labels, under, base),
            "",
        ]
    )
    pref = s["preferred"]
    lines.extend(
        [
            (
                "Net income on the preferred-dividend releases alone, counts only, no "
                f"interval: they come from {company_count(len(pref['companies']))} "
                f"({num(100 * pref['draws_without_them'], 1)}% of draws contain none of "
                "them). "
                + "; ".join(
                    f"{labels[m]} {frac(c['correct'], c['graded'])}"
                    for m, c in pref["net_income"].items()
                )
                + "."
            ),
            "",
        ]
    )
    lines.extend(render_verdicts(s, labels))
    lines.extend(render_costs(s, labels, name))
    lines.extend(render_misses(s, labels, under, base))
    lines.extend(render_abstention(s, labels))
    return lines


def outcome_text(run: Mapping[str, Any]) -> str:
    o = run["outcomes"]
    parts = [f"{o['ok']} ok"]
    if o["ok_after_retry"]:
        parts.append(f"{o['ok_after_retry']} ok after a format retry")
    if o["invalid_reply"]:
        parts.append(f"{o['invalid_reply']} invalid")
    if run["refusals"]:
        parts.append(f"{run['refusals']} refusals (graded missing)")
    return ", ".join(parts)


def difference_table(
    s: Mapping[str, Any], subset: str, labels: Mapping[str, str], under: str, base: str
) -> list[str]:
    rows = []
    for other, entry in s["differences"].items():
        d = entry[subset]
        if other == base:
            right = f"{d['first_right']} fixed, {d['second_right']} broken"
        else:
            right = (
                f"{d['first_right']} {labels[under]} right, {d['second_right']} "
                f"{labels[other]} right"
            )
        rows.append(
            [
                f"minus {labels[other]}",
                num(d["difference"], signed=True),
                bounds(d["interval"], signed=True),
                str(d["discordant"]),
                right,
                str(d["companies"]),
            ]
        )
    return table_md(
        ["difference", "points", "95% interval", "discordant", "of which", "companies"],
        rows,
    )


def render_coverage(
    s: Mapping[str, Any], name: str, labels: Mapping[str, str]
) -> list[str]:
    cov = s["coverage"]
    sectors = [
        f"{sector} {v['no_answer']} of {v['kept']}"
        for sector, v in cov["no_answer_by_sector"].items()
        if v["no_answer"]
    ]
    no_answer = sum(v["no_answer"] for v in cov["no_answer_by_sector"].values())
    lines = [
        "### Coverage",
        "",
        *table_md(
            ["answer key over the kept releases", "figures"],
            [
                [
                    "expected (answered + missing, data/stage2/releases.csv)",
                    f"{cov['expected']:,}",
                ],
                ["answered (an XBRL answer)", f"{cov['answered']:,}"],
                ["missing (no XBRL answer)", f"{cov['missing']:,}"],
                ["in the input", f"{cov['in_input']:,}"],
                ["graded", f"{cov['graded']:,}"],
            ],
        ),
        "",
        (
            f"Releases with no answer at all: {no_answer}"
            + (
                f" (by sector, of the kept releases: {'; '.join(sectors)})"
                if sectors
                else ""
            )
            + "."
        ),
        "",
        (
            f"Releases with at least one graded figure: {cov['releases_graded']} of "
            f"{cov['kept_releases']} kept; companies: {cov['companies_graded']} of "
            f"{cov['kept_companies']}. Kept releases without one: "
            + "; ".join(
                f"{r['ticker']} {r['release_accession']} "
                + (
                    "(no answer)"
                    if not r["answers"]
                    else f"(none of its {r['answers']} answers in the input)"
                )
                for r in cov["releases_without_graded"]
            )
            + "."
        ),
        "",
    ]
    if s["left_out"]:
        lines.append(
            "Left out by the context limit (prompt over 39,936 tokens; left_out.csv), "
            "for every model: "
            + "; ".join(
                f"{r['ticker']} {r['release_accession']} ({int(r['prompt_tokens']):,} tokens)"
                for r in s["left_out"]
            )
            + "."
        )
    else:
        lines.append("Left out by the context limit: none.")
    lines.append("")
    if s["attempts"]:
        for a in s["attempts"]:
            lines.extend(attempt_lines(a, labels))
    else:
        lines.append("Attempts (runs moved aside and rerun): none.")
    lines.append("")
    return lines


def attempt_lines(a: Mapping[str, Any], labels: Mapping[str, str]) -> list[str]:
    errors = ", ".join(
        f"{e['ticker']} {e['release_accession']}" for e in a["api_errors"]
    )
    statuses = ", ".join(f"HTTP {s}" for s in a["http_statuses"])
    attempts = ", ".join(str(n) for n in a["http_attempts"])
    lines = [
        (
            f"Attempt {a['run']} (run.json still says run_name {a['run_name']}; only "
            f"its folder was renamed): {len(a['api_errors'])} API errors at "
            f"{a['max_parallel']} calls at a time ({errors}), each {statuses} after "
            f"{attempts} attempts, on {a['cause']}. As the protocol says (Runs), it "
            f"was moved aside and rerun in full once: {a['rerun']}, with "
            f"{a['rerun_env_file']}, {a['rerun_settings']} (the same "
            f"{', '.join(a['same_settings'])}; max_parallel {a['rerun_max_parallel']}). "
            f"On the {a['both_answered']} releases both runs answered, "
            + (
                "every predicted figure is identical"
                if a["same_figures"] == a["both_answered"]
                else f"{a['same_figures']} releases' figures are identical"
            )
            + period_end_text(a["period_end_differs"])
            + f". The attempt cost ${a['cost']:.4f}, listed here and left out of the "
            "per-release cost."
        ),
    ]
    return lines


def period_end_text(differs: Sequence[Mapping[str, Any]]) -> str:
    if not differs:
        return ", period_end included"
    return (
        f"; {len(differs)} period_end differs ("
        + "; ".join(
            f"{d['ticker']} {d['release_accession']}: {d['attempt']} in the attempt"
            + (", after a format retry," if d["retried_in_attempt"] else "")
            + f" and {d['rerun']} in the rerun"
            for d in differs
        )
        + ")"
    )


def render_verdicts(s: Mapping[str, Any], labels: Mapping[str, str]) -> list[str]:
    shown = [v for v in VERDICTS if any(c[v] for c in s["verdicts"].values())]
    return [
        "### Verdicts and unit slips",
        "",
        *table_md(
            ["verdict", *(labels[m] for m in s["verdicts"])],
            [[v, *(f"{c[v]:,}" for c in s["verdicts"].values())] for v in shown]
            + [["unit slips", *(f"{n:,}" for n in s["unit_slips"].values())]],
        ),
        "",
        (
            "Unit slips: rows matched with a power-of-ten factor (graded.csv's factor "
            "column). not_in_input and no_answer rows aren't graded."
        ),
        "",
    ]


def render_costs(
    s: Mapping[str, Any], labels: Mapping[str, str], name: str
) -> list[str]:
    rows = []
    freeze = s.get("freeze")
    for m, c in s["cost"].items():
        lat = s["latency"][m]
        p95 = f"{lat['p95']:.2f} s"
        if freeze and freeze["run"] == s["runs"][m]["run"]:
            p95 += " (freeze)"
        if c["basis"] == "gpu":
            basis = f"${c['billed_usd_per_hour']:g}/h × {c['wall_time_s']:.1f} s"
            at_1 = f"{money(c['per_1000_at_1'])}"
        else:
            p = c["prices"]
            basis = f"tokens (${p['input']:g} in, ${p['output']:g} out per M)"
            at_1 = "–"
        rows.append(
            [
                labels[m],
                basis,
                f"${c['cost']:.4f}",
                money(c["per_1000"]),
                at_1,
                f"{lat['p50']:.2f} s",
                p95,
            ]
        )
    lines = [
        "### Cost and latency",
        "",
        *table_md(
            [
                "model",
                "cost basis",
                "cost",
                "per 1,000 releases",
                "per 1,000 at $1/h",
                "latency p50",
                "latency p95",
            ],
            rows,
        ),
        "",
        (
            f"GPU runs at the price billed (${s['billed_usd_per_hour']:g}/h) times the "
            "run's wall time (first call to last), and at $1/h as results/comparison.md "
            "shows GPU runs; they leave out each session's setup, model loading and "
            "idle time. API runs at their token prices (run.json's estimate). Latency "
            "is per call, reported but not compared across providers (16 calls at a "
            "time on the GPU, 4 or fewer on the APIs)."
        ),
        "",
    ]
    if s["recorded_price_note"]:
        gpu_runs = [m for m, c in s["cost"].items() if c["basis"] == "gpu"]
        recorded = {s["cost"][m]["recorded_usd_per_hour"] for m in gpu_runs}
        lines.extend(
            [
                (
                    f"Price: both GPU runs record gpu_hourly_usd "
                    f"{', '.join(f'{r:g}' for r in sorted(recorded))}, "
                    f"{s['recorded_price_note']}; the instance was billed "
                    f"${s['billed_usd_per_hour']:g}/h (Deviations, 2). At the recorded "
                    "price, the protocol's letter: "
                    + "; ".join(
                        f"{labels[m]} ${s['cost'][m]['cost_at_recorded']:.4f}, "
                        f"{money(s['cost'][m]['per_1000_at_recorded'])} per 1,000 releases"
                        for m in gpu_runs
                    )
                    + "."
                ),
                "",
            ]
        )
    if freeze:
        lines.extend(freeze_lines(freeze, s))
    return lines


def freeze_lines(f: Mapping[str, Any], s: Mapping[str, Any]) -> list[str]:
    return [
        (
            f"Instance freeze (descriptive): {f['run']} took {f['wall_time_s']:.1f} s, "
            f"but the whole instance froze from about {utc_clock(f['start'])} to "
            f"{utc_clock(f['end'])} UTC "
            f"({f['window_s']:.0f} s). {f['log']} has no timestamped line between "
            f"{utc_clock(f['log_gap'][0])} and {utc_clock(f['log_gap'][1])}, while vLLM "
            f"otherwise logs its throughput every {f['throughput_spacing_s']:.0f} s; "
            f"the {f['frozen_calls']} requests in flight each took "
            f"{f['frozen_latency_s'][0]:.0f}-{f['frozen_latency_s'][1]:.0f} s (the next "
            f"longest {f['next_longest_s']:.1f} s), and the run's p95 and max latency "
            "are among them. As in training's 6 slow steps (PLAN.md Progress "
            "2026-11-02), the wall clock ran ahead of the process's monotonic clock: "
            f"the run took about {f['monotonic_s']:.1f} s by its own clock (the sum of "
            f"call latencies / {f['concurrency']}), so the wall clock gained about "
            f"{f['wall_clock_gain_s']:.0f} s (after the base run's usual "
            f"{f['base_overhead_s']:.1f} s of wall time beyond its latencies / "
            f"{f['concurrency']})."
        ),
        "",
        (
            f"Freeze-excluded estimate: {f['wall_time_s']:.1f} − {f['window_s']:.0f} = "
            f"{f['excluded_s']:.1f} s, about {money(f['excluded_per_1000'])} per 1,000 "
            f"releases at ${s['billed_usd_per_hour']:g}/h "
            f"({money(f['excluded_per_1000_at_1'])} at $1/h). Cross-check: (the other "
            f"{f['other_calls']} latencies + {f['frozen_calls']} × their median) / "
            f"{f['concurrency']} + "
            f"{f['base_overhead_s']:.1f} s = {f['cross_check_s']:.1f} s. The base run "
            f"took {f['base_wall_time_s']:.1f} s."
        ),
        "",
    ]


def render_misses(
    s: Mapping[str, Any], labels: Mapping[str, str], under: str, base: str
) -> list[str]:
    summary = s["miss_summary"]
    groups = summary["groups"]
    common = groups.get(COMMON, 0)
    copied = groups.get(QUARTER_AS_YTD, 0)
    rest = {g: n for g, n in groups.items() if g not in (COMMON, QUARTER_AS_YTD)}
    rule = (
        "The rule: (1) net income whose predicted value is printed on an input row "
        "whose label names common shareholders, stockholders, shares or stock, "
        "whatever its verdict (full labels read from the input, not graded.csv's "
        "shortened evidence; table rows and prose lines); (2) a 10-Q release's "
        "year-to-date prediction equal to its quarter prediction; (3) the rest, by "
        "verdict. A miss meeting (1) and (2) goes to (2), since its value is the "
        "quarter's whichever line prints it."
    )
    lines = [
        f"### {labels[under]}'s misses (descriptive, not pre-registered)",
        "",
        rule,
        "",
        (
            f"{summary['misses']} misses: (1) {common}, (2) {copied}, (3) "
            + (", ".join(f"{g} {n}" for g, n in sorted(rest.items())) or "none")
            + "."
        ),
        "",
    ]
    if common:
        verdicts = ", ".join(
            f"{v} {n}" for v, n in sorted(summary["common_verdicts"].items())
        )
        outside = summary["outside_rows"]
        preferred_named = [t for t, rows in outside.items() if "preferred" in rows]
        participating = [
            t
            for t, rows in outside.items()
            if "participating" in rows and t not in preferred_named
        ]
        neither = [t for t, rows in outside.items() if not rows]
        parts = [
            (
                f"Group (1): {common} of the {summary['misses']} misses, from "
                f"{company_count(len(summary['common_companies']))} "
                f"({', '.join(summary['common_companies'])}); verdicts {verdicts}."
            )
        ]
        parts.append(
            f" {summary['common_in_preferred']} are in preferred.csv releases and "
            f"{summary['common_outside_preferred']} outside it"
        )
        if summary["common_outside_preferred"]:
            why = []
            if preferred_named:
                why.append(
                    f"{', '.join(preferred_named)} print preferred-stock lines (a "
                    "row naming preferred) that the XBRL preferred-dividend check "
                    "missed"
                )
            if participating:
                why.append(
                    f"{', '.join(participating)} report income after participating "
                    "securities (a row naming them)"
                )
            if neither:
                why.append(f"{', '.join(neither)} print neither")
            parts.append(": " + "; ".join(why))
        parts.append(
            f". Group (1) includes {summary['broken_common']} of the "
            f"{summary['broken']} figures broken by training (the base model's correct). "
            "The sensitivity analysis drops only preferred.csv's releases, so it "
            "understates this failure."
        )
        lines.extend(["".join(parts), ""])
    if copied:
        lines.extend(
            [
                (
                    f"Group (2): {copied} from {', '.join(summary['quarter_as_ytd_companies'])}: "
                    "the quarter copied into year to date in second- or third-quarter "
                    "releases whose income statement has quarter columns only, with the "
                    "year to date printed elsewhere (the row where the input prints the "
                    "year-to-date answer is listed)."
                ),
                "",
            ]
        )
    lines.extend(
        [
            *table_md(
                [
                    "group",
                    "ticker",
                    "release",
                    "field",
                    "period",
                    "answer",
                    "predicted",
                    "verdict",
                    f"{labels[base]} verdict",
                    "row",
                ],
                [
                    [
                        m["group"] + (" (preferred.csv)" if m["preferred"] else ""),
                        m["ticker"],
                        m["release_accession"],
                        m["field"],
                        m["period_type"],
                        m["answer"],
                        m["predicted"] or "null",
                        m["verdict"],
                        m["base_verdict"],
                        shorten(m["row"], 90) if m["row"] else "",
                    ]
                    for m in s["misses"]
                ],
            ),
            "",
        ]
    )
    return lines


def render_abstention(s: Mapping[str, Any], labels: Mapping[str, str]) -> list[str]:
    rows = [
        [
            labels[m],
            f"{a['answered']} of {a['not_in_input']}",
            f"{a['null']} of {a['figures']:,}",
        ]
        for m, a in s["abstention"].items()
    ]
    return [
        "### Abstention (descriptive, not pre-registered)",
        "",
        *table_md(
            [
                "model",
                "not_in_input figures answered anyway",
                "null figures of all predicted",
            ],
            rows,
        ),
        "",
        (
            "A not_in_input figure has an XBRL answer the release doesn't print, so "
            "the prompt's \"null when the release doesn't report a figure\" applies. "
            "Accuracy leaves these figures out, so it doesn't reward abstaining. "
            "Training had no null labels (PLAN.md, Limitation: no nulls in training)."
        ),
        "",
    ]


def render_validation(report: Mapping[str, Any]) -> list[str]:
    v = report["validation"]
    labels = report["models"]
    under = report["under_test"]
    rows = []
    for m, c in v["counts"].items():
        note = f", {CHOSEN_ON_VALIDATION}" if m == under else ""
        rows.append(
            [
                labels[m],
                c["run"],
                frac(*c["all"]) + note,
                frac(*c[WITHOUT_PREFERRED]),
            ]
        )
    gpu_recorded = sorted(
        {
            c["recorded_usd_per_hour"]
            for c in v["counts"].values()
            if c["recorded_usd_per_hour"]
        }
    )
    lines = [
        "## Validation, for reference",
        "",
        (
            "Counts only, no intervals: the 60 validation releases' graded figures, "
            f"then without {', '.join(v['preferred_tickers'])}'s "
            f"{v['preferred_releases']} preferred-dividend releases."
        ),
        "",
        *table_md(["model", "run", "all", "without preferred"], rows),
        "",
        (
            f"The qwen3-8b-2k validation runs record "
            f"{', '.join(f'${p:g}/h' for p in gpu_recorded)}; they ran on a "
            f"{report['gpu_host']} {v['gpu']} billed ${v['billed_usd_per_hour']:g}/h "
            "(PLAN.md Progress 2026-11-02), not on stage 1's A100."
        ),
    ]
    for a in v["attempts"]:
        lines.extend(
            [
                "",
                (
                    f"GPT-6.1 Sol's first validation attempt, {a['folder']}, stopped "
                    f"({a['status']}) on {', '.join(f'HTTP {s}' for s in a['http_statuses'])} "
                    f"({a['cause']}) with {a['replies']} replies; it was not a test call. "
                    "The validation reference is the run after it."
                ),
            ]
        )
    lines.append("")
    return lines


def render_bootstrap(report: Mapping[str, Any]) -> list[str]:
    b = report["bootstrap"]
    lines = [
        "## Bootstrap",
        "",
        (
            f"As the protocol states (Uncertainty): {b['draws']:,} draws of companies "
            f"with replacement from numpy's default_rng({b['seed']}), one set per split "
            "shared by every model, subset and difference; a difference is computed "
            f"within each draw. These details were chosen on {b['details_fixed']}, "
            "after the test runs were graded (Deviations, 4): the companies are the "
            "split's companies with at least one graded figure, sorted by ticker; "
            "draws = rng.integers(0, n, size=(10000, n)); each draw's accuracy = "
            "Σ(times drawn × correct) / Σ(times drawn × graded); intervals = "
            "np.percentile(values, [2.5, 97.5]). In a subset, a company with no "
            "figures left adds 0 to both sums; a draw with no graded figure would stop "
            "the report (none did), never be dropped."
        ),
        "",
    ]
    rob = []
    moves = []
    for name in TEST_SPLITS:
        r = report["splits"][name]["robustness"]
        others = {k: v for k, v in r.items() if k != "ticker"}
        move = max(abs(v[i] - r["ticker"][i]) for v in others.values() for i in (0, 1))
        moves.append(move)
        if len({tuple(v) for v in others.values()}) == 1:
            text = (
                f"{SPLIT_TITLES[name].lower()}: in links.csv order or by CIK "
                f"{bounds(next(iter(others.values())), signed=True)}"
            )
        else:
            text = (
                f"{SPLIT_TITLES[name].lower()}: in links.csv order "
                f"{bounds(r['links_order'], signed=True)}, by CIK "
                f"{bounds(r['cik'], signed=True)}"
            )
        rob.append(f"{text} (by ticker {bounds(r['ticker'], signed=True)})")
    lines.extend(
        [
            (
                "Robustness (descriptive): the primary difference's interval with the "
                f"companies in other orders, {'; '.join(rob)}. The bounds move by at "
                f"most {num(max(moves))} points."
            ),
            "",
        ]
    )
    return lines


def render_deviations(report: Mapping[str, Any]) -> list[str]:
    company, time_test = report["splits"][COMPANY_TEST], report["splits"][TIME_TEST]
    t = report["time_test"]
    attempt = time_test["attempts"][0] if time_test["attempts"] else None
    recorded = sorted(
        {
            c["recorded_usd_per_hour"]
            for c in company["cost"].values()
            if c["basis"] == "gpu"
        }
    )
    lines = [
        "## Deviations from the protocol",
        "",
        (
            "No amendment was made (none are allowed after the first test call); these are "
            "recorded here instead."
        ),
        "",
        (
            "1. The time test. The protocol set its cutoff for when the Q3 2026 releases "
            "are added, and said the provisional 346-release time test would not be "
            f"scored. Decision 2026-11-07 (commit {t['frozen_commit'][:7]}) froze that "
            f"same set (releases filed {t['first_filed']} to {t['last_filed']}) at "
            f"{utc_time(t['frozen_at'])} UTC on 2026-11-07: after the company-test runs "
            f"({utc_time(company['runs_started'])}-{utc_time(company['runs_finished'])} "
            "UTC) and before any time-test call "
            f"({utc_time(time_test['runs_started'])} UTC). No amendment records it, and "
            "none were allowed. Every time-test result was descriptive under the "
            "protocol anyway."
        ),
        (
            "2. The company test's GPU price. The protocol prices GPU runs at the $/h "
            "given to vast_setup.sh (run.json's gpu_hourly_usd). Both company-test GPU "
            f"runs record {', '.join(f'{r:g}' for r in recorded)}, "
            f"{company['recorded_price_note']}, which is the price by the protocol's "
            f"letter; the instance was billed ${company['billed_usd_per_hour']:g}/h, "
            "which the costs here use (step-00025's cost at the recorded price is also "
            "given). The time test's price, "
            f"${time_test['billed_usd_per_hour']:g}/h, was recorded correctly."
        ),
    ]
    if attempt:
        lines.append(
            "3. GPT-6.1 Sol's time-test rerun ran "
            f"{attempt['rerun_max_parallel']} calls at a time, not the protocol's 4 "
            f"(Cost and latency): its first attempt hit {attempt['cause']} at "
            f"{attempt['max_parallel']} (Time test, Coverage)."
        )
    lines.extend(
        [
            (
                f"4. The bootstrap's details were fixed on "
                f"{report['bootstrap']['details_fixed']}, after the test runs were "
                "graded (Bootstrap, with its robustness line)."
            ),
            "",
        ]
    )
    return lines


def render_notes(
    report: Mapping[str, Any], company: Mapping[str, Any], time_test: Mapping[str, Any]
) -> list[str]:
    labels = report["models"]
    lines = ["## Notes", ""]
    for s in (company, time_test):
        kind = "company-test" if s is company else "time-test"
        for n in s["notes"]:
            warned = n["period_end_warnings"]
            others = [labels[m] for m in s["runs"] if m not in warned]
            printed = n["answers"] - n["not_in_input"]
            lines.append(
                f"- {n['ticker']} {n['release_accession']}, a frozen {kind} release, "
                f"is {n['note']}; "
                + (
                    f"none of its {n['answers']} answers is in its input"
                    if not printed
                    else f"{printed} of its {n['answers']} answers are in its input"
                )
                + f". Its period_end warning appears in {len(warned)} of the "
                f"{len(s['runs'])} runs"
                + (f", not {', '.join(others)}'s" if others else "")
                + "."
            )
    lines.extend(
        [
            (
                "- Logs: "
                + "; ".join(
                    f"{SPLIT_TITLES[name].lower()} {s['logs']['server']} and "
                    f"{s['logs']['setup']}"
                    for name, s in report["splits"].items()
                )
                + "."
            ),
            "",
        ]
    )
    return lines


# Output


def to_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_json(v) for v in value]
    if isinstance(value, Decimal | Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.testreport",
        description="Write the stage 2 test report (results/test_results.md and .json) "
        "from the reported runs, as results/test_protocol.md fixes it.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        report = to_json(build_report(config))
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_text(args.out_dir / RESULTS_JSON, json.dumps(report, indent=2) + "\n")
    text = render(report)
    write_text(args.out_dir / RESULTS_MD, text)
    print(text, end="")
    print(f"wrote {args.out_dir / RESULTS_MD} and {args.out_dir / RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
