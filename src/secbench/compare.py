"""Compare evaluation runs: one table over every run in runs/.

For each run folder with a run.json (see secbench.extract): the model, provider,
prompt version and request parameters sent, accuracy overall and by field, counts
by verdict and unit slips (errors matched with a power-of-ten factor, scale and zeros
included) from the run's saved graded.csv (see secbench.grade), format retries,
format nulls (prompts with units: figures still bad on the retry, made null), the
estimated cost and per-call latency. Graded runs that cover every linked release in
links.csv go in the main table, with cost per 1,000 releases; runs on fewer releases,
stopped or not graded go in a smaller table below it, so a short run's cost per
release doesn't sit beside full runs. A run whose graded.csv doesn't match the
current answers.csv, or has other columns than the grader writes (graded by an older
grader), is marked stale (re-grade it with secbench.grade). Nothing from .env is
shown: settings come from run.json's request_parameters.

Usage: uv run python -m secbench.compare [--runs-dir runs]
       [--out results/comparison.md] [--links data/links.csv] [--answers data/answers.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from secbench.answers import DEFAULT_OUT as DEFAULT_ANSWERS
from secbench.answers import FIELDS, Answer, load_links, plural
from secbench.extract import (
    ABORTED,
    INVALID_REPLY,
    OK_AFTER_RETRY,
    RUN_INFO,
    RUNS_DIR,
    TOKENS,
    WITH_UNITS,
    describe_parameters,
)
from secbench.grade import (
    CORRECT,
    GRADED,
    GRADED_COLUMNS,
    UNGRADED,
    VERDICTS,
    GradeError,
    load_answers,
)
from secbench.link import DEFAULT_OUT as DEFAULT_LINKS
from secbench.releases import write_text

DEFAULT_OUT = Path("results/comparison.md")
FIELD_NAMES = [field.name for field in FIELDS]

GRADED_STATUS = "graded"
STALE = "stale: re-grade"
NOT_GRADED = "not graded"
STOPPED = "stopped, not graded"


@dataclass
class Run:
    name: str
    info: dict[str, Any]  # run.json
    graded: list[dict[str, str]] | None  # graded.csv rows; None if not graded
    columns: list[str]  # graded.csv's header
    stale: bool = False

    @property
    def covered(self) -> set[str]:
        """The releases the run called the model for."""
        return {call["release_accession"] for call in self.info.get("calls", [])}

    @property
    def status(self) -> str:
        if self.info.get("status") == ABORTED:
            return STOPPED
        if self.graded is None:
            return NOT_GRADED
        return STALE if self.stale else GRADED_STATUS

    def verdicts(self, field: str | None = None) -> list[str]:
        return [
            row["verdict"]
            for row in self.graded or []
            if field is None or row["field"] == field
        ]

    def unit_slips(self) -> str:
        """Figures whose verdict was matched with a power-of-ten factor; none if not
        graded, or graded before graded.csv had a factor column."""
        if self.graded is None or "factor" not in self.columns:
            return "none"
        return str(sum(bool(row["factor"]) for row in self.graded))


def is_stale(
    rows: list[dict[str, str]],
    covered: set[str],
    key: dict[tuple[str, str, str], Answer],
) -> bool:
    """Whether graded.csv's answers differ from answers.csv's for the run's releases
    (answers added, removed or changed since it was graded)."""
    expected = {k: a.value for k, a in key.items() if a.release_accession in covered}
    got = {
        (row["release_accession"], row["field"], row["period_type"]): Decimal(
            row["answer"]
        )
        for row in rows
        if row["answer"]
    }
    return got != expected


def load_runs(
    runs_dir: Path, key: dict[tuple[str, str, str], Answer]
) -> tuple[list[Run], list[str]]:
    """Every run folder with a readable run.json, by start time; and the folders
    left out, with why."""
    runs, skipped = [], []
    for folder in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        info_path = folder / RUN_INFO
        if not info_path.is_file():
            skipped.append(f"{folder.name} (no {RUN_INFO})")
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped.append(f"{folder.name} ({RUN_INFO} is not valid JSON)")
            continue
        graded = None
        columns: list[str] = []
        if (folder / GRADED).is_file():
            with (folder / GRADED).open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                graded = list(reader)
                columns = list(reader.fieldnames or [])
        run = Run(folder.name, info, graded, columns)
        if graded is not None:
            run.stale = columns != GRADED_COLUMNS or is_stale(graded, run.covered, key)
        runs.append(run)
    runs.sort(key=lambda r: (r.info.get("started", ""), r.name))
    return runs, skipped


# Cells


def accuracy(verdicts: list[str]) -> str:
    """As grade.accuracy: correct over figures graded (not_in_input and no_answer
    left out)."""
    counted = [v for v in verdicts if v not in UNGRADED]
    if not counted:
        return "0/0"
    correct = counted.count(CORRECT)
    return f"{correct}/{len(counted)} ({100 * correct / len(counted):.1f}%)"


def format_retries(info: dict[str, Any]) -> str:
    """Releases whose first reply didn't validate, and how many of those were still
    invalid after the retry."""
    outcomes = info.get("outcomes", {})
    invalid = outcomes.get(INVALID_REPLY, 0)
    retried = outcomes.get(OK_AFTER_RETRY, 0) + invalid
    return f"{retried} ({invalid} invalid)" if invalid else str(retried)


def format_nulls(info: dict[str, Any]) -> str:
    """Figures still malformed or with an unknown unit on the retry, made null; "–"
    for a prompt without units (extract_v1, and runs from before reply_format)."""
    if info.get("reply_format") != WITH_UNITS:
        return "–"
    return str(len(info["format_nulls"]))


def cost_basis(info: dict[str, Any]) -> str | None:
    """run.json's cost_basis; runs from before it was recorded priced tokens."""
    if "cost_basis" in info:
        return info["cost_basis"]
    return TOKENS if info.get("estimated_cost_usd") is not None else None


def money(value: float, places: int) -> str:
    return f"${value:,.{places}f}"


def describe_basis(info: dict[str, Any]) -> str:
    basis = cost_basis(info)
    if basis is None:
        return "none"
    if basis == TOKENS:
        prices = info["price_per_million_tokens"]
        return f"tokens (${prices['input']:g} in, ${prices['output']:g} out per M)"
    return f"GPU (${info['gpu_hourly_usd']:g}/h x {info['wall_time_s']:.1f} s)"


def cost(info: dict[str, Any]) -> str:
    value = info.get("estimated_cost_usd")
    return "none" if value is None else money(value, 4)


def cost_per_thousand(info: dict[str, Any]) -> str:
    value = info.get("estimated_cost_usd")
    if value is None or not info.get("releases"):
        return "none"
    return money(value / info["releases"] * 1000, 2)


def latency(info: dict[str, Any], percentile: str) -> str:
    value = info.get("latency_s", {}).get(percentile)
    return "none" if value is None else f"{value:.2f} s"


def settings_sent(info: dict[str, Any]) -> str:
    return describe_parameters(info["provider"], info.get("request_parameters", {}))


# Tables


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", r"\|") for c in cells) + " |"

    return [line(header), "|" + "|".join(" --- " for _ in header) + "|"] + [
        line(row) for row in rows
    ]


def main_table(runs: list[Run]) -> tuple[list[str], list[str]]:
    """The full runs' table, and the verdicts that are 0 in every one of them (not
    given a column)."""
    counts = {run.name: Counter(run.verdicts()) for run in runs}
    shown = [v for v in VERDICTS if any(c[v] for c in counts.values())]
    header = [
        "run",
        "model",
        "provider",
        "prompt",
        "settings sent",
        "concurrency",
        "releases",
        "status",
        "accuracy",
        *FIELD_NAMES,
        *shown,
        "unit slips",
        "format retries",
        "format nulls",
        "cost",
        "cost basis",
        "cost / 1,000 releases",
        "latency p50",
        "latency p95",
    ]
    rows = []
    for run in runs:
        info = run.info
        rows.append(
            [
                run.name,
                info["model"],
                info["provider"],
                info["prompt_version"],
                settings_sent(info),
                str(info.get("max_parallel", "")),
                str(info["releases"]),
                run.status,
                accuracy(run.verdicts()),
                *(accuracy(run.verdicts(name)) for name in FIELD_NAMES),
                *(str(counts[run.name][v]) for v in shown),
                run.unit_slips(),
                format_retries(info),
                format_nulls(info),
                cost(info),
                describe_basis(info),
                cost_per_thousand(info),
                latency(info, "p50"),
                latency(info, "p95"),
            ]
        )
    return markdown_table(header, rows), [v for v in VERDICTS if v not in shown]


def other_table(runs: list[Run]) -> list[str]:
    header = [
        "run",
        "model",
        "provider",
        "prompt",
        "settings sent",
        "releases",
        "status",
        "accuracy",
        "unit slips",
        "format retries",
        "format nulls",
        "cost",
        "cost basis",
        "latency p50",
        "latency p95",
    ]
    rows = [
        [
            run.name,
            run.info["model"],
            run.info["provider"],
            run.info["prompt_version"],
            settings_sent(run.info),
            str(run.info["releases"]),
            run.status,
            accuracy(run.verdicts()) if run.graded is not None else "none",
            run.unit_slips(),
            format_retries(run.info),
            format_nulls(run.info),
            cost(run.info),
            describe_basis(run.info),
            latency(run.info, "p50"),
            latency(run.info, "p95"),
        ]
        for run in runs
    ]
    return markdown_table(header, rows)


def format_comparison(
    runs: list[Run],
    skipped: list[str],
    linked: set[str],
    *,
    runs_dir: Path,
    links_path: Path,
    answers_path: Path,
) -> str:
    full = [
        r
        for r in runs
        if r.info.get("status") != ABORTED
        and r.graded is not None
        and linked <= r.covered
    ]
    others = [r for r in runs if r not in full]
    lines = [
        "# Model comparison",
        "",
        (
            f"From `uv run python -m secbench.compare`: {plural(len(runs), 'run')} in "
            f"{runs_dir}/, graded against {answers_path}; {links_path} has "
            f"{plural(len(linked), 'linked release')}."
        ),
        "",
        f"## Runs on all {len(linked)} linked releases",
        "",
    ]
    zero: list[str] = []
    if full:
        table, zero = main_table(full)
        lines.extend(table)
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Other runs: fewer releases, stopped or not graded",
            "",
            *(other_table(others) if others else ["None."]),
            "",
            "Notes:",
            (
                "- Accuracy is correct figures over figures graded; not_in_input and "
                "no_answer are left out, as in each run's summary.txt."
            ),
        ]
    )
    if zero:
        lines.append(
            f"- Verdicts 0 in every run in the first table: {', '.join(zero)}."
        )
    lines.extend(
        [
            (
                "- Unit slips: errors matched with a power-of-ten factor (graded.csv's "
                "factor column): scale and zeros, and the other errors matched only "
                "after multiplying or dividing the prediction by 10 to 1,000,000."
            ),
            (
                "- Format retries: releases whose first reply didn't validate and "
                'were asked again; "(n invalid)" were still invalid and graded as '
                "missing."
            ),
            (
                "- Format nulls (prompts with units, such as extract_v2): revenue and "
                "net income figures still malformed or with an unknown unit on the "
                "retry, made null and graded as missing; – for prompts without units."
            ),
            (
                "- Cost is run.json's estimate: tokens at the run's prices per "
                "million, or GPU dollars per hour times the run's wall-clock time. "
                "Latency is per call."
            ),
            (
                f"- {STALE}: graded.csv doesn't match {answers_path}, or has other "
                "columns than the grader writes now; run `uv run python -m "
                "secbench.grade runs/<run>/`."
            ),
        ]
    )
    if skipped:
        lines.append(f"- Not listed: {', '.join(skipped)}.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.compare",
        description="Compare every run in runs/ in one table, written to "
        "results/comparison.md.",
    )
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    args = parser.parse_args(argv)
    for path in (args.runs_dir, args.links, args.answers):
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1
    try:
        key = load_answers(args.answers)
    except GradeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    linked = {r.release_accession for r in load_links(args.links)[0]}
    runs, skipped = load_runs(args.runs_dir, key)
    text = format_comparison(
        runs,
        skipped,
        linked,
        runs_dir=args.runs_dir,
        links_path=args.links,
        answers_path=args.answers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text(args.out, text)
    print(text, end="")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
