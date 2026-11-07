"""Compare evaluation runs: one table over every run in runs/.

For each run folder with a run.json (see secbench.extract): the model, provider,
prompt version and request parameters sent, accuracy overall and by field, counts
by verdict and unit slips (errors matched with a power-of-ten factor, scale and zeros
included) from the run's saved graded.csv (see secbench.grade), format retries,
format nulls (prompts with units: figures still bad on the retry), input tokens
per release, the estimated cost and per-call latency. With --gpu-hourly, runs priced by GPU-hour are
repriced at that many dollars per GPU-hour times their wall-clock time, so runs paid
at different GPU prices compare (the cost basis column still gives the price paid).
Graded runs that cover every linked release in
links.csv go in the main table, with cost per 1,000 releases; runs on fewer releases,
stopped or not graded go in a smaller table below it, so a short run's cost per
release doesn't sit beside full runs. A run whose graded.csv doesn't match the
current answers.csv, or has other columns than the grader writes (graded by an older
grader), is marked stale (re-grade it with secbench.grade). A run whose run.json
records the sha256 of its links file (secbench.extract records it from 2026-11-05)
is left out, and named under the tables, when that isn't --links's: runs on a test
split (data/splits/<split>/links.csv) never enter the validation comparison. Runs
that record none are compared as before. Nothing from .env is shown: settings come
from run.json's request_parameters. When stage 2's checkpoint runs on validation
(qwen3-8b-2k--step-*) are listed, a note says which one was chosen on that score and
the GPU they ran on at the price billed, from the stage 2 report's config
(--report-config, configs/report/stage2.toml; no note without it).

Usage: uv run python -m secbench.compare [--runs-dir runs]
       [--out results/comparison.md] [--links data/links.csv] [--answers data/answers.csv]
       [--gpu-hourly 1.00] [--report-config configs/report/stage2.toml]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from secbench.answers import DEFAULT_OUT as DEFAULT_ANSWERS
from secbench.answers import FIELDS, Answer, load_links, plural
from secbench.extract import (
    ABORTED,
    GPU_HOURS,
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
REPORT_CONFIG = Path("configs/report/stage2.toml")
# Stage 2's training run: secbench.score names its validation runs <run>--base and
# <run>--step-NNNNN.
STAGE2_RUN = "qwen3-8b-2k"

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
    runs_dir: Path, key: dict[tuple[str, str, str], Answer], *, links_sha256: str
) -> tuple[list[Run], list[str]]:
    """Every run folder with a readable run.json and no other links file recorded
    than the one of links_sha256, by start time; and the folders left out, with
    why."""
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
        recorded = info.get("links_sha256")
        if recorded is not None and recorded != links_sha256:
            skipped.append(
                f"{folder.name} (on another links file, {info.get('links')})"
            )
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
    """Figures still malformed or with an unknown unit on the retry, which made their
    release an invalid reply; "–" for a prompt without units (extract_v1, and runs
    from before reply_format)."""
    if info.get("reply_format") != WITH_UNITS:
        return "–"
    return str(len(info["format_nulls"]))


def input_tokens_per_release(info: dict[str, Any]) -> str:
    """run.json's input tokens over its releases, every call included (a retry sends
    the conversation again); none unless every call has token counts. A call that
    failed (an HTTP error) records null input tokens and isn't counted in
    calls_without_usage."""
    tokens = info.get("tokens")
    if not tokens or tokens.get("calls_without_usage") or not info.get("releases"):
        return "none"
    calls = info.get("calls", [])
    if any("input_tokens" in c and c["input_tokens"] is None for c in calls):
        return "none"
    return f"{tokens['input'] / info['releases']:,.0f}"


def cost_basis(info: dict[str, Any]) -> str | None:
    """run.json's cost_basis; runs from before it was recorded priced tokens."""
    if "cost_basis" in info:
        return info["cost_basis"]
    return TOKENS if info.get("estimated_cost_usd") is not None else None


def money(value: float, places: int) -> str:
    return f"${value:,.{places}f}"


def describe_basis(info: dict[str, Any], gpu_hourly: float | None = None) -> str:
    basis = cost_basis(info)
    if basis is None:
        return "none"
    if basis == TOKENS:
        prices = info["price_per_million_tokens"]
        return f"tokens (${prices['input']:g} in, ${prices['output']:g} out per M)"
    paid, seconds = info["gpu_hourly_usd"], info["wall_time_s"]
    if gpu_hourly is None:
        return f"GPU (${paid:g}/h x {seconds:.1f} s)"
    return f"GPU at ${gpu_hourly:g}/h x {seconds:.1f} s (paid ${paid:g}/h)"


def cost_value(info: dict[str, Any], gpu_hourly: float | None = None) -> float | None:
    """run.json's estimated cost; with gpu_hourly, a run priced by GPU-hour is that
    price times its wall-clock time instead."""
    if gpu_hourly is not None and cost_basis(info) == GPU_HOURS:
        return gpu_hourly * info["wall_time_s"] / 3600
    return info.get("estimated_cost_usd")


def cost(info: dict[str, Any], gpu_hourly: float | None = None) -> str:
    value = cost_value(info, gpu_hourly)
    return "none" if value is None else money(value, 4)


def cost_per_thousand(info: dict[str, Any], gpu_hourly: float | None = None) -> str:
    value = cost_value(info, gpu_hourly)
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


def main_table(
    runs: list[Run], gpu_hourly: float | None = None
) -> tuple[list[str], list[str]]:
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
        "input tokens / release",
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
                input_tokens_per_release(info),
                cost(info, gpu_hourly),
                describe_basis(info, gpu_hourly),
                cost_per_thousand(info, gpu_hourly),
                latency(info, "p50"),
                latency(info, "p95"),
            ]
        )
    return markdown_table(header, rows), [v for v in VERDICTS if v not in shown]


def other_table(runs: list[Run], gpu_hourly: float | None = None) -> list[str]:
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
        "input tokens / release",
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
            input_tokens_per_release(run.info),
            cost(run.info, gpu_hourly),
            describe_basis(run.info, gpu_hourly),
            latency(run.info, "p50"),
            latency(run.info, "p95"),
        ]
        for run in runs
    ]
    return markdown_table(header, rows)


def stage2_note(runs: list[Run], report: Mapping[str, Any]) -> str | None:
    """A note on stage 2's runs on validation, if a checkpoint's is listed: the
    checkpoint chosen on that score (report's under_test), and the GPU they ran on at
    the price billed, beside the $/h they record. report is the stage 2 report's
    config (configs/report/stage2.toml)."""
    prefix = f"{STAGE2_RUN}--"
    if not any(r.name.startswith(f"{prefix}step-") for r in runs):
        return None
    chosen = next(r for r in runs if r.name == prefix + report["under_test"])
    verdicts = [v for v in chosen.verdicts() if v not in UNGRADED]
    correct = verdicts.count(CORRECT)
    recorded = sorted(
        {r.info["gpu_hourly_usd"] for r in runs if r.name.startswith(prefix)}
    )
    gpu = report["gpu"]["validation"]
    # Name the listed runs, not a pattern that also matches the test-split runs.
    names = f"{prefix}step-*"
    if any(r.name == f"{prefix}base" for r in runs):
        names = f"{prefix}base and {names}"
    return (
        f"- {names}: stage 2's base model and fine-tuned checkpoints, scored on "
        f"these releases to choose a checkpoint. {chosen.name}'s "
        f"{100 * correct / len(verdicts):.1f}% ({correct}/{len(verdicts)}) is "
        "chosen on validation (the checkpoint was picked for that score, so it may "
        "overstate; the test sets give the estimate: results/test_results.md). "
        f"These runs record {', '.join(f'${p:g}/h' for p in recorded)} but ran on a "
        f"{report['gpu_host']} {gpu['gpu']} billed ${gpu['billed_usd_per_hour']:g}/h."
    )


def format_comparison(
    runs: list[Run],
    skipped: list[str],
    linked: set[str],
    *,
    runs_dir: Path,
    links_path: Path,
    answers_path: Path,
    gpu_hourly: float | None = None,
    report: Mapping[str, Any] | None = None,
) -> str:
    full = [
        r
        for r in runs
        if r.info.get("status") != ABORTED
        and r.graded is not None
        and linked <= r.covered
    ]
    others = [r for r in runs if r not in full]
    command = "uv run python -m secbench.compare"
    if gpu_hourly is not None:
        command += f" --gpu-hourly {gpu_hourly:g}"
    where = f"{plural(len(runs), 'run')} in {runs_dir}/"
    if skipped:
        where = (
            f"{len(runs)} of the {len(runs) + len(skipped)} run folders in "
            f"{runs_dir}/ ({len(skipped)} not listed, named at the end)"
        )
    lines = [
        "# Model comparison",
        "",
        (
            f"From `{command}`: {where}, graded against {answers_path}; "
            f"{links_path} has {plural(len(linked), 'linked release')}."
        ),
        "",
        f"## Runs on all {len(linked)} linked releases",
        "",
    ]
    zero: list[str] = []
    if full:
        table, zero = main_table(full, gpu_hourly)
        lines.extend(table)
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Other runs: fewer releases, stopped or not graded",
            "",
            *(other_table(others, gpu_hourly) if others else ["None."]),
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
                'retry; each made its release an invalid reply (the "(n invalid)" of '
                "format retries); – for prompts without units."
            ),
            (
                "- Input tokens / release: run.json's input tokens over its releases, "
                "retries included. The same release counts as more or fewer tokens "
                "depending on the model's tokenizer; none if a call reported no "
                "token counts."
            ),
            (
                "- Cost is run.json's estimate: tokens at the run's prices per "
                "million, or GPU dollars per hour times the run's wall-clock time. "
                "Latency is per call."
                if gpu_hourly is None
                else "- Cost is run.json's estimate at the run's token prices per "
                f"million; runs priced by GPU-hour are repriced at ${gpu_hourly:g} "
                "per GPU-hour times the run's wall-clock time (--gpu-hourly), so "
                "they compare with each other (paid: the price each run recorded). "
                "Latency is per call."
            ),
            (
                f"- {STALE}: graded.csv doesn't match {answers_path}, or has other "
                "columns than the grader writes now; run `uv run python -m "
                "secbench.grade runs/<run>/`."
            ),
        ]
    )
    if report is not None and (note := stage2_note(runs, report)):
        lines.append(note)
    if skipped:
        lines.append(f"- Not listed: {', '.join(skipped)}.")
    return "\n".join(lines) + "\n"


def dollars(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("not a number") from None
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return number


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
    parser.add_argument(
        "--gpu-hourly",
        type=dollars,
        metavar="DOLLARS",
        help="reprice runs priced by GPU-hour at this many dollars per GPU-hour",
    )
    parser.add_argument(
        "--report-config",
        type=Path,
        default=REPORT_CONFIG,
        help="the stage 2 report's config, for the note on stage 2's runs",
    )
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
    report = None
    if args.report_config.is_file():
        report = tomllib.loads(args.report_config.read_text(encoding="utf-8"))
    linked = {r.release_accession for r in load_links(args.links)[0]}
    links_sha256 = hashlib.sha256(args.links.read_bytes()).hexdigest()
    runs, skipped = load_runs(args.runs_dir, key, links_sha256=links_sha256)
    text = format_comparison(
        runs,
        skipped,
        linked,
        runs_dir=args.runs_dir,
        links_path=args.links,
        answers_path=args.answers,
        gpu_hourly=args.gpu_hourly,
        report=report,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text(args.out, text)
    print(text, end="")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
