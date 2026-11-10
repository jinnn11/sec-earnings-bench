"""The serving benchmark's report (PLAN.md Decisions 2026-11-09), on the laptop, from
the results secbench.servebench made on the GPU (scripts/vast_setup.sh --serving packs
them into results-<run>-serving.tar; unpack it in the repository).

Usage: uv run python -m secbench.servereport
           [--config configs/serving/qwen3-8b-2k.toml] [--runs-dir runs]
           [--raw data/raw] [--out-dir results]

It reads training/<run>/serving/ (servebench.json, each variant's variant.json,
startup.json and load/summary.json, the model folders' manifests and setup.log) and
grades each accuracy run (runs/<run>--serving--<variant>--<split>, secbench.grade with
the split's links.csv and answers.csv, from the EDGAR cache; graded.csv and
summary.txt are written into each run, as stage 2's runs were graded). The stage 2
runs the config names as references are read, never graded again.

results/serving_results.md and .json:
- the variants: status, model, server command;
- memory: the weights' and KV cache's GiB, the KV cache in tokens and the maximum
  concurrency vLLM reports for 40,960-token requests (its start-up log), the weights
  on disk, and the time from launch to healthy;
- the sweep: per variant and level, the makespan, releases per hour, prompt and
  output tokens per second, latency p50, p95 and p99, time to first token p50 and
  p95, the median decode speed, errors, and the cost per 1,000 releases (the GPU's
  price times the makespan, per release without an error, at the price paid and at
  $1/h);
- the best throughput per variant (the level with the most releases per hour; ties
  to the lower level) and its cost per 1,000 releases;
- accuracy on validation, overall and by field, and the agreement of every predicted
  figure (6 per release: quarter and year to date, revenue, net income, diluted EPS;
  null equal to null) with the adapter variant (the merge check: merged should equal
  adapter) and the adapter's with stage 2's validation run;
- the time test, descriptive only (its stage 2 results are reported and no variant is
  chosen on it): accuracy and agreement with stage 2's step-00025 time-test run;
- the session's fixed overhead: the setup log's steps (install, download, the
  benchmark), the model builds and the server starts, and their cost at the price.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from secbench import extract, grade, score, serving, splits, train
from secbench.edgar import EdgarClient
from secbench.grade import FIELD_NAMES
from secbench.serving import ADAPTER, MERGED, ServingConfig

DEFAULT_CONFIG = Path("configs/serving/qwen3-8b-2k.toml")
DEFAULT_OUT = Path("results")
REPORT = "serving_results.md"
RESULTS = "serving_results.json"
GROUPS = extract.GROUPS
DIFFERENCES_SHOWN = 10
STEP = re.compile(r"^== (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (.*)$", re.MULTILINE)


class ReportError(Exception):
    """The report can't be made; the message says why."""


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# Costs


def cost_per_1000(hourly: float, makespan_s: float | None, ok: int) -> float | None:
    """Dollars per 1,000 releases: the GPU's price for the makespan, per release
    without an error."""
    if makespan_s is None or ok == 0:
        return None
    return hourly * makespan_s / 3600 / ok * 1000


def best_level(levels: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The level with the most releases per hour; ties to the lower level."""
    scored = [x for x in levels if x.get("releases_per_hour")]
    if not scored:
        return None
    return max(scored, key=lambda x: (x["releases_per_hour"], -x["level"]))


# Accuracy and agreement


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """predictions.jsonl by release: period end, and each figure as a Decimal or
    None."""
    found = {}
    for line in (path / grade.PREDICTIONS).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[row["release_accession"]] = {
            "period_end": row["period_end"],
            **{
                (group, name): None
                if row[group][name] is None
                else Decimal(str(row[group][name]))
                for group in GROUPS
                for name in FIELD_NAMES
            },
        }
    return found


def agreement(run: Path, reference: Path) -> dict[str, Any] | None:
    """How many of run's predicted figures equal reference's, on the releases both
    have (6 per release; null equals null)."""
    if (
        not (run / grade.PREDICTIONS).is_file()
        or not (reference / grade.PREDICTIONS).is_file()
    ):
        return None
    mine, theirs = read_predictions(run), read_predictions(reference)
    common = [a for a in mine if a in theirs]
    keys = [(g, n) for g in GROUPS for n in FIELD_NAMES]
    differences = [
        {
            "release_accession": a,
            "figure": f"{group}.{name}",
            "run": None
            if mine[a][(group, name)] is None
            else str(mine[a][(group, name)]),
            "reference": None
            if theirs[a][(group, name)] is None
            else str(theirs[a][(group, name)]),
        }
        for a in common
        for group, name in keys
        if mine[a][(group, name)] != theirs[a][(group, name)]
    ]
    differing = {d["release_accession"] for d in differences}
    return {
        "reference": str(reference),
        "releases": len(common),
        "figures": len(common) * len(keys),
        "same": len(common) * len(keys) - len(differences),
        "releases_identical": len(common) - len(differing),
        "period_end_differs": sum(
            mine[a]["period_end"] != theirs[a]["period_end"] for a in common
        ),
        "differences": differences[:DIFFERENCES_SHOWN],
    }


def accuracy(run: Path) -> dict[str, Any] | None:
    """graded.csv's figures correct, overall and by field (not_in_input and no_answer
    left out, as everywhere)."""
    if not (run / grade.GRADED).is_file():
        return None
    model = score.Model("-", "-", run.name, 0, 0)
    s = score.read_score(model, run, kept=False)
    return {
        "correct": s.correct,
        "graded": s.graded,
        "by_field": {f: list(s.by_field[f]) for f in FIELD_NAMES},
        "not_correct": dict(sorted(s.not_correct.items())),
        "fallbacks": s.fallbacks,
    }


def grade_run(run: Path, files: score.SplitFiles, client: EdgarClient) -> int:
    """secbench.grade on run with the split's files; its output kept quiet."""
    argv = [
        str(run),
        "--links", str(files.links),
        "--answers", str(files.answers),
        "--releases", str(files.releases),
    ]  # fmt: skip
    with contextlib.redirect_stdout(io.StringIO()):
        return grade.main(argv, client=client)


# The setup log


def setup_steps(path: Path) -> list[dict[str, Any]]:
    """Each step of scripts/vast_setup.sh's log ("== <time> <title>") and how long it
    took, to the next step's line (the last has no end)."""
    if not path.is_file():
        return []
    found = [
        (datetime.fromisoformat(m.group(1)), m.group(2))
        for m in STEP.finditer(path.read_text(errors="replace"))
    ]
    return [
        {
            "step": title,
            "started": when.isoformat(timespec="seconds"),
            "minutes": None
            if i + 1 == len(found)
            else round((found[i + 1][0] - when).total_seconds() / 60, 1),
        }
        for i, (when, title) in enumerate(found)
    ]


# The report


def variant_results(
    config: ServingConfig,
    variant: str,
    record: Mapping[str, Any] | None,
    hourly: float,
    runs_dir: Path,
) -> dict[str, Any]:
    folder = config.serving_dir / variant
    summary = read_json(folder / "load" / "summary.json")
    levels = []
    for level in (summary or {}).get("levels", []):
        levels.append(
            {
                **{k: v for k, v in level.items() if k != "error_messages"},
                "cost_per_1000_usd": cost_per_1000(
                    hourly, level["makespan_s"], level["ok"]
                ),
                "cost_per_1000_at_1_usd": cost_per_1000(
                    1.0, level["makespan_s"], level["ok"]
                ),
                "error_messages": level.get("error_messages", []),
            }
        )
    best = best_level(levels)
    startup = read_json(folder / "startup.json")
    return {
        "variant": variant,
        "served": config.served(variant),
        "status": (record or {}).get("status", "not run"),
        "error": (record or {}).get("error"),
        "model": (record or {}).get("model"),
        "server_command": ((record or {}).get("server") or {}).get("command"),
        "startup": None
        if startup is None
        else {k: v for k, v in startup.items() if k != "lines"},
        "load": None
        if summary is None
        else {
            "prompts": summary["prompts"],
            "request": summary["request"],
            "started": summary.get("started"),
            "finished": summary.get("finished"),
        },
        "levels": levels,
        "best": None
        if best is None
        else {
            "level": best["level"],
            "releases_per_hour": best["releases_per_hour"],
            "latency_p95_s": best["latency_s"]["p95"],
            "cost_per_1000_usd": best["cost_per_1000_usd"],
            "cost_per_1000_at_1_usd": best["cost_per_1000_at_1_usd"],
        },
        "runs": {
            split: str(runs_dir / config.run_name(variant, split))
            for split in config.accuracy_splits
        },
    }


def model_sizes(config: ServingConfig) -> dict[str, dict[str, Any]]:
    found = {}
    for name in (MERGED, serving.INT4_AWQ):
        manifest = read_json(config.models_dir / name / serving.MANIFEST)
        if manifest is not None:
            found[name] = {
                "weight_gib": None
                if manifest.get("weight_bytes") is None
                else manifest["weight_bytes"] / 2**30,
                "timings_s": manifest.get("timings_s"),
                "versions": manifest.get("versions"),
                "settings": manifest.get("settings"),
                "quantization_config": manifest.get("quantization_config"),
            }
    return found


def build_results(
    config: ServingConfig,
    *,
    runs_dir: Path,
    splits_dir: Path,
    client: EdgarClient,
    out: Any = print,
) -> dict[str, Any]:
    record = read_json(config.serving_dir / serving.RECORD)
    if record is None:
        raise ReportError(
            f"{config.serving_dir / serving.RECORD} not found; unpack "
            f"results-{config.run}-serving.tar in the repository first"
        )
    hourly = record["gpu_hourly_usd"]
    files = {split: score.load_split(split, splits_dir) for split in config.splits}
    variants = list(record["variants"])
    results = {
        v: variant_results(config, v, record["variants"][v], hourly, runs_dir)
        for v in variants
    }
    for result in results.values():
        result["accuracy"] = {}
        for split, path in result["runs"].items():
            run = Path(path)
            if not (run / grade.PREDICTIONS).is_file():
                result["accuracy"][split] = None
                continue
            status = grade_run(run, files[split], client)
            out(f"graded {run} ({'ok' if status == 0 else f'exit {status}'})")
            result["accuracy"][split] = accuracy(run)
    references = {
        splits.VALIDATION: config.validation_reference,
        "time_test": config.time_test_reference,
    }
    for split in config.accuracy_splits:
        adapter_run = (
            Path(results[ADAPTER]["runs"][split]) if ADAPTER in results else None
        )
        reference = references.get(split)
        for variant, result in results.items():
            run = Path(result["runs"][split])
            result.setdefault("agreement", {})[split] = {
                "with_adapter": None
                if adapter_run is None or variant == ADAPTER
                else agreement(run, adapter_run),
                "with_stage2": None if reference is None else agreement(run, reference),
            }
    reference_accuracy = {
        split: None if path is None else accuracy(path)
        for split, path in references.items()
        if split in config.accuracy_splits
    }
    steps = setup_steps(config.serving_dir / serving.SETUP_LOG)
    starts = {
        v: (r.get("startup") or {}).get("launch_to_healthy_s")
        for v, r in results.items()
    }
    builds = {
        name: model.get("wall_s")
        for name, model in record.get("models", {}).items()
        if model.get("status") == "built"
    }
    overhead_s = sum(s for s in starts.values() if s) + sum(
        s for s in builds.values() if s
    )
    return {
        "run": config.run,
        "checkpoint": config.checkpoint,
        "config": str(config.path),
        "config_sha256": config.sha256,
        "session": {
            key: record.get(key)
            for key in (
                "created",
                "finished",
                "commit",
                "clean",
                "server",
                "gpus",
                "gpu_hourly_usd",
                "base_model",
                "revision",
            )
        },
        "settings": {
            "levels": list(config.levels),
            "prompts": config.prompts,
            "warmup": config.warmup,
            "load_split": config.load_split,
            "prefix_caching": config.prefix_caching,
            "max_model_len": config.max_model_len,
            "accuracy_concurrency": config.concurrency,
        },
        "variants": results,
        "models": model_sizes(config),
        "reference_accuracy": reference_accuracy,
        "overhead": {
            "setup_steps": steps,
            "server_starts_s": starts,
            "model_builds_s": builds,
            "starts_and_builds_s": overhead_s,
            "starts_and_builds_usd": hourly * overhead_s / 3600,
        },
    }


def fmt(x: Any, spec: str = ",.1f", none: str = "–") -> str:
    return none if x is None else format(x, spec)


def fraction(pair: Sequence[int] | None) -> str:
    if not pair or not pair[1]:
        return "–"
    return f"{pair[0]}/{pair[1]}"


def percent(correct: int, graded: int) -> str:
    return f"{correct}/{graded} ({100 * correct / graded:.1f}%)" if graded else "0/0"


def agreement_text(found: Mapping[str, Any] | None) -> str:
    if not found:
        return "–"
    return (
        f"{found['same']}/{found['figures']} figures, "
        f"{found['releases_identical']}/{found['releases']} releases"
    )


def format_report(results: Mapping[str, Any]) -> str:
    session, settings = results["session"], results["settings"]
    hourly = session["gpu_hourly_usd"]
    variants = results["variants"]
    lines = [
        f"# Serving benchmark: {results['run']} {results['checkpoint']}",
        "",
        (
            f"{session['base_model']} with the {results['checkpoint']} adapter, served "
            f"by {session['server']} on {', '.join(session['gpus'] or ['an unrecorded GPU'])}, "
            f"at ${hourly:g} per GPU-hour; session {session['created']} to "
            f"{session['finished']}, commit {str(session['commit'])[:12]}"
            f"{'' if session['clean'] else ' (uncommitted changes)'}. Config "
            f"{results['config']} (sha256 {results['config_sha256'][:12]}). Prefix "
            f"caching {'on' if settings['prefix_caching'] else 'off'}, --max-model-len "
            f"{settings['max_model_len']:,}; vLLM's defaults otherwise."
        ),
        "",
        "## Variants",
        "",
        "| variant | status | served as | model | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for v in variants.values():
        note = (v["error"] or "").splitlines()[0][:160] if v["error"] else ""
        lines.append(
            f"| {v['variant']} | {v['status']} | {v['served']} | "
            f"{serving.MODEL_OF.get(v['variant']) or 'base + adapter'} | {note} |"
        )
    lines += [
        "",
        "## Memory (vLLM's start-up log)",
        "",
        (
            "| variant | weights (GiB) | KV cache (GiB) | KV cache (tokens) | max "
            "concurrency at 40,960 tokens | weights on disk (GiB) | launch to healthy (s) |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for v in variants.values():
        s = v["startup"] or {}
        disk = results["models"].get(serving.MODEL_OF[v["variant"]] or "", {})
        lines.append(
            f"| {v['variant']} | {fmt(s.get('weights_gib'), '.2f')} | "
            f"{fmt(s.get('kv_cache_gib'), '.2f')} | {fmt(s.get('kv_cache_tokens'), ',')} | "
            f"{fmt(s.get('max_concurrency'), '.2f')}"
            f"{'x' if s.get('max_concurrency') is not None else ''} | "
            f"{fmt(disk.get('weight_gib'), '.2f')} | "
            f"{fmt(s.get('launch_to_healthy_s'), '.0f')} |"
        )
    lines += [
        "",
        (
            f"## Throughput and latency ({settings['load_split']} prompts, "
            f"{settings['prompts']} per level after {settings['warmup']} warm-up)"
        ),
        "",
        (
            "Per level: the same prompts, that many requests in flight. Cost per 1,000 "
            f"releases: the GPU's price for the makespan (${hourly:g}/h, and $1/h)."
        ),
        "",
        (
            "| variant | in flight | ok | makespan (s) | releases/h | output tok/s | prompt "
            "tok/s | latency p50 / p95 / p99 (s) | first token p50 / p95 (s) | decode "
            f"tok/s (median) | $/1,000 at ${hourly:g}/h | $/1,000 at $1/h |"
        ),
        "| --- " * 12 + "|",
    ]
    for v in variants.values():
        for x in v["levels"]:
            lat, ttft = x["latency_s"], x["ttft_s"]
            lines.append(
                f"| {v['variant']} | {x['level']} | {x['ok']}/{x['requests']} | "
                f"{fmt(x['makespan_s'])} | {fmt(x['releases_per_hour'], ',.0f')} | "
                f"{fmt(x['output_tokens_per_s'], ',.0f')} | "
                f"{fmt(x['prompt_tokens_per_s'], ',.0f')} | "
                f"{fmt(lat['p50'], '.2f')} / {fmt(lat['p95'], '.2f')} / "
                f"{fmt(lat['p99'], '.2f')} | {fmt(ttft['p50'], '.2f')} / "
                f"{fmt(ttft['p95'], '.2f')} | "
                f"{fmt(x['decode_tokens_per_s_p50'], ',.0f')} | "
                f"{fmt(x['cost_per_1000_usd'], '.3f')} | "
                f"{fmt(x['cost_per_1000_at_1_usd'], '.3f')} |"
            )
    lines += [
        "",
        "## Best throughput",
        "",
        (
            "| variant | in flight | releases/h | latency p95 (s) | $/1,000 releases at "
            f"${hourly:g}/h | at $1/h |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for v in variants.values():
        b = v["best"]
        if b is None:
            lines.append(f"| {v['variant']} | – | – | – | – | – |")
            continue
        lines.append(
            f"| {v['variant']} | {b['level']} | {fmt(b['releases_per_hour'], ',.0f')} | "
            f"{fmt(b['latency_p95_s'], '.2f')} | {fmt(b['cost_per_1000_usd'], '.3f')} | "
            f"{fmt(b['cost_per_1000_at_1_usd'], '.3f')} |"
        )
    for split, title in (
        (splits.VALIDATION, "Accuracy on validation"),
        ("time_test", "The time test (descriptive)"),
    ):
        if not any(split in v["accuracy"] for v in variants.values()):
            continue
        reference = results["reference_accuracy"].get(split)
        lines += ["", f"## {title}", ""]
        if split == "time_test":
            lines += [
                (
                    "Its stage 2 results are reported and no variant is chosen on it; "
                    "this shows how far each variant's predictions move from stage 2's."
                ),
                "",
            ]
        lines += [
            "| variant | correct | "
            + " | ".join(FIELD_NAMES)
            + " | not correct | fallbacks | same as adapter | same as stage 2 |",
            "| --- " * (6 + len(FIELD_NAMES)) + "|",
        ]
        for v in variants.values():
            a = v["accuracy"].get(split)
            agree = v.get("agreement", {}).get(split, {})
            if a is None:
                lines.append(
                    f"| {v['variant']} | – |" + " – |" * (len(FIELD_NAMES) + 4)
                )
                continue
            wrong = ", ".join(f"{k} {n}" for k, n in a["not_correct"].items()) or "–"
            lines.append(
                f"| {v['variant']} | {percent(a['correct'], a['graded'])} | "
                + " | ".join(fraction(a["by_field"][f]) for f in FIELD_NAMES)
                + f" | {wrong} | {a['fallbacks']} | "
                f"{agreement_text(agree.get('with_adapter')) if v['variant'] != ADAPTER else '(itself)'} | "
                f"{agreement_text(agree.get('with_stage2'))} |"
            )
        if reference:
            lines += [
                "",
                f"Stage 2's run of the checkpoint: {percent(reference['correct'], reference['graded'])}.",
            ]
        if split == splits.VALIDATION and MERGED in variants:
            merged = variants[MERGED].get("agreement", {}).get(split, {})
            found = merged.get("with_adapter")
            if found:
                lines.append(
                    f"Merge check: merged's predictions equal adapter's in "
                    f"{found['same']} of {found['figures']} figures "
                    f"({found['releases_identical']} of {found['releases']} releases "
                    "identical)."
                )
    overhead = results["overhead"]
    lines += ["", "## Session overhead", ""]
    if overhead["setup_steps"]:
        lines += ["| step (setup log) | minutes |", "| --- | --- |"]
        lines += [
            f"| {s['step']} | {fmt(s['minutes'])} |" for s in overhead["setup_steps"]
        ]
        lines.append("")
    starts = ", ".join(
        f"{v} {fmt(s, '.0f')} s" for v, s in overhead["server_starts_s"].items()
    )
    builds = (
        ", ".join(
            f"{m} {fmt(s, '.0f')} s" for m, s in overhead["model_builds_s"].items()
        )
        or "none in this session"
    )
    lines.append(
        f"Server starts (launch to healthy): {starts}. Model builds: {builds}. "
        f"Together {overhead['starts_and_builds_s'] / 60:.1f} min, "
        f"${overhead['starts_and_builds_usd']:.2f} at ${hourly:g}/h; the install and "
        "download are in the setup log's steps."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, client: EdgarClient | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.servereport",
        description="Grade the serving benchmark's runs and write its report.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs-dir", type=Path, default=extract.RUNS_DIR)
    parser.add_argument("--splits-dir", type=Path, default=splits.DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        config = serving.load_config(args.config)
        results = build_results(
            config,
            runs_dir=args.runs_dir,
            splits_dir=args.splits_dir,
            client=client
            or EdgarClient("offline: no requests", args.raw, offline=True),
        )
    except (
        ReportError,
        serving.ServingError,
        score.ScoreError,
        train.TrainError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = format_report(results)
    (args.out_dir / REPORT).write_text(report, encoding="utf-8")
    extract.write_json(args.out_dir / RESULTS, results)
    print(report)
    print(f"wrote {args.out_dir / REPORT} and {args.out_dir / RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
