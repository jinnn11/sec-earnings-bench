"""The stage 3 serving benchmark (PLAN.md Decisions 2026-11-09): one checkpoint served
in each variant of the serving config, on one GPU, one variant at a time.

Usage: uv run --no-sync python -m secbench.servebench
           --config configs/serving/<run>.toml --gpu-hourly DOLLARS
           [--variants adapter merged fp8 int4-awq] [--server vllm|hf] [--port 8000]
           [--runs-dir runs] [--hf-cache data/raw/hf-hub] [--splits-dir data/splits]
           [--quantize-python .venv-quantize/bin/python] [--plan-only]

Before anything starts: the config, the checkpoint's adapter (its sha256 and r), the
frozen splits the config reads (validation and the time test; each split's files as
its manifest records them, every input as tokens.csv records it, every prompt within
39,936 tokens) and the prompt's sha256 are checked, and the plan is printed with a
rough time and cost estimate (plan.json).

Then, for each variant in order (training/<run>/serving/<variant>/):
1. Its model: adapter needs none (the base snapshot and the adapter); merged and
   fp8 need models/merged, built once by secbench.quantize merge in this
   environment; int4-awq needs models/int4-awq, built by secbench.quantize awq in the
   quantize group's own environment (--quantize-python) from models/merged and the
   calibration examples (data/serving/<run>/, drawn on the laptop). A model folder
   already there is kept if its manifest records these inputs; any other stops the
   variants that need it. A build's output goes to models/<model>.log.
2. The server, with the flags every variant shares (--max-model-len 40960,
   --no-enable-prefix-caching) plus its own: adapter, stage 2's LoRA flags (served
   as stage 2 served it, under the same names); fp8, --quantization fp8. Its output
   is appended to <variant>/server.log; the time from launch to healthy is
   measured, and the start-up log is read for the weights' memory, the KV cache's
   memory and size in tokens and the maximum concurrency vLLM reports for
   40,960-token requests (startup.json; a line it can't find is null).
3. The layout check: every prompt of the splits read, through /tokenize, must give
   training's token ids (secbench.score's check).
4. The load test (secbench.loadtest) at each level of the config, on the same prompts
   drawn once from the load split with the config's seed, each level after its own
   warm-up requests on other prompts: load/requests.jsonl and load/summary.json.
5. The accuracy runs: secbench.extract --no-grade on each accuracy split at the
   config's concurrency, with stage 2's settings (runs/<run>--serving--<variant>--
   <split>; the laptop grades them). A complete run with no API error, these settings
   and releases is kept; any other run there is moved to <name>--attempt-N first.
6. The server stops; variant.json records the variant: complete, failed (and why) or
   skipped. A failed variant doesn't stop the others; a rerun keeps the variants
   already complete and redoes the rest (an incomplete load test is moved to
   load--attempt-N).

fp8 and int4-awq need vLLM on a GPU: with --server hf (secbench.hfserve, the laptop
smoke test) they are skipped. servebench.json records the session (the config's
sha256, the commit, the GPUs, the server's version, the price, each variant and each
model build) and is rewritten after each variant. Exits 1 if a variant failed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shlex
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from secbench import extract, labels, loadtest, quantize, score, serving, splits, train
from secbench.answers import load_links
from secbench.edgar import EdgarClient
from secbench.serving import ADAPTER, FP8, INT4_AWQ, MERGED, ServingConfig, ServingError

COMPLETE, FAILED, SKIPPED = "complete", "failed", "skipped"
VARIANT_RECORD = "variant.json"
STARTUP = "startup.json"
LOAD = "load"
REQUESTS = "requests.jsonl"
SUMMARY = "summary.json"
DEFAULT_QUANTIZE_PYTHON = Path(".venv-quantize/bin/python")
NO_PREFIX_CACHING = "--no-enable-prefix-caching"
GPU_FREE_MIB = 2048  # GPU memory in use once a server or a build has exited
GPU_WAIT_S = 120.0

# vLLM's start-up lines (the last match in the session wins); a missing one is null.
STARTUP_PATTERNS = {
    "weights_gib": re.compile(r"Model loading took ([\d.]+) ?GiB"),
    "kv_cache_gib": re.compile(r"Available KV cache memory: ([\d.]+) ?GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "max_concurrency": re.compile(
        r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"
    ),
    "engine_init_s": re.compile(r"init engine \(.*?\) took ([\d.]+) s"),
    "cuda_graphs_s": re.compile(r"Graph capturing finished in ([\d.]+) s"),
}

# A rough estimate, in minutes and seconds on one H100, from stage 2's runs on an H100
# NVL (PLAN.md Progress 2026-11-02 and 2026-11-07) where they apply, guesses
# otherwise.
ESTIMATE_MINUTES = {"server": 3.0, "layout": 1.0, MERGED: 6.0, INT4_AWQ: 45.0}
SECONDS_ALONE = 2.0  # one request at a time (a guess: ~12,000 prompt tokens, ~140 out)
SECONDS_AT_16 = 0.81  # per release at 16 at a time: stage 2's time test, 346 in 282 s


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# vLLM's start-up log


def last_session(path: Path) -> str:
    """The log's text after its last session line ("== <time> <command>", written by
    score.run_server); all of it if there is none."""
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    starts = [m.end() for m in re.finditer(r"(?m)^== .*\n", text)]
    return text[starts[-1] :] if starts else text


def parse_startup(text: str) -> dict[str, Any]:
    """The start-up numbers vLLM logs; null for any line not found."""

    def last(name: str) -> re.Match[str] | None:
        found = list(STARTUP_PATTERNS[name].finditer(text))
        return found[-1] if found else None

    def number(name: str, group: int = 1) -> float | None:
        match = last(name)
        return float(match.group(group).replace(",", "")) if match else None

    tokens = number("kv_cache_tokens")
    for_tokens = number("max_concurrency", 1)
    return {
        "weights_gib": number("weights_gib"),
        "kv_cache_gib": number("kv_cache_gib"),
        "kv_cache_tokens": None if tokens is None else int(tokens),
        "max_concurrency": number("max_concurrency", 2),
        "max_concurrency_tokens": None if for_tokens is None else int(for_tokens),
        "engine_init_s": number("engine_init_s"),
        "cuda_graphs_s": number("cuda_graphs_s"),
        "lines": {
            name: match.group(0) for name in STARTUP_PATTERNS if (match := last(name))
        },
    }


# Server commands


def vllm_command(
    model: Path, served: str, port: int, max_model_len: int, extra: Sequence[str]
) -> list[str]:
    executable = Path(sys.executable).with_name("vllm")
    return [
        str(executable) if executable.exists() else "vllm",
        "serve",
        str(model),
        "--served-model-name",
        served,
        "--max-model-len",
        str(max_model_len),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *extra,
    ]


def server_command(
    variant: str,
    config: ServingConfig,
    *,
    server: str,
    snapshot: Path,
    adapter: score.Model,
    model_dir: Path | None,
    port: int,
    r: int,
) -> list[str]:
    """The variant's server: adapter as stage 2 served it (score's command), the
    others their model folder; prefix caching off unless the config says so."""
    served = config.served(variant)
    if server == score.HF:
        if variant == ADAPTER:
            return score.hf_command(snapshot, config.base, [adapter], port, r)
        assert model_dir is not None
        return score.hf_command(model_dir, served, [], port, r)
    if variant == ADAPTER:
        command = score.vllm_command(snapshot, config.base, [adapter], port, r)
    else:
        assert model_dir is not None
        extra = ["--quantization", "fp8"] if variant == FP8 else []
        command = vllm_command(model_dir, served, port, config.max_model_len, extra)
    if not config.prefix_caching:
        command.append(NO_PREFIX_CACHING)
    return command


def tokenize_name(variant: str, config: ServingConfig) -> str:
    """The model /tokenize is asked with: the base model's name for the adapter (as
    score asks), the variant's own otherwise."""
    return config.base if variant == ADAPTER else config.served(variant)


# The session


def run_logged(command: Sequence[str], log_path: Path) -> int:
    """Run a command, its output appended to log_path after a line with the time and
    the command; its exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"== {now()} {shlex.join(command)}\n")
        log.flush()
        return subprocess.run(
            list(command), stdout=log, stderr=subprocess.STDOUT, check=False
        ).returncode


def gpu_memory_used() -> int | None:
    """MiB in use on the first GPU; None without nvidia-smi."""
    try:
        listed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = listed.strip().splitlines()[:1]
    return int(first[0]) if first and first[0].strip().isdigit() else None


def wait_for_gpu(out: Callable[[str], None]) -> None:
    """Wait (up to GPU_WAIT_S) until the last server or build has let go of the
    GPU's memory; say so if it hasn't."""
    deadline = time.monotonic() + GPU_WAIT_S
    while (used := gpu_memory_used()) is not None and used >= GPU_FREE_MIB:
        if time.monotonic() > deadline:
            out(f"warning: {used:,} MiB of GPU memory still in use")
            return
        time.sleep(2.0)


@dataclass
class Session:
    config: ServingConfig
    server: str
    port: int
    runs_dir: Path
    snapshot: Path
    encode: Callable[[str], list[int]]
    gpu_hourly: float
    quantize_python: Path
    adapter: score.Model
    r: int
    prompt: extract.Prompt
    files: dict[str, score.SplitFiles]
    prompts: dict[str, list[tuple[str, str, str]]]
    transport: Any = None  # httpx.MockTransport in tests (sync and async)
    start_server: score.ServerStarter = score.run_server
    run_command: Callable[[Sequence[str], Path], int] = run_logged
    out: Callable[[str], None] = print
    clock: Callable[[], float] = time.perf_counter
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_failures: dict[str, str] = field(default_factory=dict)

    @property
    def root(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def variant_dir(self, variant: str) -> Path:
        return self.config.serving_dir / variant

    def every_prompt(self) -> list[tuple[str, str, str]]:
        return [p for split in self.config.splits for p in self.prompts[split]]


def load_draw(
    config: ServingConfig, prompts: Sequence[tuple[str, str, str]]
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """The load test's measured prompts and warm-up prompts: drawn once, with the
    config's seed, from the load split's prompts (its first `pool`, if set)."""
    pool = list(prompts[: config.pool] if config.pool else prompts)
    if config.prompts + config.warmup > len(pool):
        raise ServingError(
            f"the load test needs {config.prompts + config.warmup} prompts; "
            f"{config.load_split} has {len(pool)}"
        )
    drawn = random.Random(config.seed).sample(
        range(len(pool)), config.prompts + config.warmup
    )
    return [pool[i] for i in drawn[: config.prompts]], [
        pool[i] for i in drawn[config.prompts :]
    ]


def accession(label: str) -> str:
    """The release accession of a split prompt's label ("time_test <accession>")."""
    return label.split()[-1]


def selected_releases(files: score.SplitFiles, limit: int | None) -> list[str]:
    return [
        r.release_accession
        for r in extract.select_releases(load_links(files.links)[0], limit=limit)
    ]


def estimate_minutes(
    config: ServingConfig, s: Session, variants: Sequence[str]
) -> dict:
    """A rough estimate per step (see ESTIMATE_MINUTES)."""
    sweep = sum(
        (config.prompts + config.warmup) * max(SECONDS_ALONE / level, SECONDS_AT_16)
        for level in config.levels
    )
    accuracy = sum(
        len(selected_releases(s.files[split], config.limits.get(split)))
        * SECONDS_AT_16
        * 16
        / config.concurrency
        for split in config.accuracy_splits
    )
    per_variant = (
        ESTIMATE_MINUTES["server"]
        + ESTIMATE_MINUTES["layout"]
        + (sweep + accuracy) / 60
    )
    needed = {serving.MODEL_OF[v] for v in variants} - {None}
    if INT4_AWQ in needed:
        needed.add(MERGED)  # made from the merged model
    builds = [
        m
        for m in (MERGED, INT4_AWQ)
        if m in needed and read_manifest(config.models_dir / m) is None
    ]
    minutes = {
        "per_variant": round(per_variant, 1),
        "sweep": round(sweep / 60, 1),
        "accuracy": round(accuracy / 60, 1),
        **{f"build_{b}": ESTIMATE_MINUTES[b] for b in builds},
    }
    minutes["total"] = round(
        per_variant * len(variants) + sum(ESTIMATE_MINUTES[b] for b in builds), 1
    )
    return minutes


def make_plan(
    s: Session,
    variants: Sequence[str],
    to_run: Sequence[str],
    measured: Sequence[tuple[str, str, str]],
    warmup: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    config = s.config
    minutes = estimate_minutes(config, s, to_run)
    return {
        "created": now(),
        "config": str(config.path),
        "config_sha256": config.sha256,
        "server": score.server_version(s.server),
        "variants": list(variants),
        "to_run": list(to_run),
        "kept": [v for v in variants if v not in to_run],
        "levels": list(config.levels),
        "load": {
            "split": config.load_split,
            "seed": config.seed,
            "pool": config.pool,
            "measured": [accession(p[0]) for p in measured],
            "warmup": [accession(p[0]) for p in warmup],
        },
        "accuracy": {
            split: len(selected_releases(s.files[split], config.limits.get(split)))
            for split in config.accuracy_splits
        },
        "layout_prompts": len(s.every_prompt()),
        "estimate_minutes": minutes,
        "estimate_usd": round(s.gpu_hourly * minutes["total"] / 60, 2),
    }


def format_plan(plan: Mapping[str, Any], gpu_hourly: float) -> str:
    minutes = plan["estimate_minutes"]
    lines = [
        f"Plan ({plan['server']}): {', '.join(plan['to_run']) or 'nothing to run'}"
        + (
            f"; kept from an earlier session: {', '.join(plan['kept'])}"
            if plan["kept"]
            else ""
        ),
        f"  per variant: start the server, check {plan['layout_prompts']} prompts "
        f"against /tokenize, sweep {len(plan['levels'])} levels "
        f"({', '.join(map(str, plan['levels']))} in flight) of "
        f"{len(plan['load']['measured'])} {plan['load']['split']} prompts after "
        f"{len(plan['load']['warmup'])} warm-up each (~{minutes['sweep']:.0f} min), "
        "then the accuracy runs: "
        + ", ".join(f"{split} {n}" for split, n in plan["accuracy"].items())
        + f" releases (~{minutes['accuracy']:.0f} min)",
    ]
    for key, name in (("build_merged", "merge"), ("build_int4-awq", "AWQ")):
        if key in minutes:
            lines.append(f"  build: {name} (~{minutes[key]:.0f} min)")
    lines.append(
        f"  estimate: ~{minutes['total']:.0f} min (~{minutes['total'] / 60:.1f} h), "
        f"about ${plan['estimate_usd']:.2f} at ${gpu_hourly:g}/h; rough, and without "
        "the setup's install and download"
    )
    return "\n".join(lines)


# Models


def read_manifest(folder: Path) -> dict[str, Any] | None:
    path = folder / serving.MANIFEST
    return json.loads(path.read_text()) if path.is_file() else None


def build(s: Session, name: str, command: list[str]) -> Path:
    """Run a model build; the model folder, or ServingError (remembered, so the
    variants that need it fail at once)."""
    folder = s.config.models_dir / name
    log = s.config.models_dir / f"{name}.log"
    wait_for_gpu(s.out)
    s.out(f"building {folder}: {shlex.join(command)}")
    start = s.clock()
    code = s.run_command(command, log)
    wall = round(s.clock() - start, 1)
    if code != 0 or read_manifest(folder) is None:
        failure = (
            f"building {folder} exited {code}; end of {log}:\n{score.log_tail(log)}"
        )
        s.model_failures[name] = failure
        s.models[name] = {"dir": str(folder), "status": FAILED, "wall_s": wall}
        raise ServingError(failure)
    s.models[name] = {
        "dir": str(folder),
        "status": "built",
        "command": command,
        "log": str(log),
        "wall_s": wall,
        "manifest_sha256": train.file_sha256(folder / serving.MANIFEST),
    }
    return folder


def kept_model(s: Session, name: str, problem: str | None) -> Path:
    folder = s.config.models_dir / name
    if problem:
        failure = f"{folder} was made from {problem}; move it away to make it again"
        s.model_failures[name] = failure
        raise ServingError(failure)
    s.models.setdefault(
        name,
        {
            "dir": str(folder),
            "status": "kept",
            "manifest_sha256": train.file_sha256(folder / serving.MANIFEST),
        },
    )
    return folder


def merged_model(s: Session) -> Path:
    if MERGED in s.model_failures:
        raise ServingError(f"needs the merged model: {s.model_failures[MERGED]}")
    folder = s.config.models_dir / MERGED
    manifest = read_manifest(folder)
    if manifest is None:
        assert s.adapter.adapter is not None
        return build(
            s,
            MERGED,
            [
                sys.executable, "-m", "secbench.quantize", "merge",
                "--base", str(s.snapshot),
                "--adapter", str(s.adapter.adapter),
                "--out", str(folder),
            ],
        )  # fmt: skip
    inputs = manifest.get("inputs", {})
    weights = inputs.get("adapter", {}).get("files", {}).get(train.ADAPTER_WEIGHTS)
    problem = None
    if weights != s.config.adapter_sha256:
        problem = f"an adapter with sha256 {weights}"
    elif inputs.get("base", {}).get("path") != str(s.snapshot):
        problem = f"the base snapshot {inputs.get('base', {}).get('path')}"
    return kept_model(s, MERGED, problem)


def awq_model(s: Session) -> Path:
    if INT4_AWQ in s.model_failures:
        raise ServingError(f"needs the AWQ model: {s.model_failures[INT4_AWQ]}")
    merged = merged_model(s)
    config = s.config
    calibration = config.calibration_dir / serving.CALIBRATION
    folder = config.models_dir / INT4_AWQ
    manifest = read_manifest(folder)
    try:
        quantize.check_calibration(config, config.calibration_dir)
    except quantize.QuantizeError as exc:
        s.model_failures[INT4_AWQ] = str(exc)
        raise ServingError(str(exc)) from None
    if manifest is None:
        if not s.quantize_python.is_file():
            failure = (
                f"no Python at {s.quantize_python}: AWQ runs in the quantize group's "
                "own environment (scripts/vast_setup.sh --serving installs it)"
            )
            s.model_failures[INT4_AWQ] = failure
            raise ServingError(failure)
        return build(
            s,
            INT4_AWQ,
            [
                str(s.quantize_python), "-m", "secbench.quantize", "awq",
                "--model", str(merged),
                "--calibration", str(calibration),
                "--out", str(folder),
                "--scheme", config.scheme,
                "--duo-scaling", str(config.duo_scaling).lower(),
                "--ignore", *config.ignore,
                "--max-tokens", str(config.calibration_max_tokens),
            ],
        )  # fmt: skip
    inputs = manifest.get("inputs", {})
    problem = None
    merged_sha = train.file_sha256(merged / serving.MANIFEST)
    if inputs.get("model", {}).get("manifest_sha256") != merged_sha:
        problem = "another merged model"
    elif inputs.get("calibration", {}).get("sha256") != train.file_sha256(calibration):
        problem = "other calibration examples"
    elif manifest.get("settings", {}).get("scheme") != config.scheme:
        problem = f"the scheme {manifest.get('settings', {}).get('scheme')}"
    return kept_model(s, INT4_AWQ, problem)


def variant_model(s: Session, variant: str) -> Path | None:
    model = serving.MODEL_OF[variant]
    if model is None:
        return None
    return merged_model(s) if model == MERGED else awq_model(s)


# A variant


def write_envs(s: Session, variant: str) -> dict[str, Path]:
    """Each split's extract settings file for the variant (score.env_text: stage 2's
    request settings, the config's concurrency, the price)."""
    paths = {}
    for split in s.config.splits:
        path = s.variant_dir(variant) / f"{split}.env"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            score.env_text(
                f"{s.root}/v1",
                s.config.served(variant),
                s.config.concurrency,
                s.gpu_hourly,
            )
        )
        paths[split] = path
    return paths


def format_level(variant: str, level: int, summary: Mapping[str, Any]) -> str:
    def value(x: float | None, spec: str) -> str:
        return "–" if x is None else format(x, spec)

    latency, ttft = summary["latency_s"], summary["ttft_s"]
    return (
        f"{variant} at {level}: {summary['ok']}/{summary['requests']} ok in "
        f"{value(summary['makespan_s'], '.1f')} s, "
        f"{value(summary['releases_per_hour'], ',.0f')} releases/h, "
        f"{value(summary['output_tokens_per_s'], ',.0f')} output tokens/s, latency "
        f"p50 {value(latency['p50'], '.2f')} s p95 {value(latency['p95'], '.2f')} s, "
        f"first token p50 {value(ttft['p50'], '.2f')} s"
    )


def sweep(s: Session, variant: str, env: Path) -> dict[str, Any]:
    """The load test at every level (see the module docstring)."""
    config = s.config
    folder = s.variant_dir(variant) / LOAD
    if folder.exists():
        moved = score.next_attempt(folder)
        folder.rename(moved)
        s.out(f"an earlier load test moved to {moved}")
    measured, warmup = load_draw(config, s.prompts[config.load_split])
    settings = extract.load_settings(env_file=env)
    bodies = loadtest.request_bodies(
        settings, s.prompt, [p[2] for p in (*measured, *warmup)]
    )
    requests = [
        loadtest.Request(accession(p[0]), body)
        for p, body in zip((*measured, *warmup), bodies, strict=True)
    ]
    measured_requests, warmup_requests = (
        requests[: len(measured)],
        requests[len(measured) :],
    )
    record: dict[str, Any] = {
        "variant": variant,
        "served": config.served(variant),
        "url": settings.endpoint,
        "started": now(),
        "request": {
            "prompt": s.prompt.version,
            "prompt_sha256": s.prompt.sha256,
            "parameters": settings.parameters,
            "stream": loadtest.STREAM,
        },
        "prompts": {
            "split": config.load_split,
            "seed": config.seed,
            "pool": config.pool,
            "measured": [r.label for r in measured_requests],
            "warmup": [r.label for r in warmup_requests],
        },
        "levels": [],
    }
    rows: list[dict[str, Any]] = []
    for level in config.levels:
        s.out(
            f"{variant}: level {level}, {len(warmup_requests)} warm-up then "
            f"{len(measured_requests)} requests"
        )
        start = now()
        warm, results = loadtest.run_level(
            settings.endpoint,
            warmup_requests,
            measured_requests,
            level,
            transport=s.transport,
            clock=s.clock,
        )
        rows += [r.record(variant=variant, level=level) for r in (*warm, *results)]
        summary = {
            "level": level,
            "started": start,
            **loadtest.summarize(results),
            "warmup_errors": sum(bool(r.error) for r in warm),
        }
        record["levels"].append(summary)
        s.out(format_level(variant, level, summary))
        loadtest.write_jsonl(folder / REQUESTS, rows)
        extract.write_json(folder / SUMMARY, record)
    record["finished"] = now()
    extract.write_json(folder / SUMMARY, record)
    return {
        "summary": str(folder / SUMMARY),
        "requests": str(folder / REQUESTS),
        "levels": len(record["levels"]),
        "errors": sum(level["errors"] for level in record["levels"]),
    }


def accuracy_run(s: Session, variant: str, split: str, env: Path) -> dict[str, Any]:
    """The variant's predictions on a split (secbench.extract --no-grade)."""
    config = s.config
    files = s.files[split]
    limit = config.limits.get(split)
    model = score.Model(
        variant,
        config.served(variant),
        config.run_name(variant, split),
        s.adapter.step,
        s.adapter.examples_seen,
        s.adapter.adapter,
    )
    path = s.runs_dir / model.run_name
    releases = selected_releases(files, limit)
    moved = None
    try:
        kept = score.kept_test_run(
            path, model, releases, score.file_sha256(files.links), config.concurrency
        )
    except score.ScoreError:
        moved = score.next_attempt(path)
        path.rename(moved)
        s.out(f"{path} wasn't a complete run with these settings: moved to {moved}")
        kept = False
    if not kept:
        argv = [
            "--env-file", str(env),
            "--run-name", model.run_name,
            "--prompt", labels.PROMPT,
            "--runs-dir", str(s.runs_dir),
            "--links", str(files.links),
            "--answers", str(files.answers),
            "--releases", str(files.releases),
            "--no-grade",
            *(["--limit", str(limit)] if limit else []),
        ]  # fmt: skip
        s.out(f"{variant}: predictions on {split} ({len(releases)} releases)")
        status = extract.main(
            argv,
            transport=s.transport,
            edgar_client=EdgarClient("offline: no requests", offline=True),
        )
        if status:
            raise ServingError(
                f"{path} had API errors or stopped (see its run.json); a rerun moves "
                f"it to {score.next_attempt(path)}"
            )
    return {
        "split": split,
        **score.run_record(model, path, kept),
        "moved_to": None if moved is None else str(moved),
    }


def complete_record(path: Path) -> dict[str, Any] | None:
    """variant.json, if it records a complete variant whose outputs are there."""
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get("status") != COMPLETE:
        return None
    outputs = [Path(record["load"]["summary"])] + [
        Path(run["run"]) / extract.RUN_INFO for run in record.get("runs", [])
    ]
    return record if all(p.is_file() for p in outputs) else None


def run_variant(s: Session, variant: str) -> dict[str, Any]:
    config = s.config
    folder = s.variant_dir(variant)
    record_path = folder / VARIANT_RECORD
    kept = complete_record(record_path)
    if kept is not None:
        s.out(f"{variant}: complete in an earlier session; kept")
        return {**kept, "kept": True}
    record: dict[str, Any] = {
        "variant": variant,
        "served": config.served(variant),
        "status": None,
        "error": None,
        "kept": False,
        "started": now(),
    }
    if s.server == score.HF and variant in serving.VLLM_ONLY:
        record["status"] = SKIPPED
        record["error"] = (
            f"{variant} needs vLLM on a GPU; secbench.hfserve (--server hf) can't "
            "serve it"
        )
        s.out(f"{variant}: skipped: {record['error']}")
    else:
        try:
            model_dir = variant_model(s, variant)
            record["model"] = None if model_dir is None else str(model_dir)
            command = server_command(
                variant,
                config,
                server=s.server,
                snapshot=s.snapshot,
                adapter=s.adapter,
                model_dir=model_dir,
                port=s.port,
                r=s.r,
            )
            envs = write_envs(s, variant)
            log = folder / score.SERVER_LOG
            record["server"] = {
                "version": score.server_version(s.server),
                "command": command,
                "log": str(log),
            }
            wait_for_gpu(s.out)
            with httpx.Client(transport=s.transport) as http:
                s.out(f"{variant}: starting {shlex.join(command)}")
                start = s.clock()
                with s.start_server(command, log, s.root, http):
                    record["server"]["launch_to_healthy_s"] = round(
                        s.clock() - start, 1
                    )
                    startup = parse_startup(last_session(log))
                    startup["launch_to_healthy_s"] = record["server"][
                        "launch_to_healthy_s"
                    ]
                    extract.write_json(folder / STARTUP, startup)
                    record["startup"] = {
                        k: v for k, v in startup.items() if k != "lines"
                    }
                    prompts = s.every_prompt()
                    s.out(
                        f"{variant}: checking {len(prompts)} prompts against /tokenize"
                    )
                    problems = score.layout_problems(
                        http, s.root, tokenize_name(variant, config), prompts, s.encode
                    )
                    if problems:
                        raise ServingError(
                            f"{len(problems)} prompts aren't tokenized as in "
                            "training: " + "; ".join(problems[:5])
                        )
                    record["layout_check"] = {
                        "prompts": len(prompts),
                        "identical_to_training_layout": True,
                    }
                    record["load"] = sweep(s, variant, envs[config.load_split])
                    record["runs"] = [
                        accuracy_run(s, variant, split, envs[split])
                        for split in config.accuracy_splits
                    ]
            record["status"] = COMPLETE
        except Exception as exc:  # noqa: BLE001 (a failed variant mustn't stop the rest)
            record["status"] = FAILED
            record["error"] = f"{type(exc).__name__}: {exc}"
            s.out(f"{variant}: failed: {record['error']}")
            s.out(traceback.format_exc())
    record["finished"] = now()
    folder.mkdir(parents=True, exist_ok=True)
    extract.write_json(record_path, record)
    return record


def servebench(
    config: ServingConfig,
    *,
    variants: Sequence[str],
    server: str,
    port: int,
    runs_dir: Path,
    splits_dir: Path,
    gpu_hourly: float,
    snapshot: Path,
    encode: Callable[[str], list[int]],
    quantize_python: Path,
    plan_only: bool = False,
    transport: Any = None,
    start_server: score.ServerStarter = score.run_server,
    run_command: Callable[[Sequence[str], Path], int] = run_logged,
    out: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    """Check, plan, then run each variant (see the module docstring)."""
    adapter = serving.adapter_model(config)
    _, run_config, _ = train.read_run(config.run_dir)
    prompt = extract.load_prompt(labels.PROMPT)
    files = {split: score.load_split(split, splits_dir) for split in config.splits}
    prompts = {}
    for split, found in files.items():
        if prompt.sha256 != found.manifest["prompt"]["sha256"]:
            raise ServingError(
                f"{labels.PROMPT}'s sha256 isn't the one {found.folder} was built with"
            )
        prompts[split] = score.split_prompts(found, prompt.text)
    s = Session(
        config=config,
        server=server,
        port=port,
        runs_dir=runs_dir,
        snapshot=snapshot,
        encode=encode,
        gpu_hourly=gpu_hourly,
        quantize_python=quantize_python,
        adapter=adapter,
        r=run_config.r,
        prompt=prompt,
        files=files,
        prompts=prompts,
        transport=transport,
        start_server=start_server,
        run_command=run_command,
        out=out,
        clock=clock,
    )
    lengths = score.prompt_lengths(s.every_prompt(), encode)
    over = [(label, n) for label, n in lengths if n > splits.MAX_PROMPT_TOKENS]
    if over:
        raise ServingError(
            f"{len(over)} prompts over {splits.MAX_PROMPT_TOKENS:,} tokens: "
            + "; ".join(f"{label} ({n:,})" for label, n in over[:5])
        )
    measured, warmup = load_draw(config, prompts[config.load_split])
    to_run = [
        v
        for v in variants
        if complete_record(s.variant_dir(v) / VARIANT_RECORD) is None
    ]
    plan = make_plan(s, variants, to_run, measured, warmup)
    out(format_plan(plan, gpu_hourly))
    if plan_only:
        return 0
    config.serving_dir.mkdir(parents=True, exist_ok=True)
    extract.write_json(config.serving_dir / serving.PLAN, plan)
    record: dict[str, Any] = {
        "config": str(config.path),
        "config_sha256": config.sha256,
        "created": now(),
        **extract.code_version(),
        "run": config.run,
        "checkpoint": config.checkpoint,
        "adapter_sha256": config.adapter_sha256,
        "base_model": config.base,
        "revision": config.revision,
        "snapshot": str(snapshot),
        "server": score.server_version(server),
        "gpus": score.gpus(),
        "gpu_hourly_usd": gpu_hourly,
        "splits": {
            split: {
                "manifest_sha256": found.manifest_sha256,
                "links": str(found.links),
                "answers": str(found.answers),
                "releases_dir": str(found.releases),
            }
            for split, found in files.items()
        },
        "variants": {},
        "models": s.models,
    }
    path = config.serving_dir / serving.RECORD
    extract.write_json(path, record)
    for variant in variants:
        out(f"\n== {now()} {variant}")
        record["variants"][variant] = run_variant(s, variant)
        record["finished"] = now()
        extract.write_json(path, record)
    out("")
    for variant, result in record["variants"].items():
        out(
            f"{variant}: {result['status']}"
            + (" (kept)" if result.get("kept") else "")
            + (f": {result['error'].splitlines()[0]}" if result.get("error") else "")
        )
    out(f"wrote {path}")
    failed = [v for v, r in record["variants"].items() if r["status"] == FAILED]
    return 1 if failed else 0


def main(
    argv: list[str] | None = None,
    *,
    transport: Any = None,
    start_server: score.ServerStarter = score.run_server,
    run_command: Callable[[Sequence[str], Path], int] = run_logged,
    snapshot: Path | None = None,
    encode: Callable[[str], list[int]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.servebench",
        description="Serve one checkpoint in each variant of a serving config and "
        "measure memory, throughput, latency and accuracy.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--gpu-hourly",
        type=float,
        required=True,
        help="dollars per GPU-hour paid, for the cost",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=serving.VARIANTS,
        help="these, in this order (default: the config's)",
    )
    parser.add_argument("--server", choices=(score.VLLM, score.HF), default=score.VLLM)
    parser.add_argument("--port", type=int, default=score.DEFAULT_PORT)
    parser.add_argument("--runs-dir", type=Path, default=extract.RUNS_DIR)
    parser.add_argument("--hf-cache", type=Path, default=train.DEFAULT_HF_CACHE)
    parser.add_argument("--splits-dir", type=Path, default=splits.DEFAULT_OUT)
    parser.add_argument("--quantize-python", type=Path, default=DEFAULT_QUANTIZE_PYTHON)
    parser.add_argument(
        "--plan-only", action="store_true", help="check and print the plan; run nothing"
    )
    args = parser.parse_args(argv)
    if not (math.isfinite(args.gpu_hourly) and args.gpu_hourly >= 0):
        parser.error("--gpu-hourly must be 0 or more")
    if args.variants and len(set(args.variants)) != len(args.variants):
        parser.error("--variants: each once")
    try:
        config = serving.load_config(args.config)
        _, run_config, _ = train.read_run(config.run_dir)
        if snapshot is None:
            snapshot = train.snapshot_path(run_config, args.hf_cache).resolve()
        if encode is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(snapshot)

            def encode(text: str) -> list[int]:
                return train.encode(tokenizer, text)

        return servebench(
            config,
            variants=args.variants or config.variants,
            server=args.server,
            port=args.port,
            runs_dir=args.runs_dir,
            splits_dir=args.splits_dir,
            gpu_hourly=args.gpu_hourly,
            snapshot=snapshot,
            encode=encode,
            quantize_python=args.quantize_python,
            plan_only=args.plan_only,
            transport=transport,
            start_server=start_server,
            run_command=run_command,
            clock=clock,
        )
    except (
        ServingError,
        score.ScoreError,
        train.TrainError,
        extract.RunError,
        splits.SplitError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
