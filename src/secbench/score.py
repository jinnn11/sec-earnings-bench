"""Score a training run's checkpoints on the validation releases and choose one (PLAN.md
Decisions 2026-10-27: checkpoints are chosen on validation only); or, with
--test-config, make one checkpoint's and the base model's predictions on a frozen
test split (results/test_protocol.md).

Usage: uv run --group train --group serve python -m secbench.score training/<run>
           [--server vllm|hf] [--port 8000] [--limit N] [--runs-dir runs]
           [--concurrency 16] [--gpu-hourly 1] [--hf-cache data/raw/hf-hub]
       uv run --group train --group serve python -m secbench.score training/<run>
           --test-config configs/test/<run>.toml --split company_test
           --gpu-hourly DOLLARS [--server] [--port] [--limit N] [--runs-dir]
           [--hf-cache]

Validation:

1. One server for the base model (the run's snapshot, served under the model's name,
   such as Qwen/Qwen3-8B) and every complete checkpoint as a LoRA adapter named
   <run>--step-NNNNN: vLLM (vllm serve with --max-model-len 40960, README's setting
   for Qwen3-8B) or, for the laptop smoke test, secbench.hfserve (--server hf). vLLM
   always runs with LoRA on, the same flags before training as after: --max-lora-rank
   from the run's config r (every checkpoint must have that r), --max-loras 1 and
   --max-cpu-loras the number of checkpoints, at least 1. So scoring a run before it
   trains (scripts/vast_setup.sh does) serves the base model alone on the server setup
   the checkpoints get, and tries the LoRA flags before any GPU-hour of training. Its
   output is appended to training/<run>/validation/server.log, each session under a
   line with its time and command; it is stopped at the end.
2. The layout check: given each validation release's and each training example's
   system and user messages with thinking off, the server's /tokenize must return
   exactly the token ids training used for the prompt (secbench.chat, the model's
   tokenizer). Any difference stops before scoring.
3. Each model through secbench.extract on the linked releases of --links
   (data/links.csv: the 60 validation releases; --limit takes the first N): prompt
   extract_v2, temperature 0, thinking off, max_tokens 1024 (secbench.chat), graded
   as every run is, with an EDGAR client that reads only the cache. Runs go to
   <runs-dir>/<run>--base and <run>--step-NNNNN (CONTRIBUTING: raw outputs under
   runs/). A complete graded run already there with the same model, settings and
   releases is kept and not called again; any other run there stops the scoring.
4. training/<run>/validation.md (also printed) and validation.json: each model's
   accuracy, overall and by field, when its run finished and whether it was kept from
   an earlier session, and the chosen checkpoint: the most figures correct, ties to
   the earliest step. The base model is scored for reference and never chosen.

A test split (--test-config and --split), on the GPU instance, which has no SEC
cache: it reads only training/<run>/config.json, the config's checkpoint folder,
data/splits/<split>/ (secbench.splits), that split's release files and prompts/.
1. The test config (configs/test/<run>.toml) names the checkpoint and its adapter's
   sha256 and repeats validation's server and request settings; any that differ
   from validation's (or from the run's base model), another adapter, or a split that
   isn't frozen or whose files aren't its manifest's stops it before the server
   starts. So does a release whose input isn't the one tokens.csv records, or a
   prompt over 39,936 tokens by the model's own tokenizer.
2. One server for the base model and that one adapter (served as
   <run>--<checkpoint>), with validation's flags: vLLM with LoRA on, --max-lora-rank
   from the run's r, --max-loras 1, --max-cpu-loras 1, --max-model-len 40960. Every
   prompt of the split is checked against /tokenize, as for validation.
3. secbench.extract on the split's releases for each model, with validation's
   settings and the test config's concurrency, MODEL_GPU_HOURLY from --gpu-hourly,
   not graded (--no-grade): runs <run>--<split>--base and <run>--<split>--<checkpoint>,
   env files training/<run>/test/<split>--<label>.env. No checkpoint is chosen. A run
   already there is kept if it is complete, with no API error, the same model,
   settings and releases, and the split's links file; any other stops scoring, as
   does a run made now that isn't, with what to move it to (<name>--attempt-N).
4. training/<run>/test/<split>.json: the server command, the GPUs (nvidia-smi), the
   layout check, each run, the commit and the test config's sha256; the server's output is appended
   to training/<run>/test/server.log. Grading happens on the laptop (secbench.grade
   with the split's links.csv and answers.csv).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx

from secbench import chat, extract, grade, labels, splits, train
from secbench.answers import DEFAULT_OUT as DEFAULT_ANSWERS
from secbench.answers import load_links
from secbench.edgar import EdgarClient
from secbench.grade import FIELD_NAMES, UNGRADED
from secbench.inputs import build_input, release_records
from secbench.link import DEFAULT_OUT as DEFAULT_LINKS
from secbench.releases import DEFAULT_OUT_DIR as DEFAULT_RELEASES
from secbench.releases import write_text

VALIDATION = "validation"  # folder in the run, for the server log and env files
SERVER_LOG = "server.log"
REPORT = "validation.md"
RESULTS = "validation.json"
BASE = "base"
VLLM, HF = "vllm", "hf"
DEFAULT_PORT = 8000
DEFAULT_CONCURRENCY = 16
READY_TIMEOUT = 1800.0  # seconds for the server to load the model and adapters
STOP_TIMEOUT = 60.0
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)  # --max-lora-rank values
CHOICE_RULE = "the most figures correct; ties go to the earliest step"


class ScoreError(Exception):
    """Scoring can't go on; the message says why."""


@dataclass(frozen=True)
class Model:
    label: str  # "base" or "step-00025"
    served: str  # the name the server knows it by
    run_name: str  # its extract run, <runs-dir>/<run_name>
    step: int  # 0 for the base model
    examples_seen: int
    adapter: Path | None = None


def models(run_dir: Path, base_name: str) -> list[Model]:
    """The base model, then each complete checkpoint by step."""
    run = run_dir.name
    found = [Model(BASE, base_name, f"{run}--{BASE}", 0, 0)]
    for checkpoint in train.checkpoints(run_dir):
        state = json.loads((checkpoint.path / train.STATE).read_text())
        name = f"{run}--{checkpoint.path.name}"
        found.append(
            Model(
                checkpoint.path.name,
                name,
                name,
                checkpoint.step,
                state["examples_seen"],
                checkpoint.path.resolve(),
            )
        )
    return found


def lora_rank(r: int) -> int:
    """The smallest --max-lora-rank vLLM takes that fits r."""
    rank = next((rank for rank in VLLM_LORA_RANKS if rank >= r), None)
    if rank is None:
        raise ScoreError(f"LoRA r {r} is over vLLM's largest --max-lora-rank")
    return rank


def check_adapters(adapters: Sequence[Model], r: int) -> None:
    """Every checkpoint's adapter has the config's r, which the server is set for."""
    for model in adapters:
        assert model.adapter is not None
        found = json.loads((model.adapter / train.ADAPTER_CONFIG).read_text())["r"]
        if found != r:
            raise ScoreError(
                f"{model.adapter}: LoRA r {found}, but the run's config says {r}"
            )


def vllm_command(
    snapshot: Path, base_name: str, adapters: Sequence[Model], port: int, r: int
) -> list[str]:
    """vllm serve with LoRA on, whether or not there are adapters yet."""
    executable = Path(sys.executable).with_name("vllm")
    command = [
        str(executable) if executable.exists() else "vllm",
        "serve",
        str(snapshot),
        "--served-model-name",
        base_name,
        "--max-model-len",
        str(chat.MAX_MODEL_LEN),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enable-lora",
        "--max-lora-rank",
        str(lora_rank(r)),
        "--max-loras",
        "1",
        "--max-cpu-loras",
        str(max(1, len(adapters))),
    ]
    if adapters:
        command += ["--lora-modules", *(f"{m.served}={m.adapter}" for m in adapters)]
    return command


def hf_command(
    snapshot: Path, base_name: str, adapters: Sequence[Model], port: int, r: int
) -> list[str]:
    """The stand-in loads adapters as they come; r doesn't set anything."""
    command = [
        sys.executable,
        "-m",
        "secbench.hfserve",
        str(snapshot),
        "--served-model-name",
        base_name,
        "--port",
        str(port),
        "--max-model-len",
        str(chat.MAX_MODEL_LEN),
    ]
    if adapters:
        command += ["--lora-modules", *(f"{m.served}={m.adapter}" for m in adapters)]
    return command


def server_version(kind: str) -> str:
    def version(package: str) -> str:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "not installed"

    if kind == VLLM:
        return f"vLLM {version('vllm')}"
    return (
        f"secbench.hfserve (transformers {version('transformers')}, peft "
        f"{version('peft')}), a stand-in for vLLM"
    )


def log_tail(path: Path, lines: int = 20) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


@contextmanager
def run_server(
    command: Sequence[str], log_path: Path, root: str, http: httpx.Client
) -> Iterator[None]:
    """Start the server, wait until GET /health answers 200, stop it at the end. Its
    output is appended to log_path, after a line with the time and the command."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        started = datetime.now(UTC).isoformat(timespec="seconds")
        log.write(f"== {started} {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            list(command), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + READY_TIMEOUT
            while True:
                if process.poll() is not None:
                    raise ScoreError(
                        f"the server exited ({process.returncode}) before it was "
                        f"ready; end of {log_path}:\n{log_tail(log_path)}"
                    )
                try:
                    if http.get(f"{root}/health", timeout=5.0).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() > deadline:
                    raise ScoreError(
                        f"the server wasn't ready in {READY_TIMEOUT:.0f} s"
                    )
                time.sleep(2.0)
            yield
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(STOP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


# The layout check


def layout_prompts(
    info: Mapping[str, Any],
    config: train.Config,
    run_dir: Path,
    links: Path,
    releases: Path,
    prompt: str,
) -> list[tuple[str, str, str]]:
    """(label, system, user) of each validation release, then of each of the run's
    training examples."""
    records = release_records(releases)
    prompts = []
    for release in load_links(links)[0]:
        record = records.get(release.release_accession)
        if record is None:
            raise ScoreError(
                f"no saved release {release.release_accession} in {releases}"
            )
        prompts.append(
            (
                f"validation {release.release_accession}",
                prompt,
                build_input(record).text,
            )
        )
    schedule = train.make_schedule(info["schedule"]["examples_per_epoch"], config)
    accessions = {e.accession for e in train.read_order(run_dir, schedule)}
    for accession, messages in train.load_messages(config.examples, accessions).items():
        prompts.append(
            (f"training {accession}", messages[0]["content"], messages[1]["content"])
        )
    return prompts


def layout_problems(
    http: httpx.Client,
    root: str,
    served: str,
    prompts: Sequence[tuple[str, str, str]],
    encode: Callable[[str], list[int]],
) -> list[str]:
    """Each prompt whose tokens from the server's /tokenize (chat messages, thinking
    off) differ from training's."""
    problems = []
    for label, system, user in prompts:
        response = http.post(
            f"{root}/tokenize",
            json={
                "model": served,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "add_generation_prompt": True,
                "add_special_tokens": False,
                "chat_template_kwargs": chat.THINKING_OFF,
            },
            timeout=120.0,
        )
        if response.status_code != 200:
            problems.append(f"{label}: /tokenize answered HTTP {response.status_code}")
            continue
        got = response.json()["tokens"]
        want = encode(chat.prompt_text(system, user))
        if got != want:
            at = next(
                (i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b),
                min(len(got), len(want)),
            )
            problems.append(
                f"{label}: the server's {len(got)} tokens differ from training's "
                f"{len(want)} at token {at}"
            )
    return problems


# Scoring


def env_text(base_url: str, served: str, concurrency: int, gpu_hourly: float) -> str:
    """The extract settings file for one model."""
    extra = json.dumps(
        {"chat_template_kwargs": chat.THINKING_OFF, "max_tokens": chat.MAX_TOKENS}
    )
    return (
        f'MODEL_BASE_URL="{base_url}"\n'
        f'MODEL_NAME="{served}"\n'
        'MODEL_TEMPERATURE="0"\n'
        f"MODEL_EXTRA_BODY='{extra}'\n"
        f'MODEL_CONCURRENCY="{concurrency}"\n'
        f'MODEL_GPU_HOURLY="{gpu_hourly:g}"\n'
    )


def expected_parameters() -> dict[str, Any]:
    """extract's request_parameters for a model scored here."""
    return {
        "temperature": 0.0,
        "chat_template_kwargs": chat.THINKING_OFF,
        "max_tokens": chat.MAX_TOKENS,
    }


def run_releases(path: Path) -> list[str]:
    """The releases of a run's predictions, in order."""
    lines = (path / grade.PREDICTIONS).read_text().splitlines()
    return [json.loads(line)["release_accession"] for line in lines if line.strip()]


def kept_run(path: Path, model: Model, releases: Sequence[str]) -> bool:
    """Whether a complete graded run of model with these settings on exactly these
    releases is at path. Any other run there is an error: it is never overwritten."""
    if not path.exists():
        return False
    info_path = path / extract.RUN_INFO
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    if (
        info.get("status") == extract.COMPLETE
        and (path / grade.GRADED).is_file()
        and info.get("model") == model.served
        and info.get("prompt_version") == labels.PROMPT
        and info.get("request_parameters") == expected_parameters()
        and run_releases(path) == list(releases)
    ):
        return True
    raise ScoreError(
        f"{path} exists but isn't a complete graded run of {model.served} with these "
        f"settings on these {len(releases)} releases; move it away to score again"
    )


@dataclass
class Score:
    model: Model
    run_dir: Path
    correct: int
    graded: int
    by_field: dict[str, tuple[int, int]]  # field -> (correct, graded)
    not_correct: Counter[str]
    fallbacks: int  # releases with an all-null prediction (invalid reply, API error)
    finished: str  # run.json's finished time
    kept: bool  # scored in an earlier session, not called again in this one

    @property
    def accuracy(self) -> str:
        if not self.graded:
            return "0/0"
        return f"{self.correct}/{self.graded} ({100 * self.correct / self.graded:.1f}%)"


def read_score(model: Model, run_dir: Path, kept: bool) -> Score:
    with (run_dir / grade.GRADED).open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["verdict"] not in UNGRADED]
    info = json.loads((run_dir / extract.RUN_INFO).read_text())
    outcomes = info["outcomes"]

    def field(name: str) -> tuple[int, int]:
        mine = [r for r in rows if r["field"] == name]
        return sum(r["verdict"] == grade.CORRECT for r in mine), len(mine)

    return Score(
        model,
        run_dir,
        sum(r["verdict"] == grade.CORRECT for r in rows),
        len(rows),
        {name: field(name) for name in FIELD_NAMES},
        Counter(r["verdict"] for r in rows if r["verdict"] != grade.CORRECT),
        outcomes[extract.INVALID_REPLY] + outcomes[extract.API_ERROR],
        info["finished"],
        kept,
    )


def choose(scores: Sequence[Score]) -> Score | None:
    """The checkpoint with the most figures correct, ties to the earliest step; never
    the base model."""
    candidates = [s for s in scores if s.model.label != BASE]
    if not candidates:
        return None
    return max(candidates, key=lambda s: (s.correct, -s.model.step))


def format_report(
    run_dir: Path,
    scores: Sequence[Score],
    chosen: Score | None,
    header: Mapping[str, Any],
) -> str:
    def fraction(pair: tuple[int, int]) -> str:
        return f"{pair[0]}/{pair[1]}"

    checkpoints = sum(s.model.label != BASE for s in scores)
    kept = [s for s in scores if s.kept]
    lines = [
        f"# Validation: {run_dir.name}",
        "",
        (
            f"{header['base_model']} "
            + (
                f"and {checkpoints} checkpoints"
                if checkpoints
                else "alone (no checkpoints yet)"
            )
            + f" of {run_dir}, on {header['releases']} releases of {header['links']}; "
            f"prompt {labels.PROMPT}, temperature 0, thinking off, max_tokens "
            f"{chat.MAX_TOKENS}; served by {header['server']}."
        ),
        (
            f"Layout check: {header['layout_prompts']:,} prompts (validation releases "
            "and the run's training examples), token ids from the server's /tokenize "
            "identical to training's."
            if header["layout_prompts"]
            else "Layout check: not run (every model was already scored)."
        ),
    ]
    if kept:
        lines.append(
            "Kept from an earlier scoring session, not called again: "
            + ", ".join(f"{s.model.label} (scored {s.finished})" for s in kept)
            + f". Each session's server command is in {VALIDATION}/{SERVER_LOG}."
        )
    lines += [
        "",
        "| model | step | examples seen | scored | accuracy | "
        + " | ".join(FIELD_NAMES)
        + " | not correct | fallbacks | run |",
        "| --- " * (8 + len(FIELD_NAMES)) + "|",
    ]
    for s in scores:
        wrong = ", ".join(f"{v} {n}" for v, n in sorted(s.not_correct.items())) or "–"
        lines.append(
            f"| {s.model.label} | {s.model.step} | {s.model.examples_seen:,} | "
            f"{s.finished}{', kept' if s.kept else ''} | {s.accuracy} | "
            + " | ".join(fraction(s.by_field[f]) for f in FIELD_NAMES)
            + f" | {wrong} | {s.fallbacks} | {s.run_dir} |"
        )
    lines.append("")
    if chosen is None:
        lines.append("No checkpoint to choose.")
    else:
        lines.append(
            f"Chosen: {chosen.model.label}, {chosen.accuracy} ({CHOICE_RULE})."
        )
    return "\n".join(lines) + "\n"


def results_json(
    scores: Sequence[Score], chosen: Score | None, header: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **header,
        "settings": {
            "prompt": labels.PROMPT,
            "request_parameters": expected_parameters(),
        },
        "models": [
            {
                "label": s.model.label,
                "served": s.model.served,
                "step": s.model.step,
                "examples_seen": s.model.examples_seen,
                "run": str(s.run_dir),
                "scored": s.finished,  # its run.json's finished time
                "kept": s.kept,  # scored in an earlier session, not called again
                "correct": s.correct,
                "graded": s.graded,
                "by_field": {f: list(s.by_field[f]) for f in FIELD_NAMES},
                "not_correct": dict(s.not_correct),
                "fallbacks": s.fallbacks,
            }
            for s in scores
        ],
        "chosen": chosen.model.label if chosen else None,
        "rule": CHOICE_RULE,
    }


ServerStarter = Callable[
    [Sequence[str], Path, str, httpx.Client], AbstractContextManager
]


def serve_and_score(
    to_score: Sequence[Model],
    command: Sequence[str],
    folder: Path,
    root: str,
    base_name: str,
    prompts: Sequence[tuple[str, str, str]],
    encode: Callable[[str], list[int]],
    transport: httpx.BaseTransport | None,
    start_server: ServerStarter,
    out: Callable[[str], None],
    server_name: str,
    extract_args: Sequence[str],
    concurrency: int,
    gpu_hourly: float,
    edgar_client: EdgarClient,
    env_prefix: str = "",
    stop_on_failure: bool = False,
) -> list[str]:
    """Start the server, check the layout, run extract on each model; the labels of
    the models whose run failed (API errors, stopped, not graded). Each model's env
    file is <folder>/<env_prefix><label>.env. With stop_on_failure, no model is run
    after one fails."""
    failed = []
    with httpx.Client(transport=transport) as http:
        out(f"starting {server_name}: {' '.join(command)}")
        with start_server(command, folder / SERVER_LOG, root, http):
            out(f"checking {len(prompts):,} prompts against the server's /tokenize")
            problems = layout_problems(http, root, base_name, prompts, encode)
            if problems:
                raise ScoreError(
                    f"{len(problems)} prompts aren't tokenized as in training: "
                    + "; ".join(problems[:5])
                )
            for model in to_score:
                if failed and stop_on_failure:
                    break
                env_path = folder / f"{env_prefix}{model.label}.env"
                write_text(
                    env_path,
                    env_text(f"{root}/v1", model.served, concurrency, gpu_hourly),
                )
                argv = [
                    "--env-file", str(env_path),
                    "--run-name", model.run_name,
                    *extract_args,
                ]  # fmt: skip
                out(f"scoring {model.label} ({model.served})")
                if extract.main(argv, transport=transport, edgar_client=edgar_client):
                    failed.append(model.label)
    return failed


def score(
    run_dir: Path,
    *,
    server: str,
    port: int,
    links: Path,
    answers: Path,
    releases: Path,
    runs_dir: Path,
    limit: int | None,
    concurrency: int,
    gpu_hourly: float,
    snapshot: Path,
    encode: Callable[[str], list[int]],
    edgar_client: EdgarClient,
    transport: httpx.BaseTransport | None = None,
    start_server: ServerStarter = run_server,
    out: Callable[[str], None] = print,
) -> int:
    info, config, _ = train.read_run(run_dir)
    every = models(run_dir, config.model)
    adapters = every[1:]
    check_adapters(adapters, config.r)
    root = f"http://127.0.0.1:{port}"
    folder = run_dir / VALIDATION
    folder.mkdir(exist_ok=True)
    linked = load_links(links)[0]
    selected = [
        r.release_accession for r in extract.select_releases(linked, limit=limit)
    ]
    to_score = [m for m in every if not kept_run(runs_dir / m.run_name, m, selected)]
    kept = [m for m in every if m not in to_score]
    if kept:
        out(f"already scored, kept: {', '.join(m.label for m in kept)}")
    prompt = extract.load_prompt(labels.PROMPT)
    prompts = (
        layout_prompts(info, config, run_dir, links, releases, prompt.text)
        if to_score
        else []
    )
    make = vllm_command if server == VLLM else hf_command
    command = make(snapshot, config.model, adapters, port, config.r)
    header = {
        "run": str(run_dir),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_model": config.model,
        "revision": config.revision,
        "server": server_version(server),
        "server_command": command,
        "links": str(links),
        "releases": len(selected),
        "layout_prompts": len(prompts) if to_score else None,
    }
    failed = []
    extract_args = [
        "--prompt", labels.PROMPT,
        "--runs-dir", str(runs_dir),
        "--links", str(links),
        "--answers", str(answers),
        "--releases", str(releases),
        *(["--limit", str(limit)] if limit else []),
    ]  # fmt: skip
    if to_score:
        failed = serve_and_score(
            to_score,
            command,
            folder,
            root,
            config.model,
            prompts,
            encode,
            transport,
            start_server,
            out,
            header["server"],
            extract_args,
            concurrency,
            gpu_hourly,
            edgar_client,
        )
    scores = [
        read_score(m, runs_dir / m.run_name, m in kept)
        for m in every
        if (runs_dir / m.run_name / grade.GRADED).is_file()
    ]
    chosen = choose(scores)
    report = format_report(run_dir, scores, chosen, header)
    write_text(run_dir / REPORT, report)
    extract.write_json(run_dir / RESULTS, results_json(scores, chosen, header))
    out(report)
    if failed:
        out(
            f"error: {', '.join(failed)}: API errors, a stopped run or a grading "
            "error (see each run's summary above)"
        )
        return 1
    return 0


# Test splits (results/test_protocol.md)

TEST = "test"  # folder in the run: each split's results and the server log
# The test config's tables: key -> type.
TEST_TABLES: dict[str, dict[str, type]] = {
    "model": {
        "run": str,
        "base": str,
        "revision": str,
        "checkpoint": str,
        "adapter_sha256": str,
    },
    "server": {
        "max_model_len": int,
        "max_lora_rank": int,
        "max_loras": int,
        "max_cpu_loras": int,
    },
    "request": {
        "prompt": str,
        "temperature": float,
        "enable_thinking": bool,
        "max_tokens": int,
        "concurrency": int,
    },
}


@dataclass(frozen=True)
class TestConfig:
    """configs/test/<run>.toml: the checkpoint under test, and the server and request
    settings, which must be validation's."""

    __test__ = False  # not a pytest class

    path: Path
    sha256: str  # the file's
    run: str
    base: str
    revision: str
    checkpoint: str
    adapter_sha256: str
    max_model_len: int
    max_lora_rank: int
    max_loras: int
    max_cpu_loras: int
    prompt: str
    temperature: float
    enable_thinking: bool
    max_tokens: int
    concurrency: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_test_config(path: Path) -> TestConfig:
    """The test config; every key is required and no other is allowed."""
    if not path.is_file():
        raise ScoreError(f"{path} not found")
    raw = path.read_bytes()
    try:
        tables = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ScoreError(f"{path}: {exc}") from None
    problems = [f"unknown table [{name}]" for name in tables if name not in TEST_TABLES]
    values: dict[str, Any] = {}
    for name, keys in TEST_TABLES.items():
        table = tables.get(name)
        if not isinstance(table, dict):
            problems.append(f"no [{name}] table")
            continue
        problems += [f"unknown key {name}.{k}" for k in table if k not in keys]
        for key, kind in keys.items():
            if key not in table:
                problems.append(f"{name}.{key} not set")
                continue
            value = table[key]
            if kind is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if not isinstance(value, kind) or isinstance(value, bool) != (kind is bool):
                problems.append(f"{name}.{key} must be a {kind.__name__}")
                continue
            values[key] = value
    if problems:
        raise ScoreError(f"{path}: " + "; ".join(problems))
    return TestConfig(path=path, sha256=hashlib.sha256(raw).hexdigest(), **values)


def settings_problems(
    test: TestConfig, run_dir: Path, config: train.Config
) -> list[str]:
    """Each way the test config differs from the training run (its name and base
    model) or from validation's settings."""
    expected: dict[str, Any] = {
        "run": run_dir.name,
        "base": config.model,
        "revision": config.revision,
        "max_model_len": chat.MAX_MODEL_LEN,
        "max_lora_rank": lora_rank(config.r),
        "max_loras": 1,
        "max_cpu_loras": 1,
        "prompt": labels.PROMPT,
        "temperature": 0.0,
        "enable_thinking": False,
        "max_tokens": chat.MAX_TOKENS,
    }
    problems = [
        f"{key} is {getattr(test, key)!r}, not {value!r}"
        for key, value in expected.items()
        if getattr(test, key) != value
    ]
    if test.concurrency < 1:
        problems.append("concurrency must be 1 or more")
    return problems


def checkpoint_under_test(
    run_dir: Path, test: TestConfig, config: train.Config, split: str
) -> Model:
    """The test config's checkpoint, after checking its adapter's sha256 and r."""
    path = run_dir / train.CHECKPOINTS / test.checkpoint
    found = {c.path.name: c for c in train.checkpoints(run_dir)}
    if test.checkpoint not in found:
        raise ScoreError(f"{path}: no complete checkpoint (with {train.STATE})")
    weights = path / train.ADAPTER_WEIGHTS
    if not weights.is_file():
        raise ScoreError(f"{weights} not found")
    digest = file_sha256(weights)
    if digest != test.adapter_sha256:
        raise ScoreError(
            f"{weights} has sha256 {digest}, not the test config's "
            f"{test.adapter_sha256}"
        )
    state = json.loads((path / train.STATE).read_text())
    model = Model(
        test.checkpoint,
        f"{run_dir.name}--{test.checkpoint}",  # as on validation
        f"{run_dir.name}--{split}--{test.checkpoint}",
        found[test.checkpoint].step,
        state["examples_seen"],
        path.resolve(),
    )
    check_adapters([model], config.r)
    return model


@dataclass(frozen=True)
class SplitFiles:
    """data/splits/<split>/, checked against its manifest."""

    name: str
    folder: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    links: Path
    answers: Path
    releases: Path  # the release files' folder
    inputs: dict[str, str]  # each kept release's input sha256 (tokens.csv)


def load_split(name: str, splits_dir: Path, root: Path = Path()) -> SplitFiles:
    """A frozen split's files; the releases folder is relative to root."""
    folder = root / splits_dir / name
    path = folder / splits.MANIFEST
    if not path.is_file():
        raise ScoreError(f"{path} not found; run secbench.splits")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("status") != splits.FROZEN:
        raise ScoreError(
            f"{name} is {manifest.get('status')}, not {splits.FROZEN}: only a frozen "
            "split is scored (secbench.splits --freeze)"
        )
    for file, digest in manifest["files"].items():
        if not (folder / file).is_file() or file_sha256(folder / file) != digest:
            raise ScoreError(f"{folder / file} isn't the file {path} records")
    tokens = splits.read_table(folder / splits.TOKENS, splits.TOKEN_COLUMNS)
    return SplitFiles(
        name=name,
        folder=folder,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        links=folder / splits.LINKS,
        answers=folder / splits.ANSWERS,
        releases=root / manifest["releases_dir"],
        inputs={
            r["release_accession"]: r["input_sha256"]
            for r in tokens
            if r["status"] == splits.KEPT
        },
    )


def split_prompts(split: SplitFiles, prompt: str) -> list[tuple[str, str, str]]:
    """(label, system, user) of each release of the split; each input must be the
    one tokens.csv records."""
    records = release_records(split.releases)
    prompts = []
    for release in load_links(split.links)[0]:
        accession = release.release_accession
        record = records.get(accession)
        if record is None:
            raise ScoreError(f"no saved release {accession} in {split.releases}")
        text = build_input(record).text
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != split.inputs.get(
            accession
        ):
            raise ScoreError(
                f"{accession}: its input isn't the one {split.folder / splits.TOKENS} "
                "records"
            )
        prompts.append((f"{split.name} {accession}", prompt, text))
    if len(prompts) != len(split.inputs):
        raise ScoreError(
            f"{split.links} and {split.folder / splits.TOKENS} list other releases"
        )
    return prompts


def prompt_lengths(
    prompts: Sequence[tuple[str, str, str]], encode: Callable[[str], list[int]]
) -> list[tuple[str, int]]:
    """Each prompt's tokens as the server will count them (secbench.chat's layout)."""
    return [
        (label, len(encode(chat.prompt_text(system, user))))
        for label, system, user in prompts
    ]


def gpus() -> list[str] | None:
    """nvidia-smi's name, memory and driver of each GPU; None without nvidia-smi."""
    try:
        listed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in listed.splitlines() if line.strip()]


def next_attempt(path: Path) -> Path:
    """<path>--attempt-N, the first N not taken."""
    n = 1
    while path.with_name(f"{path.name}--attempt-{n}").exists():
        n += 1
    return path.with_name(f"{path.name}--attempt-{n}")


def complete_without_errors(info: Mapping[str, Any]) -> bool:
    outcomes = info.get("outcomes", {})
    return (
        info.get("status") == extract.COMPLETE
        and outcomes.get(extract.API_ERROR) == 0
        and outcomes.get(extract.ABORTED) == 0
    )


def kept_test_run(
    path: Path,
    model: Model,
    releases: Sequence[str],
    links_sha256: str,
    concurrency: int,
) -> bool:
    """Whether a complete run with no API error of model, with the test settings, on
    exactly these releases of the split's links file, is at path. Any other run there
    is an error: it is never overwritten."""
    if not path.exists():
        return False
    info_path = path / extract.RUN_INFO
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    if (
        complete_without_errors(info)
        and (path / grade.PREDICTIONS).is_file()
        and info.get("model") == model.served
        and info.get("prompt_version") == labels.PROMPT
        and info.get("request_parameters") == expected_parameters()
        and info.get("max_parallel") == concurrency
        and info.get("links_sha256") == links_sha256
        and run_releases(path) == list(releases)
    ):
        return True
    raise ScoreError(
        f"{path} exists but isn't a complete run of {model.served} with no API error, "
        f"these settings and these {len(releases)} releases; move it to "
        f"{next_attempt(path)} and score again"
    )


def run_record(model: Model, path: Path, kept: bool) -> dict[str, Any]:
    info = json.loads((path / extract.RUN_INFO).read_text())
    return {
        "label": model.label,
        "served": model.served,
        "run": str(path),
        "kept": kept,  # made in an earlier session, not called again
        "status": info["status"],
        "complete_without_errors": complete_without_errors(info),
        "outcomes": info["outcomes"],
        "started": info["started"],
        "finished": info["finished"],
        "wall_time_s": info["wall_time_s"],
        "estimated_cost_usd": info["estimated_cost_usd"],
    }


def score_test(
    run_dir: Path,
    *,
    test_config: Path,
    split: str,
    server: str,
    port: int,
    runs_dir: Path,
    limit: int | None,
    gpu_hourly: float,
    snapshot: Path,
    encode: Callable[[str], list[int]],
    splits_dir: Path = splits.DEFAULT_OUT,
    transport: httpx.BaseTransport | None = None,
    start_server: ServerStarter = run_server,
    out: Callable[[str], None] = print,
) -> int:
    """Predictions of the base model and the test config's checkpoint on a frozen
    split (see the module docstring)."""
    _, config, _ = train.read_run(run_dir)
    test = load_test_config(test_config)
    problems = settings_problems(test, run_dir, config)
    if problems:
        raise ScoreError(f"{test_config}: " + "; ".join(problems))
    adapter = checkpoint_under_test(run_dir, test, config, split)
    base = Model(BASE, config.model, f"{run_dir.name}--{split}--{BASE}", 0, 0)
    every = [base, adapter]
    files = load_split(split, splits_dir)
    prompt = extract.load_prompt(labels.PROMPT)
    if prompt.sha256 != files.manifest["prompt"]["sha256"]:
        raise ScoreError(
            f"{labels.PROMPT}'s sha256 isn't the one {files.folder} was built with"
        )
    prompts = split_prompts(files, prompt.text)
    lengths = prompt_lengths(prompts, encode)
    over = [(label, n) for label, n in lengths if n > splits.MAX_PROMPT_TOKENS]
    if over:
        raise ScoreError(
            f"{len(over)} prompts over {splits.MAX_PROMPT_TOKENS:,} tokens: "
            + "; ".join(f"{label} ({n:,})" for label, n in over[:5])
        )
    selected = [
        r.release_accession
        for r in extract.select_releases(load_links(files.links)[0], limit=limit)
    ]
    links_sha256 = file_sha256(files.links)
    to_score = [
        m
        for m in every
        if not kept_test_run(
            runs_dir / m.run_name, m, selected, links_sha256, test.concurrency
        )
    ]
    kept = [m for m in every if m not in to_score]
    if kept:
        out(f"already made, kept: {', '.join(m.run_name for m in kept)}")
    folder = run_dir / TEST
    folder.mkdir(exist_ok=True)
    make = vllm_command if server == VLLM else hf_command
    command = make(snapshot, config.model, [adapter], port, config.r)
    extract_args = [
        "--prompt", labels.PROMPT,
        "--runs-dir", str(runs_dir),
        "--links", str(files.links),
        "--answers", str(files.answers),
        "--releases", str(files.releases),
        "--no-grade",
        *(["--limit", str(limit)] if limit else []),
    ]  # fmt: skip
    failed = []
    if to_score:
        failed = serve_and_score(
            to_score,
            command,
            folder,
            f"http://127.0.0.1:{port}",
            config.model,
            prompts,
            encode,
            transport,
            start_server,
            out,
            server_version(server),
            extract_args,
            test.concurrency,
            gpu_hourly,
            EdgarClient("offline: no requests", offline=True),  # never read
            env_prefix=f"{split}--",
            stop_on_failure=True,
        )
    runs = [
        run_record(m, runs_dir / m.run_name, m in kept)
        for m in every
        if (runs_dir / m.run_name / extract.RUN_INFO).is_file()
    ]
    record = {
        "run": str(run_dir),
        "split": split,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        **extract.code_version(),
        "test_config": {"path": str(test_config), "sha256": test.sha256},
        "base_model": config.model,
        "revision": config.revision,
        "checkpoint": {
            "label": adapter.label,
            "served": adapter.served,
            "step": adapter.step,
            "examples_seen": adapter.examples_seen,
            "adapter_sha256": test.adapter_sha256,
        },
        "split_files": {
            "folder": str(files.folder),
            "manifest_sha256": files.manifest_sha256,
            "status": files.manifest["status"],
            "built_at": files.manifest["commit"],
            "files": files.manifest["files"],
            "releases_dir": str(files.releases),
        },
        "releases": len(selected),
        "limit": limit,
        "server": server_version(server),
        "server_command": command,
        "gpus": gpus(),  # for the cost's GPU type
        "layout_check": (
            {
                "prompts": len(prompts),
                "longest_prompt_tokens": max(n for _, n in lengths),
                "identical_to_training_layout": True,
            }
            if to_score
            else None  # every run was kept: no server
        ),
        "settings": {
            "prompt": labels.PROMPT,
            "request_parameters": expected_parameters(),
            "concurrency": test.concurrency,
            "gpu_hourly_usd": gpu_hourly,
        },
        "runs": runs,
        "failed": failed,
    }
    extract.write_json(folder / f"{split}.json", record)
    for run in runs:
        out(
            f"{run['label']}: {run['run']}"
            + (" (kept)" if run["kept"] else "")
            + f", {run['status']}, outcomes "
            + ", ".join(f"{k} {v}" for k, v in run["outcomes"].items() if v)
        )
    out(f"wrote {folder / f'{split}.json'}")
    if failed:
        path = runs_dir / every[[m.label for m in every].index(failed[0])].run_name
        out(
            f"error: {path} had API errors or stopped, and nothing was run after it; "
            f"move it to {next_attempt(path)} and score again"
        )
        return 1
    out(
        "grade each run on the laptop: uv run python -m secbench.grade runs/<run> "
        f"--links {files.links} --answers {files.answers} --releases {files.releases}"
    )
    return 0


def main(
    argv: list[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    edgar_client: EdgarClient | None = None,
    start_server: ServerStarter = run_server,
    snapshot: Path | None = None,
    encode: Callable[[str], list[int]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.score",
        description="Score a training run's checkpoints on validation and choose one; "
        "or, with --test-config and --split, make the predictions of the base model "
        "and one checkpoint on a frozen test split.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--server", choices=(VLLM, HF), default=VLLM)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--limit", type=extract.positive_int, help="the first N releases"
    )
    parser.add_argument(
        "--concurrency",
        type=extract.positive_int,
        help=f"validation only (default {DEFAULT_CONCURRENCY}); a test split's is "
        "the test config's",
    )
    parser.add_argument(
        "--gpu-hourly",
        type=float,
        help="dollars per GPU-hour, for the cost (default 1 on validation; required "
        "with --test-config: the price paid)",
    )
    parser.add_argument("--runs-dir", type=Path, default=extract.RUNS_DIR)
    parser.add_argument("--hf-cache", type=Path, default=train.DEFAULT_HF_CACHE)
    parser.add_argument("--links", type=Path, help=f"validation only ({DEFAULT_LINKS})")
    parser.add_argument(
        "--answers", type=Path, help=f"validation only ({DEFAULT_ANSWERS})"
    )
    parser.add_argument(
        "--releases", type=Path, help=f"validation only ({DEFAULT_RELEASES})"
    )
    parser.add_argument(
        "--test-config", type=Path, help="configs/test/<run>.toml; give --split too"
    )
    parser.add_argument("--split", choices=splits.SPLITS, help="a frozen split")
    args = parser.parse_args(argv)
    testing = args.test_config is not None
    if testing != (args.split is not None):
        parser.error("--test-config and --split go together")
    if testing:
        given = [
            f"--{name}"
            for name in ("links", "answers", "releases", "concurrency")
            if getattr(args, name) is not None
        ]
        if given:
            parser.error(
                f"{', '.join(given)}: a test split's come from data/splits/<split>/ "
                "and the test config"
            )
        if args.gpu_hourly is None:
            parser.error(
                "--test-config needs --gpu-hourly, the price paid per GPU-hour"
            )
    if args.gpu_hourly is not None and not (
        math.isfinite(args.gpu_hourly) and args.gpu_hourly >= 0
    ):
        parser.error("--gpu-hourly must be 0 or more")
    try:
        _, config, _ = train.read_run(args.run_dir)
        if snapshot is None:
            snapshot = train.snapshot_path(config, args.hf_cache).resolve()
        if encode is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(snapshot)

            def encode(text: str) -> list[int]:
                return train.encode(tokenizer, text)

        if testing:
            return score_test(
                args.run_dir,
                test_config=args.test_config,
                split=args.split,
                server=args.server,
                port=args.port,
                runs_dir=args.runs_dir,
                limit=args.limit,
                gpu_hourly=args.gpu_hourly,
                snapshot=snapshot,
                encode=encode,
                transport=transport,
                start_server=start_server,
            )
        return score(
            args.run_dir,
            server=args.server,
            port=args.port,
            links=args.links or DEFAULT_LINKS,
            answers=args.answers or DEFAULT_ANSWERS,
            releases=args.releases or DEFAULT_RELEASES,
            runs_dir=args.runs_dir,
            limit=args.limit,
            concurrency=args.concurrency or DEFAULT_CONCURRENCY,
            gpu_hourly=1.0 if args.gpu_hourly is None else args.gpu_hourly,
            snapshot=snapshot,
            encode=encode,
            edgar_client=edgar_client
            or EdgarClient("offline: no requests", offline=True),
            transport=transport,
            start_server=start_server,
        )
    except (
        ScoreError,
        train.TrainError,
        extract.RunError,
        splits.SplitError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
