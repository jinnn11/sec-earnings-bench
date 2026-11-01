"""Score a training run's checkpoints on the validation releases and choose one (PLAN.md
Decisions 2026-10-27: checkpoints are chosen on validation only).

Usage: uv run --group train --group serve python -m secbench.score training/<run>
           [--server vllm|hf] [--port 8000] [--limit N] [--runs-dir runs]
           [--concurrency 16] [--gpu-hourly 1] [--hf-cache data/raw/hf-hub]

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
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx

from secbench import chat, extract, grade, labels, train
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
) -> list[str]:
    """Start the server, check the layout, run extract on each model; the labels of
    the models whose run failed (API errors, stopped, not graded)."""
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
                env_path = folder / f"{model.label}.env"
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
        description="Score a training run's checkpoints on validation and choose one.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--server", choices=(VLLM, HF), default=VLLM)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--limit", type=extract.positive_int, help="the first N releases"
    )
    parser.add_argument(
        "--concurrency", type=extract.positive_int, default=DEFAULT_CONCURRENCY
    )
    parser.add_argument(
        "--gpu-hourly", type=float, default=1.0, help="dollars per GPU-hour (cost)"
    )
    parser.add_argument("--runs-dir", type=Path, default=extract.RUNS_DIR)
    parser.add_argument("--hf-cache", type=Path, default=train.DEFAULT_HF_CACHE)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--releases", type=Path, default=DEFAULT_RELEASES)
    args = parser.parse_args(argv)
    try:
        _, config, _ = train.read_run(args.run_dir)
        if snapshot is None:
            snapshot = train.snapshot_path(config, args.hf_cache).resolve()
        if encode is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(snapshot)

            def encode(text: str) -> list[int]:
                return train.encode(tokenizer, text)

        return score(
            args.run_dir,
            server=args.server,
            port=args.port,
            links=args.links,
            answers=args.answers,
            releases=args.releases,
            runs_dir=args.runs_dir,
            limit=args.limit,
            concurrency=args.concurrency,
            gpu_hourly=args.gpu_hourly,
            snapshot=snapshot,
            encode=encode,
            edgar_client=edgar_client
            or EdgarClient("offline: no requests", offline=True),
            transport=transport,
            start_server=start_server,
        )
    except (ScoreError, train.TrainError, extract.RunError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
