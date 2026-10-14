"""Run a model on the linked releases, save everything it returns, and grade it.

Each linked release in data/links.csv (see secbench.link) goes to an OpenAI-compatible
chat completions API at temperature 0: the instructions in prompts/<version>.txt as
the system message, the release's input (see secbench.inputs) as the user message.
The reply must be one JSON object of a Prediction (see secbench.grade) without
release_accession. An invalid reply is retried once with its validation errors. If
the retry is invalid too, or the API fails, the release gets an all-null prediction
with links.csv's period end, which grades as missing; run.json and the printed
summary list every such fallback and why. A released prompt is never edited; a
changed prompt is a new version (extract_v2).

Usage: uv run python -m secbench.extract [--tickers AAPL ...] [--accessions ...]
           [--limit N] [--run-name NAME] [--prompt extract_v1] [--dry-run]
           [--links] [--answers] [--releases] [--runs-dir] [--prompts-dir]

Settings come from the environment or .env: MODEL_BASE_URL (such as
http://localhost:8000/v1), MODEL_NAME, MODEL_API_KEY (optional), and
MODEL_PRICE_INPUT and MODEL_PRICE_OUTPUT (optional; dollars per million tokens).
--dry-run prints the full request for the first selected release, without reading
them or making a call.

A run writes runs/<run>/raw/<accession>.json (every HTTP attempt's request body and
response; no headers) as each release finishes, then predictions.jsonl and run.json
(model, prompt version, token counts, latency per call, estimated cost, fallbacks),
grades the run (see secbench.grade) and prints the summary. At most 4 calls run at a
time; 429, 5xx, timeouts and connection errors are retried with backoff. HTTP 401,
403 or 404 (wrong key, model or URL) stops the run without grading it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Self

import httpx
from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError

from secbench import grade
from secbench.answers import DEFAULT_OUT as DEFAULT_ANSWERS
from secbench.answers import LinkedRelease, load_links, plural
from secbench.edgar import (
    MAX_RETRIES,
    MAX_RETRY_WAIT,
    EdgarClient,
    normalize_ticker,
    parse_retry_after,
)
from secbench.grade import Figures, Prediction
from secbench.inputs import ReleaseInput, build_input, release_records
from secbench.link import DEFAULT_OUT as DEFAULT_LINKS
from secbench.link import shorten
from secbench.releases import DEFAULT_OUT_DIR as DEFAULT_RELEASES
from secbench.releases import write_text

log = logging.getLogger(__name__)

PROMPTS_DIR = Path("prompts")
DEFAULT_PROMPT = "extract_v1"
RUNS_DIR = Path("runs")
RAW = "raw"
RUN_INFO = "run.json"
MAX_PARALLEL = 4
TEMPERATURE = 0
# Generous: a local server can take minutes on a 70,000-character release.
TIMEOUT = httpx.Timeout(600.0, connect=10.0)
ABORT_STATUSES = (401, 403, 404)  # wrong key, model or URL: every call would fail
RETRY_MESSAGE = (
    "Your reply did not validate: {errors}\nReply with only the corrected JSON object."
)
ERROR_WIDTH = 200  # characters of a provider's error message kept in a reason

OK = "ok"
OK_AFTER_RETRY = "ok_after_retry"
INVALID_REPLY = "invalid_reply"
API_ERROR = "api_error"
ABORTED = "aborted"  # the call that stopped the run; no prediction
OUTCOMES = (OK, OK_AFTER_RETRY, INVALID_REPLY, API_ERROR)

COMPLETE = "complete"  # run status; or ABORTED

SETTINGS = (
    "MODEL_BASE_URL",
    "MODEL_NAME",
    "MODEL_API_KEY",
    "MODEL_PRICE_INPUT",
    "MODEL_PRICE_OUTPUT",
)
# A reasoning model's leading <think>...</think> block, and a ``` or ```json fence
# around the whole reply, are not part of the JSON.
THINK = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL)
FENCE = re.compile(r"```(?:json)?[ \t]*\n(.*?)\n?[ \t]*```", re.DOTALL | re.IGNORECASE)
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # a prompt version or run name


class RunError(Exception):
    """A run can't start; the message says why."""


# Settings and prompt


class SettingsError(RunError):
    """A MODEL_* setting is missing or malformed. The message names the setting,
    never its value."""


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    api_key: str = ""  # blank: no Authorization header (a local server)
    price_input: float | None = None  # dollars per million tokens
    price_output: float | None = None

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


def parse_price(name: str, value: str) -> float | None:
    if not value:
        return None
    try:
        price = float(value)
    except ValueError:
        raise SettingsError(f"{name} is not a number") from None
    if not math.isfinite(price) or price < 0:
        raise SettingsError(f"{name} must be 0 or more dollars per million tokens")
    return price


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """The MODEL_* settings from environ, or else from the environment and .env
    (the environment wins)."""
    if environ is None:
        load_dotenv(find_dotenv(usecwd=True))
        environ = os.environ
    values = {name: environ.get(name, "").strip() for name in SETTINGS}
    missing = [name for name in SETTINGS[:2] if not values[name]]
    if missing:
        raise SettingsError(
            f"{' and '.join(missing)} not set; add to .env (see .env.example)"
        )
    if not re.match(r"https?://", values["MODEL_BASE_URL"], re.IGNORECASE):
        raise SettingsError("MODEL_BASE_URL must start with http:// or https://")
    return Settings(
        base_url=values["MODEL_BASE_URL"],
        model=values["MODEL_NAME"],
        api_key=values["MODEL_API_KEY"],
        price_input=parse_price("MODEL_PRICE_INPUT", values["MODEL_PRICE_INPUT"]),
        price_output=parse_price("MODEL_PRICE_OUTPUT", values["MODEL_PRICE_OUTPUT"]),
    )


@dataclass(frozen=True)
class Prompt:
    version: str  # the file name without .txt: "extract_v1"
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def load_prompt(version: str, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    if not NAME.fullmatch(version):
        raise RunError(f"bad prompt version {version!r}")
    path = prompts_dir / f"{version}.txt"
    if not path.is_file():
        raise RunError(f"{path} not found")
    return Prompt(version, path.read_text(encoding="utf-8"))


# Requests and replies


def first_messages(prompt: Prompt, input_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": input_text},
    ]


def retry_messages(
    messages: list[dict[str, str]], reply: str, errors: str
) -> list[dict[str, str]]:
    """The conversation so far, the invalid reply, and what was wrong with it."""
    return [
        *messages,
        {"role": "assistant", "content": reply},
        {"role": "user", "content": RETRY_MESSAGE.format(errors=errors)},
    ]


def request_body(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {"model": model, "messages": messages, "temperature": TEMPERATURE}


class ReplyError(ValueError):
    """A reply isn't a valid prediction. The message says why, for the model."""


def reply_json(content: str) -> str:
    """The reply without a leading <think>...</think> block or a surrounding fence."""
    text = THINK.sub("", content, count=1).strip()
    if fenced := FENCE.fullmatch(text):
        return fenced[1].strip()
    return text


def shown(value: Any) -> str | None:
    """A scalar as JSON would show it; None for an object or a list."""
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, str):
        return shorten(json.dumps(value), 40)
    if isinstance(value, int | float | Decimal):
        return str(value)
    return None


def describe_errors(exc: ValidationError) -> str:
    """One part per error: 'quarter.revenue: Input should be a valid decimal (got
    "94.9B")'."""
    parts = []
    for error in exc.errors(include_url=False):
        where = ".".join(map(str, error["loc"]))
        part = f"{where}: {error['msg']}" if where else error["msg"]
        if got := shown(error.get("input")):  # None for a missing field's parent
            part += f" (got {got})"
        parts.append(part)
    return "; ".join(parts)


def parse_reply(content: str | None, accession: str) -> Prediction:
    """The prediction in a reply: one JSON object of a Prediction without
    release_accession. Figures are read as exact decimals."""
    if not content or not content.strip():
        raise ReplyError("the reply is empty")
    try:
        data = json.loads(reply_json(content), parse_float=Decimal)
    except json.JSONDecodeError as exc:
        raise ReplyError(f"not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise ReplyError("not a JSON object")
    if "release_accession" in data:
        raise ReplyError("release_accession: Extra inputs are not permitted")
    try:
        return Prediction.model_validate({**data, "release_accession": accession})
    except ValidationError as exc:
        raise ReplyError(describe_errors(exc)) from None


def null_prediction(release: LinkedRelease) -> Prediction:
    """Every figure null, with links.csv's period end (a Prediction needs one)."""
    return Prediction(
        release_accession=release.release_accession,
        period_end=release.period_end,
        quarter=Figures(),
        year_to_date=Figures(),
    )


# HTTP


def count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass
class Exchange:
    """One chat completion request and every HTTP attempt it took. response is the
    successful body; without one, error says why, and abort whether the run must
    stop."""

    request: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)
    response: dict[str, Any] | None = None
    error: str = ""
    abort: bool = False

    @property
    def content(self) -> str | None:
        if self.response is None:
            return None
        content = self.response["choices"][0]["message"].get("content")
        return content if isinstance(content, str) else None

    @property
    def finish_reason(self) -> str | None:
        if self.response is None:
            return None
        return self.response["choices"][0].get("finish_reason")

    @property
    def tokens(self) -> tuple[int | None, int | None]:
        """(input, output) tokens from the response's usage, if it has them."""
        usage = (self.response or {}).get("usage")
        if not isinstance(usage, dict):
            return None, None
        return count(usage.get("prompt_tokens")), count(usage.get("completion_tokens"))

    @property
    def latency_s(self) -> float | None:
        """The last HTTP attempt's time, without earlier attempts or waits."""
        return self.attempts[-1]["latency_s"] if self.attempts else None


def response_body(response: httpx.Response) -> Any:
    """The body as JSON, or as text if it isn't JSON."""
    try:
        return response.json()
    except ValueError:
        return response.text


def is_completion(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    choices = body.get("choices")
    return (
        isinstance(choices, list)
        and bool(choices)
        and isinstance(choices[0], dict)
        and isinstance(choices[0].get("message"), dict)
    )


def error_message(body: Any) -> str:
    """A provider's error message: {"error": {"message": ...}} (OpenAI),
    {"message": ...} (vLLM) or {"error": "..."}."""
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, dict):
        error = error.get("message")
    message = error if isinstance(error, str) else body.get("message")
    return shorten(message.strip(), ERROR_WIDTH) if isinstance(message, str) else ""


def backoff(attempt: int) -> float:
    base = 2.0**attempt  # 1, 2, 4, 8, 16 s
    return base + random.uniform(0, 0.25 * base)


class ChatClient:
    """Posts chat completions to the settings' endpoint. Retries 429, 5xx, timeouts
    and connection errors up to max_retries times (1-16 s backoff, Retry-After
    honored; asked to wait over 120 s, it gives up). Safe to share between
    threads."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._endpoint = settings.endpoint
        self._http = httpx.Client(headers=headers, timeout=TIMEOUT, transport=transport)
        self._sleep = sleep
        self._clock = clock
        self._wall_clock = wall_clock
        self.max_retries = max_retries

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete(self, body: dict[str, Any]) -> Exchange:
        exchange = Exchange(body)
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            retry_after = None
            start = self._clock()
            try:
                response = self._http.post(self._endpoint, json=body)
            except httpx.TransportError as exc:
                problem = f"{type(exc).__name__}: {exc}"
                exchange.attempts.append(
                    {"status": None, "latency_s": self._since(start), "error": problem}
                )
            else:
                status = response.status_code
                record: dict[str, Any] = {
                    "status": status,
                    "latency_s": self._since(start),
                }
                exchange.attempts.append(record)
                if status in ABORT_STATUSES:
                    # Not kept: the body can echo part of the key (OpenAI's 401 does).
                    record["response"] = None
                    exchange.error = (
                        f"HTTP {status}; check MODEL_BASE_URL, MODEL_NAME and "
                        "MODEL_API_KEY"
                    )
                    exchange.abort = True
                    return exchange
                record["response"] = response_body(response)
                if response.is_success:
                    if is_completion(record["response"]):
                        exchange.response = record["response"]
                    else:
                        exchange.error = f"HTTP {status} without choices[0].message"
                    return exchange
                problem = f"HTTP {status}"
                if message := error_message(record["response"]):
                    problem += f" ({message})"
                if status != 429 and status < 500:
                    exchange.error = problem
                    return exchange
                retry_after = parse_retry_after(
                    response.headers.get("Retry-After"), self._wall_clock()
                )
            if attempt + 1 == attempts:
                exchange.error = f"{problem}, after {attempts} attempts"
                return exchange
            delay = retry_after if retry_after is not None else backoff(attempt)
            if delay > MAX_RETRY_WAIT:
                exchange.error = f"{problem}, and the server asked to wait {delay:.0f}s"
                return exchange
            log.warning(
                "%s; retry %d/%d in %.1fs",
                problem,
                attempt + 1,
                self.max_retries,
                delay,
            )
            self._sleep(delay)
        raise AssertionError("unreachable")

    def _since(self, start: float) -> float:
        return round(self._clock() - start, 3)


# One release


@dataclass
class ReleaseResult:
    release: LinkedRelease
    exhibits: tuple[str, ...]  # the exhibits in the input
    prediction: Prediction | None  # None only when aborted
    outcome: str  # one of OUTCOMES, or ABORTED
    reason: str = ""  # why it fell back (or stopped the run)
    exchanges: list[Exchange] = field(default_factory=list)


def extract_release(
    chat: ChatClient,
    model: str,
    prompt: Prompt,
    release: LinkedRelease,
    release_input: ReleaseInput,
) -> ReleaseResult:
    """Ask for the release's figures; retry an invalid reply once with its errors.
    An API error, or a second invalid reply, gives an all-null prediction."""
    messages = first_messages(prompt, release_input.text)
    exchanges: list[Exchange] = []
    invalid = ""  # why the last reply was invalid
    for call in (1, 2):
        exchange = chat.complete(request_body(model, messages))
        exchanges.append(exchange)
        if exchange.response is None:
            reason = exchange.error
            if invalid:
                reason += f", on the retry of an invalid reply ({invalid})"
            if exchange.abort:
                return ReleaseResult(
                    release, release_input.exhibits, None, ABORTED, reason, exchanges
                )
            return ReleaseResult(
                release,
                release_input.exhibits,
                null_prediction(release),
                API_ERROR,
                reason,
                exchanges,
            )
        try:
            prediction = parse_reply(exchange.content, release.release_accession)
        except ReplyError as exc:
            invalid = str(exc)
            messages = retry_messages(messages, exchange.content or "", invalid)
            continue
        outcome = OK if call == 1 else OK_AFTER_RETRY
        return ReleaseResult(
            release, release_input.exhibits, prediction, outcome, "", exchanges
        )
    return ReleaseResult(
        release,
        release_input.exhibits,
        null_prediction(release),
        INVALID_REPLY,
        invalid,
        exchanges,
    )


def raw_record(result: ReleaseResult) -> dict[str, Any]:
    """Everything sent and received for a release: raw/<accession>.json."""
    release = result.release
    return {
        "ticker": release.ticker,
        "release_accession": release.release_accession,
        "linked_accession": release.linked_accession,
        "input_exhibits": list(result.exhibits),
        "outcome": result.outcome,
        "reason": result.reason,
        "prediction": (
            result.prediction.model_dump(mode="json") if result.prediction else None
        ),
        "calls": [
            {"request": e.request, "attempts": e.attempts, "error": e.error}
            for e in result.exchanges
        ],
    }


# A run


def select_releases(
    releases: list[LinkedRelease],
    tickers: Sequence[str] | None = None,
    accessions: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[LinkedRelease]:
    """Linked releases in links.csv order, kept if their ticker and accession are
    among those given, then the first `limit`. A ticker or accession that isn't
    linked is an error."""
    selected = releases
    if tickers:
        wanted = {normalize_ticker(t) for t in tickers}
        unknown = wanted - {r.ticker for r in releases}
        if unknown:
            raise RunError(f"no linked releases for {', '.join(sorted(unknown))}")
        selected = [r for r in selected if r.ticker in wanted]
    if accessions:
        wanted = set(accessions)
        unknown = wanted - {r.release_accession for r in releases}
        if unknown:
            raise RunError(f"not linked: {', '.join(sorted(unknown))}")
        selected = [r for r in selected if r.release_accession in wanted]
    if not selected:
        raise RunError("no linked release matches every filter")
    return selected[:limit] if limit else selected


def release_inputs(
    releases: list[LinkedRelease], releases_dir: Path
) -> list[tuple[LinkedRelease, ReleaseInput]]:
    """Every release's input, built before any call so a missing one costs
    nothing."""
    records = release_records(releases_dir)
    missing = [
        r.release_accession for r in releases if r.release_accession not in records
    ]
    if missing:
        raise RunError(f"no saved release in {releases_dir}: {', '.join(missing)}")
    return [(r, build_input(records[r.release_accession])) for r in releases]


def progress_line(done: int, total: int, result: ReleaseResult) -> str:
    release = result.release
    latency = sum(e.latency_s or 0.0 for e in result.exchanges)
    line = (
        f"[{done:>{len(str(total))}}/{total}] {release.ticker} "
        f"{release.release_accession} {result.outcome} {latency:.1f}s"
    )
    if result.reason:
        line += f": {shorten(result.reason, 120)}"
    return line


def run_releases(
    chat: ChatClient,
    model: str,
    prompt: Prompt,
    work: list[tuple[LinkedRelease, ReleaseInput]],
    raw_dir: Path,
    parallel: int = MAX_PARALLEL,
) -> tuple[list[ReleaseResult], str]:
    """Run every release, at most `parallel` at once, saving each raw record as it
    finishes. Returns the results in work order and the run's status: COMPLETE, or
    ABORTED when a call stopped it (no new calls start; those in flight finish)."""
    results: dict[str, ReleaseResult] = {}
    status = COMPLETE
    queue = iter(work)
    running: dict[Future[ReleaseResult], str] = {}
    with ThreadPoolExecutor(max_workers=parallel) as pool:

        def submit_next() -> bool:
            for release, release_input in queue:
                future = pool.submit(
                    extract_release, chat, model, prompt, release, release_input
                )
                running[future] = release.release_accession
                return True
            return False

        while len(running) < parallel and submit_next():
            pass
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                accession = running.pop(future)
                result = future.result()
                results[accession] = result
                write_json(raw_dir / f"{accession}.json", raw_record(result))
                print(progress_line(len(results), len(work), result), flush=True)
                if result.outcome == ABORTED:
                    status = ABORTED
            while status == COMPLETE and len(running) < parallel and submit_next():
                pass
    ordered = [
        results[r.release_accession] for r, _ in work if r.release_accession in results
    ]
    return ordered, status


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile, p from 0 to 100; None without values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


def estimated_cost(
    input_tokens: int, output_tokens: int, settings: Settings
) -> float | None:
    if settings.price_input is None or settings.price_output is None:
        return None
    return (
        input_tokens * settings.price_input + output_tokens * settings.price_output
    ) / 1_000_000


def call_rows(results: list[ReleaseResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for number, exchange in enumerate(result.exchanges, 1):
            input_tokens, output_tokens = exchange.tokens
            rows.append(
                {
                    "ticker": result.release.ticker,
                    "release_accession": result.release.release_accession,
                    "call": number,  # 2: the retry of an invalid reply
                    "status": exchange.attempts[-1]["status"],
                    "http_attempts": len(exchange.attempts),
                    "latency_s": exchange.latency_s,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "finish_reason": exchange.finish_reason,
                    "error": exchange.error,
                }
            )
    return rows


def run_info(
    *,
    run_name: str,
    status: str,
    settings: Settings,
    prompt: Prompt,
    started: datetime,
    finished: datetime,
    selection: dict[str, Any],
    results: list[ReleaseResult],
) -> dict[str, Any]:
    """run.json."""
    answered = [e for r in results for e in r.exchanges if e.response is not None]
    tokens = [e.tokens for e in answered]
    without_usage = sum(1 for t in tokens if None in t)
    input_tokens = sum(t[0] or 0 for t in tokens)
    output_tokens = sum(t[1] or 0 for t in tokens)
    cost = (
        None if without_usage else estimated_cost(input_tokens, output_tokens, settings)
    )
    latencies = [e.latency_s for e in answered if e.latency_s is not None]
    outcomes = Counter(r.outcome for r in results)
    stopped = [r for r in results if r.outcome == ABORTED]
    return {
        "run_name": run_name,
        "status": status,
        "stop_reason": stopped[0].reason if stopped else "",
        "model": settings.model,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "retry_message": RETRY_MESSAGE,
        "temperature": TEMPERATURE,
        "max_parallel": MAX_PARALLEL,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "wall_time_s": round((finished - started).total_seconds(), 1),
        "selection": selection,
        "releases": len(results),
        "outcomes": {outcome: outcomes[outcome] for outcome in (*OUTCOMES, ABORTED)},
        "fallbacks": [
            {
                "ticker": r.release.ticker,
                "release_accession": r.release.release_accession,
                "outcome": r.outcome,
                "reason": r.reason,
            }
            for r in results
            if r.outcome in (INVALID_REPLY, API_ERROR)
        ],
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "calls_without_usage": without_usage,
        },
        "price_per_million_tokens": {
            "input": settings.price_input,
            "output": settings.price_output,
        },
        "estimated_cost_usd": None if cost is None else round(cost, 6),
        "latency_s": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies, default=None),
        },
        "calls": call_rows(results),
    }


def format_run_summary(info: dict[str, Any], run_dir: Path) -> str:
    """The run's outcomes, tokens, cost and latency, and every fallback: invalid
    replies, then API errors with a command to rerun them."""
    outcomes = info["outcomes"]
    counts = ", ".join(
        f"{outcomes[o]} {o.replace('_', ' ')}"
        for o in (*OUTCOMES, ABORTED)
        if outcomes[o]
    )
    lines = [f"run {run_dir}: {plural(info['releases'], 'release')} ({counts})"]
    if info["status"] == ABORTED:
        lines.append(f"stopped, not graded: {info['stop_reason']}")
    tokens = info["tokens"]
    line = f"tokens: {tokens['input']:,} input, {tokens['output']:,} output"
    if tokens["calls_without_usage"]:
        line += (
            f" ({plural(tokens['calls_without_usage'], 'call')} without token counts)"
        )
    lines.append(line)
    prices = info["price_per_million_tokens"]
    if info["estimated_cost_usd"] is not None:
        lines.append(
            f"estimated cost: ${info['estimated_cost_usd']:.4f} "
            f"(${prices['input']:g} input, ${prices['output']:g} output per million tokens)"
        )
    elif prices["input"] is None or prices["output"] is None:
        lines.append(
            "estimated cost: none (MODEL_PRICE_INPUT or MODEL_PRICE_OUTPUT not set)"
        )
    else:
        lines.append("estimated cost: none (some calls have no token counts)")
    latency = info["latency_s"]
    if latency["p50"] is not None:
        lines.append(
            f"latency per call: p50 {latency['p50']:.2f}s, p95 {latency['p95']:.2f}s, "
            f"max {latency['max']:.2f}s"
        )
    for outcome, title in (
        (INVALID_REPLY, "invalid replies"),
        (API_ERROR, "API errors"),
    ):
        fallbacks = [f for f in info["fallbacks"] if f["outcome"] == outcome]
        lines.append(
            f"{title} (all-null predictions, graded as missing): {len(fallbacks)}"
        )
        lines.extend(
            f"  {f['ticker']} {f['release_accession']}: {f['reason']}"
            for f in fallbacks
        )
    errors = [
        f["release_accession"] for f in info["fallbacks"] if f["outcome"] == API_ERROR
    ]
    if errors:
        lines.append(
            "rerun the API errors: uv run python -m secbench.extract --accessions "
            + " ".join(errors)
        )
    return "\n".join(lines)


def format_dry_run(
    prompt: Prompt,
    release: LinkedRelease,
    release_input: ReleaseInput,
    selected: int,
    prompts_dir: Path,
) -> str:
    text = release_input.text
    exhibits = " + ".join(release_input.exhibits)
    return "\n".join(
        [
            (
                f"dry run: {release.ticker} {release.release_accession} (1 of "
                f"{plural(selected, 'selected release')}); input {exhibits}, "
                f"{len(text):,} characters"
            ),
            (
                "POST <MODEL_BASE_URL>/chat/completions, model <MODEL_NAME>, "
                f"temperature {TEMPERATURE}"
            ),
            (
                f"=== system: {prompts_dir / prompt.version}.txt (sha256 "
                f"{prompt.sha256[:12]}) ==="
            ),
            prompt.text.rstrip("\n"),
            "=== user: the release input ===",
            text.rstrip("\n"),
            "=== end of request; no call made ===",
        ]
    )


def write_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def default_run_name(model: str, prompt: Prompt, now: datetime) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-.") or "model"
    return f"{now:%Y-%m-%dT%H%M%S}_{slug}_{prompt.version}"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


def main(
    argv: list[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
    edgar_client: EdgarClient | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.extract",
        description="Run a model on the linked releases, save its replies and grade "
        "them.",
    )
    parser.add_argument("--tickers", nargs="+", metavar="TICKER")
    parser.add_argument("--accessions", nargs="+", metavar="ACCESSION")
    parser.add_argument("--limit", type=positive_int, help="the first N releases")
    parser.add_argument(
        "--run-name", help="folder under --runs-dir (default: time_model_prompt)"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompts/<PROMPT>.txt")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the full request for the first release; make no calls",
    )
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--releases", type=Path, default=DEFAULT_RELEASES)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    try:
        prompt = load_prompt(args.prompt, args.prompts_dir)
        if not args.links.is_file():
            raise RunError(f"{args.links} not found")
        selected = select_releases(
            load_links(args.links)[0], args.tickers, args.accessions, args.limit
        )
        work = release_inputs(selected, args.releases)
        if args.dry_run:
            print(format_dry_run(prompt, *work[0], len(work), args.prompts_dir))
            return 0
        settings = load_settings(environ)
        started = datetime.now(UTC)
        run_name = args.run_name or default_run_name(settings.model, prompt, started)
        if not NAME.fullmatch(run_name):
            raise RunError(f"bad run name {run_name!r}")
        run_dir = args.runs_dir / run_name
        if run_dir.exists():
            raise RunError(f"{run_dir} already exists; choose another --run-name")
        (run_dir / RAW).mkdir(parents=True)
    except RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"{run_dir}: {plural(len(work), 'release')}, prompt {prompt.version}, "
        f"up to {MAX_PARALLEL} calls at a time"
    )
    with ChatClient(settings, transport=transport, sleep=sleep) as chat:
        results, status = run_releases(
            chat, settings.model, prompt, work, run_dir / RAW
        )
    info = run_info(
        run_name=run_name,
        status=status,
        settings=settings,
        prompt=prompt,
        started=started,
        finished=datetime.now(UTC),
        selection={
            "tickers": args.tickers,
            "accessions": args.accessions,
            "limit": args.limit,
        },
        results=results,
    )
    write_json(run_dir / RUN_INFO, info)
    graded = 1
    if status == COMPLETE:
        predictions = "".join(
            r.prediction.model_dump_json() + "\n" for r in results if r.prediction
        )
        write_text(run_dir / grade.PREDICTIONS, predictions)
        graded = grade.main(
            [
                str(run_dir),
                "--links",
                str(args.links),
                "--answers",
                str(args.answers),
                "--releases",
                str(args.releases),
            ],
            client=edgar_client,
        )
    print(format_run_summary(info, run_dir))
    return 0 if graded == 0 and not info["outcomes"][API_ERROR] else 1


if __name__ == "__main__":
    sys.exit(main())
