"""Run a model on the linked releases, save everything it returns, and grade it.

Each linked release in data/links.csv (see secbench.link) goes to one model, through
an OpenAI-compatible chat completions API or Anthropic's Messages API (the anthropic
SDK): the instructions in prompts/<version>.txt as the system prompt, the release's
input (see secbench.inputs) as the user message.
The reply must be one JSON object of a Prediction (see secbench.grade) without
release_accession. An invalid reply is retried once with its validation errors. If
the retry is invalid too, or the API fails, the release gets an all-null prediction
with links.csv's period end, which grades as missing; run.json and the printed
summary list every such fallback and why. A released prompt is never edited; a
changed prompt is a new version (extract_v2).

Usage: uv run python -m secbench.extract [--tickers AAPL ...] [--accessions ...]
           [--limit N] [--run-name NAME] [--prompt extract_v1] [--dry-run]
           [--env-file .env.vllm] [--links] [--answers] [--releases] [--runs-dir]
           [--prompts-dir]

Settings come from the environment or .env (the environment wins), or with --env-file
only from that file (read literally; the environment and .env are not used for them,
and a setting the file leaves out is blank): MODEL_PROVIDER (openai, the default, or
anthropic), MODEL_NAME, MODEL_API_KEY, MODEL_CONCURRENCY (optional; calls at once,
default 4), and MODEL_PRICE_INPUT and MODEL_PRICE_OUTPUT (optional; dollars per
million tokens, for the cost estimate). openai also reads MODEL_BASE_URL (such as
http://localhost:8000/v1), MODEL_TEMPERATURE (optional; if blank, no temperature is
sent and the server's default applies), MODEL_EXTRA_BODY (optional; a JSON object
merged into every request body, such as {"reasoning_effort": "medium"}; it can't set
model, messages, stream or temperature, and if blank nothing more is sent) and
MODEL_GPU_HOURLY (optional; for a self-hosted model, dollars per hour for its GPUs:
the cost estimate is then that times the run's wall-clock time, and token prices
can't be set too); its key is optional. anthropic needs the key, sends max_tokens
16000, MODEL_EFFORT as output_config.effort (optional: low, medium, high, xhigh or
max; if blank, none is sent and the model's default applies) and never a temperature,
reads only the text blocks of a reply (not thinking blocks), and counts a refusal as
an API error; there is no fallback to another model, which would mix two models in
one run. run.json records the provider, the request parameters sent and the cost
basis. --dry-run prints the full request for
the first selected release, without reading them or making a call.

A run writes runs/<run>/raw/<accession>.json (every HTTP attempt's request body and
response; no headers) as each release finishes, then predictions.jsonl and run.json
(model, prompt version, token counts, latency per call, estimated cost, fallbacks),
grades the run (see secbench.grade) and prints the summary. At most MODEL_CONCURRENCY
calls (default 4) run at a time; for both providers, 429, 5xx, timeouts and connection
errors are retried with
the same backoff (the SDK's own retries are off). HTTP 401, 403 or 404 (wrong key,
model or URL) stops the run without grading it.
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

import anthropic
import httpx
from dotenv import dotenv_values, find_dotenv, load_dotenv
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
DEFAULT_CONCURRENCY = 4  # calls at once, unless MODEL_CONCURRENCY says otherwise
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

# The cost estimate's basis: token prices, or GPU dollars per hour times wall time.
TOKENS = "tokens"
GPU_HOURS = "gpu_hours"

OPENAI = "openai"
ANTHROPIC = "anthropic"
PROVIDERS = (OPENAI, ANTHROPIC)
ANTHROPIC_MAX_TOKENS = 16000
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Request body keys MODEL_EXTRA_BODY can't set: the runner sets the first two, a
# streamed reply can't be parsed, and MODEL_TEMPERATURE sets the temperature.
RESERVED_BODY_KEYS = ("model", "messages", "stream", "temperature")

SETTINGS = (
    "MODEL_PROVIDER",
    "MODEL_BASE_URL",
    "MODEL_NAME",
    "MODEL_API_KEY",
    "MODEL_TEMPERATURE",
    "MODEL_EXTRA_BODY",
    "MODEL_EFFORT",
    "MODEL_CONCURRENCY",
    "MODEL_PRICE_INPUT",
    "MODEL_PRICE_OUTPUT",
    "MODEL_GPU_HOURLY",
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
    base_url: str  # openai only
    model: str
    api_key: str = ""  # openai: blank sends no Authorization header (a local server)
    temperature: float | None = None  # openai only; None: not sent
    price_input: float | None = None  # dollars per million tokens
    price_output: float | None = None
    provider: str = OPENAI
    effort: str | None = None  # anthropic only; None: not sent
    extra_body: dict[str, Any] | None = None  # openai only; merged into each body
    concurrency: int = DEFAULT_CONCURRENCY  # calls at once
    gpu_hourly: float | None = None  # openai only; dollars per hour, not per token

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def parameters(self) -> dict[str, Any]:
        """What every request sends besides the model, system prompt and messages:
        anthropic, max_tokens, output_config.effort only if MODEL_EFFORT is set, and
        never a temperature; openai, temperature only if MODEL_TEMPERATURE is set,
        then MODEL_EXTRA_BODY's keys."""
        if self.provider == ANTHROPIC:
            effort = {"output_config": {"effort": self.effort}} if self.effort else {}
            return {"max_tokens": ANTHROPIC_MAX_TOKENS, **effort}
        temperature = (
            {} if self.temperature is None else {"temperature": self.temperature}
        )
        return {**temperature, **(self.extra_body or {})}


def parse_number(name: str, value: str) -> float | None:
    """A setting that is 0 or more; None if blank."""
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        raise SettingsError(f"{name} is not a number") from None
    if not math.isfinite(number) or number < 0:
        raise SettingsError(f"{name} must be 0 or more")
    return number


def parse_count(name: str, value: str, default: int) -> int:
    """A setting that is a whole number, 1 or more; default if blank."""
    if not value:
        return default
    try:
        number = int(value)
    except ValueError:
        raise SettingsError(f"{name} is not a whole number") from None
    if number < 1:
        raise SettingsError(f"{name} must be 1 or more")
    return number


def parse_extra_body(value: str) -> dict[str, Any] | None:
    """MODEL_EXTRA_BODY: a JSON object without RESERVED_BODY_KEYS; None if blank."""
    if not value:
        return None

    def no_constant(name: str) -> None:
        raise ValueError(name)  # NaN and Infinity aren't JSON

    try:
        body = json.loads(value, parse_constant=no_constant)
    except ValueError:
        raise SettingsError("MODEL_EXTRA_BODY is not valid JSON") from None
    if not isinstance(body, dict):
        raise SettingsError("MODEL_EXTRA_BODY must be a JSON object")
    reserved = [key for key in RESERVED_BODY_KEYS if key in body]
    if reserved:
        raise SettingsError(
            f"MODEL_EXTRA_BODY can't set {' or '.join(reserved)} (the runner sets "
            "model and messages, a reply can't be streamed, and MODEL_TEMPERATURE "
            "sets the temperature)"
        )
    return body


def load_settings(
    environ: Mapping[str, str] | None = None, env_file: Path | None = None
) -> Settings:
    """The MODEL_* settings from env_file alone (environ is then not used), else
    from environ, or else from the environment and .env (the environment wins)."""
    if env_file is not None:
        if not env_file.is_file():
            raise SettingsError(f"{env_file} not found")
        file_values = dotenv_values(env_file, interpolate=False)
        environ = {name: value or "" for name, value in file_values.items()}
    elif environ is None:
        load_dotenv(find_dotenv(usecwd=True))
        environ = os.environ
    values = {name: environ.get(name, "").strip() for name in SETTINGS}
    provider = values["MODEL_PROVIDER"].lower() or OPENAI
    if provider not in PROVIDERS:
        raise SettingsError("MODEL_PROVIDER must be openai or anthropic")
    # anthropic ignores MODEL_BASE_URL, MODEL_TEMPERATURE and MODEL_EXTRA_BODY, and
    # openai ignores MODEL_EFFORT, so .env can switch providers by MODEL_PROVIDER
    # alone.
    is_openai = provider == OPENAI
    required = (
        ("MODEL_BASE_URL", "MODEL_NAME")
        if is_openai
        else ("MODEL_NAME", "MODEL_API_KEY")
    )
    missing = [name for name in required if not values[name]]
    if missing:
        raise SettingsError(
            f"{' and '.join(missing)} not set; add to {env_file or '.env'} (see "
            ".env.example)"
        )
    if is_openai and not re.match(
        r"https?://", values["MODEL_BASE_URL"], re.IGNORECASE
    ):
        raise SettingsError("MODEL_BASE_URL must start with http:// or https://")
    temperature = values["MODEL_TEMPERATURE"] if is_openai else ""
    extra_body = parse_extra_body(values["MODEL_EXTRA_BODY"] if is_openai else "")
    effort = "" if is_openai else values["MODEL_EFFORT"].lower()
    if effort and effort not in EFFORTS:
        raise SettingsError(f"MODEL_EFFORT must be {', '.join(EFFORTS[:-1])} or max")
    gpu_hourly = parse_number(
        "MODEL_GPU_HOURLY", values["MODEL_GPU_HOURLY"] if is_openai else ""
    )
    if gpu_hourly is not None and (
        values["MODEL_PRICE_INPUT"] or values["MODEL_PRICE_OUTPUT"]
    ):
        raise SettingsError(
            "set MODEL_GPU_HOURLY or MODEL_PRICE_INPUT and MODEL_PRICE_OUTPUT, not both"
        )
    return Settings(
        base_url=values["MODEL_BASE_URL"] if is_openai else "",
        model=values["MODEL_NAME"],
        api_key=values["MODEL_API_KEY"],
        temperature=parse_number("MODEL_TEMPERATURE", temperature),
        price_input=parse_number("MODEL_PRICE_INPUT", values["MODEL_PRICE_INPUT"]),
        price_output=parse_number("MODEL_PRICE_OUTPUT", values["MODEL_PRICE_OUTPUT"]),
        provider=provider,
        effort=effort or None,
        extra_body=extra_body,
        concurrency=parse_count(
            "MODEL_CONCURRENCY", values["MODEL_CONCURRENCY"], DEFAULT_CONCURRENCY
        ),
        gpu_hourly=gpu_hourly,
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


# Replies


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


# Providers


def count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass(frozen=True)
class Reply:
    """A model's reply, as its provider reads it."""

    content: str | None  # the text (anthropic: the text blocks, joined)
    finish_reason: str | None  # anthropic: stop_reason
    tokens: tuple[int | None, int | None]  # (input, output), from usage
    assistant: dict[str, Any]  # the reply as a message, sent back on a retry
    refusal: str = ""  # why the model declined; "" if it didn't


@dataclass
class Attempt:
    """One HTTP attempt, as a provider reports it: its status, what raw keeps of it
    ({"response": body} or {"error": ...}), and either the reply or why it failed
    and whether to retry."""

    status: int | None  # None: no HTTP response (timeout, connection error)
    saved: dict[str, Any]
    reply: Reply | None = None
    problem: str = ""
    retry: bool = False  # 429, 5xx, timeout or connection error
    retry_after: float | None = None  # seconds, from Retry-After
    abort: bool = False  # wrong key, model or URL: every call would fail


@dataclass
class Exchange:
    """One request to the model and every HTTP attempt it took. reply is the model's
    reply, a refusal included; error says why the call failed (no reply, or a
    refusal), and abort whether the run must stop."""

    request: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)
    reply: Reply | None = None
    error: str = ""
    abort: bool = False

    @property
    def content(self) -> str | None:
        return self.reply.content if self.reply else None

    @property
    def finish_reason(self) -> str | None:
        return self.reply.finish_reason if self.reply else None

    @property
    def tokens(self) -> tuple[int | None, int | None]:
        return self.reply.tokens if self.reply else (None, None)

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


def error_message(body: Any) -> str:
    """A provider's error message: {"error": {"message": ...}} (OpenAI, Anthropic),
    {"message": ...} (vLLM) or {"error": "..."}."""
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, dict):
        error = error.get("message")
    message = error if isinstance(error, str) else body.get("message")
    return shorten(message.strip(), ERROR_WIDTH) if isinstance(message, str) else ""


def failed_attempt(
    status: int, body: Any, retry_after: str | None, now: float
) -> Attempt:
    """An HTTP error other than 401, 403 or 404: 429 and 5xx are retried."""
    problem = f"HTTP {status}"
    if message := error_message(body):
        problem += f" ({message})"
    retry = status == 429 or status >= 500
    return Attempt(
        status,
        {"response": body},
        problem=problem,
        retry=retry,
        retry_after=parse_retry_after(retry_after, now) if retry else None,
    )


def backoff(attempt: int) -> float:
    base = 2.0**attempt  # 1, 2, 4, 8, 16 s
    return base + random.uniform(0, 0.25 * base)


class ChatClient:
    """The retry loop every provider shares: retries 429, 5xx, timeouts and
    connection errors up to max_retries times (1-16 s backoff, Retry-After honored;
    asked to wait over 120 s, it gives up). A provider builds request bodies and
    makes one HTTP attempt (_send). Safe to share between threads."""

    provider = ""

    def __init__(
        self,
        settings: Settings,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.settings = settings
        self._sleep = sleep
        self._clock = clock
        self._wall_clock = wall_clock
        self.max_retries = max_retries

    def close(self) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def first_body(self, prompt: Prompt, input_text: str) -> dict[str, Any]:
        raise NotImplementedError

    def retry_body(
        self, body: dict[str, Any], reply: Reply, errors: str
    ) -> dict[str, Any]:
        """The request so far, plus the invalid reply as sent back (append-only) and
        what was wrong with it."""
        retry = {"role": "user", "content": RETRY_MESSAGE.format(errors=errors)}
        return {**body, "messages": [*body["messages"], reply.assistant, retry]}

    def _send(self, body: dict[str, Any]) -> Attempt:
        raise NotImplementedError

    def complete(self, body: dict[str, Any]) -> Exchange:
        exchange = Exchange(body)
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            start = self._clock()
            result = self._send(body)
            latency = round(self._clock() - start, 3)
            exchange.attempts.append(
                {"status": result.status, "latency_s": latency, **result.saved}
            )
            if result.reply is not None:
                exchange.reply = result.reply
                exchange.error = result.reply.refusal  # a refusal isn't retried
                return exchange
            if not result.retry:
                exchange.error, exchange.abort = result.problem, result.abort
                return exchange
            if attempt + 1 == attempts:
                exchange.error = f"{result.problem}, after {attempts} attempts"
                return exchange
            delay = result.retry_after
            if delay is None:
                delay = backoff(attempt)
            if delay > MAX_RETRY_WAIT:
                exchange.error = (
                    f"{result.problem}, and the server asked to wait {delay:.0f}s"
                )
                return exchange
            log.warning(
                "%s; retry %d/%d in %.1fs",
                result.problem,
                attempt + 1,
                self.max_retries,
                delay,
            )
            self._sleep(delay)
        raise AssertionError("unreachable")


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


def openai_reply(body: dict[str, Any]) -> Reply:
    choice = body["choices"][0]
    content = choice["message"].get("content")
    text = content if isinstance(content, str) else None
    usage = body.get("usage")
    tokens = (
        (count(usage.get("prompt_tokens")), count(usage.get("completion_tokens")))
        if isinstance(usage, dict)
        else (None, None)
    )
    assistant = {"role": "assistant", "content": text or ""}
    return Reply(text, choice.get("finish_reason"), tokens, assistant)


class OpenAIChat(ChatClient):
    """Any OpenAI-compatible chat completions API, at the settings' endpoint. The
    prompt is the first message, with role system."""

    provider = OPENAI

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        **options: Any,
    ) -> None:
        super().__init__(settings, **options)
        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._http = httpx.Client(headers=headers, timeout=TIMEOUT, transport=transport)

    def close(self) -> None:
        self._http.close()

    def first_body(self, prompt: Prompt, input_text: str) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": prompt.text},
                {"role": "user", "content": input_text},
            ],
            **self.settings.parameters,
        }

    def _send(self, body: dict[str, Any]) -> Attempt:
        try:
            response = self._http.post(self.settings.endpoint, json=body)
        except httpx.TransportError as exc:
            problem = f"{type(exc).__name__}: {exc}"
            return Attempt(None, {"error": problem}, problem=problem, retry=True)
        status = response.status_code
        if status in ABORT_STATUSES:
            # Not kept: the body can echo part of the key (OpenAI's 401 does).
            return Attempt(
                status,
                {"response": None},
                problem=f"HTTP {status}; check MODEL_BASE_URL, MODEL_NAME and "
                "MODEL_API_KEY",
                abort=True,
            )
        data = response_body(response)
        if not response.is_success:
            return failed_attempt(
                status, data, response.headers.get("Retry-After"), self._wall_clock()
            )
        if not is_completion(data):
            return Attempt(
                status,
                {"response": data},
                problem=f"HTTP {status} without choices[0].message",
            )
        return Attempt(status, {"response": data}, reply=openai_reply(data))


def refusal_reason(message: anthropic.types.Message) -> str:
    """'refusal (cyber): <explanation>', as much as stop_details gives."""
    reason = "refusal"
    details = message.stop_details
    if details and details.category:
        reason += f" ({details.category})"
    if details and details.explanation:
        reason += f": {shorten(details.explanation, ERROR_WIDTH)}"
    return reason


def anthropic_reply(message: anthropic.types.Message) -> Reply:
    """The text blocks, joined; thinking blocks are ignored, but sent back unchanged
    with the rest of the content on a retry, as the API expects."""
    text = "".join(block.text for block in message.content if block.type == "text")
    content = [block.to_dict(mode="json") for block in message.content]
    return Reply(
        text,
        message.stop_reason,
        (message.usage.input_tokens, message.usage.output_tokens),
        {"role": "assistant", "content": content},
        refusal_reason(message) if message.stop_reason == "refusal" else "",
    )


class AnthropicChat(ChatClient):
    """Anthropic's Messages API through the anthropic SDK, with its own retries off
    (max_retries=0) so the shared loop applies. The prompt is the system parameter;
    max_tokens is 16000, effort is MODEL_EFFORT's if set, and no temperature or
    thinking setting is sent, so the model's defaults apply to the rest."""

    provider = ANTHROPIC

    def __init__(
        self,
        settings: Settings,
        *,
        client: anthropic.Anthropic | None = None,
        **options: Any,
    ) -> None:
        super().__init__(settings, **options)
        self._client = client or anthropic.Anthropic(
            api_key=settings.api_key, max_retries=0
        )

    def close(self) -> None:
        self._client.close()

    def first_body(self, prompt: Prompt, input_text: str) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            **self.settings.parameters,
            "system": prompt.text,
            "messages": [{"role": "user", "content": input_text}],
        }

    def _send(self, body: dict[str, Any]) -> Attempt:
        try:
            message = self._client.messages.create(**body)
        except anthropic.APIStatusError as exc:
            status = exc.status_code
            if status in ABORT_STATUSES:
                return Attempt(
                    status,
                    {"response": None},
                    problem=f"HTTP {status}; check MODEL_NAME and MODEL_API_KEY",
                    abort=True,
                )
            return failed_attempt(
                status,
                exc.body,
                exc.response.headers.get("retry-after"),
                self._wall_clock(),
            )
        except anthropic.APIConnectionError as exc:  # a timeout too
            problem = f"{type(exc).__name__}: {shorten(str(exc), ERROR_WIDTH)}"
            return Attempt(None, {"error": problem}, problem=problem, retry=True)
        return Attempt(
            200,
            {"response": message.to_dict(mode="json")},
            reply=anthropic_reply(message),
        )


def make_chat(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ChatClient:
    if settings.provider == ANTHROPIC:
        return AnthropicChat(settings, client=anthropic_client, sleep=sleep)
    return OpenAIChat(settings, transport=transport, sleep=sleep)


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
    prompt: Prompt,
    release: LinkedRelease,
    release_input: ReleaseInput,
) -> ReleaseResult:
    """Ask for the release's figures; retry an invalid reply once with its errors.
    An API error (a refusal included), or a second invalid reply, gives an all-null
    prediction."""
    body = chat.first_body(prompt, release_input.text)
    exchanges: list[Exchange] = []
    invalid = ""  # why the last reply was invalid
    for call in (1, 2):
        exchange = chat.complete(body)
        exchanges.append(exchange)
        if exchange.reply is None or exchange.error:
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
            body = chat.retry_body(body, exchange.reply, invalid)
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
    prompt: Prompt,
    work: list[tuple[LinkedRelease, ReleaseInput]],
    raw_dir: Path,
    parallel: int = DEFAULT_CONCURRENCY,
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
                    extract_release, chat, prompt, release, release_input
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
                    "finish_reason": exchange.finish_reason,  # anthropic: stop_reason
                    "error": exchange.error,
                }
            )
    return rows


def run_info(
    *,
    run_name: str,
    status: str,
    settings: Settings,
    env_file: Path | None = None,
    prompt: Prompt,
    started: datetime,
    finished: datetime,
    selection: dict[str, Any],
    results: list[ReleaseResult],
) -> dict[str, Any]:
    """run.json."""
    # Every reply is billed, a refusal too.
    answered = [e for r in results for e in r.exchanges if e.reply is not None]
    tokens = [e.tokens for e in answered]
    without_usage = sum(1 for t in tokens if None in t)
    input_tokens = sum(t[0] or 0 for t in tokens)
    output_tokens = sum(t[1] or 0 for t in tokens)
    wall_time_s = (finished - started).total_seconds()
    if settings.gpu_hourly is not None:
        # Assumes the GPUs serve only this run; model loading and idle time aren't in.
        cost: float | None = settings.gpu_hourly * wall_time_s / 3600
        cost_basis: str | None = GPU_HOURS
    else:
        cost = (
            None
            if without_usage
            else estimated_cost(input_tokens, output_tokens, settings)
        )
        cost_basis = None if cost is None else TOKENS
    latencies = [e.latency_s for e in answered if e.latency_s is not None]
    outcomes = Counter(r.outcome for r in results)
    stopped = [r for r in results if r.outcome == ABORTED]
    return {
        "run_name": run_name,
        "status": status,
        "stop_reason": stopped[0].reason if stopped else "",
        "provider": settings.provider,
        "model": settings.model,
        "env_file": None if env_file is None else str(env_file),  # the path only
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "retry_message": RETRY_MESSAGE,
        # Sent with every request besides the model, system prompt and messages.
        "request_parameters": settings.parameters,
        "extra_body": settings.extra_body,  # MODEL_EXTRA_BODY, part of the above
        "max_parallel": settings.concurrency,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "wall_time_s": round(wall_time_s, 1),
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
        "gpu_hourly_usd": settings.gpu_hourly,
        "cost_basis": cost_basis,  # TOKENS, GPU_HOURS, or None without an estimate
        "estimated_cost_usd": None if cost is None else round(cost, 6),
        "latency_s": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies, default=None),
        },
        "calls": call_rows(results),
    }


def describe_value(value: Any) -> str:
    """0.0 as 0, a string as itself, anything else as JSON."""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, str):
        return value
    return json.dumps(value)


def describe_parameters(provider: str, parameters: dict[str, Any]) -> str:
    """Like "temperature 0, reasoning_effort medium", or "max_tokens 16000, effort
    high, no temperature"."""
    parts = []
    for name, value in parameters.items():
        if provider == ANTHROPIC and name == "output_config":
            parts.extend(f"{key} {setting}" for key, setting in value.items())
        else:
            parts.append(f"{name} {describe_value(value)}")
    if provider == ANTHROPIC and "output_config" not in parameters:
        parts.append("default effort")
    if "temperature" not in parameters:
        parts.append("no temperature")
    return ", ".join(parts)


def format_run_summary(info: dict[str, Any], run_dir: Path) -> str:
    """The run's outcomes, tokens, cost and latency, and every fallback: invalid
    replies, then API errors with a command to rerun them."""
    outcomes = info["outcomes"]
    counts = ", ".join(
        f"{outcomes[o]} {o.replace('_', ' ')}"
        for o in (*OUTCOMES, ABORTED)
        if outcomes[o]
    )
    lines = [
        f"run {run_dir}: {plural(info['releases'], 'release')} ({counts})",
        (
            f"provider {info['provider']}; sent with each request: "
            f"{describe_parameters(info['provider'], info['request_parameters'])}"
        ),
    ]
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
    if info["cost_basis"] == GPU_HOURS:
        lines.append(
            f"estimated cost: ${info['estimated_cost_usd']:.4f} "
            f"(${info['gpu_hourly_usd']:g} per GPU-hour for {info['wall_time_s']:.1f} s "
            "of wall-clock time)"
        )
    elif info["estimated_cost_usd"] is not None:
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
                "openai (the default): POST <MODEL_BASE_URL>/chat/completions, model "
                "<MODEL_NAME>, temperature <MODEL_TEMPERATURE> (not sent if blank), "
                "<MODEL_EXTRA_BODY> merged into the body (if set)"
            ),
            (
                "anthropic: messages.create (anthropic SDK), model <MODEL_NAME>, "
                f"max_tokens {ANTHROPIC_MAX_TOKENS}, effort <MODEL_EFFORT> (not sent "
                "if blank), no temperature"
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
    anthropic_client: anthropic.Anthropic | None = None,
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
        help="print the full request for the first release; make no calls (reads "
        "no settings)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="read the MODEL_* settings from this file only (such as .env.vllm)",
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
        settings = load_settings(environ, args.env_file)
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
        f"{run_dir}: {plural(len(work), 'release')}, provider {settings.provider}, "
        f"prompt {prompt.version}, "
        f"{describe_parameters(settings.provider, settings.parameters)}, up to "
        f"{settings.concurrency} calls at a time"
        + (f"; settings from {args.env_file}" if args.env_file else "")
    )
    with make_chat(
        settings, transport=transport, anthropic_client=anthropic_client, sleep=sleep
    ) as chat:
        results, status = run_releases(
            chat, prompt, work, run_dir / RAW, settings.concurrency
        )
    info = run_info(
        run_name=run_name,
        status=status,
        settings=settings,
        env_file=args.env_file,
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
