"""A load test of an OpenAI-compatible chat server, for the stage 3 serving benchmark
(PLAN.md Decisions 2026-11-09; secbench.servebench runs it). Library only.

Each request is the one secbench.extract sends (extract.OpenAIChat.first_body with a
settings file's parameters: the extract_v2 system prompt and the release's input,
temperature 0, max_tokens 1024, chat_template_kwargs enable_thinking false), plus
stream true and stream_options include_usage, so the client sees the first token and
the server's token counts.

A level sends a fixed list of requests at a fixed concurrency: that many workers take
the requests in order, each sending its next as soon as its last has finished, so
exactly `concurrency` requests are in flight until the list runs out. Per request:
when it was sent and when it finished (seconds from the level's start), the time to
first token (the first chunk with content), the end-to-end latency, the prompt and
output tokens the server reports (usage), the finish reason, the reply's text and any
error (an HTTP status and message, a transport error, an error in the stream, a stream
without usage). Nothing is retried: an error is a result.

Per level (summarize; the measured requests, warm-up requests aside): the makespan,
from the first request sent to the last finished; releases per hour (requests without
an error per makespan); prompt and output tokens per second (their sums over the
makespan); latency p50, p95, p99 and max and time to first token p50 and p95; and
the median decode speed per request, (output tokens - 1) / (latency - time to first
token), over requests with at least 2 output tokens. Percentiles are nearest-rank
(extract.percentile, as extract's latencies), over requests without an error.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from secbench import extract
from secbench.releases import write_text

STREAM = {"stream": True, "stream_options": {"include_usage": True}}
DONE = "[DONE]"
WARMUP, MEASURED = "warmup", "measured"
ERRORS_SHOWN = 5  # error messages a level's summary keeps


@dataclass(frozen=True)
class Request:
    label: str  # the release accession
    body: dict[str, Any]


@dataclass
class Result:
    label: str
    phase: str  # WARMUP or MEASURED
    index: int  # position in the level's list
    sent_s: float  # seconds from the level's start
    finished_s: float
    first_token_s: float | None  # None: no content came
    prompt_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    status: int | None  # HTTP status; None without a response
    error: str  # "" if none
    content: str

    @property
    def latency_s(self) -> float:
        return self.finished_s - self.sent_s

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_s is None:
            return None
        return self.first_token_s - self.sent_s

    def record(self, **extra: Any) -> dict[str, Any]:
        """The JSON line for this request (extra first: variant, level)."""
        row = asdict(self)
        return {
            **extra,
            **{k: row[k] for k in ("phase", "index", "label")},
            "sent_s": self.sent_s,
            "ttft_s": self.ttft_s,
            "latency_s": self.latency_s,
            "finished_s": self.finished_s,
            **{
                k: row[k]
                for k in (
                    "prompt_tokens",
                    "output_tokens",
                    "finish_reason",
                    "status",
                    "error",
                    "content",
                )
            },
        }


def request_bodies(
    settings: extract.Settings, prompt: extract.Prompt, inputs: Sequence[str]
) -> list[dict[str, Any]]:
    """extract's request body for each input, streamed."""
    with extract.OpenAIChat(settings) as client:
        return [{**client.first_body(prompt, text), **STREAM} for text in inputs]


def stream_error(chunk: Mapping[str, Any]) -> str:
    """An error vLLM sends inside the stream ({"error": {...}} or {"object": "error",
    "message": ...}); "" if the chunk isn't one."""
    if "error" in chunk or chunk.get("object") == "error":
        return "error in the stream: " + (
            extract.error_message(chunk) or json.dumps(chunk)[: extract.ERROR_WIDTH]
        )
    return ""


async def send(
    client: httpx.AsyncClient,
    url: str,
    request: Request,
    phase: str,
    index: int,
    clock: Callable[[], float],
    start: float,
) -> Result:
    """One streamed request, timed against the level's start."""
    sent = clock() - start
    first: float | None = None
    parts: list[str] = []
    finish: str | None = None
    usage: Mapping[str, Any] | None = None
    status: int | None = None
    error = ""
    try:
        async with client.stream("POST", url, json=request.body) as response:
            status = response.status_code
            if status != 200:
                await response.aread()
                body = extract.response_body(response)
                error = f"HTTP {status}"
                if message := extract.error_message(body):
                    error += f" ({message})"
            else:
                done = False
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == DONE:
                        done = True
                        break
                    chunk = json.loads(data)
                    if error := stream_error(chunk):
                        break
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if isinstance(text, str) and text:
                            if first is None:
                                first = clock() - start
                            parts.append(text)
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                if not error and not done:
                    error = "the stream ended without [DONE]"
                elif not error and usage is None:
                    error = "no usage in the stream"
    except (httpx.TransportError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"[: extract.ERROR_WIDTH]
    finished = clock() - start
    return Result(
        request.label,
        phase,
        index,
        sent,
        finished,
        first,
        extract.count(usage.get("prompt_tokens")) if usage else None,
        extract.count(usage.get("completion_tokens")) if usage else None,
        finish,
        status,
        error,
        "".join(parts),
    )


async def run_requests(
    client: httpx.AsyncClient,
    url: str,
    requests: Sequence[Request],
    concurrency: int,
    phase: str,
    clock: Callable[[], float],
    start: float,
) -> list[Result]:
    """Every request, `concurrency` at a time (a closed loop), in list order."""
    results: list[Result | None] = [None] * len(requests)
    queue = iter(enumerate(requests))

    async def worker() -> None:
        for index, request in queue:
            results[index] = await send(
                client, url, request, phase, index, clock, start
            )

    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(requests)))))
    return [r for r in results if r is not None]


async def run_level_async(
    url: str,
    warmup: Sequence[Request],
    measured: Sequence[Request],
    concurrency: int,
    transport: httpx.AsyncBaseTransport | None,
    clock: Callable[[], float],
) -> tuple[list[Result], list[Result]]:
    limits = httpx.Limits(max_connections=max(100, concurrency))
    async with httpx.AsyncClient(
        timeout=extract.TIMEOUT, limits=limits, transport=transport
    ) as client:
        start = clock()
        warm = await run_requests(
            client, url, warmup, concurrency, WARMUP, clock, start
        )
        start = clock()  # the measured requests' times start at 0
        results = await run_requests(
            client, url, measured, concurrency, MEASURED, clock, start
        )
    return warm, results


def run_level(
    url: str,
    warmup: Sequence[Request],
    measured: Sequence[Request],
    concurrency: int,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[Result], list[Result]]:
    """The warm-up requests, then the measured ones, each at `concurrency`; their
    results in list order."""
    return asyncio.run(
        run_level_async(url, warmup, measured, concurrency, transport, clock)
    )


def median(values: Sequence[float]) -> float | None:
    return extract.percentile(values, 50)


def summarize(results: Sequence[Result]) -> dict[str, Any]:
    """A level's numbers, from its measured requests (see the module docstring)."""
    ok = [r for r in results if not r.error]
    makespan = (
        max(r.finished_s for r in results) - min(r.sent_s for r in results)
        if results
        else None
    )
    prompt_tokens = sum(r.prompt_tokens or 0 for r in ok)
    output_tokens = sum(r.output_tokens or 0 for r in ok)
    latencies = [r.latency_s for r in ok]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    decode = [
        (r.output_tokens - 1) / (r.latency_s - r.ttft_s)
        for r in ok
        if r.output_tokens is not None
        and r.output_tokens >= 2
        and r.ttft_s is not None
        and r.latency_s > r.ttft_s
    ]

    def per_second(n: float) -> float | None:
        return n / makespan if makespan else None

    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "makespan_s": makespan,
        "releases_per_hour": None if not makespan else len(ok) * 3600 / makespan,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_tokens_per_s": per_second(prompt_tokens),
        "output_tokens_per_s": per_second(output_tokens),
        "latency_s": {
            "p50": extract.percentile(latencies, 50),
            "p95": extract.percentile(latencies, 95),
            "p99": extract.percentile(latencies, 99),
            "max": max(latencies, default=None),
        },
        "ttft_s": {
            "p50": extract.percentile(ttfts, 50),
            "p95": extract.percentile(ttfts, 95),
        },
        "decode_tokens_per_s_p50": median(decode),
        "error_messages": [r.error for r in results if r.error][:ERRORS_SHOWN],
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(
        path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
