"""secbench.loadtest against a fake streaming server (httpx.MockTransport): the
request extract sends plus streaming, the time to first token and usage read from the
stream, errors as results, the concurrency limit, and each level's arithmetic on
handwritten numbers."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from secbench import extract, labels, loadtest, score
from secbench.loadtest import Request, Result

PROMPTS = Path(__file__).parent.parent / "prompts"
URL = "http://127.0.0.1:8000/v1/chat/completions"


class Clock:
    """A clock the fake server moves forward, so times are exact."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def sse(chunk: dict | str) -> bytes:
    data = chunk if isinstance(chunk, str) else json.dumps(chunk)
    return f"data: {data}\n\n".encode()


class FakeServer:
    """vLLM's streamed chat completions. The reply comes in `tokens` content chunks
    (one token each) after a role chunk with empty content; the clock moves `prefill`
    seconds before the first and `per_token` before each other; then the finish
    reason, the usage chunk and [DONE]. Some labels fail, as `failures` says."""

    def __init__(
        self,
        clock: Clock,
        *,
        tokens: int = 5,
        prefill: float = 0.5,
        per_token: float = 0.1,
        failures: dict[str, str] | None = None,
        prompt_tokens: int = 1000,
    ) -> None:
        self.clock = clock
        self.tokens = tokens
        self.prefill = prefill
        self.per_token = per_token
        self.failures = failures or {}
        self.prompt_tokens = prompt_tokens
        self.bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == URL
        body = json.loads(request.content)
        self.bodies.append(body)
        label = body["messages"][1]["content"]
        failure = self.failures.get(label)
        if failure == "http":
            return httpx.Response(
                400,
                json={
                    "object": "error",
                    "message": "This model's maximum context length is 40960 tokens.",
                },
            )
        if failure == "connect":
            raise httpx.ConnectError("connection refused")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=self.events(failure),
        )

    async def events(self, failure: str | None):
        chunk = {"object": "chat.completion.chunk", "model": "m"}
        yield sse({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        yield sse(
            {
                **chunk,
                "choices": [{"index": 0, "delta": {"content": ""}}],
            }
        )
        for n in range(self.tokens):
            self.clock.now += self.prefill if n == 0 else self.per_token
            await asyncio.sleep(0)
            yield sse(
                {**chunk, "choices": [{"index": 0, "delta": {"content": f"t{n} "}}]}
            )
            if failure == "midway" and n == 1:
                yield sse({"object": "error", "message": "engine died", "code": 500})
                return
        yield sse(
            {
                **chunk,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        if failure != "no_usage":
            yield sse(
                {
                    **chunk,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": self.prompt_tokens,
                        "completion_tokens": self.tokens,
                        "total_tokens": self.prompt_tokens + self.tokens,
                    },
                }
            )
        if failure != "no_done":
            yield sse("[DONE]")


def requests(*labels_: str) -> list[Request]:
    return [
        Request(label, {"model": "m", "messages": [{}, {"content": label}]})
        for label in labels_
    ]


def test_request_bodies_are_extracts_plus_streaming(tmp_path) -> None:
    env = tmp_path / "time_test.env"
    env.write_text(score.env_text("http://127.0.0.1:8000/v1", "served", 16, 2.5))
    settings = extract.load_settings(env_file=env)
    prompt = extract.load_prompt(labels.PROMPT, PROMPTS)
    [body] = loadtest.request_bodies(settings, prompt, ["Revenue 101."])
    with extract.OpenAIChat(settings) as client:
        sent = client.first_body(prompt, "Revenue 101.")
    assert body == {**sent, "stream": True, "stream_options": {"include_usage": True}}
    assert body["model"] == "served"
    assert body["messages"] == [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": "Revenue 101."},
    ]
    assert {k: body[k] for k in score.expected_parameters()} == (
        score.expected_parameters()
    )
    assert score.expected_parameters() == {
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 1024,
    }


def test_time_to_first_token_usage_and_reply() -> None:
    clock = Clock()
    server = FakeServer(clock, tokens=5, prefill=0.5, per_token=0.1)
    warm, measured = loadtest.run_level(
        URL,
        requests("w1"),
        requests("a", "b"),
        1,
        transport=httpx.MockTransport(server.handler),
        clock=clock,
    )
    assert [r.label for r in warm] == ["w1"] and warm[0].phase == "warmup"
    a, b = measured
    assert (a.label, a.phase, a.index, b.index) == ("a", "measured", 0, 1)
    assert a.error == "" and a.status == 200
    assert a.sent_s == 0.0
    assert a.ttft_s == pytest.approx(0.5)
    assert a.latency_s == pytest.approx(0.9)  # 0.5 + 4 x 0.1
    assert b.sent_s == pytest.approx(0.9)  # sent when a finished: one at a time
    assert (a.prompt_tokens, a.output_tokens, a.finish_reason) == (1000, 5, "stop")
    assert a.content == "t0 t1 t2 t3 t4 "
    row = a.record(variant="merged", level=1)
    assert list(row)[:5] == ["variant", "level", "phase", "index", "label"]
    assert row["ttft_s"] == pytest.approx(0.5)
    assert row["latency_s"] == pytest.approx(0.9)
    assert json.loads(json.dumps(row)) == row


def test_errors_are_results_not_retries() -> None:
    clock = Clock()
    server = FakeServer(
        clock,
        failures={
            "long": "http",
            "down": "connect",
            "died": "midway",
            "quiet": "no_usage",
            "cut": "no_done",
        },
    )
    _, results = loadtest.run_level(
        URL,
        [],
        requests("ok", "long", "down", "died", "quiet", "cut"),
        2,
        transport=httpx.MockTransport(server.handler),
        clock=clock,
    )
    errors = {r.label: r.error for r in results}
    assert errors["ok"] == ""
    assert errors["long"] == (
        "HTTP 400 (This model's maximum context length is 40960 tokens.)"
    )
    assert errors["down"] == "ConnectError: connection refused"
    assert errors["died"] == "error in the stream: engine died"
    assert errors["quiet"] == "no usage in the stream"
    assert errors["cut"] == "the stream ended without [DONE]"
    assert len(server.bodies) == 6  # one call each: nothing retried
    by_label = {r.label: r for r in results}
    assert by_label["long"].status == 400 and by_label["down"].status is None
    summary = loadtest.summarize(results)
    assert (summary["requests"], summary["ok"], summary["errors"]) == (6, 1, 5)
    assert summary["error_messages"][0].startswith("HTTP 400")


@pytest.mark.parametrize("limit", [1, 3, 4])
def test_the_concurrency_limit(limit) -> None:
    # The first `limit` requests wait for each other, so all must be in flight at
    # once (a smaller pool times out). They then stay in flight long enough for one
    # more to start if the pool allowed it; it may start only when one finishes.
    state = {"now": 0, "max": 0, "started": 0}
    barrier = asyncio.Barrier(limit)

    async def events():
        state["now"] += 1
        state["started"] += 1
        state["max"] = max(state["max"], state["now"])
        if state["started"] <= limit:
            await asyncio.wait_for(barrier.wait(), timeout=5)
            await asyncio.sleep(0.05)
        state["now"] -= 1
        yield sse({"choices": [{"index": 0, "delta": {"content": "x"}}]})
        yield sse(
            {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        )
        yield sse("[DONE]")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=events())

    labels_ = [f"r{n}" for n in range(10)]
    _, results = loadtest.run_level(
        URL, [], requests(*labels_), limit, transport=httpx.MockTransport(handler)
    )
    assert state["max"] == limit
    assert [r.label for r in results] == labels_
    assert all(r.error == "" for r in results)


def result(sent, ttft, latency, output=11, prompt=1000, error="") -> Result:
    return Result(
        "x",
        "measured",
        0,
        sent,
        sent + latency,
        None if ttft is None else sent + ttft,
        prompt,
        output,
        "stop",
        200 if not error else 500,
        error,
        "",
    )


def test_summary_arithmetic() -> None:
    results = [
        result(0.0, 1.0, 2.0),  # decode 10 / 1.0 = 10 tokens/s
        result(0.5, 1.0, 3.0),  # 10 / 2.0 = 5
        result(1.0, 2.0, 4.5),  # 10 / 2.5 = 4; finishes at 5.5, the last
        result(2.0, 0.5, 1.0, output=21),  # 20 / 0.5 = 40
        result(3.0, None, 0.2, output=None, prompt=None, error="HTTP 500"),
        result(4.0, 0.2, 0.3, output=1),  # one token: no decode speed
    ]
    summary = loadtest.summarize(results)
    assert summary["requests"] == 6 and summary["ok"] == 5 and summary["errors"] == 1
    assert summary["makespan_s"] == 5.5  # 0.0 to 5.5
    assert summary["releases_per_hour"] == pytest.approx(5 * 3600 / 5.5)
    assert summary["prompt_tokens"] == 5000
    assert summary["output_tokens"] == 11 + 11 + 11 + 21 + 1
    assert summary["prompt_tokens_per_s"] == pytest.approx(5000 / 5.5)
    assert summary["output_tokens_per_s"] == pytest.approx(55 / 5.5)
    # Latencies of the 5 without an error: 0.3, 1.0, 2.0, 3.0, 4.5 (nearest rank).
    assert summary["latency_s"] == {"p50": 2.0, "p95": 4.5, "p99": 4.5, "max": 4.5}
    # Times to first token: 0.2, 0.5, 1.0, 1.0, 2.0.
    assert summary["ttft_s"] == {"p50": 1.0, "p95": 2.0}
    # Decode speeds 4, 5, 10, 40: the nearest-rank median is the 2nd, 5.
    assert summary["decode_tokens_per_s_p50"] == pytest.approx(5.0)
    assert summary["error_messages"] == ["HTTP 500"]


def test_percentiles_are_nearest_rank() -> None:
    values = [float(n) for n in range(1, 129)]  # 128 latencies, 1..128
    results = [result(0.0, 0.5, v) for v in values]
    latency = loadtest.summarize(results)["latency_s"]
    assert latency == {"p50": 64.0, "p95": 122.0, "p99": 127.0, "max": 128.0}


def test_an_empty_level() -> None:
    summary = loadtest.summarize([])
    assert summary["makespan_s"] is None and summary["releases_per_hour"] is None
    assert summary["latency_s"]["p50"] is None


def test_write_jsonl(tmp_path) -> None:
    path = tmp_path / "load" / "requests.jsonl"
    loadtest.write_jsonl(path, [{"a": 1}, {"b": "é"}])
    assert path.read_text(encoding="utf-8") == '{"a": 1}\n{"b": "é"}\n'
