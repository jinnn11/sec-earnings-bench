"""secbench.hfserve's streamed replies, on the tiny random Qwen3 (the train dependency
group): the same reply as without streaming, chunked as vLLM chunks it, and read by
secbench.loadtest over HTTP."""

import threading
from http.server import ThreadingHTTPServer

import pytest
from test_score import adapter  # noqa: F401 (a fixture)

from secbench import loadtest

MESSAGES = [
    {"role": "system", "content": "Reply with JSON."},
    {"role": "user", "content": "Revenue 101."},
]
BODY = {
    "messages": MESSAGES,
    "temperature": 0,
    "max_tokens": 6,
    "chat_template_kwargs": {"enable_thinking": False},
}


@pytest.fixture(scope="module")
def model(tiny_snapshot, adapter):  # noqa: F811
    from secbench import hfserve

    return hfserve.Model(tiny_snapshot, "base", [("tuned", adapter)], 4096)


@pytest.mark.parametrize("name", ["base", "tuned"])
def test_a_streamed_reply_is_the_reply(model, name) -> None:
    whole = model.complete({**BODY, "model": name})
    chunks = list(
        model.complete_stream(
            {
                **BODY,
                "model": name,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
    )
    first, *pieces, finish, usage = chunks
    assert first["object"] == "chat.completion.chunk" and first["model"] == name
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"]["content"] for c in pieces)
    assert text == whole["choices"][0]["message"]["content"]
    assert all(c["choices"][0]["finish_reason"] is None for c in pieces)
    assert finish["choices"][0] == {
        "index": 0,
        "delta": {},
        "finish_reason": whole["choices"][0]["finish_reason"],
    }
    assert usage["choices"] == [] and usage["usage"] == whole["usage"]
    # Without include_usage, no usage chunk.
    plain = list(model.complete_stream({**BODY, "model": name, "stream": True}))
    assert "usage" not in plain[-1] and plain[-1]["choices"][0]["finish_reason"]


def test_the_load_client_reads_the_stand_ins_stream(model) -> None:
    from secbench import hfserve

    server = ThreadingHTTPServer(("127.0.0.1", 0), hfserve.handler(model))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        whole = model.complete({**BODY, "model": "tuned"})
        streamed = {**BODY, **loadtest.STREAM}
        _, results = loadtest.run_level(
            url,
            [],
            [
                loadtest.Request("tuned", {**streamed, "model": "tuned"}),
                loadtest.Request("other", {**streamed, "model": "other"}),
            ],
            1,
        )
    finally:
        server.shutdown()
        server.server_close()
    tuned, other = results
    assert tuned.error == "" and tuned.status == 200
    assert tuned.content == whole["choices"][0]["message"]["content"]
    assert tuned.output_tokens == whole["usage"]["completion_tokens"]
    assert tuned.prompt_tokens == whole["usage"]["prompt_tokens"]
    assert tuned.ttft_s is not None and 0 < tuned.ttft_s <= tuned.latency_s
    # A request refused before the stream starts is an HTTP error.
    assert other.status == 404
    assert other.error == "HTTP 404 (The model `other` does not exist.)"
