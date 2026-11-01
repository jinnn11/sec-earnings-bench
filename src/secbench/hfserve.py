"""A stand-in for vLLM's OpenAI-compatible server, for the laptop smoke test only
(scripts/smoke_mac.sh), where vLLM doesn't run: transformers and peft on the CPU or
Apple GPU, greedy decoding, one request at a time. Scores from it show only that the
pipeline runs; results come from vLLM.

It answers what secbench.score and secbench.extract ask of vLLM, in vLLM's shapes:
GET /health; POST /tokenize with chat messages (the model's chat template with the
request's add_generation_prompt and chat_template_kwargs, then the tokens with no
special tokens added; {"count", "max_model_len", "tokens"}); POST
/v1/chat/completions (temperature 0 or none; max_tokens; chat_template_kwargs). A
reply ends at an end-of-sequence token of the model's generation config or after
max_tokens (or what --max-model-len leaves); a prompt that leaves no room for
max_tokens is HTTP 400, as in vLLM. The base model is served as
--served-model-name, each --lora-modules NAME=PATH adapter as NAME; another name is
HTTP 404.

Usage: uv run --group train python -m secbench.hfserve SNAPSHOT
           --served-model-name NAME [--lora-modules NAME=PATH ...] [--port 8000]
           [--max-model-len 40960]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from secbench import chat
from secbench.train import encode, load_model, pick_device


class RequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def parse_adapter(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not (sep and name and path):
        raise argparse.ArgumentTypeError("must be NAME=PATH")
    return name, Path(path)


class Model:
    """The base model and its adapters, loaded once."""

    def __init__(
        self,
        snapshot: Path,
        served: str,
        adapters: Sequence[tuple[str, Path]],
        max_model_len: int,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        self.served = served
        self.names = {served, *(name for name, _ in adapters)}
        self.max_model_len = max_model_len
        self.device = pick_device("auto")
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot)
        model = load_model(snapshot, dtype, self.device)
        if adapters:
            from peft import PeftModel

            (first, path), *others = adapters
            model = PeftModel.from_pretrained(model, path, adapter_name=first)
            for name, path in others:
                model.load_adapter(path, adapter_name=name)
        self.model = model.eval()
        eos = model.generation_config.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]
        self.pad = model.generation_config.pad_token_id or self.eos[0]
        self.lock = threading.Lock()

    def render(self, body: Mapping[str, Any], generation_prompt: bool) -> list[int]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RequestError(400, "messages must be a non-empty list")
        kwargs = body.get("chat_template_kwargs") or {}
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt, **kwargs
        )
        return encode(self.tokenizer, text)

    def tokenize(self, body: Mapping[str, Any]) -> dict[str, Any]:
        if body.get("add_special_tokens"):
            raise RequestError(400, "add_special_tokens is not supported here")
        ids = self.render(body, bool(body.get("add_generation_prompt", True)))
        return {"count": len(ids), "max_model_len": self.max_model_len, "tokens": ids}

    @contextlib.contextmanager
    def serving(self, name: str) -> Iterator[None]:
        if name not in self.names:
            raise RequestError(404, f"The model `{name}` does not exist.")
        if not hasattr(self.model, "disable_adapter"):
            yield
        elif name == self.served:
            with self.model.disable_adapter():
                yield
        else:
            self.model.set_adapter(name)
            yield

    def complete(self, body: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        from transformers import GenerationConfig

        if body.get("temperature") not in (None, 0, 0.0):
            raise RequestError(400, "this stand-in decodes greedily: temperature 0")
        if body.get("stream"):
            raise RequestError(400, "stream is not supported here")
        name = str(body.get("model"))
        ids = self.render(body, generation_prompt=True)
        room = self.max_model_len - len(ids)
        max_tokens = body.get("max_tokens") or room
        if max_tokens > room:
            raise RequestError(
                400,
                f"This model's maximum context length is {self.max_model_len} "
                f"tokens. However, you requested {len(ids) + max_tokens} tokens "
                f"({len(ids)} in the messages, {max_tokens} in the completion).",
            )
        with self.lock, self.serving(name), torch.inference_mode():
            prompt = torch.tensor([ids], device=self.device)
            output = self.model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                generation_config=GenerationConfig(
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    eos_token_id=self.eos,
                    pad_token_id=self.pad,
                ),
            )
        new = output[0, len(ids) :].tolist()
        finish = "stop" if new and new[-1] in self.eos else "length"
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return {
            "id": f"chatcmpl-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }
            ],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": len(new),
                "total_tokens": len(ids) + len(new),
            },
        }


def handler(model: Model) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.reply(200, {})
            elif self.path == "/v1/models":
                data = [{"id": name, "object": "model"} for name in sorted(model.names)]
                self.reply(200, {"object": "list", "data": data})
            else:
                self.reply(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self) -> None:
            routes = {
                "/tokenize": model.tokenize,
                "/v1/chat/completions": model.complete,
            }
            route = routes.get(self.path)
            try:
                if route is None:
                    raise RequestError(404, f"no route {self.path}")
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise RequestError(400, "the body must be a JSON object")
                self.reply(200, route(body))
            except RequestError as exc:
                self.reply(
                    exc.status, {"error": {"message": str(exc), "code": exc.status}}
                )

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write(f"{self.address_string()} {format % args}\n")

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.hfserve",
        description="A stand-in for vLLM's server (laptop smoke test only).",
    )
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument(
        "--lora-modules", nargs="*", type=parse_adapter, default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=chat.MAX_MODEL_LEN)
    args = parser.parse_args(argv)
    model = Model(
        args.snapshot, args.served_model_name, args.lora_modules, args.max_model_len
    )
    server = ThreadingHTTPServer((args.host, args.port), handler(model))
    print(
        f"serving {', '.join(sorted(model.names))} on http://{args.host}:{args.port} "
        f"({model.device})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
