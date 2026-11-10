"""secbench.servebench against a fake vLLM (httpx.MockTransport: /tokenize, chat
completions, streamed or not), with fake server starts and model builds, in a folder
laid out as the repository (test_splits' repo: AAPL and MSFT validation, frozen; the
byte tokenizer). The load test and the accuracy runs both use validation here: the
fixture's time test is over the context limit by the byte tokenizer's count."""

import contextlib
import hashlib
import json
import random
import shlex
import shutil
from pathlib import Path

import httpx
import pytest
from conftest import byte_tokenizer
from test_score import (  # noqa: F401 (key_dir and perfect are fixtures)
    FakeVLLM,
    add_checkpoint,
    encode,
    key_dir,
    perfect,
)
from test_serving import write_serving_config
from test_splits import AAPL, MSFT, PROMPTS, repo  # noqa: F401 (repo is a fixture)
from test_splits import run as build_splits
from test_train import prepare, write_labels

from secbench import quantize, score, servebench, serving, train
from secbench.serving import ServingError

WEIGHTS = b"adapter weights"
CONFIG = Path("configs/serving/run.toml")
SNAPSHOT = Path("/snapshots/tiny")
VLLM_LOG = """\
(EngineCore_DP0 pid=7) INFO 11-10 12:00:01 [gpu_model_runner.py:2602] Model loading took 15.2683 GiB memory and 4.512 seconds
(EngineCore_DP0 pid=7) INFO 11-10 12:00:30 [gpu_worker.py:298] Available KV cache memory: 52.34 GiB
(EngineCore_DP0 pid=7) INFO 11-10 12:00:30 [kv_cache_utils.py:1087] GPU KV cache size: 381,120 tokens
(EngineCore_DP0 pid=7) INFO 11-10 12:00:30 [kv_cache_utils.py:1091] Maximum concurrency for 40,960 tokens per request: 9.30x
(EngineCore_DP0 pid=7) INFO 11-10 12:00:52 [gpu_model_runner.py:3480] Graph capturing finished in 14 secs, took 0.61 GiB
(EngineCore_DP0 pid=7) INFO 11-10 12:00:52 [core.py:210] init engine (profile, create kv cache, warmup model) took 47.80 seconds
(APIServer pid=1) INFO 11-10 12:00:55 [api_server.py:1900] Starting vLLM API server 0 on http://127.0.0.1:8000
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sse(chunk) -> bytes:
    return (
        f"data: {chunk if isinstance(chunk, str) else json.dumps(chunk)}\n\n".encode()
    )


class StreamingVLLM(FakeVLLM):
    """FakeVLLM, plus streamed completions: the reply in pieces of 7 characters,
    the finish reason, usage and [DONE]."""

    def __init__(self, reply) -> None:
        super().__init__(reply)
        self.streamed: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not body.get("stream"):
            return super().handler(request)
        self.streamed.append(body)
        content = self.reply(body["model"], body["messages"][1]["content"])
        pieces = [content[i : i + 7] for i in range(0, len(content), 7)]
        role = {"index": 0, "delta": {"role": "assistant", "content": ""}}
        events = [sse({"choices": [role]})]
        events += [
            sse({"choices": [{"index": 0, "delta": {"content": p}}]}) for p in pieces
        ]
        events += [
            sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            sse(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": len(pieces),
                    },
                }
            ),
            sse("[DONE]"),
        ]
        return httpx.Response(200, content=b"".join(events))


@pytest.fixture
def lab(repo) -> Path:  # noqa: F811
    """The repository folder: frozen validation (and company_test), a prepared
    training run with a checkpoint at step 2 (r 4) whose adapter is WEIGHTS, the
    calibration examples, and a serving config on validation: 1 measured prompt
    after 1 warm-up, at 1 and 2 in flight, accuracy runs 4 at a time."""
    assert build_splits("--freeze", "company_test", "validation") == 0
    assert build_splits() == 0
    shutil.copytree(PROMPTS, repo / "prompts")
    write_labels(repo / "labels")
    run_dir = prepare(repo)
    add_checkpoint(run_dir, 2, 4)
    (run_dir / "checkpoints/step-00002/adapter_model.safetensors").write_bytes(WEIGHTS)
    write_serving_config(
        repo / CONFIG,
        load__split='"validation"',
        load__prompts="1",
        load__warmup="1",
        accuracy__splits='["validation"]',
        calibration__examples="2",
    )
    config = serving.load_config(CONFIG)
    quantize.make_calibration(
        config,
        labels_dir=repo / "labels",
        tokenizer=byte_tokenizer(),
        tokenizer_path=repo / "tokenizer.json",
        splits_dir=Path("data/splits"),
        frozen=Path("data/frozen_test_releases.csv"),
        out=config.calibration_dir,
        commit={},
    )
    python = repo / ".venv-quantize/bin/python"  # the quantize environment's
    python.parent.mkdir(parents=True)
    python.write_text("")
    return repo


def starter(events: list, fail: str | None = None):
    @contextlib.contextmanager
    def start(command, log_path, root, http):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            log.write(f"== 2026-11-10T12:00:00+00:00 {shlex.join(command)}\n")
            log.write(VLLM_LOG)
        served = command[command.index("--served-model-name") + 1]
        if fail and served.endswith(fail):
            raise score.ScoreError(f"the server exited (1) before it was ready: {fail}")
        events.append(("start", Path(command[2]).name, served))
        yield
        events.append(("stop", served))

    return start


def builder(events: list, fail: str | None = None):
    def run(command, log_path) -> int:
        kind = command[command.index("secbench.quantize") + 1]
        events.append((kind, command[0]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if kind == fail:
            log_path.write_text(
                "Traceback ...\ntorch.OutOfMemoryError: CUDA out of memory\n"
            )
            return 1
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True)

        def arg(name: str) -> str:
            return command[command.index(name) + 1]

        if kind == "merge":
            manifest = {
                "kind": "merged",
                "inputs": {
                    "base": {"path": arg("--base")},
                    "adapter": {
                        "files": {"adapter_model.safetensors": sha256(WEIGHTS)}
                    },
                },
            }
        else:
            manifest = {
                "kind": "int4-awq",
                "inputs": {
                    "model": {
                        "manifest_sha256": train.file_sha256(
                            Path(arg("--model")) / "manifest.json"
                        )
                    },
                    "calibration": {
                        "sha256": train.file_sha256(Path(arg("--calibration")))
                    },
                },
                "settings": {"scheme": arg("--scheme")},
            }
        (out / "manifest.json").write_text(json.dumps(manifest))
        return 0

    return run


def bench(
    lab: Path,
    server: StreamingVLLM,
    *args: str,
    events: list | None = None,
    fail_start: str | None = None,
    fail_build: str | None = None,
) -> int:
    events = [] if events is None else events
    return servebench.main(
        [
            "--config", str(CONFIG),
            "--gpu-hourly", "2.5",
            *args,
        ],
        transport=httpx.MockTransport(server.handler),
        start_server=starter(events, fail_start),
        run_command=builder(events, fail_build),
        snapshot=SNAPSHOT,
        encode=encode,
    )  # fmt: skip


def served(variant: str) -> str:
    return "run--step-00002" if variant == "adapter" else f"run--step-00002--{variant}"


def test_every_variant_in_order(lab, perfect, capsys) -> None:  # noqa: F811
    server, events = StreamingVLLM(lambda model, text: perfect[text]), []
    assert bench(lab, server, events=events) == 0
    out = capsys.readouterr().out
    models = lab / "training/run/serving/models"
    assert events == [
        ("start", "tiny", "test/tiny-qwen3"),
        ("stop", "test/tiny-qwen3"),
        ("merge", servebench.sys.executable),
        ("start", "merged", served("merged")),
        ("stop", served("merged")),
        ("start", "merged", served("fp8")),
        ("stop", served("fp8")),
        ("awq", ".venv-quantize/bin/python"),
        ("start", "int4-awq", served("int4-awq")),
        ("stop", served("int4-awq")),
    ]
    assert "Plan (vLLM " in out and "adapter, merged, fp8, int4-awq" in out
    assert "estimate: ~" in out and "at $2.5/h" in out

    serving_dir = lab / "training/run/serving"
    record = json.loads((serving_dir / "servebench.json").read_text())
    assert record["gpu_hourly_usd"] == 2.5 and record["snapshot"] == str(SNAPSHOT)
    assert {v: r["status"] for v, r in record["variants"].items()} == dict.fromkeys(
        ("adapter", "merged", "fp8", "int4-awq"), "complete"
    )
    assert record["models"]["merged"]["status"] == "built"
    assert record["models"]["int4-awq"]["status"] == "built"
    plan = json.loads((serving_dir / "plan.json").read_text())
    pool = [AAPL, MSFT]
    drawn = random.Random(7).sample(range(2), 2)
    assert plan["load"]["measured"] == [pool[drawn[0]]]
    assert plan["load"]["warmup"] == [pool[drawn[1]]]
    assert plan["accuracy"] == {"validation": 2} and plan["layout_prompts"] == 2

    for variant in ("adapter", "merged", "fp8", "int4-awq"):
        folder = serving_dir / variant
        result = json.loads((folder / "variant.json").read_text())
        assert result == record["variants"][variant]
        assert result["layout_check"] == {
            "prompts": 2,
            "identical_to_training_layout": True,
        }
        startup = json.loads((folder / "startup.json").read_text())
        assert startup["weights_gib"] == 15.2683
        assert startup["kv_cache_tokens"] == 381_120
        assert startup["max_concurrency"] == 9.30
        summary = json.loads((folder / "load/summary.json").read_text())
        assert [level["level"] for level in summary["levels"]] == [1, 2]
        assert all(level["errors"] == 0 for level in summary["levels"])
        assert all(level["requests"] == 1 for level in summary["levels"])
        assert summary["request"]["parameters"] == score.expected_parameters()
        rows = [
            json.loads(line)
            for line in (folder / "load/requests.jsonl").read_text().splitlines()
        ]
        assert [(r["level"], r["phase"]) for r in rows] == [
            (1, "warmup"),
            (1, "measured"),
            (2, "warmup"),
            (2, "measured"),
        ]
        run = lab / "runs" / f"run--serving--{variant}--validation"
        info = json.loads((run / "run.json").read_text())
        assert info["model"] == served(variant) and info["status"] == "complete"
        assert info["request_parameters"] == score.expected_parameters()
        assert (info["max_parallel"], info["gpu_hourly_usd"]) == (4, 2.5)
        assert info["skip_grading"] is True
        assert info["links"] == "data/splits/validation/links.csv"
        assert result["runs"][0]["run"] == str(Path("runs") / run.name)
    # Streamed requests are extract's, plus streaming.
    for body in server.streamed:
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert {k: body[k] for k in score.expected_parameters()} == (
            score.expected_parameters()
        )
    assert len(server.streamed) == 4 * 2 * 2  # 4 variants, 2 levels, 2 requests
    assert {b["model"] for b in server.completions} == {
        served(v) for v in ("adapter", "merged", "fp8", "int4-awq")
    }
    assert (models / "merged/manifest.json").is_file()

    # Again: every variant is kept, and nothing starts.
    server, events = StreamingVLLM(lambda model, text: perfect[text]), []
    assert bench(lab, server, events=events) == 0
    assert events == [] and server.streamed == [] and server.tokenized == 0
    assert "kept from an earlier session: adapter, merged, fp8, int4-awq" in (
        capsys.readouterr().out
    )


def test_a_failed_merge_fails_its_variants_only(lab, perfect, capsys) -> None:  # noqa: F811
    server, events = StreamingVLLM(lambda model, text: perfect[text]), []
    assert bench(lab, server, events=events, fail_build="merge") == 1
    assert [e[:2] for e in events] == [
        ("start", "tiny"),
        ("stop", "test/tiny-qwen3"),
        ("merge", servebench.sys.executable),  # tried once only
    ]
    record = json.loads((lab / "training/run/serving/servebench.json").read_text())
    statuses = {v: r["status"] for v, r in record["variants"].items()}
    assert statuses == {
        "adapter": "complete",
        "merged": "failed",
        "fp8": "failed",
        "int4-awq": "failed",
    }
    assert "exited 1" in record["variants"]["merged"]["error"]
    assert "CUDA out of memory" in record["variants"]["merged"]["error"]
    assert "needs the merged model" in record["variants"]["fp8"]["error"]
    assert record["models"]["merged"]["status"] == "failed"
    out = capsys.readouterr().out
    assert "merged: failed: ServingError: building" in out

    # A rerun keeps the adapter and builds the rest.
    server, events = StreamingVLLM(lambda model, text: perfect[text]), []
    assert bench(lab, server, events=events) == 0
    assert events[0] == ("merge", servebench.sys.executable)
    assert ("start", "tiny", "test/tiny-qwen3") not in events
    record = json.loads((lab / "training/run/serving/servebench.json").read_text())
    assert record["variants"]["adapter"]["kept"] is True
    assert all(r["status"] == "complete" for r in record["variants"].values())


def test_a_server_that_does_not_start_fails_that_variant(lab, perfect) -> None:  # noqa: F811
    server, events = StreamingVLLM(lambda model, text: perfect[text]), []
    assert bench(lab, server, events=events, fail_start="--fp8") == 1
    record = json.loads((lab / "training/run/serving/servebench.json").read_text())
    assert record["variants"]["fp8"]["status"] == "failed"
    assert "before it was ready" in record["variants"]["fp8"]["error"]
    assert record["variants"]["int4-awq"]["status"] == "complete"
    assert ("start", "int4-awq", served("int4-awq")) in events


def test_an_incomplete_run_is_moved_aside(lab, perfect, capsys) -> None:  # noqa: F811
    stale = lab / "runs" / "run--serving--merged--validation"
    stale.mkdir(parents=True)
    (stale / "run.json").write_text('{"status": "aborted"}')
    server = StreamingVLLM(lambda model, text: perfect[text])
    assert bench(lab, server, "--variants", "merged") == 0
    moved = lab / "runs" / "run--serving--merged--validation--attempt-1"
    assert json.loads((moved / "run.json").read_text()) == {"status": "aborted"}
    info = json.loads((stale / "run.json").read_text())
    assert info["status"] == "complete"
    record = json.loads((lab / "training/run/serving/merged/variant.json").read_text())
    assert record["runs"][0]["moved_to"] == str(Path("runs") / moved.name)
    assert "wasn't a complete run with these settings: moved to" in (
        capsys.readouterr().out
    )


def test_the_laptop_stand_in_skips_fp8_and_awq(lab, perfect, capsys) -> None:  # noqa: F811
    server, events = StreamingVLLM(lambda model, text: perfect[text]), []
    assert bench(lab, server, "--server", "hf", events=events) == 0
    assert [e[0] for e in events] == ["start", "stop", "merge", "start", "stop"]
    record = json.loads((lab / "training/run/serving/servebench.json").read_text())
    assert record["variants"]["fp8"]["status"] == "skipped"
    assert record["variants"]["int4-awq"]["status"] == "skipped"
    assert "needs vLLM on a GPU" in record["variants"]["fp8"]["error"]
    assert "int4-awq" not in record["models"]
    commands = [
        json.loads((lab / f"training/run/serving/{v}/variant.json").read_text())[
            "server"
        ]["command"]
        for v in ("adapter", "merged")
    ]
    assert all(c[1:3] == ["-m", "secbench.hfserve"] for c in commands)


def test_no_awq_without_the_quantize_environment(lab, perfect) -> None:  # noqa: F811
    (lab / ".venv-quantize/bin/python").unlink()
    server = StreamingVLLM(lambda model, text: perfect[text])
    assert bench(lab, server, "--variants", "int4-awq") == 1
    result = json.loads(
        (lab / "training/run/serving/int4-awq/variant.json").read_text()
    )
    assert "no Python at" in result["error"] and "quantize group" in result["error"]


def test_a_kept_model_from_other_inputs_stops(lab, perfect) -> None:  # noqa: F811
    folder = lab / "training/run/serving/models/merged"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "base": {"path": str(SNAPSHOT)},
                    "adapter": {"files": {"adapter_model.safetensors": "other"}},
                }
            }
        )
    )
    server = StreamingVLLM(lambda model, text: perfect[text])
    assert bench(lab, server, "--variants", "merged") == 1
    result = json.loads((lab / "training/run/serving/merged/variant.json").read_text())
    assert "was made from an adapter with sha256 other" in result["error"]


def test_plan_only_runs_nothing(lab, perfect, capsys) -> None:  # noqa: F811
    server, events = StreamingVLLM(lambda model, text: perfect[text]), []
    assert bench(lab, server, "--plan-only", events=events) == 0
    assert events == [] and not (lab / "training/run/serving/plan.json").exists()
    assert "Plan (" in capsys.readouterr().out


def test_another_adapter_stops_before_anything(lab, perfect, capsys) -> None:  # noqa: F811
    weights = lab / "training/run/checkpoints/step-00002/adapter_model.safetensors"
    weights.write_bytes(b"other weights")
    server, events = StreamingVLLM(lambda model, text: perfect[text]), []
    assert bench(lab, server, events=events) == 1
    assert events == []
    assert "not the serving config's" in capsys.readouterr().err


def test_server_commands() -> None:
    config = serving.ServingConfig(
        **{
            **{f: None for f in serving.ServingConfig.__dataclass_fields__},
            "path": CONFIG,
            "run": "qwen3-8b-2k",
            "base": "Qwen/Qwen3-8B",
            "checkpoint": "step-00025",
            "max_model_len": 40960,
            "prefix_caching": False,
        }
    )
    adapter = score.Model(
        "adapter", "qwen3-8b-2k--step-00025", "x", 25, 400, Path("/ckpt/step-00025")
    )
    models = Path("/models")

    def command(variant, server="vllm"):
        model_dir = None
        if serving.MODEL_OF[variant]:
            model_dir = models / serving.MODEL_OF[variant]
        found = servebench.server_command(
            variant,
            config,
            server=server,
            snapshot=Path("/snap"),
            adapter=adapter,
            model_dir=model_dir,
            port=8000,
            r=16,
        )
        return found[1:]

    common = ["--host", "127.0.0.1", "--port", "8000"]
    assert command("adapter") == [
        "serve", "/snap",
        "--served-model-name", "Qwen/Qwen3-8B",
        "--max-model-len", "40960",
        *common,
        "--enable-lora",
        "--max-lora-rank", "16",
        "--max-loras", "1",
        "--max-cpu-loras", "1",
        "--lora-modules", "qwen3-8b-2k--step-00025=/ckpt/step-00025",
        "--no-enable-prefix-caching",
    ]  # fmt: skip
    assert command("merged") == [
        "serve", "/models/merged",
        "--served-model-name", "qwen3-8b-2k--step-00025--merged",
        "--max-model-len", "40960",
        *common,
        "--no-enable-prefix-caching",
    ]  # fmt: skip
    assert command("fp8") == [
        "serve", "/models/merged",
        "--served-model-name", "qwen3-8b-2k--step-00025--fp8",
        "--max-model-len", "40960",
        *common,
        "--quantization", "fp8",
        "--no-enable-prefix-caching",
    ]  # fmt: skip
    assert command("int4-awq") == [
        "serve", "/models/int4-awq",
        "--served-model-name", "qwen3-8b-2k--step-00025--int4-awq",
        "--max-model-len", "40960",
        *common,
        "--no-enable-prefix-caching",
    ]  # fmt: skip
    # The adapter variant is exactly stage 2's server, prefix caching aside.
    stage2 = score.vllm_command(Path("/snap"), "Qwen/Qwen3-8B", [adapter], 8000, 16)
    assert command("adapter")[:-1] == stage2[1:]
    # On the laptop stand-in.
    assert command("merged", "hf")[:4] == [
        "-m",
        "secbench.hfserve",
        "/models/merged",
        "--served-model-name",
    ]
    assert servebench.tokenize_name("adapter", config) == "Qwen/Qwen3-8B"
    assert servebench.tokenize_name("fp8", config) == "qwen3-8b-2k--step-00025--fp8"


def test_parse_startup() -> None:
    found = servebench.parse_startup(VLLM_LOG)
    assert {k: v for k, v in found.items() if k != "lines"} == {
        "weights_gib": 15.2683,
        "kv_cache_gib": 52.34,
        "kv_cache_tokens": 381_120,
        "max_concurrency": 9.30,
        "max_concurrency_tokens": 40_960,
        "engine_init_s": 47.80,
        "cuda_graphs_s": 14.0,
    }
    assert found["lines"]["kv_cache_tokens"] == "GPU KV cache size: 381,120 tokens"
    # Older wording, and lines missing: null, never an error.
    older = "INFO Model loading took 5.52 GiB and 3.1 seconds\nnothing else\n"
    found = servebench.parse_startup(older)
    assert found["weights_gib"] == 5.52
    assert found["kv_cache_tokens"] is None and found["max_concurrency"] is None
    assert servebench.parse_startup("")["weights_gib"] is None


def test_last_session(tmp_path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "== 2026-11-10T10:00:00+00:00 vllm serve a\nModel loading took 1.0 GiB\n"
        "== 2026-11-10T11:00:00+00:00 vllm serve b\nModel loading took 2.0 GiB\n"
    )
    assert servebench.last_session(log) == "Model loading took 2.0 GiB\n"
    assert servebench.last_session(tmp_path / "none.log") == ""


def test_load_draw(tmp_path) -> None:
    config = serving.load_config(
        write_serving_config(tmp_path / "c.toml", load__prompts="3", load__warmup="2")
    )
    prompts = [(f"time_test {n}", "s", f"u{n}") for n in range(10)]
    measured, warmup = servebench.load_draw(config, prompts)
    drawn = random.Random(7).sample(range(10), 5)
    assert measured == [prompts[i] for i in drawn[:3]]
    assert warmup == [prompts[i] for i in drawn[3:]]
    assert not set(measured) & set(warmup)
    pooled = serving.load_config(
        write_serving_config(
            tmp_path / "p.toml", load__prompts="3", load__warmup="2", load__pool="5"
        )
    )
    measured, warmup = servebench.load_draw(pooled, prompts)
    assert {p[0] for p in (*measured, *warmup)} == {f"time_test {n}" for n in range(5)}
    with pytest.raises(ServingError, match="needs 5 prompts; time_test has 4"):
        servebench.load_draw(config, prompts[:4])
