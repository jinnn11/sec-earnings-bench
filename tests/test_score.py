import contextlib
import csv
import io
import json
import socket
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from conftest import byte_tokenizer
from test_train import prepare, write_labels

from secbench import answers, chat, extract, score, train
from secbench.answers import FULL_YEAR, QUARTER, YEAR_TO_DATE, load_links
from secbench.edgar import EdgarClient, RateLimiter
from secbench.grade import FIELDS, load_answers
from secbench.score import Model, ScoreError

FIXTURES = Path(__file__).parent / "fixtures"
LINKS = FIXTURES / "answers_links.csv"
COMPANYFACTS = FIXTURES / "answers_companyfacts.json"
SUBMISSIONS = FIXTURES / "link_submissions.json"
RELEASES = FIXTURES / "input_releases"
AAPL = "0000320193-25-000055"
FIXTURE_RELEASES = [  # links.csv order
    AAPL,
    "0000018230-25-000037",
    "0000950170-25-100226",
    "0000104169-25-000069",
    "0002115436-26-000006",
]
BASE_NAME = "test/tiny-qwen3"
ROOT = f"http://127.0.0.1:{score.DEFAULT_PORT}"


def make_client(cache_dir: Path) -> EdgarClient:
    """An EDGAR client serving the submissions and company facts fixtures."""
    bodies = json.loads(SUBMISSIONS.read_bytes()) | json.loads(
        COMPANYFACTS.read_bytes()
    )
    routes = {url: json.dumps(body).encode() for url, body in bodies.items()}

    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(str(request.url))
        return (
            httpx.Response(404) if body is None else httpx.Response(200, content=body)
        )

    return EdgarClient(
        "Test Runner test@example.com",
        cache_dir,
        transport=httpx.MockTransport(handler),
        limiter=RateLimiter(8.0, clock=lambda: 0.0, sleep=lambda s: None),
    )


@pytest.fixture(scope="module")
def key_dir(tmp_path_factory) -> Path:
    """links.csv cut to the five input_releases fixtures, and answers.csv for them."""
    tmp = tmp_path_factory.mktemp("key")
    lines = LINKS.read_text().splitlines()
    kept = [lines[0]] + [x for x in lines[1:] if x.split(",")[2] in FIXTURE_RELEASES]
    (tmp / "links.csv").write_text("\n".join(kept) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        status = answers.main(
            ["--links", str(tmp / "links.csv"), "--out", str(tmp / "answers.csv")],
            client=make_client(tmp / "raw"),
        )
    assert status == 0
    return tmp


def figure(value: Decimal | None, field: str):
    if value is None:
        return None
    number = int(value) if value == value.to_integral_value() else float(value)
    return number if field == "eps_diluted" else {"value": number, "unit": "units"}


@pytest.fixture(scope="module")
def perfect(key_dir) -> dict[str, str]:
    """Release input text -> an extract_v2 reply giving every answer."""
    key = load_answers(key_dir / "answers.csv")
    records = {p.stem: p for p in sorted(RELEASES.glob("*/*.json"))}
    replies = {}
    for release in load_links(key_dir / "links.csv")[0]:
        accession = release.release_accession
        year = FULL_YEAR if release.linked_form == "10-K" else YEAR_TO_DATE

        def group(period: str, accession=accession) -> dict:
            return {
                f.name: figure(
                    a.value if (a := key.get((accession, f.name, period))) else None,
                    f.name,
                )
                for f in FIELDS
            }

        text = score.build_input(records[accession]).text
        replies[text] = json.dumps(
            {
                "period_end": str(release.period_end),
                "quarter": group(QUARTER),
                "year_to_date": group(year),
            }
        )
    return replies


def encode(text: str) -> list[int]:
    return byte_tokenizer().encode(text, add_special_tokens=False).ids


class FakeVLLM:
    """/tokenize and /v1/chat/completions as vLLM answers them; reply(model, text)
    gives each completion's content."""

    def __init__(self, reply, tokenize=encode) -> None:
        self.reply = reply
        self.tokenize = tokenize
        self.completions: list[dict] = []
        self.tokenized = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if str(request.url) == f"{ROOT}/tokenize":
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            assert body["add_generation_prompt"] and not body["add_special_tokens"]
            self.tokenized += 1
            system, user = (m["content"] for m in body["messages"])
            return httpx.Response(
                200, json={"tokens": self.tokenize(chat.prompt_text(system, user))}
            )
        assert str(request.url) == f"{ROOT}/v1/chat/completions"
        self.completions.append(body)
        content = self.reply(body["model"], body["messages"][1]["content"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            },
        )


def add_checkpoint(run_dir: Path, step: int, examples_seen: int, r: int = 4) -> None:
    path = run_dir / "checkpoints" / train.checkpoint_name(step)
    path.mkdir(parents=True)
    (path / "state.json").write_text(json.dumps({"examples_seen": examples_seen}))
    (path / "adapter_config.json").write_text(json.dumps({"r": r}))


@pytest.fixture
def run_dir(tmp_path) -> Path:
    write_labels(tmp_path / "labels")
    run_dir = prepare(tmp_path)
    add_checkpoint(run_dir, 2, 4)
    add_checkpoint(run_dir, 4, 8)
    return run_dir


@contextlib.contextmanager
def no_server(command, log_path, root, http):
    yield


def run(run_dir: Path, key_dir: Path, server: FakeVLLM, *args: str, starts=None) -> int:
    def start(command, log_path, root, http):
        if starts is not None:
            starts.append(command)
        return no_server(command, log_path, root, http)

    return score.main(
        [
            str(run_dir),
            "--links", str(key_dir / "links.csv"),
            "--answers", str(key_dir / "answers.csv"),
            "--releases", str(RELEASES),
            "--runs-dir", str(run_dir.parent.parent / "runs"),
            *args,
        ],
        transport=httpx.MockTransport(server.handler),
        edgar_client=make_client(run_dir.parent.parent / "raw"),
        start_server=start,
        snapshot=Path("/snapshots/tiny"),
        encode=encode,
    )  # fmt: skip


def graded(run: Path) -> Counter:
    with (run / "graded.csv").open(newline="") as f:
        return Counter(r["verdict"] for r in csv.DictReader(f))


def test_score_chooses_the_earliest_best_checkpoint(
    run_dir, key_dir, perfect, capsys
) -> None:
    aapl_text = next(t for t in perfect if "Apple" in t)

    def reply(model: str, text: str) -> str:
        if model == BASE_NAME and text == aapl_text:
            return "I can't find it."  # invalid twice: an all-null prediction
        return perfect[text]

    server, starts = FakeVLLM(reply), []
    assert run(run_dir, key_dir, server, starts=starts) == 0
    runs = run_dir.parent.parent / "runs"
    names = ["run--base", "run--step-00002", "run--step-00004"]
    assert sorted(p.name for p in runs.iterdir()) == names
    for name, model in zip(names, [BASE_NAME, *names[1:]], strict=True):
        info = json.loads((runs / name / "run.json").read_text())
        assert info["model"] == model and info["prompt_version"] == "extract_v2"
        assert info["request_parameters"] == {
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 1024,
        }
        assert info["releases"] == 5
    assert {b["model"] for b in server.completions} == {BASE_NAME, *names[1:]}
    assert all(b["max_tokens"] == 1024 for b in server.completions)
    # 5 validation releases and 4 training examples, each checked once.
    assert server.tokenized == 9
    [command] = starts
    assert command[1:3] == ["serve", "/snapshots/tiny"]
    assert command[-3:] == [
        "--lora-modules",
        f"run--step-00002={run_dir.resolve()}/checkpoints/step-00002",
        f"run--step-00004={run_dir.resolve()}/checkpoints/step-00004",
    ]
    assert graded(runs / "run--step-00002") == {"correct": 26}
    assert graded(runs / "run--base")["missing"] > 0

    results = json.loads((run_dir / "validation.json").read_text())
    assert results["chosen"] == "step-00002"
    assert results["rule"] == "the most figures correct; ties go to the earliest step"
    assert [m["label"] for m in results["models"]] == [
        "base",
        "step-00002",
        "step-00004",
    ]
    assert [m["examples_seen"] for m in results["models"]] == [0, 4, 8]
    assert [m["fallbacks"] for m in results["models"]] == [1, 0, 0]
    assert results["layout_prompts"] == 9
    report = (run_dir / "validation.md").read_text()
    assert "Chosen: step-00002, 26/26 (100.0%) (the most figures correct" in report
    assert "| step-00004 | 4 | 8 | 26/26 (100.0%) |" in report
    assert "token ids from the server's /tokenize identical to training's" in report
    env = (run_dir / "validation" / "step-00004.env").read_text()
    assert 'MODEL_NAME="run--step-00004"' in env
    assert (
        'MODEL_EXTRA_BODY=\'{"chat_template_kwargs": {"enable_thinking": false}, '
        '"max_tokens": 1024}\'' in env
    )

    # Run again: every model is kept and no server is started.
    server, starts = FakeVLLM(reply), []
    assert run(run_dir, key_dir, server, starts=starts) == 0
    assert starts == [] and server.completions == [] and server.tokenized == 0
    assert (
        "already scored, kept: base, step-00002, step-00004" in capsys.readouterr().out
    )
    assert (
        json.loads((run_dir / "validation.json").read_text())["chosen"] == "step-00002"
    )
    assert "Layout check: not run" in (run_dir / "validation.md").read_text()


def test_a_later_checkpoint_that_scores_higher_is_chosen(
    run_dir, key_dir, perfect
) -> None:
    aapl_text = next(t for t in perfect if "Apple" in t)

    def reply(model: str, text: str) -> str:
        if model != "run--step-00004" and text == aapl_text:
            wrong = json.loads(perfect[text])
            wrong["quarter"]["eps_diluted"] += 0.01  # 1.66 for 1.65
            return json.dumps(wrong)
        return perfect[text]

    assert run(run_dir, key_dir, FakeVLLM(reply), "--limit", "2") == 0
    results = json.loads((run_dir / "validation.json").read_text())
    assert results["releases"] == 2
    assert [(m["correct"], m["graded"]) for m in results["models"]] == [
        (11, 12),
        (11, 12),
        (12, 12),
    ]
    assert results["models"][0]["not_correct"] == {"other": 1}
    assert results["chosen"] == "step-00004"


def test_a_layout_difference_stops_before_scoring(
    run_dir, key_dir, perfect, capsys
) -> None:
    def tokenize(text: str) -> list[int]:
        ids = encode(text)
        return ids[:-6] + ids[-2:]  # no think block

    server = FakeVLLM(lambda model, text: perfect[text], tokenize)
    assert run(run_dir, key_dir, server) == 1
    err = capsys.readouterr().err
    assert "error: 9 prompts aren't tokenized as in training: validation " in err
    assert "tokens differ from training's" in err
    assert server.completions == []
    assert not (run_dir.parent.parent / "runs").exists()


def test_another_run_in_the_way_stops_scoring(
    run_dir, key_dir, perfect, capsys
) -> None:
    (run_dir.parent.parent / "runs" / "run--step-00004").mkdir(parents=True)
    server = FakeVLLM(lambda model, text: perfect[text])
    assert run(run_dir, key_dir, server) == 1
    assert "isn't a complete graded run of run--step-00004" in capsys.readouterr().err
    assert server.completions == []


def test_models_and_vllm_command(run_dir) -> None:
    found = score.models(run_dir, "Qwen/Qwen3-8B")
    assert found[0] == Model("base", "Qwen/Qwen3-8B", "run--base", 0, 0)
    assert [(m.label, m.served, m.step, m.examples_seen) for m in found[1:]] == [
        ("step-00002", "run--step-00002", 2, 4),
        ("step-00004", "run--step-00004", 4, 8),
    ]
    command = score.vllm_command(Path("/snap"), "Qwen/Qwen3-8B", found[1:], 8000)
    assert command[1:] == [
        "serve", "/snap",
        "--served-model-name", "Qwen/Qwen3-8B",
        "--max-model-len", "40960",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--enable-lora",
        "--max-lora-rank", "8",
        "--max-loras", "1",
        "--max-cpu-loras", "2",
        "--lora-modules",
        f"run--step-00002={found[1].adapter}",
        f"run--step-00004={found[2].adapter}",
    ]  # fmt: skip
    assert score.vllm_command(Path("/snap"), "Q", [], 8000)[-1] == "8000"
    hf = score.hf_command(Path("/snap"), "Q", found[1:], 8001)
    assert hf[:3] == [sys.executable, "-m", "secbench.hfserve"]


@pytest.mark.parametrize(("r", "rank"), [(4, 8), (16, 16), (20, 32), (64, 64)])
def test_lora_rank(tmp_path, r, rank) -> None:
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": r}))
    assert score.lora_rank([tmp_path]) == rank


def test_env_file_gives_the_scoring_settings(tmp_path) -> None:
    path = tmp_path / "step.env"
    path.write_text(
        score.env_text("http://127.0.0.1:8000/v1", "run--step-00025", 16, 1)
    )
    settings = extract.load_settings(env_file=path)
    assert (settings.base_url, settings.model, settings.concurrency) == (
        "http://127.0.0.1:8000/v1",
        "run--step-00025",
        16,
    )
    assert settings.parameters == score.expected_parameters()
    assert settings.gpu_hourly == 1.0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


SERVER = """
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
print("ready", flush=True)
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def test_run_server_waits_for_health_and_stops_the_server(tmp_path) -> None:
    port = free_port()
    log = tmp_path / "server.log"
    command = [sys.executable, "-c", SERVER, str(port)]
    root = f"http://127.0.0.1:{port}"
    with httpx.Client() as http:
        with score.run_server(command, log, root, http):
            assert http.get(f"{root}/health").status_code == 200
        with pytest.raises(httpx.TransportError):
            http.get(f"{root}/health", timeout=2.0)
    assert "ready" in log.read_text()


def test_run_server_reports_a_server_that_exits(tmp_path) -> None:
    command = [sys.executable, "-c", "print('no GPU found'); raise SystemExit(3)"]
    log = tmp_path / "server.log"
    with (
        httpx.Client() as http,
        pytest.raises(ScoreError) as error,
        score.run_server(command, log, "http://127.0.0.1:9", http),
    ):
        pass
    assert "the server exited (3) before it was ready" in str(error.value)
    assert "no GPU found" in str(error.value)


# The laptop stand-in for vLLM, on the tiny random Qwen3


@pytest.fixture(scope="module")
def adapter(tiny_snapshot, tmp_path_factory) -> Path:
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(1)
    model = AutoModelForCausalLM.from_pretrained(tiny_snapshot)
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            init_lora_weights=False,
        ),
    )
    path = tmp_path_factory.mktemp("adapter")
    model.save_pretrained(path)
    return path


def test_hfserve_answers_as_vllm(tiny_snapshot, adapter) -> None:
    from secbench import hfserve

    model = hfserve.Model(tiny_snapshot, "base", [("tuned", adapter)], 4096)
    messages = [
        {"role": "system", "content": "Reply with JSON."},
        {"role": "user", "content": "Revenue 101."},
    ]
    tokens = model.tokenize(
        {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
    )
    expected = train.encode(
        model.tokenizer, chat.prompt_text("Reply with JSON.", "Revenue 101.")
    )
    assert tokens == {"count": len(expected), "max_model_len": 4096, "tokens": expected}
    body = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": 5,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    base = model.complete({**body, "model": "base"})
    tuned = model.complete({**body, "model": "tuned"})
    assert extract.is_completion(base) and base["model"] == "base"
    assert base["usage"]["prompt_tokens"] == len(expected)
    assert 1 <= base["usage"]["completion_tokens"] <= 5
    assert base["choices"][0]["finish_reason"] in ("stop", "length")
    assert tuned["model"] == "tuned"
    # Greedy: the same request gives the same reply.
    assert model.complete({**body, "model": "base"})["choices"] == base["choices"]
    with pytest.raises(hfserve.RequestError, match="does not exist") as error:
        model.complete({**body, "model": "other"})
    assert error.value.status == 404
    with pytest.raises(hfserve.RequestError, match="temperature 0"):
        model.complete({**body, "model": "base", "temperature": 0.7})
    with pytest.raises(hfserve.RequestError, match="maximum context length is 4096"):
        model.complete({**body, "model": "base", "max_tokens": 4096})
