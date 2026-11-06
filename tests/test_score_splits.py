"""secbench.score --test-config: the base model's and one checkpoint's predictions on a
frozen split, against the fake vLLM, in a folder laid out as the repository
(test_splits' repo fixture: AAPL and MSFT validation, WMT company_test, XOM and CAT
time_test, built with the byte tokenizer)."""

import contextlib
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

import httpx
import pytest
from test_score import (  # noqa: F401 (key_dir and perfect are fixtures)
    BASE_NAME,
    FakeVLLM,
    add_checkpoint,
    encode,
    key_dir,
    make_client,
    perfect,
)
from test_splits import AAPL, MSFT, PROMPTS, repo  # noqa: F401 (repo is a fixture)
from test_splits import run as build_splits
from test_train import prepare, write_labels

from secbench import bundle, grade, score, splits

WEIGHTS = b"adapter weights"
CONFIG = Path("configs/test/run.toml")
RUN = Path("training/run")
VALIDATION_LINKS = "data/splits/validation/links.csv"
SPLIT_FILES = (*splits.FILES, splits.MANIFEST)


def write_test_config(path: Path, **changes) -> Path:
    values = {
        "run": '"run"',
        "base": f'"{BASE_NAME}"',
        "revision": '"' + "0" * 40 + '"',
        "checkpoint": '"step-00002"',
        "adapter_sha256": f'"{hashlib.sha256(WEIGHTS).hexdigest()}"',
        "max_model_len": "40960",
        "max_lora_rank": "8",  # vLLM's for the test run's r 4
        "max_loras": "1",
        "max_cpu_loras": "1",
        "prompt": '"extract_v2"',
        "temperature": "0.0",
        "enable_thinking": "false",
        "max_tokens": "1024",
        "concurrency": "4",
    } | changes
    tables = {
        "model": ("run", "base", "revision", "checkpoint", "adapter_sha256"),
        "server": ("max_model_len", "max_lora_rank", "max_loras", "max_cpu_loras"),
        "request": (
            "prompt",
            "temperature",
            "enable_thinking",
            "max_tokens",
            "concurrency",
        ),
    }
    lines = []
    for table, keys in tables.items():
        lines.append(f"[{table}]")
        lines += [f"{k} = {values[k]}" for k in keys if values[k] is not None]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def lab(repo) -> Path:  # noqa: F811
    """The repository folder with frozen validation and company_test splits, prompts/,
    a prepared training run with checkpoints at steps 2 and 4 (r 4), and the test
    config for step 2."""
    assert build_splits("--freeze", "company_test", "validation") == 0
    assert build_splits() == 0
    shutil.copytree(PROMPTS, repo / "prompts")
    write_labels(repo / "labels")
    run_dir = prepare(repo)
    for step, seen in ((2, 4), (4, 8)):
        add_checkpoint(run_dir, step, seen)
        weights = run_dir / "checkpoints" / f"step-{step:05d}"
        (weights / "adapter_model.safetensors").write_bytes(WEIGHTS + bytes([step]))
    (run_dir / "checkpoints/step-00002/adapter_model.safetensors").write_bytes(WEIGHTS)
    write_test_config(repo / CONFIG)
    return repo


@contextlib.contextmanager
def no_server(command, log_path, root, http):
    yield


def score_split(server: FakeVLLM, *args: str, starts=None, split="validation") -> int:
    def start(command, log_path, root, http):
        if starts is not None:
            starts.append(command)
        return no_server(command, log_path, root, http)

    return score.main(
        [
            str(RUN),
            "--test-config", str(CONFIG),
            "--split", split,
            "--gpu-hourly", "2.5",
            *args,
        ],
        transport=httpx.MockTransport(server.handler),
        start_server=start,
        snapshot=Path("/snapshots/tiny"),
        encode=encode,
    )  # fmt: skip


def verdicts(run: Path) -> Counter:
    with (run / "graded.csv").open(newline="") as f:
        return Counter(r["verdict"] for r in csv.DictReader(f))


def test_a_frozen_split_is_scored_with_the_test_config(lab, perfect, capsys) -> None:  # noqa: F811
    server, starts = FakeVLLM(lambda model, text: perfect[text]), []
    assert score_split(server, starts=starts) == 0
    out = capsys.readouterr().out
    adapter = (lab / RUN / "checkpoints" / "step-00002").resolve()
    [command] = starts
    assert command[1:] == [
        "serve", "/snapshots/tiny",
        "--served-model-name", BASE_NAME,
        "--max-model-len", "40960",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--enable-lora",
        "--max-lora-rank", "8",
        "--max-loras", "1",
        "--max-cpu-loras", "1",
        "--lora-modules", f"run--step-00002={adapter}",
    ]  # fmt: skip
    assert server.tokenized == 2  # the split's prompts; no training examples
    assert Counter(b["model"] for b in server.completions) == {
        BASE_NAME: 2,
        "run--step-00002": 2,
    }
    for body in server.completions:
        assert body["temperature"] == 0 and body["max_tokens"] == 1024
        assert body["chat_template_kwargs"] == {"enable_thinking": False}

    runs = lab / "runs"
    names = ["run--validation--base", "run--validation--step-00002"]
    assert sorted(p.name for p in runs.iterdir()) == names
    links_sha256 = hashlib.sha256((lab / VALIDATION_LINKS).read_bytes()).hexdigest()
    for name, served in zip(names, [BASE_NAME, "run--step-00002"], strict=True):
        folder = runs / name
        assert sorted(p.name for p in folder.iterdir()) == [
            "predictions.jsonl",
            "raw",
            "run.json",
        ]
        info = json.loads((folder / "run.json").read_text())
        assert info["model"] == served and info["status"] == "complete"
        assert info["skip_grading"] is True
        assert (info["links"], info["links_sha256"]) == (VALIDATION_LINKS, links_sha256)
        assert info["answers"] == "data/splits/validation/answers.csv"
        assert info["releases_dir"] == "data/releases"
        assert (info["max_parallel"], info["gpu_hourly_usd"]) == (4, 2.5)
        assert info["request_parameters"] == score.expected_parameters()
    env = (lab / RUN / "test" / "validation--step-00002.env").read_text()
    assert 'MODEL_NAME="run--step-00002"' in env and 'MODEL_GPU_HOURLY="2.5"' in env
    assert (lab / RUN / "test" / "validation--base.env").is_file()

    record = json.loads((lab / RUN / "test" / "validation.json").read_text())
    assert record["split"] == "validation" and record["server_command"] == command
    config_sha = hashlib.sha256((lab / CONFIG).read_bytes()).hexdigest()
    assert record["test_config"] == {"path": str(CONFIG), "sha256": config_sha}
    assert record["checkpoint"] == {
        "label": "step-00002",
        "served": "run--step-00002",
        "step": 2,
        "examples_seen": 4,
        "adapter_sha256": hashlib.sha256(WEIGHTS).hexdigest(),
    }
    manifest = lab / "data/splits/validation/manifest.json"
    assert record["split_files"]["status"] == "frozen"
    assert record["split_files"]["manifest_sha256"] == (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert record["layout_check"] == {
        "prompts": 2,
        "longest_prompt_tokens": 23_399,  # MSFT's, as tokens.csv counts it
        "identical_to_training_layout": True,
    }
    assert record["settings"]["concurrency"] == 4
    assert record["settings"]["gpu_hourly_usd"] == 2.5
    assert [(r["label"], r["kept"]) for r in record["runs"]] == [
        ("base", False),
        ("step-00002", False),
    ]
    assert record["failed"] == [] and "commit" in record and "clean" in record
    assert "chosen" not in record
    assert (
        "grade each run on the laptop: uv run python -m secbench.grade runs/<run> "
        f"--links {VALIDATION_LINKS} --answers data/splits/validation/answers.csv "
        "--releases data/releases" in out
    )

    # Graded on the laptop, with the split's files.
    for name in names:
        argv = [
            str(runs / name),
            "--links", VALIDATION_LINKS,
            "--answers", "data/splits/validation/answers.csv",
            "--releases", "data/releases",
        ]  # fmt: skip
        assert grade.main(argv, client=make_client(lab / "raw")) == 0
        assert verdicts(runs / name) == {"correct": 11}

    # Again: both runs are kept, and no server is started.
    server, starts = FakeVLLM(lambda model, text: perfect[text]), []
    assert score_split(server, starts=starts) == 0
    assert starts == [] and server.completions == [] and server.tokenized == 0
    record = json.loads((lab / RUN / "test" / "validation.json").read_text())
    assert record["layout_check"] is None
    assert [(r["label"], r["kept"]) for r in record["runs"]] == [
        ("base", True),
        ("step-00002", True),
    ]


def test_a_run_with_an_api_error_stops_scoring(lab, perfect, capsys) -> None:  # noqa: F811
    aapl = next(t for t in perfect if "Apple" in t)

    class Failing(FakeVLLM):
        def handler(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if (
                request.url.path == "/v1/chat/completions"
                and body["model"] == BASE_NAME
                and body["messages"][1]["content"] == aapl
            ):
                return httpx.Response(400, json={"error": {"message": "bad request"}})
            return super().handler(request)

    server = Failing(lambda model, text: perfect[text])
    assert score_split(server) == 1
    out = capsys.readouterr().out
    runs = lab / "runs"
    base = runs / "run--validation--base"
    assert [p.name for p in runs.iterdir()] == [base.name]  # nothing after it
    assert (
        f"error: {Path('runs') / base.name} had API errors or stopped, and nothing "
        f"was run after it; move it to {Path('runs') / base.name}--attempt-1 and "
        "score again" in out
    )
    record = json.loads((lab / RUN / "test" / "validation.json").read_text())
    assert record["failed"] == ["base"]
    assert record["runs"][0]["complete_without_errors"] is False

    # Not kept, and never overwritten: scoring stops until it is moved.
    server = FakeVLLM(lambda model, text: perfect[text])
    assert score_split(server) == 1
    err = capsys.readouterr().err
    assert "isn't a complete run of test/tiny-qwen3 with no API error" in err
    assert f"move it to {Path('runs') / base.name}--attempt-1 and score again" in err
    assert server.completions == [] and server.tokenized == 0

    base.rename(base.with_name(base.name + "--attempt-1"))
    assert score_split(server) == 0
    assert sorted(p.name for p in runs.iterdir()) == [
        "run--validation--base",
        "run--validation--base--attempt-1",
        "run--validation--step-00002",
    ]


def test_a_provisional_split_is_refused(lab, perfect, capsys) -> None:  # noqa: F811
    server, starts = FakeVLLM(lambda model, text: perfect[text]), []
    assert score_split(server, starts=starts, split="time_test") == 1
    assert "time_test is provisional, not frozen: only a frozen split is scored" in (
        capsys.readouterr().err
    )
    assert starts == [] and not (lab / "runs").exists()


def test_a_prompt_over_the_limit_is_refused(lab, perfect, capsys) -> None:  # noqa: F811
    # WMT's prompt is 48,111 tokens by the byte tokenizer, over 39,936 (the split
    # files were built with a limit of 60,000).
    server, starts = FakeVLLM(lambda model, text: perfect[text]), []
    assert score_split(server, starts=starts, split="company_test") == 1
    assert (
        "1 prompts over 39,936 tokens: company_test 0000104169-25-000069 (48,111)"
        in capsys.readouterr().err
    )
    assert starts == []


def change_weights(lab: Path) -> None:
    (lab / RUN / "checkpoints/step-00002/adapter_model.safetensors").write_bytes(b"x")


def change_setting(lab: Path) -> None:
    write_test_config(lab / CONFIG, temperature="0.7")


def change_rank(lab: Path) -> None:
    write_test_config(lab / CONFIG, max_lora_rank="16")


def change_split_file(lab: Path) -> None:
    path = lab / "data/splits/validation/answers.csv"
    path.write_text(path.read_text().replace(",USD,", ",usd,", 1))


def change_input(lab: Path) -> None:
    path = lab / "data/releases/0000320193" / f"{AAPL}.md"
    path.write_text(path.read_text() + "\nA footnote.\n")


def drop_release(lab: Path) -> None:
    shutil.rmtree(lab / "data/releases/0000789019")


def unknown_key(lab: Path) -> None:
    path = lab / CONFIG
    path.write_text(path.read_text() + "seed = 1\n")


def missing_key(lab: Path) -> None:
    write_test_config(lab / CONFIG, max_tokens=None)


def wrong_type(lab: Path) -> None:
    write_test_config(lab / CONFIG, concurrency='"16"')


def other_checkpoint(lab: Path) -> None:
    write_test_config(lab / CONFIG, checkpoint='"step-00003"')


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (change_weights, "adapter_model.safetensors has sha256 "),
        (change_setting, "configs/test/run.toml: temperature is 0.7, not 0.0"),
        (change_rank, "configs/test/run.toml: max_lora_rank is 16, not 8"),
        (change_split_file, "data/splits/validation/answers.csv isn't the file"),
        (change_input, f"{AAPL}: its input isn't the one data/splits/validation/"),
        (drop_release, f"no saved release {MSFT} in data/releases"),
        (unknown_key, "configs/test/run.toml: unknown key request.seed"),
        (missing_key, "configs/test/run.toml: request.max_tokens not set"),
        (wrong_type, "configs/test/run.toml: request.concurrency must be a int"),
        (other_checkpoint, "step-00003: no complete checkpoint (with state.json)"),
    ],
)
def test_anything_unlike_validation_or_the_split_stops_before_the_server(
    lab,
    perfect,  # noqa: F811
    capsys,
    change,
    message,
) -> None:
    change(lab)
    server, starts = FakeVLLM(lambda model, text: perfect[text]), []
    assert score_split(server, starts=starts) == 1
    assert message in capsys.readouterr().err
    assert starts == [] and not (lab / "runs").exists()


TESTING = ["--test-config", "c.toml", "--split", "validation"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--split", "validation"], "--test-config and --split go together"),
        (TESTING, "--test-config needs --gpu-hourly"),
        (
            [*TESTING, "--gpu-hourly", "1", "--links", "x.csv", "--concurrency", "2"],
            "--links, --concurrency: a test split's come from data/splits/<split>/",
        ),
        (["--gpu-hourly", "-1"], "--gpu-hourly must be 0 or more"),
    ],
)
def test_test_options(args, message, capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        score.main(["training/run", *args])
    assert stop.value.code == 2
    assert message in capsys.readouterr().err


def test_split_names_are_the_splits_modules() -> None:
    assert splits.SPLITS == ("validation", "company_test", "time_test")


# The bundle for a test split, unpacked as scripts/vast_setup.sh unpacks it


def test_score_a_split_from_an_unpacked_bundle(
    lab,
    perfect,  # noqa: F811
    tmp_path_factory,
    monkeypatch,
) -> None:
    state = lab / RUN / "checkpoints/step-00002/training_state.pt"
    state.write_bytes(b"optimizer")  # not packed
    files, entry = bundle.files_for_split(CONFIG, "validation")
    assert [str(p) for p in files] == [
        "configs/test/run.toml",
        "training/run/config.json",
        "training/run/checkpoints/step-00002/adapter_config.json",
        "training/run/checkpoints/step-00002/adapter_model.safetensors",
        "training/run/checkpoints/step-00002/state.json",
        *(f"data/splits/validation/{name}" for name in SPLIT_FILES),
        f"data/releases/0000320193/{AAPL}.json",
        f"data/releases/0000320193/{AAPL}.md",
        f"data/releases/0000789019/{MSFT}.json",
        f"data/releases/0000789019/{MSFT}.md",
    ]
    assert entry == {
        "config": str(CONFIG),
        "config_sha256": hashlib.sha256((lab / CONFIG).read_bytes()).hexdigest(),
        "config_tracked": False,
        "run": "run",
        "checkpoint": "step-00002",
        "adapter_sha256": hashlib.sha256(WEIGHTS).hexdigest(),
        "split": "validation",
        "split_built_at": "c0ffee" * 6,
        "split_files": {
            f"data/splits/validation/{name}": hashlib.sha256(
                (lab / "data/splits/validation" / name).read_bytes()
            ).hexdigest()
            for name in SPLIT_FILES
        },
        "releases": 2,
        "release_files": 4,
    }
    out = tmp_path_factory.mktemp("tar") / "bundle.tar"
    branch = bundle.git("rev-parse", "--abbrev-ref", "HEAD")
    info = bundle.make_bundle(out, files, "abc123", branch, {"test": entry})
    assert info["test"] == entry and info["files"] == len(files)

    instance = tmp_path_factory.mktemp("instance")
    staging = instance / "bundle"
    with tarfile.open(out) as tar:
        tar.extractall(staging, filter="data")
    clone = instance / "sec-earnings-bench"
    git = ["git", "clone", "-q", "-b", branch, str(staging / "repo.bundle"), str(clone)]
    subprocess.run(git, check=True)
    for name in ("configs", "data", "training"):
        shutil.copytree(staging / name, clone / name, dirs_exist_ok=True)
    assert not (clone / "data/raw").exists() and not (clone / "data/stage2").exists()
    assert not (clone / RUN / "examples.csv").exists()
    assert not list(clone.rglob(state.name))
    bundle.check_test_unpacked(clone, "validation", splits.DEFAULT_OUT)

    # Anything score reads beyond the bundle (the laptop's files) would fail here.
    monkeypatch.chdir(clone)
    server = FakeVLLM(lambda model, text: perfect[text])
    assert score_split(server) == 0
    assert sorted(p.name for p in (clone / "runs").iterdir()) == [
        "run--validation--base",
        "run--validation--step-00002",
    ]
    assert server.tokenized == 2 and len(server.completions) == 4
    record = json.loads((clone / RUN / "test" / "validation.json").read_text())
    assert record["checkpoint"]["adapter_sha256"] == entry["adapter_sha256"]
    assert record["test_config"]["sha256"] == entry["config_sha256"]
    assert {
        f"data/splits/validation/{name}": digest
        for name, digest in record["split_files"]["files"].items()
    } == {k: v for k, v in entry["split_files"].items() if "manifest" not in k}


def test_a_bundle_refuses_a_provisional_split_or_another_adapter(lab) -> None:
    with pytest.raises(score.ScoreError, match="time_test is provisional"):
        bundle.files_for_split(CONFIG, "time_test")
    change_weights(lab)
    with pytest.raises(score.ScoreError, match="has sha256 .*, not the test config's"):
        bundle.files_for_split(CONFIG, "validation")


def test_a_split_without_its_release_files_fails_the_unpacked_check(
    lab, tmp_path
) -> None:
    files, _ = bundle.files_for_split(CONFIG, "validation")
    copy = tmp_path / "copy"
    bundle.symlinked_copy(copy, [f for f in files if MSFT not in f.name])
    with pytest.raises(
        bundle.BundleError, match=f"unpacked bundle: no saved release {MSFT}"
    ):
        bundle.check_test_unpacked(copy, "validation", splits.DEFAULT_OUT)
