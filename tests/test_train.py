import csv
import dataclasses
import itertools
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
from conftest import byte_tokenizer, qwen3_chat_template, save_tiny_snapshot

from secbench import chat, extract, labels, train
from secbench.labels import FIRST, FOURTH, MIDDLE
from secbench.train import Example, TrainError

PROMPTS = Path(__file__).parent.parent / "prompts"
PROMPT = extract.load_prompt(labels.PROMPT, PROMPTS).text


def tables(tmp_path: Path, **changes) -> dict:
    """A test config's tables: float32 on the CPU, 4 examples per epoch, 2 epochs,
    2 per step (4 steps), checkpoints at steps 2 and 4."""
    result = {
        "model": {"name": "test/tiny-qwen3", "revision": "0" * 40},
        "data": {
            "examples": str(tmp_path / "labels" / "train.jsonl"),
            "index": str(tmp_path / "labels" / "examples.csv"),
            "subset": 4,
            "max_tokens": 2600,
            "seed": 7,
        },
        "lora": {
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        },
        "training": {
            "epochs": 2,
            "gradient_accumulation": 2,
            "learning_rate": 1e-2,
            "warmup_steps": 1,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "save_every": 2,
            "gradient_checkpointing": True,
            "dtype": "float32",
            "device": "cpu",
        },
    }
    for key, value in changes.items():
        table, name = key.split("__")
        result[table][name] = value
    return result


# (accession, ticker, kind, user text); the prompt is 2,261 bytes, one token each.
RELEASES = [
    ("0000000001-25-000001", "AAA", FIRST, "Revenue 101. Net income 11."),
    ("0000000002-25-000002", "BBB", MIDDLE, "Revenue 202. Net income 22."),
    ("0000000003-25-000003", "CCC", MIDDLE, "Revenue 303. Net income 33."),
    ("0000000004-25-000004", "DDD", FOURTH, "Revenue 404. Net income 44."),
    ("0000000005-25-000005", "EEE", MIDDLE, "Revenue 505. " + "x" * 400),  # too long
    ("0000000006-25-000006", "FFF", FIRST, "Revenue 606. Net income 66."),
]
LONG = RELEASES[4][0]


def reply(user: str) -> str:
    revenue = user.split()[1].rstrip(".")
    return json.dumps({"period_end": "2025-06-30", "revenue": int(revenue)})


def write_labels(folder: Path, releases=RELEASES, prompt: str = PROMPT) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "release_accession": accession,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": reply(user)},
                ],
            }
        )
        for accession, _, _, user in releases
    ]
    (folder / "train.jsonl").write_text("\n".join(lines) + "\n")
    rows = [
        {
            "release_accession": a,
            "ticker": t,
            "kind": k,
            "linked_form": "10-K" if k == FOURTH else "10-Q",
            "filing_date": "2025-07-30",
        }
        for a, t, k, _ in releases
    ]
    (folder / "examples.csv").write_text(
        train.rows_csv(rows, labels.EXAMPLE_COLUMNS), encoding="utf-8"
    )
    (folder / "manifest.json").write_text('{"commit": "labels-commit"}\n')


def write_config(path: Path, tables_: dict) -> Path:
    def value(v):
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, list):
            return "[" + ", ".join(json.dumps(x) for x in v) + "]"
        return repr(v)

    lines = []
    for name, table in tables_.items():
        lines.append(f"[{name}]")
        lines += [f"{key} = {value(v)}" for key, v in table.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def labelled(tmp_path) -> Path:
    write_labels(tmp_path / "labels")
    return tmp_path


def prepare(
    tmp_path: Path, name: str = "run", snapshot: Path | None = None, **changes
) -> Path:
    """A prepared run, its tokens counted with snapshot's tokenizer.json (as prepare
    downloads the model's), or with the byte tokenizer."""
    from tokenizers import Tokenizer

    config = train.parse_config(tables(tmp_path, **changes))
    run_dir = tmp_path / "training" / name
    if snapshot is None:
        tokenizer_path = tmp_path / "tokenizer.json"
        if not tokenizer_path.exists():
            byte_tokenizer().save(str(tokenizer_path))
    else:
        tokenizer_path = snapshot / "tokenizer.json"
    train.prepare(
        config,
        tmp_path / "test.toml",
        run_dir,
        Tokenizer.from_file(str(tokenizer_path)),
        tokenizer_path,
        "test-commit",
        {},
        PROMPTS,
    )
    return run_dir


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# Settings


def test_config_round_trip(tmp_path) -> None:
    config = train.parse_config(tables(tmp_path))
    assert config.subset == 4 and config.target_modules[0] == "q_proj"
    assert train.parse_config(train.config_tables(config)) == config
    assert train.parse_config(tables(tmp_path, data__subset="all")).subset is None
    # An integer where a float goes is that float.
    assert train.parse_config(tables(tmp_path, lora__dropout=0)).dropout == 0.0


def test_the_toml_files_parse() -> None:
    configs = Path(__file__).parent.parent / "configs" / "train"
    first = train.load_config(configs / "qwen3-8b-2k.toml")
    assert (first.model, first.revision) == (labels.QWEN3, labels.QWEN3_REVISION)
    assert first.tokenizer_sha256 == labels.TOKENIZER_SHA256
    assert (first.subset, first.max_tokens, first.epochs) == (2000, 32768, 1)
    assert (first.r, first.alpha, first.dropout, first.learning_rate) == (
        16,
        32,
        0.0,
        2e-4,
    )
    assert train.make_schedule(2000, first).checkpoints == (25, 50, 75, 100, 125)
    smoke = train.load_config(configs / "smoke-qwen3-0.6b.toml")
    assert train.make_schedule(5, smoke).checkpoints == (2, 4, 5)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"data__subset": "some"}, "data.subset must be a int"),
        ({"data__subset": 0}, "data.subset must be 1 or more"),
        ({"training__epochs": True}, "training.epochs must be a int"),
        ({"training__gradient_checkpointing": 1}, "must be a bool"),
        ({"lora__dropout": 1.0}, "lora.dropout must be less than 1"),
        ({"lora__target_modules": []}, "lora.target_modules must be a list"),
        ({"training__dtype": "float16"}, "training.dtype must be bfloat16 or float32"),
        ({"training__device": "gpu"}, "training.device must be auto, cuda"),
        ({"training__save_every": 0}, "training.save_every must be 1 or more"),
    ],
)
def test_config_problems(tmp_path, changes, message) -> None:
    with pytest.raises(TrainError, match=message):
        train.parse_config(tables(tmp_path, **changes))


def test_config_keys(tmp_path) -> None:
    t = tables(tmp_path)
    del t["data"]["seed"]
    t["lora"]["rank"] = 4
    t["extra"] = {}
    with pytest.raises(TrainError) as error:
        train.parse_config(t)
    assert str(error.value) == (
        "config: unknown table [extra]; data.seed not set; unknown key lora.rank"
    )


# Drawing the examples


def test_allocate_by_largest_remainder() -> None:
    within = {FIRST: 900, MIDDLE: 1534, FOURTH: 667}  # the stage 2 data
    assert train.allocate(within, 2000) == {FIRST: 581, MIDDLE: 989, FOURTH: 430}
    assert train.allocate(within, 1000) == {FIRST: 290, MIDDLE: 495, FOURTH: 215}
    assert train.allocate(within, 3101) == within
    # Equal remainders go to the kind listed first.
    assert train.allocate({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    assert train.allocate({"a": 3, "b": 0}, 2) == {"a": 2, "b": 0}
    with pytest.raises(TrainError, match="4 examples asked for; 3 available"):
        train.allocate({"a": 3}, 4)


def examples(n: int) -> list[Example]:
    kinds = [FIRST, MIDDLE, MIDDLE, FOURTH]
    return [Example(f"acc-{i}", "T", kinds[i % 4], 100, 10) for i in range(n)]


def test_draw_is_stratified_and_seeded() -> None:
    pool = examples(40)  # 10 first, 20 middle, 10 fourth
    orders = train.draw(pool, 8, seed=1, epochs=2)
    assert Counter(e.kind for e in orders[0]) == {FIRST: 2, MIDDLE: 4, FOURTH: 2}
    assert set(orders[0]) == set(orders[1]) and orders[0] != orders[1]
    assert train.draw(pool, 8, seed=1, epochs=2) == orders
    assert train.draw(pool, 8, seed=2, epochs=2) != orders
    every = train.draw(pool, None, seed=1, epochs=1)[0]
    assert sorted(e.accession for e in every) == sorted(e.accession for e in pool)


def test_schedule_and_windows(tmp_path) -> None:
    config = train.parse_config(
        tables(tmp_path, training__gradient_accumulation=4, training__epochs=1)
    )
    schedule = train.make_schedule(10, config)
    assert (schedule.positions, schedule.steps, schedule.checkpoints) == (10, 3, (2, 3))
    assert [schedule.window(s) for s in range(3)] == [
        range(4),
        range(4, 8),
        range(8, 10),
    ]


def test_lr_factor() -> None:
    factors = [train.lr_factor(s, steps=125, warmup=10) for s in range(125)]
    assert factors[:10] == pytest.approx([k / 10 for k in range(1, 11)])
    assert factors[9] == 1.0
    assert all(a > b for a, b in itertools.pairwise(factors[9:]))
    assert 0 < factors[-1] < 1e-3
    assert train.lr_factor(0, steps=1, warmup=0) == pytest.approx(0.5)


# Prepare


def test_prepare(labelled) -> None:
    run_dir = prepare(labelled)
    info = json.loads((run_dir / "config.json").read_text())
    assert info["commit"] == "test-commit" and info["run_name"] == "run"
    assert info["settings"] == tables(labelled)
    assert info["data"]["labels_manifest"] == {"commit": "labels-commit"}
    assert info["data"]["examples_sha256"] == train.file_sha256(
        labelled / "labels" / "train.jsonl"
    )
    assert info["data"]["prompt"]["version"] == "extract_v2"
    assert info["selection"] == {
        "examples": 6,
        "over_max_tokens": 1,
        "within_max_tokens": {FIRST: 2, MIDDLE: 2, FOURTH: 1},
        "selected": 4,
        "selected_by_kind": {FIRST: 2, MIDDLE: 1, FOURTH: 1},
    }
    assert info["schedule"] == {
        "epochs": 2,
        "examples_per_epoch": 4,
        "positions": 8,
        "examples_per_step": 2,
        "steps": 4,
        "warmup_steps": 1,
        "checkpoints": [2, 4],
    }
    rows = read_rows(run_dir / "examples.csv")
    assert [r["position"] for r in rows] == [str(i) for i in range(8)]
    assert [r["epoch"] for r in rows] == ["1"] * 4 + ["2"] * 4
    assert {r["release_accession"] for r in rows[:4]} == {
        r["release_accession"] for r in rows[4:]
    }
    first = next(r for r in rows if r["release_accession"] == RELEASES[0][0])
    # One token per byte; the markers and think tags are one token each.
    system, user = PROMPT, RELEASES[0][3]
    assert int(first["prompt_tokens"]) == len(
        byte_tokenizer().encode(chat.prompt_text(system, user)).ids
    )
    assert int(first["target_tokens"]) == len(reply(user).encode()) + 1
    dropped = read_rows(run_dir / "dropped.csv")
    assert [d["release_accession"] for d in dropped] == [LONG]
    assert int(dropped[0]["tokens"]) > 2600
    assert LONG not in {r["release_accession"] for r in rows}
    printed = train.format_prepared(run_dir, info)
    assert "4 of 5 examples within 2,600 tokens (1 of 6 longer" in printed
    assert (
        "first quarter 2 of 2, second or third quarter 1 of 2, fourth quarter 1 of 1"
        in printed
    )
    assert "2 epochs, 4 steps of 2 examples, 1 warmup; checkpoints at steps 2, 4" in (
        printed
    )


def test_prepare_is_seeded(labelled) -> None:
    first = read_rows(prepare(labelled, "a") / "examples.csv")
    assert read_rows(prepare(labelled, "b") / "examples.csv") == first
    other = read_rows(prepare(labelled, "c", data__seed=8) / "examples.csv")
    assert other != first


def test_prepare_refuses(labelled) -> None:
    prepare(labelled)
    with pytest.raises(TrainError, match="already exists"):
        prepare(labelled)
    with pytest.raises(
        TrainError, match="subset 6 is more than the 5 examples within 2600 tokens"
    ):
        prepare(labelled, "big", data__subset=6)
    write_labels(labelled / "other", RELEASES, prompt="Another prompt.")
    with pytest.raises(TrainError, match="6 examples' system message isn't"):
        prepare(
            labelled,
            "other",
            data__examples=str(labelled / "other" / "train.jsonl"),
            data__index=str(labelled / "other" / "examples.csv"),
        )
    write_labels(labelled / "short", RELEASES[:3])
    with pytest.raises(TrainError, match="list different releases"):
        prepare(
            labelled,
            "mismatch",
            data__index=str(labelled / "short" / "examples.csv"),
        )


def test_main_prepare(labelled, capsys) -> None:
    config_path = write_config(labelled / "tiny.toml", tables(labelled))
    tokenizer_path = labelled / "tokenizer.json"
    byte_tokenizer().save(str(tokenizer_path))
    args = [
        "prepare",
        "--config",
        str(config_path),
        "--training-dir",
        str(labelled / "training"),
        "--prompts-dir",
        str(PROMPTS),
    ]
    fetched = []

    def fetch(config, filename, hf_cache):
        fetched.append((config.model, config.revision, filename, hf_cache))
        return tokenizer_path

    assert train.main(args + ["--subset", "all"], commit="c1", fetch=fetch) == 0
    info = json.loads((labelled / "training" / "tiny" / "config.json").read_text())
    assert info["overrides"] == {"subset": "all"}
    assert info["settings"]["data"]["subset"] == "all"
    assert info["selection"]["selected"] == 5
    assert fetched == [
        ("test/tiny-qwen3", "0" * 40, "tokenizer.json", Path("data/raw/hf-hub"))
    ]
    out = capsys.readouterr().out
    assert "5 of 5 examples within 2,600 tokens" in out
    assert train.main(args + ["--run-name", "two"], commit="c1", fetch=fetch) == 0
    info = json.loads((labelled / "training" / "two" / "config.json").read_text())
    assert info["overrides"] == {} and info["selection"]["selected"] == 4
    assert train.main(args + ["--run-name", "../up"], commit="c1", fetch=fetch) == 1
    assert "bad run name" in capsys.readouterr().err


def test_checkpoints_are_complete_folders_only(tmp_path) -> None:
    folder = tmp_path / "checkpoints"
    for name in ("step-00002", "step-00004.part", "step-00006", "notes"):
        (folder / name).mkdir(parents=True)
    (folder / "step-00002" / "state.json").write_text("{}")
    (folder / "step-00004.part" / "state.json").write_text("{}")
    assert train.checkpoints(tmp_path) == [train.Checkpoint(2, folder / "step-00002")]


# Training on a tiny random Qwen3 (the train dependency group)


def test_reply_loss_is_the_masked_loss_over_the_target(tiny_snapshot) -> None:
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_snapshot, dtype=torch.float32)
    prompt_ids, target_ids = list(range(10, 60)), [70, 71, 72, 5]
    loss = train.reply_loss_sum(model, prompt_ids, target_ids, "cpu")
    labels_ = torch.tensor([[-100] * len(prompt_ids) + target_ids])
    full = model(input_ids=torch.tensor([prompt_ids + target_ids]), labels=labels_).loss
    assert loss.item() == pytest.approx(full.item() * len(target_ids), rel=1e-5)


def adapter_weights(path: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(path / "adapter_model.safetensors")


def test_train_saves_checkpoints_and_resumes_exactly(labelled, tiny_snapshot) -> None:
    torch = pytest.importorskip("torch")
    straight = prepare(labelled, "straight", tiny_snapshot)
    interrupted = prepare(labelled, "interrupted", tiny_snapshot)
    lines: list[str] = []
    train.train(straight, tiny_snapshot, "test-commit", out=lines.append)
    assert lines[0] == "checking 4 examples against the chat template"
    assert "starting at step 1, 4 steps" in lines[1]
    log = train.read_log(straight)
    assert [row["step"] for row in log] == [1, 2, 3, 4]
    assert [row["examples_seen"] for row in log] == [2, 4, 6, 8]
    assert [row["epoch"] for row in log] == [0.5, 1.0, 1.5, 2.0]
    assert log[0]["learning_rate"] == pytest.approx(1e-2)
    assert log[-1]["learning_rate"] < log[1]["learning_rate"]
    assert all(row["loss"] > 0 and row["target_tokens"] > 0 for row in log)
    saved = train.checkpoints(straight)
    assert [c.step for c in saved] == [2, 4]
    files = {p.name for p in saved[0].path.iterdir()}
    assert {"adapter_config.json", "adapter_model.safetensors", "state.json"} <= files
    assert "training_state.pt" in files
    assert not any("tokenizer" in name for name in files)
    adapter = json.loads((saved[0].path / "adapter_config.json").read_text())
    assert (adapter["r"], adapter["lora_alpha"]) == (4, 8)
    assert set(adapter["target_modules"]) == set(
        tables(labelled)["lora"]["target_modules"]
    )
    state = json.loads((saved[1].path / "state.json").read_text())
    assert (state["step"], state["examples_seen"], state["epoch"]) == (4, 8, 2.0)
    sessions = (straight / "sessions.jsonl").read_text().splitlines()
    assert json.loads(sessions[0])["start_step"] == 0

    lines = []
    train.train(
        interrupted, tiny_snapshot, "test-commit", stop_after=2, out=lines.append
    )
    assert lines[-1] == "stopped after step 2 (--stop-after)"
    assert [c.step for c in train.checkpoints(interrupted)] == [2]
    # A step after the checkpoint, lost when the session stopped: the resume drops it.
    with (interrupted / "train_log.jsonl").open("a") as f:
        f.write(json.dumps({"step": 3, "loss": 99.0}) + "\n")
    lines = []
    train.train(interrupted, tiny_snapshot, "test-commit", out=lines.append)
    assert "resuming after step-00002, 4 steps" in lines[1]
    resumed_log = train.read_log(interrupted)
    assert [row["step"] for row in resumed_log] == [1, 2, 3, 4]
    assert [r["loss"] for r in resumed_log] == pytest.approx([r["loss"] for r in log])
    a, b = (
        adapter_weights(c.path) for c in (saved[1], train.checkpoints(interrupted)[1])
    )
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    starts = [
        json.loads(s)["start_step"]
        for s in (interrupted / "sessions.jsonl").read_text().splitlines()
    ]
    assert starts == [0, 2]

    lines = []
    train.train(straight, tiny_snapshot, "test-commit", out=lines.append)
    assert lines == [f"{straight}: finished (step-00004); nothing to do"]


def test_train_refuses(labelled, tiny_snapshot, tmp_path_factory) -> None:
    run_dir = prepare(labelled, snapshot=tiny_snapshot)
    with pytest.raises(
        TrainError, match="prepared at commit test-commit; HEAD is other"
    ):
        train.train(run_dir, tiny_snapshot, "other")
    with pytest.raises(
        TrainError, match=r"--stop-after must be a checkpoint step: \[2, 4\]"
    ):
        train.train(run_dir, tiny_snapshot, "test-commit", stop_after=3)
    other = tmp_path_factory.mktemp("other-tokenizer")
    shutil.copytree(tiny_snapshot, other, dirs_exist_ok=True)
    (other / "tokenizer.json").write_text((other / "tokenizer.json").read_text() + "\n")
    with pytest.raises(TrainError, match="isn't the tokenizer"):
        train.train(run_dir, other, "test-commit")
    path = labelled / "labels" / "train.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(TrainError, match="changed since"):
        train.train(run_dir, tiny_snapshot, "test-commit")


def test_train_refuses_a_prompt_the_template_does_not_give(
    labelled, tmp_path_factory
) -> None:
    """A template whose generation prompt has no empty think block (thinking on, or
    another model's) would train on a prompt vLLM never sends."""
    template = qwen3_chat_template().replace("{{- '<think>\\n\\n</think>\\n\\n' }}", "")
    assert template != qwen3_chat_template()
    snapshot = save_tiny_snapshot(tmp_path_factory.mktemp("other"), template)
    run_dir = prepare(labelled, snapshot=snapshot)
    with pytest.raises(TrainError) as error:
        train.train(run_dir, snapshot, "test-commit", out=lambda line: None)
    message = str(error.value)
    assert message.startswith("4 layout problems: ")
    assert "the chat template's generation prompt differs from the training prompt" in (
        message
    )
    assert not train.checkpoints(run_dir)


def test_layout_problems_count_tokens(tiny_snapshot) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_snapshot)
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "R"},
    ]
    prompt_tokens = len(train.encode(tokenizer, chat.prompt_text("S", "U")))
    good = Example("acc", "T", FIRST, prompt_tokens, 2)  # "R" and <|im_end|>
    assert train.layout_problems(tokenizer, good, messages) == []
    bad = dataclasses.replace(good, target_tokens=3)
    assert train.layout_problems(tokenizer, bad, messages) == [
        (
            f"acc: {prompt_tokens} prompt and 2 target tokens; prepare counted "
            f"{prompt_tokens} and 3"
        )
    ]
