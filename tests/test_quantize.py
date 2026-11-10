"""secbench.quantize: the calibration draw (seeded, capped, never a validation or test
release), merge on a tiny random Qwen3 with a tiny LoRA (the train group), and AWQ on
a tiny random Qwen3 (the quantize group: llm-compressor, its own environment; run with
UV_PROJECT_ENVIRONMENT=.venv-quantize uv run --group train --group quantize pytest
tests/test_quantize.py)."""

import json
import random
import shutil
from pathlib import Path

import pytest
from conftest import byte_tokenizer, qwen3_chat_template
from test_score import adapter  # noqa: F401 (a fixture)
from test_serving import write_serving_config
from test_train import RELEASES, write_labels

from secbench import chat, quantize, serving, splits, train
from secbench.quantize import QuantizeError
from secbench.train import Example

AAA, BBB, CCC, DDD, EEE, FFF = (r[0] for r in RELEASES)


def examples(*spec: tuple[str, int]) -> list[Example]:
    return [Example(a, "T", "first quarter", n - 10, 10) for a, n in spec]


def test_draw_is_seeded_capped_and_outside_the_test_splits() -> None:
    pool = examples(("a", 100), ("b", 5000), ("c", 200), ("d", 300), ("e", 400))
    chosen, counts = quantize.draw_calibration(pool, {"c"}, 2, 1000, seed=7)
    eligible = [pool[0], pool[3], pool[4]]  # b over the cap, c in a test split
    assert chosen == random.Random(7).sample(eligible, 2)
    assert counts == {
        "examples": 5,
        "over_cap": 1,
        "in_validation_or_test": 1,
        "eligible": 3,
        "drawn": 2,
    }
    again, _ = quantize.draw_calibration(pool, {"c"}, 2, 1000, seed=7)
    assert again == chosen
    other, _ = quantize.draw_calibration(pool, {"c"}, 3, 1000, seed=8)
    assert {e.accession for e in other} == {"a", "d", "e"}
    with pytest.raises(QuantizeError, match="4 calibration examples asked for; 3"):
        quantize.draw_calibration(pool, {"c"}, 4, 1000, seed=7)
    # The cap counts prompt and reply, and is inclusive: d's 300 tokens in all are
    # within a cap of 300, over one of 299.
    _, counts = quantize.draw_calibration(pool, set(), 1, 300, seed=7)
    assert counts["eligible"] == 3  # a, c and d
    chosen, counts = quantize.draw_calibration(pool, set(), 1, 299, seed=7)
    assert counts["eligible"] == 2 and chosen[0].accession in ("a", "c")


def write_split_tokens(root: Path, split: str, accessions: list[str]) -> None:
    rows = [
        {
            "ticker": "T",
            "release_accession": a,
            "prompt_tokens": "1",
            "status": splits.KEPT,
            "input_sha256": "0" * 64,
        }
        for a in accessions
    ]
    path = root / split / splits.TOKENS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(splits.csv_text(rows, splits.TOKEN_COLUMNS))


@pytest.fixture
def lab(tmp_path, monkeypatch) -> Path:
    """Labels for the six test_train releases, splits holding BBB (validation) and
    some other release, a frozen list holding CCC, and a serving config drawing 2
    examples of at most 2,600 tokens (EEE is longer)."""
    monkeypatch.chdir(tmp_path)
    write_labels(tmp_path / "labels")
    write_split_tokens(tmp_path / "splits", "validation", [BBB])
    write_split_tokens(tmp_path / "splits", "company_test", ["0000000009-25-000009"])
    write_split_tokens(tmp_path / "splits", "time_test", [])
    frozen = {name: "" for name in splits.FROZEN_COLUMNS} | {
        "split": "company_test",
        "release_accession": CCC,
    }
    (tmp_path / "frozen.csv").write_text(
        splits.csv_text([frozen], splits.FROZEN_COLUMNS)
    )
    write_serving_config(tmp_path / "serving.toml", calibration__examples="2")
    return tmp_path


def calibrate(lab: Path, *extra: str) -> int:
    return quantize.main(
        [
            "calibration",
            "--config", str(lab / "serving.toml"),
            "--labels", str(lab / "labels"),
            "--splits-dir", str(lab / "splits"),
            "--frozen", str(lab / "frozen.csv"),
            *extra,
        ],
        tokenizer=byte_tokenizer(),
    )  # fmt: skip


def test_calibration_files(lab, capsys) -> None:
    assert calibrate(lab) == 0
    out = lab / "data/serving/run"
    lines = (out / "calibration.jsonl").read_text().splitlines()
    drawn = [json.loads(line)["release_accession"] for line in lines]
    tokenizer = byte_tokenizer()
    messages = train.load_messages(lab / "labels" / "train.jsonl")
    index = train.read_csv(lab / "labels" / "examples.csv")
    every = train.count_tokens(tokenizer, messages, index)
    eligible = [e for e in every if e.accession in (AAA, DDD, FFF)]
    assert drawn == [e.accession for e in random.Random(7).sample(eligible, 2)]
    assert json.loads(lines[0])["messages"] == messages[drawn[0]]
    rows = train.read_csv(out / "calibration.csv")
    assert list(rows[0]) == quantize.CALIBRATION_COLUMNS
    assert [r["release_accession"] for r in rows] == drawn
    for row in rows:
        system, user, reply = (m["content"] for m in messages[row["release_accession"]])
        prompt_ids = tokenizer.encode(chat.prompt_text(system, user)).ids
        target_ids = tokenizer.encode(chat.target_text(reply)).ids
        assert int(row["tokens"]) == len(prompt_ids) + len(target_ids)
    manifest = json.loads((out / "calibration.json").read_text())
    assert manifest["counts"] == {
        "examples": 6,
        "over_cap": 1,  # EEE
        "in_validation_or_test": 2,  # BBB and CCC
        "eligible": 3,
        "drawn": 2,
    }
    assert manifest["settings"] == {"examples": 2, "max_tokens": 2600, "seed": 7}
    assert manifest["model"] == {"name": "test/tiny-qwen3", "revision": "0" * 40}
    assert manifest["sources"]["examples_sha256"] == train.file_sha256(
        lab / "labels" / "train.jsonl"
    )
    assert "2 of 3 eligible training examples (of 6; 1 over 2,600 tokens, 2 in a " in (
        capsys.readouterr().out
    )

    # Run again: kept. With other settings: refused.
    assert calibrate(lab) == 0
    assert "already drawn with these settings; kept" in capsys.readouterr().out
    write_serving_config(lab / "serving.toml", calibration__examples="3")
    assert calibrate(lab) == 1
    assert "settings {'examples': 2, " in capsys.readouterr().err

    # A drawn file changed, or the labels changed since: refused.
    write_serving_config(lab / "serving.toml", calibration__examples="2")
    config = serving.load_config(lab / "serving.toml")
    quantize.check_calibration(config, out, labels_dir=lab / "labels")
    with (out / "calibration.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(QuantizeError, match="calibration.csv isn't the file"):
        quantize.check_calibration(config, out)


def test_check_calibration_refuses_a_test_release(lab) -> None:
    assert calibrate(lab) == 0
    config = serving.load_config(lab / "serving.toml")
    out = lab / "data/serving/run"
    drawn = json.loads((out / "calibration.jsonl").read_text().splitlines()[0])
    with pytest.raises(QuantizeError, match="holds validation or test releases"):
        quantize.check_calibration(config, out, excluded={drawn["release_accession"]})


def test_calibration_without_the_splits_stops(lab, capsys) -> None:
    shutil.rmtree(lab / "splits" / "time_test")
    assert calibrate(lab) == 1
    assert "time_test/tokens.csv not found; run secbench.splits" in (
        capsys.readouterr().err
    )


# merge (the train group): the merged model gives the unmerged model's outputs


def logits(model, ids):
    import torch

    with torch.inference_mode():
        return model(input_ids=torch.tensor([ids])).logits


def test_merge_gives_the_unmerged_outputs(tiny_snapshot, adapter, tmp_path) -> None:  # noqa: F811
    torch = pytest.importorskip("torch")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    out = tmp_path / "merged"
    lines = []
    manifest = quantize.merge(
        tiny_snapshot,
        adapter,
        out,
        device="cpu",
        save_dtype="float32",
        out_fn=lines.append,
    )
    assert not (tmp_path / "merged.part").exists()
    unmerged = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(tiny_snapshot, dtype=torch.float32),
        adapter,
    ).eval()
    base = AutoModelForCausalLM.from_pretrained(tiny_snapshot, dtype=torch.float32)
    merged = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    ids = list(range(40, 90))
    assert torch.allclose(logits(merged, ids), logits(unmerged, ids), atol=1e-5)
    # The adapter changes the outputs, so the check means something.
    assert not torch.allclose(logits(base, ids), logits(unmerged, ids), atol=1e-3)
    prompt = torch.tensor([ids])
    kwargs = {"max_new_tokens": 8, "do_sample": False}
    with torch.inference_mode():
        assert torch.equal(
            merged.generate(input_ids=prompt, **kwargs),
            unmerged.generate(input_ids=prompt, **kwargs),
        )
    # The base's tokenizer files and generation config are copied unchanged.
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        assert (out / name).read_bytes() == (tiny_snapshot / name).read_bytes()
    assert manifest["kind"] == "merged"
    assert manifest["inputs"]["adapter"]["files"]["adapter_model.safetensors"] == (
        train.file_sha256(adapter / "adapter_model.safetensors")
    )
    assert manifest["settings"]["merge_dtype"] == "float32"
    assert manifest["outputs"] == {
        k: v for k, v in quantize.files_sha256(out).items() if k != "manifest.json"
    }
    assert json.loads((out / "manifest.json").read_text()) == manifest
    assert set(manifest["timings_s"]) == {
        "hash_inputs",
        "load",
        "merge",
        "save",
        "hash_outputs",
        "total",
    }
    with pytest.raises(QuantizeError, match="already exists"):
        quantize.merge(tiny_snapshot, adapter, out, device="cpu")


def test_merge_saves_bfloat16(tiny_snapshot, adapter, tmp_path) -> None:  # noqa: F811
    torch = pytest.importorskip("torch")
    from safetensors import safe_open

    out = tmp_path / "merged"
    quantize.merge(tiny_snapshot, adapter, out, device="cpu", out_fn=lambda _: None)
    [shard] = out.glob("*.safetensors")
    with safe_open(shard, "pt") as f:
        dtypes = {f.get_tensor(k).dtype for k in f.keys()}  # noqa: SIM118
    assert dtypes == {torch.bfloat16}


# awq (the quantize group): a tiny Qwen3 whose layers divide into groups of 128


@pytest.fixture(scope="module")
def awq_snapshot(tmp_path_factory) -> Path:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    path = tmp_path_factory.mktemp("awq-snapshot")
    raw = byte_tokenizer()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, eos_token="<|im_end|>", pad_token="<|endoftext|>"
    )
    tokenizer.chat_template = qwen3_chat_template()
    tokenizer.save_pretrained(path)
    end, text_end = raw.token_to_id("<|im_end|>"), raw.token_to_id("<|endoftext|>")
    config = transformers.Qwen3Config(
        vocab_size=raw.get_vocab_size(),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
        eos_token_id=end,
        pad_token_id=text_end,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config)
    model.generation_config = transformers.GenerationConfig(
        eos_token_id=[end, text_end], pad_token_id=text_end
    )
    model.save_pretrained(path)
    return path


def write_awq_calibration(snapshot: Path, folder: Path) -> Path:
    """Four short calibration examples for the tiny model, with their token counts
    by its tokenizer (calibration.jsonl and calibration.csv); calibration.jsonl."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    folder.mkdir()
    lines, rows = [], []

    for position, (accession, ticker, kind, user) in enumerate(RELEASES[:4]):
        messages = [
            {"role": "system", "content": "Reply with JSON."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": '{"revenue": 1}'},
        ]
        lines.append(json.dumps({"release_accession": accession, "messages": messages}))
        prompt = train.encode(
            tokenizer, chat.prompt_text(*(m["content"] for m in messages[:2]))
        )
        target = train.encode(tokenizer, chat.target_text(messages[2]["content"]))
        rows.append(
            {
                "position": position,
                "release_accession": accession,
                "ticker": ticker,
                "kind": kind,
                "prompt_tokens": len(prompt),
                "target_tokens": len(target),
                "tokens": len(prompt) + len(target),
            }
        )
    (folder / "calibration.jsonl").write_text("\n".join(lines) + "\n")
    (folder / "calibration.csv").write_text(
        train.rows_csv(rows, quantize.CALIBRATION_COLUMNS)
    )
    return folder / "calibration.jsonl"


def test_awq_on_a_tiny_qwen3(awq_snapshot, tmp_path) -> None:
    pytest.importorskip("llmcompressor")
    calibration = tmp_path / "calibration"
    write_awq_calibration(awq_snapshot, calibration)
    rows = train.read_csv(calibration / "calibration.csv")
    out = tmp_path / "int4-awq"
    manifest = quantize.awq(
        awq_snapshot,
        calibration / "calibration.jsonl",
        out,
        scheme="W4A16_ASYM",
        duo_scaling="both",
        ignore=["lm_head"],
        max_tokens=4096,
        out_fn=lambda _: None,
    )
    config = json.loads((out / "config.json").read_text())["quantization_config"]
    assert config["quant_method"] == "compressed-tensors"
    assert config["format"] == "pack-quantized"
    [group] = config["config_groups"].values()
    assert group["targets"] == ["Linear"]
    weights = group["weights"]
    assert (weights["num_bits"], weights["group_size"], weights["symmetric"]) == (
        4,
        128,
        False,
    )
    assert group["input_activations"] is None  # 16-bit activations
    assert "lm_head" in config["ignore"]
    assert manifest["quantization_config"] == config
    assert manifest["inputs"]["calibration"]["examples"] == 4
    assert manifest["settings"]["n_grid"] == 20
    for name in ("tokenizer.json", "generation_config.json"):
        assert (out / name).read_bytes() == (awq_snapshot / name).read_bytes()
    # The saved model loads (compressed-tensors) and generates.
    import torch
    from transformers import AutoModelForCausalLM

    loaded = AutoModelForCausalLM.from_pretrained(out)
    with torch.inference_mode():
        generated = loaded.generate(
            input_ids=torch.tensor([list(range(40, 60))]),
            max_new_tokens=4,
            do_sample=False,
        )
    assert generated.shape[1] > 20
    # A calibration count that isn't the tokenizer's stops before any work.
    rows[0]["tokens"] = str(int(rows[0]["tokens"]) + 1)
    (calibration / "calibration.csv").write_text(
        train.rows_csv(rows, quantize.CALIBRATION_COLUMNS)
    )
    with pytest.raises(QuantizeError, match="tokens here, .* when it was drawn"):
        quantize.awq(
            awq_snapshot,
            calibration / "calibration.jsonl",
            tmp_path / "again",
            scheme="W4A16_ASYM",
            duo_scaling="both",
            ignore=["lm_head"],
            max_tokens=4096,
        )


def test_awq_retries_out_of_gpu_memory_with_its_caches_on_the_cpu(
    awq_snapshot, tmp_path, monkeypatch
) -> None:
    llmcompressor = pytest.importorskip("llmcompressor")
    import torch

    real, calls = llmcompressor.oneshot, []

    def oneshot(**kwargs):
        awq_modifier = kwargs["recipe"][0]
        calls.append(awq_modifier.offload_device)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 30 GiB")
        if not torch.cuda.is_available():
            # llm-compressor offloads its caches to the CPU only from CUDA (on
            # Apple's GPU it refuses): the retry's real calibration runs without.
            awq_modifier.offload_device = None
        return real(**kwargs)

    monkeypatch.setattr(llmcompressor, "oneshot", oneshot)
    calibration = write_awq_calibration(awq_snapshot, tmp_path / "calibration")
    lines = []
    manifest = quantize.awq(
        awq_snapshot,
        calibration,
        tmp_path / "int4-awq",
        scheme="W4A16_ASYM",
        duo_scaling="both",
        ignore=["lm_head"],
        max_tokens=4096,
        out_fn=lines.append,
    )
    # The first attempt leaves the caches where AWQ puts them (not set), the retry
    # offloads them to the CPU.
    assert calls[0] != torch.device("cpu") and calls[1] == torch.device("cpu")
    assert manifest["settings"]["cache_offload"] == "cpu"
    [retry] = manifest["settings"]["retries"]
    assert retry.startswith("out of GPU memory with AWQ's caches on the GPU")
    assert any("again with them on the CPU" in line for line in lines)

    def broken(**kwargs):
        raise RuntimeError("a shape mismatch")

    monkeypatch.setattr(llmcompressor, "oneshot", broken)
    with pytest.raises(RuntimeError, match="shape mismatch"):  # not retried
        quantize.awq(
            awq_snapshot,
            calibration,
            tmp_path / "again",
            scheme="W4A16_ASYM",
            duo_scaling="both",
            ignore=["lm_head"],
            max_tokens=4096,
        )
