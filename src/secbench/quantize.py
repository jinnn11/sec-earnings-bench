"""The serving benchmark's models (PLAN.md Decisions 2026-11-09): the AWQ calibration
examples, the adapter merged into the base model, and the merged model quantized to
4-bit weights with AWQ.

Usage: uv run python -m secbench.quantize calibration --config configs/serving/<run>.toml
           [--labels data/stage2/labels] [--raw data/raw] [--hf-cache data/raw/hf-hub]
           [--splits-dir data/splits] [--frozen data/frozen_test_releases.csv]
       uv run --group train python -m secbench.quantize merge --base SNAPSHOT
           --adapter CHECKPOINT --out DIR [--device auto]
       .venv-quantize/bin/python -m secbench.quantize awq --model MERGED
           --calibration data/serving/<run>/calibration.jsonl --out DIR
           [--scheme W4A16_ASYM] [--duo-scaling both] [--ignore lm_head]
           [--max-tokens 16384]

calibration (the laptop; tokenizers only): every training example of labels'
train.jsonl, laid out as training lays it out (secbench.chat: the prompt with
thinking off, then the reply and <|im_end|>) and counted with the run's model's
tokenizer (Qwen3-8B's pinned tokenizer.json, as labels counts, or the downloaded
snapshot's); those over the config's token cap are left out, and so is any release
of a validation or test split (each split's tokens.csv, kept or left out, and the
frozen list; train.jsonl holds none, so this is a guard). From the rest,
random.Random(seed).sample draws the config's number of examples. Written to
data/serving/<run>/: calibration.jsonl (each drawn example's release and messages, in
drawn order), calibration.csv (its tokens) and calibration.json (the settings, the
sources' and outputs' sha256, the counts). A folder already there is kept if it
records the same settings and sources; any other stops.

merge (the train group): the base snapshot in float32 with the LoRA adapter (peft),
merge_and_unload, then the weights rounded once to bfloat16 and saved as safetensors;
the snapshot's tokenizer files and generation_config.json are copied unchanged, so
the merged model is served with the base model's tokenizer, chat template and stop
tokens. float32 first, so the merged weights carry one rounding to bfloat16 rather
than two (the base's and the sum's).

awq (the quantize group, its own environment: llm-compressor conflicts with vLLM):
the merged model in bfloat16, calibrated with llm-compressor's oneshot on the
calibration examples, each tokenized as training tokenized it (the prompt's tokens,
then the reply's; the counts must be calibration.csv's), with llm-compressor's AWQ
recipe: AWQModifier (duo_scaling) and QuantizationModifier (the scheme, here
W4A16_ASYM: 4-bit integer weights in groups of 128 with zero points, 16-bit
activations; Linear layers but lm_head), then saved compressed (compressed-tensors'
format, which vLLM loads), with the merged model's tokenizer files and generation
config copied unchanged. The sequential pipeline loads one decoder layer at a time
onto the GPU.

Each model folder is written to <out>.part and renamed when complete, with
manifest.json: the inputs' sha256, library versions, settings, timings and the
outputs' sha256 (and, for awq, the saved quantization config). An existing <out> is
never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from secbench import chat, extract, labels, serving, splits, train
from secbench.labels import DEFAULT_OUT as DEFAULT_LABELS
from secbench.labels import EXAMPLE_INDEX, EXAMPLES
from secbench.releases import write_text

CALIBRATION_COLUMNS = [
    "position",
    "release_accession",
    "ticker",
    "kind",
    "prompt_tokens",
    "target_tokens",
    "tokens",
]
# Copied unchanged from the model a folder is made from (those that exist).
COPIED = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)
MAX_SHARD = "5GB"
PARTIAL = ".part"
TARGETS = ["Linear"]


class QuantizeError(Exception):
    """A model or the calibration set can't be made; the message says why."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def versions(*packages: str) -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for package in packages:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            found[package] = None
    return found


def files_sha256(folder: Path) -> dict[str, str]:
    """Each file directly in folder (symbolic links followed, as in a Hugging Face
    snapshot), by name."""
    return {
        p.name: train.file_sha256(p) for p in sorted(folder.iterdir()) if p.is_file()
    }


def weight_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.glob("*.safetensors"))


def copy_unchanged(source: Path, target: Path) -> list[str]:
    copied = []
    for name in COPIED:
        if (source / name).is_file():
            shutil.copyfile(source / name, target / name)
            copied.append(name)
    return copied


def start_partial(out: Path) -> Path:
    if out.exists():
        raise QuantizeError(f"{out} already exists; move it away to make it again")
    partial = out.with_name(out.name + PARTIAL)
    if partial.exists():
        shutil.rmtree(partial)  # an earlier attempt's, never complete
    partial.parent.mkdir(parents=True, exist_ok=True)
    return partial


def finish_partial(partial: Path, out: Path, manifest: Mapping[str, Any]) -> None:
    extract.write_json(partial / serving.MANIFEST, manifest)
    partial.replace(out)


# Calibration


def excluded_releases(splits_dir: Path, frozen: Path) -> set[str]:
    """Every release of a validation or test split: each split's tokens.csv (kept or
    left out) and the frozen list."""
    found: set[str] = set()
    for split in splits.SPLITS:
        path = splits_dir / split / splits.TOKENS
        if not path.is_file():
            raise QuantizeError(f"{path} not found; run secbench.splits")
        rows = splits.read_table(path, splits.TOKEN_COLUMNS)
        found |= {r["release_accession"] for r in rows}
    if frozen.is_file():
        rows = splits.read_table(frozen, splits.FROZEN_COLUMNS)
        found |= {r["release_accession"] for r in rows}
    return found


def draw_calibration(
    examples: Sequence[train.Example],
    excluded: set[str],
    count: int,
    cap: int,
    seed: int,
) -> tuple[list[train.Example], dict[str, int]]:
    """count examples of at most cap tokens and outside every validation and test
    split, drawn with random.Random(seed) from those in examples' order."""
    eligible = [e for e in examples if e.tokens <= cap and e.accession not in excluded]
    counts = {
        "examples": len(examples),
        "over_cap": sum(e.tokens > cap for e in examples),
        "in_validation_or_test": sum(e.accession in excluded for e in examples),
        "eligible": len(eligible),
        "drawn": count,
    }
    if count > len(eligible):
        raise QuantizeError(
            f"{count} calibration examples asked for; {len(eligible)} are eligible"
        )
    return random.Random(seed).sample(eligible, count), counts


def calibration_tokenizer(
    config: serving.ServingConfig, raw: Path, hf_cache: Path
) -> tuple[Any, Path]:
    """The run's model's tokenizer (the tokenizers package) and its file: Qwen3-8B's
    pinned one, as labels counts tokens, or the downloaded snapshot's."""
    from tokenizers import Tokenizer

    if (config.base, config.revision) == (labels.QWEN3, labels.QWEN3_REVISION):
        path = labels.fetch_pinned(
            labels.TOKENIZER_URL, labels.tokenizer_path(raw), labels.TOKENIZER_SHA256
        )
    else:
        _, run_config, _ = train.read_run(config.run_dir)
        path = train.snapshot_path(run_config, hf_cache) / train.TOKENIZER
    return Tokenizer.from_file(str(path)), path


def calibration_settings(config: serving.ServingConfig) -> dict[str, int]:
    return {
        "examples": config.calibration_examples,
        "max_tokens": config.calibration_max_tokens,
        "seed": config.calibration_seed,
    }


def make_calibration(
    config: serving.ServingConfig,
    *,
    labels_dir: Path,
    tokenizer: Any,
    tokenizer_path: Path,
    splits_dir: Path,
    frozen: Path,
    out: Path,
    commit: dict[str, Any],
) -> dict[str, Any]:
    """Draw the calibration examples and write them to out; the manifest."""
    examples_path, index_path = labels_dir / EXAMPLES, labels_dir / EXAMPLE_INDEX
    if not index_path.is_file():
        raise QuantizeError(f"{index_path} not found")
    messages = train.load_messages(examples_path)
    index = train.read_csv(index_path)
    if [r["release_accession"] for r in index] != list(messages):
        raise QuantizeError(f"{index_path} and {examples_path} list other releases")
    examples = train.count_tokens(tokenizer, messages, index)
    excluded = excluded_releases(splits_dir, frozen)
    chosen, counts = draw_calibration(
        examples,
        excluded,
        config.calibration_examples,
        config.calibration_max_tokens,
        config.calibration_seed,
    )
    jsonl = "".join(
        json.dumps(
            {"release_accession": e.accession, "messages": messages[e.accession]},
            ensure_ascii=False,
        )
        + "\n"
        for e in chosen
    )
    index_csv = train.rows_csv(
        (
            {
                "position": position,
                "release_accession": e.accession,
                "ticker": e.ticker,
                "kind": e.kind,
                "prompt_tokens": e.prompt_tokens,
                "target_tokens": e.target_tokens,
                "tokens": e.tokens,
            }
            for position, e in enumerate(chosen)
        ),
        CALIBRATION_COLUMNS,
    )
    tokens = [e.tokens for e in chosen]
    manifest = {
        "config": str(config.path),
        "config_sha256": config.sha256,
        "run": config.run,
        "model": {"name": config.base, "revision": config.revision},
        "created": now(),
        **commit,
        "settings": calibration_settings(config),
        "rule": "training examples of at most max_tokens tokens (prompt and "
        "reply as training lays them out), outside every validation and test split, "
        "drawn with random.Random(seed).sample in train.jsonl's order",
        "sources": {
            "examples": str(examples_path),
            "examples_sha256": train.file_sha256(examples_path),
            "index": str(index_path),
            "index_sha256": train.file_sha256(index_path),
            "tokenizer": str(tokenizer_path),
            "tokenizer_sha256": train.file_sha256(tokenizer_path),
        },
        "counts": counts,
        "by_kind": dict(Counter(e.kind for e in chosen)),
        "tokens": {
            "total": sum(tokens),
            "mean": round(sum(tokens) / len(tokens)),
            "min": min(tokens),
            "max": max(tokens),
        },
        "files": {
            serving.CALIBRATION: hashlib.sha256(jsonl.encode()).hexdigest(),
            serving.CALIBRATION_INDEX: hashlib.sha256(index_csv.encode()).hexdigest(),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / serving.CALIBRATION, jsonl)
    write_text(out / serving.CALIBRATION_INDEX, index_csv)
    extract.write_json(out / serving.CALIBRATION_MANIFEST, manifest)
    return manifest


def check_calibration(
    config: serving.ServingConfig,
    folder: Path,
    *,
    labels_dir: Path | None = None,
    excluded: set[str] | None = None,
) -> dict[str, Any]:
    """The calibration folder's manifest, after checking its files are the ones it
    records, its settings the config's, its sources labels' current files (if
    labels_dir), and that it holds no release of excluded."""
    path = folder / serving.CALIBRATION_MANIFEST
    if not path.is_file():
        raise QuantizeError(
            f"{path} not found; run: uv run python -m secbench.quantize calibration "
            f"--config {config.path}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if manifest.get("settings") != calibration_settings(config):
        problems.append(
            f"settings {manifest.get('settings')}, not the config's "
            f"{calibration_settings(config)}"
        )
    if manifest.get("model") != {"name": config.base, "revision": config.revision}:
        problems.append(f"made for {manifest.get('model')}")
    for name, digest in manifest.get("files", {}).items():
        if not (folder / name).is_file() or train.file_sha256(folder / name) != digest:
            problems.append(f"{folder / name} isn't the file the manifest records")
    if labels_dir is not None:
        sources = manifest.get("sources", {})
        for key, name in (("examples", EXAMPLES), ("index", EXAMPLE_INDEX)):
            current = labels_dir / name
            if not current.is_file() or train.file_sha256(current) != sources.get(
                f"{key}_sha256"
            ):
                problems.append(f"{current} isn't the file it was drawn from")
    if not problems and excluded is not None:
        drawn = {
            json.loads(line)["release_accession"]
            for line in (folder / serving.CALIBRATION).read_text().splitlines()
        }
        if drawn & excluded:
            problems.append(
                "holds validation or test releases: "
                + ", ".join(sorted(drawn & excluded))
            )
    if problems:
        raise QuantizeError(f"{folder}: " + "; ".join(problems))
    return manifest


def read_calibration(path: Path) -> list[tuple[str, list[dict[str, str]], int]]:
    """Each calibration example's release, messages and token count (calibration.csv
    beside it), in drawn order."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    with (path.parent / serving.CALIBRATION_INDEX).open(newline="") as f:
        index = list(csv.DictReader(f))
    if [r["release_accession"] for r in rows] != [
        r["release_accession"] for r in index
    ]:
        raise QuantizeError(f"{path} and its {serving.CALIBRATION_INDEX} differ")
    return [
        (row["release_accession"], row["messages"], int(entry["tokens"]))
        for row, entry in zip(rows, index, strict=True)
    ]


# merge (the train group)


def merge(
    base: Path,
    adapter: Path,
    out: Path,
    *,
    device: str = "auto",
    save_dtype: str = "bfloat16",
    out_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The adapter merged into base, saved to out (see the module docstring)."""
    import torch
    from peft import PeftModel

    partial = start_partial(out)
    started, clock = now(), time.perf_counter()
    inputs = {
        "base": {"path": str(base), "files": files_sha256(base)},
        "adapter": {"path": str(adapter), "files": files_sha256(adapter)},
    }
    timings = {"hash_inputs": time.perf_counter() - clock}
    device = train.pick_device(device)
    out_fn(f"loading {base} in float32 on {device}, with the adapter {adapter}")
    mark = time.perf_counter()
    model = train.load_model(base, torch.float32, device)
    model = PeftModel.from_pretrained(model, adapter)
    timings["load"] = time.perf_counter() - mark
    mark = time.perf_counter()
    merged = model.merge_and_unload().to(getattr(torch, save_dtype))
    timings["merge"] = time.perf_counter() - mark
    out_fn(f"merged; saving in {save_dtype} to {out}")
    mark = time.perf_counter()
    merged.save_pretrained(partial, safe_serialization=True, max_shard_size=MAX_SHARD)
    copied = copy_unchanged(base, partial)
    timings["save"] = time.perf_counter() - mark
    mark = time.perf_counter()
    outputs = files_sha256(partial)
    timings["hash_outputs"] = time.perf_counter() - mark
    timings["total"] = time.perf_counter() - clock
    manifest = {
        "kind": serving.MERGED,
        "started": started,
        "finished": now(),
        **extract.code_version(),
        "inputs": inputs,
        "versions": versions("torch", "transformers", "peft", "safetensors"),
        "settings": {
            "merge_dtype": "float32",
            "save_dtype": save_dtype,
            "device": device,
            "max_shard_size": MAX_SHARD,
            "copied_unchanged": copied,
        },
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "outputs": outputs,
        "weight_bytes": weight_bytes(partial),
    }
    finish_partial(partial, out, manifest)
    out_fn(f"wrote {out} ({manifest['weight_bytes'] / 2**30:.2f} GiB of weights)")
    return manifest


# awq (the quantize group, in its own environment)


def calibration_ids(
    tokenizer: Any, examples: Sequence[tuple[str, list[dict[str, str]], int]]
) -> list[list[int]]:
    """Each example's tokens as training tokenized it: the prompt's, then the
    reply's; the counts must be calibration.csv's."""
    found = []
    for accession, messages, tokens in examples:
        system, user, reply = (m["content"] for m in messages)
        ids = train.encode(tokenizer, chat.prompt_text(system, user)) + train.encode(
            tokenizer, chat.target_text(reply)
        )
        if len(ids) != tokens:
            raise QuantizeError(
                f"{accession}: {len(ids)} tokens here, {tokens} when it was drawn"
            )
        found.append(ids)
    return found


def awq(
    model_dir: Path,
    calibration: Path,
    out: Path,
    *,
    scheme: str,
    duo_scaling: bool | str,
    ignore: Sequence[str],
    max_tokens: int,
    out_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """model_dir quantized with llm-compressor's AWQ recipe, saved compressed to out
    (see the module docstring)."""
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    partial = start_partial(out)
    started, clock = now(), time.perf_counter()
    examples = read_calibration(calibration)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ids = calibration_ids(tokenizer, examples)
    longest = max(len(i) for i in ids)
    if longest > max_tokens:
        raise QuantizeError(f"an example has {longest} tokens, over {max_tokens}")
    dataset = Dataset.from_dict(
        {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}
    )
    inputs = {
        "model": {
            "path": str(model_dir),
            "manifest_sha256": train.file_sha256(model_dir / serving.MANIFEST)
            if (model_dir / serving.MANIFEST).is_file()
            else None,
            "files": files_sha256(model_dir),
        },
        "calibration": {
            "path": str(calibration),
            "sha256": train.file_sha256(calibration),
            "index_sha256": train.file_sha256(
                calibration.parent / serving.CALIBRATION_INDEX
            ),
            "examples": len(ids),
            "tokens": sum(len(i) for i in ids),
            "longest": longest,
        },
    }
    timings = {"prepare": time.perf_counter() - clock}
    out_fn(f"loading {model_dir} in bfloat16")
    mark = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    timings["load"] = time.perf_counter() - mark
    awq_modifier = AWQModifier(duo_scaling=duo_scaling)
    recipe = [
        awq_modifier,
        QuantizationModifier(ignore=list(ignore), scheme=scheme, targets=TARGETS),
    ]
    out_fn(
        f"AWQ ({scheme}) on {len(ids)} calibration examples, "
        f"{inputs['calibration']['tokens']:,} tokens"
    )
    mark = time.perf_counter()
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=max_tokens,
        num_calibration_samples=len(ids),
        shuffle_calibration_samples=False,
    )
    timings["calibrate"] = time.perf_counter() - mark
    out_fn(f"saving compressed to {out}")
    mark = time.perf_counter()
    model.save_pretrained(partial, save_compressed=True, max_shard_size=MAX_SHARD)
    copied = copy_unchanged(model_dir, partial)
    timings["save"] = time.perf_counter() - mark
    timings["total"] = time.perf_counter() - clock
    config = json.loads((partial / "config.json").read_text())
    manifest = {
        "kind": serving.INT4_AWQ,
        "started": started,
        "finished": now(),
        **extract.code_version(),
        "inputs": inputs,
        "versions": versions(
            "torch", "transformers", "llmcompressor", "compressed-tensors", "datasets"
        ),
        "settings": {
            "recipe": "AWQModifier, then QuantizationModifier (llm-compressor)",
            "scheme": scheme,
            "targets": TARGETS,
            "ignore": list(ignore),
            "duo_scaling": duo_scaling,
            "n_grid": awq_modifier.n_grid,
            "max_seq_length": max_tokens,
            "num_calibration_samples": len(ids),
            "load_dtype": "bfloat16",
            "copied_unchanged": copied,
        },
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "outputs": files_sha256(partial),
        "weight_bytes": weight_bytes(partial),
        "quantization_config": config.get("quantization_config"),
    }
    finish_partial(partial, out, manifest)
    out_fn(f"wrote {out} ({manifest['weight_bytes'] / 2**30:.2f} GiB of weights)")
    return manifest


# Command line


def duo_scaling_value(value: str) -> bool | str:
    choices = {"true": True, "false": False, "both": "both"}
    if value.lower() not in choices:
        raise argparse.ArgumentTypeError("must be true, false or both")
    return choices[value.lower()]


def main(argv: list[str] | None = None, *, tokenizer: Any = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.quantize",
        description="The serving benchmark's calibration examples and models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    cal = commands.add_parser("calibration", help="draw the AWQ calibration examples")
    cal.add_argument("--config", type=Path, required=True)
    cal.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    cal.add_argument("--raw", type=Path, default=labels.DEFAULT_RAW)
    cal.add_argument("--hf-cache", type=Path, default=train.DEFAULT_HF_CACHE)
    cal.add_argument("--splits-dir", type=Path, default=splits.DEFAULT_OUT)
    cal.add_argument("--frozen", type=Path, default=splits.FROZEN_LIST)
    cal.add_argument("--out", type=Path, help="default: data/serving/<run>")
    mer = commands.add_parser("merge", help="merge the adapter into the base model")
    mer.add_argument("--base", type=Path, required=True, help="the base snapshot")
    mer.add_argument("--adapter", type=Path, required=True, help="the checkpoint")
    mer.add_argument("--out", type=Path, required=True)
    mer.add_argument("--device", choices=train.DEVICES, default="auto")
    q = commands.add_parser("awq", help="quantize a model to 4-bit weights (AWQ)")
    q.add_argument("--model", type=Path, required=True)
    q.add_argument("--calibration", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--scheme", default="W4A16_ASYM")
    q.add_argument("--duo-scaling", type=duo_scaling_value, default="both")
    q.add_argument("--ignore", nargs="*", default=["lm_head"])
    q.add_argument("--max-tokens", type=extract.positive_int, default=16384)
    args = parser.parse_args(argv)
    try:
        if args.command == "merge":
            merge(args.base, args.adapter, args.out, device=args.device)
            return 0
        if args.command == "awq":
            awq(
                args.model,
                args.calibration,
                args.out,
                scheme=args.scheme,
                duo_scaling=args.duo_scaling,
                ignore=args.ignore,
                max_tokens=args.max_tokens,
            )
            return 0
        config = serving.load_config(args.config)
        out = args.out or config.calibration_dir
        if (out / serving.CALIBRATION_MANIFEST).is_file():
            manifest = check_calibration(
                config,
                out,
                labels_dir=args.labels,
                excluded=excluded_releases(args.splits_dir, args.frozen),
            )
            print(f"{out}: already drawn with these settings; kept")
        else:
            path = None
            if tokenizer is None:
                tokenizer, path = calibration_tokenizer(config, args.raw, args.hf_cache)
            manifest = make_calibration(
                config,
                labels_dir=args.labels,
                tokenizer=tokenizer,
                tokenizer_path=path or args.labels / EXAMPLES,
                splits_dir=args.splits_dir,
                frozen=args.frozen,
                out=out,
                commit=extract.code_version(),
            )
        print(format_calibration(out, manifest))
        return 0
    except (
        QuantizeError,
        serving.ServingError,
        train.TrainError,
        labels.LabelError,
        splits.SplitError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def format_calibration(out: Path, manifest: Mapping[str, Any]) -> str:
    counts, tokens = manifest["counts"], manifest["tokens"]
    settings = manifest["settings"]
    return (
        f"{out}: {counts['drawn']} of {counts['eligible']:,} eligible training "
        f"examples (of {counts['examples']:,}; {counts['over_cap']:,} over "
        f"{settings['max_tokens']:,} tokens, {counts['in_validation_or_test']} in a "
        f"validation or test split), seed {settings['seed']}; {tokens['total']:,} "
        f"tokens, mean {tokens['mean']:,}, longest {tokens['max']:,}"
    )


if __name__ == "__main__":
    sys.exit(main())
