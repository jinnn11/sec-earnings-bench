"""Stage 2 LoRA fine-tuning (PLAN.md Decisions 2026-10-27 and 2026-11-01): Qwen3-8B
with thinking off, on secbench.labels' examples, with the loss on the reply only.

A run lives in training/<run>/ and takes three commands:

  uv run python -m secbench.train download --config configs/train/qwen3-8b-2k.toml
  uv run python -m secbench.train prepare --config configs/train/qwen3-8b-2k.toml
      [--run-name NAME] [--subset N|all]
  uv run --group train python -m secbench.train train training/<run> [--stop-after N]

download fetches the model at the config's pinned revision into --hf-cache
(data/raw/hf-hub), once (huggingface_hub; no SEC request).

prepare needs only the model's tokenizer.json (the tokenizers package). Each example
of train.jsonl is laid out as vLLM sends it with thinking off (secbench.chat): the
prompt is the system and user messages and the generation prompt, the target the
reply and <|im_end|>. Examples of more than max_tokens (prompt plus target) are left
out and listed in dropped.csv. From the rest, a seeded subset of `subset` examples (or
all) is drawn, stratified by quarter kind (labels' examples.csv): each kind gets its
share of the examples within the cap, by largest remainder. The subset is shuffled
once per epoch, from the same seed. examples.csv records the exact list in training
order; config.json the settings, seed, commit, the sha256 of train.jsonl and of
labels' examples.csv, labels' manifest, and the schedule (steps and checkpoints).
prepare refuses uncommitted changes to tracked files and an existing run folder.

train (the train dependency group: torch, transformers, peft) first checks every
example against the model's own chat template (transformers' apply_chat_template, as
vLLM renders it): the prompt must be exactly the generation-prompt rendering with
thinking off, prompt + target + "\\n" exactly the whole conversation's, and the
token counts prepare's. Then it trains a LoRA adapter on the configured projections,
the base model frozen in the config's dtype, batch size 1 with gradient accumulation
and gradient checkpointing. The forward pass computes logits only at the positions
that predict the target (logits_to_keep; the target ends the sequence), and the loss
is the cross-entropy summed over the target tokens of an accumulation window, divided
by their number. AdamW; the learning rate rises linearly over warmup_steps, then
falls along a cosine (lr_factor); the gradient norm is clipped. Every save_every steps
and at the last, a checkpoint in checkpoints/step-NNNNN/: the adapter
(adapter_config.json and adapter_model.safetensors; no tokenizer files, so vLLM uses
the base model's), the optimizer, scheduler and RNG states (training_state.pt) and
state.json. It is written to a .part folder and renamed, so a checkpoint is either
complete or absent.

train continues a run from its last checkpoint: adapter, optimizer, scheduler and RNG
states restored, train_log.jsonl cut back to that step, the examples after it in the
recorded order. A finished run is left as it is. train refuses a run prepared at
another commit, uncommitted changes to tracked files, and a train.jsonl whose sha256
isn't config.json's. --stop-after N stops after the checkpoint at step N (for testing
a resume). Each session appends the library versions, device and starting step to
sessions.jsonl; each optimizer step appends to train_log.jsonl (loss, learning rate,
gradient norm, tokens, seconds, peak memory). On a GPU a run isn't reproducible bit
for bit (some CUDA kernels aren't deterministic); its data, their order and the
adapter's initialization are.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import io
import json
import math
import platform
import random
import shutil
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from secbench import chat, extract, labels
from secbench.releases import write_text
from secbench.stage2 import current_commit

TRAINING_DIR = Path("training")
DEFAULT_HF_CACHE = Path("data/raw/hf-hub")
CONFIG = "config.json"
EXAMPLES = "examples.csv"
DROPPED = "dropped.csv"
TRAIN_LOG = "train_log.jsonl"
SESSIONS = "sessions.jsonl"
CHECKPOINTS = "checkpoints"
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
TRAINING_STATE = "training_state.pt"
STATE = "state.json"
PARTIAL = ".part"
TOKENIZER = "tokenizer.json"
LABELS_MANIFEST = "manifest.json"  # beside labels' examples.csv
EXAMPLE_COLUMNS = [
    "position",
    "epoch",
    "release_accession",
    "ticker",
    "kind",
    "prompt_tokens",
    "target_tokens",
    "tokens",
]
DROPPED_COLUMNS = [
    "release_accession",
    "ticker",
    "kind",
    "prompt_tokens",
    "target_tokens",
    "tokens",
]
DTYPES = ("bfloat16", "float32")
DEVICES = ("auto", "cuda", "mps", "cpu")
ALL = "all"
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


class TrainError(Exception):
    """A run can't be prepared or trained; the message says why."""


# Settings


@dataclass(frozen=True)
class Config:
    """A run's settings: the TOML file's [model], [data], [lora] and [training]."""

    model: str  # Hugging Face repository
    revision: str  # a commit of it
    tokenizer_sha256: str | None  # tokenizer.json's, checked if given
    examples: Path  # labels' train.jsonl
    index: Path  # labels' examples.csv
    subset: int | None  # examples per epoch; None: every one within max_tokens
    max_tokens: int  # longest example (prompt and target) trained on
    seed: int
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    epochs: int
    gradient_accumulation: int  # examples per optimizer step (batch size 1)
    learning_rate: float  # peak
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    save_every: int  # steps between checkpoints; the last step is saved too
    gradient_checkpointing: bool
    dtype: str  # of the base model's weights
    device: str


TABLES = {
    "model": ("name", "revision", "tokenizer_sha256"),
    "data": ("examples", "index", "subset", "max_tokens", "seed"),
    "lora": ("r", "alpha", "dropout", "target_modules"),
    "training": (
        "epochs",
        "gradient_accumulation",
        "learning_rate",
        "warmup_steps",
        "weight_decay",
        "max_grad_norm",
        "save_every",
        "gradient_checkpointing",
        "dtype",
        "device",
    ),
}
OPTIONAL = {("model", "tokenizer_sha256")}


def parse_config(tables: Mapping[str, Any]) -> Config:
    """A Config from its tables (the TOML file, or config.json's settings). Every key
    is required except model.tokenizer_sha256, and no other is allowed."""
    problems = [f"unknown table [{name}]" for name in tables if name not in TABLES]
    for name, keys in TABLES.items():
        table = tables.get(name)
        if not isinstance(table, Mapping):
            problems.append(f"no [{name}] table")
            continue
        problems += [f"unknown key {name}.{k}" for k in table if k not in keys]
        problems += [
            f"{name}.{k} not set"
            for k in keys
            if k not in table and (name, k) not in OPTIONAL
        ]
    if problems:
        raise TrainError("config: " + "; ".join(problems))

    def get(name: str, key: str, kind: type, minimum: float | None = None) -> Any:
        value = tables[name][key]
        if kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, kind) or isinstance(value, bool) != (kind is bool):
            problems.append(f"{name}.{key} must be a {kind.__name__}")
            return value
        if minimum is not None and value < minimum:
            problems.append(f"{name}.{key} must be {minimum} or more")
        return value

    model, data, lora = tables["model"], tables["data"], tables["lora"]
    subset = data["subset"]
    if subset != ALL:
        subset = get("data", "subset", int, 1)
    modules = lora["target_modules"]
    if not (
        isinstance(modules, list)
        and modules
        and all(isinstance(m, str) and m for m in modules)
    ):
        problems.append("lora.target_modules must be a list of module names")
    sha = model.get("tokenizer_sha256")
    if sha is not None and not isinstance(sha, str):
        problems.append("model.tokenizer_sha256 must be a str")
    config = Config(
        model=get("model", "name", str),
        revision=get("model", "revision", str),
        tokenizer_sha256=sha,
        examples=Path(get("data", "examples", str)),
        index=Path(get("data", "index", str)),
        subset=None if subset == ALL else subset,
        max_tokens=get("data", "max_tokens", int, 1),
        seed=get("data", "seed", int),
        r=get("lora", "r", int, 1),
        alpha=get("lora", "alpha", int, 1),
        dropout=get("lora", "dropout", float, 0.0),
        target_modules=tuple(modules) if isinstance(modules, list) else (),
        epochs=get("training", "epochs", int, 1),
        gradient_accumulation=get("training", "gradient_accumulation", int, 1),
        learning_rate=get("training", "learning_rate", float, 0.0),
        warmup_steps=get("training", "warmup_steps", int, 0),
        weight_decay=get("training", "weight_decay", float, 0.0),
        max_grad_norm=get("training", "max_grad_norm", float, 0.0),
        save_every=get("training", "save_every", int, 1),
        gradient_checkpointing=get("training", "gradient_checkpointing", bool),
        dtype=get("training", "dtype", str),
        device=get("training", "device", str),
    )
    if config.dropout >= 1:
        problems.append("lora.dropout must be less than 1")
    if config.dtype not in DTYPES:
        problems.append(f"training.dtype must be {' or '.join(DTYPES)}")
    if config.device not in DEVICES:
        problems.append(f"training.device must be {', '.join(DEVICES)}")
    if problems:
        raise TrainError("config: " + "; ".join(problems))
    return config


def config_tables(config: Config) -> dict[str, dict[str, Any]]:
    """config as the tables parse_config reads."""
    return {
        "model": {
            "name": config.model,
            "revision": config.revision,
            **(
                {"tokenizer_sha256": config.tokenizer_sha256}
                if config.tokenizer_sha256
                else {}
            ),
        },
        "data": {
            "examples": str(config.examples),
            "index": str(config.index),
            "subset": ALL if config.subset is None else config.subset,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
        },
        "lora": {
            "r": config.r,
            "alpha": config.alpha,
            "dropout": config.dropout,
            "target_modules": list(config.target_modules),
        },
        "training": {
            "epochs": config.epochs,
            "gradient_accumulation": config.gradient_accumulation,
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "max_grad_norm": config.max_grad_norm,
            "save_every": config.save_every,
            "gradient_checkpointing": config.gradient_checkpointing,
            "dtype": config.dtype,
            "device": config.device,
        },
    }


def load_config(path: Path) -> Config:
    if not path.is_file():
        raise TrainError(f"{path} not found")
    try:
        tables = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise TrainError(f"{path}: {exc}") from None
    return parse_config(tables)


def parse_subset(value: str) -> int | str:
    """--subset: a whole number of examples, 1 or more, or "all"."""
    return ALL if value == ALL else extract.positive_int(value)


# The examples


@dataclass(frozen=True)
class Example:
    accession: str
    ticker: str
    kind: str  # one of labels.KINDS
    prompt_tokens: int
    target_tokens: int

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.target_tokens


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_csv(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def load_messages(
    examples_path: Path, accessions: set[str] | None = None
) -> dict[str, list[dict[str, str]]]:
    """Release accession -> its system, user and assistant messages, for every
    example of train.jsonl (or only those of accessions), in file order."""
    if not examples_path.is_file():
        raise TrainError(f"{examples_path} not found")
    messages = {}
    with examples_path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            example = json.loads(line)
            accession = example["release_accession"]
            if accession in messages:
                raise TrainError(f"{examples_path}:{number}: {accession} again")
            roles = [m["role"] for m in example["messages"]]
            if roles != ["system", "user", "assistant"]:
                raise TrainError(
                    f"{examples_path}:{number}: messages {roles}, not system, user "
                    "and assistant"
                )
            if accessions is None or accession in accessions:
                messages[accession] = example["messages"]
    return messages


def check_examples(
    messages: Mapping[str, list[dict[str, str]]],
    index: Sequence[Mapping[str, str]],
    prompt: str,
    config: Config,
) -> None:
    """labels' examples.csv must list train.jsonl's releases in its order, each with a
    known kind, and every system message must be the prompt labels used."""
    if [row["release_accession"] for row in index] != list(messages):
        raise TrainError(
            f"{config.index} and {config.examples} list different releases; rebuild "
            "them with secbench.labels"
        )
    kinds = {row["kind"] for row in index} - set(labels.KINDS)
    if kinds:
        raise TrainError(f"{config.index}: unknown kinds {sorted(kinds)}")
    others = [a for a, m in messages.items() if m[0]["content"] != prompt]
    if others:
        raise TrainError(
            f"{len(others)} examples' system message isn't prompts/{labels.PROMPT}.txt "
            f"(first {others[0]})"
        )


def count_tokens(
    tokenizer: Any,
    messages: Mapping[str, list[dict[str, str]]],
    index: Sequence[Mapping[str, str]],
) -> list[Example]:
    """Every example with its prompt and target token counts (tokenizer: a tokenizers
    Tokenizer), in train.jsonl's order."""
    contents = [[m["content"] for m in messages[r["release_accession"]]] for r in index]
    prompts = [chat.prompt_text(system, user) for system, user, _ in contents]
    targets = [chat.target_text(reply) for _, _, reply in contents]

    def count(texts: list[str]) -> list[int]:
        return [
            len(e.ids) for e in tokenizer.encode_batch(texts, add_special_tokens=False)
        ]

    return [
        Example(row["release_accession"], row["ticker"], row["kind"], p, t)
        for row, p, t in zip(index, count(prompts), count(targets), strict=True)
    ]


def allocate(available: Mapping[str, int], total: int) -> dict[str, int]:
    """total split across the kinds in proportion to available: each kind's whole
    share, then one more to the largest remainders (ties to the kind listed first)."""
    whole = sum(available.values())
    if total > whole:
        raise TrainError(f"{total} examples asked for; {whole} available")
    shares = {kind: Fraction(total * n, whole) for kind, n in available.items()}
    counts = {kind: math.floor(share) for kind, share in shares.items()}
    by_remainder = sorted(available, key=lambda k: counts[k] - shares[k])
    for kind in by_remainder[: total - sum(counts.values())]:
        counts[kind] += 1
    return counts


def draw(
    examples: Sequence[Example], subset: int | None, seed: int, epochs: int
) -> list[list[Example]]:
    """The examples of each epoch, in training order: a subset stratified by kind (or
    every example), shuffled once per epoch; all from one random.Random(seed)."""
    rng = random.Random(seed)
    if subset is None:
        chosen = list(examples)
    else:
        by_kind = {k: [e for e in examples if e.kind == k] for k in labels.KINDS}
        counts = allocate({k: len(v) for k, v in by_kind.items()}, subset)
        chosen = [e for k in labels.KINDS for e in rng.sample(by_kind[k], counts[k])]
    orders = []
    for _ in range(epochs):
        order = list(chosen)
        rng.shuffle(order)
        orders.append(order)
    return orders


@dataclass(frozen=True)
class Schedule:
    per_epoch: int  # examples in an epoch
    positions: int  # examples seen in the run (every epoch)
    accumulation: int
    steps: int  # optimizer steps
    checkpoints: tuple[int, ...]  # steps after which a checkpoint is saved
    warmup: int

    def window(self, step: int) -> range:
        """Positions of optimizer step `step` (from 0); the last may be short."""
        start = step * self.accumulation
        return range(start, min(start + self.accumulation, self.positions))


def make_schedule(per_epoch: int, config: Config) -> Schedule:
    positions = per_epoch * config.epochs
    steps = math.ceil(positions / config.gradient_accumulation)
    saves = {*range(config.save_every, steps + 1, config.save_every), steps}
    return Schedule(
        per_epoch,
        positions,
        config.gradient_accumulation,
        steps,
        tuple(sorted(saves)),
        min(config.warmup_steps, steps),
    )


def lr_factor(step: int, *, steps: int, warmup: int) -> float:
    """The peak learning rate's multiplier at optimizer step `step` (from 0): k/warmup
    at the k-th step of the warmup (1/warmup, ..., 1), then a cosine from 1 at the end
    of the warmup down to 0 one step past the last."""
    k = step + 1
    if k <= warmup:
        return k / warmup
    return 0.5 * (1 + math.cos(math.pi * (k - warmup) / (steps - warmup + 1)))


# Prepare


def format_count(n: int) -> str:
    return f"{n:,}"


def prepare(
    config: Config,
    config_path: Path,
    run_dir: Path,
    tokenizer: Any,
    tokenizer_path: Path,
    commit: str,
    overrides: Mapping[str, Any],
    prompts_dir: Path = extract.PROMPTS_DIR,
) -> dict[str, Any]:
    """Write run_dir's config.json, examples.csv and dropped.csv; config.json's
    contents are returned."""
    if run_dir.exists():
        raise TrainError(f"{run_dir} already exists; choose another --run-name")
    if not config.index.is_file():
        raise TrainError(f"{config.index} not found")
    if (
        config.tokenizer_sha256
        and file_sha256(tokenizer_path) != config.tokenizer_sha256
    ):
        raise TrainError(
            f"{tokenizer_path} doesn't have the config's sha256 {config.tokenizer_sha256}"
        )
    prompt = extract.load_prompt(labels.PROMPT, prompts_dir)
    messages = load_messages(config.examples)
    index = read_csv(config.index)
    check_examples(messages, index, prompt.text, config)
    examples = count_tokens(tokenizer, messages, index)
    within = [e for e in examples if e.tokens <= config.max_tokens]
    dropped = sorted(
        (e for e in examples if e.tokens > config.max_tokens), key=lambda e: -e.tokens
    )
    if config.subset is not None and config.subset > len(within):
        raise TrainError(
            f"subset {config.subset} is more than the {len(within)} examples within "
            f"{config.max_tokens} tokens; use --subset all"
        )
    orders = draw(within, config.subset, config.seed, config.epochs)
    schedule = make_schedule(len(orders[0]), config)
    seen = [e for order in orders for e in order]
    manifest_path = config.index.parent / LABELS_MANIFEST
    info = {
        "run_name": run_dir.name,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "config_file": str(config_path),
        "overrides": dict(overrides),
        "settings": config_tables(config),
        "data": {
            "examples": str(config.examples),
            "examples_sha256": file_sha256(config.examples),
            "index": str(config.index),
            "index_sha256": file_sha256(config.index),
            "labels_manifest": json.loads(manifest_path.read_text())
            if manifest_path.is_file()
            else None,
            "prompt": {"version": labels.PROMPT, "sha256": prompt.sha256},
        },
        "tokenizer": {"file": TOKENIZER, "sha256": file_sha256(tokenizer_path)},
        "selection": {
            "examples": len(examples),
            "over_max_tokens": len(dropped),
            "within_max_tokens": dict(Counter(e.kind for e in within)),
            "selected": len(orders[0]),
            "selected_by_kind": dict(Counter(e.kind for e in orders[0])),
        },
        "schedule": {
            "epochs": config.epochs,
            "examples_per_epoch": schedule.per_epoch,
            "positions": schedule.positions,
            "examples_per_step": schedule.accumulation,
            "steps": schedule.steps,
            "warmup_steps": schedule.warmup,
            "checkpoints": list(schedule.checkpoints),
        },
        "tokens": {
            "total": sum(e.tokens for e in seen),
            "target": sum(e.target_tokens for e in seen),
            "mean": round(sum(e.tokens for e in seen) / len(seen)),
            "max": max(e.tokens for e in seen),
        },
    }
    run_dir.mkdir(parents=True)
    write_text(
        run_dir / EXAMPLES,
        rows_csv(
            (
                {
                    "position": position,
                    "epoch": epoch + 1,
                    "release_accession": e.accession,
                    "ticker": e.ticker,
                    "kind": e.kind,
                    "prompt_tokens": e.prompt_tokens,
                    "target_tokens": e.target_tokens,
                    "tokens": e.tokens,
                }
                for position, (epoch, e) in enumerate(
                    (epoch, e) for epoch, order in enumerate(orders) for e in order
                )
            ),
            EXAMPLE_COLUMNS,
        ),
    )
    write_text(
        run_dir / DROPPED,
        rows_csv(
            (
                {
                    "release_accession": e.accession,
                    "ticker": e.ticker,
                    "kind": e.kind,
                    "prompt_tokens": e.prompt_tokens,
                    "target_tokens": e.target_tokens,
                    "tokens": e.tokens,
                }
                for e in dropped
            ),
            DROPPED_COLUMNS,
        ),
    )
    extract.write_json(run_dir / CONFIG, info)
    return info


def format_prepared(run_dir: Path, info: Mapping[str, Any]) -> str:
    selection, schedule, tokens = info["selection"], info["schedule"], info["tokens"]
    settings = info["settings"]
    within = sum(selection["within_max_tokens"].values())
    max_tokens = settings["data"]["max_tokens"]
    kinds = ", ".join(
        f"{kind} {selection['selected_by_kind'].get(kind, 0):,} of "
        f"{selection['within_max_tokens'].get(kind, 0):,}"
        for kind in labels.KINDS
    )
    epochs = schedule["epochs"]
    saves = ", ".join(str(s) for s in schedule["checkpoints"])
    lines = [
        (
            f"{run_dir}: {selection['selected']:,} of {within:,} examples within "
            f"{max_tokens:,} tokens ({selection['over_max_tokens']:,} of "
            f"{selection['examples']:,} longer, listed in {DROPPED})"
        ),
        f"  {kinds}",
        (
            f"  {epochs} epoch{'s' if epochs > 1 else ''}, {schedule['steps']} steps "
            f"of {schedule['examples_per_step']} examples, "
            f"{schedule['warmup_steps']} warmup; checkpoints at steps {saves}"
        ),
        (
            f"  tokens: {tokens['total']:,} in all ({tokens['target']:,} trained "
            f"on), mean {tokens['mean']:,}, longest {tokens['max']:,}"
        ),
        f"  seed {settings['data']['seed']}; training order in {run_dir / EXAMPLES}",
    ]
    return "\n".join(lines)


# The model's files


def model_file(config: Config, filename: str, hf_cache: Path) -> Path:
    """One file of the model at its pinned revision, downloaded once into hf_cache."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            config.model, filename, revision=config.revision, cache_dir=hf_cache
        )
    )


def download(config: Config, hf_cache: Path) -> Path:
    """The model's snapshot at its pinned revision, downloaded once into hf_cache."""
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(config.model, revision=config.revision, cache_dir=hf_cache)
    )
    check_tokenizer(config, path / TOKENIZER)
    return path


def snapshot_path(config: Config, hf_cache: Path) -> Path:
    """The downloaded snapshot (no download)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = snapshot_download(
            config.model,
            revision=config.revision,
            cache_dir=hf_cache,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        raise TrainError(
            f"{config.model} at {config.revision} is not in {hf_cache}; run "
            "secbench.train download first"
        ) from None
    return Path(path)


def check_tokenizer(config: Config, path: Path) -> None:
    if config.tokenizer_sha256 and file_sha256(path) != config.tokenizer_sha256:
        raise TrainError(f"{path} doesn't have the config's sha256")


# Checkpoints


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: Path


def checkpoint_name(step: int) -> str:
    return f"step-{step:05d}"


def checkpoints(run_dir: Path) -> list[Checkpoint]:
    """The run's complete checkpoints, by step (a .part folder is not one)."""
    found = []
    for path in sorted((run_dir / CHECKPOINTS).glob("step-*")):
        name = path.name
        if (
            path.is_dir()
            and name[5:].isdigit()
            and (path / STATE).is_file()
            and name == checkpoint_name(int(name[5:]))
        ):
            found.append(Checkpoint(int(name[5:]), path))
    return sorted(found, key=lambda c: c.step)


def read_run(run_dir: Path) -> tuple[dict[str, Any], Config, Schedule]:
    """config.json's contents, its Config and its Schedule."""
    path = run_dir / CONFIG
    if not path.is_file():
        raise TrainError(f"{path} not found; run secbench.train prepare first")
    info = json.loads(path.read_text(encoding="utf-8"))
    config = parse_config(info["settings"])
    schedule = make_schedule(info["schedule"]["examples_per_epoch"], config)
    if list(schedule.checkpoints) != info["schedule"]["checkpoints"]:
        raise TrainError(f"{path}: its schedule isn't its settings'")
    return info, config, schedule


def read_order(run_dir: Path, schedule: Schedule) -> list[Example]:
    """examples.csv: the example at each position, in training order."""
    rows = read_csv(run_dir / EXAMPLES)
    if [int(r["position"]) for r in rows] != list(range(schedule.positions)):
        raise TrainError(f"{run_dir / EXAMPLES} doesn't have {schedule.positions} rows")
    return [
        Example(
            r["release_accession"],
            r["ticker"],
            r["kind"],
            int(r["prompt_tokens"]),
            int(r["target_tokens"]),
        )
        for r in rows
    ]


# Training (torch, transformers and peft: the train dependency group)


def encode(tokenizer: Any, text: str) -> list[int]:
    """text's token ids as vLLM computes them: no special tokens added (a transformers
    tokenizer)."""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def first_difference(a: str, b: str) -> int:
    return next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
        min(len(a), len(b)),
    )


def layout_problems(
    tokenizer: Any, example: Example, messages: Sequence[Mapping[str, str]]
) -> list[str]:
    """How the example's training text differs from the model's chat template with
    thinking off, or its token counts from prepare's."""
    system, user, reply = (m["content"] for m in messages)
    prompt, target = chat.prompt_text(system, user), chat.target_text(reply)
    problems = []
    generation = tokenizer.apply_chat_template(
        list(messages[:2]),
        tokenize=False,
        add_generation_prompt=True,
        **chat.THINKING_OFF,
    )
    if generation != prompt:
        problems.append(
            f"{example.accession}: the chat template's generation prompt differs from "
            f"the training prompt at character {first_difference(generation, prompt)}"
        )
    whole = tokenizer.apply_chat_template(
        list(messages), tokenize=False, **chat.THINKING_OFF
    )
    if whole != prompt + target + "\n":
        problems.append(
            f"{example.accession}: the chat template's conversation differs from "
            "prompt + target at character "
            f"{first_difference(whole, prompt + target + chr(10))}"
        )
    counts = (len(encode(tokenizer, prompt)), len(encode(tokenizer, target)))
    if counts != (example.prompt_tokens, example.target_tokens):
        problems.append(
            f"{example.accession}: {counts[0]} prompt and {counts[1]} target tokens; "
            f"prepare counted {example.prompt_tokens} and {example.target_tokens}"
        )
    return problems


def reply_loss_sum(
    model: Any, prompt_ids: Sequence[int], target_ids: Sequence[int], device: Any
) -> Any:
    """The cross-entropy of target_ids after prompt_ids, summed over the target
    tokens. Logits are computed only at the len(target) positions that predict them
    (and the last one, dropped): the target ends the sequence."""
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([[*prompt_ids, *target_ids]], device=device)
    output = model(input_ids=ids, logits_to_keep=len(target_ids) + 1, use_cache=False)
    logits = output.logits[0, :-1].float()
    targets = torch.tensor(list(target_ids), device=device)
    return F.cross_entropy(logits, targets, reduction="sum")


def pick_device(name: str) -> str:
    import torch

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if name == "auto":
        return next(d for d in ("cuda", "mps", "cpu") if available[d])
    if not available[name]:
        raise TrainError(f"training.device is {name}, which torch can't use here")
    return name


def rng_states(device: str) -> dict[str, Any]:
    import torch

    states: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if device == "cuda":
        states["cuda"] = torch.cuda.get_rng_state_all()
    elif device == "mps":
        states["mps"] = torch.mps.get_rng_state()
    return states


def restore_rng(states: Mapping[str, Any]) -> None:
    import torch

    random.setstate(states["python"])
    torch.set_rng_state(states["torch"])
    if "cuda" in states:
        torch.cuda.set_rng_state_all(states["cuda"])
    if "mps" in states:
        torch.mps.set_rng_state(states["mps"])


def save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    run_dir: Path,
    step: int,
    device: str,
    state: Mapping[str, Any],
) -> Path:
    """checkpoints/step-NNNNN/, written as a .part folder and then renamed."""
    import torch

    final = run_dir / CHECKPOINTS / checkpoint_name(step)
    partial = final.with_name(final.name + PARTIAL)
    for stale in (partial, final):
        if stale.exists():
            shutil.rmtree(stale)
    model.save_pretrained(partial)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rng_states(device),
        },
        partial / TRAINING_STATE,
    )
    extract.write_json(partial / STATE, dict(state))
    partial.rename(final)
    return final


def resume(
    model: Any, optimizer: Any, scheduler: Any, checkpoint: Checkpoint, device: str
) -> None:
    """Restore the adapter, optimizer, scheduler and RNG states saved at checkpoint."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    weights = load_file(checkpoint.path / ADAPTER_WEIGHTS, device=device)
    result = set_peft_model_state_dict(model, weights)
    if result.unexpected_keys:
        raise TrainError(
            f"{checkpoint.path}: unexpected adapter weights {result.unexpected_keys[:3]}"
        )
    saved = torch.load(checkpoint.path / TRAINING_STATE, map_location="cpu")
    if saved["step"] != checkpoint.step:
        raise TrainError(f"{checkpoint.path}: state of step {saved['step']}")
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    restore_rng(saved["rng"])


def read_log(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / TRAIN_LOG
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def environment(device: str, commit: str, start: int, snapshot: Path) -> dict[str, Any]:
    import peft
    import torch
    import transformers

    gpu = torch.cuda.get_device_name() if device == "cuda" else None
    return {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "start_step": start,
        "device": device,
        "gpu": gpu,
        "cuda": torch.version.cuda if device == "cuda" else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "snapshot": str(snapshot),
    }


def peak_memory_gb(device: str) -> float | None:
    import torch

    if device == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    if device == "mps":
        return round(torch.mps.driver_allocated_memory() / 2**30, 2)
    return None


def synchronize(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def train(
    run_dir: Path,
    snapshot: Path,
    commit: str,
    stop_after: int | None = None,
    out: Callable[[str], None] = print,
) -> None:
    """Train the run prepared in run_dir from its last checkpoint (or the start), up
    to its last step or stop_after."""
    info, config, schedule = read_run(run_dir)
    if info["commit"] != commit:
        raise TrainError(
            f"{run_dir} was prepared at commit {info['commit']}; HEAD is {commit}"
        )
    if stop_after is not None and stop_after not in schedule.checkpoints:
        raise TrainError(
            f"--stop-after must be a checkpoint step: {list(schedule.checkpoints)}"
        )
    if file_sha256(config.examples) != info["data"]["examples_sha256"]:
        raise TrainError(f"{config.examples} changed since {run_dir} was prepared")
    if file_sha256(snapshot / TOKENIZER) != info["tokenizer"]["sha256"]:
        raise TrainError(
            f"{snapshot / TOKENIZER} isn't the tokenizer {run_dir} was prepared with"
        )
    done = checkpoints(run_dir)
    last = done[-1] if done else None
    if last is not None and last.step >= schedule.steps:
        out(f"{run_dir}: finished ({last.path.name}); nothing to do")
        return
    order = read_order(run_dir, schedule)
    by_accession = {e.accession: e for e in order}
    messages = load_messages(config.examples, set(by_accession))

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    out(f"checking {len(by_accession):,} examples against the chat template")
    problems = [
        p
        for accession, example in by_accession.items()
        for p in layout_problems(tokenizer, example, messages[accession])
    ]
    if problems:
        raise TrainError(f"{len(problems)} layout problems: " + "; ".join(problems[:5]))
    end = tokenizer.convert_tokens_to_ids(chat.IM_END)

    device = pick_device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    dtype = getattr(torch, config.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=dtype, attn_implementation="sdpa", device_map={"": device}
    )
    eos = model.generation_config.eos_token_id
    if end not in (eos if isinstance(eos, list) else [eos]):
        raise TrainError(
            f"{chat.IM_END} ({end}) isn't among the model's end-of-sequence tokens "
            f"{eos}, so generation wouldn't stop where the training target ends"
        )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.r,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        functools.partial(lr_factor, steps=schedule.steps, warmup=schedule.warmup),
    )
    start = 0
    if last is not None:
        resume(model, optimizer, scheduler, last, device)
        start = last.step
        kept = [row for row in read_log(run_dir) if row["step"] <= start]
        write_text(run_dir / TRAIN_LOG, "".join(json.dumps(r) + "\n" for r in kept))
    trainable = sum(p.numel() for p in params)
    append_jsonl(run_dir / SESSIONS, environment(device, commit, start, snapshot))
    out(
        f"{run_dir}: {config.model} on {device} ({config.dtype}), LoRA r {config.r} "
        f"alpha {config.alpha} on {', '.join(config.target_modules)}: "
        f"{trainable:,} trainable parameters; "
        + (f"resuming after {last.path.name}" if last else "starting at step 1")
        + f", {schedule.steps} steps"
    )

    model.train()
    session_start, session_tokens = time.perf_counter(), 0
    since_checkpoint: list[float] = []
    for step in range(start, schedule.steps):
        window = schedule.window(step)
        window_targets = sum(order[p].target_tokens for p in window)
        began = time.perf_counter()
        loss_sum, tokens = 0.0, 0
        for position in window:
            example = order[position]
            system, user, reply = (m["content"] for m in messages[example.accession])
            prompt_ids = encode(tokenizer, chat.prompt_text(system, user))
            target_ids = encode(tokenizer, chat.target_text(reply))
            if target_ids[-1] != end:
                raise TrainError(
                    f"{example.accession}: target doesn't end {chat.IM_END}"
                )
            loss = reply_loss_sum(model, prompt_ids, target_ids, device)
            (loss / window_targets).backward()
            loss_sum += loss.item()
            tokens += len(prompt_ids) + len(target_ids)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
        learning_rate = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        seconds = time.perf_counter() - began
        session_tokens += tokens
        row = {
            "step": step + 1,
            "examples_seen": window.stop,
            "epoch": round(window.stop / schedule.per_epoch, 4),
            "loss": loss_sum / window_targets,
            "learning_rate": learning_rate,
            "grad_norm": float(grad_norm),
            "tokens": tokens,
            "target_tokens": window_targets,
            "seconds": round(seconds, 3),
            "tokens_per_second": round(tokens / seconds, 1),
            "peak_memory_gb": peak_memory_gb(device),
        }
        append_jsonl(run_dir / TRAIN_LOG, row)
        since_checkpoint.append(row["loss"])
        rate = session_tokens / (time.perf_counter() - session_start)
        left = sum(order[p].tokens for p in range(window.stop, schedule.positions))
        out(
            f"step {step + 1}/{schedule.steps}: loss {row['loss']:.4f}, lr "
            f"{learning_rate:.2e}, grad norm {row['grad_norm']:.3f}, {tokens:,} tokens "
            f"in {seconds:.1f} s ({rate:,.0f} tokens/s this session; about "
            f"{duration(left / rate)} left)"
        )
        if step + 1 in schedule.checkpoints:
            path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                run_dir,
                step + 1,
                device,
                {
                    "step": step + 1,
                    "examples_seen": window.stop,
                    "epoch": row["epoch"],
                    "learning_rate": learning_rate,
                    "mean_loss_since_last_checkpoint": sum(since_checkpoint)
                    / len(since_checkpoint),
                    "saved": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            )
            since_checkpoint = []
            out(f"saved {path}")
            if step + 1 == stop_after:
                out(f"stopped after step {stop_after} (--stop-after)")
                return
    out(f"{run_dir}: finished, {schedule.steps} steps")


# Command line


def checked_commit(commit: str | None) -> str:
    if commit is not None:
        return commit
    commit, dirty = current_commit()
    if dirty:
        raise TrainError(
            "tracked files have uncommitted changes; commit them first (config.json "
            "records the commit)"
        )
    return commit


def main(
    argv: list[str] | None = None,
    *,
    commit: str | None = None,
    fetch: Callable[[Config, str, Path], Path] = model_file,
    tokenizer_from_file: Callable[[Path], Any] | None = None,
    snapshot: Path | None = None,
    out: Callable[[str], None] = print,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.train",
        description="Stage 2 LoRA fine-tuning: download the model, prepare a run, "
        "train it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("download", "download the model at the config's revision"),
        ("prepare", "draw the examples and write training/<run>/"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
        if name == "prepare":
            command.add_argument(
                "--run-name",
                help="folder under --training-dir (default: the config's file name)",
            )
            command.add_argument(
                "--subset",
                type=parse_subset,
                help='examples per epoch, or "all" (default: the config\'s)',
            )
            command.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
            command.add_argument(
                "--prompts-dir", type=Path, default=extract.PROMPTS_DIR
            )
    command = commands.add_parser(
        "train", help="train a prepared run, from its last checkpoint"
    )
    command.add_argument("run_dir", type=Path)
    command.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    command.add_argument(
        "--stop-after", type=extract.positive_int, help="stop after this checkpoint"
    )
    args = parser.parse_args(argv)

    try:
        if args.command == "download":
            config = load_config(args.config)
            out(
                f"{config.model} at {config.revision}: {download(config, args.hf_cache)}"
            )
        elif args.command == "prepare":
            commit = checked_commit(commit)
            config = load_config(args.config)
            overrides = {}
            if args.subset is not None:
                overrides["subset"] = args.subset
                tables = config_tables(config)
                tables["data"]["subset"] = args.subset
                config = parse_config(tables)
            run_name = args.run_name or args.config.stem
            if not extract.NAME.fullmatch(run_name):
                raise TrainError(f"bad run name {run_name!r}")
            tokenizer_path = fetch(config, TOKENIZER, args.hf_cache)
            if tokenizer_from_file is None:
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(tokenizer_path))
            else:
                tokenizer = tokenizer_from_file(tokenizer_path)
            run_dir = args.training_dir / run_name
            info = prepare(
                config,
                args.config,
                run_dir,
                tokenizer,
                tokenizer_path,
                commit,
                overrides,
                args.prompts_dir,
            )
            out(format_prepared(run_dir, info))
        else:
            commit = checked_commit(commit)
            _, config, _ = read_run(args.run_dir)
            train(
                args.run_dir,
                snapshot or snapshot_path(config, args.hf_cache),
                commit,
                args.stop_after,
                out,
            )
    except (TrainError, extract.RunError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
