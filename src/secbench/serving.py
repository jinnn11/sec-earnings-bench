"""The stage 3 serving benchmark's config, variants and file layout (PLAN.md Decisions
2026-11-09), shared by secbench.quantize, secbench.servebench, secbench.bundle and
secbench.servereport. Library only.

The config (configs/serving/<run>.toml) names the checkpoint under test (as the test
config does, with its adapter's sha256), the variants to serve and in what order, the
server's flags common to every variant, the request settings (which must be stage 2's:
extract_v2, temperature 0, thinking off, max_tokens 1024), the load test (the split
whose prompts it sends, the concurrency levels, the prompts per level, the warm-up
requests and the seed of the draw), the accuracy runs (splits, concurrency), the AWQ
calibration (examples, token cap, seed, scheme) and the stage 2 runs the report
compares with. Every key is required unless OPTIONAL lists it, and no other is
allowed.

Variants: adapter (the base model with the checkpoint's LoRA adapter, as stage 2
served it), merged (the adapter merged into the base weights, bf16), fp8 (merged,
quantized to FP8 by vLLM as it loads) and int4-awq (merged, quantized offline to 4-bit
weights with llm-compressor's AWQ). fp8 and int4-awq run only on vLLM.

Layout, for a run R and a variant V:
- training/R/serving/: servebench.json (the session's record), plan.json,
  setup.log (scripts/vast_setup.sh's), models/merged and models/int4-awq (the
  weights and each one's manifest.json; models/<model>.log is its build's output),
  and V/: server.log, startup.json, variant.json, <split>.env and load/
  (requests.jsonl, summary.json);
- runs/R--serving--V--<split>: the accuracy runs (secbench.extract, not graded);
- data/serving/R/: the calibration examples drawn from the training labels
  (calibration.jsonl, calibration.csv, calibration.json).
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from secbench import chat, labels, score, splits, train

ADAPTER, MERGED, FP8, INT4_AWQ = "adapter", "merged", "fp8", "int4-awq"
VARIANTS = (ADAPTER, MERGED, FP8, INT4_AWQ)
# The model folder (under models/) each variant serves; None: the base snapshot and
# the adapter.
MODEL_OF = {ADAPTER: None, MERGED: MERGED, FP8: MERGED, INT4_AWQ: INT4_AWQ}
VLLM_ONLY = (FP8, INT4_AWQ)  # quantized by or for vLLM: not on the laptop stand-in
SERVING = "serving"
MODELS = "models"
DATA_DIR = Path("data/serving")
CALIBRATION = "calibration.jsonl"
CALIBRATION_INDEX = "calibration.csv"
CALIBRATION_MANIFEST = "calibration.json"
MANIFEST = "manifest.json"  # in each model folder
RECORD = "servebench.json"
PLAN = "plan.json"
SETUP_LOG = "setup.log"
DUO_SCALING = (True, False, "both")  # llm-compressor's AWQModifier.duo_scaling

# key -> type, per table; list[int] and list[str] as tuples of int and str.
TABLES: dict[str, dict[str, Any]] = {
    "model": {
        "run": str,
        "base": str,
        "revision": str,
        "checkpoint": str,
        "adapter_sha256": str,
    },
    "server": {
        "variants": (str,),
        "max_model_len": int,
        "prefix_caching": bool,
    },
    "request": {
        "prompt": str,
        "temperature": float,
        "enable_thinking": bool,
        "max_tokens": int,
    },
    "load": {
        "split": str,
        "levels": (int,),
        "prompts": int,
        "warmup": int,
        "seed": int,
        "pool": int,
    },
    "accuracy": {
        "splits": (str,),
        "concurrency": int,
        "limit": dict,
    },
    "calibration": {
        "examples": int,
        "max_tokens": int,
        "seed": int,
        "scheme": str,
        "duo_scaling": object,
        "ignore": (str,),
    },
    "report": {
        "validation_reference": str,
        "time_test_reference": str,
    },
}
# Keys that may be left out: the load test's pool (the split's first N releases; all
# by default), the accuracy runs' limits (split -> first N releases) and the report's
# stage 2 references.
OPTIONAL = {
    ("load", "pool"),
    ("accuracy", "limit"),
    ("report", "validation_reference"),
    ("report", "time_test_reference"),
}


class ServingError(Exception):
    """The serving benchmark can't go on; the message says why."""


@dataclass(frozen=True)
class ServingConfig:
    path: Path
    sha256: str  # the file's
    run: str
    base: str
    revision: str
    checkpoint: str
    adapter_sha256: str
    variants: tuple[str, ...]
    max_model_len: int
    prefix_caching: bool
    prompt: str
    temperature: float
    enable_thinking: bool
    max_tokens: int
    load_split: str
    levels: tuple[int, ...]
    prompts: int
    warmup: int
    seed: int
    pool: int | None
    accuracy_splits: tuple[str, ...]
    concurrency: int
    limits: dict[str, int]
    calibration_examples: int
    calibration_max_tokens: int
    calibration_seed: int
    scheme: str
    duo_scaling: bool | str
    ignore: tuple[str, ...]
    validation_reference: Path | None
    time_test_reference: Path | None

    @property
    def run_dir(self) -> Path:
        return train.TRAINING_DIR / self.run

    @property
    def serving_dir(self) -> Path:
        return self.run_dir / SERVING

    @property
    def models_dir(self) -> Path:
        return self.serving_dir / MODELS

    @property
    def calibration_dir(self) -> Path:
        return DATA_DIR / self.run

    @property
    def splits(self) -> tuple[str, ...]:
        """Every split the benchmark reads: the accuracy runs' and the load test's."""
        return tuple(dict.fromkeys((*self.accuracy_splits, self.load_split)))

    def served(self, variant: str) -> str:
        """The name a variant is served by: the adapter as stage 2 named it, the
        others after it."""
        name = f"{self.run}--{self.checkpoint}"
        return name if variant == ADAPTER else f"{name}--{variant}"

    def run_name(self, variant: str, split: str) -> str:
        return f"{self.run}--{SERVING}--{variant}--{split}"


def check_type(value: Any, kind: Any) -> Any:
    """value as kind (an int is a float where a float is wanted; a list of kind[0]
    is a tuple), or None if it isn't one."""
    if isinstance(kind, tuple):
        if not isinstance(value, list):
            return None
        items = [check_type(v, kind[0]) for v in value]
        return None if any(i is None for i in items) else tuple(items)
    if kind is object:
        return value
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, kind) or isinstance(value, bool) != (kind is bool):
        return None
    return value


NAMES = {
    int: "a whole number",
    float: "a number",
    str: "a string",
    bool: "true or false",
    dict: "a table",
}


def type_name(kind: Any) -> str:
    if isinstance(kind, tuple):
        return {int: "a list of whole numbers", str: "a list of strings"}[kind[0]]
    return NAMES.get(kind, "a value")


def parse_tables(tables: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Each table's values, by key; every problem with the file at once."""
    problems = [f"unknown table [{name}]" for name in tables if name not in TABLES]
    values: dict[str, Any] = {}
    for name, keys in TABLES.items():
        table = tables.get(name, {})
        if not isinstance(table, dict):
            problems.append(f"[{name}] must be a table")
            continue
        problems += [f"unknown key {name}.{k}" for k in table if k not in keys]
        for key, kind in keys.items():
            if key not in table:
                if (name, key) not in OPTIONAL:
                    problems.append(f"{name}.{key} not set")
                values[f"{name}.{key}"] = None
                continue
            value = check_type(table[key], kind)
            if value is None:
                problems.append(f"{name}.{key} must be {type_name(kind)}")
            values[f"{name}.{key}"] = value
    if problems:
        raise ServingError(f"{path}: " + "; ".join(problems))
    return values


def value_problems(v: Mapping[str, Any]) -> list[str]:
    """Values the benchmark can't run with, or that aren't stage 2's settings."""
    problems = []
    variants = v["server.variants"]
    unknown = [x for x in variants if x not in VARIANTS]
    if unknown or not variants or len(set(variants)) != len(variants):
        problems.append(f"server.variants must be distinct, from {', '.join(VARIANTS)}")
    expected = {
        "server.max_model_len": chat.MAX_MODEL_LEN,
        "request.prompt": labels.PROMPT,
        "request.temperature": 0.0,
        "request.enable_thinking": False,
        "request.max_tokens": chat.MAX_TOKENS,
    }
    problems += [
        f"{key} is {v[key]!r}, not stage 2's {value!r}"
        for key, value in expected.items()
        if v[key] != value
    ]
    levels = v["load.levels"]
    if not levels or any(n < 1 for n in levels) or list(levels) != sorted(set(levels)):
        problems.append("load.levels must be rising whole numbers, 1 or more")
    for key, least in (
        ("load.prompts", 1),
        ("load.warmup", 0),
        ("accuracy.concurrency", 1),
        ("calibration.examples", 1),
        ("calibration.max_tokens", 1),
    ):
        if v[key] < least:
            problems.append(f"{key} must be {least} or more")
    if (
        v["load.pool"] is not None
        and v["load.pool"] < v["load.prompts"] + v["load.warmup"]
    ):
        problems.append("load.pool must hold load.prompts + load.warmup releases")
    if v["load.split"] not in splits.SPLITS:
        problems.append(f"load.split must be one of {', '.join(splits.SPLITS)}")
    accuracy = v["accuracy.splits"]
    if any(s not in splits.SPLITS for s in accuracy) or len(set(accuracy)) != len(
        accuracy
    ):
        problems.append(
            f"accuracy.splits must be distinct, from {', '.join(splits.SPLITS)}"
        )
    for split, n in (v["accuracy.limit"] or {}).items():
        if split not in accuracy:
            problems.append(f"accuracy.limit.{split}: not one of accuracy.splits")
        elif check_type(n, int) is None or n < 1:
            problems.append(f"accuracy.limit.{split} must be a whole number, 1 or more")
    if v["calibration.duo_scaling"] not in DUO_SCALING:
        problems.append('calibration.duo_scaling must be true, false or "both"')
    if not v["calibration.scheme"]:
        problems.append("calibration.scheme must be set")
    return problems


def load_config(path: Path) -> ServingConfig:
    if not path.is_file():
        raise ServingError(f"{path} not found")
    raw = path.read_bytes()
    try:
        tables = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ServingError(f"{path}: {exc}") from None
    v = parse_tables(tables, path)
    problems = value_problems(v)
    if problems:
        raise ServingError(f"{path}: " + "; ".join(problems))

    def optional_path(key: str) -> Path | None:
        return None if v[key] is None else Path(v[key])

    return ServingConfig(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        run=v["model.run"],
        base=v["model.base"],
        revision=v["model.revision"],
        checkpoint=v["model.checkpoint"],
        adapter_sha256=v["model.adapter_sha256"],
        variants=v["server.variants"],
        max_model_len=v["server.max_model_len"],
        prefix_caching=v["server.prefix_caching"],
        prompt=v["request.prompt"],
        temperature=v["request.temperature"],
        enable_thinking=v["request.enable_thinking"],
        max_tokens=v["request.max_tokens"],
        load_split=v["load.split"],
        levels=v["load.levels"],
        prompts=v["load.prompts"],
        warmup=v["load.warmup"],
        seed=v["load.seed"],
        pool=v["load.pool"],
        accuracy_splits=v["accuracy.splits"],
        concurrency=v["accuracy.concurrency"],
        limits=dict(v["accuracy.limit"] or {}),
        calibration_examples=v["calibration.examples"],
        calibration_max_tokens=v["calibration.max_tokens"],
        calibration_seed=v["calibration.seed"],
        scheme=v["calibration.scheme"],
        duo_scaling=v["calibration.duo_scaling"],
        ignore=v["calibration.ignore"],
        validation_reference=optional_path("report.validation_reference"),
        time_test_reference=optional_path("report.time_test_reference"),
    )


def adapter_model(config: ServingConfig) -> score.Model:
    """The config's checkpoint as a score.Model (served under stage 2's name), after
    checking its adapter's sha256 and r against the training run."""
    run_dir = config.run_dir
    _, run_config, _ = train.read_run(run_dir)
    if (run_config.model, run_config.revision) != (config.base, config.revision):
        raise ServingError(
            f"{config.path}: model {config.base} at {config.revision}, but {run_dir} "
            f"trained {run_config.model} at {run_config.revision}"
        )
    found = {c.path.name: c for c in train.checkpoints(run_dir)}
    path = run_dir / train.CHECKPOINTS / config.checkpoint
    if config.checkpoint not in found:
        raise ServingError(f"{path}: no complete checkpoint (with {train.STATE})")
    weights = path / train.ADAPTER_WEIGHTS
    if not weights.is_file():
        raise ServingError(f"{weights} not found")
    digest = score.file_sha256(weights)
    if digest != config.adapter_sha256:
        raise ServingError(
            f"{weights} has sha256 {digest}, not the serving config's "
            f"{config.adapter_sha256}"
        )
    state = json.loads((path / train.STATE).read_text())
    model = score.Model(
        ADAPTER,
        config.served(ADAPTER),
        config.run_name(ADAPTER, "-"),  # not used: each split has its run name
        found[config.checkpoint].step,
        state["examples_seen"],
        path.resolve(),
    )
    score.check_adapters([model], run_config.r)
    return model
