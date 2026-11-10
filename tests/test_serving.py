"""secbench.serving: the serving config's keys, types and values, and its names."""

import hashlib
import re
from pathlib import Path

import pytest

from secbench import serving
from secbench.serving import ServingError

ROOT = Path(__file__).parent.parent

VALUES = {
    "model": {
        "run": '"run"',
        "base": '"test/tiny-qwen3"',
        "revision": '"' + "0" * 40 + '"',
        "checkpoint": '"step-00002"',
        "adapter_sha256": '"' + hashlib.sha256(b"adapter weights").hexdigest() + '"',
    },
    "server": {
        "variants": '["adapter", "merged", "fp8", "int4-awq"]',
        "max_model_len": "40960",
        "prefix_caching": "false",
    },
    "request": {
        "prompt": '"extract_v2"',
        "temperature": "0.0",
        "enable_thinking": "false",
        "max_tokens": "1024",
    },
    "load": {
        "split": '"time_test"',
        "levels": "[1, 2]",
        "prompts": "2",
        "warmup": "1",
        "seed": "7",
    },
    "accuracy": {"splits": '["validation", "time_test"]', "concurrency": "4"},
    "calibration": {
        "examples": "3",
        "max_tokens": "2600",
        "seed": "7",
        "scheme": '"W4A16_ASYM"',
        "duo_scaling": '"both"',
        "ignore": '["lm_head"]',
    },
    "report": {},
}


def write_serving_config(path: Path, **changes: str | None) -> Path:
    """A serving config; changes as table__key = TOML value (None leaves it out)."""
    tables = {name: dict(keys) for name, keys in VALUES.items()}
    for key, value in changes.items():
        table, name = key.split("__")
        if value is None:
            tables[table].pop(name, None)
        else:
            tables.setdefault(table, {})[name] = value
    lines = []
    for name, keys in tables.items():
        lines.append(f"[{name}]")
        lines += [f"{k} = {v}" for k, v in keys.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_repository_config() -> None:
    config = serving.load_config(ROOT / "configs/serving/qwen3-8b-2k.toml")
    assert config.variants == ("adapter", "merged", "fp8", "int4-awq")
    assert config.levels == (1, 2, 4, 8, 16, 32, 64)
    assert (config.prompts, config.warmup, config.load_split) == (128, 8, "time_test")
    assert config.accuracy_splits == ("validation", "time_test")
    assert config.concurrency == 16 and config.pool is None and config.limits == {}
    assert config.max_model_len == 40960 and config.prefix_caching is False
    assert (config.calibration_examples, config.calibration_max_tokens) == (
        128,
        16384,
    )
    assert config.scheme == "W4A16_ASYM" and config.ignore == ("lm_head",)
    # The checkpoint and adapter of the stage 2 test config.
    test = (ROOT / "configs/test/qwen3-8b-2k.toml").read_text()
    assert f'adapter_sha256 = "{config.adapter_sha256}"' in test
    assert f'checkpoint = "{config.checkpoint}"' in test
    assert f'revision = "{config.revision}"' in test


def test_names_and_paths(tmp_path) -> None:
    config = serving.load_config(write_serving_config(tmp_path / "c.toml"))
    assert config.served("adapter") == "run--step-00002"  # stage 2's name
    assert config.served("int4-awq") == "run--step-00002--int4-awq"
    assert config.run_name("fp8", "validation") == "run--serving--fp8--validation"
    assert config.serving_dir == Path("training/run/serving")
    assert config.models_dir == Path("training/run/serving/models")
    assert config.calibration_dir == Path("data/serving/run")
    assert config.splits == ("validation", "time_test")
    assert serving.MODEL_OF == {
        "adapter": None,
        "merged": "merged",
        "fp8": "merged",
        "int4-awq": "int4-awq",
    }


def test_optional_keys(tmp_path) -> None:
    config = serving.load_config(
        write_serving_config(
            tmp_path / "c.toml",
            load__pool="5",
            accuracy__limit="{ validation = 2, time_test = 4 }",
            report__validation_reference='"runs/a"',
        )
    )
    assert config.pool == 5
    assert config.limits == {"validation": 2, "time_test": 4}
    assert config.validation_reference == Path("runs/a")
    assert config.time_test_reference is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"server__variants": '["adapter", "awq"]'}, "server.variants must be"),
        ({"server__variants": '["fp8", "fp8"]'}, "server.variants must be"),
        (
            {"request__temperature": "0.7"},
            "request.temperature is 0.7, not stage 2's 0.0",
        ),
        ({"request__max_tokens": "512"}, "request.max_tokens is 512, not stage 2's"),
        ({"request__prompt": '"extract_v1"'}, "request.prompt is 'extract_v1'"),
        ({"server__max_model_len": "32768"}, "server.max_model_len is 32768"),
        ({"load__levels": "[2, 1]"}, "load.levels must be rising"),
        ({"load__levels": "[0, 1]"}, "load.levels must be rising"),
        ({"load__prompts": "0"}, "load.prompts must be 1 or more"),
        ({"load__split": '"train"'}, "load.split must be one of"),
        ({"load__pool": "2"}, "load.pool must hold load.prompts + load.warmup"),
        ({"accuracy__splits": '["company"]'}, "accuracy.splits must be distinct"),
        ({"accuracy__limit": "{ company_test = 2 }"}, "accuracy.limit.company_test"),
        ({"accuracy__limit": "{ validation = 0 }"}, "accuracy.limit.validation"),
        ({"calibration__duo_scaling": '"half"'}, "calibration.duo_scaling must be"),
        ({"load__seed": None}, "load.seed not set"),
        ({"load__seed": '"7"'}, "load.seed must be a whole number"),
        ({"load__levels": '["1"]'}, "load.levels must be a list of whole numbers"),
        ({"load__extra": "1"}, "unknown key load.extra"),
        ({"other__x": "1"}, "unknown table [other]"),
    ],
)
def test_config_problems(tmp_path, changes, message) -> None:
    path = write_serving_config(tmp_path / "c.toml", **changes)
    with pytest.raises(ServingError, match=re.escape(message)):
        serving.load_config(path)


def test_a_missing_or_broken_config(tmp_path) -> None:
    with pytest.raises(ServingError, match="not found"):
        serving.load_config(tmp_path / "none.toml")
    (tmp_path / "bad.toml").write_text("[model\n")
    with pytest.raises(ServingError, match="bad.toml"):
        serving.load_config(tmp_path / "bad.toml")
