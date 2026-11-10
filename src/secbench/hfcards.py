"""The model repositories' folders for the Hugging Face export (secbench.hfexport).

build_model writes the folder of <user>/qwen3-8b-sec-earnings-lora: step-00025's LoRA
adapter (adapter_model.safetensors, whose sha256 must be configs/test/qwen3-8b-2k.toml's
adapter_sha256), its adapter_config.json with two values changed (the saved
base_model_name_or_path names the GPU instance's snapshot path; it becomes the Hub's
Qwen/Qwen3-8B, and revision the pinned revision), the prompt extract_v2.txt (the
card's snippet downloads it), the test chart and the model card. build_int4 writes the
card of <user>/qwen3-8b-sec-earnings-int4-awq (the adapter merged into the base model
and quantized to 4-bit weights with AWQ on the GPU instance) once
results/serving_results.json exists and its int4-awq variant is complete (with its
build manifest); until then it writes nothing and says why, and the model card doesn't
link that repository.

No code here uploads the 4-bit model: its folder stays on the GPU instance, to be
uploaded from there by hand, and only its card goes up with the export. The build
can't see the Hub, so the card's claim that the model is in its repository is checked
when it is published: scripts/hf_publish.sh reads the repository before it uploads
anything and stops unless it already holds the model's config.json and safetensors
weights.

Every number in a card is read from a results file by code: results/test_results.json
(formatted with secbench.testreport's helpers, so it reads as results/test_results.md),
the training run's config.json, train_log.jsonl and validation.json, the checkpoint's
state.json, the test config, and results/serving_results.json (formatted with
secbench.servereport's helpers, as results/serving_results.md). The serving results'
server commands, model and run paths and error messages are never copied: they hold
the GPU instance's paths. The card's claims about the test results (step-00025 trails
every API model on the time test and costs less than each, among others) are checked
against the numbers, and a contradiction stops the build rather than overstate.

The vLLM command is score.vllm_command's for the test config (stage 2's flags), with
the base model by its Hub name at the pinned revision and the adapter from its Hub
repository; the request snippet sends exactly the body secbench.extract sent.

The int4 card's library_name is transformers: the folder is a transformers model
(config.json with a compressed-tensors quantization_config, safetensors weights),
which vLLM serves and transformers loads with compressed-tensors installed, so the Hub
files it under the library that reads it.

Library only.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from secbench import (
    chat,
    extract,
    hfcommon,
    score,
    servebench,
    servereport,
    serving,
    splits,
    testreport,
    train,
)
from secbench.hfcommon import (
    COMPANY_TEST,
    DATASET,
    FIELD_NAMES,
    INT4,
    MODEL,
    SELF_HOSTED,
    SPACE,
    TEST_SPLITS,
    TIME_TEST,
    VALIDATION,
    Context,
    ExportError,
    Sources,
    Written,
)
from secbench.labels import KINDS
from secbench.testreport import (
    COMMON,
    QUARTER_AS_YTD,
    SPLIT_TITLES,
    WITHOUT_PREFERRED,
    acc_cell,
    bounds,
    company_count,
    frac,
    money,
    num,
)

README = "README.md"
CHART = "test_accuracy_vs_cost.png"
BASE_KEY = "base_model_name_or_path"  # adapter_config.json's two changed keys
REVISION_KEY = "revision"
RELEASE_FILE = "release.md"  # the snippet's input: one release's text
# The snippet's server: vllm serve's default port, as the command sets none.
VLLM_URL = "http://localhost:8000/v1/chat/completions"
DROPPED_FLAGS = ("--host", "--port", "--served-model-name")  # each with one value
RESULTS_MD = (testreport.DEFAULT_OUT_DIR / testreport.RESULTS_MD).as_posix()
SERVING_MD = (servereport.DEFAULT_OUT / servereport.REPORT).as_posix()
SERVING_JSON = (servereport.DEFAULT_OUT / servereport.RESULTS).as_posix()
NO_SERVING_SECTION = f"model card's serving section: {SERVING_JSON} not found"
NO_INT4_CARD = f"int4-awq card: {SERVING_JSON} not found"
TAGS = ("qwen3", "finance", "sec", "xbrl", "earnings", "information-extraction")
MODEL_TAGS = ("lora", "peft", *TAGS)
INT4_TAGS = ("awq", "4-bit", "compressed-tensors", "vllm", *TAGS)
METRIC_NAME = "Share of graded figures correct (%)"
CHART_TEXT = (
    "Accuracy against cost per 1,000 releases on both test sets, with 95% intervals; "
    "the company test's fine-tuned point is joined to its cost without the instance "
    "freeze"
)
# One BibTeX entry for the project, the same in every card (secbench.hfdataset's
# citation too; a test checks they match). Bibliography styles lowercase a title's
# words: braces keep the name and the acronyms as written.
CITATION_TITLE = (
    "extracting {GAAP} figures from {SEC} earnings releases, graded against {XBRL}"
)
# vLLM 0.32.0's and torch 2.13.0's Linux wheels are CUDA 13 builds: they bring the CUDA
# 13.0 runtime (uv.lock's nvidia-cuda-runtime, which a test checks), so the GPU's
# driver must support it (scripts/vast_setup.sh).
CUDA = "13.0"
VARIANT_TEXT = {
    serving.ADAPTER: "the base model in bf16 with the LoRA adapter, as stage 2 served it",
    serving.MERGED: "the adapter merged into the base weights, bf16",
    serving.FP8: "merged, quantized to FP8 by vLLM as it loads",
    serving.INT4_AWQ: "merged, 4-bit weights with AWQ",
}
FIGURES_PER_RELEASE = len(extract.GROUPS) * len(FIELD_NAMES)
LOSS_TAIL = 5  # the steps up to the checkpoint whose single losses the card shows


def github(path: str) -> str:
    return f"{hfcommon.REPO_URL}/blob/main/{path}"


def require(condition: bool, claim: str) -> None:
    """A claim a card makes; the build stops if the numbers contradict it."""
    if not condition:
        raise ExportError(f"the numbers no longer support the card's claim: {claim}")


# Inputs


def test_config(sources: Sources) -> score.TestConfig:
    try:
        test = score.load_test_config(sources.test_config)
    except score.ScoreError as exc:
        raise ExportError(str(exc)) from None
    if (test.run, test.checkpoint) != (hfcommon.TRAINING_RUN, hfcommon.CHECKPOINT):
        raise ExportError(
            f"{sources.test_config} names {test.run} {test.checkpoint}, not "
            f"{hfcommon.TRAINING_RUN} {hfcommon.CHECKPOINT}"
        )
    return test


def served_name(test: score.TestConfig) -> str:
    """The name stage 2 served the adapter by (score.checkpoint_under_test's)."""
    return f"{test.run}--{test.checkpoint}"


def check_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ExportError(f"{path} not found")
    found = hfcommon.sha256_file(path)
    if found != expected:
        raise ExportError(
            f"{path} has sha256 {found}, not the test config's adapter_sha256 "
            f"{expected}"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ExportError(f"{path} not found")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@dataclass(frozen=True)
class Training:
    """The training run's files the model card reads."""

    config: dict[str, Any]  # training/<run>/config.json
    state: dict[str, Any]  # the checkpoint's state.json
    validation: dict[str, Any]  # validation.json: every checkpoint's score
    log: list[dict[str, Any]]  # train_log.jsonl, one line per optimizer step
    filed: tuple[str, str]  # the drawn examples' first and last filing date

    @property
    def lora(self) -> dict[str, Any]:
        return self.config["settings"]["lora"]

    def losses_to(self, step: int) -> list[float] | None:
        """Each optimizer step's loss, steps 1 to step; None if the log lacks one."""
        found = {x["step"]: x["loss"] for x in self.log}
        if step < 1 or any(s not in found for s in range(1, step + 1)):
            return None
        return [found[s] for s in range(1, step + 1)]


def load_training(sources: Sources) -> Training:
    folder = sources.training
    validation = hfcommon.read_json(folder / score.RESULTS)
    if validation.get("chosen") != hfcommon.CHECKPOINT:
        raise ExportError(
            f"{folder / score.RESULTS} chose {validation.get('chosen')}, not "
            f"{hfcommon.CHECKPOINT}"
        )
    return Training(
        config=hfcommon.read_json(folder / train.CONFIG),
        state=hfcommon.read_json(sources.checkpoint / train.STATE),
        validation=validation,
        log=read_jsonl(folder / train.TRAIN_LOG),
        filed=filing_dates(sources),
    )


def filing_dates(sources: Sources) -> tuple[str, str]:
    """The first and last filing date of the examples the run drew (labels'
    examples.csv has each example's date, the run's examples.csv its draw)."""
    index = {
        row["release_accession"]: row["filing_date"]
        for row in hfcommon.read_csv(sources.labels / train.EXAMPLES)
    }
    drawn = hfcommon.training_positions(sources)
    missing = sorted(a for a in drawn if a not in index)
    if missing or not drawn:
        raise ExportError(
            f"{sources.labels / train.EXAMPLES} lacks the run's drawn examples: "
            + ", ".join(missing[:5])
        )
    dates = sorted(index[a] for a in drawn)
    return dates[0], dates[-1]


def adapter_config(saved: Mapping[str, Any], test: score.TestConfig) -> dict[str, Any]:
    """The checkpoint's adapter_config.json, pointing at the Hub's base model at the
    pinned revision (the saved one names the GPU instance's snapshot path); every other
    key, and the order, as saved."""
    missing = [key for key in (BASE_KEY, REVISION_KEY) if key not in saved]
    if missing:
        raise ExportError(
            f"the checkpoint's {train.ADAPTER_CONFIG} has no {' or '.join(missing)}"
        )
    changed = {BASE_KEY: test.base, REVISION_KEY: test.revision}
    return {key: changed.get(key, value) for key, value in saved.items()}


def check_lora(saved: Mapping[str, Any], lora: Mapping[str, Any]) -> None:
    """The adapter is the one the run's config describes (the card shows the
    config's settings)."""
    found = (
        saved.get("r"),
        saved.get("lora_alpha"),
        saved.get("lora_dropout"),
        sorted(saved.get("target_modules") or []),
    )
    wanted = (lora["r"], lora["alpha"], lora["dropout"], sorted(lora["target_modules"]))
    if found != wanted:
        raise ExportError(
            f"the checkpoint's {train.ADAPTER_CONFIG} has LoRA {found}, the run's "
            f"config {wanted}"
        )


def load_serving(sources: Sources, test: score.TestConfig) -> dict[str, Any] | None:
    """results/serving_results.json, checked to be this checkpoint's; None if
    absent."""
    path = sources.serving_results
    if not path.is_file():
        return None
    results = hfcommon.read_json(path)
    session = results.get("session") or {}
    found = (
        results.get("run"),
        results.get("checkpoint"),
        session.get("base_model"),
        session.get("revision"),
    )
    if found != (test.run, test.checkpoint, test.base, test.revision):
        raise ExportError(
            f"{path} is for {found}, not the test config's {test.run} "
            f"{test.checkpoint} on {test.base} at {test.revision}"
        )
    return results


def int4_problem(results: Mapping[str, Any]) -> str | None:
    """Why the 4-bit model gets no card, and no link from the adapter's card, or
    None: its variant didn't complete, or its build manifest is missing. (Whether its
    files reached the Hub is checked by scripts/hf_publish.sh before any upload.)"""
    variant = (results.get("variants") or {}).get(serving.INT4_AWQ) or {}
    status = variant.get("status") or "not run"
    if status != servebench.COMPLETE:
        return (
            f"int4-awq card: its variant in {SERVING_JSON} is {status}, not "
            f"{servebench.COMPLETE}"
        )
    if not (results.get("models") or {}).get(serving.INT4_AWQ):
        return (
            f"int4-awq card: {SERVING_JSON} has no models.{serving.INT4_AWQ} (the "
            "4-bit model's build manifest)"
        )
    return None


def vllm_version(server: str) -> str:
    """'0.32.0' from 'vLLM 0.32.0'."""
    match = re.fullmatch(r"vLLM (\S+)", server or "")
    if match is None:
        raise ExportError(f"not a vLLM server: {server!r}")
    return match[1]


def test_server(results: Mapping[str, Any]) -> str:
    """The vLLM version both test splits were served with."""
    versions = {vllm_version(s) for s in results["model_under_test"]["server"].values()}
    if len(versions) != 1:
        raise ExportError(f"the test splits used different servers: {versions}")
    return versions.pop()


def variant_texts(ctx: Context, results: Mapping[str, Any]) -> dict[str, str]:
    """What each serving variant serves. The adapter variant is stage 2's server (the
    same vLLM, checked), but with prefix caching off when the session turned it off
    (stage 2 left vLLM's default, on)."""
    session = vllm_version(results["session"]["server"])
    require(
        session == test_server(ctx.test_results),
        "the serving benchmark's adapter variant ran on stage 2's vLLM version",
    )
    texts = dict(VARIANT_TEXT)
    if not results["settings"]["prefix_caching"]:
        texts[serving.ADAPTER] += " but with prefix caching off"
    return texts


def gpu_names(gpus: Iterable[str] | None, prose: bool = False) -> str:
    """'1× NVIDIA H100 NVL' (or in prose 'one NVIDIA H100 NVL') from nvidia-smi's
    lines ('NAME, MEMORY, DRIVER')."""
    found = list(gpus or [])
    if not found:
        return "an unrecorded GPU"
    names = ", ".join(sorted({g.split(",")[0].strip() for g in found}))
    if prose:
        count = {1: "one", 2: "two", 4: "four", 8: "eight"}.get(len(found))
        return f"{count or len(found)} {names}"
    return f"{len(found)}× {names}"


def split_phrase(split: str) -> str:
    """'validation', 'the company test', 'the time test'."""
    return split if split == VALIDATION else f"the {SPLIT_TITLES[split].lower()}"


# Text helpers


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    return testreport.table_md(list(header), [list(row) for row in rows])


def words(items: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def short_float(value: float) -> str:
    """A loss to two significant figures (0.00062)."""
    return f"{value:.2g}"


def dollars(value: float | None) -> str:
    """Dollars per 1,000 releases as results/serving_results.md shows them."""
    return "–" if value is None else f"${servereport.fmt(value, '.3f')}"


def py(value: Any) -> str:
    """A Python literal for a JSON-like value, strings double-quoted."""
    if value is None or isinstance(value, bool | int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{py(k)}: {py(v)}" for k, v in value.items()) + "}"
    if isinstance(value, tuple):
        inner = ", ".join(py(v) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(py(v) for v in value) + "]"
    raise ExportError(f"no Python literal for {value!r}")


# YAML front matter: the few shapes the cards need, laid out as the Hub lays it out.

PLAIN_YAML = re.compile(r"[A-Za-z][A-Za-z0-9_./-]*")
YAML_WORDS = {"y", "n", "yes", "no", "true", "false", "on", "off", "null"}


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        if not math.isfinite(value) or "e" in text:
            raise ExportError(f"no plain YAML number for {value!r}")
        return text
    if isinstance(value, str):
        if PLAIN_YAML.fullmatch(value) and value.lower() not in YAML_WORDS:
            return value
        return json.dumps(value, ensure_ascii=False)  # a YAML double-quoted scalar
    raise ExportError(f"no YAML scalar for {value!r}")


def yaml_mapping(
    data: Mapping[str, Any], indent: int, lines: list[str], first: str | None = None
) -> None:
    pad = " " * indent
    for i, (key, value) in enumerate(data.items()):
        lead = first if i == 0 and first is not None else pad
        if isinstance(value, Mapping | list | tuple) and not value:
            lines.append(f"{lead}{key}: {'{}' if isinstance(value, Mapping) else '[]'}")
        elif isinstance(value, Mapping):
            lines.append(f"{lead}{key}:")
            yaml_mapping(value, indent + 2, lines)
        elif isinstance(value, list | tuple):
            lines.append(f"{lead}{key}:")
            yaml_list(value, indent, lines)
        else:
            lines.append(f"{lead}{key}: {yaml_scalar(value)}")


def yaml_list(items: Sequence[Any], indent: int, lines: list[str]) -> None:
    pad = " " * indent
    for item in items:
        if isinstance(item, Mapping):
            yaml_mapping(item, indent + 2, lines, first=f"{pad}- ")
        elif isinstance(item, list | tuple):
            raise ExportError("a list in a list isn't needed in a card")
        else:
            lines.append(f"{pad}- {yaml_scalar(item)}")


def front_matter(metadata: Mapping[str, Any]) -> list[str]:
    lines = ["---"]
    yaml_mapping(metadata, 0, lines)
    return [*lines, "---", ""]


# Serving the model and sending a release


def serve_command(test: score.TestConfig, r: int, adapter_repo: str) -> list[str]:
    """score.vllm_command for the test config (stage 2's flags), for the Hub: the base
    model by its Hub name with its weights and tokenizer at the pinned revision
    (stage 2 served a downloaded snapshot of that revision, and its token-for-token
    check relied on that tokenizer), the adapter from its repository, and no host,
    port or --served-model-name (the model's name is its default)."""
    served = served_name(test)
    adapter = score.Model(test.checkpoint, served, served, 0, 0, Path(adapter_repo))
    try:
        command = score.vllm_command(
            Path(test.base), test.base, [adapter], score.DEFAULT_PORT, r
        )
    except score.ScoreError as exc:
        raise ExportError(str(exc)) from None
    if command[1:3] != ["serve", test.base]:
        raise ExportError(f"unexpected vllm command: {command[:3]}")
    kept = [
        "vllm",
        *command[1:3],
        "--revision",
        test.revision,
        "--tokenizer-revision",
        test.revision,
    ]
    rest = iter(command[3:])
    for word in rest:
        if word in DROPPED_FLAGS:
            value = next(rest)
            if word == "--served-model-name" and value != test.base:
                raise ExportError(f"the base model is served as {value}")
            continue
        kept.append(word)
    expected = {
        "--max-model-len": test.max_model_len,
        "--max-lora-rank": test.max_lora_rank,
        "--max-loras": test.max_loras,
        "--max-cpu-loras": test.max_cpu_loras,
    }
    for flag, value in expected.items():
        if flag not in kept or kept[kept.index(flag) + 1] != str(value):
            raise ExportError(f"vllm_command's {flag} isn't the test config's {value}")
    if "--enable-lora" not in kept:
        raise ExportError("vllm_command no longer enables LoRA")
    return kept


def shell_lines(command: Sequence[str], width: int = 80) -> list[str]:
    """command for a shell, each option with its values on one line, wrapped with
    backslashes."""
    pieces: list[list[str]] = []
    for word in command:
        if word.startswith("--") or not pieces:
            pieces.append([word])
        else:
            pieces[-1].append(word)
    lines: list[str] = []
    current = shlex.join(pieces[0])
    for piece in pieces[1:]:
        text = shlex.join(piece)
        if len(current) + 1 + len(text) > width - 2:
            lines.append(current + " \\")
            current = "    " + text
        else:
            current += " " + text
    return [*lines, current]


def request_parameters(test: score.TestConfig) -> dict[str, Any]:
    """What each request sends besides the model and messages: the test config's
    settings, which must be extract's request_parameters for stage 2."""
    found = {
        "temperature": test.temperature,
        "chat_template_kwargs": {"enable_thinking": test.enable_thinking},
        "max_tokens": test.max_tokens,
    }
    if found != score.expected_parameters():
        raise ExportError(
            f"the test config's request {found} isn't stage 2's "
            f"{score.expected_parameters()}"
        )
    return found


def request_snippet(
    prompt_repo: str, prompt_file: str, model: str, parameters: Mapping[str, Any]
) -> list[str]:
    """Python that sends one release as secbench.extract sent it (the same body) and
    converts the reply's figures to dollars as extract does."""
    return [
        "import json",
        "from decimal import Decimal",
        "",
        "import httpx",
        "from huggingface_hub import hf_hub_download",
        "",
        "prompt_path = hf_hub_download(",
        f"    repo_id={py(prompt_repo)}, filename={py(prompt_file)}",
        ")",
        'with open(prompt_path, encoding="utf-8") as f:',
        "    prompt = f.read()  # the system message, verbatim",
        f'with open({py(RELEASE_FILE)}, encoding="utf-8") as f:',
        "    release = f.read()  # one press release as Markdown",
        "",
        "response = httpx.post(",
        f"    {py(VLLM_URL)},",
        "    json={",
        f'        "model": {py(model)},',
        '        "messages": [',
        '            {"role": "system", "content": prompt},',
        '            {"role": "user", "content": release},',
        "        ],",
        *(f"        {py(key)}: {py(value)}," for key, value in parameters.items()),
        "    },",
        f"    timeout={py(extract.TIMEOUT.read)},",
        ")",
        "response.raise_for_status()",
        'content = response.json()["choices"][0]["message"]["content"]',
        "reply = json.loads(content, parse_float=Decimal)",
        "",
        "# Revenue and net income come as printed, with their unit; EPS is a number.",
        "UNITS = {",
        *(f"    {py(unit)}: {factor:_}," for unit, factor in extract.UNITS.items()),
        "}",
        'figures = {"period_end": reply["period_end"]}',
        f"for period in {py(tuple(extract.GROUPS))}:",
        "    figures[period] = {}",
        "    for name, figure in reply[period].items():",
        '        if isinstance(figure, dict):  # {"value": 12345, "unit": "millions"}',
        '            value, unit = figure["value"], figure.get("unit")',
        "            figure = None if value is None else Decimal(value) * UNITS[unit]",
        "        figures[period][name] = figure",
        "print(json.dumps(figures, indent=2, default=str))",
    ]


def reply_shape(prompt: str) -> list[str]:
    """The reply's JSON shape, as the prompt shows it."""
    match = re.search(r'^\{"period_end".*?\}\}$', prompt, re.MULTILINE | re.DOTALL)
    if match is None:
        raise ExportError("the prompt no longer shows the reply's shape")
    return match[0].splitlines()


def citation(sources: Sources) -> list[str]:
    """The project's BibTeX entry, the dataset card's own (one key, one entry)."""
    year, name = hfcommon.author(sources)
    project = hfcommon.REPO_NAMES[DATASET]  # sec-earnings-bench
    key = re.sub(r"[^a-z]", "", name.split()[-1].lower()) + year
    return [
        "```bibtex",
        f"@misc{{{key}{project.replace('-', '')},",
        f"  author = {{{name}}},",
        f"  title = {{{{{project}}}: {CITATION_TITLE}}},",
        f"  year = {{{year}}},",
        f"  url = {{{hfcommon.REPO_URL}}}",
        "}",
        "```",
    ]


def links_line(ctx: Context) -> str:
    return (
        f"[Code, method and full results (GitHub)]({hfcommon.REPO_URL}) · "
        f"[Dataset: {ctx.repo_id(DATASET)}]({ctx.hub_url(DATASET)}) · "
        f"[Results explorer (Space)]({ctx.hub_url(SPACE)})"
    )


def closing_lines(ctx: Context, base: str) -> list[str]:
    """License, citation and links, the end of both cards."""
    return [
        "## License",
        "",
        (
            f"Apache-2.0, like {base}. The benchmark's code is MIT-licensed "
            f"([GitHub]({hfcommon.REPO_URL})); the dataset card states its data's terms."
        ),
        "",
        "## Citation",
        "",
        *citation(ctx.sources),
        "",
        "## Links",
        "",
        f"- Code, method and every result: [GitHub]({hfcommon.REPO_URL})",
        f"- Dataset: [{ctx.repo_id(DATASET)}]({ctx.hub_url(DATASET)})",
        f"- Results explorer: [{ctx.repo_id(SPACE)}]({ctx.hub_url(SPACE)})",
        "",
    ]


# The test results


@dataclass(frozen=True)
class Report:
    """results/test_results.json, with the names the cards use."""

    results: dict[str, Any]

    @property
    def labels(self) -> dict[str, str]:
        return self.results["models"]

    @property
    def under(self) -> str:
        return self.results["under_test"]

    @property
    def base(self) -> str:
        return self.results["base"]

    @property
    def api(self) -> list[str]:
        """The API references, in the report's order."""
        return [m for m in self.labels if m not in (self.under, self.base)]

    def split(self, name: str) -> dict[str, Any]:
        return self.results["splits"][name]

    def accuracy(self, split: str, model: str, subset: str = "all") -> dict[str, Any]:
        return self.split(split)["accuracy"][model][subset]

    def difference(self, split: str, model: str) -> dict[str, Any]:
        return self.split(split)["differences"][model]["all"]

    def difference_cell(self, split: str, model: str) -> str:
        d = self.difference(split, model)
        return (
            f"{num(d['difference'], signed=True)} {bounds(d['interval'], signed=True)}"
        )

    def cost(self, split: str, model: str) -> float:
        return self.split(split)["cost"][model]["per_1000"]

    def trails_api_on_time_test(self) -> tuple[float, float]:
        """The smallest and largest gap to the API models on the time test, in points;
        the card says step-00025 trails each, with a 95% interval below 0."""
        gaps = []
        for model in self.api:
            d = self.difference(TIME_TEST, model)
            require(
                d["difference"] < 0 and d["interval"][1] < 0,
                f"{self.labels[self.under]} trails {self.labels[model]} on the time "
                "test, with a 95% interval below 0",
            )
            gaps.append(-d["difference"])
        return min(gaps), max(gaps)

    def cheaper_on_time_test(self) -> tuple[float, float]:
        """The API models' lowest and highest cost per 1,000 time-test releases; the
        card says step-00025 costs less than each."""
        costs = [self.cost(TIME_TEST, m) for m in self.api]
        require(
            self.cost(TIME_TEST, self.under) < min(costs),
            f"{self.labels[self.under]} costs less than every API model on the time "
            "test",
        )
        return min(costs), max(costs)


def runs_on(ctx: Context) -> dict[str, str]:
    """Where each test model ran: the GPU for the self-hosted ones, else its API."""
    gpus = {gpu_names(g) for g in ctx.test_results["model_under_test"]["gpus"].values()}
    gpu = gpus.pop() if len(gpus) == 1 else "a GPU per test split"
    found: dict[str, str] = {}
    for split in TEST_SPLITS:
        for run in ctx.test_runs[split]:
            where = gpu if run.source == SELF_HOSTED else run.source
            if found.setdefault(run.key, where) != where:
                raise ExportError(f"{run.key} ran on {found[run.key]} and {where}")
    return found


def freeze_note(report: Report, split: str, model: str) -> str:
    s = report.split(split)
    freeze = s.get("freeze")
    if freeze and freeze["run"] == s["runs"][model]["run"]:
        return (
            f" (about {money(freeze['excluded_per_1000'])} without the instance freeze)"
        )
    return ""


# The model card


def model_card(
    ctx: Context,
    test: score.TestConfig,
    training: Training,
    prompt: str,
    serving_results: dict[str, Any] | None,
) -> str:
    report = Report(ctx.test_results)
    lines = [
        *front_matter(model_metadata(ctx, report, test)),
        f"# {hfcommon.REPO_NAMES[MODEL]}",
        "",
        *intro_lines(ctx, report, test),
        *what_lines(test, prompt),
        *usage_lines(ctx, report, test, training),
        *results_lines(ctx, report, training),
        *failure_lines(report),
        *training_lines(ctx, report, test, training),
        *protocol_lines(report),
        *(serving_lines(ctx, serving_results) if serving_results else []),
        *limitation_lines(ctx, report, training),
        *closing_lines(ctx, test.base),
    ]
    return "\n".join(lines).rstrip() + "\n"


def model_metadata(
    ctx: Context, report: Report, test: score.TestConfig
) -> dict[str, Any]:
    dataset = ctx.repo_id(DATASET)
    results = [
        {
            "task": {"type": "text-generation"},
            "dataset": {
                "name": dataset,
                "type": dataset,
                "config": "answers",
                "split": split,
            },
            "metrics": [
                {
                    "type": "accuracy",
                    "value": round(report.accuracy(split, report.under)["accuracy"], 2),
                    "name": METRIC_NAME,
                }
            ],
            "source": {"name": RESULTS_MD, "url": github(RESULTS_MD)},
        }
        for split in TEST_SPLITS
    ]
    return {
        "base_model": test.base,
        "base_model_relation": "adapter",
        "library_name": "peft",
        "license": "apache-2.0",
        "language": ["en"],
        "pipeline_tag": "text-generation",
        "tags": list(MODEL_TAGS),
        "datasets": [dataset],
        "model-index": [{"name": hfcommon.REPO_NAMES[MODEL], "results": results}],
    }


def intro_lines(ctx: Context, report: Report, test: score.TestConfig) -> list[str]:
    labels, under = report.labels, report.under
    company, time_test = report.split(COMPANY_TEST), report.split(TIME_TEST)
    primary = report.difference(COMPANY_TEST, report.base)
    low, high = report.trails_api_on_time_test()
    cheapest, dearest = report.cheaper_on_time_test()
    t = report.results["time_test"]
    return [
        (
            f"A LoRA adapter for [{test.base}](https://huggingface.co/{test.base}) that "
            "reads a US company's earnings press release and returns its GAAP revenue, "
            "net income and diluted EPS as one JSON object. It is checkpoint "
            f"{test.checkpoint} of the training run {test.run} in "
            f"[sec-earnings-bench]({hfcommon.REPO_URL}), a benchmark that grades every "
            "figure against the XBRL of the company's matching 10-Q or 10-K."
        ),
        "",
        (
            f"- **Company test** ({company['coverage']['kept_releases']} releases from "
            f"{company['coverage']['kept_companies']} companies it never trained on): "
            "graded figures correct "
            f"{acc_cell(report.accuracy(COMPANY_TEST, under))}, "
            f"**{num(primary['difference'], signed=True)} points over the base "
            f"model** {bounds(primary['interval'], signed=True)}, the pre-registered "
            "primary result."
        ),
        (
            f"- **Time test** ({time_test['coverage']['kept_releases']} later releases "
            f"of the training companies, filed {t['first_filed']} to "
            f"{t['last_filed']}): graded figures correct "
            f"{acc_cell(report.accuracy(TIME_TEST, under))}, behind "
            f"{words([labels[m] for m in report.api])} by {num(low)} to "
            f"{num(high)} points (95% intervals below 0)."
        ),
        (
            "- **Its case is cost, not accuracy:** "
            f"{money(report.cost(TIME_TEST, under))} per 1,000 releases on the time "
            f"test, against {money(cheapest)} to {money(dearest)} for those API models."
        ),
        "",
        links_line(ctx),
        "",
    ]


def what_lines(test: score.TestConfig, prompt: str) -> list[str]:
    units = [f'`"{unit}"`' for unit in extract.UNITS]
    thinking = json.dumps({"enable_thinking": test.enable_thinking})
    return [
        "## What it does",
        "",
        (
            "Input: one earnings press release (an 8-K's press-release exhibit, "
            "usually EX-99.1, with the exhibit that holds the income statement "
            "appended if the release lacks one) as Markdown, its tables as Markdown "
            "tables: the `text` column of the "
            "dataset's `releases` config, which secbench.inputs builds from the filing "
            "on EDGAR. Output: one JSON object in the reply format of the prompt "
            f"{test.prompt} (`{test.prompt}.txt` in this repository):"
        ),
        "",
        "```json",
        *reply_shape(prompt),
        "```",
        "",
        (
            "- `quarter`: the latest quarter the release reports; `year_to_date`: the "
            "fiscal year to date ending on `period_end` (the full fiscal year in a "
            "fourth-quarter release, the quarter again in a first-quarter release)."
        ),
        (
            "- `revenue`: total revenue; `net_income`: net income attributable to the "
            "company; `eps_diluted`: diluted earnings per share. GAAP figures only; "
            "null for a figure the release doesn't report."
        ),
        (
            "- Revenue and net income come as printed, with the table's unit "
            f"({words(units).replace(' and ', ' or ')}), and code converts them to "
            "dollars; EPS is a plain number in dollars per share."
        ),
        "",
        (
            f"The model is {test.base} at revision `{test.revision}` with this adapter, "
            f"thinking off (`chat_template_kwargs` `{thinking}`), at temperature "
            f"{test.temperature:g}."
        ),
        "",
    ]


def driver_text(version: str) -> str:
    """Why an install that works can still fail to serve: the GPU driver."""
    return (
        f"vLLM {version}'s wheels are built for CUDA {CUDA}: pip installs them "
        f"whatever the driver, but the server needs an NVIDIA driver for CUDA {CUDA} "
        "or later."
    )


def usage_lines(
    ctx: Context, report: Report, test: score.TestConfig, training: Training
) -> list[str]:
    repo = ctx.repo_id(MODEL)
    command = serve_command(test, training.lora["r"], repo)
    limit = chat.MAX_MODEL_LEN - chat.MAX_TOKENS
    if limit != splits.MAX_PROMPT_TOKENS:
        raise ExportError("the context limit isn't splits.MAX_PROMPT_TOKENS")
    left_out = sum(not r.kept for s in TEST_SPLITS for r in ctx.releases[s])
    gpus = {
        gpu_names(g, prose=True)
        for g in report.results["model_under_test"]["gpus"].values()
    }
    version = test_server(report.results)
    snippet = request_snippet(
        repo, f"{test.prompt}.txt", served_name(test), request_parameters(test)
    )
    return [
        "## How to use",
        "",
        (
            "Serve the base model with the adapter as every evaluation did, with vLLM "
            f"{version} on Linux with an NVIDIA GPU (the test runs used "
            f"{words(sorted(gpus))}). {driver_text(version)}"
        ),
        "",
        "```sh",
        f"pip install vllm=={version}",
        *shell_lines(command),
        "```",
        "",
        (
            "Then send each release as the benchmark's runner (secbench.extract) sent "
            "it: the prompt as the system message, the release as the user message, "
            f"temperature {test.temperature:g}, thinking off, at most "
            f"{test.max_tokens:,} output tokens. With `pip install httpx "
            f"huggingface_hub`, and a release saved as `{RELEASE_FILE}` (for example "
            "a `text` value of the dataset's `releases` config: "
            f'`load_dataset("{ctx.repo_id(DATASET)}", "releases", '
            'split="company_test")`):'
        ),
        "",
        "```python",
        *snippet,
        "```",
        "",
        (
            f"The server's context is {chat.MAX_MODEL_LEN:,} tokens and a reply may "
            f"take {chat.MAX_TOKENS:,} of them, so a prompt over {limit:,} tokens "
            f"doesn't fit: such releases ({left_out} in the test sets) were left out "
            "of every evaluation, and the adapter is untested on them."
        ),
        "",
    ]


def results_lines(ctx: Context, report: Report, training: Training) -> list[str]:
    labels, under, base = report.labels, report.under, report.base
    company, time_test = report.split(COMPANY_TEST), report.split(TIME_TEST)
    cov_c, cov_t = company["coverage"], time_test["coverage"]
    t = report.results["time_test"]
    primary = report.difference(COMPANY_TEST, base)
    where = runs_on(ctx)
    low, high = report.trails_api_on_time_test()
    cheapest, dearest = report.cheaper_on_time_test()
    require(
        training.filed[1] < t["first_filed"],
        "the time test's releases were filed after the releases trained on",
    )
    rows = [
        [
            labels[m] + (" (this adapter)" if m == under else ""),
            where[m],
            acc_cell(report.accuracy(COMPANY_TEST, m)),
            acc_cell(report.accuracy(TIME_TEST, m)),
            " / ".join(
                money(report.cost(split, m)) + freeze_note(report, split, m)
                for split in TEST_SPLITS
            ),
        ]
        for m in labels
    ]
    return [
        "## Results",
        "",
        (
            f"From [{RESULTS_MD}]({github(RESULTS_MD)}) (and its .json), the report of "
            "an evaluation protocol fixed before any test call (Evaluation protocol, "
            "below). Accuracy is the share of graded figures correct, a graded "
            "figure being one whose answer the release prints (not_in_input and "
            "no_answer figures are left out; see the dataset card), with 95% bootstrap "
            "intervals over companies."
        ),
        "",
        (
            f"**Primary result** (pre-registered): on the company test, {labels[under]} "
            f"minus {labels[base]}: **{num(primary['difference'], signed=True)} points "
            f"{bounds(primary['interval'], signed=True)}**, from "
            f"{num(report.accuracy(COMPANY_TEST, base)['accuracy'])}% to "
            f"{num(report.accuracy(COMPANY_TEST, under)['accuracy'])}% of "
            f"{cov_c['graded']:,} figures; {primary['discordant']} discordant figures "
            f"({primary['first_right']} fixed by training, {primary['second_right']} "
            f"broken) from {primary['companies']} companies."
        ),
        "",
        *table(
            [
                "model",
                "runs on",
                "company test",
                "time test",
                "cost / 1,000 releases, company / time test",
            ],
            rows,
        ),
        "",
        (
            f"- **Company test:** {cov_c['kept_releases']} releases from "
            f"{cov_c['kept_companies']} companies held out of training; "
            f"{cov_c['graded']:,} graded figures."
        ),
        (
            f"- **Time test:** {cov_t['kept_releases']} releases from "
            f"{cov_t['kept_companies']} of the training companies, filed "
            f"{t['first_filed']} to {t['last_filed']}, after every release drawn for "
            f"training (filed {training.filed[0]} to {training.filed[1]}); "
            f"{cov_t['graded']:,} graded figures. Its {cov_t['releases_graded']} "
            f"releases with graded figures come from {cov_t['companies_graded']} "
            "companies, so the bootstrap there is in effect over releases."
        ),
        cost_bullet(report),
        "",
        f"{labels[under]} minus each reference, in points, with 95% intervals:",
        "",
        *table(
            [f"{labels[under]} minus", "company test", "time test"],
            [
                [
                    labels[m],
                    report.difference_cell(COMPANY_TEST, m),
                    report.difference_cell(TIME_TEST, m),
                ]
                for m in (base, *report.api)
            ],
        ),
        "",
        (
            f"On the time test {labels[under]} trails every API model, by {num(low)} "
            f"to {num(high)} points, with 95% intervals below 0. "
            f"{company_test_standing(report)} Its case is cost, not accuracy: "
            f"{money(report.cost(TIME_TEST, under))} per 1,000 releases on the time "
            f"test, against {money(cheapest)} to {money(dearest)} for the API models."
        ),
        "",
        validation_sentence(report),
        "",
        f"![{CHART_TEXT}]({CHART})",
        "",
    ]


def cost_bullet(report: Report) -> str:
    labels, under = report.labels, report.under
    company, time_test = report.split(COMPANY_TEST), report.split(TIME_TEST)
    text = (
        "- **Cost:** GPU runs at the price billed "
        f"(${company['billed_usd_per_hour']:g}/h in the company test's session, "
        f"${time_test['billed_usd_per_hour']:g}/h in the time test's) times the run's "
        "wall time (first call to last; setup, model loading and idle time left out); "
        "API runs at their token prices."
    )
    freeze = company.get("freeze")
    if freeze and freeze["run"] == company["runs"][under]["run"]:
        text += (
            f" The GPU instance froze for {freeze['window_s']:.0f} s during "
            f"{labels[under]}'s company-test run: "
            f"{money(report.cost(COMPANY_TEST, under))} per 1,000 releases with the "
            f"freeze, about {money(freeze['excluded_per_1000'])} without it."
        )
    return text


def company_test_standing(report: Report) -> str:
    """Where step-00025 stands against the API models on the company test, as README
    puts it: its errors where an API model made none first (an interval that only
    reaches 0 isn't parity), then the models it is level with."""
    labels, under = report.labels, report.under
    for model in report.api:
        low, high = report.difference(COMPANY_TEST, model)["interval"]
        require(
            low <= 0 <= high,
            f"on the company test, {labels[under]} minus {labels[model]}'s interval "
            "includes 0",
        )
    mine = report.accuracy(COMPANY_TEST, under)
    misses = mine["graded"] - mine["correct"]
    perfect = [
        m
        for m in report.api
        if misses and (a := report.accuracy(COMPANY_TEST, m))["correct"] == a["graded"]
    ]
    level = [m for m in report.api if m not in perfect]
    parts = []
    if perfect:
        parts.append(
            f"it made {misses} {'error' if misses == 1 else 'errors'} where "
            f"{words([labels[m] for m in perfect])} made none "
            f"({differences_text(report, perfect, points=True)})"
        )
    if level:
        parts.append(
            f"it is level with {words([labels[m] for m in level])} "
            f"({differences_text(report, level, points=not parts)})"
        )
    return f"On the company test {', and '.join(parts)}."


def differences_text(report: Report, models: Sequence[str], points: bool) -> str:
    """'−0.76 points, interval −1.95 to 0.00' on the company test, once if every model
    has it, else per model."""

    def text(model: str) -> str:
        d = report.difference(COMPANY_TEST, model)
        low, high = (num(x, signed=True) for x in d["interval"])
        unit = " points" if points else ""
        return f"{num(d['difference'], signed=True)}{unit}, interval {low} to {high}"

    found = {m: text(m) for m in models}
    if len(set(found.values())) == 1:
        return next(iter(found.values()))
    return "; ".join(f"{report.labels[m]} {t}" for m, t in found.items())


def validation_sentence(report: Report) -> str:
    correct, graded = report.results["validation"]["counts"][report.under]["all"]
    return (
        f"{report.labels[report.under]}'s validation score, "
        f"{num(100 * correct / graded, 1)}% ({correct}/{graded}), is "
        f"{testreport.CHOSEN_ON_VALIDATION}."
    )


def failure_lines(report: Report) -> list[str]:
    labels, under = report.labels, report.under
    lines = [
        "## Where it fails",
        "",
        (
            f"Descriptive, not pre-registered: {labels[under]}'s misses on the test "
            f"sets, grouped by the rule in [{RESULTS_MD}]({github(RESULTS_MD)}), "
            "which lists every one."
        ),
        "",
        common_line_text(report),
        "",
        *preferred_lines(report),
    ]
    copied = copied_text(report)
    if copied:
        lines += [copied, ""]
    return [*lines, other_misses_text(report), "", abstention_text(report), ""]


def outside_reasons(summary: Mapping[str, Any]) -> list[str]:
    """Why group (1)'s misses outside preferred.csv's releases still name common
    shareholders (testreport.render_misses's reading of the release rows)."""
    outside = summary["outside_rows"]
    preferred = [t for t, rows in outside.items() if "preferred" in rows]
    participating = [
        t
        for t, rows in outside.items()
        if "participating" in rows and t not in preferred
    ]
    neither = [t for t, rows in outside.items() if not rows]
    why = []
    if preferred:
        why.append(
            f"{', '.join(preferred)} print preferred-stock lines that the XBRL "
            "preferred-dividend check missed"
        )
    if participating:
        why.append(
            f"{', '.join(participating)} report income after participating securities"
        )
    if neither:
        why.append(f"{', '.join(neither)} print neither")
    return why


def common_line_text(report: Report) -> str:
    parts = [
        (
            "**The common shareholders' line.** Net income read from a line naming "
            "common shareholders, stockholders, shares or stock: income after "
            "preferred dividends or participating securities, where the answer key's "
            "NetIncomeLoss comes before them."
        )
    ]
    for split in TEST_SPLITS:
        s = report.split(split)["miss_summary"]
        common = s["groups"].get(COMMON, 0)
        if not common:
            parts.append(f"{SPLIT_TITLES[split]}: none of its {s['misses']} misses.")
            continue
        text = (
            f"{SPLIT_TITLES[split]}: {common} of its {s['misses']} misses, from "
            f"{company_count(len(s['common_companies']))} "
            f"({', '.join(s['common_companies'])}); {s['common_in_preferred']} in "
            "releases whose filing reports preferred dividends and "
            f"{s['common_outside_preferred']} in others"
        )
        # The sensitivity analysis below drops preferred.csv's releases: all of this
        # split's misses of this kind, or not (then it understates them here).
        if s["common_outside_preferred"]:
            text += (
                f" ({'; '.join(outside_reasons(s))}), so dropping the "
                "preferred-dividend releases, as below, understates this failure on "
                f"{split_phrase(split)}"
            )
        else:
            text += ", so dropping those releases, as below, drops all of them"
        parts.append(
            text + f"; {s['broken_common']} of the {s['broken']} figures training "
            "broke (the base model had them right)."
        )
    return " ".join(parts)


def preferred_lines(report: Report) -> list[str]:
    labels = report.labels
    counts = {s: report.split(s)["preferred"] for s in TEST_SPLITS}
    net_income = []
    for split, p in counts.items():
        figures = next(iter(p["net_income"].values()))["graded"]
        net_income.append(
            f"{SPLIT_TITLES[split].lower()} ({figures} figures from "
            f"{company_count(len(p['companies']))}): "
            + ", ".join(
                f"{labels[m]} {frac(c['correct'], c['graded'])}"
                for m, c in p["net_income"].items()
            )
        )
    return [
        (
            "**Preferred dividends.** Accuracy without the releases whose filing "
            "reports preferred dividends (whole releases dropped: "
            f"{counts[COMPANY_TEST]['releases']} on the company test, "
            f"{counts[TIME_TEST]['releases']} on the time test), with 95% intervals:"
        ),
        "",
        *table(
            ["model", "company test", "time test"],
            [
                [
                    labels[m],
                    acc_cell(report.accuracy(COMPANY_TEST, m, WITHOUT_PREFERRED)),
                    acc_cell(report.accuracy(TIME_TEST, m, WITHOUT_PREFERRED)),
                ]
                for m in labels
            ],
        ),
        "",
        (
            "Net income on those releases alone, counts only: "
            + "; ".join(net_income)
            + "."
        ),
        "",
    ]


def copied_text(report: Report) -> str | None:
    found = []
    for split in TEST_SPLITS:
        counts = Counter(
            m["ticker"]
            for m in report.split(split)["misses"]
            if m["group"] == QUARTER_AS_YTD
        )
        if counts:
            found.append(
                f"{', '.join(f'{t} {n}' for t, n in sorted(counts.items()))} on the "
                f"{SPLIT_TITLES[split].lower()}"
            )
    if not found:
        return None
    return (
        "**The quarter copied into year to date.** In second- or third-quarter "
        "releases whose income statement has quarter columns only, with the year to "
        "date printed elsewhere, it repeated the quarter's figure as the year to date: "
        f"{' and '.join(found)}."
    )


def other_misses_text(report: Report) -> str:
    parts = []
    for split in TEST_SPLITS:
        groups = report.split(split)["miss_summary"]["groups"]
        rest = sorted(
            (g, n) for g, n in groups.items() if g not in (COMMON, QUARTER_AS_YTD)
        )
        parts.append(
            f"{SPLIT_TITLES[split].lower()} "
            + (", ".join(f"{g} {n}" for g, n in rest) or "none")
        )
    return f"**Other misses**, by verdict: {'; '.join(parts)}."


def abstention_text(report: Report) -> str:
    labels, under = report.labels, report.under
    found = {s: report.split(s)["abstention"] for s in TEST_SPLITS}
    mine = {s: a[under] for s, a in found.items()}
    nulls = sum(a["null"] for a in mine.values())
    figures = ", ".join(
        f"{a['null']} of {a['figures']:,} on {split_phrase(s)}" for s, a in mine.items()
    )
    head = (
        f"**It never answered null.** None of its predicted figures was null ({figures})"
        if nulls == 0
        else f"**It rarely answered null.** Its null predicted figures: {figures}"
    )
    others = ", ".join(
        f"{labels[m]} " + " and ".join(str(found[s][m]["null"]) for s in TEST_SPLITS)
        for m in report.api
    )
    if all(a["answered"] == a["not_in_input"] for a in mine.values()):
        answered = "every not_in_input figure (" + ", ".join(
            f"{a['not_in_input']} on {split_phrase(s)}" for s, a in mine.items()
        )
    else:
        answered = "not_in_input figures (" + ", ".join(
            f"{a['answered']} of {a['not_in_input']} on {split_phrase(s)}"
            for s, a in mine.items()
        )
    return (
        f"{head}; the API models' null figures, company test and time test: {others}. "
        f"It answered {answered}), XBRL answers the grader doesn't find printed. Some "
        "of those releases print the figure in another form, so answering isn't "
        "always wrong; but where a release doesn't report a figure the prompt asks "
        "for null, and no training label was null. Accuracy leaves these figures out."
    )


def loss_sentence(training: Training, step: int, examples_per_step: int) -> str:
    """The training loss up to the checkpoint, from train_log.jsonl: the first step,
    the mean over every step, and the last few steps' range, since one step's loss
    covers only its own examples (empty if the log lacks a step)."""
    losses = training.losses_to(step)
    if not losses:
        return ""
    tail = losses[-LOSS_TAIL:]
    examples = f"{examples_per_step} example{'' if examples_per_step == 1 else 's'}"
    return (
        " Training loss (cross-entropy per reply token, train_log.jsonl): "
        f"{short_float(losses[0])} at step 1, and {short_float(sum(losses) / step)} "
        f"averaged over steps 1 to {step}; a step's loss covers only its {examples}, "
        f"so it is noisy (from {short_float(min(tail))} to {short_float(max(tail))} "
        f"over steps {step - len(tail) + 1} to {step})."
    )


def training_lines(
    ctx: Context, report: Report, test: score.TestConfig, training: Training
) -> list[str]:
    settings = training.config["settings"]
    selection, schedule = training.config["selection"], training.config["schedule"]
    lora, optimizer = training.lora, settings["training"]
    data = settings["data"]
    drawn = selection["selected_by_kind"]
    kinds = ", ".join(
        f"{kind} {drawn.get(kind, 0):,}"
        for kind in (*KINDS, *(k for k in drawn if k not in KINDS))
    )
    modules = words([f"`{m}`" for m in lora["target_modules"]])
    step = training.state["step"]
    checkpointing = (
        ", with gradient checkpointing"
        if optimizer.get("gradient_checkpointing")
        else ""
    )
    loss_text = loss_sentence(training, step, schedule["examples_per_step"])
    v = training.validation
    companies = len({r.ticker for r in ctx.releases[VALIDATION]})
    chosen = next(m for m in v["models"] if m["label"] == v["chosen"])
    rows = []
    for m in v["models"]:
        name = report.labels.get(m["label"], m["label"])
        rows.append(
            [
                name + (" (chosen)" if m["label"] == v["chosen"] else ""),
                str(m["step"]),
                f"{m['examples_seen']:,}",
                servereport.percent(m["correct"], m["graded"]),
                *(servereport.fraction(m["by_field"][f]) for f in FIELD_NAMES),
            ]
        )
    gpu = report.results["validation"]["gpu"]
    return [
        "## Training",
        "",
        (
            f"- **Data.** {selection['examples']:,} releases of the training companies, "
            "labelled by code from the XBRL answer key: each figure as the release "
            "prints it (the value and its table's unit), never from a model's output. "
            "A release with any figure that couldn't be labelled that way was left "
            f"out, so no label is null. {sum(selection['within_max_tokens'].values()):,} "
            f"fit the {data['max_tokens']:,}-token cap (prompt and reply); "
            f"{selection['selected']:,} of them were drawn with seed {data['seed']}, "
            f"stratified by quarter kind ({kinds}), filed {training.filed[0]} to "
            f"{training.filed[1]}. {test.checkpoint} had seen the first "
            f"{training.state['examples_seen']:,} of them, in training order."
        ),
        (
            f"- **Method.** LoRA r {lora['r']}, alpha {lora['alpha']}, dropout "
            f"{lora['dropout']:g}, on {modules} of every layer; the base model frozen in "
            f"{optimizer['dtype']}{checkpointing}; thinking off, each example laid out "
            "as vLLM sends it, with the loss on the reply's tokens only. AdamW (betas "
            f"{train.ADAM_BETAS[0]:g} and {train.ADAM_BETAS[1]:g}, eps "
            f"{train.ADAM_EPS:g}), peak learning rate {optimizer['learning_rate']:g}, "
            f"a linear warmup over {schedule['warmup_steps']} steps, then a cosine "
            f"decay; weight decay {optimizer['weight_decay']:g}; gradient norm clipped "
            f"at {optimizer['max_grad_norm']:g}. One epoch of {schedule['steps']} "
            f"steps of {schedule['examples_per_step']} examples (gradient "
            "accumulation), a checkpoint every "
            f"{optimizer['save_every']} steps.{loss_text}"
        ),
        (
            f"- **Hardware.** One {report.results['gpu_host']} {gpu}, in the session "
            "that also scored the checkpoints on validation."
        ),
        (
            f"- **Checkpoint choice, on validation only.** Every checkpoint was scored "
            f"on the {v['releases']} validation releases of {companies} companies "
            f"({chosen['graded']} graded figures), with the base model for reference, "
            f"never chosen; the rule: {score.CHOICE_RULE}. {test.checkpoint}'s score "
            f"is {testreport.CHOSEN_ON_VALIDATION}."
        ),
        "",
        *table(
            ["model", "step", "examples seen", "correct / graded", *FIELD_NAMES], rows
        ),
        "",
        (
            f"`{train.ADAPTER_CONFIG}` is the checkpoint's own with two values "
            f"changed: `{BASE_KEY}` named the training instance's local copy of the "
            f"base model and is now `{test.base}`, and `{REVISION_KEY}` is now "
            f"`{test.revision}`. Every other key is as saved."
        ),
        "",
    ]


def protocol_lines(report: Report) -> list[str]:
    labels, under, base = report.labels, report.under, report.base
    protocol, draws = report.results["protocol"], report.results["bootstrap"]
    deviations = testreport.render_deviations(report.results)[1:]  # no heading
    return [
        "## Evaluation protocol",
        "",
        (
            "The analysis was fixed before any test call, in "
            f"[{protocol['path']}]({github(protocol['path'])}) (commit "
            f"{protocol['commit'][:7]}, sha256 {protocol['sha256'][:8]}…), with the "
            "company test's releases frozen by sha256. Primary result: on the company "
            f"test, {labels[under]}'s accuracy minus {labels[base]}'s over all kept "
            "releases, with a 95% bootstrap interval over companies "
            f"({draws['draws']:,} draws, numpy default_rng({draws['seed']})); "
            "everything else is descriptive. No checkpoint or setting was chosen on a "
            "test set."
        ),
        "",
        (
            f"Deviations, as [{RESULTS_MD}]({github(RESULTS_MD)}) records them (the "
            "section names in parentheses are that report's):"
        ),
        *deviations,
    ]


def serving_lines(ctx: Context, results: Mapping[str, Any]) -> list[str]:
    session, settings = results["session"], results["settings"]
    price = session["gpu_hourly_usd"]
    texts = variant_texts(ctx, results)
    checkpoint = results["checkpoint"]
    rows = []
    for name, v in results["variants"].items():
        startup = v.get("startup") or {}
        best = v.get("best") or {}
        accuracy = (v.get("accuracy") or {}).get(VALIDATION)
        agreement = ((v.get("agreement") or {}).get(VALIDATION) or {}).get(
            "with_adapter"
        )
        rows.append(
            [
                name,
                texts.get(name, ""),
                v.get("status") or "not run",
                servereport.fmt(startup.get("weights_gib"), ".2f"),
                servereport.fmt(startup.get("kv_cache_tokens"), ","),
                f"{servereport.fmt(best['releases_per_hour'], ',.0f')} ({best['level']})"
                if best
                else "–",
                servereport.fmt(best.get("latency_p95_s"), ".2f"),
                dollars(best.get("cost_per_1000_usd")),
                dollars(best.get("cost_per_1000_at_1_usd")),
                servereport.percent(accuracy["correct"], accuracy["graded"])
                if accuracy
                else "–",
                "(itself)"
                if name == serving.ADAPTER
                else servereport.agreement_text(agreement),
            ]
        )
    # The link only when the 4-bit model has a card (its variant is complete).
    int4 = (
        f" The {serving.INT4_AWQ} model is on the Hub as "
        f"[{ctx.repo_id(INT4)}]({ctx.hub_url(INT4)})."
        if int4_problem(results) is None
        else ""
    )
    return [
        "## Serving",
        "",
        (
            f"From the serving benchmark ([{SERVING_MD}]({github(SERVING_MD)})): "
            f"{gpu_names(session.get('gpus'), prose=True)} at ${price:g} per GPU-hour, "
            f"{session['server']}, "
            f"prefix caching {'on' if settings['prefix_caching'] else 'off'}, "
            f"--max-model-len {settings['max_model_len']:,}; vLLM's defaults otherwise. "
            f"Load test: the {SPLIT_TITLES.get(settings['load_split'], settings['load_split']).lower()}'s "
            f"prompts, {settings['prompts']} per level after {settings['warmup']} "
            f"warm-up requests, at {words([str(x) for x in settings['levels']])} "
            "requests in flight; the best level is the one with the most releases per "
            "hour, and a cost per 1,000 releases is the GPU's price for the level's "
            "makespan. Same figures as adapter: the validation figures "
            f"({FIGURES_PER_RELEASE} per release) equal to the adapter variant's. "
            f"Validation is where {checkpoint} was chosen, so every variant's "
            "validation score may overstate: it compares the variants, and the test "
            f"results above give the level.{int4}"
        ),
        "",
        *table(
            [
                "variant",
                "serves",
                "status",
                "weights (GiB)",
                "KV cache (tokens)",
                "best releases/h (in flight)",
                "latency p95 there (s)",
                f"$/1,000 releases at ${price:g}/h",
                "at $1/h",
                f"validation correct ({checkpoint} chosen on it)",
                "same figures as adapter",
            ],
            rows,
        ),
        "",
    ]


def limitation_lines(ctx: Context, report: Report, training: Training) -> list[str]:
    labels, under = report.labels, report.under
    low, high = report.trails_api_on_time_test()
    common = report.split(TIME_TEST)["miss_summary"]
    t = report.results["time_test"]
    limit = chat.MAX_MODEL_LEN - chat.MAX_TOKENS
    left_out = sum(not r.kept for s in TEST_SPLITS for r in ctx.releases[s])
    steps = sorted(m["step"] for m in training.validation["models"] if m["step"])
    require(
        bool(steps) and training.state["step"] == steps[0],
        f"{labels[under]} is the first checkpoint saved",
    )
    nulls = sum(report.split(s)["abstention"][under]["null"] for s in TEST_SPLITS)
    time_cov = report.split(TIME_TEST)["coverage"]
    company_cov = report.split(COMPANY_TEST)["coverage"]
    gaps = (
        f"no XBRL answer for {time_cov['missing']:,} of the {time_cov['expected']:,} "
        "figures the time test's releases should report "
        f"({num(100 * time_cov['missing'] / time_cov['expected'], 1)}%), and for "
        f"{company_cov['missing']:,} of {company_cov['expected']:,} on the company "
        f"test ({num(100 * company_cov['missing'] / company_cov['expected'], 1)}%)"
    )
    return [
        "## Limitations",
        "",
        (
            "- **Behind the API models on the time test,** by "
            f"{num(low)} to {num(high)} points; its case is cost."
        ),
        (
            f"- **The common shareholders' line:** {common['groups'].get(COMMON, 0)} of "
            f"its {common['misses']} time-test misses are net income read from a line "
            "after preferred dividends or participating securities (Where it fails). "
            "Check net income where a release prints such a line."
        ),
        (
            "- **No nulls in training:** a release was labelled only when every figure "
            "it should report is printed, so no training label is null, and "
            + (
                "the adapter never answered null on the test sets."
                if nulls == 0
                else f"the adapter answered null for only {nulls} test figures."
            )
        ),
        f"- **Answer-key gaps:** {gaps}; those figures aren't graded.",
        (
            "- **Fourth-quarter EPS isn't graded:** XBRL has no fourth-quarter value, "
            "and EPS can't be derived by subtraction."
        ),
        (
            "- **One training run, untuned:** its settings were fixed before training, "
            f"and the chosen checkpoint is the first one saved (step {steps[0]} of "
            f"{steps[-1]})."
        ),
        (
            "- **Survivorship:** the companies are the S&P 500's members on one date, "
            "financial companies excluded, so companies that have left the index are "
            "absent."
        ),
        (
            f"- **No Q3 2026 releases in the time test:** it holds releases filed "
            f"{t['first_filed']} to {t['last_filed']}; its cutoff was set before all "
            "the Q3 2026 10-Qs, which give the answer key, were filed (a deviation "
            "from the protocol)."
        ),
        (
            f"- **Long releases:** prompts over {limit:,} tokens were left out "
            f"({left_out} test releases), so the adapter is untested on them."
        ),
        (
            "- **Scope:** English press releases of US companies reporting under US "
            "GAAP, read with the prompt it was trained on; other documents, languages, "
            "accounting standards or prompts are untested."
        ),
        (
            "- **Not financial advice:** a figure can be wrong; check it against the "
            "filing before relying on it."
        ),
        "",
    ]


def build_model(ctx: Context, out: Path) -> Written:
    """The LoRA adapter's repository folder (see the module docstring). Everything is
    read and checked before any file is written."""
    sources = ctx.sources
    test = test_config(sources)
    weights = sources.checkpoint / train.ADAPTER_WEIGHTS
    check_sha256(weights, test.adapter_sha256)
    saved = hfcommon.read_json(sources.checkpoint / train.ADAPTER_CONFIG)
    training = load_training(sources)
    check_lora(saved, training.lora)
    try:
        prompt = extract.load_prompt(test.prompt, sources.prompts)
    except extract.RunError as exc:
        raise ExportError(str(exc)) from None
    if not sources.test_chart.is_file():
        raise ExportError(f"{sources.test_chart} not found")
    serving_results = load_serving(sources, test)
    card = model_card(ctx, test, training, prompt.text, serving_results)

    files = [
        out / train.ADAPTER_WEIGHTS,
        out / train.ADAPTER_CONFIG,
        out / f"{test.prompt}.txt",
        out / CHART,
        out / README,
    ]
    hfcommon.copy_file(weights, files[0])
    check_sha256(files[0], test.adapter_sha256)
    hfcommon.write_json(files[1], adapter_config(saved, test))
    hfcommon.copy_file(sources.prompts / f"{test.prompt}.txt", files[2])
    hfcommon.copy_file(sources.test_chart, files[3])
    hfcommon.write_text(files[4], card)
    return Written(files, [] if serving_results else [NO_SERVING_SECTION])


# The 4-bit card


def build_int4(ctx: Context, out: Path) -> Written:
    """out/README.md, the card of the 4-bit AWQ model, once the serving results exist,
    its variant completed with its build manifest (int4_problem);
    otherwise nothing, with the reason."""
    test = test_config(ctx.sources)
    results = load_serving(ctx.sources, test)
    if results is None:
        return Written(skipped=[NO_INT4_CARD])
    problem = int4_problem(results)
    if problem:
        return Written(skipped=[problem])
    card = int4_card(ctx, test, results)
    path = out / README
    hfcommon.write_text(path, card)
    return Written([path])


def quant_weights(config: Mapping[str, Any] | None) -> str | None:
    """The weights' format from the saved quantization config, if it says: '4-bit
    int weights with zero points, in groups of 128'."""
    found = set()
    for group in ((config or {}).get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        if weights.get("num_bits") is None:
            continue
        text = f"{weights['num_bits']}-bit"
        if weights.get("type"):
            text += f" {weights['type']}"
        text += " weights"
        if weights.get("symmetric") is False:
            text += " with zero points"
        if weights.get("group_size"):
            text += f", in groups of {weights['group_size']}"
        if (group or {}).get("input_activations") is None:
            text += ", activations not quantized"
        found.add(text)
    return "; ".join(sorted(found)) or None


def int4_card(ctx: Context, test: score.TestConfig, results: Mapping[str, Any]) -> str:
    repo, adapter_repo = ctx.repo_id(INT4), ctx.repo_id(MODEL)
    session, settings = results["session"], results["settings"]
    variants = results["variants"]
    mine = variants[serving.INT4_AWQ]
    version = vllm_version(session["server"])
    command = ["vllm", "serve", repo, "--max-model-len", str(settings["max_model_len"])]
    snippet = request_snippet(
        adapter_repo, f"{test.prompt}.txt", repo, request_parameters(test)
    )
    metadata = {
        "base_model": adapter_repo,
        "base_model_relation": "quantized",
        "library_name": "transformers",
        "license": "apache-2.0",
        "language": ["en"],
        "pipeline_tag": "text-generation",
        "tags": list(INT4_TAGS),
        "datasets": [ctx.repo_id(DATASET)],
    }
    lines = [
        *front_matter(metadata),
        f"# {hfcommon.REPO_NAMES[INT4]}",
        "",
        (
            f"The LoRA adapter [{adapter_repo}]({ctx.hub_url(MODEL)}) ({test.checkpoint} "
            f"of {test.run}) merged into [{test.base}](https://huggingface.co/"
            f"{test.base}) and quantized to 4-bit weights with llm-compressor's AWQ, "
            "for serving with vLLM. It reads a US company's earnings press release and "
            "returns its GAAP revenue, net income and diluted EPS as one JSON object. "
            "The adapter's card describes the task, the reply format, the training and "
            "the pre-registered test results; this card has the 4-bit model's own "
            "numbers from the serving benchmark."
        ),
        "",
        links_line(ctx),
        "",
        "## What it is",
        "",
        *int4_build_lines(test, results),
        "",
        "## How to use",
        "",
        (
            f"Serve it with vLLM {version} on Linux with an NVIDIA GPU. "
            f"{driver_text(version)}"
        ),
        "",
        "```sh",
        f"pip install vllm=={version}",
        *shell_lines(command),
        "```",
        "",
        (
            "Then send each release as the adapter's card does, with this repository "
            f"as the model (the prompt, `{test.prompt}.txt`, is in the adapter's "
            f"repository; `{RELEASE_FILE}` is one press release as Markdown):"
        ),
        "",
        "```python",
        *snippet,
        "```",
        "",
        *int4_results_lines(ctx, results, mine),
        "## Limitations",
        "",
        *int4_limitation_lines(ctx, results, mine),
        *closing_lines(ctx, test.base),
    ]
    return "\n".join(lines).rstrip() + "\n"


def int4_build_lines(test: score.TestConfig, results: Mapping[str, Any]) -> list[str]:
    built = results["models"][serving.INT4_AWQ]
    merged = results["models"].get(serving.MERGED) or {}
    merged_settings = merged.get("settings") or {}
    merged_versions = merged.get("versions") or {}
    merge = (
        f"- **Merge:** the adapter merged into {test.base} (revision "
        f"`{test.revision}`) with peft's merge_and_unload"
    )
    if merged_settings.get("merge_dtype") and merged_settings.get("save_dtype"):
        merge += (
            f", in {merged_settings['merge_dtype']}, then saved in "
            f"{merged_settings['save_dtype']}"
        )
    if merged_versions.get("peft"):
        merge += f" (peft {merged_versions['peft']})"
    if merged.get("weight_gib") is not None:
        merge += f"; {merged['weight_gib']:.2f} GiB of weights"
    s = built.get("settings") or {}
    versions = built.get("versions") or {}
    libraries = words(  # quantize.versions records None for a package it didn't find
        [
            f"{n} {versions[n]}"
            for n in ("llmcompressor", "compressed-tensors")
            if versions.get(n)
        ]
    )
    quantize = "- **Quantization:** llm-compressor's AWQ" + (
        f" ({libraries})" if libraries else ""
    )
    if s.get("duo_scaling") is not None:
        quantize += f", duo_scaling {s['duo_scaling']}"
    if s.get("scheme"):
        quantize += f", then the {s['scheme']} scheme"
    if s.get("targets"):
        quantize += f" on the {words(list(s['targets']))} layers"
    if s.get("ignore"):
        quantize += f" except {words(list(s['ignore']))}"
    weights = quant_weights(built.get("quantization_config"))
    if weights:
        quantize += f": {weights}"
    config = built.get("quantization_config") or {}
    if config.get("format"):
        quantize += f"; saved in compressed-tensors' {config['format']} format"
    if built.get("weight_gib") is not None:
        quantize += f", {built['weight_gib']:.2f} GiB of weights"
    lines = [merge + ".", quantize + "."]
    if s.get("num_calibration_samples"):
        calibration = (
            f"- **Calibration:** {s['num_calibration_samples']} examples of the "
            "training data (prompt and reply, laid out as in training)"
        )
        if s.get("max_seq_length"):
            calibration += f", at most {s['max_seq_length']:,} tokens each"
        calibration += (
            ", drawn with a fixed seed; none from a validation or test split."
        )
        if s.get("cache_offload"):
            calibration += (
                f" AWQ ran out of GPU memory and ran again with its caches on the "
                f"{s['cache_offload'].upper()} (slower, the same result)."
            )
        lines.append(calibration)
    lines.append(
        "- **Files:** a transformers model folder (config.json with its "
        "quantization_config, the weights as safetensors in compressed-tensors' "
        "format), hence `library_name: transformers`: vLLM serves it as it is, and "
        "transformers loads it with compressed-tensors installed."
    )
    return lines


def int4_results_lines(
    ctx: Context, results: Mapping[str, Any], mine: Mapping[str, Any]
) -> list[str]:
    session, settings = results["session"], results["settings"]
    price = session["gpu_hourly_usd"]
    checkpoint = results["checkpoint"]
    variants = results["variants"]
    columns = [serving.INT4_AWQ] + (
        [serving.ADAPTER] if serving.ADAPTER in variants else []
    )
    adapter = (
        f" The {serving.ADAPTER} column is {variant_texts(ctx, results)[serving.ADAPTER]}."
        if serving.ADAPTER in columns
        else ""
    )

    def accuracy(variant: str, split: str) -> str:
        found = (variants[variant].get("accuracy") or {}).get(split)
        return servereport.percent(found["correct"], found["graded"]) if found else "–"

    def agreement(variant: str, split: str, key: str) -> str:
        if key == "with_adapter" and variant == serving.ADAPTER:
            return "(itself)"
        found = ((variants[variant].get("agreement") or {}).get(split) or {}).get(key)
        return servereport.agreement_text(found)

    def best(variant: str, key: str, spec: str) -> str:
        found = variants[variant].get("best") or {}
        if key.startswith("cost"):
            return dollars(found.get(key))
        return servereport.fmt(found.get(key), spec)

    def startup(variant: str, key: str, spec: str) -> str:
        return servereport.fmt((variants[variant].get("startup") or {}).get(key), spec)

    def disk(variant: str) -> str:
        model = serving.MODEL_OF.get(variant)
        found = (results["models"].get(model) or {}) if model else {}
        return servereport.fmt(found.get("weight_gib"), ".2f")

    max_tokens = (mine.get("startup") or {}).get("max_concurrency_tokens")
    rows = [
        [
            f"validation ({checkpoint} chosen on it), correct / graded",
            *(accuracy(v, VALIDATION) for v in columns),
        ],
        [
            "time test, correct / graded (descriptive)",
            *(accuracy(v, TIME_TEST) for v in columns),
        ],
        *(
            [
                f"{split_phrase(split).removeprefix('the ')} figures equal to {who}",
                *(agreement(v, split, key) for v in columns),
            ]
            for key, who in (
                ("with_adapter", "the adapter variant's"),
                ("with_stage2", "stage 2's run of the adapter"),
            )
            for split in (VALIDATION, TIME_TEST)
        ),
        [
            "best throughput, releases/h",
            *(best(v, "releases_per_hour", ",.0f") for v in columns),
        ],
        ["at requests in flight", *(best(v, "level", "") for v in columns)],
        ["latency p95 there (s)", *(best(v, "latency_p95_s", ".2f") for v in columns)],
        [
            f"$/1,000 releases there, at ${price:g}/h",
            *(best(v, "cost_per_1000_usd", "") for v in columns),
        ],
        ["at $1/h", *(best(v, "cost_per_1000_at_1_usd", "") for v in columns)],
        [
            "weights in GPU memory (GiB)",
            *(startup(v, "weights_gib", ".2f") for v in columns),
        ],
        ["KV cache (GiB)", *(startup(v, "kv_cache_gib", ".2f") for v in columns)],
        ["KV cache (tokens)", *(startup(v, "kv_cache_tokens", ",") for v in columns)],
        [
            f"max concurrency at {servereport.fmt(max_tokens, ',')} tokens",
            *(startup(v, "max_concurrency", ".2f") for v in columns),
        ],
        [
            "launch to healthy (s)",
            *(startup(v, "launch_to_healthy_s", ".0f") for v in columns),
        ],
        ["model on disk (GiB)", *(disk(v) for v in columns)],
    ]
    levels = [
        [
            str(x["level"]),
            f"{x['ok']}/{x['requests']}",
            servereport.fmt(x.get("releases_per_hour"), ",.0f"),
            " / ".join(
                servereport.fmt((x.get("latency_s") or {}).get(p), ".2f")
                for p in ("p50", "p95")
            ),
            servereport.fmt((x.get("ttft_s") or {}).get("p50"), ".2f"),
            servereport.fmt(x.get("decode_tokens_per_s_p50"), ",.0f"),
            dollars(x.get("cost_per_1000_usd")),
            dollars(x.get("cost_per_1000_at_1_usd")),
        ]
        for x in mine.get("levels") or []
    ]
    return [
        "## Results",
        "",
        (
            f"From the serving benchmark ([{SERVING_MD}]({github(SERVING_MD)})): "
            f"{gpu_names(session.get('gpus'), prose=True)} at ${price:g} per GPU-hour, "
            f"{session['server']}, "
            f"prefix caching {'on' if settings['prefix_caching'] else 'off'}, "
            f"--max-model-len {settings['max_model_len']:,}; vLLM's defaults otherwise."
            f"{adapter} Accuracy is correct over graded figures, as in the benchmark. "
            f"Validation is where {checkpoint} was chosen, so every validation score "
            "below may overstate: it compares the variants, and the time test gives "
            "the level (descriptive here, as it is in the adapter's card)."
            f"{stage2_scores(results)} Agreement counts the predicted figures "
            f"({FIGURES_PER_RELEASE} per release; null equals null) equal to another "
            "run's."
        ),
        "",
        *table(["measure", *columns], rows),
        "",
        (
            f"{serving.INT4_AWQ} per level (the "
            f"{SPLIT_TITLES.get(settings['load_split'], settings['load_split']).lower()}'s "
            f"prompts, {settings['prompts']} per level after {settings['warmup']} "
            "warm-up requests; a cost per 1,000 releases is the GPU's price for the "
            "level's makespan):"
        ),
        "",
        *table(
            [
                "in flight",
                "ok",
                "releases/h",
                "latency p50 / p95 (s)",
                "first token p50 (s)",
                "decode tok/s (median)",
                f"$/1,000 at ${price:g}/h",
                "at $1/h",
            ],
            levels,
        ),
        "",
    ]


def stage2_scores(results: Mapping[str, Any]) -> str:
    """Stage 2's own runs of the adapter on the serving benchmark's splits, the
    validation score labelled as the protocol requires."""
    reference = results.get("reference_accuracy") or {}
    scores = []
    for split in hfcommon.SPLIT_NAMES:
        r = reference.get(split)
        if not r:
            continue
        text = (
            f"{servereport.percent(r['correct'], r['graded'])} on {split_phrase(split)}"
        )
        scores.append(text + (", chosen on validation" if split == VALIDATION else ""))
    if not scores:
        return ""
    return f" Stage 2's runs of the adapter scored {', and '.join(scores)}."


def int4_limitation_lines(
    ctx: Context, results: Mapping[str, Any], mine: Mapping[str, Any]
) -> list[str]:
    found = {
        split: ((mine.get("agreement") or {}).get(split) or {}).get("with_adapter")
        for split in (VALIDATION, TIME_TEST)
    }
    same = " and ".join(
        f"{servereport.agreement_text(a)} on {split_phrase(split)}"
        for split, a in found.items()
    )
    changed = any(a and a["same"] < a["figures"] for a in found.values())
    s = (results["models"][serving.INT4_AWQ].get("settings")) or {}
    return [
        (
            f"- Every limitation in [the adapter's card]({ctx.hub_url(MODEL)}) applies "
            "to this model too."
        ),
        (
            (
                "- **Quantization changes some predictions:**"
                if changed
                else "- **Quantization may change predictions:**"
            )
            + f" the figures equal to the adapter variant's are {same}."
        ),
        (
            "- **One session on one GPU type:** throughput, latency and cost depend on "
            "the GPU, the vLLM version and its settings."
        ),
        (
            "- **Calibrated on training releases only**"
            + (
                f" ({s['num_calibration_samples']} examples)"
                if s.get("num_calibration_samples")
                else ""
            )
            + "."
        ),
        "",
    ]
