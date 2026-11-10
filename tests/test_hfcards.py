"""Tests for secbench.hfcards, the model repositories' folders of the Hugging Face
export, on the small repository of tests/hf_fixture.py; the last ones build the real
model card and skip without the real data (runs/, training/ and data/ aren't in the
repository)."""

from __future__ import annotations

import dataclasses
import json
import re
import runpy
import shlex
import shutil
import sys
import tomllib
import types
from collections.abc import Callable
from pathlib import Path

import hf_fixture
import httpx
import pytest

from secbench import extract, hfcards, hfcommon, score, servereport, splits
from secbench.hfcommon import FIELD_NAMES, ExportError, Sources
from secbench.testreport import acc_cell, bounds, frac, money, num

REPO = Path(__file__).resolve().parents[1]
FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "extract_v2.txt",
    "test_accuracy_vs_cost.png",
    "README.md",
]
SERVING_NOTE = "model card's serving section: results/serving_results.json not found"
CHART = "test_accuracy_vs_cost.png"
INT4_NOTE = "int4-awq card: results/serving_results.json not found"
INT4_REPO = "test-user/qwen3-8b-sec-earnings-int4-awq"


@pytest.fixture(scope="module")
def built(hf_context, tmp_path_factory) -> tuple[Path, hfcommon.Written]:
    out = tmp_path_factory.mktemp("cards") / "model"
    return out, hfcards.build_model(hf_context, out)


@pytest.fixture(scope="module")
def card(built) -> str:
    return (built[0] / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def serving_repo(hf_repo, tmp_path_factory):
    """A copy of the mini repo with results/serving_results.json, its context and the
    serving results."""
    root = tmp_path_factory.mktemp("serving") / "repo"
    shutil.copytree(hf_repo.root, root)
    data = hf_fixture.add_serving_results(root)
    return hfcommon.load_context(Sources(root), "test-user"), data


@pytest.fixture(scope="module")
def serving_cards(serving_repo, tmp_path_factory):
    context, _ = serving_repo
    out = tmp_path_factory.mktemp("serving-cards")
    model = hfcards.build_model(context, out / "model")
    int4 = hfcards.build_int4(context, out / "int4-awq")
    return out, model, int4


def front_matter(text: str) -> dict:
    yaml = pytest.importorskip("yaml")
    match = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "no front matter"
    return yaml.safe_load(match[1])


def code_block(text: str, language: str, containing: str) -> str:
    found = [
        block
        for block in re.findall(
            rf"^```{language}\n(.*?)^```$", text, re.MULTILINE | re.DOTALL
        )
        if containing in block
    ]
    assert len(found) == 1, f"{len(found)} {language} blocks with {containing!r}"
    return found[0]


def section(text: str, heading: str) -> str:
    return text.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


def shell_command(block: str) -> tuple[list[str], list[str]]:
    """The pip line and the command, from a sh block."""
    pip, *rest = block.strip().splitlines()
    return shlex.split(pip), shlex.split("\n".join(rest).replace("\\\n", " "))


def read_test_config(root: Path) -> score.TestConfig:
    return score.load_test_config(root / "configs" / "test" / "qwen3-8b-2k.toml")


# The model folder


def test_the_model_folder_holds_the_adapter_and_its_card(built, hf_repo):
    out, written = built
    assert [p.name for p in written.files] == FILES
    assert all(p.parent == out for p in written.files)
    assert sorted(p.name for p in out.iterdir()) == sorted(FILES)
    assert written.skipped == [SERVING_NOTE]
    checkpoint = (
        hf_repo.root / "training" / "qwen3-8b-2k" / "checkpoints" / "step-00025"
    )
    weights = out / "adapter_model.safetensors"
    assert (
        weights.read_bytes() == (checkpoint / "adapter_model.safetensors").read_bytes()
    )
    assert hfcommon.sha256_file(weights) == hf_repo.adapter_sha256
    assert hf_repo.adapter_sha256 == read_test_config(hf_repo.root).adapter_sha256
    for name, source in (
        ("extract_v2.txt", hf_repo.root / "prompts" / "extract_v2.txt"),
        ("test_accuracy_vs_cost.png", hf_repo.root / "results" / CHART),
    ):
        assert (out / name).read_bytes() == source.read_bytes()


def test_adapter_config_changes_exactly_two_keys(built, hf_repo):
    out, _ = built
    checkpoint = (
        hf_repo.root / "training" / "qwen3-8b-2k" / "checkpoints" / "step-00025"
    )
    saved = json.loads((checkpoint / "adapter_config.json").read_text())
    text = (out / "adapter_config.json").read_text()
    mine = json.loads(text)
    assert list(mine) == list(saved)
    assert {k for k in saved if saved[k] != mine[k]} == {
        "base_model_name_or_path",
        "revision",
    }
    test = read_test_config(hf_repo.root)
    assert (mine["base_model_name_or_path"], mine["revision"]) == (
        "Qwen/Qwen3-8B",
        test.revision,
    )
    assert text == json.dumps(mine, indent=2) + "\n"
    assert "/workspace" not in text


def test_a_saved_config_without_the_keys_stops(hf_repo):
    test = read_test_config(hf_repo.root)
    with pytest.raises(ExportError, match="no revision"):
        hfcards.adapter_config({"base_model_name_or_path": "x", "r": 16}, test)


def test_an_adapter_with_another_sha256_stops_the_build(hf_repo_copy, tmp_path):
    weights = (
        hf_repo_copy.root
        / "training"
        / "qwen3-8b-2k"
        / "checkpoints"
        / "step-00025"
        / "adapter_model.safetensors"
    )
    weights.write_bytes(weights.read_bytes() + b"\0")
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "test-user")
    with pytest.raises(ExportError, match="sha256"):
        hfcards.build_model(context, tmp_path / "model")
    assert not (tmp_path / "model").exists()  # nothing written


def test_an_adapter_unlike_the_runs_config_stops_the_build(hf_repo_copy, tmp_path):
    path = (
        hf_repo_copy.root
        / "training"
        / "qwen3-8b-2k"
        / "checkpoints"
        / "step-00025"
        / "adapter_config.json"
    )
    saved = json.loads(path.read_text())
    path.write_text(json.dumps({**saved, "lora_alpha": 64}, indent=2))
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "test-user")
    with pytest.raises(ExportError, match="LoRA"):
        hfcards.build_model(context, tmp_path / "model")


def test_two_builds_are_byte_identical(hf_context, built, tmp_path):
    out, _ = built
    again = tmp_path / "model"
    hfcards.build_model(hf_context, again)
    for name in FILES:
        assert (again / name).read_bytes() == (out / name).read_bytes(), name


# The card's metadata


def test_metadata(card, hf_repo):
    meta = front_matter(card)
    results = hf_repo.test_results
    assert {k: v for k, v in meta.items() if k != "model-index"} == {
        "base_model": "Qwen/Qwen3-8B",
        "base_model_relation": "adapter",
        "library_name": "peft",
        "license": "apache-2.0",
        "language": ["en"],
        "pipeline_tag": "text-generation",
        "tags": [
            "lora",
            "peft",
            "qwen3",
            "finance",
            "sec",
            "xbrl",
            "earnings",
            "information-extraction",
        ],
        "datasets": ["test-user/sec-earnings-bench"],
    }
    (index,) = meta["model-index"]
    assert index["name"] == "qwen3-8b-sec-earnings-lora"
    found = []
    for result in index["results"]:
        assert result["task"] == {"type": "text-generation"}
        dataset = result["dataset"]
        assert (dataset["name"], dataset["type"], dataset["config"]) == (
            "test-user/sec-earnings-bench",
            "test-user/sec-earnings-bench",
            "answers",
        )
        (metric,) = result["metrics"]
        assert metric["type"] == "accuracy"
        assert "graded figures correct" in metric["name"]
        found.append((dataset["split"], metric["value"]))
    under = results["under_test"]
    assert found == [
        (
            split,
            round(results["splits"][split]["accuracy"][under]["all"]["accuracy"], 2),
        )
        for split in ("company_test", "time_test")
    ]
    assert found == [("company_test", 99.24), ("time_test", 97.64)]


def test_the_hub_reads_the_model_index(card):
    hub = pytest.importorskip("huggingface_hub")
    data = hub.ModelCard(card).data
    assert data.base_model == "Qwen/Qwen3-8B"
    assert data.library_name == "peft"
    assert [
        (r.dataset_split, r.dataset_config, r.metric_type, r.metric_value)
        for r in data.eval_results
    ] == [
        ("company_test", "answers", "accuracy", 99.24),
        ("time_test", "answers", "accuracy", 97.64),
    ]
    assert all(
        r.source_url.endswith("results/test_results.md") for r in data.eval_results
    )


@pytest.mark.parametrize(
    "value, shown",
    [
        ("apache-2.0", "apache-2.0"),
        ("HF_USER/sec-earnings-bench", "HF_USER/sec-earnings-bench"),
        ("4-bit", '"4-bit"'),
        ("yes", '"yes"'),
        ("2026-11-09", '"2026-11-09"'),
        ("a: b # c", '"a: b # c"'),
        ("https://x.example/a", '"https://x.example/a"'),
        (99.24, "99.24"),
        (True, "true"),
    ],
)
def test_yaml_scalars_read_back(value, shown):
    yaml = pytest.importorskip("yaml")
    assert hfcards.yaml_scalar(value) == shown
    assert yaml.safe_load(f"key: {shown}") == {"key": value}


def test_front_matter_reads_back():
    yaml = pytest.importorskip("yaml")
    metadata = {
        "a": "x",
        "tags": ["4-bit", "yes"],
        "empty": [],
        "none": {},
        "index": [{"name": "n", "results": [{"task": {"type": "t"}, "v": 1.5}]}],
    }
    lines = hfcards.front_matter(metadata)
    assert lines[0] == "---" and lines[-2:] == ["---", ""]
    assert yaml.safe_load("\n".join(lines[1:-2])) == metadata


# The card's numbers


def difference(entry: dict) -> str:
    return f"{num(entry['difference'], signed=True)} {bounds(entry['interval'], signed=True)}"


def test_headline_numbers_are_the_reports(card, hf_repo):
    r = hf_repo.test_results
    labels, under = r["models"], r["under_test"]
    company, time_test = r["splits"]["company_test"], r["splits"]["time_test"]
    primary = company["differences"]["base"]["all"]
    assert (
        f"**{num(primary['difference'], signed=True)} points "
        f"{bounds(primary['interval'], signed=True)}**" in card
    )
    assert "**+10.12 points [+5.08, +15.58]**" in card
    assert (
        f"{primary['discordant']} discordant figures ({primary['first_right']} fixed by "
        f"training, {primary['second_right']} broken) from {primary['companies']} "
        "companies" in card
    )
    results = section(card, "Results")
    for model, label in labels.items():
        row = next(
            line for line in results.splitlines() if line.startswith(f"| {label}")
        )
        for s in (company, time_test):
            assert acc_cell(s["accuracy"][model]["all"]) in row
            assert money(s["cost"][model]["per_1000"]) in row
        if model != under:
            assert difference(company["differences"][model]["all"]) in results
            assert difference(time_test["differences"][model]["all"]) in results
    assert "657/662 (99.24%) [98.05, 100.00]" in card
    assert "1,699/1,740 (97.64%) [96.64, 98.57]" in card
    freeze = company["freeze"]
    assert (
        f"about {money(freeze['excluded_per_1000'])} without the instance freeze"
        in card
    )
    correct, graded = r["validation"]["counts"][under]["all"]
    assert (
        f"step-00025's validation score, {num(100 * correct / graded, 1)}% "
        f"({correct}/{graded}), is chosen on validation" in card
    )
    assert "98.0% (287/293), is chosen on validation" in card
    gaps = [
        -time_test["differences"][m]["all"]["difference"]
        for m in labels
        if m not in (under, "base")
    ]
    assert f"by {num(min(gaps))} to {num(max(gaps))} points" in card
    assert "![" in card and "](test_accuracy_vs_cost.png)" in card
    assert (
        "https://github.com/jinnn11/sec-earnings-bench/blob/main/results/test_results.md"
        in card
    )


def readme_text() -> str:
    """The tracked README.md as plain prose: no bold markers, one space between
    words."""
    text = (REPO / "README.md").read_text(encoding="utf-8").replace("**", "")
    return " ".join(text.split())


def test_the_company_test_standing_is_readmes(card, hf_repo):
    r = hf_repo.test_results
    company = r["splits"]["company_test"]
    mine = company["accuracy"][r["under_test"]]["all"]

    def cell(model: str, unit: str) -> str:
        d = company["differences"][model]["all"]
        low, high = (num(x, signed=True) for x in d["interval"])
        return f"{num(d['difference'], signed=True)}{unit}, interval {low} to {high}"

    expected = (
        f"On the company test it made {mine['graded'] - mine['correct']} errors where "
        "GPT-6.1 Sol and Claude Haiku 5.5 made none "
        f"({cell('gpt-6.1-sol', ' points')}), and it is level with GPT-6 Luna "
        f"({cell('gpt-6-luna', '')})."
    )
    assert cell("gpt-6.1-sol", "") == cell("claude-haiku-5-5", "")
    assert expected in card
    assert expected in readme_text()  # README's own sentence
    assert "every interval against an API model includes 0" not in card


def test_the_company_test_standing_groups_the_api_models(hf_context):
    results = json.loads(json.dumps(hf_context.test_results))
    company = results["splits"]["company_test"]
    company["accuracy"]["gpt-6.1-sol"]["all"]["correct"] -= 1  # no longer perfect
    text = hfcards.company_test_standing(hfcards.Report(results))
    assert text == (
        "On the company test it made 5 errors where Claude Haiku 5.5 made none "
        "(−0.76 points, interval −1.95 to 0.00), and it is level with GPT-6.1 Sol "
        "and GPT-6 Luna (GPT-6.1 Sol −0.76, interval −1.95 to 0.00; GPT-6 Luna "
        "+0.45, interval −1.41 to +2.67)."
    )
    under = company["accuracy"][results["under_test"]]["all"]
    under["correct"] = under["graded"]  # no misses: level with every API model
    text = hfcards.company_test_standing(hfcards.Report(results))
    assert text.startswith("On the company test it is level with GPT-6.1 Sol, ")
    assert "errors" not in text and "(GPT-6.1 Sol −0.76 points, " in text


def test_failures_are_the_reports(card, hf_repo):
    r = hf_repo.test_results
    labels = r["models"]
    text = section(card, "Where it fails")
    for name in ("company_test", "time_test"):
        s = r["splits"][name]
        summary = s["miss_summary"]
        common = summary["groups"]["common shareholders' line"]
        assert f"{common} of its {summary['misses']} misses" in text
        assert ", ".join(summary["common_companies"]) in text
        assert (
            f"{summary['broken_common']} of the {summary['broken']} figures training broke"
            in text
        )
        for model, label in labels.items():
            assert acc_cell(s["accuracy"][model]["without_preferred"]) in text
            c = s["preferred"]["net_income"][model]
            assert f"{label} {frac(c['correct'], c['graded'])}" in text
        a = s["abstention"][r["under_test"]]
        assert f"{a['null']} of {a['figures']:,}" in text
    assert "29 of its 41 misses, from 16 companies" in text
    assert "HPE, KDP, LHX print preferred-stock lines" in text
    assert "DOC, TDG report income after participating securities" in text
    # Dropping the preferred-dividend releases understates the failure only where
    # some of these misses are outside them: the time test, not the company test.
    common = text.split("**The common shareholders' line.**", 1)[1].split("\n")[0]
    company_part, time_part = common.split(" Company test: ", 1)[1].split(
        " Time test: "
    )
    assert r["splits"]["company_test"]["miss_summary"]["common_outside_preferred"] == 0
    assert "so dropping those releases, as below, drops all of them" in company_part
    assert "understates" not in company_part
    assert (
        "so dropping the preferred-dividend releases, as below, understates this "
        "failure on the time test" in time_part
    )
    assert common.count("understates") == 1
    assert "WY 2 on the company test and TWLO 1 on the time test" in text
    assert "It never answered null." in text
    assert (
        "every not_in_input figure (38 on the company test, 52 on the time test)"
        in text
    )


def test_training_reads_the_runs_files(card, hf_repo):
    training = hf_repo.root / "training" / "qwen3-8b-2k"
    validation = json.loads((training / "validation.json").read_text())
    config = json.loads((training / "config.json").read_text())
    text = section(card, "Training")
    for m in validation["models"]:
        label = "Qwen3-8B base" if m["label"] == "base" else m["label"]
        chosen = " (chosen)" if m["label"] == validation["chosen"] else ""
        fields = " | ".join(f"{c}/{g}" for c, g in m["by_field"].values())
        assert (
            f"| {label}{chosen} | {m['step']} | {m['examples_seen']:,} | "
            f"{servereport.percent(m['correct'], m['graded'])} | {fields} |" in text
        )
    lora = config["settings"]["lora"]
    assert f"LoRA r {lora['r']}, alpha {lora['alpha']}, dropout 0, on `q_proj`" in text
    assert f"drawn with seed {config['settings']['data']['seed']}" in text
    assert "AdamW (betas 0.9 and 0.999, eps 1e-08), peak learning rate 0.0002" in text
    assert "One Vast H100 NVL" in text  # test_results.json's validation GPU
    assert "the most figures correct; ties go to the earliest step" in text
    # The fixture's train_log.jsonl has steps 1 and 2, not 1 to 25: no loss sentence.
    assert "Training loss" not in text


def test_the_loss_up_to_the_checkpoint_comes_from_train_log(hf_context):
    training = hfcards.load_training(hf_context.sources)
    losses = [0.12, *(0.01 for _ in range(19)), 0.0013, 0.0088, 0.00045, 0.00064]
    losses.append(0.00062)  # step 25's own loss: one step's examples only
    log = [{"step": i + 1, "loss": x} for i, x in enumerate(losses)]
    report = hfcards.Report(hf_context.test_results)
    test = read_test_config(hf_context.sources.root)
    text = "\n".join(
        hfcards.training_lines(
            hf_context, report, test, dataclasses.replace(training, log=log)
        )
    )
    mean = sum(losses) / 25
    assert (
        "Training loss (cross-entropy per reply token, train_log.jsonl): 0.12 at "
        f"step 1, and {mean:.2g} averaged over steps 1 to 25; a step's loss covers "
        "only its 1 example, so it is noisy (from 0.00045 to 0.0088 over steps 21 "
        "to 25)." in text
    )
    assert "0.00062" not in text  # no single step quoted as the loss
    gap = dataclasses.replace(training, log=[x for x in log if x["step"] != 7])
    assert hfcards.loss_sentence(gap, 25, 16) == ""


def test_the_protocol_and_its_deviations_are_the_reports(card, hf_repo):
    text = section(card, "Evaluation protocol")
    protocol = hf_repo.test_results["protocol"]
    assert f"commit {protocol['commit'][:7]}, sha256 {protocol['sha256'][:8]}…" in text
    assert "10,000 draws, numpy default_rng(20261105)" in text
    report = (REPO / "results" / "test_results.md").read_text()  # the tracked report
    deviations = report.split("## Deviations from the protocol\n", 1)[1].split("\n## ")[
        0
    ]
    assert deviations.strip() in text


def test_limitations(card):
    text = section(card, "Limitations")
    assert "no XBRL answer for 124 of the 1,916 figures" in text
    assert "(6.5%)" in text and "28 of 728 on the company test (3.8%)" in text
    assert f"prompts over {splits.MAX_PROMPT_TOKENS:,} tokens were left out" in text
    assert "Survivorship" in text and "Not financial advice" in text
    assert "filed 2026-07-01 to 2026-09-30" in text


def test_the_citation_is_the_dataset_cards(card, hf_context):
    """One BibTeX key, one entry: the model cards cite the project as the dataset card
    does, with the name and the acronyms braced so styles keep their capitals."""
    from secbench import hfdataset

    mine = hfcards.citation(hf_context.sources)
    theirs = hfdataset.citation_section(hf_context)
    start = theirs.index("```bibtex")
    assert mine == theirs[start : theirs.index("```", start + 1) + 1]
    entry = "\n".join(mine)
    assert entry in card
    assert "@misc{sankaran2026secearningsbench," in entry
    for word in ("{sec-earnings-bench}", "{GAAP}", "{SEC}", "{XBRL}"):
        assert word in entry
    assert entry.count("{") == entry.count("}")


@pytest.mark.parametrize(
    "change, claim",
    [
        (
            lambda r: r["splits"]["time_test"]["differences"]["gpt-6-luna"][
                "all"
            ].update(difference=0.5, interval=[-0.2, 1.1]),
            "trails GPT-6 Luna",
        ),
        (
            lambda r: r["splits"]["time_test"]["cost"]["gpt-6-luna"].update(
                per_1000=0.1
            ),
            "costs less",
        ),
        (
            lambda r: r["splits"]["company_test"]["differences"]["gpt-6-luna"][
                "all"
            ].update(interval=[0.4, 2.6]),
            "includes 0",
        ),
    ],
)
def test_a_claim_the_numbers_contradict_stops_the_card(
    hf_context, tmp_path, change, claim
):
    results = json.loads(json.dumps(hf_context.test_results))
    change(results)
    context = dataclasses.replace(hf_context, test_results=results)
    with pytest.raises(ExportError, match=claim):
        hfcards.build_model(context, tmp_path / "model")
    assert not (tmp_path / "model").exists()


# Serving the adapter and sending a release


def test_the_vllm_command_is_stage_2s(card, hf_repo):
    test = read_test_config(hf_repo.root)
    pip, command = shell_command(code_block(card, "sh", "vllm serve"))
    assert pip == ["pip", "install", "vllm==0.32.0"]
    assert set(hf_repo.test_results["model_under_test"]["server"].values()) == {
        "vLLM 0.32.0"
    }
    adapter = score.Model(
        "step-00025", "qwen3-8b-2k--step-00025", "x", 25, 1, Path("A")
    )
    stage2 = score.vllm_command(Path("/snapshot"), test.base, [adapter], 8000, 16)
    expected, words = [], iter(stage2)
    for word in words:
        if word in ("--host", "--port", "--served-model-name"):
            next(words)
            continue
        expected.append(word)
    expected[0], expected[2] = "vllm", test.base
    expected[3:3] = ["--revision", test.revision]
    expected[-1] = "qwen3-8b-2k--step-00025=test-user/qwen3-8b-sec-earnings-lora"
    assert command == expected
    for flag, value in (
        ("--max-model-len", test.max_model_len),
        ("--max-lora-rank", test.max_lora_rank),
        ("--max-loras", test.max_loras),
        ("--max-cpu-loras", test.max_cpu_loras),
    ):
        assert command[command.index(flag) + 1] == str(value)
    assert "--enable-lora" in command
    usage = section(card, "How to use")
    assert "the server needs an NVIDIA driver for CUDA 13.0 or later" in usage


def test_the_cuda_version_is_the_locked_vllms():
    """The cards' driver requirement: the CUDA runtime uv.lock installs with the vLLM
    version the test runs used (pip installs vLLM whatever the driver)."""
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])
    results = json.loads((REPO / "results" / "test_results.json").read_text())
    assert versions["vllm"] == {hfcards.test_server(results)}
    runtime = versions["nvidia-cuda-runtime"]
    assert runtime and all(v.startswith(hfcards.CUDA + ".") for v in runtime)


def test_shell_lines_split_back_into_the_command():
    command = ["vllm", "serve", "m", *(f"--flag-{i}" for i in range(12)), "--x", "a b"]
    lines = hfcards.shell_lines(command, width=40)
    assert all(len(line) <= 40 for line in lines)
    assert shlex.split("\n".join(lines).replace("\\\n", " ")) == command


REPLIES = [
    {
        "period_end": "2026-06-30",
        "quarter": {
            "revenue": {"value": 1234, "unit": "millions"},
            "net_income": {"value": -120.5, "unit": "millions"},
            "eps_diluted": 0.45,
        },
        "year_to_date": {
            "revenue": {"value": 2401000, "unit": "thousands"},
            "net_income": None,
            "eps_diluted": -0.87,
        },
    },
    {
        "period_end": "2026-06-30",
        "quarter": {
            "revenue": {"value": 1.234, "unit": "billions"},
            "net_income": {"value": None, "unit": "millions"},
            "eps_diluted": None,
        },
        "year_to_date": {
            "revenue": {"value": 2401000000, "unit": "units"},
            "net_income": {"value": 231, "unit": "millions"},
            "eps_diluted": 2,
        },
    },
]


def run_snippet(
    code: str,
    folder: Path,
    release_text: str,
    reply: str,
    downloads: dict[str, Path],
    monkeypatch,
) -> tuple[dict, list[tuple]]:
    """Runs a card's Python snippet with huggingface_hub stubbed (each filename from
    downloads) and httpx.post faked; returns its namespace and the requests."""
    (folder / "release.md").write_text(release_text, encoding="utf-8")
    monkeypatch.chdir(folder)
    calls: list[tuple] = []

    def hf_hub_download(repo_id: str, filename: str, **kwargs) -> str:
        calls.append(("download", repo_id, filename))
        return str(downloads[filename])

    stub = types.ModuleType("huggingface_hub")
    stub.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    def post(url: str, **kwargs) -> httpx.Response:
        calls.append(("post", url, kwargs))
        body = {"choices": [{"message": {"role": "assistant", "content": reply}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    script = folder / "snippet.py"
    script.write_text(code, encoding="utf-8")
    return runpy.run_path(str(script), run_name="__main__"), calls


def extract_body(
    root: Path, served: str, text: str, tmp_path: Path
) -> tuple[str, dict]:
    """The endpoint and the first request body secbench.extract sends for a model
    scored by secbench.score (its settings file)."""
    env = tmp_path / "model.env"
    env.write_text(score.env_text("http://localhost:8000/v1", served, 16, 1.0))
    settings = extract.load_settings(env_file=env)
    prompt = extract.load_prompt("extract_v2", root / "prompts")
    with extract.OpenAIChat(settings) as client:
        return settings.endpoint, client.first_body(prompt, text)


def check_figures(namespace: dict, reply: str, accession: str) -> None:
    parsed = extract.parse_reply(reply, accession, extract.WITH_UNITS).prediction
    figures = namespace["figures"]
    assert figures["period_end"] == str(parsed.period_end)
    for group in extract.GROUPS:
        assert set(figures[group]) == set(FIELD_NAMES)
        for name in FIELD_NAMES:
            assert figures[group][name] == getattr(getattr(parsed, group), name), (
                group,
                name,
            )


@pytest.mark.parametrize("reply", REPLIES)
def test_the_snippet_sends_extracts_request(
    card, built, hf_context, hf_repo, tmp_path, monkeypatch, reply
):
    out, _ = built
    release = hf_context.releases["validation"][0]
    code = code_block(card, "python", "httpx.post")
    text = json.dumps(reply)
    namespace, calls = run_snippet(
        code,
        tmp_path,
        release.text,
        text,
        {"extract_v2.txt": out / "extract_v2.txt"},
        monkeypatch,
    )
    assert calls[0] == (
        "download",
        "test-user/qwen3-8b-sec-earnings-lora",
        "extract_v2.txt",
    )
    endpoint, expected = extract_body(
        hf_repo.root, "qwen3-8b-2k--step-00025", release.text, tmp_path
    )
    [(_, url, sent)] = calls[1:]
    assert url == endpoint
    assert sent["json"] == expected
    assert list(sent["json"]) == list(expected)
    assert sent["timeout"] == extract.TIMEOUT.read
    check_figures(namespace, text, release.release_accession)


# Serving results: the card's serving section and the int4 card


def test_no_serving_section_and_no_int4_card_without_serving_results(
    card, hf_context, tmp_path
):
    assert "## Serving" not in card
    assert "int4-awq" not in card
    written = hfcards.build_int4(hf_context, tmp_path / "int4-awq")
    assert written.files == [] and written.skipped == [INT4_NOTE]
    assert not (tmp_path / "int4-awq").exists()


def test_serving_results_add_the_serving_section(serving_repo, serving_cards):
    context, data = serving_repo
    out, model, _ = serving_cards
    assert model.skipped == []
    card = (out / "model" / "README.md").read_text()
    text = section(card, "Serving")
    session = data["session"]
    assert (
        f"one NVIDIA H100 NVL at ${session['gpu_hourly_usd']:g} per GPU-hour, "
        f"{session['server']}" in text
    )
    assert "at 1, 16 and 64 requests in flight" in text
    # Every variant is step-00025, chosen on validation: its scores there may overstate.
    assert (
        "Validation is where step-00025 was chosen, so every variant's validation "
        "score may overstate: it compares the variants, and the test results above "
        "give the level." in text
    )
    assert "| validation correct (step-00025 chosen on it) |" in text
    texts = hfcards.variant_texts(context, data)
    assert texts["adapter"] == (
        "the base model in bf16 with the LoRA adapter, as stage 2 served it but with "
        "prefix caching off"
    )
    for variant, v in data["variants"].items():
        best, startup = v["best"], v["startup"]
        accuracy = v["accuracy"]["validation"]
        agreement = v["agreement"]["validation"]["with_adapter"]
        row = (
            f"| {variant} | {texts[variant]} | {v['status']} | "
            f"{startup['weights_gib']:.2f} | {startup['kv_cache_tokens']:,} | "
            f"{best['releases_per_hour']:,.0f} ({best['level']}) | "
            f"{best['latency_p95_s']:.2f} | ${best['cost_per_1000_usd']:.3f} | "
            f"${best['cost_per_1000_at_1_usd']:.3f} | "
            f"{servereport.percent(accuracy['correct'], accuracy['graded'])} | "
            + (
                "(itself)"
                if agreement is None
                else servereport.agreement_text(agreement)
            )
            + " |"
        )
        assert row in text, variant
    # The int4-awq variant is complete, so the card links its repository.
    assert (
        "The int4-awq model is on the Hub as [test-user/qwen3-8b-sec-earnings-int4-awq]"
        "(https://huggingface.co/test-user/qwen3-8b-sec-earnings-int4-awq)." in text
    )


def test_the_adapter_variant_is_described_by_the_sessions_settings(serving_repo):
    context, data = serving_repo
    on = json.loads(json.dumps(data))
    on["settings"]["prefix_caching"] = True
    assert hfcards.variant_texts(context, on)["adapter"] == (
        "the base model in bf16 with the LoRA adapter, as stage 2 served it"
    )
    other = json.loads(json.dumps(data))
    other["session"]["server"] = "vLLM 0.33.0"
    with pytest.raises(ExportError, match="stage 2's vLLM version"):
        hfcards.variant_texts(context, other)


def test_the_int4_card(serving_repo, serving_cards, hf_repo):
    _, data = serving_repo
    out, _, written = serving_cards
    assert written.files == [out / "int4-awq" / "README.md"] and written.skipped == []
    assert sorted(p.name for p in (out / "int4-awq").iterdir()) == ["README.md"]
    card = (out / "int4-awq" / "README.md").read_text()
    meta = front_matter(card)
    assert meta["base_model"] == "test-user/qwen3-8b-sec-earnings-lora"
    assert meta["base_model_relation"] == "quantized"
    assert meta["library_name"] == "transformers"
    assert meta["license"] == "apache-2.0"
    assert {"awq", "4-bit", "compressed-tensors", "vllm"} <= set(meta["tags"])
    assert meta["datasets"] == ["test-user/sec-earnings-bench"]
    pip, command = shell_command(code_block(card, "sh", "vllm serve"))
    assert pip == ["pip", "install", "vllm==0.32.0"]
    assert command == [
        "vllm",
        "serve",
        "test-user/qwen3-8b-sec-earnings-int4-awq",
        "--max-model-len",
        str(data["settings"]["max_model_len"]),
    ]
    built = data["models"]["int4-awq"]
    what = section(card, "What it is")
    for text in (
        "llmcompressor 0.15.0 and compressed-tensors 0.20.1",
        "the W4A16_ASYM scheme on the Linear layers except lm_head",
        "pack-quantized format",
        f"{built['weight_gib']:.2f} GiB of weights",
        "128 examples of the training data",
        "at most 16,384 tokens each",
        "`library_name: transformers`",
    ):
        assert text in what, text
    results = section(card, "Results")
    mine, adapter = data["variants"]["int4-awq"], data["variants"]["adapter"]
    for split in ("validation", "time_test"):
        for v in (mine, adapter):
            a = v["accuracy"][split]
            assert servereport.percent(a["correct"], a["graded"]) in results
        assert (
            servereport.agreement_text(mine["agreement"][split]["with_adapter"])
            in results
        )
        assert (
            servereport.agreement_text(mine["agreement"][split]["with_stage2"])
            in results
        )
    for level in mine["levels"]:
        assert (
            f"| {level['level']} | {level['ok']}/{level['requests']} | "
            f"{level['releases_per_hour']:,.0f} | {level['latency_s']['p50']:.2f} / "
            f"{level['latency_s']['p95']:.2f} |" in results
        )
    assert f"| KV cache (tokens) | {mine['startup']['kv_cache_tokens']:,} |" in results
    assert f"${mine['best']['cost_per_1000_usd']:.3f}" in results
    # Validation scores carry the protocol's label; the time test is descriptive.
    reference = data["reference_accuracy"]
    stage2 = {
        s: servereport.percent(r["correct"], r["graded"]) for s, r in reference.items()
    }
    assert (
        "Validation is where step-00025 was chosen, so every validation score below "
        "may overstate: it compares the variants, and the time test gives the level "
        "(descriptive here, as it is in the adapter's card). Stage 2's runs of the "
        f"adapter scored {stage2['validation']} on validation, chosen on validation, "
        f"and {stage2['time_test']} on the time test." in results
    )
    assert "| validation (step-00025 chosen on it), correct / graded |" in results
    assert "pre-registered" not in results
    assert (
        "The adapter column is the base model in bf16 with the LoRA adapter, as stage "
        "2 served it but with prefix caching off." in results
    )
    usage = section(card, "How to use")
    assert "the server needs an NVIDIA driver for CUDA 13.0 or later" in usage


def test_the_int4_snippet_names_its_repository(
    serving_cards, serving_repo, hf_repo, tmp_path, monkeypatch
):
    out, _, _ = serving_cards
    context, _ = serving_repo
    card = (out / "int4-awq" / "README.md").read_text()
    release = context.releases["company_test"][1]
    text = json.dumps(REPLIES[0])
    namespace, calls = run_snippet(
        code_block(card, "python", "httpx.post"),
        tmp_path,
        release.text,
        text,
        {"extract_v2.txt": out / "model" / "extract_v2.txt"},
        monkeypatch,
    )
    assert calls[0][1:] == ("test-user/qwen3-8b-sec-earnings-lora", "extract_v2.txt")
    _, expected = extract_body(
        hf_repo.root, "test-user/qwen3-8b-sec-earnings-int4-awq", release.text, tmp_path
    )
    assert calls[1][2]["json"] == expected
    check_figures(namespace, text, release.release_accession)


def change_serving(root: Path, change: Callable[[dict], None]) -> None:
    path = root / "results" / "serving_results.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_no_int4_card_when_its_variant_failed(hf_repo_copy, tmp_path):
    hf_fixture.add_serving_results(hf_repo_copy.root)
    change_serving(
        hf_repo_copy.root,
        lambda d: d["variants"]["int4-awq"].update(status="failed", best=None),
    )
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "test-user")
    written = hfcards.build_int4(context, tmp_path / "int4-awq")
    assert written.files == []
    assert written.skipped == [
        (
            "int4-awq card: its variant in results/serving_results.json is failed, "
            "not complete"
        )
    ]
    model = hfcards.build_model(context, tmp_path / "model")
    text = section((tmp_path / "model" / "README.md").read_text(), "Serving")
    assert "| int4-awq | merged, 4-bit weights with AWQ | failed |" in text
    assert "qwen3-8b-sec-earnings-int4-awq" not in text
    assert model.skipped == []


def test_the_int4_card_needs_its_variant_and_its_build_manifest(serving_repo):
    _, data = serving_repo
    assert hfcards.int4_problem(data) is None
    results = json.loads(json.dumps(data))
    del results["models"]["int4-awq"]
    assert hfcards.int4_problem(results) == (
        "int4-awq card: results/serving_results.json has no models.int4-awq (the 4-bit "
        "model's build manifest)"
    )
    results["variants"]["int4-awq"]["status"] = "skipped"
    assert hfcards.int4_problem(results) == (
        "int4-awq card: its variant in results/serving_results.json is skipped, not "
        "complete"
    )
    del results["variants"]["int4-awq"]
    assert "is not run, not complete" in hfcards.int4_problem(results)


def test_a_library_version_not_recorded_is_left_out(serving_repo):
    context, data = serving_repo
    results = json.loads(json.dumps(data))
    test = read_test_config(context.sources.root)
    versions = results["models"]["int4-awq"]["versions"]
    versions["llmcompressor"] = None  # quantize.versions: the package wasn't found
    lines = "\n".join(hfcards.int4_build_lines(test, results))
    assert "llm-compressor's AWQ (compressed-tensors 0.20.1), duo_scaling both" in lines
    assert "None" not in lines
    versions["compressed-tensors"] = None
    lines = "\n".join(hfcards.int4_build_lines(test, results))
    assert "llm-compressor's AWQ, duo_scaling both" in lines


def test_serving_results_of_another_checkpoint_stop_the_build(hf_repo_copy, tmp_path):
    hf_fixture.add_serving_results(hf_repo_copy.root)
    change_serving(hf_repo_copy.root, lambda d: d.update(checkpoint="step-00050"))
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "test-user")
    with pytest.raises(ExportError, match="is for"):
        hfcards.build_int4(context, tmp_path / "int4-awq")
    with pytest.raises(ExportError, match="is for"):
        hfcards.build_model(context, tmp_path / "model")


def test_the_int4_card_is_deterministic(serving_repo, serving_cards, tmp_path):
    context, _ = serving_repo
    out, _, _ = serving_cards
    hfcards.build_int4(context, tmp_path / "int4-awq")
    hfcards.build_model(context, tmp_path / "model")
    for path in ("int4-awq/README.md", "model/README.md"):
        assert (tmp_path / path).read_bytes() == (out / path).read_bytes()


def test_quantization_details_from_the_saved_config():
    config = {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": False,
                    "group_size": 128,
                    "strategy": "group",
                },
            }
        },
        "format": "pack-quantized",
        "ignore": ["lm_head"],
        "quant_method": "compressed-tensors",
    }
    assert hfcards.quant_weights(config) == (
        "4-bit int weights with zero points, in groups of 128, activations not "
        "quantized"
    )
    assert hfcards.quant_weights({"format": "pack-quantized"}) is None
    assert hfcards.quant_weights(None) is None


def test_the_int4_limits_say_what_quantization_changed(serving_repo, hf_context):
    _, data = serving_repo
    mine = json.loads(json.dumps(data["variants"]["int4-awq"]))
    lines = hfcards.int4_limitation_lines(hf_context, data, mine)
    assert any(
        "**Quantization changes some predictions:** the figures equal to the adapter "
        "variant's are 357/360 figures, 57/60 releases on validation and 2073/2076 "
        "figures, 343/346 releases on the time test." in line
        for line in lines
    )
    for split in mine["agreement"].values():
        found = split["with_adapter"]
        found.update(same=found["figures"], releases_identical=found["releases"])
    lines = hfcards.int4_limitation_lines(hf_context, data, mine)
    assert any("**Quantization may change predictions:**" in line for line in lines)


# Nothing private


def private_strings(root: Path) -> list[str]:
    return [
        hf_fixture.SECRET_ORG,
        hf_fixture.SECRET_EMAIL,
        hf_fixture.RELEASE_EMAIL,
        hf_fixture.INSTANCE_PATH,
        "/workspace",
        "/Users/",
        "/home/",
        str(Path.home()),
        str(root),
    ]


def test_the_folders_hold_nothing_private(built, serving_cards, hf_repo):
    out, _ = built
    serving_out, _, _ = serving_cards
    texts = {
        name: (out / name).read_text()
        for name in ("README.md", "adapter_config.json", "extract_v2.txt")
    }
    texts["int4"] = (serving_out / "int4-awq" / "README.md").read_text()
    texts["serving"] = (serving_out / "model" / "README.md").read_text()
    for name, text in texts.items():
        for secret in private_strings(hf_repo.root):
            assert secret not in text, (name, secret)
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text), name  # no e-mail
        prose = re.sub(r"^```.*?^```$", "", text, flags=re.MULTILINE | re.DOTALL)
        # no unfilled value (the prose itself says "None of")
        assert not re.search(r"\bNone\b(?! of)|\bnan\b", prose), name


# The real data (skipped without runs/, training/ and data/splits/)

REAL = Sources(REPO)
needs_real_data = pytest.mark.skipif(
    not (
        REAL.runs.is_dir()
        and REAL.training.is_dir()
        and REAL.splits_dir.is_dir()
        and (REAL.checkpoint / "adapter_model.safetensors").is_file()
        and (REAL.labels / "examples.csv").is_file()
    ),
    reason="needs the real runs, training files and splits (not in the repository)",
)


@pytest.fixture(scope="module")
def real_card(tmp_path_factory):
    context = hfcommon.load_context(REAL, hfcommon.PLACEHOLDER_USER)
    out = tmp_path_factory.mktemp("real") / "model"
    written = hfcards.build_model(context, out)
    return out, written, (out / "README.md").read_text(encoding="utf-8")


@needs_real_data
def test_real_card_headline_numbers(real_card):
    _, written, card = real_card
    results = json.loads(REAL.test_results.read_text())
    under = results["under_test"]
    company, time_test = (results["splits"][s] for s in ("company_test", "time_test"))
    primary = company["differences"]["base"]["all"]
    assert (
        f"**{num(primary['difference'], signed=True)} points "
        f"{bounds(primary['interval'], signed=True)}**" in card
    )
    assert "**+10.12 points [+5.08, +15.58]**" in card
    for split, shown in ((company, "99.24%"), (time_test, "97.64%")):
        cell = acc_cell(split["accuracy"][under]["all"])
        assert f"({shown})" in cell and cell in card
    validation = json.loads((REAL.training / "validation.json").read_text())
    chosen = next(m for m in validation["models"] if m["label"] == "step-00025")
    assert (chosen["correct"], chosen["graded"]) == (287, 293)
    assert results["validation"]["counts"][under]["all"] == [287, 293]
    assert "98.0% (287/293), is chosen on validation" in card
    assert (
        "| step-00025 (chosen) | 25 | 400 | 287/293 (98.0%) | 102/102 | 96/102 | 89/89 |"
        in card
    )
    meta = front_matter(card)
    values = [r["metrics"][0]["value"] for r in meta["model-index"][0]["results"]]
    assert values == [99.24, 97.64]
    assert written.skipped == ([] if REAL.serving_results.is_file() else [SERVING_NOTE])


@needs_real_data
def test_real_card_training_and_adapter(real_card):
    out, _, card = real_card
    results = json.loads(REAL.test_results.read_text())
    m = results["model_under_test"]
    text = section(card, "Training")
    assert (
        f"{m['labelled']:,} releases of the training companies" in text
        and f"{m['within_max_tokens']:,} fit the {m['max_tokens']:,}-token cap" in text
        and f"{m['selected']:,} of them were drawn with seed 20261101" in text
    )
    assert (
        "(first quarter 581, second or third quarter 989, fourth quarter 430)" in text
    )
    assert f"had seen the first {m['checkpoint']['examples_seen']:,} of them" in text
    assert (
        f"One epoch of {m['steps']} steps of {m['examples_per_step']} examples" in text
    )
    assert "a linear warmup over 10 steps" in text and "every 25 steps" in text
    # The mean over steps 1 to 25 is the checkpoint's own record of it (state.json).
    state = json.loads((REAL.checkpoint / "state.json").read_text())
    assert f"{state['mean_loss_since_last_checkpoint']:.2g}" == "0.015"
    assert (
        "Training loss (cross-entropy per reply token, train_log.jsonl): 0.12 at step "
        "1, and 0.015 averaged over steps 1 to 25; a step's loss covers only its 16 "
        "examples, so it is noisy (from 0.00045 to 0.0088 over steps 21 to 25)." in text
    )
    assert "0.00062" not in text
    assert "One Vast H100 NVL" in text
    test = read_test_config(REPO)
    assert (
        hfcommon.sha256_file(out / "adapter_model.safetensors") == test.adapter_sha256
    )
    assert m["checkpoint"]["adapter_sha256"] == test.adapter_sha256
    saved = json.loads((REAL.checkpoint / "adapter_config.json").read_text())
    mine = json.loads((out / "adapter_config.json").read_text())
    assert list(mine) == list(saved)
    assert {k for k in saved if saved[k] != mine[k]} == {
        "base_model_name_or_path",
        "revision",
    }
    assert "9 in the test sets" in card
    for name in ("README.md", "adapter_config.json", "extract_v2.txt"):
        text = (out / name).read_text()
        for secret in ("/workspace", "/Users/", "/home/", str(Path.home())):
            assert secret not in text, (name, secret)
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text), name
