"""Tests for secbench.hfcommon, the Hugging Face export's data model, on the small
repository of tests/hf_fixture.py; the last ones read the real data and skip
without it (runs/, training/ and data/ aren't in the repository)."""

from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from secbench import grade, hfcommon, inputs
from secbench.hfcommon import ExportError, Sources

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


# Releases


def test_releases_rebuild_the_frozen_splits(hf_repo, hf_context):
    releases = hf_context.releases
    assert {k: [r.ticker for r in v] for k, v in releases.items()} == {
        "validation": ["AAA", "BBB"],
        "company_test": ["CCC", "DDD"],
        "time_test": ["EEE", "FFF", "GGG"],
    }
    eee, fff, ggg = releases["time_test"]
    assert (eee.kept, fff.kept, ggg.kept) == (True, True, False)
    assert ggg.left_out_reason.startswith("prompt over 39,936 tokens")
    assert ggg.prompt_tokens == 45000 and eee.left_out_reason is None
    ddd = releases["company_test"][1]
    assert ddd.preferred_dividends and not eee.preferred_dividends
    assert (ddd.company, ddd.sector, ddd.cik) == ("Delta Corp", "Utilities", 1004)
    assert ddd.exhibits == ("EX-99.1",)
    assert ddd.release_url.startswith("https://www.sec.gov/Archives/edgar/data/1004/")
    assert ddd.filing_url == (
        "https://www.sec.gov/Archives/edgar/data/1004/000000100425000041/"
        "0000001004-25-000041-index.htm"
    )
    assert ddd.linked_url.endswith("/000000100425000044/0000001004-25-000044-index.htm")
    bbb = releases["validation"][1]
    assert (bbb.linked_form, bbb.period_type) == ("10-K", "full year")
    record = hf_repo.root / "data" / "releases" / "0000001002"
    assert bbb.text == inputs.build_input(record / f"{bbb.release_accession}.json").text


def test_releases_stop_when_a_frozen_input_changes(hf_repo_copy):
    spec = next(s for s in hf_repo_copy.specs if s.ticker == "CCC")
    path = (
        hf_repo_copy.root
        / "data"
        / "stage2"
        / "releases"
        / f"{spec.cik:010d}"
        / f"{spec.accession}.md"
    )
    path.write_text(path.read_text() + "\nA late correction.\n")
    with pytest.raises(ExportError, match="frozen splits don't rebuild"):
        hfcommon.load_releases(Sources(hf_repo_copy.root))


def test_releases_stop_when_a_split_file_disagrees(hf_repo_copy):
    tokens = hf_repo_copy.root / "data" / "splits" / "time_test" / "tokens.csv"
    tokens.write_text(tokens.read_text().replace("left_out", "kept"))
    with pytest.raises(ExportError, match="disagrees"):
        hfcommon.load_releases(Sources(hf_repo_copy.root))


# Answers and where they are printed


def printed(text: str, spans) -> list[str]:
    return [text[a:b] for a, b in spans]


def test_answers_with_where_the_grader_finds_them(hf_context):
    answers = hf_context.answers
    assert {k: len(v) for k, v in answers.items()} == {
        "validation": 11,
        "company_test": 9,
        "time_test": 11,  # GGG is left out: its answers aren't in the split
    }
    texts = {
        r.release_accession: r.text
        for rows in hf_context.releases.values()
        for r in rows
    }
    by_key = {
        (a.ticker, a.field, a.period_type): a for rows in answers.values() for a in rows
    }
    cases = {
        ("AAA", "revenue", "quarter"): ("1,234", ["1,234"]),
        ("BBB", "revenue", "quarter"): ("512,300", ["512,300"]),
        ("BBB", "net_income", "quarter"): ("(30,250)", ["(30,250)"]),
        ("EEE", "net_income", "quarter"): ("-15", ["−15"]),
        ("EEE", "eps_diluted", "quarter"): ("(0.11)", ["(0.11)"]),
        ("CCC", "eps_diluted", "quarter"): (None, []),
    }
    for key, (form, shown) in cases.items():
        a = by_key[key]
        assert a.printed_as == form, key
        assert printed(texts[a.release_accession], a.spans) == shown, key
        assert a.in_input == (form is not None)
    loss = by_key["EEE", "net_income", "year_to_date"]
    assert loss.printed_as.startswith("40 on a loss row (Net loss")
    assert printed(texts[loss.release_accession], loss.spans) == ["40"]
    derived = by_key["BBB", "revenue", "quarter"]
    assert derived.derived and derived.nine_months_accession == "0000001002-25-000030"
    assert by_key["AAA", "revenue", "quarter"].full_year_accession is None


def release_records() -> list[Path]:
    return [
        path
        for folder in ("input_releases", "label_releases")
        for path in sorted((FIXTURES / folder).glob("*/*.json"))
    ]


@pytest.mark.parametrize("record", release_records(), ids=lambda p: p.stem)
def test_answer_spans_agree_with_the_grader_on_real_releases(record):
    """Figures of a real release, as answers in each unit and sign: the form
    answer_spans reports is grade.find_in_input's, and its spans show that form."""
    text = inputs.build_input(record).text
    rows = grade.input_rows(text)
    values = set()
    for row in rows:
        for figure in row.figures:
            for unit in (1, 10**3, 10**6):
                values.add(abs(figure.value) * unit)
    checked = 0
    for value in sorted(values)[::4][:120]:
        field = "eps_diluted" if value != value.to_integral_value() else "revenue"
        for signed in (value, -value):
            expected = grade.find_in_input(signed, field, text, rows)
            form, spans = hfcommon.answer_spans(signed, field, text, rows)
            assert form == expected, (signed, field)
            assert bool(spans) == (form is not None)
            for a, b in spans:
                assert form.split(" ")[0].strip("()-") in text[a:b]
            checked += 1
    assert checked > 0


def test_answer_spans_and_the_grader_disagreeing_stops_the_export(
    hf_repo_copy, monkeypatch
):
    monkeypatch.setattr(hfcommon, "answer_spans", lambda *a, **k: (None, ()))
    sources = Sources(hf_repo_copy.root)
    with pytest.raises(ExportError, match="grade.find_in_input"):
        hfcommon.load_answers(sources, hfcommon.load_releases(sources))


# Runs


def comparison_main_table(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## Runs on all"))
    table = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        table.append([c.strip() for c in line.strip("|").split(" | ")])
    header = table[0]
    return [dict(zip(header, row, strict=True)) for row in table[2:]]


COMPARISON_COLUMNS = {
    "model_id": "model",
    "prompt": "prompt",
    "settings": "settings sent",
    "concurrency": "concurrency",
    "releases": "releases",
    "accuracy": "accuracy",
    "revenue": "revenue",
    "net_income": "net_income",
    "eps_diluted": "eps_diluted",
    "unit_slips": "unit slips",
    "format_retries": "format retries",
    "input_tokens_per_release": "input tokens / release",
    "cost": "cost",
    "cost_basis": "cost basis",
    "cost_per_1000": "cost / 1,000 releases",
    "latency_p50": "latency p50",
    "latency_p95": "latency p95",
}


def assert_matches_comparison(rows: list[dict], comparison: Path) -> None:
    table = comparison_main_table(comparison)
    assert [r["run"] for r in rows] == [r["run"] for r in table]
    for mine, theirs in zip(rows, table, strict=True):
        for key, column in COMPARISON_COLUMNS.items():
            assert mine[key] == theirs[column], (mine["run"], key)


def test_validation_runs_are_comparisons_main_table(hf_repo, hf_context):
    runs = hf_context.validation_runs
    assert [r.name for r in runs] == sorted(hf_repo.validation_runs)
    assert [(r.model, r.source) for r in runs] == [
        ("Claude Haiku 5.5", "Anthropic API"),
        ("Qwen3-8B base", "vLLM (self-hosted)"),
        ("step-00025", "vLLM (self-hosted)"),
        ("Qwen3-8B (no thinking)", "vLLM (self-hosted)"),
    ]
    rows = hfcommon.validation_table(runs)
    assert_matches_comparison(rows, hf_repo.root / "results" / "comparison.md")
    assert (rows[0]["correct"], rows[0]["graded"]) == (11, 11)
    assert (rows[1]["correct"], rows[1]["graded"]) == (9, 11)  # two scale errors


def test_a_stale_validation_run_stops_the_export(hf_repo_copy):
    graded = hf_repo_copy.root / "runs" / "qwen3-8b-2k--base" / "graded.csv"
    graded.write_text(graded.read_text().replace(",1234000000,", ",1234000001,", 1))
    with pytest.raises(ExportError, match="stale"):
        hfcommon.validation_runs(Sources(hf_repo_copy.root))


def test_test_runs_in_the_reports_order(hf_repo, hf_context):
    assert {k: [r.name for r in v] for k, v in hf_context.test_runs.items()} == (
        hf_repo.test_runs
    )
    sources = [r.source for r in hf_context.test_runs["company_test"]]
    assert sources == [
        "vLLM (self-hosted)",
        "vLLM (self-hosted)",
        "OpenAI API",
        "Anthropic API",
        "OpenAI API",
    ]
    assert [r.model for r in hf_context.test_runs["time_test"]] == list(
        hf_repo.test_results["models"].values()
    )


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda info: info.update(links_sha256="0" * 64), "isn't on"),
        (lambda info: info["calls"].pop(), "doesn't cover"),
    ],
)
def test_a_test_run_must_be_on_its_split(hf_repo_copy, change, match):
    path = hf_repo_copy.root / "runs" / "gpt-6-luna--company_test" / "run.json"
    info = json.loads(path.read_text())
    change(info)
    path.write_text(json.dumps(info))
    sources = Sources(hf_repo_copy.root)
    with pytest.raises(ExportError, match=match):
        hfcommon.test_runs(sources, hfcommon.load_test_results(sources))


def test_prediction_rows_are_graded_csv(hf_context):
    rows = hfcommon.prediction_rows(hf_context.test_runs["company_test"])
    # 2 releases, 9 answers, and a no_answer row for CCC's year to date, per run.
    assert len(rows) == 5 * 10
    base = [r for r in rows if r.model == "Qwen3-8B base"]
    scale = next(r for r in base if r.verdict == grade.SCALE)
    assert (scale.ticker, scale.predicted, scale.factor) == (
        "CCC",
        "845000000000",
        "0.001",
    )
    no_answer = next(r for r in base if r.verdict == grade.NO_ANSWER)
    assert no_answer.answer is None and no_answer.answer_tag is None
    sol = next(r for r in rows if r.model == "GPT-6.1 Sol" and r.ticker == "CCC")
    assert sol.source == "OpenAI API"
    not_printed = [r for r in rows if r.verdict == grade.NOT_IN_INPUT]
    assert {r.predicted for r in not_printed} == {None, "0.32"}


def test_runs_raw_folders_are_never_read(hf_repo_copy):
    for folder in (hf_repo_copy.root / "runs").glob("*/raw"):
        shutil.rmtree(folder)
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "someone")
    assert len(context.validation_runs) == 4


# Training examples


def test_train_rows(hf_repo):
    rows = list(hfcommon.train_rows(Sources(hf_repo.root)))
    assert [(r.ticker, r.trained, r.position, r.seen_by_checkpoint) for r in rows] == [
        ("EEE", True, 1, False),
        ("FFF", True, 0, True),
        ("GGG", False, None, False),
    ]
    assert [m["role"] for m in rows[0].messages] == ["system", "user", "assistant"]
    prompt = (hf_repo.root / "prompts" / "extract_v2.txt").read_text()
    assert rows[0].messages[0]["content"] == prompt


def test_train_rows_need_examples_csvs_order(hf_repo_copy):
    path = hf_repo_copy.root / "data" / "stage2" / "labels" / "examples.csv"
    header, first, second, *rest = path.read_text().splitlines()
    path.write_text("\n".join([header, second, first, *rest]) + "\n")
    with pytest.raises(ExportError, match="order"):
        list(hfcommon.train_rows(Sources(hf_repo_copy.root)))


# Names, links and writers


@pytest.mark.parametrize(
    "run, info, name",
    [
        ("qwen3-8b-2k--step-00050", {"model": "qwen3-8b-2k--step-00050"}, "step-00050"),
        ("qwen3-8b-2k--base", {"model": "Qwen/Qwen3-8B"}, "Qwen3-8B base"),
        ("qwen3-8b-v2", {"model": "Qwen/Qwen3-8B"}, "Qwen3-8B (thinking)"),
        (
            "x",
            {
                "model": "Qwen/Qwen3-8B",
                "request_parameters": {
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            },
            "Qwen3-8B (no thinking)",
        ),
        ("x", {"model": "openai/gpt-oss-20b"}, "gpt-oss-20b"),
        ("x", {"model": "some/new-model"}, "some/new-model"),
    ],
)
def test_display_names(run, info, name):
    assert hfcommon.display_name(run, info) == name


@pytest.mark.parametrize(
    "info, expected",
    [
        ({"provider": "anthropic"}, "Anthropic API"),
        ({"provider": "openai", "cost_basis": "tokens"}, "OpenAI API"),
        ({"provider": "openai", "cost_basis": "gpu_hours"}, "vLLM (self-hosted)"),
        ({"provider": "openai", "estimated_cost_usd": 0.1}, "OpenAI API"),
    ],
)
def test_sources(info, expected):
    assert hfcommon.source(info) == expected


@pytest.mark.parametrize("user", ["jinnn11", "HF_USER", "a.b-c_d", "x"])
def test_hub_names(user):
    assert hfcommon.check_user(user) == user


@pytest.mark.parametrize("user", ["", "-a", "a-", "a--b", "a..b", "a/b", "a b"])
def test_bad_hub_names(user):
    with pytest.raises(ExportError, match="Hub name"):
        hfcommon.check_user(user)


def test_repo_ids_and_urls():
    assert hfcommon.repo_id("me", "dataset") == "me/sec-earnings-bench"
    assert hfcommon.hub_url("me", "space") == (
        "https://huggingface.co/spaces/me/sec-earnings-explorer"
    )
    assert hfcommon.hub_url("me", "model") == (
        "https://huggingface.co/me/qwen3-8b-sec-earnings-lora"
    )
    assert hfcommon.hub_url("me", "int4-awq").endswith(
        "me/qwen3-8b-sec-earnings-int4-awq"
    )


def test_author_from_license(hf_repo):
    assert hfcommon.author(Sources(hf_repo.root)) == ("2026", "Swaminathan Sankaran")


def test_writers_are_deterministic(tmp_path, hf_context):
    rows = hf_context.answers["validation"]
    first = tmp_path / "a" / "answers.jsonl"
    second = tmp_path / "b" / "answers.jsonl"
    assert hfcommon.write_jsonl(first, rows) == len(rows)
    hfcommon.write_jsonl(second, rows)
    assert first.read_bytes() == second.read_bytes()
    line = json.loads(first.read_text().splitlines()[0])
    assert line["spans"] == [list(s) for s in rows[0].spans]
    assert not list(tmp_path.rglob("*.part"))
    hfcommon.write_json(tmp_path / "x.json", {"value": Decimal("1.50"), "t": (1, 2)})
    assert (
        tmp_path / "x.json"
    ).read_text() == '{\n  "value": "1.50",\n  "t": [\n    1,\n    2\n  ]\n}\n'
    assert hfcommon.json_text({"é": 1}, indent=None) == '{"é":1}\n'


# The real data (skipped without runs/, training/ and data/splits/)

REAL = Sources(REPO)
needs_real_data = pytest.mark.skipif(
    not (
        REAL.runs.is_dir()
        and REAL.training.is_dir()
        and REAL.splits_dir.is_dir()
        and (REAL.labels / "train.jsonl").is_file()
    ),
    reason="needs the real runs, training files and splits (not in the repository)",
)


@pytest.fixture(scope="module")
def real_context():
    return hfcommon.load_context(REAL, hfcommon.PLACEHOLDER_USER)


@needs_real_data
def test_real_counts(real_context):
    releases = real_context.releases
    assert {k: sum(r.kept for r in v) for k, v in releases.items()} == {
        "validation": 60,
        "company_test": 148,
        "time_test": 346,
    }
    assert {k: sum(not r.kept for r in v) for k, v in releases.items()} == {
        "validation": 0,
        "company_test": 0,
        "time_test": 9,
    }
    for split, rows in real_context.answers.items():
        manifest = json.loads((REAL.splits_dir / split / "manifest.json").read_text())
        assert len(rows) == manifest["counts"]["answers"]
        assert sum(a.in_input for a in rows) == manifest["counts"]["answers_in_input"]
    assert [len(real_context.answers[s]) for s in hfcommon.SPLIT_NAMES] == [
        293,
        700,
        1792,
    ]


@needs_real_data
def test_real_validation_table_is_comparison_md(real_context):
    rows = hfcommon.validation_table(real_context.validation_runs)
    assert len(rows) == 24
    assert_matches_comparison(rows, REAL.comparison)


@needs_real_data
def test_real_train_rows():
    rows = list(hfcommon.train_rows(REAL))
    config = json.loads((REAL.training / "config.json").read_text())
    assert len(rows) == config["selection"]["examples"] == 3200
    assert sum(r.trained for r in rows) == config["selection"]["selected"] == 2000
    seen = json.loads((REAL.checkpoint / "state.json").read_text())["examples_seen"]
    assert sum(r.seen_by_checkpoint for r in rows) == seen == 400
    splits_accessions = {
        r["release_accession"]
        for s in hfcommon.SPLIT_NAMES
        for r in hfcommon.read_csv(REAL.splits_dir / s / "tokens.csv")
    }
    assert not splits_accessions & {r.release_accession for r in rows}
