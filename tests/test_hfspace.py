"""Tests for secbench.hfspace, the results explorer's Space, on the small repository of
tests/hf_fixture.py: the built folder (files, README metadata, the gradio pin, sizes,
determinism, privacy), the app's logic on the built data (space/app.py imported by
path, without gradio: choices, filters, the figures table and its colours, the
highlighted text, the Errors and Leaderboard tables), and, when gradio is installed
(the hf group), the Blocks with and without serving data. The last tests build the
Space from the real data and skip without it."""

from __future__ import annotations

import ast
import dataclasses
import html
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import hf_fixture
import pytest
from test_hfcommon import comparison_main_table

from secbench import grade, hfcommon, hfspace, servereport, splits, testreport
from secbench.hfcommon import ExportError, Sources

REPO = Path(__file__).resolve().parents[1]
SPACE = REPO / "space"
TEST_RESULTS_MD = (REPO / "results" / "test_results.md").read_text(encoding="utf-8")
SPLITS = ("validation", "company_test", "time_test")
DATA_FILES = {
    "data/meta.json",
    "data/leaderboard.json",
    "data/runs.json",
    "data/releases.json",
    "data/answers.json",
    "data/errors.json",
    *(f"data/predictions/{split}.json" for split in SPLITS),
    "data/charts/accuracy_vs_cost.png",
    "data/charts/test_accuracy_vs_cost.png",
}
# hfcommon.validation_table's keys, in the leaderboard's column order ("runs on" is
# the GPU for a GPU run, the source otherwise).
VALIDATION_KEYS = [
    "model",
    "runs_on",
    "prompt",
    "accuracy",
    "revenue",
    "net_income",
    "eps_diluted",
    "unit_slips",
    "cost_per_1000",
    "concurrency",
    "latency_p50",
    "latency_p95",
]
COMPARISON_COLUMNS = {
    "accuracy": "accuracy",
    "revenue": "revenue",
    "net_income": "net_income",
    "eps_diluted": "eps_diluted",
    "unit slips": "unit slips",
    "cost / 1,000 releases": "cost / 1,000 releases",
    "calls at a time": "concurrency",
    "latency p50": "latency p50",
    "latency p95": "latency p95",
}


def load_app(folder: Path):
    """space/app.py as built into folder, imported by path (no gradio needed)."""
    spec = importlib.util.spec_from_file_location("space_app", folder / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relative_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def built(hf_context, tmp_path_factory):
    out = tmp_path_factory.mktemp("space") / "space"
    return out, hfspace.build_space(hf_context, out)


@pytest.fixture(scope="module")
def app(built):
    return load_app(built[0])


@pytest.fixture(scope="module")
def data(app, built):
    return app.load_data(built[0] / "data")


def run_name(context, split: str, key: str) -> str:
    """A run's name by its key (test_results.json's model key on a test split)."""
    return next(r.name for r in context.runs(split) if r.key == key)


def accession(context, ticker: str) -> str:
    return next(
        r.release_accession
        for rows in context.releases.values()
        for r in rows
        if r.ticker == ticker
    )


# The built folder


def test_the_folder_holds_the_app_its_readme_and_its_data(hf_context, built):
    out, written = built
    texts = {
        f"data/texts/{split}/{r.release_accession}.md"
        for split, rows in hf_context.releases.items()
        for r in rows
    }
    expected = {"app.py", "requirements.txt", "README.md", *DATA_FILES, *texts}
    assert relative_files(out) == expected
    assert {p.relative_to(out).as_posix() for p in written.files} == expected
    assert written.skipped == [
        "Space's serving tab: results/serving_results.json not found"
    ]
    for name in ("app.py", "requirements.txt"):
        assert (out / name).read_bytes() == (SPACE / name).read_bytes()
    for name in ("accuracy_vs_cost.png", "test_accuracy_vs_cost.png"):
        source = hf_context.sources.results / name
        assert (out / "data" / "charts" / name).read_bytes() == source.read_bytes()
    assert sum(p.stat().st_size for p in out.rglob("*")) < 100 * 2**20


def test_the_texts_are_the_model_inputs_left_out_ones_too(hf_context, built):
    out, _ = built
    for split, rows in hf_context.releases.items():
        for r in rows:
            path = out / "data" / "texts" / split / f"{r.release_accession}.md"
            assert path.read_bytes() == r.text.encode("utf-8")
    ggg = hf_context.releases["time_test"][2]
    assert (
        not ggg.kept
        and (out / "data/texts/time_test" / f"{ggg.release_accession}.md").is_file()
    )


def front_matter(text: str) -> dict:
    """The README's YAML front matter, read as the simple YAML it is: key: value
    lines and lists of "  - item"."""
    assert text.startswith("---\n")
    block = text[4:].split("\n---\n", 1)[0]
    found: dict = {}
    key = None
    for line in block.splitlines():
        if line.startswith("  - "):
            found[key].append(line[4:])
            continue
        key, _, value = line.partition(": ")
        key = key.rstrip(":")
        found[key] = value.strip('"') if value else []
    return found


def pyproject_gradio_pin() -> str:
    groups = tomllib.loads((REPO / "pyproject.toml").read_text())["dependency-groups"]
    pins = [d.split("==", 1)[1] for d in groups["hf"] if d.startswith("gradio==")]
    assert len(pins) == 1
    return pins[0]


def test_the_requirements_pin_gradio_as_pyproject_does():
    lines = (SPACE / "requirements.txt").read_text().splitlines()
    assert lines == [f"gradio=={pyproject_gradio_pin()}"]
    assert hfspace.gradio_pin(SPACE / "requirements.txt") == pyproject_gradio_pin()


def test_readme_metadata(built):
    out, _ = built
    readme = (out / "README.md").read_text(encoding="utf-8")
    meta = front_matter(readme)
    assert meta == {
        "title": "SEC Earnings Explorer",
        "emoji": "📊",
        "colorFrom": "blue",
        "colorTo": "indigo",
        "sdk": "gradio",
        "sdk_version": pyproject_gradio_pin(),
        "python_version": "3.11",
        "app_file": "app.py",
        "pinned": "false",
        "license": "mit",
        "short_description": hfspace.SHORT_DESCRIPTION,
        "datasets": ["test-user/sec-earnings-bench"],
        "models": ["test-user/qwen3-8b-sec-earnings-lora"],
    }
    assert len(meta["short_description"]) <= 60
    assert 'python_version: "3.11"' in readme  # a string, not the float 3.11
    for link in (
        hfcommon.REPO_URL,
        "https://huggingface.co/datasets/test-user/sec-earnings-bench",
        "https://huggingface.co/test-user/qwen3-8b-sec-earnings-lora",
    ):
        assert f"({link})" in readme
    assert "**Serving:**" not in readme
    yaml = pytest.importorskip("yaml")  # as the Hub reads it
    parsed = yaml.safe_load(readme.split("---\n")[1])
    assert parsed == {**meta, "pinned": False}


def test_json_files_are_compact_and_hold_nothing_private(hf_repo, built):
    out, _ = built
    for path in (out / "data").rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert text.count("\n") == 1 and text.endswith("\n"), path
        json.loads(text)
        assert "@" not in text, path  # press contacts stay in the texts
        for private in (str(hf_repo.root), "/Users/", "/workspace/", "raw/"):
            assert private not in text, (path, private)
        assert hf_fixture.SECRET_ORG not in text
    assert hf_fixture.RELEASE_EMAIL in (
        out / "data/texts/validation/0000001001-26-000010.md"
    ).read_text(encoding="utf-8")


def test_two_builds_are_the_same_bytes(hf_context, built, tmp_path):
    first, _ = built
    second = tmp_path / "again"
    hfspace.build_space(hf_context, second)
    assert relative_files(first) == relative_files(second)
    for name in relative_files(first):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_an_email_outside_the_texts_stops_the_build(hf_repo_copy, tmp_path):
    graded = hf_repo_copy.root / "runs" / "gpt-6-luna--time_test" / "graded.csv"
    text = graded.read_text()
    assert "'Adjusted sales'" in text
    graded.write_text(
        text.replace("'Adjusted sales'", "'Adjusted sales, ir@eee.example'")
    )
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "u")
    with pytest.raises(ExportError, match="e-mail address"):
        hfspace.build_space(context, tmp_path / "space")


def test_a_missing_chart_stops_the_build(hf_repo_copy, tmp_path):
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "u")
    (hf_repo_copy.root / "results" / "test_accuracy_vs_cost.png").unlink()
    with pytest.raises(ExportError, match="test_accuracy_vs_cost.png not found"):
        hfspace.build_space(context, tmp_path / "space")


@pytest.mark.parametrize("text", ["", "gradio\n", "gradio==1\ngradio==2\n"])
def test_requirements_must_pin_gradio_once(tmp_path, text):
    path = tmp_path / "requirements.txt"
    path.write_text(text)
    with pytest.raises(ExportError, match="pin gradio once"):
        hfspace.gradio_pin(path)


# The app's source


def test_the_app_imports_neither_secbench_nor_gradio_at_import():
    source = (SPACE / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            top.add(node.module.split(".")[0])
    assert top <= sys.stdlib_module_names, top - sys.stdlib_module_names
    inner = {
        a.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and node not in tree.body
        for a in node.names
    }
    assert inner == {"gradio"}
    imported = {
        name.split(".")[0]
        for node in ast.walk(tree)
        for name in (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert "secbench" not in imported
    # No `from __future__ import annotations` (see the app's docstring).
    assert "__future__" not in imported


BLOCKED_IMPORTS = """
import importlib.abc, importlib.util, sys

class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("gradio", "pandas", "secbench"):
            raise ImportError(name + " is blocked")

sys.meta_path.insert(0, Blocked())
spec = importlib.util.spec_from_file_location("app", sys.argv[1])
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
data = app.load_data()
state = app.explore_state(data)
view = app.release_view(data, state.split, state.accession, state.runs)
print(state.split, state.ticker, "<mark" in view.text)
"""


def test_the_app_logic_runs_without_gradio_pandas_or_secbench(built):
    out, _ = built
    done = subprocess.run(
        [sys.executable, "-c", BLOCKED_IMPORTS, str(out / "app.py")],
        capture_output=True,
        text=True,
        check=True,
        cwd=out.parent,
    )
    assert done.stdout.split() == ["company_test", "CCC", "True"]


def test_the_apps_file_names_are_the_builds(app):
    for name in ("META", "LEADERBOARD", "RUNS", "RELEASES", "ANSWERS", "ERRORS"):
        assert getattr(app, name) == getattr(hfspace, name)
    assert (app.PREDICTIONS, app.SERVING) == (hfspace.PREDICTIONS, hfspace.SERVING)


# Explore: choices and filters


def test_split_and_model_choices(app, data, hf_context, hf_repo):
    assert app.split_choices(data) == [
        ("Validation", "validation"),
        ("Company test", "company_test"),
        ("Time test", "time_test"),
    ]
    assert app.run_choices(data, "company_test") == [
        (label, run)
        for label, run in zip(
            hf_repo.test_results["models"].values(),
            hf_repo.test_runs["company_test"],
            strict=True,
        )
    ]
    assert [label for label, _ in app.run_choices(data, "validation")] == [
        "Claude Haiku 5.5 · v1",
        "Qwen3-8B base · v2",
        "step-00025 · v2",
        "Qwen3-8B (no thinking) · v2",
    ]
    # Validation shows the test sets' models' validation runs first, in their order:
    # here only stage 2's two are in the mini repo.
    assert app.default_runs(data, "validation") == [
        "qwen3-8b-2k--step-00025",
        "qwen3-8b-2k--base",
    ]
    assert app.default_runs(data, "time_test") == hf_repo.test_runs["time_test"]
    verdicts = [v for _, v in app.verdict_filter_choices(data, "company_test")]
    assert verdicts == [
        app.ANY,
        app.MISS,
        "correct",
        "scale",
        "wrong_line",
        "not_in_input",
        "no_answer",
    ]


def test_filters_by_a_models_misses_or_a_verdict(app, data, hf_context):
    def tickers(split, model=app.ANY, verdict=app.ANY):
        return [r["ticker"] for r in app.matching_releases(data, split, model, verdict)]

    base = run_name(hf_context, "company_test", "base")
    step = run_name(hf_context, "company_test", "step-00025")
    assert tickers("company_test", base, app.MISS) == ["CCC"]  # a scale error
    assert tickers("company_test", step, app.MISS) == ["DDD"]  # the common line
    assert tickers("company_test", step, "scale") == []
    assert tickers("company_test", verdict="not_in_input") == ["CCC"]
    assert tickers("time_test", verdict="adjusted") == ["EEE"]
    assert tickers("time_test") == ["EEE", "FFF", "GGG"]  # left out too
    assert tickers("validation", verdict=app.MISS) == ["AAA"]
    assert app.verdict_after_model_change(base, app.ANY) == app.MISS
    assert app.verdict_after_model_change(base, "scale") == "scale"
    assert app.verdict_after_model_change(app.ANY, app.ANY) == app.ANY


def test_explore_state(app, data, hf_context):
    first = app.explore_state(data)
    assert (first.split, first.ticker, first.model, first.verdict) == (
        "company_test",
        "CCC",
        app.ANY,
        app.ANY,
    )
    assert first.accession == accession(hf_context, "CCC")
    assert first.company_choices == [
        ("CCC · Gamma Co", "CCC"),
        ("DDD · Delta Corp", "DDD"),
    ]
    assert first.status == "2 releases in the company test split."
    assert app.explore_state(data, "validation").status == (
        "2 releases in the validation split."
    )
    # The left-out release is counted, and said to be (the Leaderboard counts only
    # the kept ones).
    assert app.explore_state(data, "time_test").status == (
        "3 releases in the time test split, 1 of them left out for every model."
    )
    step = run_name(hf_context, "company_test", "step-00025")
    misses = app.explore_state(data, "company_test", step, app.MISS)
    assert (misses.ticker, misses.accession) == ("DDD", accession(hf_context, "DDD"))
    assert misses.status == "1 of the 2 releases match, from 1 company."
    # A link to a release: its split and company, with filters that hide it cleared.
    ggg = accession(hf_context, "GGG")
    linked = app.explore_state(data, accession=ggg)
    assert (linked.split, linked.ticker, linked.accession) == ("time_test", "GGG", ggg)
    assert linked.release_choices == [
        ("2026-08-01 · 10-Q, period ended 2026-06-30 · left out", ggg)
    ]
    ccc = accession(hf_context, "CCC")
    cleared = app.explore_state(data, "company_test", step, app.MISS, accession=ccc)
    assert (cleared.model, cleared.verdict, cleared.accession) == (
        app.ANY,
        app.ANY,
        ccc,
    )
    # Unknown values fall back to the defaults.
    odd = app.explore_state(data, "nope", "nobody", "bad verdict", "ZZZ", "0-0", ["x"])
    assert (odd.split, odd.model, odd.verdict, odd.ticker) == (
        "company_test",
        app.ANY,
        app.ANY,
        "CCC",
    )
    assert odd.runs == app.default_runs(data, "company_test")
    nothing = app.explore_state(data, "company_test", step, "scale")
    assert (nothing.ticker, nothing.accession) == (None, None)
    assert app.release_view(data, "company_test", None, []).header.startswith(
        "<p>No release matches"
    )


def test_a_repeated_run_shows_under_its_own_name(hf_repo_copy, tmp_path):
    """Two runs of one model and prompt (a repeat, as PLAN's repeat runs would make)
    are told apart by their run's name, instead of stopping the build."""
    runs = hf_repo_copy.root / "runs"
    repeat = runs / "qwen3-8b-nothink-v2-r2"
    shutil.copytree(runs / "qwen3-8b-nothink-v2", repeat)
    info = json.loads((repeat / "run.json").read_text())
    info["run_name"] = repeat.name
    (repeat / "run.json").write_text(json.dumps(info))
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "u")
    assert [hfspace.run_label(r) for r in context.validation_runs].count(
        "Qwen3-8B (no thinking) · v2"
    ) == 2
    out = tmp_path / "space"
    hfspace.build_space(context, out)
    app = load_app(out)
    data = app.load_data(out / "data")
    labels = {run: label for label, run in app.run_choices(data, "validation")}
    assert labels == {
        "2026-10-14T023302_claude-haiku-5-5_extract_v1": "Claude Haiku 5.5 · v1",
        "qwen3-8b-2k--base": "Qwen3-8B base · v2",
        "qwen3-8b-2k--step-00025": "step-00025 · v2",
        "qwen3-8b-nothink-v2": "Qwen3-8B (no thinking) · v2 · qwen3-8b-nothink-v2",
        repeat.name: f"Qwen3-8B (no thinking) · v2 · {repeat.name}",
    }
    _, rows = app.verdict_table(data, "validation")
    assert sorted(row[0] for row in rows) == sorted(labels.values())
    board = data.leaderboard["validation"]
    models = dict(zip(board["runs"], (row[0] for row in board["rows"]), strict=True))
    assert models[repeat.name] == f"Qwen3-8B (no thinking) · {repeat.name}"
    assert models["qwen3-8b-2k--base"] == "Qwen3-8B base"
    assert (
        "- **Repeated runs** of one model and prompt show their run's name"
        in (board["notes"])
    )


def test_left_out_releases_say_why_without_citing_the_plan(
    app, data, hf_context, built
):
    """The split's own reason cites PLAN.md, which isn't published; the Space says why
    from the prompt's token count and the limit."""
    out, _ = built
    ggg = next(r for r in hf_context.releases["time_test"] if not r.kept)
    assert ggg.left_out_reason == splits.left_out_reason(splits.MAX_PROMPT_TOKENS)
    reason = data.index["time_test"][ggg.release_accession]["left_out_reason"]
    assert reason == (
        f"its prompt is {ggg.prompt_tokens:,} tokens, over the 39,936-token limit "
        "(the GPU model's context of 40,960 tokens minus 1,024 for the reply)"
    )
    header = app.release_header(data, "time_test", ggg.release_accession)
    assert f"Left out for every model: {html.escape(reason)}." in header
    assert html.escape(reason) in app.figures_table(
        data, "time_test", ggg.release_accession, []
    )
    for path in out.rglob("*"):
        if path.suffix in (".json", ".md") and "texts" not in path.parts:
            assert "PLAN.md" not in path.read_text(encoding="utf-8"), path
    assert hfspace.left_out_text(hf_context.releases["time_test"][0]) is None
    for odd in (
        dataclasses.replace(ggg, left_out_reason="too long"),
        dataclasses.replace(ggg, prompt_tokens=splits.MAX_PROMPT_TOKENS),
    ):
        with pytest.raises(ExportError, match="doesn't describe"):
            hfspace.left_out_text(odd)


# Explore: the release


def cells_by_slot(rows, run):
    return {
        (row.field, row.period): next(c for c in row.cells if c.run == run)
        for row in rows
    }


def test_the_figures_table_and_its_colours(app, data, hf_context):
    step = run_name(hf_context, "company_test", "step-00025")
    base = run_name(hf_context, "company_test", "base")
    ddd = accession(hf_context, "DDD")
    rows = app.figure_rows(data, "company_test", ddd, [step, base])
    assert [(r.field, r.period) for r in rows] == [
        (f, p) for f in grade.FIELD_NAMES for p in ("quarter", "year_to_date")
    ]
    mine = cells_by_slot(rows, step)
    wrong = mine["net_income", "quarter"]
    assert (wrong.kind, wrong.prediction["verdict"], wrong.prediction["predicted"]) == (
        app.ERROR,
        "wrong_line",
        "190000000",
    )
    assert mine["revenue", "quarter"].kind == app.CORRECT
    assert cells_by_slot(rows, base)["net_income", "quarter"].kind == app.CORRECT
    table = app.figures_table(data, "company_test", ddd, [step, base])
    background = app.KIND_COLOURS[app.ERROR][0]
    evidence = html.escape(wrong.prediction["evidence"], quote=True)
    assert re.search(
        rf'background:{background};[^"]*" title="wrong_line: {re.escape(evidence)}"',
        table,
    )
    assert "190,000,000" in table and "200,000,000" in table

    ccc = accession(hf_context, "CCC")
    theirs = cells_by_slot(app.figure_rows(data, "company_test", ccc, [base]), base)
    slip = theirs["revenue", "quarter"]
    assert (slip.kind, slip.prediction["factor"]) == (app.SLIP, "0.001")
    assert theirs["eps_diluted", "quarter"].prediction["verdict"] == "not_in_input"
    assert theirs["eps_diluted", "quarter"].kind == app.UNGRADED
    assert theirs["revenue", "year_to_date"].prediction["verdict"] == "no_answer"
    assert theirs["net_income", "year_to_date"].prediction is None  # null, no answer
    assert theirs["net_income", "year_to_date"].kind == app.UNGRADED
    table = app.figures_table(data, "company_test", ccc, [base])
    for colour in (app.SLIP, app.UNGRADED, app.CORRECT):
        assert f"background:{app.KIND_COLOURS[colour][0]}" in table
    assert "scale (×0.001)" in table and "not printed: not graded" in table

    # A fourth quarter: the full year's column, and no quarter EPS answer.
    bbb = accession(hf_context, "BBB")
    rows = app.figure_rows(
        data, "validation", bbb, app.default_runs(data, "validation")
    )
    assert [r.period for r in rows[:2]] == ["quarter", "full_year"]
    assert rows[4].answer is None and rows[0].answer["derived"]
    ggg = accession(hf_context, "GGG")
    assert app.figure_rows(data, "time_test", ggg, ["x"]) == []
    assert "Left out for every model" in app.figures_table(data, "time_test", ggg, [])


def marked(html_text: str) -> list[str]:
    return [
        html.unescape(m) for m in re.findall(r"<mark [^>]*>(.*?)</mark>", html_text)
    ]


def test_highlights_land_on_the_printed_form(app, data, hf_context):
    bbb, eee = accession(hf_context, "BBB"), accession(hf_context, "EEE")
    text = app.read_release_text(data, "validation", bbb)
    shown = marked(app.release_view(data, "validation", bbb, []).text)
    assert "(30,250)" in shown and "512,300" in shown and "0.59" in shown
    text = app.read_release_text(data, "time_test", eee)
    view = app.release_view(data, "time_test", eee, [])
    assert {"640", "1,255", "−15", "40", "(0.11)", "(0.29)"} <= set(marked(view.text))
    marks = app.text_marks(data, "time_test", eee)
    loss = [m for m in marks if m[3].startswith("Net income, year to date")]
    assert [text[a:b] for a, b, _, _ in loss] == ["40"]
    number, line, start, end = app.line_of(text, loss[0][0], loss[0][1])
    assert line.startswith("| Net loss |") and line[start:end] == "40"
    assert view.answers.count("<li>") == 6
    assert "40 on a loss row (Net loss" in html.unescape(view.answers)
    assert f"line {number}: <code>" in view.answers
    assert "<b>40</b>" in view.answers


def test_highlights_split_overlapping_marks_and_escape_the_text(app):
    text = "a<b>cdef"
    marks = [(0, 5, "revenue", "first"), (3, 8, "net_income", "second")]
    shown = app.highlighted_text(text, marks)
    pieces = re.findall(r'<mark [^>]*title="([^"]*)">(.*?)</mark>', shown)
    assert [(html.unescape(t), html.unescape(x)) for t, x in pieces] == [
        ("first", "a<b"),
        ("first; second", ">c"),
        ("second", "def"),
    ]
    assert "<b>" not in shown.replace("&lt;b&gt;", "")
    assert app.excerpt("x" * 500 + "123" + "y" * 500, 500, 503).count("…") == 2


def test_figures_are_exact_with_separators(app):
    assert app.figure("1234000000") == "1,234,000,000"
    assert app.figure("-30250000") == "-30,250,000"
    assert app.figure("202000000.0") == "202,000,000.0"
    assert app.figure("0.45") == "0.45"
    assert app.figure("1.25E+7") == "12,500,000"
    assert app.figure(None) == "null"


# Errors


def test_the_errors_tab_counts_graded_csv(app, data, hf_context):
    for split in SPLITS:
        runs = hf_context.runs(split)
        columns, rows = app.verdict_table(data, split)
        assert columns[0] == "model" and columns[-1] == "unit slips"
        assert [row[0] for row in rows] == [hfspace.run_label(r) for r in runs]
        for run, row in zip(runs, rows, strict=True):
            tally = Counter(g["verdict"] for g in run.graded)
            assert dict(zip(columns[1:-1], row[1:-1], strict=True)) == {
                v: tally[v] for v in columns[1:-1]
            }
            assert set(tally) <= set(columns[1:-1])
            assert row[-1] == sum(bool(g["factor"]) for g in run.graded)
        columns, examples = app.examples_table(data, split)
        expected = []
        for run in runs:
            first = {}
            for g in run.graded:
                first.setdefault(g["verdict"], g)
            for verdict in grade.VERDICTS:
                if verdict in first:
                    g = first[verdict]
                    expected.append(
                        (hfspace.run_label(run), verdict, g["release_accession"])
                    )
        assert [(r[0], r[1], r[3]) for r in examples] == expected
    base = run_name(hf_context, "company_test", "base")
    _, rows = app.examples_table(data, "company_test", base, app.MISS)
    assert [(r[0], r[1], r[6], r[7], r[8]) for r in rows] == [
        ("Qwen3-8B base", "scale", "845,000,000", "845,000,000,000", "0.001")
    ]


# Leaderboard


def test_the_validation_leaderboard_is_comparison_md(app, data, hf_context, hf_repo):
    columns, rows = app.validation_board(data)
    table = hfcommon.validation_table(hf_context.validation_runs)
    # A GPU run's "runs on" is its GPU, as README's tables: stage 2's runs on the
    # validation GPU test_results.json names, the others on stage 1's A100.
    runs_on = {
        "2026-10-14T023302_claude-haiku-5-5_extract_v1": "Anthropic API",
        "qwen3-8b-2k--base": "1× H100 NVL",
        "qwen3-8b-2k--step-00025": "1× H100 NVL",
        "qwen3-8b-nothink-v2": "1× A100 80GB",
    }
    assert hf_repo.test_results["validation"]["gpu"] == "H100 NVL"
    assert rows == [
        [{**r, "runs_on": runs_on[r["run"]]}[k] for k in VALIDATION_KEYS] for r in table
    ]
    comparison = comparison_main_table(hf_repo.root / "results" / "comparison.md")
    assert len(comparison) == len(rows)
    for row, theirs in zip(rows, comparison, strict=True):
        mine = dict(zip(columns, row, strict=True))
        for column, their_column in COMPARISON_COLUMNS.items():
            assert mine[column] == theirs[their_column], (theirs["run"], column)
    board = data.leaderboard["validation"]
    assert board["runs"] == [r["run"] for r in table]
    notes = board["notes"]
    assert "98.0% (287/293), is chosen on validation" in notes
    # Unit slips as comparison.md defines them: not only the answer off by a power
    # of ten.
    assert (
        "- **Unit slips:** the figures the grader matched with a power-of-ten factor "
        "(scale and zeros, and the other errors matched only after multiplying or "
        "dividing the prediction by 10 to 1,000,000)."
    ) in notes
    assert "match the answer once multiplied" not in notes
    # Two GPUs at one price per GPU-hour: costs and latencies compare within one.
    assert (
        "one A100 80GB for stage 1's open models and one H100 NVL (Vast, billed "
        "$2.776/h) for the qwen3-8b-2k runs (Qwen3-8B base and the step-* "
        "checkpoints)"
    ) in notes
    assert "compare GPU costs and latencies only between runs on the same GPU" in notes
    assert "so they compare with each other" not in notes
    assert "Repeated runs" not in notes


def test_the_test_leaderboard_reads_as_test_results_md(app, data, hf_repo):
    results = hf_repo.test_results
    boards = app.test_boards(data)
    assert [b["name"] for b in boards] == ["company_test", "time_test"]
    runs_on = {
        "step-00025": "1× H100 NVL",  # model_under_test.gpus, as README's table
        "base": "1× H100 NVL",
        "gpt-6.1-sol": "OpenAI API",
        "claude-haiku-5-5": "Anthropic API",
        "gpt-6-luna": "OpenAI API",
    }
    for board in boards:
        s = results["splits"][board["name"]]
        assert board["keys"] == list(results["models"])
        for key, row in zip(board["keys"], board["rows"], strict=True):
            assert row[0] == results["models"][key]
            assert row[1] == runs_on[key]
            accuracy = s["accuracy"][key]
            assert row[2:6] == [
                testreport.acc_cell(accuracy[sub])
                for sub in ("all", *grade.FIELD_NAMES)
            ]
            for cell in row[2:6]:
                assert f" {cell} |" in TEST_RESULTS_MD  # a cell of its tables
            assert row[6].startswith(testreport.money(s["cost"][key]["per_1000"]))
        assert board["differences"].splitlines() == testreport.difference_table(
            s, "all", results["models"], "step-00025", "base"
        )
        for line in board["differences"].splitlines()[2:]:
            assert line in TEST_RESULTS_MD
    company = boards[0]
    assert company["rows"][0][6] == "$2.66 (about $0.56 without the freeze)"
    assert company["rows"][1][6] == "$0.54"
    assert boards[1]["title"] == (
        "Time test: 346 releases from 344 companies (9 more left out for every model)"
    )
    primary = data.leaderboard["test"]["primary"]
    assert (
        primary.removeprefix("**Primary result (pre-registered).** ") in TEST_RESULTS_MD
    )
    summary = data.leaderboard["test"]["summary"]
    assert "trails all three API models, by 1.78 to 1.95 points" in summary
    assert "$0.59 per 1,000 releases, against $1.24 for GPT-6 Luna" in summary
    # The price by the protocol's letter, beside the price billed (README gives both).
    cost = results["splits"]["company_test"]["cost"]
    notes = data.leaderboard["test"]["notes"]
    assert (
        "The company test's GPU runs record $2.776/h (an example price passed to "
        "vast_setup.sh by mistake), the price by the protocol's letter; at it, per "
        f"1,000 releases: step-00025 "
        f"{testreport.money(cost['step-00025']['per_1000_at_recorded'])} and Qwen3-8B "
        f"base {testreport.money(cost['base']['per_1000_at_recorded'])}."
    ) in notes
    assert "$2.91" in notes and "time test's GPU runs record" not in notes
    assert "on one Vast H100 NVL per test split" in notes


def test_tabs_from_the_url_query(app, data):
    tabs = app.available_tabs(data)
    assert tabs == ["leaderboard", "explore", "errors", "about"]
    assert app.tab_from_query({"tab": "errors"}, tabs) == "errors"
    assert app.tab_from_query({"tab": " About "}, tabs) == "about"
    assert app.tab_from_query({"tab": "serving"}, tabs) == "leaderboard"
    assert app.tab_from_query({}, tabs) == "leaderboard"


def test_meta_describes_every_verdict_and_split(data, hf_context):
    assert [v["name"] for v in data.meta["verdicts"]] == list(grade.VERDICTS)
    assert {v["name"] for v in data.meta["verdicts"] if not v["graded"]} == set(
        grade.UNGRADED
    )
    scale = next(v for v in data.meta["verdicts"] if v["name"] == "scale")
    assert "1,000 or 1,000,000" in scale["meaning"]
    counts = {s["name"]: (s["kept"], s["left_out"]) for s in data.meta["splits"]}
    assert counts == {"validation": (2, 0), "company_test": (2, 0), "time_test": (2, 1)}
    about = data.meta["about"]
    assert "at most 100 days before it" in about
    assert "whole millions, millions to one decimal or whole thousands" in about
    assert "39,936 tokens" in about and "1 more is left out" in about
    assert "neither step-00025 nor Qwen3-8B base ever answered null" in about
    assert "left 31-50% of the not_in_input figures empty" in about
    assert "Unit slips are the figures the grader matched with a power-of-ten" in about


def test_about_lists_the_limitations_and_the_failures(data, hf_repo):
    """README's limitations, and where the fine-tuned model fails, with the counts
    of test_results.json (the mini repo has the real one)."""
    results = hf_repo.test_results
    about = data.meta["about"]
    company, time_test = (
        results["splits"][s]["miss_summary"] for s in ("company_test", "time_test")
    )
    common = testreport.COMMON
    expected = [
        (
            "- **Small, and not held out:** 11 graded figures from 2 companies, and "
            "prompt v2 was designed on these releases"
        ),
        (
            "- **No Q3 2026 releases in the time test:** it holds the releases filed "
            "2026-07-01 to 2026-09-30"
        ),
        (
            "Its cutoff was set on 2026-11-07, after the company test was scored and "
            "before any time-test call, though the protocol said"
        ),
        "results/test_results.md#deviations-from-the-protocol",
        "- **Answer-key gaps:** the company test has no XBRL answer for 28 of the 728",
        (
            "- **Company test release CCI 0001051470-26-000052** is a divestiture and "
            "outlook announcement, not results; none of its 3 answers is printed, so it "
            "adds no graded figure."
        ),
        # validation.json here saves two checkpoints, step-00025 first.
        (
            "- **One training run, untuned:** the checkpoint chosen on validation, "
            "step-00025, is the first of the 2 it saved."
        ),
        "- **Survivorship:**",
        "- **Abstention isn't scored:**",
        "- **Fourth-quarter EPS isn't graded:**",
        "- **Sampling:** one run per model and split.",
        "- **GPU cost:** each run's wall-clock time only",
        (
            f"Time test: {time_test['groups'][common]} of step-00025's "
            f"{time_test['misses']} misses, from "
            f"{len(time_test['common_companies'])} companies; "
            f"{time_test['common_in_preferred']} are in releases whose filing reports "
            f"preferred dividends and {time_test['common_outside_preferred']} are not, "
            "so the preferred-dividend sensitivity understates it; they include "
            f"{time_test['broken_common']} of the {time_test['broken']} figures training "
            "broke (the base model had them right)."
        ),
        (
            f"Company test: {company['groups'][common]} of its {company['misses']} misses, "
            "from BA; all are in releases whose filing reports preferred dividends"
        ),
        (
            "repeated the quarter: 2 figures on the company test, from WY; 1 figure on "
            "the time test, from TWLO."
        ),
    ]
    for text in expected:
        assert text in about, text
    # Every company drawn for the company test has a release here.
    assert "companies drawn" not in about


def test_about_names_drawn_companies_without_a_release(hf_repo_copy, tmp_path):
    companies = hf_repo_copy.root / "data" / "companies.csv"
    companies.write_text(
        companies.read_text() + "ZZZ,1999,Omega Inc,Utilities,company_test\n"
    )
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "u")
    out = tmp_path / "space"
    hfspace.build_space(context, out)
    about = json.loads((out / "data" / "meta.json").read_text())["about"]
    assert (
        "- **2 of the 3 companies drawn for the company test:** ZZZ has no linked "
        "release."
    ) in about


# Serving


@pytest.fixture
def serving_build(hf_repo_copy, tmp_path):
    results = hf_fixture.add_serving_results(hf_repo_copy.root)
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "test-user")
    out = tmp_path / "space"
    return results, out, hfspace.build_space(context, out)


def test_serving_data_without_commands_or_paths(serving_build):
    results, out, written = serving_build
    assert written.skipped == []
    assert (out / "data/charts/serving_throughput_latency.png").is_file()
    assert (
        "- **Serving:** the fine-tuned model's throughput, latency, cost and accuracy "
        "on vLLM, as an adapter, merged, FP8 and 4-bit AWQ."
    ) in (out / "README.md").read_text(encoding="utf-8")
    text = (out / "data" / "serving.json").read_text(encoding="utf-8")
    for private in ("server_command", "vllm", "MODEL", "training/", "runs/", "error"):
        assert private not in text, private
    app = load_app(out)
    data = app.load_data(out / "data")
    assert app.available_tabs(data) == [
        "leaderboard",
        "explore",
        "errors",
        "serving",
        "about",
    ]
    columns, rows = app.serving_table(data, "summary")
    merged = dict(zip(columns, rows[1], strict=True))
    variant = results["variants"]["merged"]
    best = variant["best"]
    assert merged["best releases/hour"] == (
        f"{servereport.fmt(best['releases_per_hour'], ',.0f')} at {best['level']} in "
        "flight"
    )
    assert merged["$/1,000 at $2.5/h"] == servereport.fmt(
        best["cost_per_1000_usd"], ".3f"
    )
    # Accuracy, with each variant's agreement with the adapter variant and with
    # stage 2's runs (adapter's own: the check that serving reproduces stage 2).
    columns, rows = app.serving_table(data, "accuracy")
    by_variant = {row[0]: dict(zip(columns, row, strict=True)) for row in rows}
    assert list(by_variant) == ["adapter", "merged", "fp8", "int4-awq"]
    merged = by_variant["merged"]
    assert merged["validation accuracy"] == "286/293 (97.6%)"
    assert merged["same as adapter (validation)"] == "359/360 figures, 59/60 releases"
    for name, row in by_variant.items():
        for split, title in (("validation", "validation"), ("time_test", "time test")):
            assert row[f"same as stage 2 ({title})"] == servereport.agreement_text(
                results["variants"][name]["agreement"][split]["with_stage2"]
            )
    adapter = by_variant["adapter"]
    assert adapter["same as adapter (time test)"] == "(itself)"
    assert adapter["same as stage 2 (validation)"] == (
        "360/360 figures, 60/60 releases"
    )
    assert "Same as stage 2:" in data.serving["notes"]
    columns, rows = app.serving_table(data, "memory")
    assert rows[3] == ["int4-awq", "5.60", "70.00", "566,000", "13.80x", "85"]
    columns, rows = app.serving_table(data, "level")
    assert len(rows) == 4 * 3 and rows[0][:3] == ["adapter", "1", "128/128"]
    assert "1, 16 or 64 requests in flight" in data.serving["intro"]


def test_a_failed_serving_variant(hf_repo_copy, tmp_path):
    """The README names only the variants that completed; the tables show the failed
    one's status with empty cells."""
    results = hf_fixture.add_serving_results(hf_repo_copy.root)
    fp8 = results["variants"]["fp8"]
    fp8.update(
        status="failed",
        error="server exited",
        startup=None,
        levels=[],
        best=None,
        accuracy={},
        agreement={},
    )
    path = hf_repo_copy.root / "results" / "serving_results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "u")
    out = tmp_path / "space"
    hfspace.build_space(context, out)
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "on vLLM, as an adapter, merged and 4-bit AWQ." in readme
    serving = json.loads((out / "data" / "serving.json").read_text())
    assert "in 4 variants, one at a time (fp8 failed)." in serving["intro"]
    assert serving["summary"][2] == ["fp8", "failed", "–", "–", "–", "–"]
    assert serving["accuracy"][2] == ["fp8", "–", "–", "–", "–", "–", "–"]
    assert "server exited" not in json.dumps(serving)
    assert hfspace.serving_bullet(None) == []
    for v in results["variants"].values():
        v["status"] = "skipped"
    assert hfspace.serving_bullet(results) == [
        (
            "- **Serving:** the fine-tuned model's throughput, latency, cost and accuracy "
            "on vLLM: no variant completed."
        )
    ]


def test_serving_without_its_chart(hf_repo_copy, tmp_path):
    hf_fixture.add_serving_results(hf_repo_copy.root)
    (hf_repo_copy.root / "results" / "serving_throughput_latency.png").unlink()
    context = hfcommon.load_context(Sources(hf_repo_copy.root), "u")
    written = hfspace.build_space(context, tmp_path / "space")
    assert written.skipped == [
        "Space's serving chart: results/serving_throughput_latency.png not found"
    ]
    serving = json.loads((tmp_path / "space/data/serving.json").read_text())
    assert serving["chart"] is None


def test_html_table(app):
    columns = ["model", "wrong_period", "cost / 1,000", "latency", "accuracy", "note"]
    rows = [
        [
            "GPT <6>",
            2,
            "$1.23",
            "3.29 s",
            "293/293 (100.0%)",
            "657/662 (99.24%) [98.05, 100.00]",
        ],
        ["Qwen3-8B base", 10, "$0.54", "1.10 s", "262/293 (89.4%)", "x" * 50],
        [
            "step-00025",
            None,
            "\u22120.76",
            "",
            "287/293 (98.0%)",
            "$2.66 (about $0.56 a)",
        ],
    ]
    out = app.html_table(columns, rows)
    assert out.startswith('<div style="overflow-x:auto"><table')
    assert "wrong_<wbr>period" in out  # a header breaks at its underscores only
    assert "GPT &lt;6&gt;" in out
    # Columns of numbers (counts, dollars, seconds) are right-aligned.
    aligned = re.findall(r'<th style="[^"]*text-align:(\w+)', out)
    assert aligned == ["left", "right", "right", "right", "left", "left"]
    # Only the 32-character cell with spaces may wrap, and only between its parts.
    assert out.count("white-space:nowrap") == 17
    assert out.count("white-space:normal") == 1
    assert "657/662\u00a0(99.24%) [98.05,\u00a0100.00]" in out
    boxed = app.html_table(columns, rows, max_height=700)
    assert boxed.startswith(
        '<div style="overflow-x:auto;max-height:700px;overflow-y:auto">'
    )
    assert app.html_table(columns, []) == "<p>Nothing to show.</p>"


def test_the_answer_list_shows_a_table_row_when_there_is_one(app):
    text = "Capital spending of about $600 million.\n| Net income | $600 | $458 |\n"
    first = text.index("600")
    prose = (first, first + 3)
    row = (text.index("600", first + 3), text.index("600", first + 3) + 3)
    assert app.shown_span(text, [prose, row]) == row
    assert app.shown_span(text, [prose]) == prose


def test_the_release_header_says_whether_models_saw_it(app, data, hf_context):
    aaa = accession(hf_context, "AAA")
    tokens = data.index["validation"][aaa]["prompt_tokens"]
    header = app.release_header(data, "validation", aaa)
    assert f"Sent to every model: its prompt is {tokens:,} tokens." in header
    ggg = accession(hf_context, "GGG")
    assert "Left out for every model: its prompt is 45,000 tokens" in (
        app.release_header(data, "time_test", ggg)
    )


# The Gradio app (the hf group; CI has no gradio)


def tab_ids(demo, gr) -> list[str]:
    return [b.id for b in demo.blocks.values() if isinstance(b, gr.Tab)]


def handler(demo, name: str):
    return next(f.fn for f in demo.fns.values() if f.name == name)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_the_blocks_and_their_tabs(app, data, hf_context):
    gr = pytest.importorskip("gradio")
    demo = app.build_demo(data)
    assert tab_ids(demo, gr) == ["leaderboard", "explore", "errors", "about"]
    on_load = handler(demo, "on_load")
    tabs, split, *_ = on_load(gr.Request(query_params={"tab": "errors"}))
    assert (tabs.selected, split.value) == ("errors", "company_test")
    tabs, *_ = on_load(gr.Request(query_params={"tab": "serving"}))
    assert tabs.selected == "leaderboard"
    ggg = accession(hf_context, "GGG")
    updates = on_load(gr.Request(query_params={"tab": "explore", "release": ggg}))
    assert updates[0].selected == "explore"
    assert (updates[1].value, updates[5].value, updates[6].value) == (
        "time_test",
        "GGG",
        ggg,
    )
    assert "Left out for every model" in updates[9]
    # The Errors tab's tables are HTML; the examples scroll in their box.
    counts, _, _, examples = handler(demo, "on_error_split")("time_test")
    assert counts.startswith('<div style="overflow-x:auto"><table')
    assert examples.startswith(
        f'<div style="overflow-x:auto;max-height:{app.EXAMPLES_HEIGHT}px;'
    )
    assert handler(demo, "on_example_filter")("time_test", app.ANY, app.ANY) == (
        examples
    )


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_the_blocks_with_serving_data(serving_build):
    gr = pytest.importorskip("gradio")
    _, out, _ = serving_build
    app = load_app(out)
    data = app.load_data(out / "data")
    demo = app.build_demo(data)
    assert tab_ids(demo, gr) == ["leaderboard", "explore", "errors", "serving", "about"]
    tabs, *_ = handler(demo, "on_load")(gr.Request(query_params={"tab": "serving"}))
    assert tabs.selected == "serving"
    shown = [b.value for b in demo.blocks.values() if isinstance(b, gr.HTML)]
    for name in ("summary", "accuracy", "memory", "level"):
        assert app.html_table(*app.serving_table(data, name)) in shown, name


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
def real(tmp_path_factory):
    context = hfcommon.load_context(REAL, hfcommon.PLACEHOLDER_USER)
    out = tmp_path_factory.mktemp("real-space") / "space"
    written = hfspace.build_space(context, out)
    return context, out, written


@needs_real_data
def test_real_space_folder(real):
    _, out, written = real
    assert sum(p.stat().st_size for p in written.files) < 100 * 2**20
    releases = json.loads((out / "data" / "releases.json").read_text())
    assert {s: len(r) for s, r in releases.items()} == {
        "validation": 60,
        "company_test": 148,
        "time_test": 355,
    }
    assert {s: sum(not x["kept"] for x in r) for s, r in releases.items()} == {
        "validation": 0,
        "company_test": 0,
        "time_test": 9,
    }
    reasons = {
        r["ticker"]: r["left_out_reason"]
        for r in releases["time_test"]
        if not r["kept"]
    }
    assert reasons["UDR"] == (
        "its prompt is 50,970 tokens, over the 39,936-token limit (the GPU model's "
        "context of 40,960 tokens minus 1,024 for the reply)"
    )
    answers = json.loads((out / "data" / "answers.json").read_text())
    assert {s: sum(map(len, a.values())) for s, a in answers.items()} == {
        "validation": 293,
        "company_test": 700,
        "time_test": 1792,
    }
    assert len(list((out / "data" / "texts").rglob("*.md"))) == 563
    home = str(Path.home())
    for path in out.rglob("*"):
        if path.suffix in (".json", ".md", ".py", ".txt"):
            text = path.read_text(encoding="utf-8")
            assert home not in text and "/workspace/" not in text, path
            if path.suffix == ".json":
                assert "@" not in text, path
            if "texts" not in path.parts:  # PLAN.md isn't published
                assert "PLAN.md" not in text, path


@needs_real_data
def test_real_leaderboard_is_comparison_md_and_test_results(real):
    context, out, _ = real
    app = load_app(out)
    data = app.load_data(out / "data")
    columns, rows = app.validation_board(data)
    comparison = comparison_main_table(REAL.comparison)
    assert len(rows) == len(comparison) == 24
    for row, theirs in zip(rows, comparison, strict=True):
        mine = dict(zip(columns, row, strict=True))
        for column, their_column in COMPARISON_COLUMNS.items():
            assert mine[column] == theirs[their_column], (theirs["run"], column)
    # The GPU each GPU run had, as README's tables write it: stage 1's open models
    # on the A100, stage 2's runs on the H100 NVL.
    runs = data.leaderboard["validation"]["runs"]
    runs_on = dict(zip(runs, (row[1] for row in rows), strict=True))
    assert (
        runs_on["qwen3-8b-nothink-v2"] == runs_on["gpt-oss-20b-v2"] == ("1× A100 80GB")
    )
    assert (
        runs_on["qwen3-8b-2k--base"]
        == runs_on["qwen3-8b-2k--step-00125"]
        == ("1× H100 NVL")
    )
    assert runs_on["2026-10-14T023027_claude-opus-5-5_extract_v1"] == "Anthropic API"
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for place in set(runs_on.values()):
        assert f"| {place} |" in readme, place
    assert app.explore_state(data, "time_test").status == (
        "355 releases in the time test split, 9 of them left out for every model."
    )
    assert app.explore_state(data, "validation").status == (
        "60 releases in the validation split."
    )
    results = context.test_results
    for board in app.test_boards(data):
        for row in board["rows"]:
            for cell in row[2:6]:
                assert cell in TEST_RESULTS_MD
    company, time_test = app.test_boards(data)
    assert company["rows"][0][2] == "657/662 (99.24%) [98.05, 100.00]"
    assert time_test["rows"][0][2] == "1,699/1,740 (97.64%) [96.64, 98.57]"
    primary = data.leaderboard["test"]["primary"]
    assert "**+10.12 points [+5.08, +15.58]**" in primary
    assert (
        primary.removeprefix("**Primary result (pre-registered).** ") in TEST_RESULTS_MD
    )
    # Explore shows the test sets' models' validation runs first.
    assert app.default_runs(data, "validation") == [
        c["run"] for c in results["validation"]["counts"].values()
    ]
    # The Errors tab's counts are test_results.json's, from the same graded.csv.
    for split in ("company_test", "time_test"):
        for run in context.runs(split):
            counts = data.errors[split]["counts"][run.name]
            assert counts == results["splits"][split]["verdicts"][run.key]
            assert (
                data.errors[split]["unit_slips"][run.name]
                == results["splits"][split]["unit_slips"][run.key]
            )


@needs_real_data
def test_real_about_has_readmes_limitations_and_failures(real):
    """The facts of README's Limitations and Failure modes, as the About tab says
    them from test_results.json, validation.json and data/companies.csv."""
    _, out, _ = real
    about = json.loads((out / "data" / "meta.json").read_text())["about"]
    for text in (
        "- **Small, and not held out:** 293 graded figures from 10 companies",
        "- **No Q3 2026 releases in the time test:**",
        (
            "- **Answer-key gaps:** the company test has no XBRL answer for 28 of the "
            "728 figures its kept releases should report (3.8%); the time test has no "
            "XBRL answer for 124 of the 1,916 figures its kept releases should report "
            "(6.5%)."
        ),
        (
            "- **Company test release CCI 0001051470-26-000052** (Crown Castle) is a "
            "divestiture and outlook announcement, not results; none of its 3 answers "
            "is printed, so it adds no graded figure."
        ),
        (
            "- **37 of the 40 companies drawn for the company test:** AEP, FDXF and VYLR "
            "have no linked release."
        ),
        (
            "- **One training run, untuned:** the checkpoint chosen on validation, "
            "step-00025, is the first of the 5 it saved."
        ),
        (
            "Time test: 29 of step-00025's 41 misses, from 16 companies; 20 are in "
            "releases whose filing reports preferred dividends and 9 are not"
        ),
        "they include 11 of the 14 figures training broke",
        (
            "Company test: 3 of its 5 misses, from Boeing (BA); all are in releases "
            "whose filing reports preferred dividends"
        ),
        (
            "2 figures on the company test, from Weyerhaeuser (WY); 1 figure on the "
            "time test, from Twilio (TWLO)."
        ),
    ):
        assert text in about, text


@needs_real_data
def test_real_misses_are_found_by_the_filters(real):
    context, out, _ = real
    app = load_app(out)
    data = app.load_data(out / "data")
    for split in ("company_test", "time_test"):
        step = run_name(context, split, "step-00025")
        found = {
            r["release_accession"]
            for r in app.matching_releases(data, split, step, app.MISS)
        }
        misses = context.test_results["splits"][split]["misses"]
        assert found == {m["release_accession"] for m in misses}
        miss = misses[0]
        rows = app.figure_rows(data, split, miss["release_accession"], [step])
        cell = next(
            row.cells[0]
            for row in rows
            if (row.field, row.period) == (miss["field"], miss["period_type"])
        )
        assert cell.kind in (app.ERROR, app.SLIP)
        assert cell.prediction["predicted"] == miss["predicted"]
        assert cell.prediction["verdict"] == miss["verdict"]
