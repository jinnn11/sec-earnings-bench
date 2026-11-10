"""Tests for secbench.hfspace and secbench.hfsite, the results explorer's Space (a
static site), on the small repository of tests/hf_fixture.py: the built folder (files,
README metadata, sizes, determinism, privacy), the explorer's logic on the built data
(choices, filters, the figures table and its colours, the highlighted text, the Errors
and Leaderboard tables), and the pages (their tabs, links, markup, the filters' data
and the hooks the script needs). The last tests build the Space from the real data and
skip without it."""

from __future__ import annotations

import dataclasses
import html
import json
import re
import shutil
import urllib.parse
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import hf_fixture
import pytest
from test_hfcommon import comparison_main_table

from secbench import grade, hfcommon, hfsite, hfspace, servereport, splits, testreport
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


def relative_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def built(hf_context, tmp_path_factory):
    out = tmp_path_factory.mktemp("space") / "space"
    return out, hfspace.build_space(hf_context, out)


@pytest.fixture(scope="module")
def data(built):
    return hfsite.load_data(built[0] / "data")


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


def site_pages(context) -> set[str]:
    """The pages of a build without serving results: one per tab, one per release."""
    return {
        "index.html",
        "explore.html",
        "errors.html",
        "about.html",
        *(
            f"release/{r.release_accession}.html"
            for rows in context.releases.values()
            for r in rows
        ),
    }


def test_the_folder_holds_its_readme_pages_assets_and_data(hf_context, built):
    out, written = built
    texts = {
        f"data/texts/{split}/{r.release_accession}.md"
        for split, rows in hf_context.releases.items()
        for r in rows
    }
    assets = {"assets/style.css", "assets/explorer.js"}
    expected = {"README.md", *assets, *site_pages(hf_context), *DATA_FILES, *texts}
    assert relative_files(out) == expected
    assert {p.relative_to(out).as_posix() for p in written.files} == expected
    assert written.skipped == [
        "Space's serving tab: results/serving_results.json not found"
    ]
    for name in ("style.css", "explorer.js"):
        assert (out / "assets" / name).read_bytes() == (SPACE / name).read_bytes()
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


def test_readme_metadata(built):
    out, _ = built
    readme = (out / "README.md").read_text(encoding="utf-8")
    meta = front_matter(readme)
    assert meta == {
        "title": "SEC Earnings Explorer",
        "emoji": "📊",
        "colorFrom": "blue",
        "colorTo": "indigo",
        "sdk": "static",
        "app_file": "index.html",
        "pinned": "false",
        "license": "mit",
        "short_description": hfspace.SHORT_DESCRIPTION,
        "datasets": ["test-user/sec-earnings-bench"],
        "models": ["test-user/qwen3-8b-sec-earnings-lora"],
    }
    assert len(meta["short_description"]) <= 60
    assert "The Space is a static site" in readme
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


# Explore: choices and filters


def test_split_and_model_choices(data, hf_context, hf_repo):
    assert hfsite.split_choices(data) == [
        ("Validation", "validation"),
        ("Company test", "company_test"),
        ("Time test", "time_test"),
    ]
    assert hfsite.run_choices(data, "company_test") == [
        (label, run)
        for label, run in zip(
            hf_repo.test_results["models"].values(),
            hf_repo.test_runs["company_test"],
            strict=True,
        )
    ]
    assert [label for label, _ in hfsite.run_choices(data, "validation")] == [
        "Claude Haiku 5.5 · v1",
        "Qwen3-8B base · v2",
        "step-00025 · v2",
        "Qwen3-8B (no thinking) · v2",
    ]
    # Validation shows the test sets' models' validation runs first, in their order:
    # here only stage 2's two are in the mini repo.
    assert hfsite.default_runs(data, "validation") == [
        "qwen3-8b-2k--step-00025",
        "qwen3-8b-2k--base",
    ]
    assert hfsite.default_runs(data, "time_test") == hf_repo.test_runs["time_test"]
    verdicts = [v for _, v in hfsite.verdict_filter_choices(data, "company_test")]
    assert verdicts == [
        hfsite.ANY,
        hfsite.MISS,
        "correct",
        "scale",
        "wrong_line",
        "not_in_input",
        "no_answer",
    ]


def test_filters_by_a_models_misses_or_a_verdict(data, hf_context):
    def tickers(split, model=hfsite.ANY, verdict=hfsite.ANY):
        return [
            r["ticker"] for r in hfsite.matching_releases(data, split, model, verdict)
        ]

    base = run_name(hf_context, "company_test", "base")
    step = run_name(hf_context, "company_test", "step-00025")
    assert tickers("company_test", base, hfsite.MISS) == ["CCC"]  # a scale error
    assert tickers("company_test", step, hfsite.MISS) == ["DDD"]  # the common line
    assert tickers("company_test", step, "scale") == []
    assert tickers("company_test", verdict="not_in_input") == ["CCC"]
    assert tickers("time_test", verdict="adjusted") == ["EEE"]
    assert tickers("time_test") == ["EEE", "FFF", "GGG"]  # left out too
    assert tickers("validation", verdict=hfsite.MISS) == ["AAA"]


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
    data = hfsite.load_data(out / "data")
    labels = {run: label for label, run in hfsite.run_choices(data, "validation")}
    assert labels == {
        "2026-10-14T023302_claude-haiku-5-5_extract_v1": "Claude Haiku 5.5 · v1",
        "qwen3-8b-2k--base": "Qwen3-8B base · v2",
        "qwen3-8b-2k--step-00025": "step-00025 · v2",
        "qwen3-8b-nothink-v2": "Qwen3-8B (no thinking) · v2 · qwen3-8b-nothink-v2",
        repeat.name: f"Qwen3-8B (no thinking) · v2 · {repeat.name}",
    }
    _, rows = hfsite.verdict_table(data, "validation")
    assert sorted(row[0] for row in rows) == sorted(labels.values())
    board = data.leaderboard["validation"]
    models = dict(zip(board["runs"], (row[0] for row in board["rows"]), strict=True))
    assert models[repeat.name] == f"Qwen3-8B (no thinking) · {repeat.name}"
    assert models["qwen3-8b-2k--base"] == "Qwen3-8B base"
    assert (
        "- **Repeated runs** of one model and prompt show their run's name"
        in (board["notes"])
    )


def test_left_out_releases_say_why_without_citing_the_plan(data, hf_context, built):
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
    header = hfsite.release_header(data, "time_test", ggg.release_accession)
    assert f"Left out for every model: {html.escape(reason)}." in header
    assert html.escape(reason) in hfsite.figures_table(
        data, "time_test", ggg.release_accession, []
    )
    for path in out.rglob("*"):
        if path.suffix in (".json", ".md", ".html") and not (
            {"texts", "release"} & set(path.relative_to(out).parts)
        ):
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


def test_the_figures_table_and_its_colours(data, hf_context):
    step = run_name(hf_context, "company_test", "step-00025")
    base = run_name(hf_context, "company_test", "base")
    ddd = accession(hf_context, "DDD")
    rows = hfsite.figure_rows(data, "company_test", ddd, [step, base])
    assert [(r.field, r.period) for r in rows] == [
        (f, p) for f in grade.FIELD_NAMES for p in ("quarter", "year_to_date")
    ]
    mine = cells_by_slot(rows, step)
    wrong = mine["net_income", "quarter"]
    assert (wrong.kind, wrong.prediction["verdict"], wrong.prediction["predicted"]) == (
        hfsite.ERROR,
        "wrong_line",
        "190000000",
    )
    assert mine["revenue", "quarter"].kind == hfsite.CORRECT
    assert cells_by_slot(rows, base)["net_income", "quarter"].kind == hfsite.CORRECT
    table = hfsite.figures_table(data, "company_test", ddd, [step, base])
    background = hfsite.KIND_COLOURS[hfsite.ERROR][0]
    evidence = html.escape(wrong.prediction["evidence"], quote=True)
    assert re.search(
        rf'background:{background};[^"]*" title="wrong_line: {re.escape(evidence)}"',
        table,
    )
    assert "190,000,000" in table and "200,000,000" in table

    ccc = accession(hf_context, "CCC")
    theirs = cells_by_slot(hfsite.figure_rows(data, "company_test", ccc, [base]), base)
    slip = theirs["revenue", "quarter"]
    assert (slip.kind, slip.prediction["factor"]) == (hfsite.SLIP, "0.001")
    assert theirs["eps_diluted", "quarter"].prediction["verdict"] == "not_in_input"
    assert theirs["eps_diluted", "quarter"].kind == hfsite.UNGRADED
    assert theirs["revenue", "year_to_date"].prediction["verdict"] == "no_answer"
    assert theirs["net_income", "year_to_date"].prediction is None  # null, no answer
    assert theirs["net_income", "year_to_date"].kind == hfsite.UNGRADED
    table = hfsite.figures_table(data, "company_test", ccc, [base])
    for colour in (hfsite.SLIP, hfsite.UNGRADED, hfsite.CORRECT):
        assert f"background:{hfsite.KIND_COLOURS[colour][0]}" in table
    assert "scale (×0.001)" in table and "not printed: not graded" in table

    # A fourth quarter: the full year's column, and no quarter EPS answer.
    bbb = accession(hf_context, "BBB")
    rows = hfsite.figure_rows(
        data, "validation", bbb, hfsite.default_runs(data, "validation")
    )
    assert [r.period for r in rows[:2]] == ["quarter", "full_year"]
    assert rows[4].answer is None and rows[0].answer["derived"]
    ggg = accession(hf_context, "GGG")
    assert hfsite.figure_rows(data, "time_test", ggg, ["x"]) == []
    assert "Left out for every model" in hfsite.figures_table(
        data, "time_test", ggg, []
    )


def marked(html_text: str) -> list[str]:
    return [
        html.unescape(m) for m in re.findall(r"<mark [^>]*>(.*?)</mark>", html_text)
    ]


def test_highlights_land_on_the_printed_form(data, hf_context):
    bbb, eee = accession(hf_context, "BBB"), accession(hf_context, "EEE")
    text = hfsite.read_release_text(data, "validation", bbb)
    shown = marked(hfsite.release_view(data, "validation", bbb, []).text)
    assert "(30,250)" in shown and "512,300" in shown and "0.59" in shown
    text = hfsite.read_release_text(data, "time_test", eee)
    view = hfsite.release_view(data, "time_test", eee, [])
    assert {"640", "1,255", "−15", "40", "(0.11)", "(0.29)"} <= set(marked(view.text))
    marks = hfsite.text_marks(data, "time_test", eee)
    loss = [m for m in marks if m[3].startswith("Net income, year to date")]
    assert [text[a:b] for a, b, _, _ in loss] == ["40"]
    number, line, start, end = hfsite.line_of(text, loss[0][0], loss[0][1])
    assert line.startswith("| Net loss |") and line[start:end] == "40"
    assert view.answers.count("<li>") == 6
    assert "40 on a loss row (Net loss" in html.unescape(view.answers)
    assert f"line {number}: <code>" in view.answers
    assert "<b>40</b>" in view.answers


def test_highlights_split_overlapping_marks_and_escape_the_text():
    text = "a<b>cdef"
    marks = [(0, 5, "revenue", "first"), (3, 8, "net_income", "second")]
    shown = hfsite.highlighted_text(text, marks)
    pieces = re.findall(r'<mark [^>]*title="([^"]*)">(.*?)</mark>', shown)
    assert [(html.unescape(t), html.unescape(x)) for t, x in pieces] == [
        ("first", "a<b"),
        ("first; second", ">c"),
        ("second", "def"),
    ]
    assert "<b>" not in shown.replace("&lt;b&gt;", "")
    assert hfsite.excerpt("x" * 500 + "123" + "y" * 500, 500, 503).count("…") == 2


def test_figures_are_exact_with_separators():
    assert hfsite.figure("1234000000") == "1,234,000,000"
    assert hfsite.figure("-30250000") == "-30,250,000"
    assert hfsite.figure("202000000.0") == "202,000,000.0"
    assert hfsite.figure("0.45") == "0.45"
    assert hfsite.figure("1.25E+7") == "12,500,000"
    assert hfsite.figure(None) == "null"


# Errors


def test_the_errors_tab_counts_graded_csv(data, hf_context):
    for split in SPLITS:
        runs = hf_context.runs(split)
        columns, rows = hfsite.verdict_table(data, split)
        assert columns[0] == "model" and columns[-1] == "unit slips"
        assert [row[0] for row in rows] == [hfspace.run_label(r) for r in runs]
        for run, row in zip(runs, rows, strict=True):
            tally = Counter(g["verdict"] for g in run.graded)
            assert dict(zip(columns[1:-1], row[1:-1], strict=True)) == {
                v: tally[v] for v in columns[1:-1]
            }
            assert set(tally) <= set(columns[1:-1])
            assert row[-1] == sum(bool(g["factor"]) for g in run.graded)
        columns, examples = hfsite.examples_table(data, split)
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
    _, rows = hfsite.examples_table(data, "company_test", base, hfsite.MISS)
    assert [(r[0], r[1], r[6], r[7], r[8]) for r in rows] == [
        ("Qwen3-8B base", "scale", "845,000,000", "845,000,000,000", "0.001")
    ]


# Leaderboard


def test_the_validation_leaderboard_is_comparison_md(data, hf_context, hf_repo):
    columns, rows = hfsite.validation_board(data)
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


def test_the_test_leaderboard_reads_as_test_results_md(data, hf_repo):
    results = hf_repo.test_results
    boards = hfsite.test_boards(data)
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
    data = hfsite.load_data(out / "data")
    assert hfsite.available_tabs(data) == [
        "leaderboard",
        "explore",
        "errors",
        "serving",
        "about",
    ]
    columns, rows = hfsite.serving_table(data, "summary")
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
    columns, rows = hfsite.serving_table(data, "accuracy")
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
    columns, rows = hfsite.serving_table(data, "memory")
    assert rows[3] == ["int4-awq", "5.60", "70.00", "566,000", "13.80x", "85"]
    columns, rows = hfsite.serving_table(data, "level")
    assert len(rows) == 4 * 3 and rows[0][:3] == ["adapter", "1", "128/128"]
    assert "1, 16 or 64 requests in flight" in data.serving["intro"]
    page = (out / "serving.html").read_text(encoding="utf-8")
    for name in ("summary", "accuracy", "memory", "level"):
        assert hfsite.html_table(*hfsite.serving_table(data, name)) in page, name
    assert 'href="serving.html" aria-current="page"' in page
    assert f'<img src="data/{data.serving["chart"]}"' in page
    assert 'href="serving.html">Serving</a>' in (out / "index.html").read_text(
        encoding="utf-8"
    )


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
    page = (tmp_path / "space/serving.html").read_text(encoding="utf-8")
    assert "<img " not in page and "<h3>Every load level</h3>" in page


def test_html_table():
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
    out = hfsite.html_table(columns, rows)
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
    boxed = hfsite.html_table(columns, rows, max_height=700)
    assert boxed.startswith(
        '<div style="overflow-x:auto;max-height:700px;overflow-y:auto">'
    )
    assert hfsite.html_table(columns, []) == "<p>Nothing to show.</p>"


def test_the_answer_list_shows_a_table_row_when_there_is_one():
    text = "Capital spending of about $600 million.\n| Net income | $600 | $458 |\n"
    first = text.index("600")
    prose = (first, first + 3)
    row = (text.index("600", first + 3), text.index("600", first + 3) + 3)
    assert hfsite.shown_span(text, [prose, row]) == row
    assert hfsite.shown_span(text, [prose]) == prose


def test_the_release_header_says_whether_models_saw_it(data, hf_context):
    aaa = accession(hf_context, "AAA")
    tokens = data.index["validation"][aaa]["prompt_tokens"]
    header = hfsite.release_header(data, "validation", aaa)
    assert f"Sent to every model: its prompt is {tokens:,} tokens." in header
    ggg = accession(hf_context, "GGG")
    assert "Left out for every model: its prompt is 45,000 tokens" in (
        hfsite.release_header(data, "time_test", ggg)
    )


# The pages

# Where a page may link off the site (in a new tab: the Hub shows a Space in a frame);
# nothing is loaded from anywhere else.
LINKED_HOSTS = {"github.com", "huggingface.co", "www.sec.gov"}
VOID_ELEMENTS = {"br", "img", "input", "link", "meta", "wbr"}


class Markup(HTMLParser):
    """A page's elements, checked to nest (each closed, in order), with their
    attributes."""

    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.open: list[str] = []
        self.problems: list[str] = []
        self.tags: list[tuple[str, dict]] = []
        self.feed(text)
        self.close()
        if self.open:
            self.problems.append(f"never closed: {self.open}")

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag not in VOID_ELEMENTS:
            self.open.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.problems.append(f"<{tag} /> (no element here closes itself)")

    def handle_endtag(self, tag):
        if tag in VOID_ELEMENTS or not self.open or self.open[-1] != tag:
            self.problems.append(f"</{tag}> where {self.open[-1:]} is open")
        else:
            self.open.pop()


def page_files(out: Path) -> list[Path]:
    return sorted(out.glob("*.html")) + sorted((out / "release").glob("*.html"))


def sections(text: str) -> dict[str, str]:
    """A page's split sections, by split, in order."""
    found = re.findall(
        r'<section class="split(?: current)?" data-split="(\w+)">(.*?)</section>',
        text,
        re.DOTALL,
    )
    return dict(found)


def test_every_page_nests_and_links_only_where_it_should(built):
    """Every element closed in order; every link and file on the site found; off
    the site, only links (to GitHub, the Hub and EDGAR), each in a new tab."""
    out, _ = built
    pages = page_files(out)
    assert len(pages) == len(list(out.rglob("*.html")))
    for path in pages:
        text = path.read_text(encoding="utf-8")
        assert text.startswith('<!doctype html>\n<html lang="en">\n'), path
        markup = Markup(text)
        assert markup.problems == [], (path, markup.problems[:3])
        for tag, attrs in markup.tags:
            for name in ("href", "src"):
                url = attrs.get(name)
                if url is None:
                    continue
                if url.startswith("data:"):
                    assert (tag, attrs.get("rel")) == ("link", "icon"), (path, url)
                    continue
                parts = urllib.parse.urlsplit(url)
                if parts.scheme:
                    assert parts.scheme == "https", (path, url)
                    assert parts.hostname in LINKED_HOSTS, (path, url)
                    assert tag == "a", (path, url)
                    assert (attrs.get("target"), attrs.get("rel")) == (
                        "_blank",
                        "noopener",
                    ), (path, url)
                    continue
                assert parts.path, (path, url)
                target = (path.parent / urllib.parse.unquote(parts.path)).resolve()
                assert target.is_file(), (path, url)
                assert out.resolve() in target.parents, (path, url)


def test_each_page_names_its_tab(built, data, hf_context):
    out, _ = built
    tabs = hfsite.available_tabs(data)
    assert tabs == ["leaderboard", "explore", "errors", "about"]
    pages = {hfsite.PAGES[t]: t for t in tabs}
    ddd = accession(hf_context, "DDD")
    for name, tab, root in [
        *((name, tab, "") for name, tab in pages.items()),
        (f"release/{ddd}.html", "explore", "../"),
    ]:
        text = (out / name).read_text(encoding="utf-8")
        nav = re.search(r'<nav class="tabs" aria-label="Pages">(.*?)</nav>', text)[1]
        assert re.findall(r'href="([^"]+)"', nav) == [root + p for p in pages], name
        assert re.findall(r'href="([^"]+)" aria-current="page"', nav) == [
            root + hfsite.PAGES[tab]
        ], name
        assert f'<link rel="stylesheet" href="{root}assets/style.css">' in text
        assert f'<script src="{root}assets/explorer.js" defer></script>' in text
        title = re.search(r"<title>(.*?)</title>", text)[1]
        assert title.endswith(f" · {html.escape(data.meta['title'])}"), name
    assert "<title>Errors · " in (out / "errors.html").read_text(encoding="utf-8")


def test_the_explore_page_lists_every_release_and_what_each_filter_matches(
    built, data, hf_context
):
    """Each split's releases by ticker and date, each linked to its page; for each
    model and verdict choice, the build's list of the releases that match, which is
    matching_releases' (the script only looks it up)."""
    out, _ = built
    text = (out / "explore.html").read_text(encoding="utf-8")
    found = sections(text)
    assert list(found) == list(SPLITS)
    assert '<section class="split current" data-split="company_test">' in text
    assert text.count("split current") == 1  # the default split first
    for split, body in found.items():
        order = hfsite.release_order(data, split)
        rows = re.findall(
            r'<tr data-i="(\d+)" data-ticker="([^"]+)" data-search="([^"]*)">'
            r'<td><a href="release/([\d-]+)\.html">',
            body,
        )
        assert [int(i) for i, *_ in rows] == list(range(len(order)))
        assert [a for *_, a in rows] == [r["release_accession"] for r in order]
        tickers = [t for _, t, _, _ in rows]
        assert tickers == sorted(tickers)
        for (_, _, search, _), r in zip(rows, order, strict=True):
            assert search == html.escape(f"{r['ticker']} {r['company']}".lower())
        models = [m for _, m in hfsite.model_filter_choices(data, split)]
        verdicts = [v for _, v in hfsite.verdict_filter_choices(data, split)]
        for name, choices in (
            ("model", hfsite.model_filter_choices(data, split)),
            ("verdict", hfsite.verdict_filter_choices(data, split)),
        ):
            assert (
                f'<select data-filter="{name}">{hfsite.options(choices)}</select>'
                in (body)
            )
        matches = json.loads(
            re.search(
                r'<script type="application/json" class="filter-data">(.*?)</script>',
                body,
                re.DOTALL,
            )[1]
        )
        assert list(matches) == models
        for model in models:
            wanted = [v for v in verdicts if (model, v) != (hfsite.ANY, hfsite.ANY)]
            assert list(matches[model]) == wanted
            for verdict in wanted:
                shown = {order[i]["release_accession"] for i in matches[model][verdict]}
                assert shown == {
                    r["release_accession"]
                    for r in hfsite.matching_releases(data, split, model, verdict)
                }, (split, model, verdict)
        status = html.escape(hfsite.split_meta(data, split)["status"])
        assert f'<p class="status" data-default="{status}">{status}</p>' in body
    step = run_name(hf_context, "company_test", "step-00025")
    order = hfsite.release_order(data, "company_test")
    picked = json.loads(
        re.search(r'class="filter-data">(.*?)</script>', found["company_test"])[1]
    )[step][hfsite.MISS]
    assert [order[i]["ticker"] for i in picked] == ["DDD"]
    # The left-out release says so, and why on hover.
    ggg = accession(hf_context, "GGG")
    assert re.search(
        rf'release/{ggg}\.html">GGG · [^<]*</a></td>(?:<td[^>]*>[^<]*</td>)*?'
        r'<td class="num" title="its prompt is 45,000 tokens[^"]*">'
        r"45,000 tokens · left out</td>",
        found["time_test"],
    )


def test_a_release_page(built, data, hf_context):
    out, _ = built
    ccc, ddd = accession(hf_context, "CCC"), accession(hf_context, "DDD")
    text = (out / f"release/{ddd}.html").read_text(encoding="utf-8")
    shown = hfsite.default_runs(data, "company_test")
    for part in hfsite.release_view(data, "company_test", ddd, shown):
        assert part in text
    assert (
        '<a href="../explore.html#split=company_test">← Company test releases</a>'
        in (text)
    )
    assert f'<a href="../release/{ccc}.html" rel="prev">‹ CCC ' in text
    assert 'rel="next"' not in text  # DDD is the company test's last
    assert "<title>DDD 20" in text
    assert "<details>" not in text  # the company test shows every model already
    assert "<h3>Where the release prints each answer</h3>" in text
    # Validation shows its default models, and every model in a fold.
    aaa = accession(hf_context, "AAA")
    text = (out / f"release/{aaa}.html").read_text(encoding="utf-8")
    every = [run for _, run in hfsite.run_choices(data, "validation")]
    assert len(every) == 4 > len(hfsite.default_runs(data, "validation"))
    assert (
        "<details><summary>Every model on this split (4)</summary>"
        f"{hfsite.figures_table(data, 'validation', aaa, every)}</details>"
    ) in text
    # A left-out release: why, and its text, with no answers to list.
    ggg = accession(hf_context, "GGG")
    text = (out / f"release/{ggg}.html").read_text(encoding="utf-8")
    assert "Left out for every model" in text and "<mark " not in text
    assert "Where the release prints each answer" not in text
    assert "<details>" not in text
    assert html.escape(hfsite.read_release_text(data, "time_test", ggg)) in text


def test_the_errors_page_holds_every_example_with_what_its_filters_read(built, data):
    out, _ = built
    text = (out / "errors.html").read_text(encoding="utf-8")
    found = sections(text)
    assert list(found) == list(SPLITS)
    assert '(see <a href="about.html">About</a>)' in text
    assert hfsite.markdown(hfsite.verdict_glossary(data)) in text
    for split, body in found.items():
        assert hfsite.html_table(*hfsite.verdict_table(data, split)) in body
        rows = re.findall(
            r'<tr data-run="([^"]*)" data-verdict="([^"]*)" data-miss="([01])">', body
        )
        assert rows == [
            (
                row["run"],
                row["verdict"],
                "1" if hfsite.is_miss(data, row["verdict"]) else "0",
            )
            for row, _ in hfsite.example_rows(data, split)
        ]
        assert len(rows) == len(hfsite.examples_table(data, split)[1])
        assert '<p class="empty" hidden>' in body


def test_the_leaderboard_and_about_pages(built, data):
    out, _ = built
    text = (out / "index.html").read_text(encoding="utf-8")
    assert hfsite.html_table(*hfsite.validation_board(data)) in text
    for board in hfsite.test_boards(data):
        assert hfsite.html_table(board["columns"], board["rows"]) in text
        assert hfsite.markdown(board["differences"]) in text
    for key in ("validation", "test"):
        board = data.leaderboard[key]
        assert hfsite.markdown(board["notes"]) in text
        width, height = hfsite.png_size(out / "data" / board["chart"])
        assert width > 0 and height > 0
        assert (
            f'<img src="data/{board["chart"]}" width="{width}" height="{height}"'
            in (text)
        )
    assert "<strong>Unit slips:</strong>" in text
    about = (out / "about.html").read_text(encoding="utf-8")
    assert hfsite.markdown(data.meta["about"]) in about
    assert "<table>" in about  # its markdown tables


def test_markdown_escapes_html_and_opens_off_site_links_in_a_new_tab():
    out = hfsite.markdown(
        "See [About](about.html) and [GitHub](https://github.com/x). <b>raw</b> "
        "not_in_input\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    )
    assert '<a href="about.html">About</a>' in out
    assert '<a href="https://github.com/x" target="_blank" rel="noopener">' in out
    assert "&lt;b&gt;raw&lt;/b&gt; not_in_input" in out
    assert "<td>1</td>" in out
    assert hfsite.markdown_inline("*a* b") == "<em>a</em> b"


def test_the_script_finds_what_it_needs_in_the_pages(built):
    """explorer.js's hooks (classes, data attributes, filters) are on the pages, and
    the miss filter's value is the build's (the script reads it there)."""
    out, _ = built
    script = (SPACE / "explorer.js").read_text(encoding="utf-8")
    pages = "".join(
        (out / name).read_text(encoding="utf-8")
        for name in ("explore.html", "errors.html")
    )
    attributes = set(re.findall(r'getAttribute\("([\w-]+)"\)', script))
    assert attributes >= {"data-i", "data-miss", "data-run", "data-split"}
    for name in attributes:
        assert f' {name}="' in pages, name
    filters = set(re.findall(r'data-filter="([\w-]+)"', script))
    assert filters == {
        "model",
        "verdict",
        "company",
        "example-model",
        "example-verdict",
    }
    for name in filters:
        assert f'data-filter="{name}"' in pages, name
    for selector, markup in (
        (".split-tabs", 'class="split-tabs"'),
        ("section.split", '<section class="split'),
        ("script.filter-data", 'class="filter-data"'),
        (".filters", 'class="filters js-only"'),
        (".status", 'class="status"'),
        (".empty", 'class="empty"'),
    ):
        assert f'"{selector}' in script or f"'{selector}" in script, selector
        assert markup in pages, selector
    assert pages.count(f'data-miss="{hfsite.MISS}"') == 2 * len(SPLITS)
    assert '"miss"' not in script  # read from the page, not written twice


def test_the_style_sheet_hides_only_what_the_script_shows():
    """Without the script (no "js" class), every split and row shows, and the
    filters, which need it, don't."""
    style = (SPACE / "style.css").read_text(encoding="utf-8")
    assert ".js section.split:not(.current) {\n  display: none;" in style
    assert ".split-tabs,\n.js-only {\n  display: none;" in style
    rules = re.sub(r"/\*.*?\*/", "", style, flags=re.DOTALL)
    hidden = re.findall(r"([^{}]+)\{[^}]*display: none;", rules)
    for selector in hidden:
        names = [n.strip() for n in selector.split(",")]
        assert all(
            n.startswith(".js ")
            or n in (".split-tabs", ".js-only", "tr[hidden]", "p[hidden]")
            for n in names
        ), selector


def test_html_table_row_attributes():
    rows = [["a", 1], ["b", 2]]
    out = hfsite.html_table(
        ["name", "n"], rows, row_attributes=[{"data-run": 'x"<y'}, {}]
    )
    assert '<tr data-run="x&quot;&lt;y"><td' in out and "<tr><td" in out
    with pytest.raises(ValueError, match="one mapping per row"):
        hfsite.html_table(["name", "n"], rows, row_attributes=[{}])


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
    data = hfsite.load_data(out / "data")
    columns, rows = hfsite.validation_board(data)
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
    assert hfsite.split_meta(data, "time_test")["status"] == (
        "355 releases in the time test split, 9 of them left out for every model."
    )
    assert hfsite.split_meta(data, "validation")["status"] == (
        "60 releases in the validation split."
    )
    results = context.test_results
    for board in hfsite.test_boards(data):
        for row in board["rows"]:
            for cell in row[2:6]:
                assert cell in TEST_RESULTS_MD
    company, time_test = hfsite.test_boards(data)
    assert company["rows"][0][2] == "657/662 (99.24%) [98.05, 100.00]"
    assert time_test["rows"][0][2] == "1,699/1,740 (97.64%) [96.64, 98.57]"
    primary = data.leaderboard["test"]["primary"]
    assert "**+10.12 points [+5.08, +15.58]**" in primary
    assert (
        primary.removeprefix("**Primary result (pre-registered).** ") in TEST_RESULTS_MD
    )
    # Explore shows the test sets' models' validation runs first.
    assert hfsite.default_runs(data, "validation") == [
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
    data = hfsite.load_data(out / "data")
    for split in ("company_test", "time_test"):
        step = run_name(context, split, "step-00025")
        found = {
            r["release_accession"]
            for r in hfsite.matching_releases(data, split, step, hfsite.MISS)
        }
        misses = context.test_results["splits"][split]["misses"]
        assert found == {m["release_accession"] for m in misses}
        miss = misses[0]
        rows = hfsite.figure_rows(data, split, miss["release_accession"], [step])
        cell = next(
            row.cells[0]
            for row in rows
            if (row.field, row.period) == (miss["field"], miss["period_type"])
        )
        assert cell.kind in (hfsite.ERROR, hfsite.SLIP)
        assert cell.prediction["predicted"] == miss["predicted"]
        assert cell.prediction["verdict"] == miss["verdict"]
