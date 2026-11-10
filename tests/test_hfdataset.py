"""Tests for secbench.hfdataset, the Hugging Face export's dataset repo, on the small
repository of tests/hf_fixture.py. The tests that load the build with
datasets.load_dataset skip without the hf dependency group (CI has none of it), and
the last ones read the real data and skip without it (runs/, training/ and data/
aren't in the repository)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import fields
from decimal import Decimal
from pathlib import Path

import hf_fixture
import pytest

from secbench import grade, hfcommon, hfdataset, testreport
from secbench.hfcommon import ExportError, Sources

REPO = Path(__file__).resolve().parents[1]
DATA_FILES = [
    f"data/{config}/{split}.jsonl"
    for config in ("releases", "answers", "train", "predictions")
    for split in (
        ("train",) if config == "train" else ("validation", "company_test", "time_test")
    )
]
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


@pytest.fixture(scope="module")
def built(hf_context, tmp_path_factory):
    out = tmp_path_factory.mktemp("dataset") / "dataset"
    return out, hfdataset.build_dataset(hf_context, out)


def read_jsonl(path: Path) -> list[dict]:
    # split("\n"), not splitlines(): a JSON string may hold other line separators.
    return [json.loads(line) for line in path.read_text("utf-8").split("\n") if line]


def rows_of(out: Path, config: str, split: str) -> list[dict]:
    return read_jsonl(out / hfdataset.data_file(config, split))


def front_matter(readme: Path) -> dict:
    yaml = pytest.importorskip("yaml")
    text = readme.read_text("utf-8")
    assert text.startswith("---\n")
    return yaml.safe_load(text[4 : text.index("\n---\n", 4)])


def card_table(text: str, header: str) -> list[list[str]]:
    """The body rows of the markdown table whose header line starts with header."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip().strip("|").split(" | ")])
    return rows


def section(text: str, heading: str) -> str:
    start = text.index(f"\n{heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else len(text)]


# The files and their rows


def test_files_written(built, hf_repo):
    out, written = built
    names = [str(p.relative_to(out)) for p in written.files]
    assert names == [
        *DATA_FILES,
        "prompts/extract_v1.txt",
        "prompts/extract_v2.txt",
        "README.md",
    ]
    on_disk = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert on_disk == sorted(names)  # nothing else, no .part file left
    assert written.skipped == []
    for name in ("extract_v1.txt", "extract_v2.txt"):
        source = hf_repo.root / "prompts" / name
        assert (out / "prompts" / name).read_bytes() == source.read_bytes()


def test_row_counts(built, hf_context):
    out, _ = built
    counts = {name: len(read_jsonl(out / name)) for name in DATA_FILES}
    assert counts == {
        "data/releases/validation.jsonl": 2,
        "data/releases/company_test.jsonl": 2,
        "data/releases/time_test.jsonl": 3,  # GGG, left out, included
        "data/answers/validation.jsonl": 11,
        "data/answers/company_test.jsonl": 9,
        "data/answers/time_test.jsonl": 11,
        "data/train/train.jsonl": 3,
        # 4 validation runs of 11 figures; per test run 9 answers and CCC's
        # no_answer year to date, or 11 answers.
        "data/predictions/validation.jsonl": 4 * 11,
        "data/predictions/company_test.jsonl": 5 * 10,
        "data/predictions/time_test.jsonl": 5 * 11,
    }
    for split in hfcommon.SPLIT_NAMES:
        graded = sum(len(r.graded) for r in hf_context.runs(split))
        assert counts[f"data/predictions/{split}.jsonl"] == graded


@pytest.mark.parametrize("config", hfdataset.CONFIGS)
def test_columns_are_the_rows_fields_in_order(built, config):
    out, _ = built
    expected = [f.name for f in fields(hfdataset.ROW_TYPES[config])]
    for split in hfdataset.config_splits(config):
        rows = rows_of(out, config, split)
        assert rows and all(list(row) == expected for row in rows)


def test_release_rows_are_the_contexts(built, hf_context):
    out, _ = built
    for split in hfcommon.SPLIT_NAMES:
        rows = rows_of(out, "releases", split)
        assert rows == [hfcommon.as_json(r) for r in hf_context.releases[split]]
        for row in rows:
            assert row["split"] == split
            digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            assert digest == row["input_sha256"]
    bbb = rows_of(out, "releases", "validation")[1]
    assert bbb["exhibits"] == ["EX-99.1"] and bbb["cik"] == 1002
    assert (bbb["linked_form"], bbb["period_type"]) == ("10-K", "full year")


def test_left_out_releases_stay_flagged(built):
    out, _ = built
    time_test = {r["ticker"]: r for r in rows_of(out, "releases", "time_test")}
    ggg = time_test["GGG"]
    assert ggg["kept"] is False and ggg["prompt_tokens"] == 45000
    assert ggg["left_out_reason"].startswith("prompt over 39,936 tokens")
    assert all(
        time_test[t]["kept"] and time_test[t]["left_out_reason"] is None
        for t in ("EEE", "FFF")
    )
    # The split's answer key covers kept releases only, and no model was called.
    assert "GGG" not in {a["ticker"] for a in rows_of(out, "answers", "time_test")}
    assert "GGG" not in {p["ticker"] for p in rows_of(out, "predictions", "time_test")}


def test_spans_land_on_the_printed_forms(built):
    out, _ = built
    texts = {
        r["release_accession"]: r["text"]
        for split in hfcommon.SPLIT_NAMES
        for r in rows_of(out, "releases", split)
    }
    answers = {
        (a["ticker"], a["field"], a["period_type"]): a
        for split in hfcommon.SPLIT_NAMES
        for a in rows_of(out, "answers", split)
    }
    cases = {
        ("AAA", "revenue", "quarter"): ["1,234"],
        ("BBB", "net_income", "quarter"): ["(30,250)"],  # a loss in parentheses
        ("EEE", "net_income", "quarter"): ["−15"],  # after a minus sign
        ("EEE", "net_income", "year_to_date"): ["40"],  # unsigned on a loss row
        ("EEE", "eps_diluted", "quarter"): ["(0.11)"],
        ("CCC", "eps_diluted", "quarter"): [],  # not printed: not graded
    }
    for key, shown in cases.items():
        answer = answers[key]
        text = texts[answer["release_accession"]]
        assert all(len(span) == 2 for span in answer["spans"])
        assert [text[a:b] for a, b in answer["spans"]] == shown, key
        assert answer["in_input"] == bool(shown)
    assert answers["CCC", "eps_diluted", "quarter"]["printed_as"] is None
    loss = answers["EEE", "net_income", "year_to_date"]
    assert loss["printed_as"].startswith("40 on a loss row (Net loss")
    derived = answers["BBB", "revenue", "quarter"]
    assert derived["derived"] is True
    assert derived["full_year_accession"] == derived["linked_accession"]
    assert answers["AAA", "revenue", "quarter"]["nine_months_accession"] is None


def test_train_rows(built, hf_repo):
    out, _ = built
    rows = rows_of(out, "train", "train")
    assert [
        (r["ticker"], r["trained"], r["position"], r["seen_by_checkpoint"])
        for r in rows
    ] == [
        ("EEE", True, 1, False),
        ("FFF", True, 0, True),
        ("GGG", False, None, False),
    ]
    prompt = (hf_repo.root / "prompts" / "extract_v2.txt").read_text("utf-8")
    for row in rows:
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
        assert list(row["messages"][0]) == ["role", "content"]
        assert row["messages"][0]["content"] == prompt
        json.loads(row["messages"][2]["content"])  # the label: one JSON object


def test_prediction_rows_name_their_source(built, hf_context):
    out, _ = built
    expected_source = {}
    for split in hfcommon.SPLIT_NAMES:
        for run in hf_context.runs(split):
            info = run.info
            if info["provider"] == "anthropic":
                expected_source[run.name] = "Anthropic API"
            elif info["cost_basis"] == "gpu_hours":
                expected_source[run.name] = "vLLM (self-hosted)"
            else:
                expected_source[run.name] = "OpenAI API"
    commercial = Counter()
    for split in hfcommon.SPLIT_NAMES:
        for row in rows_of(out, "predictions", split):
            assert row["source"] == expected_source[row["run"]]
            assert row["split"] == split
            if row["source"] in hfcommon.COMMERCIAL_SOURCES:
                commercial[row["run"]] += 1
    # The commercial runs: Claude Haiku on validation, and Sol, Haiku and Luna on
    # each test split.
    assert len(commercial) == 7 and sum(commercial.values()) == 11 + 6 * 10 + 3
    card = (out / "README.md").read_text("utf-8")
    assert (
        f'Rows whose `source` is "Anthropic API" or "OpenAI API" '
        f"({sum(commercial.values())} rows from 7 runs) are outputs of commercial APIs"
    ) in card
    assert "never the reply text" in card


# The card


def test_card_intro_counts_what_it_holds(built, hf_context):
    """What each part holds, without overstating: the left-out releases were sent
    to no model, prediction rows can be null, and the published checkpoint saw
    only the first of the examples drawn for training."""
    out, _ = built
    intro = section((out / "README.md").read_text("utf-8"), "# sec-earnings-bench")
    releases = [r for s in hfcommon.SPLIT_NAMES for r in rows_of(out, "releases", s)]
    answers = [a for s in hfcommon.SPLIT_NAMES for a in rows_of(out, "answers", s)]
    predictions = [
        p for s in hfcommon.SPLIT_NAMES for p in rows_of(out, "predictions", s)
    ]
    assert any(p["predicted"] is None for p in predictions)  # Sol's not_in_input
    runs = sum(len(hf_context.runs(s)) for s in hfcommon.SPLIT_NAMES)
    left_out = sum(not r["kept"] for r in releases)
    assert left_out == 1
    for text in (
        (
            f"holds the {len(releases)} releases of the validation and test splits "
            f"with their model inputs ({left_out} was left out by the context limit "
            "and sent to no model); "
        ),
        (
            f"their XBRL answer key ({len(answers)} answers; the "
            f"{sum(a['in_input'] for a in answers)} found in their release are "
            "located in its text); "
        ),
        (
            f"and {len(predictions)} prediction rows from {runs} runs, each with the "
            "model's figure (null when it gave none) and the grader's verdict."
        ),
        (
            "the 3 training examples labelled from XBRL: 2 were drawn for the "
            "qwen3-8b-2k training run, and the published checkpoint, step-00025, "
            "had seen the first 1."
        ),
        "- The fine-tuned model, step-00025 (Qwen3-8B with a LoRA adapter): ",
    ):
        assert text in intro, text


def test_card_metadata_declares_every_file(built, hf_context):
    out, _ = built
    meta = front_matter(out / "README.md")
    assert meta["license"] == "cc-by-4.0" and meta["language"] == ["en"]
    assert meta["pretty_name"]
    assert meta["size_categories"] == ["n<1K"]  # the predictions' 149 rows
    assert set(meta["task_categories"]) >= {"text-generation"}
    assert set(meta["tags"]) >= {
        "finance",
        "sec",
        "edgar",
        "xbrl",
        "earnings",
        "information-extraction",
        "benchmark",
    }
    declared = {
        (config["config_name"], f["split"]): f["path"]
        for config in meta["configs"]
        for f in config["data_files"]
    }
    assert [c["config_name"] for c in meta["configs"]] == list(hfdataset.CONFIGS)
    assert sorted(declared.values()) == sorted(DATA_FILES)
    for (config, split), path in declared.items():
        assert path == f"data/{config}/{split}.jsonl" and (out / path).is_file()
    info = {i["config_name"]: i["features"] for i in meta["dataset_info"]}
    assert list(info) == list(hfdataset.CONFIGS)
    for config, features in info.items():
        assert features == hfdataset.features(config)
        assert [f["name"] for f in features] == [
            f.name for f in fields(hfdataset.ROW_TYPES[config])
        ]
    answers = {f["name"]: f for f in info["answers"]}
    assert answers["spans"] == {"name": "spans", "list": {"list": "int64"}}
    assert answers["printed_as"] == {"name": "printed_as", "dtype": "string"}
    train = {f["name"]: f for f in info["train"]}
    assert train["messages"] == {
        "name": "messages",
        "list": [
            {"name": "role", "dtype": "string"},
            {"name": "content", "dtype": "string"},
        ],
    }
    assert train["position"] == {"name": "position", "dtype": "int64"}
    releases = {f["name"]: f for f in info["releases"]}
    assert releases["exhibits"] == {"name": "exhibits", "list": "string"}
    assert releases["cik"]["dtype"] == "int64" and releases["kept"]["dtype"] == "bool"


def test_yaml_writer_round_trips():
    yaml = pytest.importorskip("yaml")
    tricky = [
        "yes",
        "No",
        "null",
        "on",
        "y",
        "2026-07-01",
        "1.5",
        "0x10",
        "a: b",
        "#x",
        "-x",
        "1K<n<10K",
        "é",
        "x y",
        "",
        "plain-path/file_1.jsonl",
    ]
    data = {
        "a": tricky,
        "b": {"c": True, "d": 3, "e": [{"f": "g", "h": [{"i": "j"}]}]},
        "k": [],
        "m": {},
        "configs": [{"config_name": "x", "data_files": [{"split": "s", "path": "p"}]}],
    }
    text = "\n".join(hfdataset.yaml_lines(data))
    assert yaml.safe_load(text) == data
    lines = text.splitlines()
    assert '- "yes"' in lines and '- "2026-07-01"' in lines  # quoted: not a bool, date
    assert "- plain-path/file_1.jsonl" in lines  # plain


def test_features_of_unsupported_annotations_stop_the_build():
    for annotation, column in ((float, "x"), (dict[str, str], "x"), (int | str, "x")):
        with pytest.raises(ExportError, match="no datasets feature"):
            hfdataset.feature(annotation, column)
    assert hfdataset.feature(tuple[tuple[int, int], ...], "spans") == {
        "list": {"list": "int64"}
    }
    assert hfdataset.type_text(str | None, "x") == "string or null"


@pytest.mark.parametrize(
    "rows, category",
    [(0, "n<1K"), (999, "n<1K"), (1_000, "1K<n<10K"), (23_386, "10K<n<100K")],
)
def test_size_categories(rows, category):
    assert hfdataset.size_category(rows) == category


def test_card_counts_are_the_rows(built, hf_context):
    out, _ = built
    card = (out / "README.md").read_text("utf-8")
    counts = card_table(card, "| config | validation |")
    expected = {
        config: [
            str(len(rows_of(out, config, split)))
            if split in hfdataset.config_splits(config)
            else "–"
            for split in (*hfcommon.SPLIT_NAMES, "train")
        ]
        for config in hfdataset.CONFIGS
    }
    assert {row[0].strip("`"): row[1:] for row in counts} == expected
    answer_rows = card_table(card, "| split | answers |")
    for row in answer_rows:
        answers = rows_of(out, "answers", row[0].strip("`"))
        assert row[1:] == [
            str(len(answers)),
            str(sum(a["in_input"] for a in answers)),
            str(sum(not a["in_input"] for a in answers)),
            str(sum(a["derived"] for a in answers)),
        ]
    assert "3 training releases were labelled" in card
    assert "and 2 of those were drawn" in card and "had seen the first 1 " in card
    assert "That leaves out 1 release: GGG (45,000 tokens). It stays" in card
    assert (
        "37 of the 40" not in card
        and "Every drawn company has a linked release" in card
    )


def test_card_grading_lists_every_verdict(built):
    out, _ = built
    card = (out / "README.md").read_text("utf-8")
    rows = card_table(card, "| verdict | graded |")
    assert [r[0].strip("`") for r in rows] == list(grade.VERDICTS)
    assert {r[0].strip("`") for r in rows if r[1] == "no"} == set(grade.UNGRADED)
    meaning = {r[0].strip("`"): r[2] for r in rows}
    assert "1,000 or 1,000,000" in meaning["scale"]
    assert "10, 100, 10,000 or 100,000" in meaning["zeros"]
    # Unit slips go both ways: a factor below 1 is a division (the mini repo's
    # scale rows record 0.001).
    assert "multiplied or divided by each power of ten from 10 to 1,000,000" in card
    assert "so a division is a factor below 1 (0.001: the prediction divided by" in card
    factors = {
        Decimal(r["factor"])
        for split in hfcommon.SPLIT_NAMES
        for r in rows_of(out, "predictions", split)
        if r["factor"] is not None
    }
    assert factors == {Decimal("0.001")} and factors <= set(grade.FACTORS)


def test_unit_slips_text_follows_grade_factors(monkeypatch):
    for factors in (
        tuple(f for f in grade.FACTORS if f > 1),  # multiplied only
        tuple(f for f in grade.FACTORS if f not in (100, Decimal("0.01"))),
    ):
        monkeypatch.setattr(grade, "FACTORS", factors)
        with pytest.raises(ExportError, match="grade.FACTORS isn't each power"):
            hfdataset.grading_section()


def test_card_columns_and_models(built, hf_context):
    out, _ = built
    card = (out / "README.md").read_text("utf-8")
    for config in hfdataset.CONFIGS:
        rows = card_table(
            section(card, "## Columns").split(f"### {config}\n")[1], "| column |"
        )
        assert [r[0].strip("`") for r in rows] == [
            f.name for f in fields(hfdataset.ROW_TYPES[config])
        ]
    columns = {
        r[0].strip("`"): r[1]
        for r in card_table(card.split("### answers\n")[1], "| column |")
    }
    assert columns["spans"] == "list of list of int64"
    assert columns["printed_as"] == "string or null"
    described = {
        config: {
            r[0].strip("`"): r[2]
            for r in card_table(card.split(f"### {config}\n")[1], "| column |")
        }
        for config in ("releases", "answers")
    }
    # A derived quarter's value and start are computed, not an XBRL fact's.
    assert (
        "(for a derived quarter, the 10-K's full year minus the nine months)"
        in (described["answers"]["value"])
    )
    assert described["answers"]["start"] == (
        "the XBRL period's start; for a derived quarter, the day after the nine "
        "months end"
    )
    releases = described["releases"]
    assert releases["cik"] == "the CIK the 8-K is filed under"  # one CIK each here
    assert releases["preferred_dividends"].endswith(
        "; the flag missed some releases that print preferred-stock lines (HPE, KDP "
        "and LHX in the time test)"
    )
    assert releases["left_out_reason"].endswith(
        "null when kept. It cites PLAN.md, the project's planning notes, which "
        "aren't published; the Context limit section states the rule"
    )
    models = section(card, "## Predictions by model")
    for split in hfcommon.SPLIT_NAMES:
        rows = card_table(models.split(f"### {split}\n")[1], "| model |")
        runs = hf_context.runs(split)
        assert [r[3].strip("`") for r in rows] == [r.name for r in runs]
        assert [(r[0], r[1], r[2], r[4]) for r in rows] == [
            (r.model, r.source, r.prompt, str(len(r.graded))) for r in runs
        ]


def test_card_says_the_preferred_flag_is_an_xbrl_check(built, hf_repo):
    out, _ = built
    card = (out / "README.md").read_text("utf-8")
    flagged = [
        r["ticker"]
        for split in hfcommon.SPLIT_NAMES
        for r in rows_of(out, "releases", split)
        if r["preferred_dividends"]
    ]
    assert flagged == ["DDD"]
    # The companies results/test_results.md names (its miss summary), read from
    # the real results file the mini repository holds.
    outside = hf_repo.test_results["splits"]["time_test"]["miss_summary"]
    named = [t for t, rows in outside["outside_rows"].items() if "preferred" in rows]
    assert named == ["HPE", "KDP", "LHX"]
    assert (
        "`preferred_dividends` flags the releases whose linked filing tags nonzero "
        "preferred dividends for their periods in XBRL, under "
        "`PreferredStockDividendsIncomeStatementImpact`, `DividendsPreferredStock`, "
        "`DividendsPreferredStockCash` or `PreferredStockDividendsAndOtherAdjustments` "
        "(1 release here). The flag reads XBRL only, and it missed some "
        "releases that print preferred-stock lines: results/test_results.md names "
        "HPE, KDP and LHX in the time test."
    ) in section(card, "## Answer key")


def test_column_notes_follow_the_rows(hf_context):
    """The notes added to the columns' descriptions come from the rows and the
    test results: changed here to give a company releases under two CIKs (as
    XOM's are), no company named by the test report, and a left-out reason that
    cites nothing."""
    notes = hfdataset.column_notes(hf_context)
    assert set(notes) == {
        ("releases", "preferred_dividends"),
        ("releases", "left_out_reason"),
    }
    eee, fff, ggg = hf_context.releases["time_test"]
    results = json.loads(json.dumps(hf_context.test_results))
    for split in hfcommon.TEST_SPLITS:
        results["splits"][split]["miss_summary"]["outside_rows"] = {
            "DOC": ["participating"]
        }
    ctx = dataclasses.replace(
        hf_context,
        releases={
            **hf_context.releases,
            "time_test": [
                eee,
                dataclasses.replace(fff, ticker="EEE", cik=eee.cik + 1000),
                dataclasses.replace(ggg, left_out_reason="prompt over the limit"),
            ],
        },
        test_results=results,
    )
    assert hfdataset.column_notes(ctx) == {
        ("releases", "cik"): (
            f"; a company's releases can be under more than one (EEE: {eee.cik} and "
            f"{eee.cik + 1000}), so group by `ticker`"
        )
    }
    answer_key = "\n".join(hfdataset.answers_section(ctx))
    assert "(1 release here). The flag reads XBRL only.\n" in answer_key


def test_card_limitations_are_test_results(built, hf_repo):
    out, _ = built
    card = (out / "README.md").read_text("utf-8")
    results = hf_repo.test_results
    for split, name in (("company_test", "company test"), ("time_test", "time test")):
        cov = results["splits"][split]["coverage"]
        share = testreport.num(100 * cov["missing"] / cov["expected"], 1)
        assert (
            f"{cov['missing']:,} of {cov['expected']:,} on the {name} ({share}%)"
            in card
        )
    sectors = results["splits"]["time_test"]["coverage"]["no_answer_by_sector"]
    without = sum(v["no_answer"] for v in sectors.values())
    worst = max(sectors, key=lambda s: sectors[s]["no_answer"])
    assert (
        f"{without} time-test releases have no answer at all, "
        f"{sectors[worst]['no_answer']} of them among the time test's "
        f"{sectors[worst]['kept']} {worst} releases"
    ) in card
    note = results["splits"]["company_test"]["notes"][0]
    assert f"{note['ticker']} {note['release_accession']} is {note['note']}" in card
    assert "step-00025 and Qwen3-8B base never answered null" in card
    assert f"The time test stops at {results['time_test']['last_filed']}" in card


def test_card_links_and_citation(built, hf_context):
    out, _ = built
    card = (out / "README.md").read_text("utf-8")
    assert hfcommon.REPO_URL in card
    assert "https://huggingface.co/test-user/qwen3-8b-sec-earnings-lora" in card
    assert "https://huggingface.co/spaces/test-user/sec-earnings-explorer" in card
    assert 'repo = "test-user/sec-earnings-bench"' in card
    year, name = hfcommon.author(hf_context.sources)
    citation = card.split("```bibtex\n")[1].split("```")[0]
    assert f"author = {{{name}}}" in citation and f"year = {{{year}}}" in citation
    assert f"url = {{{hfcommon.REPO_URL}}}" in citation
    assert citation.count("{") == citation.count("}")
    # BibTeX styles lowercase a title outside braces: every capital is braced.
    title = re.search(r"^  title = \{(.*)\},$", citation, re.MULTILINE)[1]
    depth = 0
    for char in title:
        depth += {"{": 1, "}": -1}.get(char, 0)
        assert depth >= 0 and (depth > 0 or not char.isupper()), title
    assert {"{GAAP}", "{SEC}", "{XBRL}"} <= set(re.findall(r"\{[A-Z]+\}", title))


def test_two_builds_are_identical(hf_context, tmp_path):
    for name in ("a", "b"):
        hfdataset.build_dataset(hf_context, tmp_path / name)
    files = {
        name: {
            str(p.relative_to(tmp_path / name)): p.read_bytes()
            for p in sorted((tmp_path / name).rglob("*"))
            if p.is_file()
        }
        for name in ("a", "b")
    }
    assert files["a"] == files["b"] and len(files["a"]) == len(DATA_FILES) + 3


def test_nothing_private_is_written(built, hf_repo):
    out, _ = built
    secrets = [
        hf_fixture.SECRET_ORG,
        hf_fixture.SECRET_EMAIL,
        "chatcmpl-",
        "/workspace/",
        str(hf_repo.root),
        str(REPO),
        str(Path.home()),
    ]
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text("utf-8")
            assert not [s for s in secrets if s in text], path
    # The press contact appears only in release texts: releases' text and the
    # training examples' user messages.
    assert hf_fixture.RELEASE_EMAIL not in (out / "README.md").read_text("utf-8")
    for name in DATA_FILES:
        for row in read_jsonl(out / name):
            for key, value in row.items():
                if key == "text":
                    continue
                if key == "messages":
                    value = [m["content"] for m in value if m["role"] != "user"]
                assert not EMAIL.search(json.dumps(value)), (name, key)


# What the build checks


def change_json(path: Path, change) -> None:
    data = json.loads(path.read_text("utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=2) + "\n", "utf-8")


def replace_text(path: Path, old: str, new: str) -> None:
    text = path.read_text("utf-8")
    assert old in text
    path.write_text(text.replace(old, new), "utf-8")


def training_selection(root: Path) -> None:
    change_json(
        root / "training" / "qwen3-8b-2k" / "config.json",
        lambda d: d["selection"].update(selected=3),
    )


def scored_release_in_training(root: Path) -> None:
    # GGG's training example (not trained on) given a time-test release's accession.
    labels = root / "data" / "stage2" / "labels"
    for name in ("train.jsonl", "examples.csv"):
        replace_text(labels / name, "0000001007-25-000008", "0000001005-26-000031")


def held_out_company_in_training(root: Path) -> None:
    replace_text(
        root / "data" / "stage2" / "labels" / "examples.csv",
        "0000001007-25-000008,GGG,",
        "0000001007-25-000008,AAA,",
    )


def training_after_its_window(root: Path) -> None:
    replace_text(
        root / "data" / "stage2" / "labels" / "examples.csv", "2025-02-12", "2026-07-15"
    )


def another_system_prompt(root: Path) -> None:
    path = root / "data" / "stage2" / "labels" / "train.jsonl"
    lines = path.read_text("utf-8").splitlines()
    first = json.loads(lines[0])
    first["messages"][0]["content"] += "\nAnother rule."
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n", "utf-8")


def company_in_another_split(root: Path) -> None:
    replace_text(
        root / "data" / "companies.csv",
        "CCC,1003,Gamma Co,Health Care,company_test",
        "CCC,1003,Gamma Co,Health Care,train",
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (training_selection, "differ from .*config.json's selection"),
        (scored_release_in_training, "scored releases among the training examples"),
        (held_out_company_in_training, "training examples of held-out companies"),
        (training_after_its_window, "training examples filed .* outside"),
        (another_system_prompt, "isn't prompts/extract_v2.txt"),
        (company_in_another_split, "outside its companies.csv split"),
    ],
)
def test_the_build_checks_what_the_card_states(
    hf_repo_copy, hf_context, tmp_path, change, message
):
    change(hf_repo_copy.root)
    ctx = dataclasses.replace(hf_context, sources=Sources(hf_repo_copy.root))
    with pytest.raises(ExportError, match=message):
        hfdataset.build_dataset(ctx, tmp_path / "out")


def test_the_time_tests_window_is_checked(hf_context, tmp_path):
    results = json.loads(json.dumps(hf_context.test_results))
    results["time_test"]["first_filed"] = "2026-08-01"  # after EEE's 2026-07-30
    ctx = dataclasses.replace(hf_context, test_results=results)
    with pytest.raises(ExportError, match="time_test releases filed outside"):
        hfdataset.build_dataset(ctx, tmp_path / "out")


# Loading with datasets, as the Hub does (the hf dependency group)


def load(out: Path, config: str, cache: Path):
    datasets = pytest.importorskip("datasets")
    datasets.disable_progress_bars()
    return datasets.load_dataset(str(out), config, cache_dir=str(cache))


def check_loads_as_declared(out: Path, cache: Path, configs) -> dict:
    datasets = pytest.importorskip("datasets")
    meta = front_matter(out / "README.md")
    declared = {i["config_name"]: i["features"] for i in meta["dataset_info"]}
    loaded = {}
    for config in meta["configs"]:
        name = config["config_name"]
        if name not in configs:
            continue
        found = load(out, name, cache)
        assert list(found) == [f["split"] for f in config["data_files"]]
        features = datasets.Features._from_yaml_list(declared[name])
        for f in config["data_files"]:
            split = found[f["split"]]
            assert split.features == features, (name, f["split"])
            assert split.num_rows == len(read_jsonl(out / f["path"]))
        loaded[name] = found
    return loaded


def test_every_config_loads_as_the_card_declares(built, tmp_path):
    out, _ = built
    loaded = check_loads_as_declared(out, tmp_path / "cache", hfdataset.CONFIGS)
    answers = loaded["answers"]["time_test"]
    eee = [a for a in answers if a["ticker"] == "EEE" and a["period_type"] == "quarter"]
    assert eee[1]["spans"] and all(len(s) == 2 for s in eee[1]["spans"])
    assert loaded["train"]["train"]["position"] == [1, 0, None]
    validation = loaded["releases"]["validation"]  # every left_out_reason null here
    assert validation["left_out_reason"] == [None, None]
    assert loaded["releases"]["time_test"]["kept"] == [True, True, False]


def test_the_cards_loading_snippet_runs(built, tmp_path):
    """The card's snippet, run as a script on the local build (its repo id
    replaced by the folder), offline and with its own datasets cache."""
    pytest.importorskip("datasets")
    out, _ = built
    card = (out / "README.md").read_text("utf-8")
    snippet = card.split("```python\n")[1].split("```")[0]
    assert 'repo = "test-user/sec-earnings-bench"' in snippet
    script = tmp_path / "snippet.py"
    script.write_text(
        snippet.replace('"test-user/sec-earnings-bench"', repr(str(out))), "utf-8"
    )
    env = {
        **os.environ,
        "HF_DATASETS_CACHE": str(tmp_path / "cache"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    done = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    printed = done.stdout
    # CCC's quarter revenue, the first company-test answer the release prints.
    assert printed.splitlines()[0] == "revenue 845000000 845"
    rows = rows_of(out, "predictions", "time_test")
    for model in {r["model"] for r in rows}:
        graded = [
            r
            for r in rows
            if r["model"] == model and r["verdict"] not in grade.UNGRADED
        ]
        share = sum(r["verdict"] == "correct" for r in graded) / len(graded)
        line = rf"^{re.escape(model)}\s+{share:.6f}$"
        assert re.search(line, printed, re.MULTILINE), model


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
def real_built(tmp_path_factory):
    ctx = hfcommon.load_context(REAL, hfcommon.PLACEHOLDER_USER)
    out = tmp_path_factory.mktemp("real") / "dataset"
    hfdataset.build_dataset(ctx, out)
    return ctx, out


def manifest(split: str) -> dict:
    return json.loads((REAL.splits_dir / split / "manifest.json").read_text("utf-8"))


@needs_real_data
def test_real_counts(real_built):
    ctx, out = real_built
    releases = {s: rows_of(out, "releases", s) for s in hfcommon.SPLIT_NAMES}
    answers = {s: rows_of(out, "answers", s) for s in hfcommon.SPLIT_NAMES}
    assert {
        s: (sum(r["kept"] for r in v), sum(not r["kept"] for r in v))
        for s, v in releases.items()
    } == {
        "validation": (60, 0),
        "company_test": (148, 0),
        "time_test": (346, 9),
    }
    assert [len(answers[s]) for s in hfcommon.SPLIT_NAMES] == [293, 700, 1792]
    for split in hfcommon.SPLIT_NAMES:
        counts = manifest(split)["counts"]
        kept = [r for r in releases[split] if r["kept"]]
        assert (len(kept), len(releases[split]) - len(kept)) == (
            counts["kept"],
            counts.get("left_out", 0),
        )
        assert len(answers[split]) == counts["answers"]
        assert sum(a["in_input"] for a in answers[split]) == counts["answers_in_input"]
        predictions = rows_of(out, "predictions", split)
        assert len(predictions) == sum(len(r.graded) for r in ctx.runs(split))
    assert [len(ctx.runs(s)) for s in hfcommon.SPLIT_NAMES] == [24, 5, 5]
    config = json.loads((REAL.training / "config.json").read_text("utf-8"))
    seen = json.loads((REAL.checkpoint / "state.json").read_text("utf-8"))[
        "examples_seen"
    ]
    with (out / "data" / "train" / "train.jsonl").open(encoding="utf-8") as f:
        train = [
            (row["trained"], row["seen_by_checkpoint"]) for row in map(json.loads, f)
        ]
    assert len(train) == config["selection"]["examples"] == 3200
    assert sum(t for t, _ in train) == config["selection"]["selected"] == 2000
    assert sum(s for _, s in train) == seen == 400
    card = (out / "README.md").read_text("utf-8")
    for text in (
        "60 releases of 10 companies",
        "148 releases of 37 companies never trained on",
        "37 of the 40 drawn companies have a linked release (AEP, FDXF and VYLR have none)",
        "355 releases of 353 training companies, filed 2026-07-01 to 2026-09-30",
        "346 kept and 9 left out by the context limit",
        "3,200 training releases were labelled",
        "3,101 fit the training cap of 32,768 tokens, and 2,000 of those were drawn",
        "had seen the first 400 ",
        "28 of 728 on the company test (3.8%) and 124 of 1,916 on the time test (6.5%)",
        "That leaves out 9 releases: UDR (50,970 tokens)",
        "They stay in `releases`, with `kept` false",
        "BXP (106,137 tokens)",
        "| `answers` | 293 | 700 | 1,792 | – |",
        "| `time_test` | 1,792 | 1,740 | 52 | 49 |",
        "24 runs (",
        "CCI 0001051470-26-000052 is a divestiture and outlook announcement",
        # The intro: the reviewed counts, and what the published checkpoint saw.
        (
            "This dataset holds the 563 releases of the validation and test splits "
            "with their model inputs (9 were left out by the context limit and sent "
            "to no model); their XBRL answer key (2,785 answers; the 2,695 found in "
            "their release are located in its text); and 23,386 prediction rows from "
            "34 runs, each with the model's figure (null when it gave none) and the "
            "grader's verdict. It also holds the 3,200 training examples labelled "
            "from XBRL: 2,000 were drawn for the qwen3-8b-2k training run, and the "
            "published checkpoint, step-00025, had seen the first 400."
        ),
        (
            "(35 releases here). The flag reads XBRL only, and it missed some "
            "releases that print preferred-stock lines: results/test_results.md "
            "names HPE, KDP and LHX in the time test."
        ),
        (
            "| `cik` | int64 | the CIK the 8-K is filed under; a company's releases "
            "can be under more than one (XOM: 34088 and 2115436), so group by "
            "`ticker` |"
        ),
        "so a division is a factor below 1 (0.001: the prediction divided by 1,000)",
    ):
        assert text in card, text
    for text in (
        "fine-tuned Qwen3-8B",
        "as the models saw them",
        "predicted figures from",
    ):
        assert text not in card, text
    assert front_matter(out / "README.md")["size_categories"] == ["10K<n<100K"]
    # Most of the test splits' unit slips are divisions (factors below 1).
    for split in hfcommon.TEST_SPLITS:
        factors = [
            Decimal(r["factor"])
            for r in rows_of(out, "predictions", split)
            if r["factor"] is not None
        ]
        assert sum(f < 1 for f in factors) > len(factors) / 2, split


@needs_real_data
def test_real_unflagged_preferred_releases(real_built):
    """The releases behind the companies the card names: not flagged, yet their
    text prints preferred-stock lines, and the key's net income is NetIncomeLoss
    (before preferred dividends)."""
    ctx, out = real_built
    named = hfdataset.unflagged_preferred(ctx.test_results)
    assert named == {"time_test": ["HPE", "KDP", "LHX"]}
    misses = {
        m["release_accession"]: m["ticker"]
        for m in ctx.test_results["splits"]["time_test"]["misses"]
        if "preferred" in m.get("release_rows", []) and not m["preferred"]
    }
    assert sorted(set(misses.values())) == named["time_test"]
    releases = {
        r["release_accession"]: r for r in rows_of(out, "releases", "time_test")
    }
    net_income = {
        (a["release_accession"], a["period_type"]): a["tag"]
        for a in rows_of(out, "answers", "time_test")
        if a["field"] == "net_income"
    }
    for accession in misses:
        release = releases[accession]
        assert release["preferred_dividends"] is False
        assert "preferred" in release["text"].lower()
        tags = {t for (acc, _), t in net_income.items() if acc == accession}
        assert tags == {"NetIncomeLoss"}, accession


@needs_real_data
def test_real_build_holds_nothing_private(real_built):
    _, out = real_built
    home = str(Path.home())
    for path in out.rglob("*"):
        if path.suffix in (".md", ".txt"):
            text = path.read_text("utf-8")
            assert home not in text and "/workspace/" not in text, path
            assert not EMAIL.search(text), path
    for name in DATA_FILES:
        with (out / name).open(encoding="utf-8") as f:
            for line in f:
                assert home not in line and "/workspace/" not in line, name
                row = json.loads(line)
                for key, value in row.items():
                    if key == "text":
                        continue
                    if key == "messages":
                        value = [m["content"] for m in value if m["role"] != "user"]
                    assert not EMAIL.search(json.dumps(value)), (name, key)


@needs_real_data
def test_real_small_configs_load(real_built, tmp_path):
    _, out = real_built
    loaded = check_loads_as_declared(
        out, tmp_path / "cache", ("releases", "answers", "predictions")
    )
    assert loaded["releases"]["time_test"].num_rows == 355
    assert sum(loaded["releases"]["time_test"]["kept"]) == 346


@needs_real_data
def test_real_train_loads(real_built, tmp_path):
    _, out = real_built
    train = check_loads_as_declared(out, tmp_path / "cache", ("train",))["train"][
        "train"
    ]
    assert train.num_rows == 3200 and sum(train["trained"]) == 2000
