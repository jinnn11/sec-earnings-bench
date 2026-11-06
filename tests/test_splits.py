"""secbench.splits on the five input_releases fixtures, with the byte tokenizer (one
token per byte, plus Qwen3's special tokens) and a limit of 60,000 prompt tokens
instead of 39,936, so the time test has one release left out. No data/ files."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from conftest import byte_tokenizer
from test_score import RELEASES, key_dir  # noqa: F401 (key_dir is a fixture)

from secbench import answers, chat, extract, labels, link, splits
from secbench.stage2 import RELEASE_COLUMNS

PROMPTS = Path(__file__).parent.parent / "prompts"
LIMIT = 60_000
AAPL = "0000320193-25-000055"  # 13,984 prompt tokens with the byte tokenizer
MSFT = "0000950170-25-100226"  # 23,399
WMT = "0000104169-25-000069"  # 48,111
XOM = "0002115436-26-000006"  # 57,525
CAT = "0000018230-25-000037"  # 67,391: over LIMIT
TOKENS = {AAPL: 13_984, MSFT: 23_399, WMT: 48_111, XOM: 57_525, CAT: 67_391}
EXPECTED = {"validation": {}, "company_test": {}, "time_test": {CAT: 67_391}}
FOLDERS = {
    AAPL: "0000320193",
    MSFT: "0000789019",
    WMT: "0000104169",
    XOM: "0002115436",
    CAT: "0000018230",
}
# accession -> (ticker, cik, split, preferred dividends). Answers: AAPL 6 (a second
# quarter: quarter and six months), MSFT 5 (a 10-K: no fourth-quarter EPS), WMT 3 (a
# first quarter: no year to date), XOM 6, CAT 6.
META = {
    AAPL: ("AAPL", "320193", "validation", False),
    MSFT: ("MSFT", "789019", "validation", False),
    WMT: ("WMT", "104169", "company_test", True),
    XOM: ("XOM", "2115436", "time_test", False),
    CAT: ("CAT", "18230", "time_test", True),
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_releases_csv(path: Path, meta: dict) -> None:
    rows = []
    for accession, (ticker, cik, split, preferred) in meta.items():
        rows.append(
            {
                **dict.fromkeys(RELEASE_COLUMNS, ""),
                "ticker": ticker,
                "cik": cik,
                "company_split": split,
                "release_accession": accession,
                "filing_date": "2025-05-01",
                "saved": "True",
                "link_status": "linked",
                "split": split,
                "preferred_dividends": str(preferred),
            }
        )
    path.write_text(splits.csv_text(rows, RELEASE_COLUMNS))


@pytest.fixture
def repo(tmp_path, key_dir, monkeypatch) -> Path:  # noqa: F811
    """data/ as the repository lays it out: AAPL and MSFT validation (stage 1), WMT
    company_test, XOM and CAT time_test (stage 2)."""
    data = tmp_path / "data"
    (data / "stage2").mkdir(parents=True)
    links = (key_dir / "links.csv").read_text().splitlines(keepends=True)
    key = (key_dir / "answers.csv").read_text().splitlines(keepends=True)
    (data / "links.csv").write_text(
        links[0] + "".join(x for x in links[1:] if x.split(",")[2] in (AAPL, MSFT))
    )
    (data / "answers.csv").write_text(
        key[0] + "".join(x for x in key[1:] if x.split(",")[1] in (AAPL, MSFT))
    )
    shutil.copy(key_dir / "links.csv", data / "stage2" / "links.csv")
    shutil.copy(key_dir / "answers.csv", data / "stage2" / "answers.csv")
    for accession, folder in FOLDERS.items():
        stage = "releases" if accession in (AAPL, MSFT) else "stage2/releases"
        shutil.copytree(RELEASES / folder, data / stage / folder)
    write_releases_csv(data / "stage2" / "releases.csv", META)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(*args: str, expected=None, commit: str = "c0ffee" * 6) -> int:
    return splits.main(
        ["--prompts-dir", str(PROMPTS), *args],
        commit=commit,
        tokenizer=byte_tokenizer(),
        expected_left_out=EXPECTED if expected is None else expected,
        limit=LIMIT,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_limit_is_the_context_less_the_output_allowance() -> None:
    assert splits.MAX_PROMPT_TOKENS == 40_960 - 1_024 == 39_936
    assert chat.MAX_MODEL_LEN == 40_960 and chat.MAX_TOKENS == 1_024
    decision = splits.EXPECTED_LEFT_OUT["time_test"]
    assert len(decision) == 9 and decision["0001037540-26-000031"] == 106_137
    assert min(decision.values()) == 44_808  # INVH
    assert splits.EXPECTED_LEFT_OUT["validation"] == {}
    assert splits.EXPECTED_LEFT_OUT["company_test"] == {}


def test_main_writes_each_split(repo, key_dir, capsys) -> None:  # noqa: F811
    assert run() == 0
    out = repo / "data" / "splits"
    assert sorted(p.name for p in out.iterdir()) == [
        "company_test",
        "time_test",
        "validation",
    ]
    report = capsys.readouterr().out
    assert "| validation | provisional | 2 | 0 | 2 | 23,399 | 11 (11) | 0 |" in report
    assert "| company_test | provisional | 1 | 0 | 1 | 48,111 | 3 (3) | 1 |" in report
    assert "| time_test | provisional | 1 | 1 | 1 | 57,525 | 6 (6) | 0 |" in report
    assert f"- CAT {CAT}: 67,391 tokens, preferred dividends" in report

    # links.csv and answers.csv: the sources' rows for the kept releases, verbatim.
    source = (repo / "data" / "links.csv").read_text()
    assert (out / "validation" / "links.csv").read_text() == source
    assert (out / "validation" / "answers.csv").read_text() == (
        repo / "data" / "answers.csv"
    ).read_text()
    test_links = read(out / "time_test" / "links.csv")
    assert [r["release_accession"] for r in test_links] == [XOM]
    with (out / "time_test" / "answers.csv").open() as f:
        assert f.readline().rstrip("\n").split(",") == answers.COLUMNS
    assert {
        r["release_accession"] for r in read(out / "time_test" / "answers.csv")
    } == {XOM}
    with (out / "company_test" / "links.csv").open() as f:
        assert f.readline().rstrip("\n").split(",") == link.COLUMNS

    # tokens.csv: every release; left_out.csv and preferred.csv.
    tokens = read(out / "time_test" / "tokens.csv")
    assert [
        (r["release_accession"], r["prompt_tokens"], r["status"]) for r in tokens
    ] == [
        (CAT, "67391", "left_out"),  # links.csv's order
        (XOM, "57525", "kept"),
    ]
    records = {
        p.stem: p for p in (repo / "data" / "stage2" / "releases").glob("*/*.json")
    }
    text = splits.build_input(records[CAT]).text
    assert tokens[0]["input_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert read(out / "time_test" / "left_out.csv") == [
        {
            "ticker": "CAT",
            "release_accession": CAT,
            "prompt_tokens": "67391",
            "reason": "prompt over 60,000 tokens (PLAN.md Decisions 2026-11-01)",
        }
    ]
    assert read(out / "time_test" / "preferred.csv") == [
        {"ticker": "CAT", "release_accession": CAT, "status": "left_out"}
    ]
    assert read(out / "company_test" / "preferred.csv") == [
        {"ticker": "WMT", "release_accession": WMT, "status": "kept"}
    ]
    assert read(out / "validation" / "left_out.csv") == []

    manifest = json.loads((out / "time_test" / "manifest.json").read_text())
    assert manifest["split"] == "time_test" and manifest["status"] == "provisional"
    assert manifest["commit"] == "c0ffee" * 6
    assert manifest["releases_dir"] == "data/stage2/releases"
    assert manifest["counts"] == {
        "releases": 2,
        "kept": 1,
        "left_out": 1,
        "companies_kept": 1,
        "answers": 6,
        "answers_in_input": 6,
        "preferred_kept": 0,
        "preferred_left_out": 1,
    }
    assert manifest["longest_kept_prompt"] == {
        "ticker": "XOM",
        "release_accession": XOM,
        "tokens": 57_525,
    }
    assert manifest["context_limit"]["max_prompt_tokens"] == LIMIT
    assert manifest["tokenizer"]["revision"] == labels.QWEN3_REVISION
    prompt = extract.load_prompt("extract_v2", PROMPTS)
    assert manifest["prompt"] == {"version": "extract_v2", "sha256": prompt.sha256}
    assert manifest["inputs"] == {
        "data/stage2/links.csv": sha256(repo / "data/stage2/links.csv"),
        "data/stage2/answers.csv": sha256(repo / "data/stage2/answers.csv"),
        "data/stage2/releases.csv": sha256(repo / "data/stage2/releases.csv"),
        f"{PROMPTS}/extract_v2.txt": prompt.sha256,
    }
    assert manifest["files"] == {
        name: sha256(out / "time_test" / name) for name in splits.FILES
    }
    validation = json.loads((out / "validation" / "manifest.json").read_text())
    assert validation["releases_dir"] == "data/releases"
    assert "data/links.csv" in validation["inputs"]


def test_token_counts_are_the_prompt_in_qwen3s_layout(repo) -> None:
    """With the byte tokenizer, the count is the prompt's bytes, each special or
    think token counting as one."""
    assert run() == 0
    prompt = extract.load_prompt("extract_v2", PROMPTS).text
    records = {p.stem: p for p in (repo / "data" / "releases").glob("*/*.json")}
    text = chat.prompt_text(prompt, splits.build_input(records[AAPL]).text)
    special = ["<|im_start|>"] * 3 + ["<|im_end|>"] * 2 + ["<think>", "</think>"]
    expected = len(text.encode()) - sum(len(s) for s in special) + len(special)
    tokens = read(repo / "data" / "splits" / "validation" / "tokens.csv")
    assert int(tokens[0]["prompt_tokens"]) == expected == TOKENS[AAPL]


def test_the_same_inputs_give_the_same_bytes(repo) -> None:
    assert run("--out", "first") == 0
    assert run("--out", "second") == 0
    for split in splits.SPLITS:
        for name in (*splits.FILES, splits.MANIFEST):
            first = (repo / "first" / split / name).read_bytes()
            assert first == (repo / "second" / split / name).read_bytes()


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        (
            {"time_test": {}},
            f"time_test {CAT}: left out (67,391 tokens), not in the expected list",
        ),
        (
            {"time_test": {CAT: 67_390}},
            f"time_test {CAT}: 67,391 tokens, expected 67,390",
        ),
        (
            {"time_test": {CAT: 67_391, XOM: 57_525}},
            f"time_test {XOM}: expected left out (57,525 tokens), but it isn't",
        ),
        (
            {"time_test": {CAT: 67_391}, "company_test": {WMT: 48_111}},
            f"company_test {WMT}: expected left out (48,111 tokens), but it isn't",
        ),
    ],
)
def test_other_left_out_releases_write_nothing(repo, expected, message, capsys) -> None:
    assert run(expected=expected) == 1
    err = capsys.readouterr().err
    assert "the releases left out aren't the expected ones" in err
    assert message in err
    assert not (repo / "data" / "splits").exists()


def test_frozen_splits_are_built_from_the_frozen_list(repo, capsys) -> None:
    frozen = repo / "data" / "frozen_test_releases.csv"
    assert run("--freeze", "company_test", "validation") == 0
    out = capsys.readouterr().out
    assert "froze company_test, validation in data/frozen_test_releases.csv" in out
    rows = read(frozen)
    assert [(r["split"], r["release_accession"]) for r in rows] == [
        ("validation", AAPL),
        ("validation", MSFT),
        ("company_test", WMT),
    ]
    assert rows[2]["preferred_dividends"] == "True"
    assert rows[2]["linked_accession"] == "0000104169-25-000090"
    # Built from releases.csv in this run: provisional until the list is committed.
    manifest = json.loads((repo / "data/splits/company_test/manifest.json").read_text())
    assert manifest["status"] == "provisional"
    before = (repo / "data/splits/company_test/links.csv").read_bytes()

    # releases.csv later moves WMT out of company_test and XOM into it.
    meta = {**META, WMT: (*META[WMT][:2], "unused", True)}
    meta[XOM] = (*META[XOM][:2], "company_test", False)
    write_releases_csv(repo / "data/stage2/releases.csv", meta)
    assert run() == 0
    out = capsys.readouterr().out
    assert "| company_test | frozen | 1 | 0 | 1 | 48,111 | 3 (3) | 1 |" in out
    assert "| validation | frozen | 2 |" in out
    assert "| time_test | provisional | 0 | 1 | 0 | 0 | 0 (0) | 0 |" in out
    assert (
        "Note, company_test: releases.csv now gives 1 releases not in the frozen list "
        "and lacks 1 of it; built from the frozen list" in out
    )
    assert (repo / "data/splits/company_test/links.csv").read_bytes() == before
    manifest = json.loads((repo / "data/splits/company_test/manifest.json").read_text())
    assert manifest["status"] == "frozen"
    assert manifest["inputs"]["data/frozen_test_releases.csv"] == sha256(frozen)

    assert run("--freeze", "company_test") == 1
    assert "company_test is already frozen" in capsys.readouterr().err
    assert read(frozen) == rows


def change_answer(repo: Path) -> None:
    path = repo / "data/stage2/answers.csv"
    path.write_text(path.read_text().replace(",4487000000,", ",4488000000,", 1))


def change_input(repo: Path) -> None:
    path = repo / "data/stage2/releases/0000104169" / f"{WMT}.md"
    path.write_text(path.read_text() + "\nA footnote.\n")


def change_link(repo: Path) -> None:
    path = repo / "data/stage2/links.csv"
    path.write_text(path.read_text().replace(f"{WMT},2025-05-15", f"{WMT},2025-05-16"))


def change_preferred(repo: Path) -> None:
    meta = {**META, WMT: (*META[WMT][:3], False)}
    write_releases_csv(repo / "data/stage2/releases.csv", meta)


@pytest.mark.parametrize(
    ("change", "column"),
    [
        (change_answer, "answers_sha256"),
        (change_input, "input_sha256"),
        (change_link, "links_sha256"),
        (change_preferred, "preferred_dividends"),
    ],
)
def test_a_frozen_release_that_changed_stops_the_build(
    repo, change, column, capsys
) -> None:
    assert run("--freeze", "company_test") == 0
    capsys.readouterr()
    change(repo)
    assert run() == 1
    err = capsys.readouterr().err
    assert "1 differences from the frozen list: " in err
    assert f"company_test {WMT}: {column} is " in err
    assert ", not the frozen " in err


def test_answers_hash_ignores_row_order(repo) -> None:
    rows = read(repo / "data" / "answers.csv")
    release = splits.Release("validation", "AAPL", AAPL, "x", False, {}, rows, "")
    shuffled = splits.Release(
        "validation", "AAPL", AAPL, "x", False, {}, rows[::-1], ""
    )
    assert release.answers_sha256 == shuffled.answers_sha256
    changed = [{**rows[0], "value": "1"}, *rows[1:]]
    other = splits.Release("validation", "AAPL", AAPL, "x", False, {}, changed, "")
    assert other.answers_sha256 != release.answers_sha256


def test_validation_must_be_releases_csvs(repo, capsys) -> None:
    write_releases_csv(
        repo / "data/stage2/releases.csv",
        {**META, MSFT: ("MSFT", "789019", "train", False)},
    )
    assert run() == 1
    assert (
        "data/links.csv has linked releases that releases.csv doesn't put in "
        f"validation: {MSFT}" in capsys.readouterr().err
    )


def test_a_linked_release_without_its_files_stops_the_build(repo, capsys) -> None:
    shutil.rmtree(repo / "data/stage2/releases/0002115436")
    assert run() == 1
    assert f"time_test: no saved release {XOM} in data/stage2/releases" in (
        capsys.readouterr().err
    )


def test_uncommitted_changes_are_refused(repo, monkeypatch, capsys) -> None:
    monkeypatch.setattr(splits, "current_commit", lambda: ("abc", True))
    assert splits.main(["--prompts-dir", str(PROMPTS)], tokenizer=byte_tokenizer()) == 1
    assert "tracked files have uncommitted changes" in capsys.readouterr().err
    assert not (repo / "data" / "splits").exists()


def test_an_uncommitted_frozen_list_is_refused(repo, monkeypatch, capsys) -> None:
    assert run("--freeze", "company_test") == 0
    monkeypatch.setattr(splits, "current_commit", lambda: ("abc", False))
    monkeypatch.setattr(splits, "is_tracked", lambda path: False)
    assert splits.main(["--prompts-dir", str(PROMPTS)], tokenizer=byte_tokenizer()) == 1
    assert "data/frozen_test_releases.csv isn't committed; commit it first" in (
        capsys.readouterr().err
    )
