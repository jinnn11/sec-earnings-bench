import dataclasses
import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from secbench import compare, grade
from secbench.answers import Answer, answers_csv
from secbench.compare import is_stale
from secbench.grade import Graded, load_answers

D = Decimal
R1, R2, R3 = "0000000001-26-000001", "0000000001-26-000002", "0000000001-26-000003"
LINKED = {
    R1: "0000000001-26-000011",
    R2: "0000000001-26-000012",
    R3: "0000000001-26-000013",
}
LINKS_HEADER = (
    "ticker,release_cik,release_accession,release_date,linked_form,linked_cik,"
    "linked_accession,period_end,period_type,status,reason"
)
# Each release's answers: revenue 100 and net income 10 for the quarter.
VALUES = {"revenue": "100", "net_income": "10"}
TAGS = {"revenue": "Revenues", "net_income": "NetIncomeLoss"}


def write_links(path: Path) -> Path:
    lines = [LINKS_HEADER]
    for release, filing in LINKED.items():
        lines.append(
            f"TST,1,{release},2026-04-30,10-Q,1,{filing},2026-03-31,quarter,linked,"
        )
    lines.append("TST,1,0000000001-26-000004,2026-05-01,,,,,,excluded,no tables")
    path.write_text("\n".join(lines) + "\n")
    return path


def key_answers(revenue_r3: str = "100") -> list[Answer]:
    return [
        Answer(
            "TST",
            release,
            filing,
            field,
            "quarter",
            date(2026, 1, 1),
            date(2026, 3, 31),
            D(revenue_r3 if (release, field) == (R3, "revenue") else value),
            "USD",
            TAGS[field],
        )
        for release, filing in LINKED.items()
        for field, value in VALUES.items()
    ]


def graded_rows(*verdicts: str, extra: tuple[str, ...] = ()) -> list[Graded]:
    """Graded rows for R1, R2, R3 (revenue then net income) with these verdicts,
    plus a no_answer EPS row per release in extra."""
    rows = []
    answers = iter(key_answers())
    for verdict in verdicts:
        a = next(answers)
        predicted = a.value if verdict == grade.CORRECT else None
        rows.append(
            Graded(
                a.ticker,
                a.release_accession,
                a.linked_accession,
                a.field,
                a.period_type,
                a.value,
                a.tag,
                predicted,
                verdict,
                "",
            )
        )
    for release in extra:
        rows.append(
            Graded(
                "TST",
                release,
                LINKED[release],
                "eps_diluted",
                "quarter",
                None,
                "",
                D("1.5"),
                grade.NO_ANSWER,
                "no quarter answer for this release",
            )
        )
    return rows


def run_info(
    *,
    started: str,
    model: str,
    provider: str,
    parameters: dict,
    releases: list[str],
    outcomes: dict | None = None,
    latency: tuple = (None, None),
    status: str = "complete",
    max_parallel: int = 4,
    **cost: object,
) -> dict:
    info = {
        "run_name": "x",
        "status": status,
        "provider": provider,
        "model": model,
        "prompt_version": "extract_v1",
        "request_parameters": parameters,
        "max_parallel": max_parallel,
        "started": started,
        "releases": len(releases),
        "outcomes": {
            "ok": len(releases),
            "ok_after_retry": 0,
            "invalid_reply": 0,
            "api_error": 0,
            "aborted": 0,
        }
        | (outcomes or {}),
        "price_per_million_tokens": {"input": None, "output": None},
        "estimated_cost_usd": None,
        "latency_s": {"p50": latency[0], "p95": latency[1], "max": latency[1]},
        "calls": [{"release_accession": r, "call": 1} for r in releases],
    }
    return info | cost


def write_run(runs: Path, name: str, info: dict, graded: list[Graded] | None) -> None:
    folder = runs / name
    (folder / "raw").mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps(info))
    if graded is not None:
        (folder / "graded.csv").write_text(grade.graded_csv(graded))


ALL = [R1, R2, R3]


def vllm_rows() -> list[Graded]:
    """Two of the errors matched with a factor (unit slips)."""
    rows = graded_rows(
        grade.CORRECT,
        grade.WRONG_LINE,
        grade.CORRECT,
        grade.CORRECT,
        grade.ZEROS,
        grade.MISSING,
    )
    rows[1] = dataclasses.replace(rows[1], factor=D("0.1"))
    rows[4] = dataclasses.replace(rows[4], factor=D(10))
    return rows


OPUS = "max_tokens 16000, effort medium, no temperature"


@pytest.fixture
def fake_runs(tmp_path) -> tuple[Path, Path, Path]:
    """runs/ with seven folders, links.csv (3 linked releases) and answers.csv."""
    runs = tmp_path / "runs"
    runs.mkdir()
    # From before cost_basis was recorded (like the 2026-10-14 runs): token prices.
    write_run(
        runs,
        "a-opus",
        run_info(
            started="2026-10-14T01:00:00+00:00",
            model="claude-test",
            provider="anthropic",
            parameters={"max_tokens": 16000, "output_config": {"effort": "medium"}},
            releases=ALL,
            outcomes={"ok": 2, "ok_after_retry": 1},
            latency=(1.5, 2.5),
            # 4 calls (one release retried), 45,000 input tokens: 15,000 a release.
            tokens={"input": 45000, "output": 900, "calls_without_usage": 0},
            price_per_million_tokens={"input": 4.0, "output": 20.0},
            estimated_cost_usd=0.012,
        ),
        graded_rows(*[grade.CORRECT] * 6, extra=(R2,)),
    )
    write_run(
        runs,
        "b-vllm",
        run_info(
            started="2026-10-14T02:00:00+00:00",
            model="qwen3-4b",
            provider="openai",
            parameters={"temperature": 0.0, "reasoning_effort": "medium"},
            releases=ALL,
            outcomes={"ok": 2, "invalid_reply": 1},
            latency=(0.8, 1.25),
            max_parallel=16,
            # One call without token counts: no tokens per release.
            tokens={"input": 31000, "output": 700, "calls_without_usage": 1},
            # Its invalid reply had a format null: a figure still bad on the retry.
            prompt_version="extract_v2",
            reply_format="with_units",
            format_nulls=[
                {
                    "ticker": "TST",
                    "release_accession": R3,
                    "figure": "year_to_date.net_income",
                    "kind": "malformed",
                    "message": "year_to_date.net_income: should be null or ...",
                    "replied": "123",
                }
            ],
            gpu_hourly_usd=2.5,
            wall_time_s=72.0,
            cost_basis="gpu_hours",
            estimated_cost_usd=0.05,
        ),
        vllm_rows(),
    )
    # One release: the second table, though it started first.
    write_run(
        runs,
        "c-smoke",
        run_info(
            started="2026-10-14T00:30:00+00:00",
            model="claude-test",
            provider="anthropic",
            parameters={"max_tokens": 16000},
            releases=[R1],
            latency=(1.0, 1.0),
            tokens={"input": 12345, "output": 300, "calls_without_usage": 0},
            price_per_million_tokens={"input": 4.0, "output": 20.0},
            cost_basis="tokens",
            estimated_cost_usd=0.004,
        ),
        graded_rows(grade.CORRECT, grade.CORRECT),
    )
    write_run(
        runs,
        "d-stopped",
        run_info(
            started="2026-10-14T04:00:00+00:00",
            model="gpt-test",
            provider="openai",
            parameters={"temperature": 0.0},
            releases=[R1, R2],
            status="aborted",
            cost_basis=None,
        ),
        None,
    )
    # Graded when R3's revenue answer was 99.
    stale = graded_rows(*[grade.CORRECT] * 6)
    stale[4] = dataclasses.replace(stale[4], answer=D(99), predicted=D(99))
    write_run(
        runs,
        "e-stale",
        run_info(
            started="2026-10-14T03:00:00+00:00",
            model="gpt-test",
            provider="openai",
            parameters={},
            releases=ALL,
            latency=(2.0, 3.0),
            cost_basis=None,
            reply_format="raw_dollars",
            format_nulls=[],
        ),
        stale,
    )
    (runs / "f-empty").mkdir()
    (runs / "g-bad").mkdir()
    (runs / "g-bad" / "run.json").write_text("{")
    (runs / "notes.txt").write_text("not a run")
    links = write_links(tmp_path / "links.csv")
    answers = tmp_path / "answers.csv"
    answers.write_text(answers_csv(key_answers()))
    return runs, links, answers


MAIN_HEADER = (
    "| run | model | provider | prompt | settings sent | concurrency | releases | "
    "status | accuracy | revenue | net_income | eps_diluted | correct | missing | "
    "zeros | wrong_line | no_answer | unit slips | format retries | format nulls | "
    "input tokens / release | cost | cost basis | cost / 1,000 releases | "
    "latency p50 | latency p95 |"
)
OTHER_HEADER = (
    "| run | model | provider | prompt | settings sent | releases | status | "
    "accuracy | unit slips | format retries | format nulls | input tokens / release "
    "| cost | cost basis | latency p50 | latency p95 |"
)


def test_compare(fake_runs, tmp_path, capsys) -> None:
    runs, links, answers = fake_runs
    out = tmp_path / "results" / "comparison.md"
    argv = ["--runs-dir", str(runs), "--out", str(out), "--links", str(links)]
    assert compare.main([*argv, "--answers", str(answers)]) == 0
    text = out.read_text()
    printed = capsys.readouterr().out
    assert printed == text + f"wrote {out}\n"
    assert text.splitlines() == [
        "# Model comparison",
        "",
        (
            f"From `uv run python -m secbench.compare`: 5 runs in {runs}/, graded "
            f"against {answers}; {links} has 3 linked releases."
        ),
        "",
        "## Runs on all 3 linked releases",
        "",
        MAIN_HEADER,
        "|" + "|".join([" --- "] * 26) + "|",
        # 0.012 / 3 releases x 1,000 = $4.00.
        (
            f"| a-opus | claude-test | anthropic | extract_v1 | {OPUS} | 4 | 3 | graded "
            "| 6/6 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 0/0 | 6 | 0 | 0 | 0 | 1 | "
            "0 | 1 | – | 15,000 | $0.0120 | tokens ($4 in, $20 out per M) | $4.00 | "
            "1.50 s | 2.50 s |"
        ),
        # $2.50 per hour for 72 s is $0.05; / 3 x 1,000 = $16.67.
        (
            "| b-vllm | qwen3-4b | openai | extract_v2 | temperature 0, "
            "reasoning_effort medium | 16 | 3 | graded | 3/6 (50.0%) | 2/3 (66.7%) | "
            "1/3 (33.3%) | 0/0 | 3 | 1 | 1 | 1 | 0 | 2 | 1 (1 invalid) | 1 | none | "
            "$0.0500 | GPU ($2.5/h x 72.0 s) | $16.67 | 0.80 s | 1.25 s |"
        ),
        (
            "| e-stale | gpt-test | openai | extract_v1 | no temperature | 4 | 3 | "
            "stale: re-grade | 6/6 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 0/0 | 6 | "
            "0 | 0 | 0 | 0 | 0 | 0 | – | none | none | none | none | 2.00 s | 3.00 s |"
        ),
        "",
        "## Other runs: fewer releases, stopped or not graded",
        "",
        OTHER_HEADER,
        "|" + "|".join([" --- "] * 16) + "|",
        (
            "| c-smoke | claude-test | anthropic | extract_v1 | max_tokens 16000, "
            "default effort, no temperature | 1 | graded | 2/2 (100.0%) | 0 | 0 | – | "
            "12,345 | $0.0040 | tokens ($4 in, $20 out per M) | 1.00 s | 1.00 s |"
        ),
        (
            "| d-stopped | gpt-test | openai | extract_v1 | temperature 0 | 2 | "
            "stopped, not graded | none | none | 0 | – | none | none | none | none | "
            "none |"
        ),
        "",
        "Notes:",
        (
            "- Accuracy is correct figures over figures graded; not_in_input and "
            "no_answer are left out, as in each run's summary.txt."
        ),
        (
            "- Verdicts 0 in every run in the first table: sign, scale, wrong_period, "
            "noncontrolling_interest, rounded, adjusted, other, not_in_input."
        ),
        (
            "- Unit slips: errors matched with a power-of-ten factor (graded.csv's "
            "factor column): scale and zeros, and the other errors matched only after "
            "multiplying or dividing the prediction by 10 to 1,000,000."
        ),
        (
            "- Format retries: releases whose first reply didn't validate and were "
            'asked again; "(n invalid)" were still invalid and graded as missing.'
        ),
        (
            "- Format nulls (prompts with units, such as extract_v2): revenue and net "
            "income figures still malformed or with an unknown unit on the retry; each "
            'made its release an invalid reply (the "(n invalid)" of format retries); '
            "– for prompts without units."
        ),
        (
            "- Input tokens / release: run.json's input tokens over its releases, "
            "retries included. The same release counts as more or fewer tokens "
            "depending on the model's tokenizer; none if a call reported no token "
            "counts."
        ),
        (
            "- Cost is run.json's estimate: tokens at the run's prices per million, or "
            "GPU dollars per hour times the run's wall-clock time. Latency is per call."
        ),
        (
            f"- stale: re-grade: graded.csv doesn't match {answers}, or has other "
            "columns than the grader writes now; run `uv run python -m secbench.grade "
            "runs/<run>/`."
        ),
        "- Not listed: f-empty (no run.json), g-bad (run.json is not valid JSON).",
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runs_on_another_links_file_are_left_out(fake_runs, tmp_path) -> None:
    runs, links, answers = fake_runs
    split_links = tmp_path / "splits" / "company_test" / "links.csv"
    split_links.parent.mkdir(parents=True)
    split_links.write_text(LINKS_HEADER + "\n")
    info = json.loads((runs / "a-opus" / "run.json").read_text())
    # A test split's run (its predictions graded on the laptop), and a validation
    # run that recorded the same links file compare reads.
    test_run = info | {"links": str(split_links), "links_sha256": sha256(split_links)}
    graded = graded_rows(*[grade.CORRECT] * 6)
    write_run(runs, "h-test", test_run, graded)
    same = info | {"links": str(links), "links_sha256": sha256(links)}
    write_run(runs, "i-validation", same, graded)
    out = tmp_path / "comparison.md"
    argv = ["--runs-dir", str(runs), "--out", str(out), "--links", str(links)]
    assert compare.main([*argv, "--answers", str(answers)]) == 0
    text = out.read_text()
    assert "| h-test |" not in text
    assert "| i-validation | claude-test | anthropic |" in text
    assert f"6 runs in {runs}/" in text  # a-opus to e-stale, and i-validation
    assert text.splitlines()[-1] == (
        f"- Not listed: f-empty (no run.json), g-bad (run.json is not valid JSON), "
        f"h-test (on another links file, {split_links})."
    )


def test_no_full_runs(tmp_path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    links = write_links(tmp_path / "links.csv")
    answers = tmp_path / "answers.csv"
    answers.write_text(answers_csv(key_answers()))
    out = tmp_path / "comparison.md"
    argv = ["--runs-dir", str(runs), "--out", str(out), "--links", str(links)]
    assert compare.main([*argv, "--answers", str(answers)]) == 0
    lines = out.read_text().splitlines()
    assert lines[4:12] == [
        "## Runs on all 3 linked releases",
        "",
        "None.",
        "",
        "## Other runs: fewer releases, stopped or not graded",
        "",
        "None.",
        "",
    ]


def test_is_stale(tmp_path) -> None:
    path = tmp_path / "answers.csv"
    path.write_text(answers_csv(key_answers()))
    key = load_answers(path)
    rows = [
        {k: "" if v is None else str(v) for k, v in vars(g).items()}
        for g in graded_rows(*[grade.CORRECT] * 6)
    ]
    assert not is_stale(rows, set(ALL), key)
    assert not is_stale(rows[:2], {R1}, key)  # only the run's releases count
    # The key gained an answer since grading (as the derived fourth quarters did).
    assert is_stale(rows[:5], set(ALL), key)
    path.write_text(answers_csv(key_answers(revenue_r3="99")))  # an answer changed
    assert is_stale(rows, set(ALL), load_answers(path))


def test_graded_by_an_older_grader_is_stale(fake_runs) -> None:
    # graded.csv from before the factor column: its answers match, but unit slips
    # can't be counted.
    runs, links, answers = fake_runs
    old = runs / "a-opus" / "graded.csv"
    lines = old.read_text().splitlines()
    assert lines[0].endswith(",factor")
    old.write_text("".join(line.rsplit(",", 1)[0] + "\n" for line in lines))
    runs_loaded, _ = compare.load_runs(
        runs, load_answers(answers), links_sha256=sha256(links)
    )
    run = next(r for r in runs_loaded if r.name == "a-opus")
    assert (run.status, run.unit_slips()) == ("stale: re-grade", "none")
    assert compare.accuracy(run.verdicts()) == "6/6 (100.0%)"


def test_missing_inputs(fake_runs, tmp_path, capsys) -> None:
    runs, links, answers = fake_runs
    out = ["--out", str(tmp_path / "c.md")]
    base = ["--runs-dir", str(runs), "--links", str(links), "--answers", str(answers)]
    for name in ("--runs-dir", "--links", "--answers"):
        argv = [*base, *out]
        argv[argv.index(name) + 1] = str(tmp_path / "none")
        assert compare.main(argv) == 1
        assert f"error: {tmp_path / 'none'} not found" in capsys.readouterr().err
    assert not (tmp_path / "c.md").exists()


def test_compare_at_a_reference_gpu_price(fake_runs, tmp_path, capsys) -> None:
    runs, links, answers = fake_runs
    argv = ["--runs-dir", str(runs), "--links", str(links), "--answers", str(answers)]
    paid, reference = tmp_path / "paid.md", tmp_path / "reference.md"
    assert compare.main([*argv, "--out", str(paid)]) == 0
    assert compare.main([*argv, "--out", str(reference), "--gpu-hourly", "1"]) == 0
    capsys.readouterr()
    paid_lines = paid.read_text().splitlines()
    lines = reference.read_text().splitlines()
    assert lines[2].startswith(
        "From `uv run python -m secbench.compare --gpu-hourly 1`: 5 runs in "
    )
    # b-vllm paid $2.50 per hour; at $1, its 72 s cost $0.02, / 3 releases x 1,000 =
    # $6.67.
    [vllm] = [line for line in lines if line.startswith("| b-vllm ")]
    assert vllm.endswith(
        "| 1 (1 invalid) | 1 | none | $0.0200 | GPU at $1/h x 72.0 s (paid $2.5/h) | "
        "$6.67 | 0.80 s | 1.25 s |"
    )
    # Token-priced runs, and runs without a cost, are unchanged.
    for name in ("a-opus", "c-smoke", "d-stopped", "e-stale"):
        [before] = [line for line in paid_lines if line.startswith(f"| {name} ")]
        [after] = [line for line in lines if line.startswith(f"| {name} ")]
        assert after == before
    assert (
        "- Cost is run.json's estimate at the run's token prices per million; runs "
        "priced by GPU-hour are repriced at $1 per GPU-hour times the run's "
        "wall-clock time (--gpu-hourly), so they compare with each other (paid: the "
        "price each run recorded). Latency is per call."
    ) in lines


@pytest.mark.parametrize(
    ("value", "message"), [("cheap", "not a number"), ("-1", "must be 0 or more")]
)
def test_a_bad_gpu_price_is_refused(capsys, value, message) -> None:
    with pytest.raises(SystemExit) as raised:
        compare.main(["--gpu-hourly", value])
    assert raised.value.code == 2
    assert f"argument --gpu-hourly: {message}" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("info", "shown"),
    [
        # Every call counts, a retry's too: 45,000 over 3 releases.
        (
            {"releases": 3, "tokens": {"input": 45000, "calls_without_usage": 0}},
            "15,000",
        ),
        (
            {"releases": 60, "tokens": {"input": 917738, "calls_without_usage": 0}},
            "15,296",
        ),
        ({"releases": 3, "tokens": {"input": 45000, "calls_without_usage": 1}}, "none"),
        ({"releases": 3}, "none"),  # no token counts recorded
        ({"releases": 0, "tokens": {"input": 0, "calls_without_usage": 0}}, "none"),
    ],
)
def test_input_tokens_per_release(info, shown) -> None:
    assert compare.input_tokens_per_release(info) == shown
