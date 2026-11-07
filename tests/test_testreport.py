import dataclasses
import hashlib
import json
import subprocess
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from secbench import grade, testreport
from secbench.answers import Answer, LinkedRelease, answers_csv
from secbench.grade import Figures, Prediction
from secbench.testreport import (
    COMMON,
    QUARTER_AS_YTD,
    Bootstrap,
    Config,
    Freeze,
    Frozen,
    Model,
    ReportError,
    Table,
)

D = Decimal
SPLIT = testreport.COMPANY_TEST
PERIOD_END = date(2026, 6, 30)
A1, A2 = "0000000001-26-000001", "0000000001-26-000002"
B1, C1 = "0000000002-26-000001", "0000000003-26-000001"
TICKERS = {A1: "AAA", A2: "AAA", B1: "BBB", C1: "CCC"}
LINKS_HEADER = (
    "ticker,release_cik,release_accession,release_date,linked_form,linked_cik,"
    "linked_accession,period_end,period_type,status,reason"
)
# Each release's figures as printed, in millions: (quarter, six months).
PRINTED = {
    A1: {"revenue": (1000, 1900), "net_income": (100, 180), "eps": ("0.95", "1.70")},
    A2: {"revenue": (1100, 2100), "net_income": (110, 200), "eps": ("1.05", "1.90")},
    B1: {"revenue": (500, 990), "net_income": (50, 90), "eps": ("0.50", "0.90")},
    C1: {"revenue": (70, 140), "net_income": (7, 14), "eps": ("0.07", "0.14")},
}
PREFERRED = {A1: (5, 10)}  # preferred dividends: A1 also prints common's line


def text_lines(*lines: str) -> str:
    return "\n".join(lines)


def statement(accession: str) -> str:
    """A release's input: one income statement, quarter and six months, in
    millions."""
    p = PRINTED[accession]
    lines = [
        "# Second quarter results",
        "",
        "(in millions, except per share)",
        "",
        "| | Three Months Ended June 30, 2026 | Six Months Ended June 30, 2026 |",
        "| --- | --- | --- |",
        f"| Revenue | {p['revenue'][0]:,} | {p['revenue'][1]:,} |",
        f"| Net income | {p['net_income'][0]:,} | {p['net_income'][1]:,} |",
    ]
    if accession in PREFERRED:
        q, y = PREFERRED[accession]
        lines += [
            f"| Preferred stock dividends | ({q}) | ({y}) |",
            (
                "| Net income available to common stockholders | "
                f"{p['net_income'][0] - q:,} | {p['net_income'][1] - y:,} |"
            ),
        ]
    lines += [f"| Diluted earnings per share | ${p['eps'][0]} | ${p['eps'][1]} |", ""]
    return "\n".join(lines)


def answers() -> list[Answer]:
    """The key: every printed figure, except B1's six-month revenue, whose answer
    (991) isn't printed (not_in_input); C1 has no answer."""
    key = []
    for accession in (A1, A2, B1):
        p = PRINTED[accession]
        for period, start, i in (
            ("quarter", date(2026, 4, 1), 0),
            ("year_to_date", date(2026, 1, 1), 1),
        ):
            for name, value, unit, tag in (
                ("revenue", D(p["revenue"][i]) * 10**6, "USD", "Revenues"),
                ("net_income", D(p["net_income"][i]) * 10**6, "USD", "NetIncomeLoss"),
                (
                    "eps_diluted",
                    D(p["eps"][i]),
                    "USD/shares",
                    "EarningsPerShareDiluted",
                ),
            ):
                if (accession, name, period) == (B1, "revenue", "year_to_date"):
                    value = D(991) * 10**6
                key.append(
                    Answer(
                        TICKERS[accession],
                        accession,
                        accession.replace("-26-", "-27-"),
                        name,
                        period,
                        start,
                        PERIOD_END,
                        value,
                        unit,
                        tag,
                    )
                )
    return key


def as_printed(accession: str) -> dict[str, tuple[Decimal, Decimal]]:
    p = PRINTED[accession]
    return {
        "revenue": tuple(D(v) * 10**6 for v in p["revenue"]),
        "net_income": tuple(D(v) * 10**6 for v in p["net_income"]),
        "eps_diluted": tuple(D(v) for v in p["eps"]),
    }


def prediction(accession: str, **changes: Decimal | None) -> Prediction:
    """The figures as printed (C1: all null), with changes keyed field_period
    ("net_income_quarter")."""
    values = as_printed(accession)
    groups = {}
    for i, group in enumerate(("quarter", "year_to_date")):
        groups[group] = Figures(
            **{
                name: changes.get(
                    f"{name}_{group}", None if accession == C1 else values[name][i]
                )
                for name in values
            }
        )
    return Prediction(release_accession=accession, period_end=PERIOD_END, **groups)


# step-00025: A1's quarter net income on common's line (95), B1's quarter copied
# into year to date (50), and B1's unprinted six-month revenue answered (990).
STEP = {
    A1: prediction(A1, net_income_quarter=D(95_000_000)),
    A2: prediction(A2),
    B1: prediction(B1, net_income_year_to_date=D(50_000_000)),
    C1: prediction(C1),
}
# The base model: A1's quarter revenue in millions (scale), A2's quarter net income a
# zero short (zeros), and B1's unprinted six-month revenue left null.
BASE = {
    A1: prediction(A1, revenue_quarter=D(1000)),
    A2: prediction(A2, net_income_quarter=D(11_000_000)),
    B1: prediction(B1, revenue_year_to_date=None),
    C1: prediction(C1),
}


def links() -> list[LinkedRelease]:
    return [
        LinkedRelease(TICKERS[a], a, "10-Q", a.replace("-26-", "-27-"), PERIOD_END)
        for a in (A1, A2, B1, C1)
    ]


def graded_text(predictions: dict[str, Prediction]) -> str:
    key = {(a.release_accession, a.field, a.period_type): a for a in answers()}
    rows = []
    for release in links():
        rows.extend(
            grade.grade_prediction(
                predictions[release.release_accession],
                release,
                key,
                set(),
                statement(release.release_accession),
            )
        )
    return grade.graded_csv(rows)


def run_info(model: str, name: str, releases_dir: Path, **changes) -> dict:
    calls = [
        {
            "ticker": TICKERS[a],
            "release_accession": a,
            "call": 1,
            "status": 200,
            "http_attempts": 1,
            "latency_s": 2.0,
            "error": "",
        }
        for a in (A1, A2, B1, C1)
    ]
    info = {
        "run_name": name,
        "status": "complete",
        "commit": testreport.RUN_COMMITS[SPLIT],
        "clean": True,
        "links_sha256": testreport.SPLIT_FILES[SPLIT]["links.csv"],
        "answers_sha256": testreport.SPLIT_FILES[SPLIT]["answers.csv"],
        "releases_dir": str(releases_dir),
        "provider": "openai",
        "model": model,
        "env_file": f"training/test/{name}.env",
        "prompt_version": "extract_v2",
        "prompt_sha256": testreport.PROMPT_SHA256,
        "reply_format": "with_units",
        "request_parameters": {"temperature": 0.0},
        "max_parallel": 16,
        "started": "2026-11-07T11:00:00+00:00",
        "finished": "2026-11-07T11:00:36+00:00",
        "wall_time_s": 36.0,
        "selection": {"tickers": None, "accessions": None, "limit": None},
        "releases": 4,
        "outcomes": {
            "ok": 4,
            "ok_after_retry": 0,
            "invalid_reply": 0,
            "api_error": 0,
            "aborted": 0,
        },
        "fallbacks": [],
        "price_per_million_tokens": {"input": None, "output": None},
        "gpu_hourly_usd": 2.0,
        "cost_basis": "gpu_hours",
        "estimated_cost_usd": 0.02,
        "latency_s": {"p50": 2.0, "p95": 2.0, "max": 2.0},
        "calls": calls,
    }
    return info | changes


def write_run(folder: Path, info: dict, predictions: dict, graded: str | None) -> Path:
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps(info))
    (folder / "predictions.jsonl").write_text(
        "".join(p.model_dump_json() + "\n" for p in predictions.values())
    )
    if graded is not None:
        (folder / "graded.csv").write_text(graded)
    return folder


@dataclasses.dataclass
class World:
    root: Path
    config: Config
    splits_dir: Path
    stage2_releases: Path
    companies_csv: Path

    def run(self, model: str) -> testreport.Run:
        return testreport.load_run(model, self.config.model(model).runs[SPLIT])

    def split(self) -> testreport.Split:
        return testreport.load_split(SPLIT, self.splits_dir)


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A company test of four releases from three companies (CCC's has no answer),
    a step-00025 and a base run on it, their validation references, and the files
    the report reads."""
    releases = tmp_path / "releases"
    for accession in (A1, A2, B1, C1):
        folder = releases / accession[:10]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{accession}.md").write_text(statement(accession))
        record = {
            "accession_number": accession,
            "exhibits": [
                {"type": "EX-99.1", "file": f"{accession}.md", "chosen": True}
            ],
        }
        (folder / f"{accession}.json").write_text(json.dumps(record))
    folder = tmp_path / "splits" / SPLIT
    folder.mkdir(parents=True)
    (folder / "links.csv").write_text(
        "\n".join(
            [LINKS_HEADER]
            + [
                f"{r.ticker},1,{r.release_accession},2026-07-20,10-Q,1,"
                f"{r.linked_accession},{r.period_end},quarter,linked,"
                for r in links()
            ]
        )
        + "\n"
    )
    (folder / "answers.csv").write_text(answers_csv(answers()))
    (folder / "preferred.csv").write_text(
        "ticker,release_accession,status\nAAA," + A1 + ",kept\n"
    )
    (folder / "left_out.csv").write_text(
        "ticker,release_accession,prompt_tokens,reason\n"
    )
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "releases_dir": str(releases),
                "counts": {"answers": 18, "kept": 4, "answers_in_input": 17},
            }
        )
    )
    stage2 = tmp_path / "releases.csv"
    columns = testreport.RELEASE_COLUMNS
    stage2.write_text(
        ",".join(columns)
        + "\n"
        + "".join(
            ",".join(
                {
                    "ticker": TICKERS[a],
                    "release_accession": a,
                    "answers": "0" if a == C1 else "6",
                    "missing": "6" if a == C1 else "0",
                }.get(c, "")
                for c in columns
            )
            + "\n"
            for a in (A1, A2, B1, C1)
        )
    )
    companies = tmp_path / "companies.csv"
    companies.write_text(
        "ticker,cik,name,sector,split\n"
        "AAA,3,A Corp,Industrials,company_test\n"
        "BBB,2,B Corp,Industrials,company_test\n"
        "CCC,1,C Corp,Utilities,company_test\n"
    )
    runs = tmp_path / "runs"
    models = []
    for key, model, predictions in (
        ("step-00025", "qwen3-8b-2k--step-00025", STEP),
        ("base", "Qwen/Qwen3-8B", BASE),
    ):
        name = f"qwen3-8b-2k--{SPLIT}--{key}"
        write_run(
            runs / name,
            run_info(model, name, releases),
            predictions,
            graded_text(predictions),
        )
        validation = f"qwen3-8b-2k--{key}"
        write_run(
            runs / validation,
            run_info(model, validation, releases, gpu_hourly_usd=1.0),
            {},
            None,
        )
        models.append(
            Model(key, key, runs / validation, {SPLIT: runs / name, "time_test": runs})
        )
    logs = {}
    for kind in ("server", "setup"):
        logs[kind] = tmp_path / f"{kind}.log"
        logs[kind].write_text("")
    config = Config(
        path=tmp_path / "stage2.toml",
        under_test="step-00025",
        base="base",
        bootstrap_details_fixed=date(2026, 11, 7),
        gpu_host="Vast",
        models=tuple(models),
        billed={"validation": 2.776, SPLIT: 2.5, "time_test": 2.5},
        validation_gpu="H100 NVL",
        recorded_price_note={},
        logs={SPLIT: logs},
        freeze=None,
        attempts=(),
        notes=(),
    )
    return World(tmp_path, config, tmp_path / "splits", stage2, companies)


def same_text(run: testreport.Run, split: testreport.Split) -> str:
    return run.graded_text or ""


# Bootstrap


def test_times_drawn_counts_each_company_in_each_draw():
    picks = np.array([[0, 0], [0, 1], [1, 1]])
    assert testreport.times_drawn(picks, 2).tolist() == [[2, 0], [1, 1], [0, 2]]


def test_bootstrap_on_a_case_computed_by_hand():
    # AAA: 2 figures, 1 correct; BBB: 1 figure, correct. Draws AAA+AAA, AAA+BBB and
    # BBB+BBB give (2 x 1) / (2 x 2) = 50%, (1 + 1) / (2 + 1) = 66.67% and 2 / 2 =
    # 100%. numpy's linear percentiles over three values: the 2.5th is 0.05 of the
    # way from 50 to 66.67, the 97.5th 0.95 of the way from 66.67 to 100.
    boot = Bootstrap(["AAA", "BBB"], draws=3)
    boot.counts = np.array([[2, 0], [1, 1], [0, 2]])
    figures = [("AAA", True), ("AAA", False), ("BBB", True)]
    values = boot.accuracies(figures, "all")
    assert values.tolist() == pytest.approx([50, 200 / 3, 100])
    low, high = testreport.interval(values)
    assert low == pytest.approx(50 + 0.05 * 50 / 3)
    assert high == pytest.approx(200 / 3 + 0.95 * 100 / 3)


def test_bootstrap_draws_as_the_protocol_states():
    boot = Bootstrap(["AAA", "BBB", "CCC"])
    picks = np.random.default_rng(20261105).integers(0, 3, size=(10_000, 3))
    assert boot.counts.shape == (10_000, 3)
    assert (boot.counts == testreport.times_drawn(picks, 3)).all()
    assert (boot.counts.sum(axis=1) == 3).all()


def test_a_difference_is_computed_within_each_draw():
    r1, r2, r3 = (
        ("r1", "revenue", "quarter"),
        ("r2", "revenue", "quarter"),
        ("r3", "revenue", "quarter"),
    )
    # Both models vary across draws, so subtracting their intervals, or pairing
    # other draws, gives another interval ([-45.83, +96.67]).
    table = Table(
        [r1, r2, r3],
        {r1: "AAA", r2: "BBB", r3: "AAA"},
        {
            "a": {r1: True, r2: False, r3: True},
            "b": {r1: True, r2: False, r3: False},
        },
    )
    boot = Bootstrap(["AAA", "BBB"], draws=3)
    boot.counts = np.array([[2, 0], [1, 1], [0, 2]])
    entry = testreport.difference_entry(table, "a", "b", table.keys, boot, "minus b")
    # Within each draw, a minus b: 100 - 50, 66.67 - 33.33, 0 - 0.
    assert entry["difference"] == pytest.approx(100 / 3)
    assert entry["interval"] == pytest.approx(
        list(np.percentile([50, 100 / 3, 0], [2.5, 97.5]))
    )
    assert (entry["discordant"], entry["first_right"], entry["second_right"]) == (
        1,
        1,
        0,
    )
    assert entry["companies"] == 1


def test_a_company_without_figures_in_a_subset_adds_nothing():
    boot = Bootstrap(["AAA", "BBB"], draws=2)
    boot.counts = np.array([[1, 1], [0, 2]])
    # Only AAA has figures in this subset: the second draw has none of AAA's.
    with pytest.raises(ReportError, match="1 of 2 draws have no graded figure"):
        boot.accuracies([("AAA", True)], "subset")
    boot.counts = np.array([[1, 1], [2, 0]])
    # BBB adds 0 to both sums: each draw is AAA's accuracy.
    assert boot.accuracies([("AAA", True), ("AAA", False)], "subset").tolist() == [
        50,
        50,
    ]


def test_the_share_of_draws_without_some_companies():
    boot = Bootstrap(["AAA", "BBB", "CCC"], draws=4)
    boot.counts = np.array([[3, 0, 0], [1, 1, 1], [0, 2, 1], [0, 0, 3]])
    assert boot.without(["BBB", "CCC"]) == 0.25
    assert boot.without(["AAA"]) == 0.5


# Checks: frozen items


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


SPLIT_NAMES = ("validation", SPLIT, "time_test")
CSV_NAMES = ("links.csv", "answers.csv", "tokens.csv", "left_out.csv", "preferred.csv")
RECORD = {
    "test_config": {"sha256": sha("config\n")},
    "checkpoint": {"adapter_sha256": "adapter"},
    "layout_check": {"identical_to_training_layout": True},
}


@pytest.fixture
def frozen(tmp_path: Path) -> Frozen:
    """A repository holding every frozen item, committed; a Frozen of their hashes."""
    files = {
        "results/test_protocol.md": "protocol\n",
        "prompts/extract_v2.txt": "prompt\n",
        "configs/test/qwen3-8b-2k.toml": "config\n",
        "src/secbench/grade.py": "# grade\n",
        "src/secbench/inputs.py": "# inputs\n",
        "data/frozen_test_releases.csv": (
            "split,ticker\nvalidation,AAA\ncompany_test,BBB\ntime_test,CCC\n"
        ),
    }
    for split in SPLIT_NAMES:
        for name in CSV_NAMES:
            files[f"data/splits/{split}/{name}"] = f"{split} {name}\n"
    for split in (SPLIT, "time_test"):
        files[f"training/run/test/{split}.json"] = json.dumps(RECORD)
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "frozen")
    return Frozen(
        root=tmp_path,
        protocol=Path("results/test_protocol.md"),
        protocol_sha256=sha("protocol\n"),
        splits_dir=Path("data/splits"),
        split_files={
            split: {name: sha(f"{split} {name}\n") for name in CSV_NAMES}
            for split in SPLIT_NAMES
        },
        frozen_list=Path("data/frozen_test_releases.csv"),
        frozen_list_sha256=sha(files["data/frozen_test_releases.csv"]),
        frozen_list_at_protocol_sha256=sha(
            "split,ticker\nvalidation,AAA\ncompany_test,BBB\n"
        ),
        prompt=Path("prompts/extract_v2.txt"),
        prompt_sha256=sha("prompt\n"),
        test_config=Path("configs/test/qwen3-8b-2k.toml"),
        test_config_sha256=sha("config\n"),
        adapter_sha256="adapter",
        test_records=Path("training/run/test"),
        commit=git(tmp_path, "rev-parse", "HEAD"),
        code=(
            "src/secbench/grade.py",
            "src/secbench/inputs.py",
            "prompts/extract_v2.txt",
            "configs/test/qwen3-8b-2k.toml",
        ),
    )


def test_frozen_items_unchanged(frozen: Frozen):
    lines = testreport.check_frozen(frozen)
    assert "15 split files" in lines[1]
    assert "no change since" in lines[-1]


def edit(frozen: Frozen, path: str, text: str) -> None:
    (frozen.root / path).write_text(text)


@pytest.mark.parametrize(
    ("path", "text", "message"),
    [
        ("results/test_protocol.md", "amended\n", "protocol results/test_protocol.md"),
        ("data/splits/time_test/answers.csv", "x\n", "split file"),
        ("data/splits/company_test/extra.csv", "x\n", "CSV files"),
        ("data/frozen_test_releases.csv", "split,ticker\n", "frozen release list"),
        ("prompts/extract_v2.txt", "prompt v3\n", "prompt prompts/extract_v2.txt"),
        ("configs/test/qwen3-8b-2k.toml", "x\n", "test config"),
        ("src/secbench/inputs.py", "# changed\n", "changed since"),
    ],
)
def test_a_changed_frozen_item_stops_the_report(frozen, path, text, message):
    edit(frozen, path, text)
    with pytest.raises(ReportError, match="frozen items changed") as error:
        testreport.check_frozen(frozen)
    assert message in str(error.value)


def test_a_committed_change_to_frozen_code_stops_the_report(frozen: Frozen):
    edit(frozen, "src/secbench/grade.py", "# a grader change\n")
    git(frozen.root, "commit", "-q", "-am", "change the grader")
    with pytest.raises(ReportError, match="changed since .*src/secbench/grade.py"):
        testreport.check_frozen(frozen)


def test_the_frozen_list_keeps_the_protocol_commits_rows(frozen: Frozen):
    text = "split,ticker\nvalidation,AAA\ncompany_test,ZZZ\ntime_test,CCC\n"
    edit(frozen, "data/frozen_test_releases.csv", text)
    # Even with the new file's own sha256 expected, a changed company_test row stops.
    changed = dataclasses.replace(frozen, frozen_list_sha256=sha(text))
    with pytest.raises(ReportError, match="rows differ from the protocol commit's"):
        testreport.check_frozen(changed)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"checkpoint": {"adapter_sha256": "other"}}, "another adapter"),
        ({"test_config": {"sha256": "other"}}, "another test config"),
        (
            {"layout_check": {"identical_to_training_layout": False}},
            "not identical to training's layout",
        ),
    ],
)
def test_a_test_record_must_name_the_frozen_config_and_adapter(frozen, change, message):
    edit(frozen, "training/run/test/time_test.json", json.dumps(RECORD | change))
    with pytest.raises(ReportError, match=message):
        testreport.check_frozen(frozen)


# Checks: runs


def test_reported_runs_pass_every_check(world: World):
    split = world.split()
    for model in ("step-00025", "base"):
        run = world.run(model)
        testreport.check_run(run, split, testreport.RUN_COMMITS[SPLIT])
        testreport.check_graded(run, split, same_text)
    assert testreport.check_same_keys([world.run("step-00025"), world.run("base")]) == (
        17,
        1,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"status": "aborted"}, "status 'aborted'"),
        ({"selection": {"tickers": None, "accessions": None, "limit": 2}}, "selection"),
        ({"links_sha256": "0" * 64}, "links_sha256"),
        ({"answers_sha256": "0" * 64}, "answers_sha256"),
        ({"commit": "0" * 40}, "commit 0000000"),
        ({"clean": False}, "clean"),
        (
            {
                "outcomes": {
                    "ok": 3,
                    "ok_after_retry": 0,
                    "invalid_reply": 0,
                    "api_error": 0,
                    "aborted": 1,
                }
            },
            "1 aborted calls",
        ),
    ],
)
def test_a_run_that_fails_a_check_stops_the_report(world, change, message):
    run = world.run("step-00025")
    run.info |= change
    with pytest.raises(ReportError) as error:
        testreport.check_run(run, world.split(), testreport.RUN_COMMITS[SPLIT])
    assert str(error.value).startswith(f"qwen3-8b-2k--{SPLIT}--step-00025: ")
    assert message in str(error.value)


def test_a_run_must_cover_exactly_the_kept_releases(world: World):
    run = world.run("step-00025")
    run.info["calls"] = run.info["calls"][:-1]
    with pytest.raises(ReportError, match="run.json's calls cover 3 releases"):
        testreport.check_run(run, world.split(), testreport.RUN_COMMITS[SPLIT])
    run = world.run("step-00025")
    del run.predictions[C1]
    with pytest.raises(ReportError, match="predictions.jsonl cover 3 releases"):
        testreport.check_run(run, world.split(), testreport.RUN_COMMITS[SPLIT])


def api_error(run: testreport.Run, reason: str) -> None:
    run.info["outcomes"] = run.info["outcomes"] | {"ok": 3, "api_error": 1}
    run.info["fallbacks"] = [
        {
            "ticker": "CCC",
            "release_accession": C1,
            "outcome": "api_error",
            "reason": reason,
        }
    ]


def test_a_refusal_is_the_models_answer(world: World):
    run = world.run("step-00025")
    api_error(run, "refusal (cyber): the model declined")
    testreport.check_run(run, world.split(), testreport.RUN_COMMITS[SPLIT])
    assert run.refusals == 1


def test_an_infrastructure_error_stops_the_report_without_its_text(world: World):
    run = world.run("step-00025")
    reason = "HTTP 429 (Rate limit reached for organization org-secret)"
    api_error(run, reason)
    with pytest.raises(ReportError, match="CCC .*isn't a refusal") as error:
        testreport.check_run(run, world.split(), testreport.RUN_COMMITS[SPLIT])
    assert "org-secret" not in str(error.value)
    assert "429" not in str(error.value)


def test_graded_csv_must_be_current(world: World):
    split = world.split()
    run = world.run("step-00025")
    with pytest.raises(ReportError, match="doesn't give graded.csv byte for byte"):
        testreport.check_graded(run, split, lambda r, s: "other\n")


def test_stale_graded_csv_stops_the_report(world: World):
    split = world.split()
    key = (A2, "revenue", "quarter")
    split.key[key] = dataclasses.replace(split.key[key], value=D(1_200_000_000))
    with pytest.raises(ReportError, match="graded.csv is stale"):
        testreport.check_graded(world.run("step-00025"), split, same_text)


def test_graded_values_must_be_the_predictions(world: World):
    run = world.run("step-00025")
    run.predictions[A2] = prediction(A2, revenue_year_to_date=D(2_000_000_000))
    with pytest.raises(ReportError, match=f"{A2} revenue year_to_date: predicted"):
        testreport.check_graded(run, world.split(), same_text)


def test_a_predicted_figure_needs_a_row(world: World):
    run = world.run("step-00025")
    # C1 has no answer: a figure predicted for it is a no_answer row.
    run.predictions[C1] = prediction(C1, eps_diluted_quarter=D("0.07"))
    with pytest.raises(ReportError, match=f"{C1} quarter eps_diluted: predicted"):
        testreport.check_graded(run, world.split(), same_text)


def test_graded_csv_must_have_the_graders_columns(world: World):
    run = world.run("step-00025")
    run.graded_text = (run.graded_text or "").replace(",factor\n", "\n", 1)
    with pytest.raises(ReportError, match="has columns"):
        testreport.check_graded(run, world.split(), same_text)


def test_full_year_figures_are_the_year_to_date_prediction():
    p = prediction(A1)
    assert testreport.predicted_value(p, (A1, "revenue", "full_year")) == D(
        1_900_000_000
    )
    assert testreport.predicted_value(p, (A1, "revenue", "quarter")) == D(1_000_000_000)


def test_settings_must_equal_the_validation_reference(world: World):
    run = world.run("step-00025")
    reference = testreport.load_run("step-00025", world.config.models[0].validation)
    notes = testreport.check_settings(run, reference)
    assert notes == [
        (
            f"env_file training/test/qwen3-8b-2k--{SPLIT}--step-00025.env (reference "
            "training/test/qwen3-8b-2k--step-00025.env)"
        ),
        "gpu_hourly_usd 2.0 (reference 1.0)",
    ]
    # A reference run made without --env-file records none.
    reference.info["env_file"] = None
    assert testreport.check_settings(run, reference)[0] == (
        f"env_file training/test/qwen3-8b-2k--{SPLIT}--step-00025.env (reference none)"
    )
    run.info["request_parameters"] = {"temperature": 0.7}
    with pytest.raises(ReportError, match="request_parameters differ"):
        testreport.check_settings(run, reference)


def test_every_run_on_a_split_has_the_same_keys(world: World):
    step, base = world.run("step-00025"), world.run("base")
    step.graded_text = "\n".join(
        line
        for line in (step.graded_text or "").splitlines()
        if not line.startswith(f"AAA,{A2},") or ",revenue,quarter," not in line
    )
    with pytest.raises(ReportError, match="different graded keys"):
        testreport.check_same_keys([base, step])
    step = world.run("step-00025")
    step.graded_text = (step.graded_text or "").replace("not_in_input", "missing")
    with pytest.raises(ReportError, match="different graded keys"):
        testreport.check_same_keys([base, step])


# Analysis


def test_a_split_analysed_end_to_end(world: World):
    result, checks = testreport.analyse_split(
        world.config,
        SPLIT,
        world.splits_dir,
        same_text,
        world.stage2_releases,
        world.companies_csv,
    )
    assert "same 17 graded and 1 not_in_input" in checks[-1]
    step, base = result["accuracy"]["step-00025"], result["accuracy"]["base"]
    assert (step["all"]["correct"], step["all"]["graded"]) == (15, 17)
    assert (base["all"]["correct"], base["all"]["graded"]) == (15, 17)
    assert "full_year" not in step  # no full-year figure in this split
    # Without A1 (preferred dividends): 11 figures.
    assert step["without_preferred"]["graded"] == 11
    difference = result["differences"]["base"]["all"]
    assert difference["difference"] == 0
    assert (difference["first_right"], difference["second_right"]) == (2, 2)
    assert difference["companies"] == 2
    assert result["preferred"]["net_income"]["step-00025"] == {
        "correct": 1,
        "graded": 2,
    }
    coverage = result["coverage"]
    assert (coverage["expected"], coverage["answered"], coverage["missing"]) == (
        24,
        18,
        6,
    )
    assert (coverage["releases_graded"], coverage["companies_graded"]) == (3, 2)
    assert coverage["no_answer_by_sector"]["Utilities"] == {"no_answer": 1, "kept": 1}
    assert result["abstention"]["step-00025"] == {
        "not_in_input": 1,
        "answered": 1,
        "figures": 24,
        "null": 6,
    }
    assert result["abstention"]["base"]["answered"] == 0
    # By ticker and in links.csv order, AAA then BBB; by CIK, BBB first.
    robustness = result["robustness"]
    assert robustness["links_order"] == robustness["ticker"]
    assert set(robustness) == {"ticker", "links_order", "cik"}
    assert result["unit_slips"]["base"] == 2  # scale and zeros
    groups = [(m["ticker"], m["group"], m["base_verdict"]) for m in result["misses"]]
    assert groups == [("AAA", COMMON, "correct"), ("BBB", QUARTER_AS_YTD, "correct")]
    report = {
        "models": {"step-00025": "step-00025", "base": "Qwen3-8B base"},
        "under_test": "step-00025",
        "base": "base",
        "splits": {SPLIT: result},
    }
    text = "\n".join(testreport.render_split(report, SPLIT))
    assert "| minus Qwen3-8B base | 0.00 |" in text
    assert "Group (1): 1 of the 2 misses, from 1 company (AAA)" in text
    # A1 is in preferred.csv: the sensitivity analysis drops the whole group.
    assert "outside it, so the sensitivity analysis drops all of them." in text
    assert "understates" not in text
    # Net income on the preferred-dividend releases: A1's two figures.
    assert "interval: the 2 figures come from 1 company (" in text
    assert "upper bound on missed abstentions" in text
    assert "doesn't print" not in text
    # With a group (1) miss outside preferred.csv, the sentence says why and that
    # the sensitivity analysis understates the failure.
    summary = result["miss_summary"]
    summary |= {
        "common_in_preferred": 0,
        "common_outside_preferred": 1,
        "outside_rows": {"AAA": ["participating"]},
    }
    text = "\n".join(testreport.render_split(report, SPLIT))
    assert "1 outside it: AAA report income after participating securities" in text
    assert "drops all of them" not in text
    assert "so it understates this failure." in text


def test_companies_in_other_orders(world: World):
    orders = testreport.company_orders(
        world.split(), ["BBB", "AAA"], {"AAA": 3, "BBB": 2}
    )
    assert orders == {
        "ticker": ["AAA", "BBB"],
        "links_order": ["AAA", "BBB"],
        "cik": ["BBB", "AAA"],
    }


def test_robustness_draws_each_order_as_given(monkeypatch):
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    keys = [(f"r{i}", "revenue", "quarter") for i in range(len(tickers))]
    table = Table(
        keys,
        dict(zip(keys, tickers, strict=True)),
        {
            "a": {k: i < 6 for i, k in enumerate(keys)},
            "b": {k: i < 3 for i, k in enumerate(keys)},
        },
    )
    seen = []
    real = testreport.Bootstrap

    def recording(companies, *args, **kwargs):
        seen.append(list(companies))
        return real(companies, *args, **kwargs)

    monkeypatch.setattr(testreport, "Bootstrap", recording)
    orders = {"ticker": tickers, "reversed": tickers[::-1]}
    result = testreport.robustness(table, "a", "b", orders)
    # Each order is drawn as given (not sorted), and the interval is within draws.
    assert seen == [tickers, tickers[::-1]]
    for name, order in orders.items():
        boot = real(order)
        within = boot.accuracies(table.figures("a", keys), name) - boot.accuracies(
            table.figures("b", keys), name
        )
        assert result[name] == pytest.approx(testreport.interval(within))


def test_the_offline_regrade_grades_against_the_split_files(world: World, monkeypatch):
    seen = {}

    def grade_run(run_dir, links, answers, releases, client):
        seen.update(
            run_dir=run_dir,
            links=links,
            answers=answers,
            releases=releases,
            offline=client.offline,
        )
        return [], None

    monkeypatch.setattr(grade, "grade_run", grade_run)
    run, split = world.run("step-00025"), world.split()
    assert testreport.regrade_offline(run, split) == grade.graded_csv([])
    assert seen == {
        "run_dir": run.path,
        "links": split.folder / "links.csv",
        "answers": split.folder / "answers.csv",
        "releases": Path(run.info["releases_dir"]),
        "offline": True,
    }


# Misses


def miss_world(
    texts: dict[str, str], predictions: dict[str, Prediction], key: list[Answer]
) -> tuple[testreport.Run, testreport.Split]:
    releases = [
        LinkedRelease("TST", a, "10-Q", "0000000009-26-000001", PERIOD_END)
        for a in predictions
    ]
    by_key = {(a.release_accession, a.field, a.period_type): a for a in key}
    rows = []
    for release in releases:
        rows.extend(
            grade.grade_prediction(
                predictions[release.release_accession],
                release,
                by_key,
                set(),
                texts[release.release_accession],
            )
        )
    run = testreport.Run("x", Path("x"), {}, grade.graded_csv(rows), predictions)
    split = testreport.Split(
        SPLIT, Path("."), Path("."), by_key, releases, {A1}, [], {}
    )
    return run, split


def net_income_answers(accession: str, quarter: int, ytd: int) -> list[Answer]:
    return [
        Answer(
            "TST",
            accession,
            "0000000009-26-000001",
            "net_income",
            period,
            date(2026, 1, 1),
            PERIOD_END,
            D(value),
            "USD",
            "NetIncomeLoss",
        )
        for period, value in (("quarter", quarter), ("year_to_date", ytd))
    ]


def only_net_income(accession: str, quarter, ytd) -> Prediction:
    return Prediction(
        release_accession=accession,
        period_end=PERIOD_END,
        quarter=Figures(net_income=quarter),
        year_to_date=Figures(net_income=ytd),
    )


def test_misses_grouped_by_the_stated_rule():
    texts = {
        A1: statement(A1),
        A2: text_lines(
            "(in thousands)",
            "",
            "| | Three Months Ended June 30, 2026 | Six Months Ended June 30, 2026 |",
            "| --- | --- | --- |",
            "| Net income | 110,000 | 200,000 |",
            "| Less: income allocated to participating securities | (1,000) | (2,000) |",
            "| Net income applicable to common shares | 109,000 | 198,000 |",
            "",
            (
                "GAAP net income attributable to common shareholders rose to $107 "
                "million."
            ),
            "",
        ),
        B1: statement(B1),
    }
    predictions = {
        # On common's line (95, in millions), with a preferred-dividend row.
        A1: only_net_income(A1, D(95_000_000), D(180_000_000)),
        # Common's line in thousands; the year to date on a prose line.
        A2: only_net_income(A2, D(109_000_000), D(107_000_000)),
        # The quarter copied into year to date: (2), though no common line here.
        B1: only_net_income(B1, D(50_000_000), D(50_000_000)),
    }
    key = (
        net_income_answers(A1, 100_000_000, 180_000_000)
        + net_income_answers(A2, 110_000_000, 200_000_000)
        + net_income_answers(B1, 50_000_000, 90_000_000)
    )
    run, split = miss_world(texts, predictions, key)
    base, _ = miss_world(
        texts,
        {a: only_net_income(a, None, None) for a in predictions},
        key,
    )
    misses = testreport.group_misses(run, base, split, texts)
    found = {(m["release_accession"], m["period_type"]): m for m in misses}
    assert len(misses) == 4
    a1 = found[A1, "quarter"]
    assert (a1["group"], a1["row"]) == (
        COMMON,
        "Net income available to common stockholders",
    )
    assert a1["release_rows"] == ["preferred"] and a1["preferred"]
    a2 = found[A2, "quarter"]
    assert (a2["group"], a2["row"]) == (
        COMMON,
        "Net income applicable to common shares",
    )
    assert a2["release_rows"] == ["participating"]
    prose = found[A2, "year_to_date"]
    assert prose["group"] == COMMON
    assert prose["row"].startswith(
        "GAAP net income attributable to common shareholders"
    )
    b1 = found[B1, "year_to_date"]
    assert (b1["group"], b1["verdict"], b1["base_verdict"]) == (
        QUARTER_AS_YTD,
        "wrong_period",
        "missing",
    )
    assert b1["row"].endswith("› Net income")  # where the year to date is printed
    summary = testreport.miss_summary(misses, broken=0)
    assert summary["common_outside_preferred"] == 2
    assert summary["outside_rows"] == {"TST": ["participating"]}


def test_a_copied_quarter_on_commons_line_goes_to_group_2():
    text = text_lines(
        "| | Three Months Ended June 30, 2026 | Six Months Ended June 30, 2026 |",
        "| --- | --- | --- |",
        "| Net income attributable to common stockholders | 60 | 100 |",
        "",
    )
    predictions = {B1: only_net_income(B1, D(60_000_000), D(60_000_000))}
    key = net_income_answers(B1, 60_000_000, 100_000_000)
    run, split = miss_world({B1: text}, predictions, key)
    [miss] = testreport.group_misses(run, run, split, {B1: text})
    assert miss["group"] == QUARTER_AS_YTD


def test_other_misses_are_grouped_by_verdict():
    text = text_lines(
        "| | Three Months Ended June 30, 2026 | Six Months Ended June 30, 2026 |",
        "| --- | --- | --- |",
        "| Net income | 60 | 100 |",
        "| Operating income | 70 | 120 |",
        "",
    )
    predictions = {B1: only_net_income(B1, D(70_000_000), D(100_000_000))}
    key = net_income_answers(B1, 60_000_000, 100_000_000)
    run, split = miss_world({B1: text}, predictions, key)
    [miss] = testreport.group_misses(run, run, split, {B1: text})
    assert (miss["group"], miss["verdict"]) == ("wrong_line", "wrong_line")


# Costs


def test_gpu_cost_at_the_price_billed_and_at_one_dollar(world: World):
    run = world.run("step-00025")
    run.info |= {"wall_time_s": 559.0, "releases": 148, "gpu_hourly_usd": 2.776}
    cost = testreport.cost_entry(run, 2.538)
    assert cost["cost"] == pytest.approx(2.538 * 559.0 / 3600)
    assert cost["per_1000"] == pytest.approx(2.538 * 559.0 / 3600 / 148 * 1000)
    assert round(cost["per_1000"], 2) == 2.66
    assert round(cost["per_1000_at_1"], 2) == 1.05
    assert round(cost["per_1000_at_recorded"], 2) == 2.91


def test_api_cost_is_run_jsons_estimate(world: World):
    run = world.run("step-00025")
    run.info |= {
        "cost_basis": "tokens",
        "estimated_cost_usd": 3.519332,
        "releases": 148,
        "price_per_million_tokens": {"input": 2.0, "output": 10.0},
    }
    cost = testreport.cost_entry(run, None)
    assert cost["basis"] == "tokens"
    assert round(cost["per_1000"], 2) == 23.78


def test_a_gpu_run_needs_a_billed_price(world: World):
    with pytest.raises(ReportError, match="without a billed price"):
        testreport.cost_entry(world.run("step-00025"), None)


def freeze_log(
    path: Path, start: datetime, gap: tuple[datetime, datetime], end: datetime
):
    lines, t = [], start
    while t <= end:
        if not gap[0] < t < gap[1]:
            stamp = t.strftime("%m-%d %H:%M:%S")
            lines.append(f"(APIServer) INFO {stamp} [loggers.py] Avg prompt throughput")
            lines.append('INFO:     127.0.0.1 - "POST /v1/chat/completions" 200 OK')
        t += timedelta(seconds=10)
    path.write_text("\n".join(lines) + "\n")


def test_the_freeze_excluded_estimate(world: World, tmp_path: Path):
    run, base = world.run("step-00025"), world.run("base")
    # The other 32 calls: median 7.5 s, mean 8.125 s.
    latencies = [5.0] * 16 + [10.0] * 15 + [30.0] + [135.0] * 16
    run.info |= {
        "started": "2026-11-07T11:20:00+00:00",
        "finished": "2026-11-07T11:30:00+00:00",
        "wall_time_s": 600.0,
        "releases": 48,
        "latency_s": {"p50": 10.0, "p95": 135.0, "max": 135.0},
        "calls": [{"latency_s": x} for x in latencies],
    }
    base.info |= {"wall_time_s": 21.0, "calls": [{"latency_s": 10.0}] * 32}
    start = datetime(2026, 11, 7, 11, 20, tzinfo=UTC)
    freeze = Freeze(
        SPLIT, "step-00025", start.replace(minute=22), start.replace(minute=29)
    )
    log = tmp_path / "server.log"
    freeze_log(
        log,
        start,
        (start.replace(minute=21, second=55), start.replace(minute=29, second=5)),
        start.replace(minute=30),
    )
    entry = testreport.freeze_entry(run, base, freeze, log, billed=3.6)
    assert entry["window_s"] == 420
    assert entry["excluded_s"] == 180
    assert entry["excluded_per_1000"] == pytest.approx(3.6 * 180 / 3600 / 48 * 1000)
    assert entry["monotonic_s"] == sum(latencies) / 16
    assert entry["base_overhead_s"] == 21.0 - 320 / 16
    assert entry["cross_check_s"] == (260 + 16 * 7.5) / 16 + 1.0
    assert entry["throughput_spacing_s"] == 10
    assert entry["p95_and_max_frozen"]
    run.info["latency_s"] = {"p50": 10.0, "p95": 30.0, "max": 135.0}
    entry = testreport.freeze_entry(run, base, freeze, log, billed=3.6)
    assert not entry["p95_and_max_frozen"]
    late = Freeze(
        SPLIT, "step-00025", start.replace(minute=22), start.replace(minute=30)
    )
    with pytest.raises(ReportError, match="no gap covering"):
        testreport.freeze_entry(run, base, late, log, billed=3.6)


def test_the_freeze_sentence_gives_the_window_to_the_second(
    world: World, tmp_path: Path
):
    # Seconds of 30 or more: a time rounded to the minute and then given its
    # seconds again would read a minute late (11:23:37 for 11:22:37).
    run, base = world.run("step-00025"), world.run("base")
    run.info |= {
        "started": "2026-11-07T11:20:00+00:00",
        "finished": "2026-11-07T11:31:00+00:00",
        "wall_time_s": 600.0,
        "releases": 48,
        "latency_s": {"p50": 10.0, "p95": 135.0, "max": 135.0},
        "calls": [{"latency_s": x} for x in [10.0] * 32 + [135.0] * 16],
    }
    base.info |= {"wall_time_s": 21.0, "calls": [{"latency_s": 10.0}] * 32}
    start = datetime(2026, 11, 7, 11, 20, tzinfo=UTC)
    freeze = Freeze(
        SPLIT,
        "step-00025",
        start.replace(minute=22, second=37),
        start.replace(minute=29, second=58),
    )
    log = tmp_path / "server.log"
    freeze_log(
        log,
        start,
        (start.replace(minute=22, second=31), start.replace(minute=29, second=59)),
        start.replace(minute=31),
    )
    entry = testreport.freeze_entry(run, base, freeze, log, billed=3.6)
    text = "\n".join(testreport.freeze_lines(entry, {"billed_usd_per_hour": 3.6}))
    assert "froze from about 11:22:37 to 11:29:58 UTC (441 s)" in text
    assert "no timestamped line between 11:22:30 and 11:30:00," in text
    assert "(the next longest 10.0 s), and the run's p95 and max latency are" in text
    # A p95 outside the frozen calls: the sentence doesn't claim it.
    run.info["latency_s"] = {"p50": 10.0, "p95": 30.0, "max": 135.0}
    entry = testreport.freeze_entry(run, base, freeze, log, billed=3.6)
    text = "\n".join(testreport.freeze_lines(entry, {"billed_usd_per_hour": 3.6}))
    assert "(the next longest 10.0 s). As in training's" in text
    assert "p95" not in text


# Config and formatting


def test_the_tracked_config_loads():
    config = testreport.load_config(testreport.DEFAULT_CONFIG)
    assert [m.key for m in config.models][:2] == ["step-00025", "base"]
    assert config.references == [
        "base",
        "gpt-6.1-sol",
        "claude-haiku-5-5",
        "gpt-6-luna",
    ]
    assert config.billed == {"validation": 2.776, SPLIT: 2.538, "time_test": 2.589}
    assert config.freeze is not None and config.freeze.seconds == 441
    assert [a.split for a in config.attempts] == ["time_test", "validation"]


def test_numbers_are_shown_with_a_minus_sign_and_no_negative_zero():
    assert testreport.num(-0.76) == "−0.76"
    assert testreport.num(10.12, signed=True) == "+10.12"
    assert testreport.num(-0.001, signed=True) == "0.00"
    assert testreport.bounds([-1.95, 0.0], signed=True) == "[−1.95, 0.00]"
    assert testreport.utc_time("2026-11-07T11:30:59+00:00") == "11:31"
