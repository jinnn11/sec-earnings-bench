"""secbench.servereport on fake outputs: servebench's files from the fake vLLM
(test_servebench's lab), graded on the laptop with the split's files, plus a stage 2
reference run and a setup log; and its arithmetic on handwritten numbers."""

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
from test_score import key_dir, make_client, perfect  # noqa: F401 (fixtures)
from test_servebench import CONFIG, StreamingVLLM, bench, lab  # noqa: F401
from test_serving import write_serving_config
from test_splits import AAPL, MSFT, repo  # noqa: F401

from secbench import score, servereport

SETUP_LOG = """\
== 2026-11-10T12:00:00Z GPU, driver and disk
NVIDIA H100 80GB HBM3, 81559 MiB, 580.178.04
== 2026-11-10T12:01:00Z Unpack gpu-bundle-x-serving.tar
== 2026-11-10T12:02:30Z Install uv 0.13.0, Python and the train and serve groups
== 2026-11-10T12:09:30Z Download the base model at training/run's revision
== 2026-11-10T12:13:00Z Serving benchmark (secbench.servebench): every variant of the config
== 2026-11-10T14:40:00Z Pack the results (exit status 0)
"""


def wrong_eps(reply: str) -> str:
    """The reply with the quarter's diluted EPS a cent higher."""
    data = json.loads(reply)
    data["quarter"]["eps_diluted"] = round(data["quarter"]["eps_diluted"] + 0.01, 2)
    return json.dumps(data)


@pytest.fixture
def served(lab, perfect):  # noqa: F811
    """servebench's outputs for adapter and merged; merged gets AAPL's quarter EPS
    wrong. A stage 2 validation reference (a copy of adapter's run) and a setup
    log."""
    write_serving_config(
        lab / CONFIG,
        load__split='"validation"',
        load__prompts="1",
        load__warmup="1",
        accuracy__splits='["validation"]',
        calibration__examples="2",
        report__validation_reference='"runs/stage2-validation"',
    )
    aapl = next(t for t in perfect if "Apple" in t)

    def reply(model: str, text: str) -> str:
        if model.endswith("--merged") and text == aapl:
            return wrong_eps(perfect[text])
        return perfect[text]

    assert bench(lab, StreamingVLLM(reply), "--variants", "adapter", "merged") == 0
    reference = lab / "runs/stage2-validation"
    shutil.copytree(lab / "runs/run--serving--adapter--validation", reference)
    files = score.load_split("validation", Path("data/splits"))
    assert servereport.grade_run(reference, files, make_client(lab / "raw")) == 0
    (lab / "training/run/serving/setup.log").write_text(SETUP_LOG)
    return lab


def report(root: Path, *args: str) -> int:
    return servereport.main(
        ["--config", str(CONFIG), "--out-dir", "results", *args],
        client=make_client(root / "raw"),
    )


def test_the_report(served, capsys) -> None:
    assert report(served) == 0
    out = capsys.readouterr().out
    results = json.loads((served / "results/serving_results.json").read_text())
    md = (served / "results/serving_results.md").read_text()
    assert md in out
    assert results["session"]["gpu_hourly_usd"] == 2.5
    adapter, merged = results["variants"]["adapter"], results["variants"]["merged"]
    assert (adapter["status"], merged["status"]) == ("complete", "complete")

    # Graded on the laptop: graded.csv in each run.
    for variant in ("adapter", "merged"):
        run = served / f"runs/run--serving--{variant}--validation"
        assert (run / "graded.csv").is_file() and (run / "summary.txt").is_file()
    assert adapter["accuracy"]["validation"]["correct"] == 11
    assert adapter["accuracy"]["validation"]["graded"] == 11
    assert merged["accuracy"]["validation"]["correct"] == 10
    assert results["reference_accuracy"]["validation"]["correct"] == 11

    # Agreement: merged differs from adapter in one figure of AAPL's.
    found = merged["agreement"]["validation"]["with_adapter"]
    assert (found["figures"], found["same"]) == (12, 11)
    assert (found["releases"], found["releases_identical"]) == (2, 1)
    [difference] = found["differences"]
    assert difference["release_accession"] == AAPL
    assert difference["figure"] == "quarter.eps_diluted"
    assert Decimal(difference["run"]) - Decimal(difference["reference"]) == Decimal(
        "0.01"
    )
    assert adapter["agreement"]["validation"]["with_adapter"] is None
    stage2 = adapter["agreement"]["validation"]["with_stage2"]
    assert (stage2["same"], stage2["figures"]) == (12, 12)
    assert (
        "Merge check: merged's predictions equal adapter's in 11 of 12 figures "
        "(1 of 2 releases identical)." in md
    )

    # The sweep: costs from each level's makespan.
    for v in (adapter, merged):
        assert [x["level"] for x in v["levels"]] == [1, 2]
        for x in v["levels"]:
            assert x["cost_per_1000_usd"] == pytest.approx(
                2.5 * x["makespan_s"] / 3600 / x["ok"] * 1000
            )
            assert x["cost_per_1000_at_1_usd"] == pytest.approx(
                x["cost_per_1000_usd"] / 2.5
            )
        best = max(v["levels"], key=lambda x: (x["releases_per_hour"], -x["level"]))
        assert v["best"]["level"] == best["level"]
    assert adapter["startup"]["weights_gib"] == 15.2683
    assert "| adapter | 15.27 | 52.34 | 381,120 | 9.30x |" in md
    assert "| merged | 1 | 1/1 |" in md

    # The setup log's steps.
    steps = results["overhead"]["setup_steps"]
    assert [s["minutes"] for s in steps] == [1.0, 1.5, 7.0, 3.5, 147.0, None]
    assert steps[2]["step"].startswith("Install uv")
    # The fake builds' manifests record no times.
    assert results["overhead"]["model_builds_s"] == {"merged": None}
    assert "| Install uv 0.13.0, Python and the train and serve groups | 7.0 |" in md


def test_a_failed_variant_is_shown(served, capsys) -> None:
    record_path = served / "training/run/serving/servebench.json"
    record = json.loads(record_path.read_text())
    record["variants"]["fp8"] = {
        "status": "failed",
        "error": "ServingError: the server exited (1) before it was ready\nmore",
    }
    record_path.write_text(json.dumps(record))
    assert report(served) == 0
    md = (served / "results/serving_results.md").read_text()
    assert (
        "| fp8 | failed | run--step-00002--fp8 | merged | ServingError: the server "
        in md
    )
    assert "| fp8 | – | – | – | – | – |" in md  # no best throughput


def test_without_the_results(lab, capsys) -> None:  # noqa: F811
    assert report(lab) == 1
    assert "unpack results-run-serving.tar in the repository first" in (
        capsys.readouterr().err
    )


def test_cost_and_best_level() -> None:
    assert servereport.cost_per_1000(2.5, 3600.0, 1000) == 2.5
    assert servereport.cost_per_1000(2.0, 90.0, 120) == pytest.approx(
        2.0 * 90 / 3600 / 120 * 1000
    )
    assert servereport.cost_per_1000(2.0, None, 10) is None
    assert servereport.cost_per_1000(2.0, 10.0, 0) is None
    levels = [
        {"level": 1, "releases_per_hour": 100.0},
        {"level": 8, "releases_per_hour": 900.0},
        {"level": 16, "releases_per_hour": 900.0},  # a tie: the lower level
        {"level": 32, "releases_per_hour": None},
    ]
    assert servereport.best_level(levels)["level"] == 8
    assert servereport.best_level([{"level": 1, "releases_per_hour": None}]) is None


def test_agreement_counts_nulls_and_period_ends(tmp_path) -> None:
    def write(folder: Path, rows: list[dict]) -> Path:
        folder.mkdir()
        (folder / "predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        return folder

    def row(accession, revenue, eps, period_end="2026-06-30"):
        group = {"revenue": revenue, "net_income": None, "eps_diluted": eps}
        return {
            "release_accession": accession,
            "period_end": period_end,
            "quarter": group,
            "year_to_date": dict(group),
        }

    a = write(
        tmp_path / "a",
        [row("r1", "100", "1.50"), row("r2", None, "2.00"), row("r3", "5", "1")],
    )
    b = write(
        tmp_path / "b",
        [row("r1", "100", "1.5"), row("r2", "7", "2.00", "2026-07-01")],
    )
    found = servereport.agreement(a, b)
    assert found["releases"] == 2  # r3 only in a
    assert found["figures"] == 12
    assert found["same"] == 10  # r2's revenue differs in both groups; 1.50 == 1.5
    assert found["releases_identical"] == 1
    assert found["period_end_differs"] == 1
    assert {d["figure"] for d in found["differences"]} == {
        "quarter.revenue",
        "year_to_date.revenue",
    }
    assert servereport.agreement(a, tmp_path / "none") is None
