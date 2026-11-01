import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pytest
from test_score import RELEASES, key_dir  # noqa: F401 (key_dir is a fixture)
from test_train import write_labels

from secbench import bundle
from secbench.bundle import BundleError

ARGS = [
    Path("data/stage2/labels"),
    Path("data/links.csv"),
    Path("data/answers.csv"),
    Path("data/releases"),
    Path("data/raw"),
]


@pytest.fixture
def repo(tmp_path, key_dir, monkeypatch) -> Path:  # noqa: F811
    """A folder laid out as the repository's data: labels, the five validation
    fixtures and the EDGAR cache answers.py filled for them."""
    root = tmp_path / "repo"
    data = root / "data"
    write_labels(data / "stage2" / "labels")
    for name in ("links.csv", "answers.csv"):
        shutil.copy(key_dir / name, data / name)
    shutil.copytree(RELEASES, data / "releases")
    shutil.copytree(key_dir / "raw", data / "raw")
    (data / "raw" / "unrelated.json").write_text("{}")
    monkeypatch.chdir(root)
    return root


def test_bundle_holds_what_the_instance_needs(repo, tmp_path) -> None:
    files = bundle.data_files(*ARGS)
    labels = [
        Path("data/stage2/labels") / n
        for n in ("train.jsonl", "examples.csv", "manifest.json")
    ]
    assert files[:5] == [*labels, Path("data/links.csv"), Path("data/answers.csv")]
    cached = [f for f in files if f.parts[:2] == ("data", "raw")]
    # XOM's facts are under both its CIKs (edgar.TICKER_CIKS); the grader reads both.
    assert [f.name for f in cached] == [
        "CIK0000018230.json",
        "CIK0000034088.json",
        "CIK0000104169.json",
        "CIK0000320193.json",
        "CIK0000789019.json",
        "CIK0002115436.json",
        "company_tickers.json",
    ]
    releases = [f for f in files if f.parts[:2] == ("data", "releases")]
    assert releases and all(f.is_file() for f in releases)

    out = tmp_path / "bundle.tar"
    branch = bundle.git("rev-parse", "--abbrev-ref", "HEAD")
    info = bundle.make_bundle(out, files, "abc123", branch)
    assert (info["commit"], info["branch"], info["files"]) == (
        "abc123",
        branch,
        len(files),
    )
    unpacked = tmp_path / "unpacked"
    with tarfile.open(out) as tar:
        tar.extractall(unpacked, filter="data")
    assert json.loads((unpacked / "BUNDLE.json").read_text()) == info
    sums = (unpacked / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == len(files) + 1 and sums[0].endswith("  repo.bundle")
    for line in sums:
        digest, name = line.split("  ")
        assert hashlib.sha256((unpacked / name).read_bytes()).hexdigest() == digest
    assert not (unpacked / "data" / "raw" / "unrelated.json").exists()
    bundle.check_unpacked(unpacked, *ARGS[1:4])


def test_bundle_refuses_a_missing_cache_file(repo) -> None:
    facts = Path("data/raw/data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json")
    facts.unlink()
    with pytest.raises(BundleError, match="the grader's files: not in the cache"):
        bundle.data_files(*ARGS)


def test_bundle_refuses_absolute_paths(repo) -> None:
    args = [ARGS[0].resolve(), *ARGS[1:]]
    with pytest.raises(BundleError, match="paths must be relative to the repository"):
        bundle.data_files(*args)
