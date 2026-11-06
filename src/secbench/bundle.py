"""Pack what a GPU instance needs to train and score a stage 2 run, or to make one
checkpoint's predictions on a frozen test split; scripts/vast_setup.sh unpacks it
there. The instance makes no SEC request.

Usage: uv run python -m secbench.bundle [--out gpu-bundle-<commit>.tar]
       uv run python -m secbench.bundle --test-config configs/test/<run>.toml
           --split company_test [--out gpu-bundle-<commit>-<split>.tar]

The tar holds:
- repo.bundle, a git bundle of HEAD's branch (uncommitted changes to tracked files
  are refused);
- data/stage2/labels/train.jsonl, examples.csv and manifest.json: the training
  examples;
- data/links.csv, data/answers.csv and data/releases/: the validation releases, their
  answer key and their inputs;
- the cached SEC files the grader reads for them (SEC's ticker map and the company
  facts of each CIK), found by running the grader's fact loading on the validation
  releases with a client that records every URL it reads;
- SHA256SUMS (sha256sum -c) and BUNDLE.json (commit, branch, created, files).

Before it is written, the bundle is checked from a temporary unpacked copy: the
grader's fact loading with a client that reads only that copy's cache, and every
validation input built from that copy's releases.

With --test-config and --split (secbench.score's test mode), the tar holds
repo.bundle and only: the test config, training/<run>/config.json, the config's
checkpoint folder (not training_state.pt, the optimizer's), data/splits/<split>/
and that split's release files (each record and its saved exhibits). No SEC cache:
the instance doesn't grade. The test config, the adapter's sha256 and the split
(frozen, its files as its manifest records) are checked as score checks them, and
every input of the split is built from the unpacked copy and checked against
tokens.csv. BUNDLE.json's "test" entry records the test config and its sha256, the
checkpoint and the adapter's sha256, the split and its files' sha256s, and the
commit the split files were built at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from secbench import extract, grade, labels, score, splits, train
from secbench.answers import DEFAULT_OUT as DEFAULT_ANSWERS
from secbench.answers import load_links
from secbench.edgar import DEFAULT_CACHE_DIR, EdgarClient, EdgarError, cache_path
from secbench.inputs import build_input, release_records
from secbench.labels import DEFAULT_OUT as DEFAULT_LABELS
from secbench.labels import EXAMPLE_INDEX, EXAMPLES, MANIFEST
from secbench.link import DEFAULT_OUT as DEFAULT_LINKS
from secbench.releases import DEFAULT_OUT_DIR as DEFAULT_RELEASES
from secbench.stage2 import REPO, current_commit

REPO_BUNDLE = "repo.bundle"
SUMS = "SHA256SUMS"
INFO = "BUNDLE.json"
OFFLINE_AGENT = "offline: no requests"


class BundleError(Exception):
    """The bundle can't be made; the message says why."""


class RecordingClient(EdgarClient):
    """An offline client that records the cache file of every URL it reads."""

    def __init__(self, cache_dir: Path) -> None:
        super().__init__(OFFLINE_AGENT, cache_dir, offline=True)
        self.read: set[Path] = set()

    def fetch(self, url: str, *, refresh: bool = False) -> bytes:
        body = super().fetch(url, refresh=refresh)
        self.read.add(cache_path(url, self.cache_dir))
        return body


def grader_files(links: Path, answers: Path, client: EdgarClient) -> None:
    """Load the company facts the grader reads to grade every linked release of
    links (raises EdgarError if one isn't in the client's cache)."""
    releases = load_links(links)[0]
    accessions = {r.release_accession for r in releases}
    key = grade.load_answers(answers)
    grade.load_filing_facts(
        client, releases, [a for a in key.values() if a.release_accession in accessions]
    )


def data_files(
    labels_dir: Path, links: Path, answers: Path, releases: Path, cache: Path
) -> list[Path]:
    """Every data file the instance needs, as paths relative to the repository."""
    files = [labels_dir / EXAMPLES, labels_dir / EXAMPLE_INDEX, labels_dir / MANIFEST]
    files += [links, answers]
    files += sorted(p for p in releases.rglob("*") if p.is_file())
    recorder = RecordingClient(cache)
    try:
        grader_files(links, answers, recorder)
    except EdgarError as exc:
        raise BundleError(f"the grader's files: {exc}") from None
    files += sorted(recorder.read)
    check_relative(files)
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_unpacked(root: Path, links: Path, answers: Path, releases: Path) -> None:
    """Grade's fact loading and every validation input from the unpacked copy alone."""
    client = EdgarClient(OFFLINE_AGENT, root / DEFAULT_CACHE_DIR, offline=True)
    try:
        grader_files(root / links, root / answers, client)
    except EdgarError as exc:
        raise BundleError(f"unpacked bundle: {exc}") from None
    records = release_records(root / releases)
    for release in load_links(root / links)[0]:
        record = records.get(release.release_accession)
        if record is None:
            raise BundleError(f"unpacked bundle: no {release.release_accession}")
        build_input(record)


def release_files(links: Path, releases: Path) -> list[Path]:
    """Each linked release's saved record and the exhibit files it lists."""
    records = release_records(releases)
    files = []
    for release in load_links(links)[0]:
        record = records.get(release.release_accession)
        if record is None:
            raise BundleError(f"no saved release {release.release_accession}")
        files.append(record)
        exhibits = json.loads(record.read_text(encoding="utf-8"))["exhibits"]
        files += [record.parent / e["file"] for e in exhibits if e["file"]]
    return files


def check_relative(files: Iterable[Path]) -> None:
    files = list(files)
    absolute = [str(p) for p in files if p.is_absolute()]
    if absolute:
        raise BundleError(
            f"paths must be relative to the repository: {', '.join(absolute[:3])}"
        )
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise BundleError(f"not found: {', '.join(missing[:5])}")


def files_for_split(
    test_config: Path, split: str, splits_dir: Path = splits.DEFAULT_OUT
) -> tuple[list[Path], dict[str, Any]]:
    """The files a test split's instance needs besides the code, and BUNDLE.json's
    "test" entry."""
    test = score.load_test_config(test_config)
    run_dir = train.TRAINING_DIR / test.run
    _, config, _ = train.read_run(run_dir)
    problems = score.settings_problems(test, run_dir, config)
    if problems:
        raise BundleError(f"{test_config}: " + "; ".join(problems))
    score.checkpoint_under_test(run_dir, test, config, split)  # sha256 and r
    found = score.load_split(split, splits_dir)  # frozen; files as recorded
    checkpoint = run_dir / train.CHECKPOINTS / test.checkpoint
    split_files = [found.folder / n for n in (*splits.FILES, splits.MANIFEST)]
    releases = release_files(found.links, found.releases)
    files = [
        test_config,
        run_dir / train.CONFIG,
        *sorted(
            p
            for p in checkpoint.iterdir()
            if p.is_file() and p.name != train.TRAINING_STATE
        ),
        *split_files,
        *releases,
    ]
    check_relative(files)
    entry = {
        "config": str(test_config),
        "config_sha256": test.sha256,
        "config_tracked": splits.is_tracked(test_config),
        "run": test.run,
        "checkpoint": test.checkpoint,
        "adapter_sha256": test.adapter_sha256,
        "split": split,
        "split_built_at": found.manifest["commit"],
        "split_files": {str(p): sha256(p) for p in split_files},
        "releases": len(found.inputs),
        "release_files": len(releases),
    }
    return files, entry


def check_test_unpacked(root: Path, split: str, splits_dir: Path) -> None:
    """The split's files and every input of it, from the unpacked copy alone."""
    try:
        found = score.load_split(split, splits_dir, root)
        prompt = extract.load_prompt(labels.PROMPT)
        score.split_prompts(found, prompt.text)
    except score.ScoreError as exc:
        raise BundleError(f"unpacked bundle: {exc}") from None


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def make_bundle(
    out: Path,
    files: Iterable[Path],
    commit: str,
    branch: str,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Write the tar; BUNDLE.json's contents (with extra) are returned."""
    files = list(files)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        git("bundle", "create", str(staging / REPO_BUNDLE), branch)
        sums = [f"{sha256(staging / REPO_BUNDLE)}  {REPO_BUNDLE}"]
        sums += [f"{sha256(path)}  {path.as_posix()}" for path in files]
        (staging / SUMS).write_text("\n".join(sums) + "\n")
        info = {
            "commit": commit,
            "branch": branch,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            **(extra or {}),
        }
        (staging / INFO).write_text(json.dumps(info, indent=2) + "\n")
        partial = out.with_name(out.name + ".part")
        with tarfile.open(partial, "w") as tar:
            for name in (INFO, SUMS, REPO_BUNDLE):
                tar.add(staging / name, arcname=name)
            for path in files:
                tar.add(path, arcname=path.as_posix())
        partial.replace(out)
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.bundle",
        description="Pack the repository and the data a GPU instance needs.",
    )
    parser.add_argument("--out", type=Path, help="default: gpu-bundle-<commit>.tar")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--releases", type=Path, default=DEFAULT_RELEASES)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--test-config",
        type=Path,
        help="pack for secbench.score's test mode (configs/test/<run>.toml); give "
        "--split too",
    )
    parser.add_argument("--split", choices=splits.SPLITS, help="a frozen split")
    args = parser.parse_args(argv)
    if (args.test_config is None) != (args.split is None):
        parser.error("--test-config and --split go together")
    if args.test_config is not None:
        return main_test(args)
    try:
        commit, dirty = current_commit()
        if dirty:
            raise BundleError(
                "tracked files have uncommitted changes; commit them first"
            )
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        if branch == "HEAD":
            raise BundleError("HEAD is detached; check out a branch")
        files = data_files(
            args.labels, args.links, args.answers, args.releases, args.cache
        )
        with tempfile.TemporaryDirectory() as tmp:
            symlinked_copy(Path(tmp), files)
            check_unpacked(Path(tmp), args.links, args.answers, args.releases)
        out = args.out or Path(f"gpu-bundle-{commit[:12]}.tar")
        info = make_bundle(out, files, commit, branch)
    except (BundleError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"wrote {out}: {branch} at {commit[:12]}, {info['files']:,} data files "
        f"({info['bytes'] / 2**20:,.0f} MiB); copy it and scripts/vast_setup.sh to "
        "the instance"
    )
    return 0


def symlinked_copy(root: Path, files: Iterable[Path]) -> None:
    for path in files:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).symlink_to(path.resolve())


def main_test(args: argparse.Namespace) -> int:
    """The test mode of main."""
    try:
        commit, dirty = current_commit()
        if dirty:
            raise BundleError(
                "tracked files have uncommitted changes; commit them first"
            )
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        if branch == "HEAD":
            raise BundleError("HEAD is detached; check out a branch")
        files, entry = files_for_split(args.test_config, args.split)
        with tempfile.TemporaryDirectory() as tmp:
            symlinked_copy(Path(tmp), files)
            check_test_unpacked(Path(tmp), args.split, splits.DEFAULT_OUT)
        out = args.out or Path(f"gpu-bundle-{commit[:12]}-{args.split}.tar")
        info = make_bundle(out, files, commit, branch, {"test": entry})
    except (
        BundleError,
        score.ScoreError,
        train.TrainError,
        splits.SplitError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"wrote {out}: {branch} at {commit[:12]}, {info['files']:,} files "
        f"({info['bytes'] / 2**20:,.0f} MiB): {entry['run']} {entry['checkpoint']} "
        f"on {entry['split']} ({entry['releases']} releases)"
        + ("" if entry["config_tracked"] else f"; {args.test_config} isn't tracked")
        + "; copy it and scripts/vast_setup.sh to the instance and run there: bash "
        f"vast_setup.sh {out.name} --split {args.split} --gpu-hourly <price>"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
