"""Build everything the project's Hugging Face repositories need, by code, into
dist/hf/ (gitignored), so it can be rebuilt at any time. Nothing is uploaded here:
scripts/hf_publish.sh does that later, after a check (--verify).

Usage: uv run python -m secbench.hfexport --hf-user NAME [--out dist/hf]
       uv run python -m secbench.hfexport --verify [--out dist/hf]

NAME is the Hugging Face user or organization the repositories go under; local builds
use the placeholder HF_USER, which hf_publish.sh refuses. Folders, one per repository:

- model/: <NAME>/qwen3-8b-sec-earnings-lora, step-00025's LoRA adapter (its sha256
  checked against configs/test/qwen3-8b-2k.toml) and its model card
  (secbench.hfcards);
- int4-awq/: the card of <NAME>/qwen3-8b-sec-earnings-int4-awq, whose model files were
  uploaded from the GPU instance; built only when results/serving_results.json exists,
  and skipped (said so) until then;
- dataset/: <NAME>/sec-earnings-bench, the benchmark as JSONL configs (releases, answers,
  train, predictions) with its card (secbench.hfdataset);
- space/: <NAME>/sec-earnings-explorer, the results explorer (a Gradio app on the free CPU
  tier that reads precomputed files only; secbench.hfspace, app source in space/).

Every number comes from a results file (see secbench.hfcommon). The build goes into
<out>.part and replaces <out> only when complete: manifest.json lists every file with
its size and sha256, and the privacy scan (secbench.hfscan) must find nothing (home or
instance paths, keys, tokens, account IDs, the git author's or any other e-mail address
outside the release texts). On a finding the build stops, leaving <out>.part to
inspect and <out> as it was. Deterministic: the same inputs at the same commit give
byte-identical files (manifest.json records the commit and whether tracked files had
uncommitted changes).

--verify checks an existing build against its manifest (no file missing, extra or
changed) and scans it again; exit 0 when both pass.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from secbench import hfcards, hfcommon, hfdataset, hfscan, hfspace
from secbench.hfcommon import (
    DATASET,
    INT4,
    MODEL,
    REPO_TYPES,
    SPACE,
    Context,
    ExportError,
    Sources,
    Written,
)
from secbench.stage2 import REPO

DEFAULT_OUT = Path("dist/hf")
MANIFEST = "manifest.json"
# The folders in build order, each with its builder.
BUILDERS: dict[str, Callable[[Context, Path], Written]] = {
    MODEL: hfcards.build_model,
    INT4: hfcards.build_int4,
    DATASET: hfdataset.build_dataset,
    SPACE: hfspace.build_space,
}


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def file_entries(root: Path) -> list[dict[str, Any]]:
    """Every file under root but the manifest: path (POSIX, relative), size, sha256."""
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST:
            continue
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": hfcommon.sha256_file(path),
            }
        )
    return entries


def repos_entry(hf_user: str, built: Sequence[str]) -> dict[str, dict[str, Any]]:
    repos = {}
    for kind in BUILDERS:
        if kind not in built:
            continue
        entry: dict[str, Any] = {
            "id": hfcommon.repo_id(hf_user, kind),
            "type": REPO_TYPES[kind],
            "folder": kind,
            "private": True,
        }
        if kind == SPACE:
            entry["sdk"] = "gradio"
        if kind == INT4:
            entry["card_only"] = True
        repos[kind] = entry
    return repos


def manifest(
    hf_user: str, root: Path, built: Sequence[str], skipped: Sequence[str]
) -> dict[str, Any]:
    return {
        "hf_user": hf_user,
        "build": hfcommon.build_info(),
        "repos": repos_entry(hf_user, built),
        "skipped": list(skipped),
        "files": file_entries(root),
    }


def scan(root: Path) -> list[hfscan.Finding]:
    """The privacy scan, with this repository's git author e-mails and the home
    directory."""
    return hfscan.scan(root, emails=hfscan.known_emails(REPO), home=str(Path.home()))


def build(
    sources: Sources, out: Path, hf_user: str, *, say: Callable[[str], None] = print
) -> dict[str, Any]:
    """Builds every folder into <out>.part, writes the manifest, scans it, then
    replaces out. Returns the manifest. ExportError on any problem (out unchanged)."""
    context = hfcommon.load_context(sources, hf_user)
    part = out.with_name(out.name + ".part")
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True)
    built, skipped = [], []
    for kind, builder in BUILDERS.items():
        written = builder(context, part / kind)
        skipped += written.skipped
        if written.files:
            built.append(kind)
            say(f"built {kind}/: {len(written.files)} files")
        for note in written.skipped:
            say(f"skipped {note}")
    record = manifest(context.hf_user, part, built, skipped)
    hfcommon.write_json(part / MANIFEST, record)
    findings = scan(part)
    if findings:
        for finding in findings:
            say(f"privacy: {format_finding(finding)}")
        raise ExportError(
            f"the privacy scan found {len(findings)} problem(s) in {part}; nothing "
            f"replaced {out}"
        )
    if out.exists():
        shutil.rmtree(out)
    part.replace(out)
    return record


def format_finding(finding: hfscan.Finding) -> str:
    return f"{finding.path} ({finding.where}): {finding.kind}: {finding.excerpt}"


def summary(record: dict[str, Any], out: Path) -> str:
    files = record["files"]
    build = record["build"]
    state = "uncommitted changes" if build["dirty"] else "clean"
    lines = [
        f"built {out} for {record['hf_user']} (commit {build['commit'][:7]}, {state}):"
    ]
    for kind, repo in record["repos"].items():
        mine = [f for f in files if f["path"].startswith(f"{kind}/")]
        size = sum(f["size"] for f in mine)
        lines.append(
            f"  {kind + '/':<10} {len(mine):>5} files {human_size(size):>12}  -> "
            f"{repo['id']} ({repo['type']})"
        )
    total = sum(f["size"] for f in files)
    lines.append(f"  {'total':<10} {len(files):>5} files {human_size(total):>12}")
    for note in record["skipped"]:
        lines.append(f"skipped: {note}")
    lines.append(f"privacy scan: no findings in {len(files) + 1} files")
    return "\n".join(lines)


def verify(out: Path) -> list[str]:
    """Problems with a build: files missing, extra or changed since its manifest, and
    the privacy scan's findings."""
    path = out / MANIFEST
    if not path.is_file():
        return [f"{path} not found; build first"]
    record = hfcommon.read_json(path)
    listed = {f["path"]: f for f in record["files"]}
    found = {f["path"]: f for f in file_entries(out)}
    problems = [f"missing: {p}" for p in sorted(set(listed) - set(found))]
    problems += [f"not in the manifest: {p}" for p in sorted(set(found) - set(listed))]
    for name in sorted(set(listed) & set(found)):
        if (listed[name]["size"], listed[name]["sha256"]) != (
            found[name]["size"],
            found[name]["sha256"],
        ):
            problems.append(f"changed since the build: {name}")
    for kind, repo in record["repos"].items():
        if not (out / repo["folder"]).is_dir():
            problems.append(f"{kind}: folder {repo['folder']}/ not found")
    problems += [f"privacy: {format_finding(f)}" for f in scan(out)]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.hfexport",
        description="Build the Hugging Face model card, dataset and results explorer "
        "into dist/hf/ (nothing is uploaded).",
    )
    parser.add_argument(
        "--hf-user",
        help=f"the Hugging Face user or organization ({hfcommon.PLACEHOLDER_USER} for "
        "a local build)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check an existing build against its manifest and scan it again",
    )
    args = parser.parse_args(argv)
    if args.verify:
        try:
            problems = verify(args.out)
        except (hfscan.ScanError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for problem in problems:
            print(problem)
        if problems:
            print(f"error: {args.out} failed verification", file=sys.stderr)
            return 1
        print(f"{args.out}: every file matches its manifest; privacy scan clean")
        return 0
    if not args.hf_user:
        parser.error("--hf-user is required to build")
    try:
        hfcommon.check_user(args.hf_user)
        record = build(Sources(), args.out, args.hf_user)
    except (ExportError, hfscan.ScanError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(summary(record, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
