"""The files each scored split is graded on (PLAN.md Decisions 2026-10-27 and
2026-11-01): data/splits/<split>/ for validation, company_test and time_test.

A split's releases are its linked releases: validation's are data/links.csv's (stage
1: data/links.csv, data/answers.csv, data/releases), the test splits' those of
<stage2>/releases.csv's split column (stage 2: <stage2>/links.csv, answers.csv,
releases/). A frozen split is instead the releases listed for it in the tracked file
data/frozen_test_releases.csv, whatever releases.csv says, each with the sha256 of
its links row, its answers rows and its input text, which must be unchanged (any
difference stops the build). Why: company_test is each company's latest 4 linked
releases, so a later stage 2 build with newer releases would otherwise swap test
releases. --freeze SPLIT adds a split to that list; commit the list, then run again
to build the split as frozen. A split not in the list is provisional, and
secbench.score scores only frozen ones.

Context limit (Decisions 2026-11-01): a release whose prompt is over
chat.MAX_MODEL_LEN - chat.MAX_TOKENS (39,936) tokens is left out of its split for
every model and listed. The prompt is the extract_v2 system prompt and the release's
input (secbench.inputs) in chat.prompt_text's layout (Qwen3, thinking off, generation
prompt included), counted with Qwen3-8B's tokenizer at labels.QWEN3_REVISION
(labels.load_tokenizer). The releases left out must be exactly EXPECTED_LEFT_OUT, the
Decision's list with its token counts, or nothing is written and the command exits 1.

Each data/splits/<split>/ holds links.csv and answers.csv (their sources' rows, in
the same format, for the kept releases), tokens.csv (every release of the split: its
prompt tokens, kept or left_out, and its input's sha256), left_out.csv, preferred.csv
(the releases whose linked filing reports nonzero preferred dividends: releases.csv's
preferred_dividends), and manifest.json (counts, the longest kept prompt, the
tokenizer, the prompt's sha256, the sha256 of every input and output file, the
releases directory, the commit, and whether the split is frozen or provisional).
Deterministic: the same inputs at the same commit give the same bytes. Refuses to run
while tracked files have uncommitted changes, or while the frozen list isn't
committed.

Usage: uv run python -m secbench.splits [--freeze SPLIT ...] [--out data/splits]
       [--stage2 data/stage2] [--links data/links.csv] [--answers data/answers.csv]
       [--releases data/releases] [--frozen data/frozen_test_releases.csv]
       [--raw data/raw] [--prompts-dir prompts]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from secbench import chat, extract, grade, labels
from secbench.answers import COLUMNS as ANSWER_COLUMNS
from secbench.answers import DEFAULT_OUT as DEFAULT_ANSWERS
from secbench.companies import COMPANY_TEST, VALIDATION
from secbench.inputs import build_input, release_records
from secbench.link import COLUMNS as LINK_COLUMNS
from secbench.link import DEFAULT_OUT as DEFAULT_LINKS
from secbench.link import LINKED
from secbench.releases import DEFAULT_OUT_DIR as DEFAULT_RELEASES
from secbench.releases import write_text
from secbench.stage2 import DEFAULT_OUT as DEFAULT_STAGE2
from secbench.stage2 import RELEASE_COLUMNS, REPO, TIME_TEST, current_commit

SPLITS = (VALIDATION, COMPANY_TEST, TIME_TEST)
DEFAULT_OUT = Path("data/splits")
FROZEN_LIST = Path("data/frozen_test_releases.csv")
MAX_PROMPT_TOKENS = chat.MAX_MODEL_LEN - chat.MAX_TOKENS  # 39,936
KEPT, LEFT_OUT = "kept", "left_out"
FROZEN, PROVISIONAL = "frozen", "provisional"

# Decisions 2026-11-01: the releases over the limit, with their prompt tokens. The
# rule applies to releases added later (the time test's Q3 2026 releases); a release
# it leaves out must be added here, so the list is never changed silently.
EXPECTED_LEFT_OUT: dict[str, dict[str, int]] = {
    VALIDATION: {},
    COMPANY_TEST: {},
    TIME_TEST: {
        "0001037540-26-000031": 106_137,  # BXP
        "0001104659-26-093361": 104_334,  # SPG
        "0001035443-26-000067": 87_822,  # ARE
        "0001104659-26-086270": 70_776,  # DLR
        "0001534701-26-000030": 52_257,  # PSX
        "0000074208-26-000070": 50_970,  # UDR
        "0000831259-26-000033": 50_171,  # FCX
        "0001164727-26-000034": 46_825,  # NEM
        "0001687229-26-000041": 44_808,  # INVH
    },
}

LINKS = "links.csv"
ANSWERS = "answers.csv"
TOKENS = "tokens.csv"
LEFT_OUT_FILE = "left_out.csv"
PREFERRED = "preferred.csv"
MANIFEST = "manifest.json"
FILES = (LINKS, ANSWERS, TOKENS, LEFT_OUT_FILE, PREFERRED)  # besides the manifest

FROZEN_COLUMNS = [
    "split",
    "ticker",
    "release_accession",
    "linked_accession",
    "preferred_dividends",
    "links_sha256",  # its links.csv row
    "answers_sha256",  # its answers.csv rows, sorted
    "input_sha256",  # its input text (secbench.inputs)
]
TOKEN_COLUMNS = [
    "ticker",
    "release_accession",
    "prompt_tokens",
    "status",
    "input_sha256",
]
LEFT_OUT_COLUMNS = ["ticker", "release_accession", "prompt_tokens", "reason"]
PREFERRED_COLUMNS = ["ticker", "release_accession", "status"]


class SplitError(Exception):
    """The split files can't be built; the message says why."""


# CSV rows and hashes


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_text(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """rows under a header, as the rest of the project writes CSV."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def csv_line(row: Mapping[str, str], columns: Sequence[str]) -> str:
    """One row as it is written in a CSV file of these columns."""
    return csv_text([row], columns).split("\n", 1)[1]


def read_table(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    """A CSV file's rows; its header must be exactly columns."""
    if not path.is_file():
        raise SplitError(f"{path} not found")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if reader.fieldnames != list(columns):
        raise SplitError(f"{path} has columns {reader.fieldnames}, not {columns}")
    return rows


# Releases


@dataclass(frozen=True)
class Sources:
    """Where a split's links, answers and release files come from."""

    links: Path
    answers: Path
    releases: Path


@dataclass
class Release:
    split: str
    ticker: str
    accession: str
    linked_accession: str
    preferred: bool
    links_row: dict[str, str]
    answer_rows: list[dict[str, str]]
    text: str  # the model input
    tokens: int = 0  # the prompt's, set by count_tokens
    status: str = ""  # KEPT or LEFT_OUT, set by count_tokens

    @property
    def links_sha256(self) -> str:
        return sha256_text(csv_line(self.links_row, LINK_COLUMNS))

    @property
    def answers_sha256(self) -> str:
        """Order-free: answers.csv's row order is not part of the answer key."""
        lines = sorted(csv_line(row, ANSWER_COLUMNS) for row in self.answer_rows)
        return sha256_text("".join(lines))

    @property
    def input_sha256(self) -> str:
        return sha256_text(self.text)

    def frozen_row(self) -> dict[str, str]:
        return {
            "split": self.split,
            "ticker": self.ticker,
            "release_accession": self.accession,
            "linked_accession": self.linked_accession,
            "preferred_dividends": str(self.preferred),
            "links_sha256": self.links_sha256,
            "answers_sha256": self.answers_sha256,
            "input_sha256": self.input_sha256,
        }


@dataclass
class Split:
    name: str
    sources: Sources
    status: str  # FROZEN or PROVISIONAL
    releases: list[Release]
    notes: list[str]  # shown in the report

    @property
    def kept(self) -> list[Release]:
        return [r for r in self.releases if r.status == KEPT]

    @property
    def left_out(self) -> list[Release]:
        return [r for r in self.releases if r.status == LEFT_OUT]


def selected_from_releases_csv(
    name: str, meta: Sequence[Mapping[str, str]], links: Sequence[Mapping[str, str]]
) -> list[str]:
    """The split's linked releases by releases.csv, in links.csv's order."""
    wanted = {
        r["release_accession"]
        for r in meta
        if r["split"] == name and r["link_status"] == LINKED
    }
    linked = [r["release_accession"] for r in links if r["status"] == LINKED]
    missing = sorted(wanted - set(linked))
    if missing:
        raise SplitError(
            f"{name}: linked in releases.csv but not in links.csv: "
            + ", ".join(missing[:5])
        )
    return [a for a in linked if a in wanted]


def build_split(
    name: str,
    sources: Sources,
    accessions: Sequence[str],
    status: str,
    preferred: Mapping[str, bool],
    notes: Sequence[str] = (),
) -> Split:
    """The split's releases with their links row, answers rows and input text."""
    links = {
        r["release_accession"]: r
        for r in read_table(sources.links, LINK_COLUMNS)
        if r["status"] == LINKED
    }
    answers: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_table(sources.answers, ANSWER_COLUMNS):
        answers[row["release_accession"]].append(row)
    records = release_records(sources.releases)
    releases = []
    for accession in accessions:
        row = links.get(accession)
        if row is None:
            raise SplitError(f"{name}: {accession} is not linked in {sources.links}")
        record = records.get(accession)
        if record is None:
            raise SplitError(
                f"{name}: no saved release {accession} in {sources.releases}"
            )
        if accession not in preferred:
            raise SplitError(f"{name}: {accession} is not in releases.csv")
        releases.append(
            Release(
                split=name,
                ticker=row["ticker"],
                accession=accession,
                linked_accession=row["linked_accession"],
                preferred=preferred[accession],
                links_row=row,
                answer_rows=answers.get(accession, []),
                text=build_input(record).text,
            )
        )
    return Split(name, sources, status, releases, list(notes))


def frozen_problems(split: Split, frozen: Sequence[Mapping[str, str]]) -> list[str]:
    """Each way the split's releases differ from their frozen rows."""
    problems = []
    for release, row in zip(split.releases, frozen, strict=True):
        built = release.frozen_row()
        for column in FROZEN_COLUMNS[1:]:
            if built[column] != row[column]:
                problems.append(
                    f"{split.name} {release.accession}: {column} is {built[column]}, "
                    f"not the frozen {row[column]}"
                )
    return problems


def count_tokens(
    splits: Sequence[Split], prompt: str, tokenizer: Any, limit: int
) -> None:
    """Set each release's prompt tokens (the system prompt and its input in Qwen3's
    layout with thinking off, generation prompt included: chat.prompt_text), and
    whether it is kept: at most limit tokens."""
    releases = [r for s in splits for r in s.releases]
    texts = [chat.prompt_text(prompt, r.text) for r in releases]
    encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
    for release, tokens in zip(releases, encoded, strict=True):
        release.tokens = len(tokens.ids)
        release.status = KEPT if release.tokens <= limit else LEFT_OUT


def left_out_problems(
    splits: Sequence[Split], expected: Mapping[str, Mapping[str, int]]
) -> list[str]:
    """Differences between each split's left-out releases and the expected ones."""
    problems = []
    for split in splits:
        found = {r.accession: r.tokens for r in split.left_out}
        want = expected.get(split.name, {})
        for accession in sorted(found.keys() | want.keys()):
            got, wanted = found.get(accession), want.get(accession)
            if got == wanted:
                continue
            if wanted is None:
                problems.append(
                    f"{split.name} {accession}: left out ({got:,} tokens), not in the "
                    "expected list"
                )
            elif got is None:
                problems.append(
                    f"{split.name} {accession}: expected left out ({wanted:,} tokens), "
                    "but it isn't"
                )
            else:
                problems.append(
                    f"{split.name} {accession}: {got:,} tokens, expected {wanted:,}"
                )
    return problems


# Output


def answers_in_input(release: Release) -> int:
    """How many of the release's answers its input prints (the grader's check)."""
    rows = grade.input_rows(release.text)
    return sum(
        grade.find_in_input(Decimal(a["value"]), a["field"], release.text, rows)
        is not None
        for a in release.answer_rows
    )


def left_out_reason(limit: int) -> str:
    return f"prompt over {limit:,} tokens (PLAN.md Decisions 2026-11-01)"


def split_files(split: Split, limit: int) -> dict[str, str]:
    """The text of each file but the manifest."""
    kept = split.kept
    return {
        LINKS: csv_text((r.links_row for r in kept), LINK_COLUMNS),
        ANSWERS: csv_text((a for r in kept for a in r.answer_rows), ANSWER_COLUMNS),
        TOKENS: csv_text(
            (
                {
                    "ticker": r.ticker,
                    "release_accession": r.accession,
                    "prompt_tokens": r.tokens,
                    "status": r.status,
                    "input_sha256": r.input_sha256,
                }
                for r in split.releases
            ),
            TOKEN_COLUMNS,
        ),
        LEFT_OUT_FILE: csv_text(
            (
                {
                    "ticker": r.ticker,
                    "release_accession": r.accession,
                    "prompt_tokens": r.tokens,
                    "reason": left_out_reason(limit),
                }
                for r in split.left_out
            ),
            LEFT_OUT_COLUMNS,
        ),
        PREFERRED: csv_text(
            (
                {
                    "ticker": r.ticker,
                    "release_accession": r.accession,
                    "status": r.status,
                }
                for r in split.releases
                if r.preferred
            ),
            PREFERRED_COLUMNS,
        ),
    }


@dataclass(frozen=True)
class Counts:
    releases: int
    kept: int
    left_out: int
    companies_kept: int
    answers: int  # of the kept releases
    answers_in_input: int
    preferred_kept: int
    preferred_left_out: int


def counts(split: Split) -> Counts:
    kept = split.kept
    return Counts(
        releases=len(split.releases),
        kept=len(kept),
        left_out=len(split.left_out),
        companies_kept=len({r.ticker for r in kept}),
        answers=sum(len(r.answer_rows) for r in kept),
        answers_in_input=sum(answers_in_input(r) for r in kept),
        preferred_kept=sum(r.preferred for r in kept),
        preferred_left_out=sum(r.preferred for r in split.left_out),
    )


def longest(releases: Sequence[Release]) -> Release | None:
    """The release with the most prompt tokens; ties to the first."""
    return max(releases, key=lambda r: r.tokens, default=None)


def manifest(
    split: Split,
    found: Counts,
    files: Mapping[str, str],
    inputs: Mapping[str, str],
    prompt: extract.Prompt,
    commit: str,
    limit: int,
) -> dict[str, Any]:
    top = longest(split.kept)
    return {
        "split": split.name,
        "status": split.status,
        "commit": commit,
        "releases_dir": str(split.sources.releases),
        "counts": asdict(found),
        "longest_kept_prompt": None
        if top is None
        else {
            "ticker": top.ticker,
            "release_accession": top.accession,
            "tokens": top.tokens,
        },
        "context_limit": {
            "max_model_len": chat.MAX_MODEL_LEN,
            "max_tokens": chat.MAX_TOKENS,
            "max_prompt_tokens": limit,
        },
        "tokenizer": {
            "model": labels.QWEN3,
            "revision": labels.QWEN3_REVISION,
            "sha256": labels.TOKENIZER_SHA256,
        },
        "prompt": {"version": prompt.version, "sha256": prompt.sha256},
        "inputs": dict(inputs),  # path -> sha256
        "files": {name: sha256_text(text) for name, text in files.items()},
    }


def format_report(
    splits: Sequence[Split],
    found: Mapping[str, Counts],
    out: Path,
    commit: str,
    limit: int,
) -> str:
    lines = [
        (
            f"Split files in {out} at {commit[:12]}: prompts over {limit:,} tokens "
            "left out (PLAN.md Decisions 2026-11-01), counted by "
            f"{labels.QWEN3} at {labels.QWEN3_REVISION[:12]}."
        ),
        "",
        (
            "| split | status | kept | left out | companies kept | longest kept "
            "prompt | answers (in the input) | preferred-dividend releases kept |"
        ),
        "| --- " * 8 + "|",
    ]
    for split in splits:
        c = found[split.name]
        top = longest(split.kept)
        lines.append(
            f"| {split.name} | {split.status} | {c.kept:,} | {c.left_out:,} | "
            f"{c.companies_kept:,} | {top.tokens if top else 0:,} | "
            f"{c.answers:,} ({c.answers_in_input:,}) | {c.preferred_kept:,} |"
        )
    for split in splits:
        if split.left_out:
            lines += ["", f"Left out of {split.name}:"]
            lines += [
                f"- {r.ticker} {r.accession}: {r.tokens:,} tokens"
                + (", preferred dividends" if r.preferred else "")
                for r in sorted(split.left_out, key=lambda r: -r.tokens)
            ]
        lines += [f"Note, {split.name}: {note}" for note in split.notes]
    return "\n".join(lines) + "\n"


# Command


def disagreement(
    name: str,
    frozen: Sequence[str],
    meta: Sequence[Mapping[str, str]],
    links: Sequence[Mapping[str, str]],
) -> str | None:
    """A note when releases.csv's split no longer matches the frozen list."""
    try:
        current = set(selected_from_releases_csv(name, meta, links))
    except SplitError as exc:
        return (
            f"releases.csv doesn't give this split ({exc}); built from the frozen list"
        )
    added, gone = current - set(frozen), set(frozen) - current
    if not added and not gone:
        return None
    return (
        f"releases.csv now gives {len(added)} releases not in the frozen list and "
        f"lacks {len(gone)} of it; built from the frozen list"
    )


def build_all(
    stage1: Sources,
    stage2: Path,
    frozen_rows: Sequence[Mapping[str, str]],
    freeze: Sequence[str],
) -> list[Split]:
    meta = read_table(stage2 / "releases.csv", RELEASE_COLUMNS)
    preferred = {
        r["release_accession"]: r["preferred_dividends"] == "True" for r in meta
    }
    splits = []
    for name in SPLITS:
        sources = (
            stage1
            if name == VALIDATION
            else Sources(stage2 / LINKS, stage2 / ANSWERS, stage2 / "releases")
        )
        links = read_table(sources.links, LINK_COLUMNS)
        mine = [r for r in frozen_rows if r["split"] == name]
        if mine and name in freeze:
            raise SplitError(f"{name} is already frozen")
        if mine:
            accessions = [r["release_accession"] for r in mine]
            note = disagreement(name, accessions, meta, links)
            split = build_split(
                name, sources, accessions, FROZEN, preferred, [note] if note else []
            )
            problems = frozen_problems(split, mine)
            if problems:
                raise SplitError(
                    f"{len(problems)} differences from the frozen list: "
                    + "; ".join(problems[:5])
                )
        else:
            accessions = selected_from_releases_csv(name, meta, links)
            if name == VALIDATION:
                others = [
                    r["release_accession"]
                    for r in links
                    if r["status"] == LINKED
                    and r["release_accession"] not in accessions
                ]
                if others:
                    raise SplitError(
                        f"{sources.links} has linked releases that releases.csv doesn't "
                        f"put in validation: {', '.join(others[:5])}"
                    )
            split = build_split(name, sources, accessions, PROVISIONAL, preferred)
        splits.append(split)
    return splits


def is_tracked(path: Path, repo: Path = REPO) -> bool:
    """Whether git tracks path (relative to the working directory)."""
    return (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "ls-files",
                "--error-unmatch",
                str(path.resolve()),
            ],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def main(
    argv: list[str] | None = None,
    *,
    commit: str | None = None,
    tokenizer: Any = None,
    expected_left_out: Mapping[str, Mapping[str, int]] = EXPECTED_LEFT_OUT,
    limit: int = MAX_PROMPT_TOKENS,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.splits",
        description="Write data/splits/<split>/ for validation, company_test and "
        "time_test.",
    )
    parser.add_argument(
        "--freeze",
        nargs="+",
        choices=SPLITS,
        default=[],
        help="add these splits to the frozen list",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--releases", type=Path, default=DEFAULT_RELEASES)
    parser.add_argument("--frozen", type=Path, default=FROZEN_LIST)
    parser.add_argument(
        "--raw", type=Path, default=labels.DEFAULT_RAW, help="the tokenizer's cache"
    )
    parser.add_argument("--prompts-dir", type=Path, default=extract.PROMPTS_DIR)
    args = parser.parse_args(argv)
    try:
        if commit is None:
            commit, dirty = current_commit()
            if dirty:
                raise SplitError(
                    "tracked files have uncommitted changes; commit them first (each "
                    "manifest.json records the commit)"
                )
            if args.frozen.is_file() and not is_tracked(args.frozen):
                raise SplitError(f"{args.frozen} isn't committed; commit it first")
        frozen_rows = (
            read_table(args.frozen, FROZEN_COLUMNS) if args.frozen.is_file() else []
        )
        prompt = extract.load_prompt(labels.PROMPT, args.prompts_dir)
        if tokenizer is None:
            tokenizer = labels.load_tokenizer(args.raw)
        stage1 = Sources(args.links, args.answers, args.releases)
        splits = build_all(stage1, args.stage2, frozen_rows, args.freeze)
        count_tokens(splits, prompt.text, tokenizer, limit)
        problems = left_out_problems(splits, expected_left_out)
        if problems:
            raise SplitError(
                "the releases left out aren't the expected ones (PLAN.md Decisions "
                "2026-11-01; splits.EXPECTED_LEFT_OUT): " + "; ".join(problems)
            )
    except (SplitError, extract.RunError, labels.LabelError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.freeze:
        rows = [*frozen_rows]
        for split in splits:
            if split.name in args.freeze:
                rows += [r.frozen_row() for r in split.releases]
        rows.sort(key=lambda r: SPLITS.index(r["split"]))  # stable within a split
        args.frozen.parent.mkdir(parents=True, exist_ok=True)
        write_text(args.frozen, csv_text(rows, FROZEN_COLUMNS))
    found = {}
    for split in splits:
        files = split_files(split, limit)
        found[split.name] = counts(split)
        inputs = {
            str(path): file_sha256(path)
            for path in (
                split.sources.links,
                split.sources.answers,
                args.stage2 / "releases.csv",
                args.prompts_dir / f"{prompt.version}.txt",
                *([args.frozen] if split.status == FROZEN else []),
            )
        }
        folder = args.out / split.name
        folder.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            write_text(folder / name, text)
        record = manifest(
            split, found[split.name], files, inputs, prompt, commit, limit
        )
        write_text(folder / MANIFEST, json.dumps(record, indent=2) + "\n")
    print(format_report(splits, found, args.out, commit, limit), end="")
    if args.freeze:
        print(
            f"froze {', '.join(args.freeze)} in {args.frozen}: commit it, then run "
            "secbench.splits again to build them as frozen"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
