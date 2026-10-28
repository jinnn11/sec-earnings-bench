"""Stage 2 data (PLAN.md Decisions 2026-10-27): releases, links and XBRL answers for
the companies in data/companies.csv, one company at a time in one process, resumable,
with a failure log.

Train and company_test companies: every Item 2.02 8-K filed SINCE to UNTIL, saved
under <out>/releases/ (secbench.releases), linked to its 10-Q or 10-K
(secbench.link) and answered from XBRL company facts (secbench.answers).
Validation companies: the stage 1 releases and links as they are (data/links.csv;
nothing fetched), answered again from the cached company facts, which must give
data/answers.csv's rows.

Filings, submissions and company facts are fetched by the company's CIK in
companies.csv, plus the other CIKs of its edgar.TICKER_CIKS group (ExxonMobil's
predecessor), never by looking up its ticker. Where SEC's ticker map (the cached
company_tickers.json) disagrees, that is reported in <out>/ticker_map.csv, not
followed.

Each company's results go to <out>/companies/<cik>/ (releases.csv, links.csv,
answers.csv, preferred_dividends.csv, revenue_checks.csv, failures.csv), then
status.json, which records the git commit that built them. A company counts as done
only if the current commit (HEAD) built it and finished: it is then skipped (--redo
runs it again); any other runs again, from the cache. A build refuses to start
while tracked files have uncommitted changes, so the commit is the code that ran
(untracked files such as data/ don't count). Every request goes through the
EDGAR cache at 8 requests per second. Failures never stop the run; each is logged
with its company and release: no Item 2.02 8-K in the window (no_item_202), an 8-K
without an HTML press release or that isn't an earnings release (no_press_release),
a release without its 10-Q/10-K (no_linked_filing), a figure without an XBRL answer
(no_xbrl_answer), and any other error (error). After the selected companies, the
files of every company built by the current commit are merged into
<out>/{releases,links,answers,preferred_dividends,revenue_checks,failures}.csv
(failures from every such company, the rest from finished ones); companies built by
another commit, or not at all, are left out and named. <out>/merged.json records the
commit, the companies left out and the rows. --merge-only merges without building.
revenue_checks.csv lists the revenue answers between 1/5 and 1/2 of a larger
revenue-like fact (see secbench.answers.RevenueCheck).

Release splits (releases.csv's split column): a train company's releases filed
before TIME_TEST_START are train and the rest time_test; a company_test company's
latest COMPANY_TEST_RELEASES linked releases are company_test and its others unused;
a validation company's are validation.

--offline builds from the cache alone: anything not in it is an error for that
company, never a request.

Usage: uv run python -m secbench.stage2 [--tickers T ...] [--redo] [--merge-only]
       [--offline] [--out data/stage2]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from secbench.answers import COLUMNS as ANSWER_COLUMNS
from secbench.answers import DEFAULT_OUT as STAGE1_ANSWERS
from secbench.answers import (
    LinkedRelease,
    Results,
    RevenueCheck,
    answer_ticker,
    answers_csv,
)
from secbench.companies import (
    COMPANY_TEST,
    TRAIN,
    VALIDATION,
    Company,
    cik_mismatches,
    load_companies,
)
from secbench.companies import DEFAULT_OUT as DEFAULT_COMPANIES
from secbench.edgar import (
    TICKER_CIKS,
    TICKERS_URL,
    EdgarClient,
    EdgarError,
    default_client,
    format_cik,
    parse_company_tickers,
)
from secbench.link import COLUMNS as LINK_COLUMNS
from secbench.link import (
    CONFLICT,
    EXCLUDED,
    LINKED,
    UNLINKED,
    Link,
    link_releases,
    links_csv,
    load_release,
    ticker_filings,
)
from secbench.link import DEFAULT_OUT as STAGE1_LINKS
from secbench.releases import collect_releases, write_text

log = logging.getLogger(__name__)

DEFAULT_OUT = Path("data/stage2")
SINCE = date(2023, 10, 1)
# Q3 2026 releases come in November, once their 10-Qs are out (PLAN.md To do).
UNTIL = date(2026, 9, 30)
TIME_TEST_START = date(2026, 7, 1)
COMPANY_TEST_RELEASES = 4
TIME_TEST = "time_test"
UNUSED = "unused"  # a company_test company's releases other than its latest four
RELEASE_SPLITS = (VALIDATION, COMPANY_TEST, TRAIN, TIME_TEST, UNUSED)

# Failure kinds.
NO_ITEM_202 = "no_item_202"
NO_PRESS_RELEASE = "no_press_release"
NO_LINKED_FILING = "no_linked_filing"
NO_XBRL_ANSWER = "no_xbrl_answer"
ERROR = "error"
KINDS = (NO_ITEM_202, NO_PRESS_RELEASE, NO_LINKED_FILING, NO_XBRL_ANSWER, ERROR)

DONE = "done"
FAILED = "failed"
STATUS = "status.json"
MERGED = "merged.json"
REPO = Path(__file__).resolve().parents[2]
LEFT_OUT_SHOWN = 10
TICKER_MAP = "ticker_map.csv"
TICKER_MAP_COLUMNS = ["ticker", "cik", "sec_cik"]


@dataclass(frozen=True)
class Failure:
    ticker: str
    cik: int
    split: str  # the company's
    release_accession: str  # blank for the whole company
    step: str  # releases, link or answers
    kind: str  # one of KINDS
    reason: str


@dataclass
class ReleaseRow:
    """One Item 2.02 8-K of a company in the window, and what became of it."""

    ticker: str
    cik: int  # the company's (a release may be under an older CIK)
    company_split: str
    release_accession: str
    filing_date: date
    saved: bool  # an HTML press release was saved
    link_status: str = ""  # linked, excluded, unlinked or conflict; blank if not saved
    linked_form: str = ""
    linked_accession: str = ""
    period_end: date | None = None
    split: str = ""  # see release_splits
    answers: int = 0
    missing: int = 0
    preferred_dividends: bool = False


FAILURE_COLUMNS = [f.name for f in fields(Failure)]
RELEASE_COLUMNS = [f.name for f in fields(ReleaseRow)]
DIVIDEND_COLUMNS = [
    "ticker",
    "release_accession",
    "linked_accession",
    "period_type",
    "tag",
    "value",
]
REVENUE_CHECK_COLUMNS = [f.name for f in fields(RevenueCheck)]
FILES = {
    "releases.csv": RELEASE_COLUMNS,
    "links.csv": LINK_COLUMNS,
    "answers.csv": ANSWER_COLUMNS,
    "preferred_dividends.csv": DIVIDEND_COLUMNS,
    "revenue_checks.csv": REVENUE_CHECK_COLUMNS,
    "failures.csv": FAILURE_COLUMNS,
}


def rows_csv(rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: value.isoformat()
                if isinstance(value, date)
                else ("" if value is None else value)
                for key, value in row.items()
            }
        )
    return buffer.getvalue()


def release_splits(company_split: str, rows: list[ReleaseRow]) -> None:
    """Set each row's split (see the module docstring)."""
    if company_split == COMPANY_TEST:
        linked = [r for r in rows if r.link_status == LINKED]
        latest = sorted(
            linked, key=lambda r: (r.filing_date, r.release_accession), reverse=True
        )[:COMPANY_TEST_RELEASES]
        chosen = {r.release_accession for r in latest}
    for row in rows:
        if company_split == VALIDATION:
            row.split = VALIDATION
        elif company_split == TRAIN:
            row.split = TRAIN if row.filing_date < TIME_TEST_START else TIME_TEST
        else:
            row.split = COMPANY_TEST if row.release_accession in chosen else UNUSED


@dataclass
class CompanyResult:
    company: Company
    status: str = DONE
    releases: list[ReleaseRow] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    results: Results = field(default_factory=Results)
    failures: list[Failure] = field(default_factory=list)

    def fail(self, step: str, kind: str, reason: str, accession: str = "") -> None:
        c = self.company
        self.failures.append(
            Failure(c.ticker, c.cik, c.split, accession, step, kind, reason)
        )


def company_ciks(company: Company) -> list[int]:
    """The CIKs whose filings are the company's: its CIK in companies.csv, then the
    others of the edgar.TICKER_CIKS group holding it. Never a ticker lookup."""
    for ciks in TICKER_CIKS.values():
        if company.cik in ciks:
            return [company.cik, *(cik for cik in ciks if cik != company.cik)]
    return [company.cik]


def ticker_map_rows(
    client: EdgarClient, companies: Sequence[Company]
) -> list[dict[str, Any]]:
    """Companies whose CIK SEC's ticker map (company_tickers.json, from the cache
    once downloaded) doesn't give for their ticker (see companies.cik_mismatches):
    reported, never followed. sec_cik is blank for a ticker not in the map."""
    sec_ciks = parse_company_tickers(client.fetch_json(TICKERS_URL))
    return [asdict(m) for m in cik_mismatches(companies, sec_ciks)]


def link_failures(result: CompanyResult) -> None:
    for link in result.links:
        if link.status == EXCLUDED:
            # link's reason says "not an earnings release: ..."
            result.fail("link", NO_PRESS_RELEASE, link.reason, link.release_accession)
        elif link.status in (UNLINKED, CONFLICT):
            result.fail(
                "link",
                NO_LINKED_FILING,
                f"{link.status}: {link.reason}",
                link.release_accession,
            )


def linked_releases(links: Iterable[Link]) -> list[LinkedRelease]:
    return [
        LinkedRelease(
            link.ticker,
            link.release_accession,
            link.linked_form,
            link.linked_accession,
            link.period_end,
        )
        for link in links
        if link.status == LINKED and link.period_end is not None
    ]


def answer(client: EdgarClient, result: CompanyResult) -> None:
    """Answers for the company's linked releases; missing company facts are a
    company-wide no_xbrl_answer."""
    linked = linked_releases(result.links)
    if not linked:
        return
    try:
        result.results = answer_ticker(
            client, result.company.ticker, linked, ciks=company_ciks(result.company)
        )
    except EdgarError as exc:
        result.fail("answers", NO_XBRL_ANSWER, f"company facts: {exc}")
        return
    for m in result.results.missing:
        result.fail(
            "answers",
            NO_XBRL_ANSWER,
            f"{m.field} {m.period_type}: {m.reason} ({m.detail})",
            m.release_accession,
        )


def fill_release_rows(result: CompanyResult) -> None:
    """Link status, answer counts and preferred dividends on each release row, then
    the release splits."""
    links = {link.release_accession: link for link in result.links}
    answered = Counter(a.release_accession for a in result.results.answers)
    missing = Counter(m.release_accession for m in result.results.missing)
    dividends = {d.release_accession for d in result.results.preferred_dividends}
    for row in result.releases:
        link = links.get(row.release_accession)
        if link is not None:
            row.link_status = link.status
            row.linked_form = link.linked_form
            row.linked_accession = link.linked_accession
            row.period_end = link.period_end
        row.answers = answered[row.release_accession]
        row.missing = missing[row.release_accession]
        row.preferred_dividends = row.release_accession in dividends
    release_splits(result.company.split, result.releases)


def build_company(
    client: EdgarClient,
    company: Company,
    releases_dir: Path,
    since: date = SINCE,
    until: date = UNTIL,
) -> CompanyResult:
    """Releases, links and answers for a train or company_test company."""
    result = CompanyResult(company)
    ciks, outcomes = collect_releases(
        client, company.ticker, since, until, releases_dir, ciks=company_ciks(company)
    )
    if not outcomes:
        result.fail(
            "releases", NO_ITEM_202, f"no Item 2.02 8-K filed {since} to {until}"
        )
    saved = []
    for outcome in outcomes:
        result.releases.append(
            ReleaseRow(
                company.ticker,
                company.cik,
                company.split,
                outcome.accession_number,
                outcome.filing_date,
                saved=not outcome.skip_reason,
            )
        )
        if outcome.skip_reason:
            result.fail(
                "releases",
                NO_PRESS_RELEASE,
                outcome.skip_reason,
                outcome.accession_number,
            )
        else:
            saved.append(outcome.accession_number)
    records = {
        path.stem: path
        for cik in ciks
        for path in (releases_dir / format_cik(cik)).glob("*.json")
    }
    releases = []
    for accession in saved:
        try:
            releases.append(load_release(records[accession]))
        except (KeyError, ValueError) as exc:
            result.fail(
                "link", NO_PRESS_RELEASE, f"saved record unreadable: {exc}", accession
            )
    if releases:
        filings = ticker_filings(client, company.ticker, ciks=ciks)
        result.links = link_releases(releases, filings)
        link_failures(result)
        answer(client, result)
    fill_release_rows(result)
    return result


def stage1_rows(path: Path, ticker: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["ticker"] == ticker]


def build_validation(
    client: EdgarClient, company: Company, links_path: Path, answers_path: Path
) -> CompanyResult:
    """A validation company from the stage 1 links, answered again from the cached
    company facts; the answers must be stage 1's."""
    result = CompanyResult(company)
    for row in stage1_rows(links_path, company.ticker):
        period_end = row["period_end"]
        link = Link(
            row["ticker"],
            int(row["release_cik"]),
            row["release_accession"],
            date.fromisoformat(row["release_date"]),
            row["linked_form"],
            int(row["linked_cik"]) if row["linked_cik"] else None,
            row["linked_accession"],
            date.fromisoformat(period_end) if period_end else None,
            row["period_type"],
            row["status"],
            row["reason"],
        )
        result.links.append(link)
        result.releases.append(
            ReleaseRow(
                company.ticker,
                company.cik,
                company.split,
                link.release_accession,
                link.release_date,
                saved=True,
            )
        )
    link_failures(result)
    answer(client, result)
    expected = rows_csv(stage1_rows(answers_path, company.ticker), ANSWER_COLUMNS)
    if answers_csv(result.results.answers) != expected:
        result.fail("answers", ERROR, f"answers differ from the stage 1 {answers_path}")
    fill_release_rows(result)
    return result


# Writing, resuming and merging


def company_dir(out: Path, company: Company) -> Path:
    return out / "companies" / format_cik(company.cik)


def current_commit(repo: Path = REPO) -> tuple[str, bool]:
    """HEAD's commit in repo, and whether tracked files differ from it (untracked
    files, such as data/, don't count)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout

    commit = git("rev-parse", "HEAD").strip()
    return commit, bool(git("status", "--porcelain", "--untracked-files=no").strip())


def read_status(out: Path, company: Company) -> dict[str, Any] | None:
    path = company_dir(out, company) / STATUS
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def built_by(status: dict[str, Any] | None, commit: str) -> bool:
    """Whether the company's status says commit built it (finished or not)."""
    return status is not None and status.get("commit") == commit


def is_done(status: dict[str, Any] | None, commit: str) -> bool:
    return built_by(status, commit) and status is not None and status["status"] == DONE


def write_company(
    out: Path, result: CompanyResult, seconds: float, commit: str
) -> None:
    """The company's files, then its status, last (an interrupted write leaves no
    done status, so the company runs again). The status records commit."""
    folder = company_dir(out, result.company)
    folder.mkdir(parents=True, exist_ok=True)
    texts = {
        "releases.csv": rows_csv((asdict(r) for r in result.releases), RELEASE_COLUMNS),
        "links.csv": links_csv(result.links),
        "answers.csv": answers_csv(result.results.answers),
        "preferred_dividends.csv": rows_csv(
            (asdict(d) for d in result.results.preferred_dividends), DIVIDEND_COLUMNS
        ),
        "revenue_checks.csv": rows_csv(
            (asdict(c) for c in result.results.revenue_checks), REVENUE_CHECK_COLUMNS
        ),
        "failures.csv": rows_csv((asdict(f) for f in result.failures), FAILURE_COLUMNS),
    }
    for name, text in texts.items():
        write_text(folder / name, text)
    status = {
        "ticker": result.company.ticker,
        "cik": result.company.cik,
        "split": result.company.split,
        "status": result.status,
        "commit": commit,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(seconds, 1),
        "releases": len(result.releases),
        "linked": sum(r.link_status == LINKED for r in result.releases),
        "answers": len(result.results.answers),
        "failures": dict(Counter(f.kind for f in result.failures)),
    }
    write_text(folder / STATUS, json.dumps(status, indent=2) + "\n")


def merge(
    out: Path, companies: Sequence[Company], commit: str
) -> tuple[dict[str, int], list[Company]]:
    """Concatenate the files of every company that commit built into <out>/<name>,
    in companies.csv order: failures from each, the rest from finished ones; then
    write <out>/merged.json. Returns the rows per file and the companies left out
    (built by another commit, or not at all)."""
    out.mkdir(parents=True, exist_ok=True)
    statuses = {c.cik: read_status(out, c) for c in companies}
    left_out = [c for c in companies if not built_by(statuses[c.cik], commit)]
    counts = {}
    for name, columns in FILES.items():
        rows: list[dict[str, str]] = []
        for company in companies:
            status = statuses[company.cik]
            if not built_by(status, commit):
                continue
            assert status is not None
            if status["status"] != DONE and name != "failures.csv":
                continue
            path = company_dir(out, company) / name
            if not path.is_file():
                raise ValueError(f"{path}: missing (built before it existed? --redo)")
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != list(columns):
                    raise ValueError(f"{path}: unexpected columns")
                rows.extend(reader)
        write_text(out / name, rows_csv(rows, columns))
        counts[name] = len(rows)
    note = {
        "commit": commit,
        "merged": datetime.now(UTC).isoformat(timespec="seconds"),
        "companies": len(companies) - len(left_out),
        "left_out": [c.ticker for c in left_out],
        "rows": counts,
    }
    write_text(out / MERGED, json.dumps(note, indent=2) + "\n")
    return counts, left_out


def run_company(
    client: EdgarClient,
    company: Company,
    out: Path,
    links_path: Path,
    answers_path: Path,
    since: date,
    until: date,
) -> CompanyResult:
    """One company, with any error logged as a failure instead of raised."""
    try:
        if company.split == VALIDATION:
            return build_validation(client, company, links_path, answers_path)
        return build_company(client, company, out / "releases", since, until)
    except Exception as exc:  # logged as a failure; the run goes on
        if isinstance(exc, EdgarError):
            log.warning("%s failed: %s", company.ticker, exc)
        else:
            log.exception("%s failed", company.ticker)
        result = CompanyResult(company, status=FAILED)
        result.fail("build", ERROR, f"{type(exc).__name__}: {exc}")
        return result


def progress_line(done: int, total: int, result: CompanyResult, seconds: float) -> str:
    c = result.company
    kinds = Counter(f.kind for f in result.failures)
    problems = ", ".join(f"{k} {n}" for k, n in kinds.items()) or "no failures"
    return (
        f"[{done:>{len(str(total))}}/{total}] {c.ticker} ({c.split}): "
        f"{len(result.releases)} 8-Ks, "
        f"{sum(r.link_status == LINKED for r in result.releases)} linked, "
        f"{len(result.results.answers)} answers; {problems}; {result.status} "
        f"{seconds:.1f}s"
    )


def main(
    argv: list[str] | None = None,
    client: EdgarClient | None = None,
    commit: str | None = None,
) -> int:
    """commit: the commit to record and merge by; None reads it from git (see
    current_commit) and refuses to build with uncommitted changes."""
    parser = argparse.ArgumentParser(
        prog="python -m secbench.stage2",
        description="Build stage 2 releases, links and answers, one company at a time.",
    )
    parser.add_argument("--tickers", nargs="+", metavar="TICKER")
    parser.add_argument(
        "--redo", action="store_true", help="run finished companies again"
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="merge the companies the current commit built, without building any",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="read only the EDGAR cache; anything not in it is an error",
    )
    parser.add_argument("--companies", type=Path, default=DEFAULT_COMPANIES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--links", type=Path, default=STAGE1_LINKS)
    parser.add_argument("--answers", type=Path, default=STAGE1_ANSWERS)
    parser.add_argument("--since", type=date.fromisoformat, default=SINCE)
    parser.add_argument("--until", type=date.fromisoformat, default=UNTIL)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("secbench.releases").setLevel(logging.ERROR)

    companies = load_companies(args.companies)
    selected = companies
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        unknown = wanted - {c.ticker for c in companies}
        if unknown:
            print(
                f"error: not in {args.companies}: {', '.join(sorted(unknown))}",
                file=sys.stderr,
            )
            return 1
        selected = [c for c in companies if c.ticker in wanted]
    if commit is None:
        try:
            commit, dirty = current_commit()
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"error: no git commit to record: {exc}", file=sys.stderr)
            return 1
        if dirty and not args.merge_only:
            print(
                "error: tracked files have uncommitted changes; commit them first "
                "(status.json records the commit that built each company)",
                file=sys.stderr,
            )
            return 1
    if args.merge_only:
        return report_merge(args.out, companies, commit)
    todo = [
        c
        for c in selected
        if args.redo or not is_done(read_status(args.out, c), commit)
    ]
    print(
        f"{len(selected)} companies selected, {len(selected) - len(todo)} already done "
        f"at {commit[:12]}, {len(todo)} to run (releases filed {args.since} to "
        f"{args.until})",
        flush=True,
    )
    if client is None:
        client = EdgarClient(offline=True) if args.offline else default_client()
    try:
        mismatches = ticker_map_rows(client, selected)
    except EdgarError as exc:
        print(f"SEC's ticker map not checked: {exc}")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        write_text(args.out / TICKER_MAP, rows_csv(mismatches, TICKER_MAP_COLUMNS))
        print(
            f"SEC's ticker map disagrees with companies.csv for {len(mismatches)} "
            f"of {len(selected)} companies (in {args.out / TICKER_MAP}; not followed)"
            + "".join(
                f"\n  {m['ticker']}: companies.csv {m['cik']}, SEC "
                f"{m['sec_cik'] if m['sec_cik'] is not None else 'not in the map'}"
                for m in mismatches
            )
        )
    errors = 0
    for number, company in enumerate(todo, 1):
        start = time.monotonic()
        result = run_company(
            client, company, args.out, args.links, args.answers, args.since, args.until
        )
        seconds = time.monotonic() - start
        write_company(args.out, result, seconds, commit)
        errors += result.status == FAILED or any(
            f.kind == ERROR for f in result.failures
        )
        print(progress_line(number, len(todo), result, seconds), flush=True)
    report_merge(args.out, companies, commit)
    if errors:
        print(f"{errors} companies with errors; see {args.out / 'failures.csv'}")
    return 1 if errors else 0


def report_merge(out: Path, companies: Sequence[Company], commit: str) -> int:
    """Merge (see merge) and print what was merged and left out."""
    counts, left_out = merge(out, companies, commit)
    print(
        f"merged {len(companies) - len(left_out)} companies built by {commit[:12]}: "
        + ", ".join(f"{out / name} {n} rows" for name, n in counts.items())
    )
    if left_out:
        shown = ", ".join(c.ticker for c in left_out[:LEFT_OUT_SHOWN])
        more = len(left_out) - LEFT_OUT_SHOWN
        print(
            f"left out {len(left_out)} companies not built by {commit[:12]}: {shown}"
            + (f" and {more} more" if more > 0 else "")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
