"""Coverage of the stage 2 data (see secbench.stage2), by release split and by
quarter.

Reads the merged files in data/stage2/ (releases.csv, answers.csv, failures.csv) and
data/companies.csv, and writes data/stage2/coverage.md (also printed):

- by split, and by calendar quarter of the 8-K's filing date: companies, Item 2.02
  8-Ks found, without a press release (no HTML EX-99.1), not earnings releases
  (excluded by secbench.link's filter), linked to their 10-Q/10-K, not linked,
  figures with an XBRL answer and without, and releases whose linked filing reports
  nonzero preferred dividends;
- every company_test company with fewer than COMPANY_TEST_RELEASES linked releases
  (it stays in company_test with what it has; the split is not redrawn);
- the answer tag used, per field and split;
- revenue answers between 1/5 and 1/2 of a larger revenue-like fact (revenue_checks.csv;
  at most 1/5 is dropped as a component), with the row of the release input that
  prints each;
- failures by kind and company split, and the companies with errors.

Usage: uv run python -m secbench.coverage [--stage2 data/stage2]
       [--companies data/companies.csv] [--stage1-releases data/releases]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path

from secbench.companies import COMPANY_TEST, load_companies
from secbench.companies import DEFAULT_OUT as DEFAULT_COMPANIES
from secbench.companies import SPLITS as COMPANY_SPLITS
from secbench.grade import input_rows, match_figure
from secbench.inputs import build_input, release_records
from secbench.link import CONFLICT, EXCLUDED, LINKED, UNLINKED, shorten
from secbench.releases import DEFAULT_OUT_DIR as STAGE1_RELEASES
from secbench.releases import write_text
from secbench.stage2 import COMPANY_TEST_RELEASES, ERROR, KINDS, RELEASE_SPLITS
from secbench.stage2 import DEFAULT_OUT as DEFAULT_STAGE2

REPORT = "coverage.md"
COLUMNS = [
    "8-Ks",
    "no press release",
    "not earnings",
    "linked",
    "not linked",
    "figures answered",
    "figures missing",
    "answered",
    "preferred dividends",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quarter(filing_date: str) -> str:
    day = date.fromisoformat(filing_date)
    return f"{day.year}Q{(day.month - 1) // 3 + 1}"


def counts(rows: Sequence[dict[str, str]]) -> list[str]:
    """The COLUMNS cells for a group of releases.csv rows."""
    answered = sum(int(r["answers"]) for r in rows)
    missing = sum(int(r["missing"]) for r in rows)
    share = (
        f"{100 * answered / (answered + missing):.1f}%" if answered + missing else "–"
    )
    return [
        str(len(rows)),
        str(sum(r["saved"] != "True" for r in rows)),
        str(sum(r["link_status"] == EXCLUDED for r in rows)),
        str(sum(r["link_status"] == LINKED for r in rows)),
        str(sum(r["link_status"] in (UNLINKED, CONFLICT) for r in rows)),
        str(answered),
        str(missing),
        share,
        str(sum(r["preferred_dividends"] == "True" for r in rows)),
    ]


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    return [line(header), "|" + "|".join(" --- " for _ in header) + "|"] + [
        line(row) for row in rows
    ]


def short_company_test(
    releases: Sequence[dict[str, str]], company_splits: dict[str, str]
) -> list[list[str]]:
    """Rows (ticker, 8-Ks, linked) for each company_test company with fewer than
    COMPANY_TEST_RELEASES linked releases, by ticker; one without releases in the
    data (none found, or not finished) counts 0."""
    eight_ks = Counter(r["ticker"] for r in releases)
    linked = Counter(r["ticker"] for r in releases if r["link_status"] == LINKED)
    return [
        [ticker, str(eight_ks[ticker]), str(linked[ticker])]
        for ticker in sorted(company_splits)
        if company_splits[ticker] == COMPANY_TEST
        and linked[ticker] < COMPANY_TEST_RELEASES
    ]


def printed_in(value: Decimal, record: Path | None) -> str:
    """Where a release's model input shows value as revenue: the row's label and
    the figure as shown (in dollars, thousands or millions), or why not."""
    if record is None:
        return "release not saved"
    rows = input_rows(build_input(record).text)
    found = match_figure(
        value, "revenue", ((row, figure) for row in rows for figure in row.figures)
    )
    if found is None:
        return "not printed"
    row, figure, _ = found
    return f"{shorten(row.label)}: {figure.text}".replace("|", r"\|")


def revenue_check_rows(
    checks: Sequence[dict[str, str]],
    split_of: dict[str, str],
    printed: Callable[[dict[str, str]], str],
) -> list[list[str]]:
    """One table row per revenue_checks.csv row."""
    rows = []
    for c in checks:
        derived = c["derived"] == "True"
        share = Decimal(c["value"]) / Decimal(c["larger_value"])
        rows.append(
            [
                c["ticker"],
                c["release_accession"],
                split_of.get(c["release_accession"], ""),
                c["period_type"] + (", derived" if derived else ""),
                c["answer"],
                # A derived quarter's fact compared: its full year or nine months.
                c["tag"]
                + (f" {c['start']}..{c['end']} {c['value']}" if derived else ""),
                f"{c['larger_tag']} {c['larger_value']}",
                f"{100 * share:.0f}%",
                printed(c),
            ]
        )
    return rows


def format_coverage(
    releases: list[dict[str, str]],
    answers: list[dict[str, str]],
    failures: list[dict[str, str]],
    company_splits: dict[str, str],
    revenue_checks: Sequence[dict[str, str]] = (),
    printed: Callable[[dict[str, str]], str] = lambda check: "",
) -> str:
    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_quarter: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in releases:
        by_split[row["split"]].append(row)
        by_quarter[quarter(row["filing_date"])].append(row)
    listed = Counter(company_splits.values())
    with_releases = {r["ticker"] for r in releases}
    lines = [
        "# Stage 2 coverage",
        "",
        (
            "Companies in the list: "
            + ", ".join(f"{s} {listed[s]}" for s in COMPANY_SPLITS)
            + f"; with releases in the data: {len(with_releases)}."
        ),
        "",
        "## By release split",
        "",
        *table(
            ["split", "companies", *COLUMNS],
            (
                [
                    split,
                    str(len({r["ticker"] for r in by_split[split]})),
                    *counts(by_split[split]),
                ]
                for split in RELEASE_SPLITS
                if by_split[split]
            ),
        ),
        "",
        (
            f"## company_test companies with fewer than {COMPANY_TEST_RELEASES} "
            "linked releases"
        ),
        "",
    ]
    short = short_company_test(releases, company_splits)
    lines.extend(table(["ticker", "8-Ks", "linked"], short) if short else ["None."])
    lines += [
        "",
        "## By quarter of filing",
        "",
        *table(
            ["quarter", *COLUMNS],
            ([q, *counts(by_quarter[q])] for q in sorted(by_quarter)),
        ),
        "",
        "## Answer tags by release split",
        "",
    ]
    split_of = {r["release_accession"]: r["split"] for r in releases}
    tags: Counter[tuple[str, str, str]] = Counter(
        (a["field"], a["tag"], split_of.get(a["release_accession"], ""))
        for a in answers
    )
    used = [s for s in RELEASE_SPLITS if any(s == key[2] for key in tags)]
    fields_tags = sorted({(f, t) for f, t, _ in tags})
    if fields_tags:
        lines.extend(
            table(
                ["field", "tag", *used],
                ([f, t, *(str(tags[f, t, s]) for s in used)] for f, t in fields_tags),
            )
        )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Revenue answers between 1/5 and 1/2 of a larger revenue-like fact",
            "",
            (
                "Listed, not dropped; an answer at most 1/5 of one is dropped as a "
                "component. Share: the fact compared over the larger one."
            ),
            "",
        ]
    )
    checks = revenue_check_rows(revenue_checks, split_of, printed)
    header = [
        "ticker",
        "release",
        "split",
        "period",
        "answer",
        "tag",
        "larger",
        "share",
        "printed in",
    ]
    lines.extend(table(header, checks) if checks else ["None."])
    lines.extend(["", "## Failures by kind and company split", ""])
    kinds = Counter((f["kind"], f["split"]) for f in failures)
    if kinds:
        lines.extend(
            table(
                ["kind", *COMPANY_SPLITS],
                (
                    [kind, *(str(kinds[kind, s]) for s in COMPANY_SPLITS)]
                    for kind in KINDS
                    if any(kinds[kind, s] for s in COMPANY_SPLITS)
                ),
            )
        )
    else:
        lines.append("None.")
    errors = [f for f in failures if f["kind"] == ERROR]
    if errors:
        lines.extend(["", "Companies with errors:"])
        lines.extend(
            f"- {f['ticker']} ({f['split']}): {f['step']}: {f['reason']}"
            for f in errors
        )
    lines.extend(
        [
            "",
            "Notes:",
            (
                "- Splits: a train company's releases filed before 2026-07-01 are "
                "train, later ones time_test; a company_test company's latest 4 linked "
                "releases are company_test, its others unused; validation is the stage 1 "
                "releases."
            ),
            (
                "- Figures: answers and missing answers for every expected field and "
                "period of each linked release (quarter, year to date or full year, "
                "derived fourth quarter)."
            ),
            (
                "- Preferred dividends: releases whose linked filing reports a nonzero "
                "preferred dividend for one of their periods."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.coverage",
        description="Report the stage 2 data's coverage by split and quarter.",
    )
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--companies", type=Path, default=DEFAULT_COMPANIES)
    parser.add_argument("--stage1-releases", type=Path, default=STAGE1_RELEASES)
    args = parser.parse_args(argv)
    names = ("releases.csv", "answers.csv", "failures.csv", "revenue_checks.csv")
    needed = [args.stage2 / n for n in names]
    absent = [str(p) for p in [*needed, args.companies] if not p.is_file()]
    if absent:
        print(
            f"error: not found: {', '.join(absent)}; run secbench.stage2",
            file=sys.stderr,
        )
        return 1
    releases, answers, failures, revenue_checks = (read_csv(p) for p in needed)
    company_splits = {c.ticker: c.split for c in load_companies(args.companies)}
    # Validation releases are stage 1's; the others are saved under <stage2>/releases.
    records = release_records(args.stage1_releases) | release_records(
        args.stage2 / "releases"
    )
    text = format_coverage(
        releases,
        answers,
        failures,
        company_splits,
        revenue_checks,
        lambda c: printed_in(Decimal(c["answer"]), records.get(c["release_accession"])),
    )
    write_text(args.stage2 / REPORT, text)
    print(text, end="")
    print(f"wrote {args.stage2 / REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
