"""Answer key: GAAP headline figures from the XBRL of each release's linked filing.

For each linked release in links.csv (see secbench.link), reads the XBRL company facts
of every CIK of the ticker and keeps only us-gaap facts from the linked 10-Q or 10-K
(matched on accession number) that end on its period end. A 10-Q gives the
three-month value and, if the filing reports a longer period ending then, the
year-to-date value; a 10-K gives the full-year value. A 10-K's fourth-quarter
revenue and net income are derived: the full year minus the nine months (same tag,
same start) reported by that fiscal year's third-quarter 10-Q, the 10-Q whose period
ends one quarter before the 10-K's (from the cached submissions). EPS is not derived.

Usage: uv run python -m secbench.answers [--links data/links.csv]
       [--out data/answers.csv] [--refresh]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from dataclasses import field as dataclass_field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from secbench.edgar import EdgarClient, EdgarError, default_client, format_cik
from secbench.link import DEFAULT_OUT as DEFAULT_LINKS
from secbench.link import LINKED, ticker_filings
from secbench.releases import write_text

log = logging.getLogger(__name__)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/"
DEFAULT_OUT = Path("data/answers.csv")
TAXONOMY = "us-gaap"

QUARTER = "quarter"
YEAR_TO_DATE = "year_to_date"
FULL_YEAR = "full_year"
PERIOD_TYPES = (QUARTER, YEAR_TO_DATE, FULL_YEAR)
# Period length in days (end minus start) for each period type: 13- and 14-week
# quarters, 26- to 40-week years to date, 52- and 53-week years.
PERIOD_DAYS = {
    QUARTER: range(80, 101),
    YEAR_TO_DATE: range(150, 301),
    FULL_YEAR: range(350, 381),
}
# Form -> the release's headline period.
HEADLINE = {"10-Q": QUARTER, "10-K": FULL_YEAR}


@dataclass(frozen=True)
class Field:
    name: str
    tags: tuple[str, ...]  # us-gaap tags, most preferred first
    unit: str


FIELDS = (
    Field(
        "revenue",
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RegulatedAndUnregulatedOperatingRevenue",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ),
        "USD",
    ),
    Field(
        "net_income",
        (
            "NetIncomeLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
            "ProfitLoss",
        ),
        "USD",
    ),
    Field(
        "eps_diluted",
        (
            "EarningsPerShareDiluted",
            "EarningsPerShareBasicAndDiluted",
            "IncomeLossFromContinuingOperationsPerDilutedShare",
        ),
        "USD/shares",
    ),
)
REVENUE = FIELDS[0]
NET_INCOME = FIELDS[1]
# Tag -> a tag that must be tagged 0 for the same filing and period before the first
# is used. ProfitLoss includes noncontrolling interests, so it is net income only
# when they are 0.
NONCONTROLLING = "NetIncomeLossAttributableToNoncontrollingInterest"
ZERO_REQUIRED = {"ProfitLoss": NONCONTROLLING}
# Tags that are net income after preferred dividends: net income only when there are
# none, so never used for a period with nonzero preferred dividends (Decisions
# 2026-10-11).
AFTER_PREFERRED_DIVIDENDS = {"NetIncomeLossAvailableToCommonStockholdersBasic"}
PREFERRED_DIVIDEND_TAGS = (
    "PreferredStockDividendsIncomeStatementImpact",
    "DividendsPreferredStock",
    "DividendsPreferredStockCash",
    "PreferredStockDividendsAndOtherAdjustments",
)
# Revenue tags after the current two count only for a filing with no fact under
# either current tag (Decisions 2026-10-28): a utility's
# RevenueFromContractWithCustomerIncludingAssessedTax can be one part of the total in
# its RegulatedAndUnregulatedOperatingRevenue.
CURRENT_REVENUE_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
)
WITHOUT_CURRENT_REVENUE = {
    "RegulatedAndUnregulatedOperatingRevenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
}
# Continuing-operations EPS is diluted EPS only for a period without discontinued
# operations (Decisions 2026-10-28), so never used for a period where one of these is
# nonzero.
CONTINUING_OPERATIONS = {"IncomeLossFromContinuingOperationsPerDilutedShare"}
DISCONTINUED_OPERATION_TAGS = (
    "IncomeLossFromDiscontinuedOperationsNetOfTax",
    "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
    "IncomeLossFromDiscontinuedOperationsNetOfTaxIncludingPortionAttributableToNoncontrollingInterest",
    "IncomeLossFromDiscontinuedOperationsNetOfTaxPerBasicShare",
    "IncomeLossFromDiscontinuedOperationsNetOfTaxPerDilutedShare",
    "IncomeLossFromDiscontinuedOperationsNetOfTaxPerBasicAndDilutedShare",
    "DiscontinuedOperationIncomeLossFromDiscontinuedOperationNetOfTaxPerDilutedShare",
)
# Other tags named like revenue, reported when they aren't the answer's tag.
REVENUE_LIKE = re.compile(r"^(Revenue|SalesRevenue)")

# Why a field has no answer, from a tag that got least far to one that got furthest.
NO_TAG = "no tag"
NO_FACT = "no fact for that accession"
NO_PERIOD = "no matching period"
CONDITION_NOT_MET = "condition not met"  # see unmet_condition
CONFLICTING = "conflicting values"
REASONS = (NO_TAG, NO_FACT, NO_PERIOD, CONDITION_NOT_MET, CONFLICTING)
PERIODS_IN_DETAIL = 4

# A 10-K's fourth quarter is derived for these fields (EPS doesn't subtract).
DERIVED_FIELDS = (REVENUE, NET_INCOME)
# Why a derived fourth quarter has none: no third-quarter 10-Q, else the furthest
# any tag got: "full year: <reason>", "nine months: <reason>" or different starts.
NO_THIRD_QUARTER = "no third-quarter 10-Q"
SEVERAL_THIRD_QUARTERS = "several third-quarter 10-Qs"
FULL_YEAR_PART = "full year"
NINE_MONTHS_PART = "nine months"
STARTS_DIFFER = "starts differ"


# Company facts


@dataclass(frozen=True)
class Fact:
    """One us-gaap fact from company facts."""

    tag: str
    unit: str
    start: date | None  # None for an instant
    end: date
    value: Decimal


def companyfacts_url(cik: int | str) -> str:
    return f"{COMPANYFACTS_URL}CIK{format_cik(cik)}.json"


def load_company_facts(
    client: EdgarClient, cik: int, *, refresh: bool = False
) -> dict[str, Any]:
    """A CIK's company facts JSON, with decimals parsed exactly (EPS 1.57 stays
    1.57); refresh re-downloads it."""
    return json.loads(
        client.fetch(companyfacts_url(cik), refresh=refresh), parse_float=Decimal
    )


@dataclass
class FactIndex:
    """us-gaap facts of one ticker, merged across its CIKs."""

    by_accession: dict[str, set[Fact]]  # only the accessions asked for
    known: set[tuple[str, str]]  # (tag, unit) with any fact under any of the CIKs


def index_facts(
    companyfacts: Iterable[dict[str, Any]], accessions: set[str]
) -> FactIndex:
    """Index the facts of the given accessions. A fact reported under several CIKs
    (ExxonMobil's joint 10-Q) is kept once. Values must be ints or Decimals (see
    load_company_facts); a float has already lost the exact figure."""
    by_accession: dict[str, set[Fact]] = defaultdict(set)
    known: set[tuple[str, str]] = set()
    for document in companyfacts:
        for tag, body in document.get("facts", {}).get(TAXONOMY, {}).items():
            for unit, entries in body.get("units", {}).items():
                if entries:
                    known.add((tag, unit))
                for entry in entries:
                    if entry["accn"] not in accessions:
                        continue
                    if isinstance(entry["val"], float):
                        raise TypeError(
                            f"{tag} value {entry['val']!r} is a float; parse company "
                            "facts with parse_float=Decimal"
                        )
                    start = entry.get("start")
                    by_accession[entry["accn"]].add(
                        Fact(
                            tag=tag,
                            unit=unit,
                            start=date.fromisoformat(start) if start else None,
                            end=date.fromisoformat(entry["end"]),
                            value=Decimal(entry["val"]),
                        )
                    )
    return FactIndex(dict(by_accession), known)


def classify_period(start: date | None, end: date) -> str | None:
    """The period type a fact's length fits, or None (instants, other lengths)."""
    if start is None:
        return None
    days = (end - start).days
    for name, window in PERIOD_DAYS.items():
        if days in window:
            return name
    return None


def describe_periods(facts: Iterable[Fact]) -> str:
    """Distinct periods, latest end first, then shortest."""
    periods = sorted(
        {(f.end, f.start) for f in facts},
        key=lambda p: (p[0], p[1] or p[0]),
        reverse=True,
    )
    shown = [
        f"{end} (instant)"
        if start is None
        else f"{start}..{end} ({(end - start).days} days)"
        for end, start in periods[:PERIODS_IN_DETAIL]
    ]
    more = len(periods) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more else "")


# Looking up answers


@dataclass(frozen=True)
class LinkedRelease:
    """A linked row of links.csv."""

    ticker: str
    release_accession: str
    linked_form: str
    linked_accession: str
    period_end: date


@dataclass(frozen=True)
class Lookup:
    """The fact for one field and period of a filing, or why there is none."""

    fact: Fact | None
    reason: str = ""  # one of REASONS: how far the furthest tag got
    detail: str = ""  # what each tag lacked


def lookup(
    field: Field,
    period: str,
    filing_facts: set[Fact],
    period_end: date,
    known: set[tuple[str, str]],
) -> Lookup:
    """The first of field's tags with one value for period (ending on period_end)
    in the filing that meets its condition, if any (see unmet_condition).
    Conflicting values for a tag stop the search rather than fall through to the
    next tag."""
    notes = []
    furthest = 0
    for tag in field.tags:
        facts = [f for f in filing_facts if f.tag == tag and f.unit == field.unit]
        matches = in_period(facts, period, period_end)
        values = sorted({f.value for f in matches})
        if len(values) == 1:
            chosen = min(matches, key=lambda f: f.start or f.end)
            unmet = unmet_condition(tag, chosen, filing_facts)
            if unmet is None:
                return Lookup(chosen)
            reason, note = CONDITION_NOT_MET, f"{chosen.value}, {unmet}"
        elif values:
            reason = CONFLICTING
            note = f"values {', '.join(map(str, values))}"
        elif facts:
            reason, note = NO_PERIOD, f"this filing has {describe_periods(facts)}"
        elif (tag, field.unit) in known:
            reason, note = NO_FACT, "none from this filing"
        else:
            reason, note = NO_TAG, f"no {field.unit} facts in company facts"
        furthest = max(furthest, REASONS.index(reason))
        notes.append(f"{tag}: {note}")
        if reason == CONFLICTING:
            break
    detail = "; ".join(notes)
    if not filing_facts:
        detail += (
            "; company facts have nothing from this filing (downloaded before it "
            "was filed? try --refresh)"
        )
    return Lookup(None, REASONS[furthest], detail)


def unmet_condition(tag: str, fact: Fact, filing_facts: set[Fact]) -> str | None:
    """Why tag's value fact can't be the answer, or None if it can:
    - ProfitLoss needs its ZERO_REQUIRED tag to be 0 for exactly that period;
    - a tag after preferred dividends needs none (see preferred_dividends);
    - a revenue tag after the current two needs a filing with no fact under either
      current tag (any period);
    - continuing-operations EPS needs no nonzero discontinued operations for exactly
      that period."""
    required = ZERO_REQUIRED.get(tag)
    if required:
        zero = required_values(required, fact, filing_facts)
        if zero != [0]:
            shown = ", ".join(map(str, zero)) if zero else "not tagged"
            return f"but {required} is {shown} for that period"
    if tag in AFTER_PREFERRED_DIVIDENDS:
        dividends = preferred_dividends(fact, filing_facts)
        if dividends:
            return "after preferred dividends: " + ", ".join(
                f"{f.tag} {describe_fact_period(f)} {f.value}" for f in dividends
            )
    if tag in WITHOUT_CURRENT_REVENUE:
        current = sorted({f.tag for f in filing_facts if f.tag in CURRENT_REVENUE_TAGS})
        if current:
            return f"but this filing has {' and '.join(current)} facts"
    if tag in CONTINUING_OPERATIONS:
        discontinued = discontinued_operations(fact, filing_facts)
        if discontinued:
            return "but discontinued operations: " + ", ".join(
                f"{f.tag} {describe_fact_period(f)} {f.value}" for f in discontinued
            )
    return None


def discontinued_operations(fact: Fact, filing_facts: set[Fact]) -> list[Fact]:
    """Nonzero discontinued-operations facts in the same filing for exactly fact's
    period, in any unit."""
    return sorted(
        (
            f
            for f in filing_facts
            if f.tag in DISCONTINUED_OPERATION_TAGS
            and f.value != 0
            and (f.start, f.end) == (fact.start, fact.end)
        ),
        key=lambda f: (f.tag, f.unit),
    )


def in_period(facts: Iterable[Fact], period: str, period_end: date) -> list[Fact]:
    """The facts for a period of the given type ending on period_end."""
    return [
        f
        for f in facts
        if f.end == period_end and classify_period(f.start, f.end) == period
    ]


def required_values(
    required: str, fact: Fact, filing_facts: set[Fact]
) -> list[Decimal]:
    """The values of tag required for exactly fact's period, in the same filing."""
    return sorted(
        {
            f.value
            for f in filing_facts
            if f.tag == required
            and f.unit == fact.unit
            and (f.start, f.end) == (fact.start, fact.end)
        }
    )


def preferred_dividends(fact: Fact, filing_facts: set[Fact]) -> list[Fact]:
    """Nonzero preferred dividends in the same filing for fact's period, or for a
    longer period ending the same day that contains it: a quarter whose dividends are
    tagged only for the year to date may have some too, and a missing answer is
    better than one after preferred dividends."""
    return sorted(
        (
            f
            for f in filing_facts
            if f.tag in PREFERRED_DIVIDEND_TAGS
            and f.unit == "USD"
            and f.value != 0
            and f.end == fact.end
            and f.start is not None
            and fact.start is not None
            and f.start <= fact.start
        ),
        key=lambda f: (f.tag, f.start or f.end),
    )


def describe_fact_period(fact: Fact) -> str:
    return f"{fact.start}..{fact.end}" if fact.start else str(fact.end)


def expected_periods(form: str, filing_facts: set[Fact], period_end: date) -> list[str]:
    """The headline period, plus year to date for a 10-Q that reports a longer USD
    period ending on its period end (a first-quarter 10-Q has none)."""
    if form == "10-Q" and any(
        f.unit == "USD"
        and f.end == period_end
        and classify_period(f.start, f.end) == YEAR_TO_DATE
        for f in filing_facts
    ):
        return [QUARTER, YEAR_TO_DATE]
    return [HEADLINE[form]]


# Fourth quarters


@dataclass(frozen=True)
class ThirdQuarter:
    """A 10-K's third-quarter 10-Q: the 10-Q whose period ends one quarter before the
    10-K's. accession is None when there is none (reason and detail say why)."""

    accession: str | None
    period_end: date | None
    reason: str = ""
    detail: str = ""


def find_third_quarter(filings: pd.DataFrame, period_end: date) -> ThirdQuarter:
    """The filing of form exactly "10-Q" (not "10-Q/A") whose report date leaves a
    quarter to period_end: the day after it to period_end is a quarter by
    classify_period. filings is a submissions table (see link.ticker_filings)."""
    candidates = filings[(filings["form"] == "10-Q") & filings["report_date"].notna()]
    found = sorted(
        (accession, end.date())
        for accession, end in zip(
            candidates["accession_number"], candidates["report_date"], strict=True
        )
        if classify_period(end.date() + timedelta(days=1), period_end) == QUARTER
    )
    if len(found) == 1:
        return ThirdQuarter(*found[0])
    if not found:
        return ThirdQuarter(
            None,
            None,
            NO_THIRD_QUARTER,
            f"no 10-Q's period ends a quarter before {period_end}",
        )
    return ThirdQuarter(
        None,
        None,
        SEVERAL_THIRD_QUARTERS,
        ", ".join(f"{accession} (ends {end})" for accession, end in found),
    )


@dataclass(frozen=True)
class Derived:
    """The full year and nine months a fourth quarter is derived from, or why there
    are none."""

    full_year: Fact | None
    nine_months: Fact | None
    reason: str = ""
    detail: str = ""


def derive_quarter(
    field: Field,
    filing_facts: set[Fact],
    period_end: date,
    third_quarter: ThirdQuarter,
    nine_month_facts: set[Fact],
    known: set[tuple[str, str]],
) -> Derived:
    """The first of field's tags with one full-year value in the 10-K (filing_facts,
    ending on period_end) and one nine-month value from the same start in its
    third-quarter 10-Q (nine_month_facts), each found as lookup finds a period, so
    each filing must meet the tag's condition (see unmet_condition).
    Conflicting values stop the search."""
    if third_quarter.accession is None:
        return Derived(None, None, third_quarter.reason, third_quarter.detail)
    assert third_quarter.period_end is not None
    notes = []
    furthest = (-1, "")  # (progress, reason): full year, then nine months, then start
    for tag in field.tags:
        one_tag = Field(field.name, (tag,), field.unit)
        full = lookup(one_tag, FULL_YEAR, filing_facts, period_end, known)
        if full.fact is None:
            progress = REASONS.index(full.reason)
            reason = f"{FULL_YEAR_PART}: {full.reason}"
            note = f"{FULL_YEAR_PART} {full.detail}"
            stop = full.reason == CONFLICTING
        else:
            nine = lookup(
                one_tag,
                YEAR_TO_DATE,
                nine_month_facts,
                third_quarter.period_end,
                known,
            )
            if nine.fact is None:
                progress = len(REASONS) + REASONS.index(nine.reason)
                reason = f"{NINE_MONTHS_PART}: {nine.reason}"
                note = f"{NINE_MONTHS_PART} {nine.detail}"
                stop = nine.reason == CONFLICTING
            elif nine.fact.start != full.fact.start:
                progress = 2 * len(REASONS)
                reason = STARTS_DIFFER
                note = (
                    f"{tag}: full year starts {full.fact.start}, nine months "
                    f"{nine.fact.start}"
                )
                stop = False
            else:
                return Derived(full.fact, nine.fact)
        furthest = max(furthest, (progress, reason))
        notes.append(note)
        if stop:
            break
    detail = f"10-Q {third_quarter.accession}: " + "; ".join(notes)
    return Derived(None, None, furthest[1], detail)


@dataclass(frozen=True)
class Answer:
    """One row of answers.csv: a fact of the linked filing, or a derived row whose
    value is the full year in full_year_accession minus the nine months in
    nine_months_accession, both facts of tag starting on fiscal_year_start (the
    last three are blank on other rows)."""

    ticker: str
    release_accession: str
    linked_accession: str
    field: str
    period_type: str
    start: date
    end: date
    value: Decimal
    unit: str
    tag: str
    derived: bool = False
    full_year_accession: str = ""
    nine_months_accession: str = ""
    fiscal_year_start: date | None = None


COLUMNS = [f.name for f in fields(Answer)]


@dataclass(frozen=True)
class Missing:
    """An expected answer that the linked filing's XBRL doesn't give."""

    ticker: str
    release_accession: str
    linked_form: str
    linked_accession: str
    field: str
    period_type: str
    reason: str
    detail: str


@dataclass(frozen=True)
class OtherRevenue:
    """A revenue-like tag at a filing's headline period that wasn't the answer."""

    ticker: str
    linked_accession: str
    tag: str
    value: Decimal
    answer: Decimal | None  # the filing's revenue answer, if it has one


@dataclass(frozen=True)
class FallbackCheck:
    """NetIncomeLoss and a net income fallback tag, both tagged for one filing and
    period."""

    ticker: str
    linked_accession: str
    period_type: str
    tag: str
    value: Decimal
    net_income_loss: Decimal
    usable: bool  # the tag's condition holds: ZERO_REQUIRED, no preferred dividends


@dataclass(frozen=True)
class PreferredDividend:
    """A nonzero preferred dividend in a release's linked filing, for one of its
    expected periods."""

    ticker: str
    release_accession: str
    linked_accession: str
    period_type: str
    tag: str
    value: Decimal


@dataclass
class Results:
    answers: list[Answer] = dataclass_field(default_factory=list)
    missing: list[Missing] = dataclass_field(default_factory=list)
    other_revenue: list[OtherRevenue] = dataclass_field(default_factory=list)
    fallback_checks: list[FallbackCheck] = dataclass_field(default_factory=list)
    preferred_dividends: list[PreferredDividend] = dataclass_field(default_factory=list)

    def extend(self, other: Results) -> None:
        self.answers.extend(other.answers)
        self.missing.extend(other.missing)
        self.other_revenue.extend(other.other_revenue)
        self.fallback_checks.extend(other.fallback_checks)
        self.preferred_dividends.extend(other.preferred_dividends)


def answer_release(
    release: LinkedRelease,
    index: FactIndex,
    third_quarter: ThirdQuarter | None = None,
) -> Results:
    """Answers for every field and expected period of one linked release. A 10-K
    release needs its third_quarter (see find_third_quarter): its derived fourth
    quarters come first in their field, as quarters do in a 10-Q's."""
    filing_facts = index.by_accession.get(release.linked_accession, set())
    if release.linked_form == "10-K" and third_quarter is None:
        raise ValueError(
            f"{release.release_accession}: a 10-K release needs its third quarter"
        )
    results = Results()
    headline_revenue: Fact | None = None
    periods = expected_periods(release.linked_form, filing_facts, release.period_end)
    for field in FIELDS:
        if third_quarter is not None and field in DERIVED_FIELDS:
            derived, missing = derive_answer(
                release, field, filing_facts, third_quarter, index
            )
            if derived:
                results.answers.append(derived)
            if missing:
                results.missing.append(missing)
        for period in periods:
            found = lookup(field, period, filing_facts, release.period_end, index.known)
            if found.fact is None:
                results.missing.append(
                    Missing(
                        release.ticker,
                        release.release_accession,
                        release.linked_form,
                        release.linked_accession,
                        field.name,
                        period,
                        found.reason,
                        found.detail,
                    )
                )
                continue
            fact = found.fact
            assert fact.start is not None
            results.answers.append(
                Answer(
                    release.ticker,
                    release.release_accession,
                    release.linked_accession,
                    field.name,
                    period,
                    fact.start,
                    fact.end,
                    fact.value,
                    fact.unit,
                    fact.tag,
                )
            )
            if field is REVENUE and period == HEADLINE[release.linked_form]:
                headline_revenue = fact
    results.other_revenue = other_revenue_tags(release, filing_facts, headline_revenue)
    for period in periods:
        results.fallback_checks.extend(check_fallbacks(release, filing_facts, period))
        results.preferred_dividends.extend(
            PreferredDividend(
                release.ticker,
                release.release_accession,
                release.linked_accession,
                period,
                f.tag,
                f.value,
            )
            for f in sorted(
                in_period(
                    (
                        f
                        for f in filing_facts
                        if f.tag in PREFERRED_DIVIDEND_TAGS and f.unit == "USD"
                    ),
                    period,
                    release.period_end,
                ),
                key=lambda f: f.tag,
            )
            if f.value != 0
        )
    return results


def derive_answer(
    release: LinkedRelease,
    field: Field,
    filing_facts: set[Fact],
    third_quarter: ThirdQuarter,
    index: FactIndex,
) -> tuple[Answer | None, Missing | None]:
    """A 10-K release's derived fourth quarter for field, or why it has none."""
    nine_month_facts = index.by_accession.get(third_quarter.accession or "", set())
    found = derive_quarter(
        field,
        filing_facts,
        release.period_end,
        third_quarter,
        nine_month_facts,
        index.known,
    )
    if found.full_year is None or found.nine_months is None:
        return None, Missing(
            release.ticker,
            release.release_accession,
            release.linked_form,
            release.linked_accession,
            field.name,
            QUARTER,
            found.reason,
            found.detail,
        )
    full_year, nine_months = found.full_year, found.nine_months
    start = nine_months.end + timedelta(days=1)
    assert classify_period(start, full_year.end) == QUARTER  # see find_third_quarter
    return Answer(
        release.ticker,
        release.release_accession,
        release.linked_accession,
        field.name,
        QUARTER,
        start,
        full_year.end,
        full_year.value - nine_months.value,
        field.unit,
        full_year.tag,
        derived=True,
        full_year_accession=release.linked_accession,
        nine_months_accession=third_quarter.accession or "",
        fiscal_year_start=full_year.start,
    ), None


def check_fallbacks(
    release: LinkedRelease, filing_facts: set[Fact], period: str
) -> list[FallbackCheck]:
    """Each net income fallback tag with one value for period where NetIncomeLoss
    has one too."""
    first, *fallbacks = NET_INCOME.tags

    def single(tag: str) -> Fact | None:
        facts = [f for f in filing_facts if f.tag == tag and f.unit == NET_INCOME.unit]
        matches = in_period(facts, period, release.period_end)
        return matches[0] if len({f.value for f in matches}) == 1 else None

    net_income_loss = single(first)
    if net_income_loss is None:
        return []
    checks = []
    for tag in fallbacks:
        fact = single(tag)
        if fact is None:
            continue
        required = ZERO_REQUIRED.get(tag)
        checks.append(
            FallbackCheck(
                release.ticker,
                release.linked_accession,
                period,
                tag,
                fact.value,
                net_income_loss.value,
                (
                    required is None
                    or required_values(required, fact, filing_facts) == [0]
                )
                and not (
                    tag in AFTER_PREFERRED_DIVIDENDS
                    and preferred_dividends(fact, filing_facts)
                ),
            )
        )
    return checks


def other_revenue_tags(
    release: LinkedRelease, filing_facts: set[Fact], answer: Fact | None
) -> list[OtherRevenue]:
    """USD tags named like revenue ("Revenue...", "SalesRevenue...") with a value
    for the filing's headline period, other than the tag of the revenue answer."""
    headline = HEADLINE[release.linked_form]
    return [
        OtherRevenue(
            release.ticker,
            release.linked_accession,
            f.tag,
            f.value,
            answer.value if answer else None,
        )
        for f in sorted(filing_facts, key=lambda f: (f.tag, f.value))
        if REVENUE_LIKE.match(f.tag)
        and f.unit == REVENUE.unit
        and (answer is None or f.tag != answer.tag)
        and f.end == release.period_end
        and classify_period(f.start, f.end) == headline
    ]


def answer_ticker(
    client: EdgarClient,
    ticker: str,
    releases: list[LinkedRelease],
    *,
    refresh: bool = False,
) -> Results:
    """Answers for one ticker's linked releases, from the company facts of every
    CIK in edgar.TICKER_CIKS, and for a 10-K, its third-quarter 10-Q from their
    submissions; refresh re-downloads both."""
    companyfacts = [
        load_company_facts(client, cik, refresh=refresh)
        for cik in client.ticker_to_ciks(ticker)
    ]
    third_quarters: dict[str, ThirdQuarter] = {}
    tenks = [r for r in releases if r.linked_form == "10-K"]
    if tenks:
        filings = ticker_filings(client, ticker, refresh=refresh)
        third_quarters = {
            r.release_accession: find_third_quarter(filings, r.period_end)
            for r in tenks
        }
    accessions = {r.linked_accession for r in releases} | {
        q.accession for q in third_quarters.values() if q.accession
    }
    index = index_facts(companyfacts, accessions)
    results = Results()
    for release in releases:
        third_quarter = third_quarters.get(release.release_accession)
        results.extend(answer_release(release, index, third_quarter))
    return results


# Input and output


def load_links(path: Path) -> tuple[list[LinkedRelease], int]:
    """The linked rows of links.csv, in file order, and how many rows were not
    linked."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    releases = []
    for row in rows:
        if row["status"] != LINKED:
            continue
        if row["linked_form"] not in HEADLINE:
            raise ValueError(
                f"{path}: {row['release_accession']} is linked to a "
                f"{row['linked_form']!r}, not a 10-Q or 10-K"
            )
        releases.append(
            LinkedRelease(
                ticker=row["ticker"],
                release_accession=row["release_accession"],
                linked_form=row["linked_form"],
                linked_accession=row["linked_accession"],
                period_end=date.fromisoformat(row["period_end"]),
            )
        )
    return releases, len(rows) - len(releases)


def answers_csv(answers: list[Answer]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for answer in answers:
        writer.writerow({key: csv_cell(value) for key, value in asdict(answer).items()})
    return buffer.getvalue()


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    return value.isoformat() if isinstance(value, date) else str(value)


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def format_report(releases: list[LinkedRelease], results: Results) -> str:
    """Headline answers per ticker and field, year-to-date and derived fourth-quarter
    counts, every missing answer with its reason, and the revenue-like tags not
    used."""
    names = [field.name for field in FIELDS]
    tickers = sorted({r.ticker for r in releases})
    linked = Counter(r.ticker for r in releases)
    headline = Counter(
        (a.ticker, a.field)
        for a in results.answers
        if a.period_type != YEAR_TO_DATE and not a.derived
    )

    def line(name: str, values: list[Any]) -> str:
        return f"{name:<8}" + "".join(f"{v:>13}" for v in values)

    lines = [line("ticker", ["releases", *names])]
    for ticker in tickers:
        lines.append(
            line(ticker, [linked[ticker], *(headline[ticker, n] for n in names)])
        )
    totals = [sum(headline[t, n] for t in tickers) for n in names]
    lines.append(line("total", [len(releases), *totals]))

    ytd_filings = {
        x.linked_accession
        for x in [*results.answers, *results.missing]
        if x.period_type == YEAR_TO_DATE
    }
    ytd = Counter(a.field for a in results.answers if a.period_type == YEAR_TO_DATE)
    lines.append(
        f"year to date ({plural(len(ytd_filings), '10-Q')} report one): "
        + ", ".join(f"{n} {ytd[n]}" for n in names)
    )

    tenks = sum(r.linked_form == "10-K" for r in releases)
    derived = [a for a in results.answers if a.derived]
    counts = Counter(a.field for a in derived)
    lines.append(
        f"fourth quarter derived ({plural(tenks, '10-K')}): "
        + ", ".join(f"{f.name} {counts[f.name]}" for f in DERIVED_FIELDS)
    )
    full_year_tags = {
        (a.release_accession, a.field): a.tag
        for a in results.answers
        if a.period_type == FULL_YEAR
    }
    for a in derived:
        full_year_tag = full_year_tags.get((a.release_accession, a.field))
        if a.tag != full_year_tag:
            lines.append(
                f"  {a.ticker} {a.linked_accession} {a.field}: {a.tag} "
                f"(full-year answer {full_year_tag})"
            )

    lines.append(f"missing: {len(results.missing)}")
    for m in results.missing:
        lines.append(
            f"  {m.ticker} {m.release_accession} {m.field} {m.period_type} "
            f"({m.linked_form} {m.linked_accession}): {m.reason} ({m.detail})"
        )

    lines.append("other revenue tags at the headline period (reported, not used):")
    by_tag: dict[tuple[str, str], list[OtherRevenue]] = defaultdict(list)
    for other in results.other_revenue:
        by_tag[other.ticker, other.tag].append(other)
    for (ticker, tag), others in sorted(by_tag.items()):
        filings = {o.linked_accession for o in others}
        equal = {o.linked_accession for o in others if o.value == o.answer}
        no_answer = {o.linked_accession for o in others if o.answer is None}
        text = (
            f"  {ticker} {tag}: {plural(len(filings), 'filing')}, "
            f"{len(equal)} equal to the revenue answer"
        )
        if no_answer:
            text += f", {len(no_answer)} without one"
        differ = [o for o in others if o.answer is not None and o.value != o.answer]
        if differ:
            text += (
                f" (e.g. {differ[0].linked_accession}: {differ[0].value}, "
                f"answer {differ[0].answer})"
            )
        lines.append(text)
    if not by_tag:
        lines.append("  none")

    dividends = sorted(
        {(d.ticker, d.release_accession) for d in results.preferred_dividends}
    )
    lines.append(
        "releases whose linked filing reports nonzero preferred dividends: "
        f"{len(dividends)}"
    )
    for ticker, accession in dividends:
        found = [
            d for d in results.preferred_dividends if d.release_accession == accession
        ]
        lines.append(
            f"  {ticker} {accession}: "
            + ", ".join(f"{d.period_type} {d.tag} {d.value}" for d in found)
        )
    lines.append(
        f"net income fallbacks where {NET_INCOME.tags[0]} is also tagged "
        "(same filing and period):"
    )
    by_fallback: dict[tuple[str, str], list[FallbackCheck]] = defaultdict(list)
    for check in results.fallback_checks:
        by_fallback[check.ticker, check.tag].append(check)
    for (ticker, tag), checks in sorted(by_fallback.items()):
        agree = [c for c in checks if c.value == c.net_income_loss]
        text = f"  {ticker} {tag}: {plural(len(checks), 'period')}, {len(agree)} agree"
        usable = sum(c.usable for c in checks)
        if tag in ZERO_REQUIRED:
            text += f"; {ZERO_REQUIRED[tag]} tagged 0 in {usable}"
        if tag in AFTER_PREFERRED_DIVIDENDS:
            text += f"; no preferred dividends in {usable}"
        differ = [c for c in checks if c.value != c.net_income_loss]
        if differ:
            d = differ[0]
            text += (
                f" (e.g. {d.linked_accession} {d.period_type}: {d.value}, "
                f"{NET_INCOME.tags[0]} {d.net_income_loss})"
            )
        lines.append(text)
    if not by_fallback:
        lines.append("  none")
    return "\n".join(lines)


def main(argv: list[str] | None = None, client: EdgarClient | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.answers",
        description="Build the answer key from the XBRL of each release's linked "
        "10-Q or 10-K.",
    )
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download company facts (to see filings made since the cache)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if not args.links.is_file():
        print(
            f"error: {args.links} not found; run secbench.link first", file=sys.stderr
        )
        return 1
    releases, not_linked = load_links(args.links)
    if not releases:
        print(f"error: no linked releases in {args.links}", file=sys.stderr)
        return 1
    by_ticker: dict[str, list[LinkedRelease]] = defaultdict(list)
    for release in releases:
        by_ticker[release.ticker].append(release)

    status = 0
    done: list[LinkedRelease] = []
    results = Results()
    for ticker in sorted(by_ticker):
        try:
            client = client or default_client()
            one = answer_ticker(client, ticker, by_ticker[ticker], refresh=args.refresh)
        except EdgarError as exc:
            print(
                f"error: {ticker}: {exc}; its releases have no answers", file=sys.stderr
            )
            status = 1
            continue
        done.extend(by_ticker[ticker])
        results.extend(one)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text(args.out, answers_csv(results.answers))
    print(format_report(done, results))
    if not_linked:
        print(f"skipped {plural(not_linked, 'release')} in {args.links} not linked")
    print(f"wrote {len(results.answers)} rows to {args.out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
