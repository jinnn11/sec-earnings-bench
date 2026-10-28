import contextlib
import csv
import io
import json
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx
import pandas as pd
import pytest

from secbench import answers
from secbench.answers import (
    COLUMNS,
    CONDITION_NOT_MET,
    CONFLICTING,
    FIELDS,
    FULL_YEAR,
    IMPLAUSIBLE_EPS,
    LOOKS_LIKE_COMPONENT,
    NO_FACT,
    NO_PERIOD,
    NO_TAG,
    NO_THIRD_QUARTER,
    PERIOD_TYPES,
    QUARTER,
    SEVERAL_THIRD_QUARTERS,
    STARTS_DIFFER,
    YEAR_TO_DATE,
    Answer,
    LinkedRelease,
    RevenueCheck,
    ThirdQuarter,
    answer_release,
    answers_csv,
    classify_period,
    find_third_quarter,
    format_report,
    index_facts,
    is_revenue_like,
    load_company_facts,
    load_links,
)
from secbench.edgar import EdgarClient, RateLimiter, parse_submissions

FIXTURES = Path(__file__).parent / "fixtures"
UA = "Test Runner test@example.com"
COMPANYFACTS = FIXTURES / "answers_companyfacts.json"
# Submissions (10-Q and 10-K rows since 2025-01-01), for third-quarter 10-Qs.
SUBMISSIONS = FIXTURES / "link_submissions.json"
LINKS = FIXTURES / "answers_links.csv"
HANDCHECKED = FIXTURES / "answers_handchecked.csv"
HANDCHECKED_COLUMNS = [
    "ticker",
    "linked_accession",
    "field",
    "period_type",
    "expected_value",
]
FIELD_NAMES = [field.name for field in FIELDS]

ACCN = "0000000001-26-000010"  # the linked filing in synthetic tests
LATER = "0000000001-26-000020"  # another filing of the same company
REV = "Revenues"
RFC = "RevenueFromContractWithCustomerExcludingAssessedTax"
NI = "NetIncomeLoss"
AVAIL = "NetIncomeLossAvailableToCommonStockholdersBasic"
PL = "ProfitLoss"
NCI = "NetIncomeLossAttributableToNoncontrollingInterest"
PSD = "PreferredStockDividendsIncomeStatementImpact"
# What the net income fallbacks add to a missing answer's detail when untagged.
NO_NI_FALLBACKS = (
    f"; {AVAIL}: no USD facts in company facts; {PL}: no USD facts in company facts"
)
EPS = "EarningsPerShareDiluted"
EPS_BD = "EarningsPerShareBasicAndDiluted"
RUU = "RegulatedAndUnregulatedOperatingRevenue"
RIA = "RevenueFromContractWithCustomerIncludingAssessedTax"
CONT = "IncomeLossFromContinuingOperationsPerDilutedShare"
DISC = "IncomeLossFromDiscontinuedOperationsNetOfTax"
# What the revenue and EPS tags after the current ones add when untagged.
NO_REV_FALLBACKS = (
    f"; {RUU}: no USD facts in company facts; {RIA}: no USD facts in company facts"
)
NO_EPS_FALLBACK = f"; {CONT}: no USD/shares facts in company facts"
USD = "USD"
PER_SHARE = "USD/shares"

# Periods of a calendar-year company's second-quarter 10-Q (ends 2026-06-30) and
# its comparatives.
Q2 = ("2026-04-01", "2026-06-30")
H1 = ("2026-01-01", "2026-06-30")
Q2_PRIOR = ("2025-04-01", "2025-06-30")
H1_PRIOR = ("2025-01-01", "2025-06-30")
Q1 = ("2026-01-01", "2026-03-31")
# A 10-K (ACCN, ending 2025-12-31) and its fiscal year's third-quarter 10-Q.
FY = ("2025-01-01", "2025-12-31")
Q4 = ("2025-10-01", "2025-12-31")
NINE_MONTHS = ("2025-01-01", "2025-09-30")
Q3_10Q = "0000000001-25-000030"
THIRD_QUARTER = ThirdQuarter(Q3_10Q, date(2025, 9, 30))


def fact(
    period: tuple[str, str] | str, val: float, accn: str = ACCN
) -> dict[str, object]:
    """A company-facts entry as load_company_facts returns it (decimals as
    Decimal); a str period is an instant."""
    start, end = (None, period) if isinstance(period, str) else period
    entry: dict[str, object] = {} if start is None else {"start": start}
    entry |= {
        "end": end,
        "val": Decimal(repr(val)) if isinstance(val, float) else val,
        "accn": accn,
        "fy": 2026,
        "fp": "Q2",
        "form": "10-Q",
        "filed": "2026-07-30",
    }
    return entry


def companyfacts(tags: dict[str, dict[str, list[dict]]], cik: int = 1) -> dict:
    """A company facts document holding the given us-gaap tags ({tag: {unit:
    [entries]}})."""
    return {
        "cik": cik,
        "entityName": "Test Co",
        "facts": {
            "dei": {},
            "us-gaap": {
                tag: {"label": tag, "description": "", "units": units}
                for tag, units in tags.items()
            },
        },
    }


def linked(form: str = "10-Q", end: str = "2026-06-30", accn: str = ACCN):
    return LinkedRelease(
        "TEST", "0000000001-26-000009", form, accn, date.fromisoformat(end)
    )


def run(
    release: LinkedRelease,
    *documents: dict,
    third_quarter: ThirdQuarter | None = None,
) -> answers.Results:
    """Answers for release; a 10-K's third quarter is THIRD_QUARTER unless given."""
    if release.linked_form == "10-K" and third_quarter is None:
        third_quarter = THIRD_QUARTER
    accessions = {release.linked_accession}
    if third_quarter is not None and third_quarter.accession:
        accessions.add(third_quarter.accession)
    return answer_release(release, index_facts(documents, accessions), third_quarter)


def values(results: answers.Results) -> dict[tuple[str, str], tuple[str, str]]:
    """(field, period type) -> (value, tag) of each answer."""
    return {(a.field, a.period_type): (str(a.value), a.tag) for a in results.answers}


def missing(results: answers.Results) -> dict[tuple[str, str], tuple[str, str]]:
    """(field, period type) -> (reason, detail) of each missing answer."""
    return {(m.field, m.period_type): (m.reason, m.detail) for m in results.missing}


def second_quarter(**overrides: dict[str, list[dict]]) -> dict:
    """A full second-quarter 10-Q: each field for the quarter, the half year and
    both prior-year comparatives."""
    tags = {
        REV: {
            USD: [
                fact(Q2, 300),
                fact(H1, 550),
                fact(Q2_PRIOR, 280),
                fact(H1_PRIOR, 500),
            ]
        },
        NI: {USD: [fact(Q2, 30), fact(H1, 52), fact(Q2_PRIOR, 25), fact(H1_PRIOR, 45)]},
        EPS: {
            PER_SHARE: [
                fact(Q2, 1.57),
                fact(H1, 2.84),
                fact(Q2_PRIOR, 1.3),
                fact(H1_PRIOR, 2.4),
            ]
        },
    }
    return companyfacts(tags | overrides)


# Periods


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (79, None),
        (80, QUARTER),
        (88, QUARTER),  # Walmart Feb 1 - Apr 30
        (89, QUARTER),  # Jan 1 - Mar 31
        (90, QUARTER),  # 13 weeks
        (91, QUARTER),
        (97, QUARTER),  # 14 weeks
        (100, QUARTER),
        (101, None),
        (149, None),
        (150, YEAR_TO_DATE),
        (180, YEAR_TO_DATE),  # Jan 1 - Jun 30
        (188, YEAR_TO_DATE),  # 27 weeks
        (273, YEAR_TO_DATE),  # Jul 1 - Mar 31
        (279, YEAR_TO_DATE),  # 40 weeks
        (300, YEAR_TO_DATE),
        (301, None),
        (349, None),
        (350, FULL_YEAR),
        (363, FULL_YEAR),  # 52 weeks
        (364, FULL_YEAR),  # calendar year
        (365, FULL_YEAR),  # leap year
        (370, FULL_YEAR),  # 53 weeks
        (380, FULL_YEAR),
        (381, None),
        (0, None),
    ],
)
def test_classify_period_by_length(days, expected) -> None:
    end = date(2026, 6, 30)
    assert classify_period(end - timedelta(days=days), end) == expected


def test_classify_period_instant() -> None:
    assert classify_period(None, date(2026, 6, 30)) is None


# Choosing facts


def test_10q_gives_quarter_and_year_to_date() -> None:
    results = run(linked(), second_quarter())
    assert values(results) == {
        ("revenue", QUARTER): ("300", REV),
        ("revenue", YEAR_TO_DATE): ("550", REV),
        ("net_income", QUARTER): ("30", NI),
        ("net_income", YEAR_TO_DATE): ("52", NI),
        ("eps_diluted", QUARTER): ("1.57", EPS),
        ("eps_diluted", YEAR_TO_DATE): ("2.84", EPS),
    }
    assert results.missing == []
    first = results.answers[0]
    assert first == Answer(
        ticker="TEST",
        release_accession="0000000001-26-000009",
        linked_accession=ACCN,
        field="revenue",
        period_type=QUARTER,
        start=date(2026, 4, 1),
        end=date(2026, 6, 30),
        value=Decimal(300),
        unit=USD,
        tag=REV,
    )


def test_answers_follow_field_then_period_order() -> None:
    results = run(linked(), second_quarter())
    assert [(a.field, a.period_type) for a in results.answers] == [
        (field, period) for field in FIELD_NAMES for period in (QUARTER, YEAR_TO_DATE)
    ]


def test_first_quarter_10q_has_no_year_to_date() -> None:
    document = companyfacts(
        {
            REV: {USD: [fact(Q1, 100), fact(("2025-01-01", "2025-03-31"), 90)]},
            NI: {USD: [fact(Q1, 10)]},
            EPS: {PER_SHARE: [fact(Q1, 0.5)]},
        }
    )
    results = run(linked(end="2026-03-31"), document)
    assert set(values(results)) == {(name, QUARTER) for name in FIELD_NAMES}
    assert results.missing == []


def test_year_to_date_expected_when_any_usd_fact_has_one() -> None:
    # Net income has a half-year value, so revenue's missing one is reported.
    document = companyfacts(
        {
            REV: {USD: [fact(Q2, 300)]},
            NI: {USD: [fact(Q2, 30), fact(H1, 52)]},
            EPS: {PER_SHARE: [fact(Q2, 1.57)]},
        }
    )
    results = run(linked(), document)
    assert ("net_income", YEAR_TO_DATE) in values(results)
    assert missing(results) == {
        ("revenue", YEAR_TO_DATE): (
            NO_PERIOD,
            (
                f"{REV}: this filing has 2026-04-01..2026-06-30 (90 days); "
                f"{RFC}: no USD facts in company facts" + NO_REV_FALLBACKS
            ),
        ),
        ("eps_diluted", YEAR_TO_DATE): (
            NO_PERIOD,
            (
                f"{EPS}: this filing has 2026-04-01..2026-06-30 (90 days); "
                f"{EPS_BD}: no USD/shares facts in company facts" + NO_EPS_FALLBACK
            ),
        ),
    }


# Fourth quarters


def fiscal_year(**overrides: dict[str, list[dict]]) -> dict:
    """A 10-K (ACCN) with each field for the full year and the prior year, and its
    third-quarter 10-Q (Q3_10Q) with each for the nine months."""
    tags = {
        REV: {
            USD: [
                fact(FY, 1200),
                fact(("2024-01-01", "2024-12-31"), 1100),
                fact(NINE_MONTHS, 880, accn=Q3_10Q),
            ]
        },
        NI: {USD: [fact(FY, 120), fact(NINE_MONTHS, 87, accn=Q3_10Q)]},
        EPS: {PER_SHARE: [fact(FY, 6.7), fact(NINE_MONTHS, 4.9, accn=Q3_10Q)]},
    }
    return companyfacts(tags | overrides)


def ten_k() -> LinkedRelease:
    return linked(form="10-K", end="2025-12-31")


def test_10k_gives_full_year_and_derived_fourth_quarter() -> None:
    results = run(ten_k(), fiscal_year())
    assert values(results) == {
        ("revenue", QUARTER): ("320", REV),
        ("revenue", FULL_YEAR): ("1200", REV),
        ("net_income", QUARTER): ("33", NI),
        ("net_income", FULL_YEAR): ("120", NI),
        ("eps_diluted", FULL_YEAR): ("6.7", EPS),  # EPS doesn't subtract
    }
    assert results.missing == []
    assert [(a.field, a.period_type) for a in results.answers] == [
        ("revenue", QUARTER),
        ("revenue", FULL_YEAR),
        ("net_income", QUARTER),
        ("net_income", FULL_YEAR),
        ("eps_diluted", FULL_YEAR),
    ]
    assert results.answers[0] == Answer(
        ticker="TEST",
        release_accession="0000000001-26-000009",
        linked_accession=ACCN,
        field="revenue",
        period_type=QUARTER,
        start=date(2025, 10, 1),
        end=date(2025, 12, 31),
        value=Decimal(320),
        unit=USD,
        tag=REV,
        derived=True,
        full_year_accession=ACCN,
        nine_months_accession=Q3_10Q,
        fiscal_year_start=date(2025, 1, 1),
    )
    assert not results.answers[1].derived


def test_10ks_own_fourth_quarter_is_not_the_answer() -> None:
    # Some 10-Ks also report the fourth quarter; the quarter is still derived.
    document = fiscal_year(
        **{
            REV: {
                USD: [
                    fact(FY, 1200),
                    fact(Q4, 321),
                    fact(NINE_MONTHS, 880, accn=Q3_10Q),
                ]
            },
            EPS: {PER_SHARE: [fact(FY, 6.7), fact(Q4, 1.8)]},
        }
    )
    got = values(run(ten_k(), document))
    assert got["revenue", QUARTER] == ("320", REV)
    assert ("eps_diluted", QUARTER) not in got


def test_nine_months_come_from_the_third_quarter_10q() -> None:
    # A later 10-Q repeats the nine months as a comparative (here restated).
    document = fiscal_year(
        **{
            REV: {
                USD: [
                    fact(FY, 1200),
                    fact(NINE_MONTHS, 870, accn=LATER),
                    fact(NINE_MONTHS, 880, accn=Q3_10Q),
                ]
            },
            NI: {USD: [fact(FY, 120), fact(NINE_MONTHS, 86, accn=LATER)]},
        }
    )
    results = run(ten_k(), document)
    assert values(results)["revenue", QUARTER] == ("320", REV)
    assert missing(results) == {
        ("net_income", QUARTER): (
            f"nine months: {NO_FACT}",
            (
                f"10-Q {Q3_10Q}: nine months {NI}: none from this filing; "
                f"full year {AVAIL}: no USD facts in company facts; "
                f"full year {PL}: no USD facts in company facts"
            ),
        )
    }


def test_derived_quarter_needs_the_same_start() -> None:
    document = fiscal_year(
        **{
            REV: {
                USD: [fact(FY, 1200), fact(("2025-01-02", "2025-09-30"), 880, Q3_10Q)]
            },
        }
    )
    assert missing(run(ten_k(), document)) == {
        ("revenue", QUARTER): (
            STARTS_DIFFER,
            (
                f"10-Q {Q3_10Q}: {REV}: full year starts 2025-01-01, nine months "
                f"2025-01-02; full year {RFC}: no USD facts in company facts; "
                f"full year {RUU}: no USD facts in company facts; "
                f"full year {RIA}: no USD facts in company facts"
            ),
        )
    }


def test_derived_tag_falls_back_per_period() -> None:
    # Oracle: its 10-K tags both revenue tags, its 10-Qs only the second.
    document = fiscal_year(
        **{
            REV: {USD: [fact(FY, 1200)]},
            RFC: {USD: [fact(FY, 1200), fact(NINE_MONTHS, 880, accn=Q3_10Q)]},
        }
    )
    results = run(ten_k(), document)
    got = values(results)
    assert (got["revenue", QUARTER], got["revenue", FULL_YEAR]) == (
        ("320", RFC),
        ("1200", REV),
    )
    lines = format_report([ten_k()], results).splitlines()
    assert lines[4:6] == [
        "fourth quarter derived (1 10-K): revenue 1, net_income 1",
        f"  TEST {ACCN} revenue: {RFC} (full-year answer {REV})",
    ]


def test_derived_conflicting_values_stop_the_search() -> None:
    document = fiscal_year(
        **{
            REV: {
                USD: [
                    fact(FY, 1200),
                    fact(NINE_MONTHS, 880, accn=Q3_10Q),
                    fact(NINE_MONTHS, 881, accn=Q3_10Q),
                ]
            },
            RFC: {USD: [fact(FY, 1200), fact(NINE_MONTHS, 880, accn=Q3_10Q)]},
        }
    )
    assert missing(run(ten_k(), document)) == {
        ("revenue", QUARTER): (
            f"nine months: {CONFLICTING}",
            f"10-Q {Q3_10Q}: nine months {REV}: values 880, 881",
        )
    }


@pytest.mark.parametrize(
    ("nine_month_noncontrolling", "expected"),
    [(0, ("35", PL)), (2, None)],
)
def test_derived_profit_loss_needs_zero_noncontrolling_in_both(
    nine_month_noncontrolling, expected
) -> None:
    document = fiscal_year(
        **{
            PL: {USD: [fact(FY, 125), fact(NINE_MONTHS, 90, accn=Q3_10Q)]},
            NCI: {
                USD: [
                    fact(FY, 0),
                    fact(NINE_MONTHS, nine_month_noncontrolling, accn=Q3_10Q),
                ]
            },
        }
    )
    del document["facts"]["us-gaap"][NI]
    results = run(ten_k(), document)
    assert values(results).get(("net_income", QUARTER)) == expected
    if expected is None:
        assert missing(results)["net_income", QUARTER] == (
            f"nine months: {CONDITION_NOT_MET}",
            (
                f"10-Q {Q3_10Q}: full year {NI}: no USD facts in company facts; "
                f"full year {AVAIL}: no USD facts in company facts; nine months "
                f"{PL}: 90, but {NCI} is 2 for that period"
            ),
        )


def test_no_third_quarter_10q() -> None:
    none = ThirdQuarter(None, None, NO_THIRD_QUARTER, "no 10-Q's period ends ...")
    results = run(ten_k(), fiscal_year(), third_quarter=none)
    assert set(values(results)) == {(f, FULL_YEAR) for f in FIELD_NAMES}
    assert missing(results) == {
        ("revenue", QUARTER): (NO_THIRD_QUARTER, "no 10-Q's period ends ..."),
        ("net_income", QUARTER): (NO_THIRD_QUARTER, "no 10-Q's period ends ..."),
    }


def test_10k_release_needs_its_third_quarter() -> None:
    index = index_facts([fiscal_year()], {ACCN})
    with pytest.raises(ValueError, match="a 10-K release needs its third quarter"):
        answer_release(ten_k(), index)


def filings(*rows: tuple[str, str, str]) -> pd.DataFrame:
    """A submissions table of (accession, form, report date or "") rows."""
    return parse_submissions(
        [
            {
                "accessionNumber": [r[0] for r in rows],
                "form": [r[1] for r in rows],
                "filingDate": ["2026-02-01"] * len(rows),
                "reportDate": [r[2] for r in rows],
                "items": [""] * len(rows),
                "primaryDocument": ["x.htm"] * len(rows),
            }
        ]
    )


def test_find_third_quarter() -> None:
    table = filings(
        ("a", "10-Q", "2025-09-30"),  # third quarter
        ("b", "10-Q", "2026-09-30"),  # next year's, with this year as comparative
        ("c", "10-Q", "2025-06-30"),
        ("d", "10-Q/A", "2025-09-30"),
        ("e", "10-K", "2025-12-31"),
        ("f", "8-K", ""),
    )
    assert find_third_quarter(table, date(2025, 12, 31)) == ThirdQuarter(
        "a", date(2025, 9, 30)
    )


@pytest.mark.parametrize(
    ("report_date", "found"),
    [
        ("2025-06-28", True),  # 13-week fourth quarter, ends 2025-09-27
        ("2025-06-21", True),  # 14 weeks
        ("2025-06-18", True),  # 101 days before: a 100-day quarter
        ("2025-06-17", False),
        ("2025-07-08", True),  # 81 days before: an 80-day quarter
        ("2025-07-09", False),
    ],
)
def test_third_quarter_leaves_a_quarter(report_date, found) -> None:
    got = find_third_quarter(
        filings(("a", "10-Q", report_date)), date(2025, 9, 27)
    ).accession
    assert got == ("a" if found else None)


def test_no_or_several_third_quarters() -> None:
    end = date(2025, 12, 31)
    assert find_third_quarter(filings(("c", "10-Q", "2025-06-30")), end) == (
        ThirdQuarter(
            None,
            None,
            NO_THIRD_QUARTER,
            "no 10-Q's period ends a quarter before 2025-12-31",
        )
    )
    several = filings(("b", "10-Q", "2025-09-27"), ("a", "10-Q", "2025-09-30"))
    assert find_third_quarter(several, end) == ThirdQuarter(
        None,
        None,
        SEVERAL_THIRD_QUARTERS,
        "a (ends 2025-09-30), b (ends 2025-09-27)",
    )


def test_comparatives_in_the_same_filing_are_not_answers() -> None:
    # Only last year's quarter is tagged: the filing's own period is missing.
    document = second_quarter(**{NI: {USD: [fact(Q2_PRIOR, 25), fact(H1, 52)]}})
    assert missing(run(linked(), document))[("net_income", QUARTER)] == (
        NO_PERIOD,
        (
            f"{NI}: this filing has 2026-01-01..2026-06-30 (180 days), "
            "2025-04-01..2025-06-30 (90 days)" + NO_NI_FALLBACKS
        ),
    )


def test_facts_from_other_filings_are_not_used() -> None:
    # A later filing reports the same period (as a comparative); only the linked
    # filing's own facts count.
    document = second_quarter(
        **{NI: {USD: [fact(Q2, 31, accn=LATER), fact(H1, 52, accn=LATER)]}}
    )
    results = run(linked(), document)
    assert missing(results)[("net_income", QUARTER)] == (
        NO_FACT,
        f"{NI}: none from this filing" + NO_NI_FALLBACKS,
    )
    assert ("revenue", QUARTER) in values(results)


def test_linked_filing_wins_over_another_filings_value() -> None:
    document = second_quarter(
        **{REV: {USD: [fact(Q2, 300), fact(Q2, 305, accn=LATER), fact(H1, 550)]}}
    )
    assert values(run(linked(), document))[("revenue", QUARTER)] == ("300", REV)


# Tags


def test_revenues_comes_before_revenue_from_contracts() -> None:
    document = second_quarter(**{RFC: {USD: [fact(Q2, 290), fact(H1, 530)]}})
    results = run(linked(), document)
    assert values(results)[("revenue", QUARTER)] == ("300", REV)
    assert values(results)[("revenue", YEAR_TO_DATE)] == ("550", REV)


def test_revenue_falls_back_to_revenue_from_contracts() -> None:
    document = second_quarter(**{RFC: {USD: [fact(Q2, 290), fact(H1, 530)]}})
    del document["facts"]["us-gaap"][REV]
    assert values(run(linked(), document))[("revenue", QUARTER)] == ("290", RFC)


def test_fallback_is_per_period() -> None:
    # Revenues is in the filing but only for last year; RevenueFrom... has this
    # quarter.
    document = second_quarter(
        **{
            REV: {USD: [fact(Q2_PRIOR, 280), fact(H1, 550)]},
            RFC: {USD: [fact(Q2, 290)]},
        }
    )
    results = run(linked(), document)
    assert values(results)[("revenue", QUARTER)] == ("290", RFC)
    assert values(results)[("revenue", YEAR_TO_DATE)] == ("550", REV)


def test_eps_falls_back_to_basic_and_diluted() -> None:
    document = second_quarter(**{EPS_BD: {PER_SHARE: [fact(Q2, 0.42), fact(H1, 0.8)]}})
    del document["facts"]["us-gaap"][EPS]
    results = run(linked(), document)
    assert values(results)[("eps_diluted", QUARTER)] == ("0.42", EPS_BD)
    assert values(results)[("eps_diluted", YEAR_TO_DATE)] == ("0.8", EPS_BD)


def without_net_income_loss(**tags: dict[str, list[dict]]) -> dict:
    """A second-quarter 10-Q with the given tags and no NetIncomeLoss."""
    document = second_quarter(**tags)
    del document["facts"]["us-gaap"][NI]
    return document


def test_net_income_falls_back_to_available_to_common() -> None:
    # Caterpillar tags its bottom line this way and never as NetIncomeLoss.
    document = without_net_income_loss(**{AVAIL: {USD: [fact(Q2, 29), fact(H1, 50)]}})
    results = run(linked(), document)
    assert values(results)[("net_income", QUARTER)] == ("29", AVAIL)
    assert values(results)[("net_income", YEAR_TO_DATE)] == ("50", AVAIL)


def test_net_income_loss_comes_first() -> None:
    # Available to common is after preferred dividends (Oracle from 2026).
    document = second_quarter(**{AVAIL: {USD: [fact(Q2, 29), fact(H1, 50)]}})
    assert values(run(linked(), document))[("net_income", QUARTER)] == ("30", NI)


def test_available_to_common_comes_before_profit_loss() -> None:
    document = without_net_income_loss(
        **{
            AVAIL: {USD: [fact(Q2, 29)]},
            PL: {USD: [fact(Q2, 30)]},
            NCI: {USD: [fact(Q2, 0)]},
        }
    )
    assert values(run(linked(), document))[("net_income", QUARTER)] == ("29", AVAIL)


def test_profit_loss_when_noncontrolling_interest_is_zero() -> None:
    document = without_net_income_loss(
        **{
            PL: {USD: [fact(Q2, 30), fact(H1, 52)]},
            NCI: {USD: [fact(Q2, 0), fact(H1, 0)]},
        }
    )
    results = run(linked(), document)
    assert values(results)[("net_income", QUARTER)] == ("30", PL)
    assert values(results)[("net_income", YEAR_TO_DATE)] == ("52", PL)


@pytest.mark.parametrize(
    ("noncontrolling", "why"),
    [
        ([], "not tagged"),
        ([fact(Q2, 4)], "4"),
        ([fact(Q2, -2)], "-2"),
        ([fact(H1, 0)], "not tagged"),  # zero for another period
        ([fact(Q2_PRIOR, 0)], "not tagged"),
        ([fact(Q2, 0, accn=LATER)], "not tagged"),  # zero in another filing
        ([fact(Q2, 0), fact(Q2, 3)], "0, 3"),
    ],
)
def test_profit_loss_needs_zero_for_the_same_filing_and_period(
    noncontrolling, why
) -> None:
    tags = {PL: {USD: [fact(Q2, 30), fact(H1, 52)]}}
    if noncontrolling:
        tags[NCI] = {USD: noncontrolling}
    results = run(linked(), without_net_income_loss(**tags))
    assert missing(results)[("net_income", QUARTER)] == (
        CONDITION_NOT_MET,
        (
            f"{NI}: no USD facts in company facts; "
            f"{AVAIL}: no USD facts in company facts; "
            f"{PL}: 30, but {NCI} is {why} for that period"
        ),
    )


def test_profit_loss_condition_is_per_period() -> None:
    document = without_net_income_loss(
        **{PL: {USD: [fact(Q2, 30), fact(H1, 52)]}, NCI: {USD: [fact(H1, 0)]}}
    )
    results = run(linked(), document)
    assert ("net_income", QUARTER) in missing(results)
    assert values(results)[("net_income", YEAR_TO_DATE)] == ("52", PL)


# Preferred dividends: available to common is after them (Decisions 2026-10-11)


@pytest.mark.parametrize(
    "tag",
    [
        "PreferredStockDividendsIncomeStatementImpact",
        "DividendsPreferredStock",
        "DividendsPreferredStockCash",
        "PreferredStockDividendsAndOtherAdjustments",
    ],
)
def test_available_to_common_is_not_used_with_preferred_dividends(tag) -> None:
    document = without_net_income_loss(
        **{AVAIL: {USD: [fact(Q2, 29), fact(H1, 50)]}, tag: {USD: [fact(Q2, 1)]}}
    )
    results = run(linked(), document)
    assert missing(results)[("net_income", QUARTER)] == (
        CONDITION_NOT_MET,
        (
            f"{NI}: no USD facts in company facts; {AVAIL}: 29, after preferred "
            f"dividends: {tag} 2026-04-01..2026-06-30 1; {PL}: no USD facts in "
            "company facts"
        ),
    )
    # The half year has none tagged, so it keeps the fallback.
    assert values(results)[("net_income", YEAR_TO_DATE)] == ("50", AVAIL)
    assert [
        (d.period_type, d.tag, str(d.value)) for d in results.preferred_dividends
    ] == [(QUARTER, tag, "1")]


@pytest.mark.parametrize(
    ("dividends", "used"),
    [
        ([fact(Q2, 0)], True),  # tagged zero: none
        ([fact(Q2_PRIOR, 3)], True),  # last year's quarter
        ([fact(Q2, 3, accn=LATER)], True),  # in another filing
        ([fact(H1, 3)], False),  # the half year contains the quarter
        ([fact(Q2, -2)], False),  # nonzero either way
    ],
)
def test_which_preferred_dividends_count(dividends, used) -> None:
    document = without_net_income_loss(
        **{AVAIL: {USD: [fact(Q2, 29)]}, PSD: {USD: dividends}}
    )
    found = values(run(linked(), document)).get(("net_income", QUARTER))
    assert found == (("29", AVAIL) if used else None)


def test_preferred_dividends_fall_through_to_profit_loss() -> None:
    # ProfitLoss with noncontrolling interest 0 is net income before preferred
    # dividends.
    document = without_net_income_loss(
        **{
            AVAIL: {USD: [fact(Q2, 29)]},
            PSD: {USD: [fact(Q2, 1)]},
            PL: {USD: [fact(Q2, 30)]},
            NCI: {USD: [fact(Q2, 0)]},
        }
    )
    assert values(run(linked(), document))[("net_income", QUARTER)] == ("30", PL)


def test_net_income_loss_needs_no_check_for_preferred_dividends() -> None:
    # Oracle: NetIncomeLoss is before preferred dividends.
    document = second_quarter(**{PSD: {USD: [fact(Q2, 22)]}})
    results = run(linked(), document)
    assert values(results)[("net_income", QUARTER)] == ("30", NI)
    assert [d.period_type for d in results.preferred_dividends] == [QUARTER]


@pytest.mark.parametrize("where", ["full year", "nine months"])
def test_derived_available_to_common_needs_no_preferred_dividends_in_both(
    where,
) -> None:
    dividends = (
        fact(FY, 4) if where == "full year" else fact(NINE_MONTHS, 3, accn=Q3_10Q)
    )
    document = fiscal_year(
        **{
            AVAIL: {USD: [fact(FY, 120), fact(NINE_MONTHS, 88, accn=Q3_10Q)]},
            PSD: {USD: [dividends]},
        }
    )
    del document["facts"]["us-gaap"][NI]
    results = run(ten_k(), document)
    assert ("net_income", QUARTER) not in values(results)
    reason, detail = missing(results)["net_income", QUARTER]
    assert reason == f"{where}: {CONDITION_NOT_MET}"
    assert "after preferred dividends" in detail


def test_derived_available_to_common_without_preferred_dividends() -> None:
    document = fiscal_year(
        **{
            AVAIL: {USD: [fact(FY, 120), fact(NINE_MONTHS, 88, accn=Q3_10Q)]},
            PSD: {USD: [fact(FY, 0), fact(NINE_MONTHS, 0, accn=Q3_10Q)]},
        }
    )
    del document["facts"]["us-gaap"][NI]
    assert values(run(ten_k(), document))[("net_income", QUARTER)] == ("32", AVAIL)


def test_fallback_checks_mark_available_to_common_unusable_with_dividends() -> None:
    document = second_quarter(
        **{AVAIL: {USD: [fact(Q2, 29), fact(H1, 50)]}, PSD: {USD: [fact(Q2, 1)]}}
    )
    checks = run(linked(), document).fallback_checks
    assert [(c.period_type, c.tag, c.usable) for c in checks] == [
        (QUARTER, AVAIL, False),
        (YEAR_TO_DATE, AVAIL, True),
    ]


def test_preferred_dividends_in_report() -> None:
    release = linked()
    document = second_quarter(**{PSD: {USD: [fact(Q2, 22), fact(H1, 40)]}})
    lines = format_report([release], run(release, document)).splitlines()
    start = lines.index(
        "releases whose linked filing reports nonzero preferred dividends: 1"
    )
    assert lines[start + 1] == (
        f"  TEST 0000000001-26-000009: quarter {PSD} 22, year_to_date {PSD} 40"
    )


def test_tag_in_another_unit_counts_as_no_tag() -> None:
    document = second_quarter(**{NI: {"EUR": [fact(Q2, 30), fact(H1, 52)]}})
    assert missing(run(linked(), document))[("net_income", QUARTER)][0] == NO_TAG


def test_reason_is_the_furthest_any_tag_got() -> None:
    # Revenues has nothing from this filing; RevenueFrom... is in it but for
    # another period.
    document = second_quarter(
        **{
            REV: {USD: [fact(Q2, 300, accn=LATER)]},
            RFC: {USD: [fact(Q2_PRIOR, 280)]},
        }
    )
    assert missing(run(linked(), document))[("revenue", QUARTER)] == (
        NO_PERIOD,
        (
            f"{REV}: none from this filing; "
            f"{RFC}: this filing has 2025-04-01..2025-06-30 (90 days)"
            + NO_REV_FALLBACKS
        ),
    )


# Revenue and EPS tags after the current ones (Decisions 2026-10-28)


@pytest.mark.parametrize("tag", [RUU, RIA])
def test_later_revenue_tags_answer_a_filing_without_current_tags(tag) -> None:
    document = second_quarter(**{REV: {}, tag: {USD: [fact(Q2, 300)]}})
    assert values(run(linked(), document))["revenue", QUARTER] == ("300", tag)


def test_regulated_revenue_comes_before_including_assessed_tax() -> None:
    # American Water, Duke, NextEra: the second is one part of the first.
    document = second_quarter(
        **{REV: {}, RUU: {USD: [fact(Q2, 300)]}, RIA: {USD: [fact(Q2, 260)]}}
    )
    assert values(run(linked(), document))["revenue", QUARTER] == ("300", RUU)


@pytest.mark.parametrize("current", [REV, RFC])
@pytest.mark.parametrize("tag", [RUU, RIA])
def test_later_revenue_tags_need_a_filing_without_current_tags(current, tag) -> None:
    # A current tag anywhere in the filing, here only for the prior year's
    # quarter, rules out the later tags for every period.
    tags = {REV: {}, current: {USD: [fact(Q2_PRIOR, 280)]}, tag: {USD: [fact(Q2, 300)]}}
    reason, detail = missing(run(linked(), second_quarter(**tags)))["revenue", QUARTER]
    assert reason == CONDITION_NOT_MET
    assert f"{tag}: 300, but this filing has {current} facts" in detail


def test_current_revenue_tags_in_another_filing_do_not_count() -> None:
    document = second_quarter(
        **{REV: {USD: [fact(Q2_PRIOR, 280, accn=LATER)]}, RIA: {USD: [fact(Q2, 300)]}}
    )
    assert values(run(linked(), document))["revenue", QUARTER] == ("300", RIA)


def test_derived_revenue_from_later_tag_needs_both_filings_without_current() -> None:
    # The 10-K has no current tag, but the third-quarter 10-Q does.
    ria = {USD: [fact(FY, 1200), fact(NINE_MONTHS, 880, accn=Q3_10Q)]}
    clean = fiscal_year(**{REV: {}, RIA: ria})
    assert values(run(ten_k(), clean))["revenue", QUARTER] == ("320", RIA)
    tainted = fiscal_year(
        **{REV: {USD: [fact(("2024-01-01", "2024-09-30"), 800, accn=Q3_10Q)]}, RIA: ria}
    )
    reason, detail = missing(run(ten_k(), tainted))["revenue", QUARTER]
    assert reason == f"nine months: {CONDITION_NOT_MET}"
    assert f"{RIA}: 880, but this filing has {REV} facts" in detail
    assert values(run(ten_k(), tainted))["revenue", FULL_YEAR] == ("1200", RIA)


def test_continuing_operations_eps_without_discontinued_operations() -> None:
    document = second_quarter(**{EPS: {}, CONT: {PER_SHARE: [fact(Q2, 1.25)]}})
    assert values(run(linked(), document))["eps_diluted", QUARTER] == ("1.25", CONT)


@pytest.mark.parametrize(
    ("tag", "unit", "value"),
    [
        (DISC, USD, -1000000),
        (
            "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
            USD,
            5,
        ),
        (
            "IncomeLossFromDiscontinuedOperationsNetOfTaxPerDilutedShare",
            PER_SHARE,
            0.01,
        ),
    ],
)
def test_continuing_operations_eps_not_with_discontinued_operations(
    tag, unit, value
) -> None:
    document = second_quarter(
        **{EPS: {}, CONT: {PER_SHARE: [fact(Q2, 1.25)]}, tag: {unit: [fact(Q2, value)]}}
    )
    reason, detail = missing(run(linked(), document))["eps_diluted", QUARTER]
    assert reason == CONDITION_NOT_MET
    assert f"{CONT}: 1.25, but discontinued operations: {tag} " in detail


@pytest.mark.parametrize(
    "discontinued",
    [
        [fact(Q2, 0)],  # zero
        [fact(H1, -1000000)],  # another period: the half year, not the quarter
        [fact(Q2, -1000000, accn=LATER)],  # another filing
    ],
    ids=["zero", "other-period", "other-filing"],
)
def test_which_discontinued_operations_count(discontinued) -> None:
    document = second_quarter(
        **{EPS: {}, CONT: {PER_SHARE: [fact(Q2, 1.25)]}, DISC: {USD: discontinued}}
    )
    assert values(run(linked(), document))["eps_diluted", QUARTER] == ("1.25", CONT)


def test_diluted_eps_needs_no_check_for_discontinued_operations() -> None:
    document = second_quarter(**{DISC: {USD: [fact(Q2, -1000000)]}})
    assert values(run(linked(), document))["eps_diluted", QUARTER][1] == EPS


# Revenue components (Decisions 2026-10-28)

LEASE = "OperatingLeaseLeaseIncome"


@pytest.mark.parametrize(
    ("larger", "dropped", "listed"),
    [
        (Decimal(300), True, False),  # 5x: a component
        (Decimal("299.99"), False, True),  # just under 5x
        (Decimal(120), False, True),  # 2x
        (Decimal("119.99"), False, False),  # just under 2x
    ],
)
def test_revenue_component_is_dropped(larger, dropped, listed) -> None:
    # Camden: RevenueFromContractWithCustomerExcludingAssessedTax 1,077K beside
    # OperatingLeaseLeaseIncome 390,778K.
    document = second_quarter(
        **{
            REV: {USD: [fact(Q2, 60), fact(H1, 550)]},
            LEASE: {USD: [fact(Q2, larger)]},
        }
    )
    results = run(linked(), document)
    if dropped:
        assert missing(results)["revenue", QUARTER] == (
            LOOKS_LIKE_COMPONENT,
            # A drop ends the search: no later tag is tried.
            f"{REV}: 60, but {LEASE} is 300 (5.0x)",
        )
    else:
        assert values(results)["revenue", QUARTER] == ("60", REV)
    assert values(results)["revenue", YEAR_TO_DATE] == ("550", REV)  # not compared
    checks = [(c.period_type, c.tag, c.larger_tag) for c in results.revenue_checks]
    assert checks == ([(QUARTER, REV, LEASE)] if listed else [])


def test_component_is_not_replaced_by_a_later_tag() -> None:
    # General Mills FY2024: Revenues 2,037.8M beside net sales tagged
    # RevenueFromContractWithCustomerExcludingAssessedTax, 19,857.2M.
    document = second_quarter(
        **{REV: {USD: [fact(Q2, 60), fact(H1, 550)]}, RFC: {USD: [fact(Q2, 300)]}}
    )
    assert missing(run(linked(), document))["revenue", QUARTER] == (
        LOOKS_LIKE_COMPONENT,
        f"{REV}: 60, but {RFC} is 300 (5.0x)",
    )


@pytest.mark.parametrize(
    "larger",
    [
        {LEASE: {USD: [fact(Q2_PRIOR, 300)]}},  # another period
        {LEASE: {USD: [fact(Q2, 300, accn=LATER)]}},  # another filing
        {"PaymentsToAcquireAvailableForSaleSecuritiesDebt": {USD: [fact(Q2, 300)]}},
        {"RevenueRemainingPerformanceObligation": {USD: [fact("2026-06-30", 900)]}},
        {REV: {USD: [fact(Q2, 60)]}, "Revenues2": {"EUR": [fact(Q2, 300)]}},
    ],
    ids=["other-period", "other-filing", "not-revenue-like", "instant", "not-usd"],
)
def test_which_facts_count_against_a_revenue_answer(larger) -> None:
    document = second_quarter(**({REV: {USD: [fact(Q2, 60)]}} | larger))
    results = run(linked(), document)
    assert values(results)["revenue", QUARTER] == ("60", REV)
    assert results.revenue_checks == []


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        (REV, True),
        (RFC, True),
        (RUU, True),
        (RIA, True),
        ("SalesRevenueNet", True),
        ("RevenueFromRelatedParties", True),
        (LEASE, True),
        ("LeaseIncome", True),
        ("CostOfRevenue", False),
        ("ContractWithCustomerLiabilityRevenueRecognized", False),
        ("PaymentsToAcquireAvailableForSaleSecuritiesDebt", False),
        ("InterestIncomeOperating", False),
    ],
)
def test_revenue_like_tags(tag, expected) -> None:
    assert is_revenue_like(tag) is expected


def test_only_positive_revenue_can_be_a_component() -> None:
    document = second_quarter(
        **{REV: {USD: [fact(Q2, 0)]}, LEASE: {USD: [fact(Q2, 300)]}}
    )
    assert values(run(linked(), document))["revenue", QUARTER] == ("0", REV)


def test_net_income_and_eps_are_not_checked_for_components() -> None:
    # Net income is a tenth of revenue in second_quarter.
    results = run(linked(), second_quarter())
    assert values(results)["net_income", QUARTER] == ("30", NI)
    assert values(results)["eps_diluted", QUARTER] == ("1.57", EPS)


def test_derived_quarter_with_a_component_full_year_is_dropped() -> None:
    document = fiscal_year(LeaseIncome={USD: [fact(FY, 6000)]})
    results = run(ten_k(), document)
    reasons = {key: reason for key, (reason, _) in missing(results).items()}
    assert reasons == {
        ("revenue", QUARTER): f"full year: {LOOKS_LIKE_COMPONENT}",
        ("revenue", FULL_YEAR): LOOKS_LIKE_COMPONENT,
    }
    assert (
        f"full year {REV}: 1200, but LeaseIncome is 6000 (5.0x)"
        in (missing(results)["revenue", QUARTER][1])
    )


def test_derived_quarter_with_a_component_nine_months_is_dropped() -> None:
    document = fiscal_year(LeaseIncome={USD: [fact(NINE_MONTHS, 4400, accn=Q3_10Q)]})
    results = run(ten_k(), document)
    reason, detail = missing(results)["revenue", QUARTER]
    assert reason == f"nine months: {LOOKS_LIKE_COMPONENT}"
    assert f"nine months {REV}: 880, but LeaseIncome is 4400 (5.0x)" in detail
    assert values(results)["revenue", FULL_YEAR] == ("1200", REV)


def test_derived_quarter_near_a_component_is_listed() -> None:
    document = fiscal_year(LeaseIncome={USD: [fact(NINE_MONTHS, 1760, accn=Q3_10Q)]})
    results = run(ten_k(), document)
    assert values(results)["revenue", QUARTER] == ("320", REV)
    assert results.revenue_checks == [
        RevenueCheck(
            ticker="TEST",
            release_accession="0000000001-26-000009",
            period_type=QUARTER,
            derived=True,
            answer=Decimal(320),
            accession=Q3_10Q,
            tag=REV,
            start=date(2025, 1, 1),
            end=date(2025, 9, 30),
            value=Decimal(880),
            larger_tag="LeaseIncome",
            larger_value=Decimal(1760),
        )
    ]


def test_near_components_in_report() -> None:
    document = second_quarter(
        **{REV: {USD: [fact(Q2, 100)]}, LEASE: {USD: [fact(Q2, 300)]}}
    )
    release = linked()
    report = format_report([release], run(release, document)).splitlines()
    start = report.index(
        "revenue answers between 1/5 and 1/2 of a larger revenue-like fact (listed, "
        "not dropped): 1"
    )
    assert report[start + 1] == (
        f"  TEST 0000000001-26-000009 quarter {REV} 100; {LEASE} 300"
    )


# Implausible EPS (Decisions 2026-10-28)


@pytest.mark.parametrize(
    ("value", "kept"),
    [
        (Decimal(790000), False),  # Halliburton's $0.79
        (Decimal(1000), False),
        (Decimal(-1000), False),
        (Decimal("999.99"), True),
        (Decimal("-999.99"), True),
        (Decimal("506.69"), True),  # NVR's full year
    ],
)
def test_eps_of_1000_or_more_is_dropped(value, kept) -> None:
    document = second_quarter(**{EPS: {PER_SHARE: [fact(Q2, value), fact(H1, 2.84)]}})
    results = run(linked(), document)
    if kept:
        assert values(results)["eps_diluted", QUARTER] == (str(value), EPS)
    else:
        assert missing(results)["eps_diluted", QUARTER] == (
            IMPLAUSIBLE_EPS,
            f"{EPS}: {value}, 1,000 or more per share",
        )
    assert values(results)["eps_diluted", YEAR_TO_DATE] == ("2.84", EPS)


def test_implausible_eps_is_not_replaced_or_rescaled() -> None:
    document = second_quarter(
        **{
            EPS: {PER_SHARE: [fact(Q2, 790000)]},
            EPS_BD: {PER_SHARE: [fact(Q2, 0.79)]},
            CONT: {PER_SHARE: [fact(Q2, 0.79)]},
        }
    )
    results = run(linked(), document)
    assert missing(results)["eps_diluted", QUARTER] == (
        IMPLAUSIBLE_EPS,
        f"{EPS}: 790000, 1,000 or more per share",
    )
    assert ("eps_diluted", QUARTER) not in values(results)


def test_large_revenue_and_net_income_are_not_implausible() -> None:
    document = second_quarter(
        **{REV: {USD: [fact(Q2, 10**12)]}, NI: {USD: [fact(Q2, -(10**11))]}}
    )
    results = run(linked(), document)
    assert values(results)["revenue", QUARTER] == (str(10**12), REV)
    assert values(results)["net_income", QUARTER] == (str(-(10**11)), NI)


def test_period_detail_is_capped() -> None:
    periods = [(f"20{y}-04-01", f"20{y}-06-30") for y in range(19, 25)]
    document = second_quarter(**{NI: {USD: [fact(p, 1) for p in periods]}})
    detail = missing(run(linked(), document))[("net_income", QUARTER)][1]
    assert detail == (
        f"{NI}: this filing has 2024-04-01..2024-06-30 (90 days), "
        "2023-04-01..2023-06-30 (90 days), 2022-04-01..2022-06-30 (90 days), "
        "2021-04-01..2021-06-30 (90 days) and 2 more" + NO_NI_FALLBACKS
    )


def test_instants_are_described() -> None:
    document = second_quarter(**{NI: {USD: [fact("2026-06-30", 5)]}})
    assert missing(run(linked(), document))[("net_income", QUARTER)][1] == (
        f"{NI}: this filing has 2026-06-30 (instant)" + NO_NI_FALLBACKS
    )


def test_conflicting_values_stop_the_search() -> None:
    document = second_quarter(
        **{
            REV: {USD: [fact(Q2, 300), fact(Q2, 301), fact(H1, 550)]},
            RFC: {USD: [fact(Q2, 290)]},
        }
    )
    results = run(linked(), document)
    assert missing(results)[("revenue", QUARTER)] == (
        CONFLICTING,
        f"{REV}: values 300, 301",
    )
    assert values(results)[("revenue", YEAR_TO_DATE)] == ("550", REV)


def test_equal_duplicates_are_one_value() -> None:
    document = second_quarter(
        **{EPS: {PER_SHARE: [fact(Q2, 1.57), fact(Q2, 1.570), fact(H1, 2.84)]}}
    )
    assert values(run(linked(), document))[("eps_diluted", QUARTER)] == ("1.57", EPS)


def test_facts_merge_across_ciks() -> None:
    # ExxonMobil: the joint 10-Q's facts are under the new holding company only.
    old = companyfacts({REV: {USD: [fact(Q2_PRIOR, 280, accn=LATER)]}}, cik=34088)
    new = second_quarter()
    results = run(linked(), new, old)
    assert len(results.answers) == 6
    assert results.missing == []


def test_same_fact_under_two_ciks_is_kept_once() -> None:
    results = run(linked(), second_quarter(), second_quarter())
    assert len(results.answers) == 6
    assert results.missing == []


def test_tag_known_from_another_cik_is_no_fact() -> None:
    old = companyfacts({NI: {USD: [fact(Q2_PRIOR, 25, accn=LATER)]}}, cik=34088)
    new = second_quarter()
    del new["facts"]["us-gaap"][NI]
    assert missing(run(linked(), new, old))[("net_income", QUARTER)][0] == NO_FACT


def test_filing_missing_from_company_facts_suggests_refresh() -> None:
    document = second_quarter()
    results = run(linked(accn="0000000001-26-000099"), document)
    assert results.answers == []
    assert set(missing(results)) == {(name, QUARTER) for name in FIELD_NAMES}
    reason, detail = missing(results)[("revenue", QUARTER)]
    assert reason == NO_FACT
    assert detail.endswith(
        "company facts have nothing from this filing (downloaded before it was "
        "filed? try --refresh)"
    )


def test_float_values_are_refused() -> None:
    document = second_quarter()
    document["facts"]["us-gaap"][EPS]["units"][PER_SHARE][0]["val"] = 1.57
    with pytest.raises(TypeError, match="parse_float=Decimal"):
        index_facts([document], {ACCN})


def test_other_tags_and_taxonomies_are_ignored() -> None:
    document = second_quarter()
    document["facts"]["ifrs-full"] = {
        "Revenue": {"label": "Revenue", "units": {USD: [fact(Q2, 999), fact(H1, 999)]}}
    }
    del document["facts"]["us-gaap"][REV]
    assert missing(run(linked(), document))[("revenue", QUARTER)][0] == NO_TAG


# Other revenue tags


def test_other_revenue_tags_reported() -> None:
    document = second_quarter(
        **{
            RFC: {USD: [fact(Q2, 290), fact(H1, 530)]},
            "SalesRevenueNet": {USD: [fact(Q2, 300)]},
            "RevenueFromRelatedParties": {USD: [fact(Q2_PRIOR, 3)]},  # wrong period
            "RevenueRemainingPerformanceObligation": {USD: [fact("2026-06-30", 900)]},
            "CostOfRevenue": {USD: [fact(Q2, 150)]},
            "ContractWithCustomerLiabilityRevenueRecognized": {USD: [fact(Q2, 40)]},
        }
    )
    others = run(linked(), document).other_revenue
    assert [(o.tag, str(o.value), str(o.answer)) for o in others] == [
        (RFC, "290", "300"),
        ("SalesRevenueNet", "300", "300"),
    ]


def test_other_revenue_tags_without_an_answer() -> None:
    document = second_quarter(SalesRevenueNet={USD: [fact(Q2, 300)]})
    del document["facts"]["us-gaap"][REV]
    others = run(linked(), document).other_revenue
    assert [(o.tag, o.answer) for o in others] == [("SalesRevenueNet", None)]


def test_other_revenue_tags_use_the_headline_period() -> None:
    document = companyfacts(
        {
            REV: {USD: [fact(FY, 1200)]},
            RFC: {USD: [fact(FY, 1150), fact(Q4, 300)]},
        }
    )
    others = run(linked(form="10-K", end="2025-12-31"), document).other_revenue
    assert [(o.tag, str(o.value)) for o in others] == [(RFC, "1150")]


# Net income fallbacks alongside NetIncomeLoss


def test_fallback_checks_compare_with_net_income_loss() -> None:
    document = second_quarter(
        **{
            AVAIL: {USD: [fact(Q2, 30), fact(H1, 50)]},
            PL: {USD: [fact(Q2, 31)]},
            NCI: {USD: [fact(Q2, 1)]},
        }
    )
    checks = run(linked(), document).fallback_checks
    assert [
        (c.period_type, c.tag, str(c.value), str(c.net_income_loss), c.usable)
        for c in checks
    ] == [
        (QUARTER, AVAIL, "30", "30", True),
        (QUARTER, PL, "31", "30", False),
        (YEAR_TO_DATE, AVAIL, "50", "52", True),
    ]


def test_fallback_checks_mark_profit_loss_usable_when_zero() -> None:
    document = second_quarter(**{PL: {USD: [fact(Q2, 30)]}, NCI: {USD: [fact(Q2, 0)]}})
    checks = run(linked(), document).fallback_checks
    assert [(c.tag, c.usable, c.value == c.net_income_loss) for c in checks] == [
        (PL, True, True)
    ]


def test_no_fallback_checks_without_net_income_loss() -> None:
    document = without_net_income_loss(**{AVAIL: {USD: [fact(Q2, 29)]}})
    assert run(linked(), document).fallback_checks == []


def test_fallback_checks_in_report() -> None:
    release = linked()
    document = second_quarter(
        **{
            AVAIL: {USD: [fact(Q2, 30), fact(H1, 50)]},
            PL: {USD: [fact(Q2, 31)]},
            NCI: {USD: [fact(Q2, 1)]},
        }
    )
    lines = format_report([release], run(release, document)).splitlines()
    start = lines.index(
        "net income fallbacks where NetIncomeLoss is also tagged "
        "(same filing and period):"
    )
    assert lines[start + 1 :] == [
        (
            f"  TEST {AVAIL}: 2 periods, 1 agree; no preferred dividends in 2 "
            f"(e.g. {ACCN} year_to_date: 50, NetIncomeLoss 52)"
        ),
        (
            f"  TEST {PL}: 1 period, 0 agree; {NCI} tagged 0 in 0 "
            f"(e.g. {ACCN} quarter: 31, NetIncomeLoss 30)"
        ),
    ]


def test_report_without_fallback_checks() -> None:
    release = linked()
    lines = format_report([release], run(release, second_quarter())).splitlines()
    assert lines[-2:] == [
        (
            "net income fallbacks where NetIncomeLoss is also tagged "
            "(same filing and period):"
        ),
        "  none",
    ]


# Values


def make_client(
    cache_dir: Path, routes: dict[str, bytes], requests: list[str] | None = None
):
    """A client that serves routes (URL -> body); other URLs 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(str(request.url))
        body = routes.get(str(request.url))
        return (
            httpx.Response(404) if body is None else httpx.Response(200, content=body)
        )

    return EdgarClient(
        UA,
        cache_dir,
        transport=httpx.MockTransport(handler),
        limiter=RateLimiter(8.0, clock=lambda: 0.0, sleep=lambda s: None),
    )


def fixture_routes() -> dict[str, bytes]:
    """link_submissions.json and answers_companyfacts.json as served: URL -> JSON
    body."""
    routes = json.loads(SUBMISSIONS.read_bytes()) | json.loads(
        COMPANYFACTS.read_bytes()
    )
    return {url: json.dumps(body).encode() for url, body in routes.items()}


def test_values_are_parsed_exactly(tmp_path) -> None:
    url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    body = (
        b'{"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": '
        b'[{"start": "2026-04-01", "end": "2026-06-30", "val": 0.1, "accn": "'
        + ACCN.encode()
        + b'"}]}}, "NetIncomeLoss": {"units": {"USD": [{"start": "2026-04-01", '
        b'"end": "2026-06-30", "val": 94036000000, "accn": "'
        + ACCN.encode()
        + b'"}]}}}}}'
    )
    client = make_client(tmp_path, {url: body})
    document = load_company_facts(client, 1)
    eps = document["facts"]["us-gaap"][EPS]["units"][PER_SHARE][0]["val"]
    assert eps == Decimal("0.1")  # not the float 0.1000000000000000055...
    results = run(linked(end="2026-06-30"), document)
    csv_text = answers_csv(results.answers)
    assert csv_text.splitlines() == [
        ",".join(COLUMNS),
        (
            "TEST,0000000001-26-000009,0000000001-26-000010,net_income,quarter,"
            "2026-04-01,2026-06-30,94036000000,USD,NetIncomeLoss,False,,,"
        ),
        (
            "TEST,0000000001-26-000009,0000000001-26-000010,eps_diluted,quarter,"
            "2026-04-01,2026-06-30,0.1,USD/shares,EarningsPerShareDiluted,False,,,"
        ),
    ]


# links.csv


def test_load_links_keeps_linked_rows() -> None:
    releases, not_linked = load_links(LINKS)
    assert len(releases) == 60
    assert not_linked == 1  # NKE's CFO announcement
    assert releases[0] == LinkedRelease(
        "AAPL",
        "0000320193-25-000055",
        "10-Q",
        "0000320193-25-000057",
        date(2025, 3, 29),
    )
    assert {r.linked_form for r in releases} == {"10-Q", "10-K"}


def test_load_links_rejects_other_forms(tmp_path) -> None:
    lines = LINKS.read_text().splitlines()
    path = tmp_path / "links.csv"
    path.write_text("\n".join([lines[0], lines[1].replace(",10-Q,", ",10-Q/A,")]))
    with pytest.raises(ValueError, match="linked to a '10-Q/A'"):
        load_links(path)


# Command line


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory) -> tuple[int, str, list[dict[str, str]], Path]:
    """answers.main over answers_links.csv and the company facts fixture: exit
    code, stdout, answers.csv rows and answers.csv path."""
    tmp = tmp_path_factory.mktemp("answers")
    out = tmp / "answers.csv"
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        status = answers.main(
            ["--links", str(LINKS), "--out", str(out)],
            client=make_client(tmp / "raw", fixture_routes()),
        )
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return status, stdout.getvalue(), rows, out


def test_cli_report_on_fixtures(fixture_run) -> None:
    status, stdout, _, out = fixture_run
    assert status == 0
    lines = stdout.splitlines()
    assert lines[0].split() == ["ticker", "releases", *FIELD_NAMES]
    assert [line.split() for line in lines[1:12]] == [
        [ticker, "6", "6", "6", "6"]
        for ticker in [
            "AAPL",
            "CAT",
            "HD",
            "MSFT",
            "NFLX",
            "NKE",
            "NVDA",
            "ORCL",
            "WMT",
            "XOM",
        ]
    ] + [["total", "60", "60", "60", "60"]]
    assert lines[12] == (
        "year to date (29 10-Qs report one): revenue 29, net_income 29, eps_diluted 29"
    )
    assert lines[13:16] == [
        "fourth quarter derived (13 10-Ks): revenue 13, net_income 13",
        f"  ORCL 0000950170-25-087926 revenue: {RFC} (full-year answer {REV})",
        f"  ORCL 0001193125-26-277521 revenue: {RFC} (full-year answer {REV})",
    ]
    assert lines[16] == "missing: 0"
    assert lines[17:] == [
        (
            "revenue answers between 1/5 and 1/2 of a larger revenue-like fact "
            "(listed, not dropped): 0"
        ),
        "other revenue tags at the headline period (reported, not used):",
        f"  ORCL {RFC}: 2 filings, 2 equal to the revenue answer",
        (
            f"  WMT {RFC}: 6 filings, 0 equal to the revenue answer "
            "(e.g. 0000104169-25-000090: 163981000000, answer 165609000000)"
        ),
        # Oracle's from its Q3 FY26 release (Findings 2026-10-14).
        "releases whose linked filing reports nonzero preferred dividends: 3",
        (
            f"  ORCL 0001193125-26-100148: quarter {PSD} 22000000, year_to_date "
            f"{PSD} 22000000"
        ),
        (
            "  ORCL 0001193125-26-265848: full_year DividendsPreferredStockCash "
            f"103000000, full_year {PSD} 103000000"
        ),
        f"  ORCL 0001193125-26-387905: quarter {PSD} 81000000",
        (
            "net income fallbacks where NetIncomeLoss is also tagged "
            "(same filing and period):"
        ),
        (
            f"  ORCL {AVAIL}: 4 periods, 0 agree; no preferred dividends in 0 "
            "(e.g. 0001193125-26-101045 quarter: 3699000000, NetIncomeLoss 3721000000)"
        ),
        (
            f"  ORCL {PL}: 2 periods, 0 agree; {NCI} tagged 0 in 0 "
            "(e.g. 0000950170-25-087926 full_year: 12627000000, "
            "NetIncomeLoss 12443000000)"
        ),
        (
            f"  WMT {PL}: 9 periods, 0 agree; {NCI} tagged 0 in 0 "
            "(e.g. 0000104169-25-000090 quarter: 4639000000, NetIncomeLoss 4487000000)"
        ),
        (
            f"  XOM {PL}: 9 periods, 0 agree; {NCI} tagged 0 in 0 "
            "(e.g. 0000034088-25-000024 quarter: 8033000000, NetIncomeLoss 7713000000)"
        ),
        f"skipped 1 release in {LINKS} not linked",
        f"wrote 293 rows to {out}",
    ]


def test_cli_answers_csv_on_fixtures(fixture_run) -> None:
    _, _, rows, _ = fixture_run
    assert list(rows[0]) == COLUMNS
    assert len(rows) == 293
    apple_q3 = [r for r in rows if r["linked_accession"] == "0000320193-25-000073"]
    assert [
        (r["field"], r["period_type"], r["start"], r["end"], r["value"], r["tag"])
        for r in apple_q3
    ] == [
        ("revenue", "quarter", "2025-03-30", "2025-06-28", "94036000000", RFC),
        ("revenue", "year_to_date", "2024-09-29", "2025-06-28", "313695000000", RFC),
        ("net_income", "quarter", "2025-03-30", "2025-06-28", "23434000000", NI),
        ("net_income", "year_to_date", "2024-09-29", "2025-06-28", "84544000000", NI),
        ("eps_diluted", "quarter", "2025-03-30", "2025-06-28", "1.57", EPS),
        ("eps_diluted", "year_to_date", "2024-09-29", "2025-06-28", "5.62", EPS),
    ]
    # ExxonMobil's joint 10-Q, filed under the old CIK, has its XBRL under the new.
    xom = [r for r in rows if r["linked_accession"] == "0000034088-26-000093"]
    assert xom[0] == {
        "ticker": "XOM",
        "release_accession": "0002115436-26-000006",
        "linked_accession": "0000034088-26-000093",
        "field": "revenue",
        "period_type": "quarter",
        "start": "2026-04-01",
        "end": "2026-06-30",
        "value": "116017000000",
        "unit": "USD",
        "tag": REV,
        "derived": "False",
        "full_year_accession": "",
        "nine_months_accession": "",
        "fiscal_year_start": "",
    }
    assert len(xom) == 6
    full_year = [r for r in rows if r["period_type"] == "full_year"]
    assert len({r["linked_accession"] for r in full_year}) == 13
    assert {r["unit"] for r in rows if r["field"] == "eps_diluted"} == {PER_SHARE}
    # Caterpillar never tags NetIncomeLoss in its 10-Qs and 10-K.
    cat = {
        (r["linked_accession"], r["period_type"]): (r["value"], r["tag"])
        for r in rows
        if r["ticker"] == "CAT" and r["field"] == "net_income"
    }
    assert len(cat) == 10  # with the 10-K's derived fourth quarter
    assert {tag for _, tag in cat.values()} == {AVAIL}
    assert cat["0000018230-25-000040", "quarter"] == ("2179000000", AVAIL)
    assert cat["0000018230-26-000008", "quarter"] == ("2402000000", AVAIL)
    net_income_tags = [r["tag"] for r in rows if r["field"] == "net_income"]
    assert (net_income_tags.count(NI), net_income_tags.count(AVAIL)) == (92, 10)


def test_cli_derived_fourth_quarters_on_fixtures(fixture_run) -> None:
    _, _, rows, _ = fixture_run
    derived = [r for r in rows if r["derived"] == "True"]
    assert len(derived) == 26
    assert {(r["field"], r["period_type"]) for r in derived} == {
        ("revenue", "quarter"),
        ("net_income", "quarter"),
    }
    assert all(r["full_year_accession"] == r["linked_accession"] for r in derived)
    # Microsoft FY25: 281,724 - 205,283 and 101,832 - 74,599 (millions). The nine
    # months are from the FY25 third-quarter 10-Q, not FY26's, which repeats them.
    msft = [r for r in derived if r["linked_accession"] == "0000950170-25-100235"]
    assert msft == [
        {
            "ticker": "MSFT",
            "release_accession": "0000950170-25-100226",
            "linked_accession": "0000950170-25-100235",
            "field": field,
            "period_type": "quarter",
            "start": "2025-04-01",
            "end": "2025-06-30",
            "value": value,
            "unit": "USD",
            "tag": tag,
            "derived": "True",
            "full_year_accession": "0000950170-25-100235",
            "nine_months_accession": "0000950170-25-061046",
            "fiscal_year_start": "2024-07-01",
        }
        for field, value, tag in [
            ("revenue", "76441000000", RFC),
            ("net_income", "27233000000", NI),
        ]
    ]
    # Oracle FY25: its third-quarter 10-Q is no linked filing; revenue uses the tag
    # that 10-Q has.
    orcl = {
        r["field"]: (r["value"], r["tag"], r["nine_months_accession"])
        for r in derived
        if r["linked_accession"] == "0000950170-25-087926"
    }
    assert orcl == {
        "revenue": ("15903000000", RFC, "0000950170-25-037143"),
        "net_income": ("3427000000", NI, "0000950170-25-037143"),
    }


def test_cli_uses_the_cache(tmp_path) -> None:
    links = tmp_path / "links.csv"
    lines = LINKS.read_text().splitlines()
    links.write_text(
        "\n".join([lines[0], *[line for line in lines if line.startswith("AAPL,")]])
    )
    argv = ["--links", str(links), "--out", str(tmp_path / "answers.csv")]
    requests: list[str] = []
    client = make_client(tmp_path / "raw", fixture_routes(), requests)
    with contextlib.redirect_stdout(io.StringIO()):
        assert answers.main(argv, client=client) == 0
        assert answers.main(argv, client=client) == 0
        assert answers.main([*argv, "--refresh"], client=client) == 0
    facts_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    assert requests.count(facts_url) == 2  # first run, then --refresh
    # Apple's 10-K release needs its third-quarter 10-Q from the submissions.
    submissions_url = "https://data.sec.gov/submissions/CIK0000320193.json"
    assert requests.count(submissions_url) == 2


def test_cli_continues_after_edgar_error(tmp_path, capsys) -> None:
    lines = LINKS.read_text().splitlines()
    apple = [line for line in lines if line.startswith("AAPL,")][:2]
    unknown = apple[0].replace("AAPL,", "ZZZZ,", 1)
    links = tmp_path / "links.csv"
    links.write_text("\n".join([lines[0], *apple, unknown]))
    out = tmp_path / "answers.csv"
    argv = ["--links", str(links), "--out", str(out)]
    assert (
        answers.main(argv, client=make_client(tmp_path / "raw", fixture_routes())) == 1
    )
    captured = capsys.readouterr()
    assert "error: ZZZZ: ticker 'ZZZZ' is not in" in captured.err
    assert "its releases have no answers" in captured.err
    report = captured.out.splitlines()
    assert report[1].split() == ["AAPL", "2", "2", "2", "2"]
    assert report[2].split() == ["total", "2", "2", "2", "2"]
    with out.open(newline="") as f:
        assert {row["ticker"] for row in csv.DictReader(f)} == {"AAPL"}


def test_cli_without_links_file(tmp_path, capsys) -> None:
    argv = ["--links", str(tmp_path / "none.csv"), "--out", str(tmp_path / "a.csv")]
    assert answers.main(argv, client=make_client(tmp_path / "raw", {})) == 1
    assert "not found; run secbench.link first" in capsys.readouterr().err
    assert not (tmp_path / "a.csv").exists()


def test_cli_without_linked_rows(tmp_path, capsys) -> None:
    lines = LINKS.read_text().splitlines()
    links = tmp_path / "links.csv"
    links.write_text(
        "\n".join([lines[0], *[line for line in lines if ",excluded," in line]])
    )
    argv = ["--links", str(links), "--out", str(tmp_path / "a.csv")]
    assert answers.main(argv, client=make_client(tmp_path / "raw", {})) == 1
    assert "error: no linked releases in" in capsys.readouterr().err
    assert not (tmp_path / "a.csv").exists()


# Hand-checked answers (tests/fixtures/answers_handchecked.csv)


def parse_expected(text: str) -> Decimal | None:
    """An expected_value cell: blank means no answer; thousands commas allowed."""
    text = text.strip().replace(",", "")
    return None if not text else Decimal(text)


def handchecked_rows() -> list:
    if not HANDCHECKED.exists():
        return [pytest.param(None, id="file-missing")]
    with HANDCHECKED.open(newline="", encoding="utf-8") as f:
        return [
            pytest.param(
                row,
                id=f"{row['ticker']}-{row['linked_accession']}-{row['field']}-"
                f"{row['period_type']}".replace(" ", ""),
            )
            for row in csv.DictReader(f)
        ]


def test_handchecked_file_is_well_formed() -> None:
    assert HANDCHECKED.exists(), f"{HANDCHECKED} is missing"
    with HANDCHECKED.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == HANDCHECKED_COLUMNS
    assert rows
    keys = []
    for row in rows:
        assert row["ticker"].strip(), row
        assert row["field"].strip() in FIELD_NAMES, row
        assert row["period_type"].strip() in PERIOD_TYPES, row
        try:
            parse_expected(row["expected_value"])
        except InvalidOperation:
            pytest.fail(f"expected_value is not a number: {row}")
        keys.append(tuple(row[c].strip() for c in HANDCHECKED_COLUMNS[:4]))
    duplicates = {key for key in keys if keys.count(key) > 1}
    assert not duplicates, f"rows listed twice: {sorted(duplicates)}"


@pytest.mark.parametrize("expected", handchecked_rows())
def test_handchecked_answer(expected, fixture_run) -> None:
    assert expected is not None, f"{HANDCHECKED} is missing"
    _, stdout, rows, _ = fixture_run
    ticker, accession, field, period = (
        expected[c].strip() for c in HANDCHECKED_COLUMNS[:4]
    )
    releases, _ = load_links(LINKS)
    assert any(
        r.ticker == ticker and r.linked_accession == accession for r in releases
    ), f"{ticker} {accession} is not a linked filing in {LINKS.name}"
    got = [
        row
        for row in rows
        if (row["ticker"], row["linked_accession"], row["field"], row["period_type"])
        == (ticker, accession, field, period)
    ]
    want = parse_expected(expected["expected_value"])
    if want is None:
        assert not got, f"expected no answer, got {got[0]['value']} ({got[0]['tag']})"
        return
    reason = next(
        (
            line.strip()
            for line in stdout.splitlines()
            if f" {field} {period} (" in line and accession in line
        ),
        "not in the missing list either (period not expected for this filing?)",
    )
    assert got, f"no answer: {reason}"
    assert Decimal(got[0]["value"]) == want, (
        f"{got[0]['tag']} {got[0]['start']}..{got[0]['end']}"
    )
