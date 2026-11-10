"""A small repository for the Hugging Face export's tests (secbench.hfexport and its
modules), built in a temporary folder: the frozen splits, runs, training files and
results the export reads, with synthetic releases so every case is known.

- validation (stage 1 layout): AAA, a second-quarter 10-Q; BBB, a fourth quarter
  linked to a 10-K, with a derived quarter and a net loss printed in parentheses.
- company_test: CCC, a first quarter whose EPS answer isn't printed (not_in_input);
  DDD, a third quarter whose filing reports preferred dividends, with a "Net income
  available to common stockholders" row.
- time_test: EEE, a second quarter whose net loss is printed with a minus sign
  (quarter) and unsigned on a "Net loss" row (year to date); FFF, a fourth quarter;
  GGG, left out (its prompt is over the context limit).
- train: three labelled examples (EEE, FFF, GGG releases from 2024-2025), two of
  them trained on, one of those seen by step-00025.

results/test_results.json is the repository's own (its run names are the runs made
here, with synthetic figures); results/comparison.md is written by secbench.compare
from these runs. Each run's raw/ holds made-up account details that the export must
never copy. Library for the tests: build(root) returns a MiniRepo.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import struct
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from secbench import answers as answers_module
from secbench import compare, grade, link, splits
from secbench.stage2 import RELEASE_COLUMNS

REPO = Path(__file__).resolve().parents[1]
PROMPT_V2 = (REPO / "prompts" / "extract_v2.txt").read_text(encoding="utf-8")
SECRET_ORG = "org-FixtureOrgIdAbCdEfGhIjKl"
SECRET_EMAIL = "billing-owner@account-holder.example"
RELEASE_EMAIL = "investors@aaa-corp.example"  # a press contact, in a release text
INSTANCE_PATH = (
    "/workspace/sec-earnings-bench/data/raw/hf-hub/models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)


@dataclass
class Spec:
    ticker: str
    cik: int
    accession: str
    filing_date: str
    split: str  # validation, company_test, time_test or train
    form: str  # the linked 10-Q or 10-K
    linked_accession: str
    period_end: str
    kind: str  # first quarter, second or third quarter, fourth quarter
    markdown: str
    answers: list[dict[str, str]] = field(default_factory=list)
    preferred: bool = False
    tokens: int = 5000
    left_out: bool = False


COMPANIES = [
    ("AAA", 1001, "Alpha Corp", "Industrials", "validation"),
    ("BBB", 1002, "Beta Inc", "Information Technology", "validation"),
    ("CCC", 1003, "Gamma Co", "Health Care", "company_test"),
    ("DDD", 1004, "Delta Corp", "Utilities", "company_test"),
    ("EEE", 1005, "Epsilon Ltd", "Materials", "train"),
    ("FFF", 1006, "Zeta Holdings", "Energy", "train"),
    ("GGG", 1007, "Eta Group", "Real Estate", "train"),
]


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join([" --- "] * len(header)) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def release_text(name: str, intro: str, statement: str) -> str:
    return (
        f"# {name} reports results\n\n{intro}\n\n"
        "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS\n\n"
        f"{statement}\n\n"
        f"Investor contact: {RELEASE_EMAIL}\n"
    )


def answer(
    spec_ticker: str,
    accession: str,
    linked: str,
    name: str,
    period: str,
    start: str,
    end: str,
    value: str,
    tag: str,
    derived: tuple[str, str, str] | None = None,
) -> dict[str, str]:
    full_year, nine_months, year_start = derived or ("", "", "")
    return {
        "ticker": spec_ticker,
        "release_accession": accession,
        "linked_accession": linked,
        "field": name,
        "period_type": period,
        "start": start,
        "end": end,
        "value": value,
        "unit": "USD/shares" if name == "eps_diluted" else "USD",
        "tag": tag,
        "derived": str(derived is not None),
        "full_year_accession": full_year,
        "nine_months_accession": nine_months,
        "fiscal_year_start": year_start,
    }


def specs() -> list[Spec]:
    found = []
    # AAA: second quarter, three- and six-month columns, in millions.
    acc, linked = "0000001001-26-000010", "0000001001-26-000012"
    found.append(
        Spec(
            "AAA",
            1001,
            acc,
            "2026-07-28",
            "validation",
            "10-Q",
            linked,
            "2026-06-30",
            "second or third quarter",
            release_text(
                "Alpha Corp",
                "Alpha Corp reported revenue of $1.2 billion for the quarter.",
                table(
                    [
                        "(in millions, except per share data)",
                        "Three Months Ended June 30, 2026",
                        "Three Months Ended June 30, 2025",
                        "Six Months Ended June 30, 2026",
                        "Six Months Ended June 30, 2025",
                    ],
                    [
                        ["Total revenues", "$ 1,234", "$ 1,100", "$ 2,401", "$ 2,150"],
                        ["Net income", "120", "95", "231", "180"],
                        [
                            "Diluted earnings per share",
                            "$ 0.45",
                            "$ 0.36",
                            "$ 0.87",
                            "$ 0.68",
                        ],
                    ],
                ),
            ),
            [
                answer(
                    "AAA",
                    acc,
                    linked,
                    "revenue",
                    "quarter",
                    "2026-04-01",
                    "2026-06-30",
                    "1234000000",
                    "Revenues",
                ),
                answer(
                    "AAA",
                    acc,
                    linked,
                    "revenue",
                    "year_to_date",
                    "2026-01-01",
                    "2026-06-30",
                    "2401000000",
                    "Revenues",
                ),
                answer(
                    "AAA",
                    acc,
                    linked,
                    "net_income",
                    "quarter",
                    "2026-04-01",
                    "2026-06-30",
                    "120000000",
                    "NetIncomeLoss",
                ),
                answer(
                    "AAA",
                    acc,
                    linked,
                    "net_income",
                    "year_to_date",
                    "2026-01-01",
                    "2026-06-30",
                    "231000000",
                    "NetIncomeLoss",
                ),
                answer(
                    "AAA",
                    acc,
                    linked,
                    "eps_diluted",
                    "quarter",
                    "2026-04-01",
                    "2026-06-30",
                    "0.45",
                    "EarningsPerShareDiluted",
                ),
                answer(
                    "AAA",
                    acc,
                    linked,
                    "eps_diluted",
                    "year_to_date",
                    "2026-01-01",
                    "2026-06-30",
                    "0.87",
                    "EarningsPerShareDiluted",
                ),
            ],
            tokens=4100,
        )
    )
    # BBB: fourth quarter (10-K), a net loss in the quarter, in thousands.
    acc, linked = "0000001002-26-000004", "0000001002-26-000009"
    derived = (linked, "0000001002-25-000030", "2025-01-01")
    found.append(
        Spec(
            "BBB",
            1002,
            acc,
            "2026-02-05",
            "validation",
            "10-K",
            linked,
            "2025-12-31",
            "fourth quarter",
            release_text(
                "Beta Inc",
                "Beta Inc reported a fourth-quarter net loss.",
                table(
                    [
                        "(in thousands, except per share amounts)",
                        "Three Months Ended December 31, 2025",
                        "Three Months Ended December 31, 2024",
                        "Year Ended December 31, 2025",
                        "Year Ended December 31, 2024",
                    ],
                    [
                        [
                            "Net sales",
                            "$ 512,300",
                            "$ 498,000",
                            "$ 2,050,100",
                            "$ 1,900,000",
                        ],
                        ["Net income (loss)", "(30,250)", "12,000", "84,700", "60,000"],
                        [
                            "Net income (loss) per diluted share",
                            "$ (0.21)",
                            "$ 0.08",
                            "$ 0.59",
                            "$ 0.42",
                        ],
                    ],
                ),
            ),
            [
                answer(
                    "BBB",
                    acc,
                    linked,
                    "revenue",
                    "quarter",
                    "2025-10-01",
                    "2025-12-31",
                    "512300000",
                    "Revenues",
                    derived,
                ),
                answer(
                    "BBB",
                    acc,
                    linked,
                    "revenue",
                    "full_year",
                    "2025-01-01",
                    "2025-12-31",
                    "2050100000",
                    "Revenues",
                ),
                answer(
                    "BBB",
                    acc,
                    linked,
                    "net_income",
                    "quarter",
                    "2025-10-01",
                    "2025-12-31",
                    "-30250000",
                    "NetIncomeLoss",
                    derived,
                ),
                answer(
                    "BBB",
                    acc,
                    linked,
                    "net_income",
                    "full_year",
                    "2025-01-01",
                    "2025-12-31",
                    "84700000",
                    "NetIncomeLoss",
                ),
                answer(
                    "BBB",
                    acc,
                    linked,
                    "eps_diluted",
                    "full_year",
                    "2025-01-01",
                    "2025-12-31",
                    "0.59",
                    "EarningsPerShareDiluted",
                ),
            ],
            tokens=6200,
        )
    )
    # CCC: first quarter; its EPS answer (0.33) isn't printed (the release says 0.32).
    acc, linked = "0000001003-26-000020", "0000001003-26-000022"
    found.append(
        Spec(
            "CCC",
            1003,
            acc,
            "2026-05-01",
            "company_test",
            "10-Q",
            linked,
            "2026-03-31",
            "first quarter",
            release_text(
                "Gamma Co",
                "Gamma Co reported first-quarter results.",
                table(
                    [
                        "(in millions, except per share data)",
                        "Three Months Ended March 31, 2026",
                        "Three Months Ended March 31, 2025",
                    ],
                    [
                        ["Total revenues", "$ 845", "$ 801"],
                        ["Net earnings", "77", "70"],
                        ["Diluted earnings per share", "$ 0.32", "$ 0.29"],
                    ],
                ),
            ),
            [
                answer(
                    "CCC",
                    acc,
                    linked,
                    "revenue",
                    "quarter",
                    "2026-01-01",
                    "2026-03-31",
                    "845000000",
                    "Revenues",
                ),
                answer(
                    "CCC",
                    acc,
                    linked,
                    "net_income",
                    "quarter",
                    "2026-01-01",
                    "2026-03-31",
                    "77000000",
                    "NetIncomeLoss",
                ),
                answer(
                    "CCC",
                    acc,
                    linked,
                    "eps_diluted",
                    "quarter",
                    "2026-01-01",
                    "2026-03-31",
                    "0.33",
                    "EarningsPerShareDiluted",
                ),
            ],
            tokens=3900,
        )
    )
    # DDD: third quarter, preferred dividends, an "available to common" line.
    acc, linked = "0000001004-25-000041", "0000001004-25-000044"
    found.append(
        Spec(
            "DDD",
            1004,
            acc,
            "2025-10-30",
            "company_test",
            "10-Q",
            linked,
            "2025-09-30",
            "second or third quarter",
            release_text(
                "Delta Corp",
                "Delta Corp reported third-quarter results.",
                table(
                    [
                        "(in millions, except per share amounts)",
                        "Three Months Ended September 30, 2025",
                        "Three Months Ended September 30, 2024",
                        "Nine Months Ended September 30, 2025",
                        "Nine Months Ended September 30, 2024",
                    ],
                    [
                        [
                            "Operating revenues",
                            "$ 3,010",
                            "$ 2,900",
                            "$ 8,800",
                            "$ 8,500",
                        ],
                        ["Net income", "200", "180", "560", "500"],
                        ["Preferred stock dividends", "10", "10", "30", "30"],
                        [
                            "Net income available to common stockholders",
                            "190",
                            "170",
                            "530",
                            "470",
                        ],
                        [
                            "Diluted earnings per share",
                            "$ 1.12",
                            "$ 1.01",
                            "$ 3.12",
                            "$ 2.80",
                        ],
                    ],
                ),
            ),
            [
                answer(
                    "DDD",
                    acc,
                    linked,
                    "revenue",
                    "quarter",
                    "2025-07-01",
                    "2025-09-30",
                    "3010000000",
                    "Revenues",
                ),
                answer(
                    "DDD",
                    acc,
                    linked,
                    "revenue",
                    "year_to_date",
                    "2025-01-01",
                    "2025-09-30",
                    "8800000000",
                    "Revenues",
                ),
                answer(
                    "DDD",
                    acc,
                    linked,
                    "net_income",
                    "quarter",
                    "2025-07-01",
                    "2025-09-30",
                    "200000000",
                    "NetIncomeLoss",
                ),
                answer(
                    "DDD",
                    acc,
                    linked,
                    "net_income",
                    "year_to_date",
                    "2025-01-01",
                    "2025-09-30",
                    "560000000",
                    "NetIncomeLoss",
                ),
                answer(
                    "DDD",
                    acc,
                    linked,
                    "eps_diluted",
                    "quarter",
                    "2025-07-01",
                    "2025-09-30",
                    "1.12",
                    "EarningsPerShareDiluted",
                ),
                answer(
                    "DDD",
                    acc,
                    linked,
                    "eps_diluted",
                    "year_to_date",
                    "2025-01-01",
                    "2025-09-30",
                    "3.12",
                    "EarningsPerShareDiluted",
                ),
            ],
            preferred=True,
            tokens=7700,
        )
    )
    # EEE: second quarter; a net loss with a minus sign (quarter), and unsigned on a
    # "Net loss" row (year to date).
    acc, linked = "0000001005-26-000031", "0000001005-26-000033"
    found.append(
        Spec(
            "EEE",
            1005,
            acc,
            "2026-07-30",
            "time_test",
            "10-Q",
            linked,
            "2026-06-30",
            "second or third quarter",
            release_text(
                "Epsilon Ltd",
                "Epsilon Ltd reported a second-quarter loss.",
                table(
                    [
                        "(in millions, except per share data)",
                        "Three Months Ended June 30, 2026",
                        "Three Months Ended June 30, 2025",
                        "Six Months Ended June 30, 2026",
                        "Six Months Ended June 30, 2025",
                    ],
                    [
                        ["Total sales", "$ 640", "$ 600", "$ 1,255", "$ 1,190"],
                        ["Net loss", "−15", "−9", "40", "22"],
                        [
                            "Diluted loss per share",
                            "$ (0.11)",
                            "$ (0.07)",
                            "$ (0.29)",
                            "$ (0.16)",
                        ],
                    ],
                ),
            ),
            [
                answer(
                    "EEE",
                    acc,
                    linked,
                    "revenue",
                    "quarter",
                    "2026-04-01",
                    "2026-06-30",
                    "640000000",
                    "Revenues",
                ),
                answer(
                    "EEE",
                    acc,
                    linked,
                    "revenue",
                    "year_to_date",
                    "2026-01-01",
                    "2026-06-30",
                    "1255000000",
                    "Revenues",
                ),
                answer(
                    "EEE",
                    acc,
                    linked,
                    "net_income",
                    "quarter",
                    "2026-04-01",
                    "2026-06-30",
                    "-15000000",
                    "NetIncomeLoss",
                ),
                answer(
                    "EEE",
                    acc,
                    linked,
                    "net_income",
                    "year_to_date",
                    "2026-01-01",
                    "2026-06-30",
                    "-40000000",
                    "NetIncomeLoss",
                ),
                answer(
                    "EEE",
                    acc,
                    linked,
                    "eps_diluted",
                    "quarter",
                    "2026-04-01",
                    "2026-06-30",
                    "-0.11",
                    "EarningsPerShareDiluted",
                ),
                answer(
                    "EEE",
                    acc,
                    linked,
                    "eps_diluted",
                    "year_to_date",
                    "2026-01-01",
                    "2026-06-30",
                    "-0.29",
                    "EarningsPerShareDiluted",
                ),
            ],
            tokens=5100,
        )
    )
    # FFF: fourth quarter of a fiscal year ending in June.
    acc, linked = "0000001006-26-000050", "0000001006-26-000055"
    derived = (linked, "0000001006-26-000021", "2025-07-01")
    found.append(
        Spec(
            "FFF",
            1006,
            acc,
            "2026-08-06",
            "time_test",
            "10-K",
            linked,
            "2026-06-30",
            "fourth quarter",
            release_text(
                "Zeta Holdings",
                "Zeta Holdings reported fiscal 2026 results.",
                table(
                    [
                        "(in millions, except per share data)",
                        "Three Months Ended June 30, 2026",
                        "Three Months Ended June 30, 2025",
                        "Fiscal Year 2026",
                        "Fiscal Year 2025",
                    ],
                    [
                        [
                            "Total revenues",
                            "$ 9,876",
                            "$ 9,100",
                            "$ 38,402",
                            "$ 35,000",
                        ],
                        [
                            "Net income attributable to Zeta",
                            "1,045",
                            "990",
                            "4,120",
                            "3,800",
                        ],
                        [
                            "Diluted earnings per share",
                            "$ 2.31",
                            "$ 2.10",
                            "$ 9.08",
                            "$ 8.33",
                        ],
                    ],
                ),
            ),
            [
                answer(
                    "FFF",
                    acc,
                    linked,
                    "revenue",
                    "quarter",
                    "2026-04-01",
                    "2026-06-30",
                    "9876000000",
                    "Revenues",
                    derived,
                ),
                answer(
                    "FFF",
                    acc,
                    linked,
                    "revenue",
                    "full_year",
                    "2025-07-01",
                    "2026-06-30",
                    "38402000000",
                    "Revenues",
                ),
                answer(
                    "FFF",
                    acc,
                    linked,
                    "net_income",
                    "quarter",
                    "2026-04-01",
                    "2026-06-30",
                    "1045000000",
                    "NetIncomeLoss",
                    derived,
                ),
                answer(
                    "FFF",
                    acc,
                    linked,
                    "net_income",
                    "full_year",
                    "2025-07-01",
                    "2026-06-30",
                    "4120000000",
                    "NetIncomeLoss",
                ),
                answer(
                    "FFF",
                    acc,
                    linked,
                    "eps_diluted",
                    "full_year",
                    "2025-07-01",
                    "2026-06-30",
                    "9.08",
                    "EarningsPerShareDiluted",
                ),
            ],
            tokens=8800,
        )
    )
    # GGG: left out for every model (its prompt is over the context limit).
    acc, linked = "0000001007-26-000061", "0000001007-26-000064"
    found.append(
        Spec(
            "GGG",
            1007,
            acc,
            "2026-08-01",
            "time_test",
            "10-Q",
            linked,
            "2026-06-30",
            "second or third quarter",
            release_text(
                "Eta Group",
                "Eta Group reported second-quarter results with a long supplement.",
                table(
                    [
                        "(in thousands, except per share data)",
                        "Three Months Ended June 30, 2026",
                        "Six Months Ended June 30, 2026",
                    ],
                    [
                        ["Total revenues", "$ 410,500", "$ 800,200"],
                        ["Net income", "55,100", "101,900"],
                        ["Diluted earnings per share", "$ 0.51", "$ 0.95"],
                    ],
                ),
            ),
            [
                answer(
                    "GGG",
                    acc,
                    linked,
                    "revenue",
                    "quarter",
                    "2026-04-01",
                    "2026-06-30",
                    "410500000",
                    "Revenues",
                ),
                answer(
                    "GGG",
                    acc,
                    linked,
                    "net_income",
                    "quarter",
                    "2026-04-01",
                    "2026-06-30",
                    "55100000",
                    "NetIncomeLoss",
                ),
            ],
            tokens=45000,
            left_out=True,
        )
    )
    return found


TRAIN = [
    (
        "EEE",
        1005,
        "0000001005-24-000031",
        "2024-07-30",
        "second or third quarter",
        "10-Q",
    ),
    ("FFF", 1006, "0000001006-24-000012", "2024-10-31", "first quarter", "10-Q"),
    ("GGG", 1007, "0000001007-25-000008", "2025-02-12", "fourth quarter", "10-K"),
]
# training/qwen3-8b-2k/examples.csv: positions 0 and 1; step-00025 saw position 0.
TRAINED = {"0000001006-24-000012": 0, "0000001005-24-000031": 1}
EXAMPLES_SEEN = 1


@dataclass
class MiniRepo:
    root: Path
    specs: list[Spec]
    adapter_sha256: str
    validation_runs: list[str]
    test_runs: dict[str, list[str]]
    test_results: dict


def csv_text(rows: list[dict], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    write(path, json.dumps(data, indent=2) + "\n")


def save_release(folder: Path, spec: Spec) -> None:
    cik10 = f"{spec.cik:010d}"
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{spec.cik}/"
        f"{spec.accession.replace('-', '')}/ex991.htm"
    )
    record = {
        "ticker": spec.ticker,
        "cik": spec.cik,
        "filing_date": spec.filing_date,
        "accession_number": spec.accession,
        "exhibit": "EX-99.1",
        "exhibit_url": url,
        "exhibits": [
            {
                "type": "EX-99.1",
                "description": "Press release",
                "url": url,
                "file": f"{spec.accession}.md",
                "chosen": True,
            }
        ],
    }
    write(folder / cik10 / f"{spec.accession}.md", spec.markdown)
    write_json(folder / cik10 / f"{spec.accession}.json", record)


def links_row(spec: Spec) -> dict[str, str]:
    return {
        "ticker": spec.ticker,
        "release_cik": str(spec.cik),
        "release_accession": spec.accession,
        "release_date": spec.filing_date,
        "linked_form": spec.form,
        "linked_cik": str(spec.cik),
        "linked_accession": spec.linked_accession,
        "period_end": spec.period_end,
        "period_type": "full year" if spec.form == "10-K" else "quarter",
        "status": link.LINKED,
        "reason": "",
    }


def releases_csv_row(spec: Spec) -> dict[str, str]:
    company_split = next(c[4] for c in COMPANIES if c[0] == spec.ticker)
    return {
        "ticker": spec.ticker,
        "cik": str(spec.cik),
        "company_split": company_split,
        "release_accession": spec.accession,
        "filing_date": spec.filing_date,
        "saved": "True",
        "link_status": link.LINKED,
        "linked_form": spec.form,
        "linked_accession": spec.linked_accession,
        "period_end": spec.period_end,
        "split": spec.split,
        "answers": str(len(spec.answers)),
        "missing": "0",
        "preferred_dividends": str(spec.preferred),
    }


def safetensors_bytes() -> bytes:
    """A valid safetensors file: one 1 x 2 float32 tensor."""
    header = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [1, 2],
                "data_offsets": [0, 8],
            },
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<2f", 0.5, -0.25)


def verdict_row(
    model: str, ans: dict[str, str], printed: bool
) -> tuple[str, str, str, str]:
    """(predicted, verdict, evidence, factor) of model for the answer."""
    value = Decimal(ans["value"])
    if not printed:
        searched = ", ".join(f.text for f in grade.printed_forms(value, ans["field"]))
        predicted = "" if model in ("gpt-6.1-sol", "sol") else "0.32"
        return predicted, grade.NOT_IN_INPUT, f"searched {searched}", ""
    key = (model, ans["ticker"], ans["field"], ans["period_type"])
    if key[0] in ("step-00025", "step") and key[1:3] == ("DDD", "net_income"):
        return (
            str(value - 10_000_000),
            grade.WRONG_LINE,
            "'Net income available to common stockholders': 190 (x1,000,000)",
            "",
        )
    if key[0] in ("base", "qwen3-nothink") and key[1:3] in (
        ("CCC", "revenue"),
        ("AAA", "revenue"),
    ):
        return (
            str(value * 1000),
            grade.SCALE,
            "prediction /1,000 is the answer",
            "0.001",
        )
    if key[0] == "gpt-6-luna" and key[1:4] == ("EEE", "revenue", "quarter"):
        return "655000000", grade.ADJUSTED, "'Adjusted sales': 655 (x1,000,000)", ""
    return str(value), grade.CORRECT, "", ""


def graded_and_predictions(
    model: str, chosen: list[Spec], printed: dict[tuple[str, str, str], bool]
) -> tuple[str, str]:
    rows, predictions = [], []
    for spec in chosen:
        figures: dict[str, dict[str, str | None]] = {
            "quarter": dict.fromkeys(grade.FIELD_NAMES),
            "year_to_date": dict.fromkeys(grade.FIELD_NAMES),
        }
        for ans in spec.answers:
            key = (spec.accession, ans["field"], ans["period_type"])
            predicted, verdict, evidence, factor = verdict_row(model, ans, printed[key])
            rows.append(
                {
                    "ticker": spec.ticker,
                    "release_accession": spec.accession,
                    "linked_accession": spec.linked_accession,
                    "field": ans["field"],
                    "period_type": ans["period_type"],
                    "answer": ans["value"],
                    "answer_tag": ans["tag"],
                    "predicted": predicted,
                    "verdict": verdict,
                    "evidence": evidence,
                    "factor": factor,
                }
            )
            group = "quarter" if ans["period_type"] == "quarter" else "year_to_date"
            figures[group][ans["field"]] = predicted or None
        if spec.kind == "first quarter":  # a year to date without an answer
            rows.append(
                {
                    "ticker": spec.ticker,
                    "release_accession": spec.accession,
                    "linked_accession": spec.linked_accession,
                    "field": "revenue",
                    "period_type": "year_to_date",
                    "answer": "",
                    "answer_tag": "",
                    "predicted": figures["quarter"]["revenue"] or "",
                    "verdict": grade.NO_ANSWER,
                    "evidence": "",
                    "factor": "",
                }
            )
        predictions.append(
            json.dumps(
                {
                    "release_accession": spec.accession,
                    "period_end": spec.period_end,
                    **figures,
                }
            )
        )
    return csv_text(rows, grade.GRADED_COLUMNS), "\n".join(predictions) + "\n"


def run_info(
    name: str,
    provider: str,
    model: str,
    prompt: str,
    parameters: dict,
    chosen: list[Spec],
    *,
    gpu: bool,
    links: Path | None = None,
    status: str = "complete",
) -> dict:
    calls = [
        {
            "ticker": spec.ticker,
            "release_accession": spec.accession,
            "call": 1,
            "status": 200,
            "http_attempts": 1,
            "latency_s": 1.5 + i,
            "input_tokens": 5000 + 10 * i,
            "output_tokens": 120,
            "finish_reason": "stop",
        }
        for i, spec in enumerate(chosen)
    ]
    info = {
        "run_name": name,
        "status": status,
        "stop_reason": "",
        "provider": provider,
        "model": model,
        "env_file": f".env.{name}",
        "prompt_version": prompt,
        "reply_format": "with_units" if prompt == "extract_v2" else "raw_dollars",
        "request_parameters": parameters,
        "max_parallel": 16 if gpu else 4,
        "started": "2026-11-01T10:00:00+00:00",
        "finished": "2026-11-09T10:05:00+00:00",
        "wall_time_s": 90.0,
        "releases": len(chosen),
        "outcomes": {
            "ok": len(chosen),
            "ok_after_retry": 0,
            "invalid_reply": 0,
            "api_error": 0,
            "aborted": 0,
        },
        "fallbacks": [],
        "format_nulls": [],
        "tokens": {
            "input": sum(c["input_tokens"] for c in calls),
            "output": 120 * len(calls),
            "calls_without_usage": 0,
        },
        "price_per_million_tokens": (
            {"input": None, "output": None} if gpu else {"input": 0.1, "output": 0.5}
        ),
        "gpu_hourly_usd": 2.5 if gpu else None,
        "cost_basis": "gpu_hours" if gpu else "tokens",
        "estimated_cost_usd": 0.0625 if gpu else 0.0011,
        "latency_s": {"p50": 1.5, "p95": 2.5, "max": 2.5},
        "calls": calls,
    }
    if links is not None:
        info["links"] = str(links)
        info["links_sha256"] = hashlib.sha256(links.read_bytes()).hexdigest()
    return info


def save_run(
    root: Path,
    name: str,
    info: dict,
    model_key: str,
    chosen: list[Spec],
    printed: dict,
    *,
    graded: bool = True,
) -> None:
    folder = root / "runs" / name
    graded_text, predictions = graded_and_predictions(model_key, chosen, printed)
    write_json(folder / "run.json", info)
    write(folder / "predictions.jsonl", predictions)
    if graded:
        write(folder / "graded.csv", graded_text)
    for spec in chosen:  # never read by the export
        write_json(
            folder / "raw" / f"{spec.accession}.json",
            {
                "response": {
                    "id": "chatcmpl-FixtureCompletionId0001",
                    "organization": SECRET_ORG,
                    "billing_email": SECRET_EMAIL,
                }
            },
        )


def training_files(root: Path) -> None:
    folder = root / "training" / "qwen3-8b-2k"
    lora = {
        "r": 16,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }
    write_json(
        folder / "config.json",
        {
            "run_name": "qwen3-8b-2k",
            "created": "2026-11-01T22:42:35+00:00",
            "commit": "b5e9990b44cb0edb77dd94934d3f57ddcfef5250",
            "config_file": "configs/train/qwen3-8b-2k.toml",
            "overrides": {},
            "settings": {
                "model": {
                    "name": "Qwen/Qwen3-8B",
                    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                },
                "data": {"subset": 2, "max_tokens": 32768, "seed": 20261101},
                "lora": lora,
                "training": {
                    "epochs": 1,
                    "gradient_accumulation": 1,
                    "learning_rate": 0.0002,
                    "warmup_steps": 1,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "save_every": 1,
                    "gradient_checkpointing": True,
                    "dtype": "bfloat16",
                    "device": "cuda",
                },
            },
            "selection": {
                "examples": 3,
                "over_max_tokens": 0,
                "within_max_tokens": {
                    "second or third quarter": 1,
                    "first quarter": 1,
                    "fourth quarter": 1,
                },
                "selected": 2,
                "selected_by_kind": {"second or third quarter": 1, "first quarter": 1},
            },
            "schedule": {
                "epochs": 1,
                "examples_per_epoch": 2,
                "positions": 2,
                "examples_per_step": 1,
                "steps": 2,
                "warmup_steps": 1,
                "checkpoints": [1, 2],
            },
            "tokens": {"total": 12000, "target": 260, "mean": 6000, "max": 6500},
        },
    )
    write(
        folder / "train_log.jsonl",
        "".join(
            json.dumps(
                {
                    "step": step,
                    "examples_seen": step,
                    "epoch": step / 2,
                    "loss": 0.1 / step,
                    "learning_rate": 0.0002,
                    "grad_norm": 0.5,
                    "tokens": 6000,
                    "target_tokens": 130,
                    "seconds": 2.0,
                    "tokens_per_second": 3000.0,
                    "peak_memory_gb": 30.5,
                }
            )
            + "\n"
            for step in (1, 2)
        ),
    )
    models = []
    for label, step, correct, by_field in (
        (
            "base",
            0,
            9,
            {"revenue": [3, 4], "net_income": [3, 4], "eps_diluted": [3, 3]},
        ),
        (
            "step-00025",
            25,
            11,
            {"revenue": [4, 4], "net_income": [4, 4], "eps_diluted": [3, 3]},
        ),
        (
            "step-00050",
            50,
            10,
            {"revenue": [4, 4], "net_income": [3, 4], "eps_diluted": [3, 3]},
        ),
    ):
        models.append(
            {
                "label": label,
                "served": "Qwen/Qwen3-8B" if step == 0 else f"qwen3-8b-2k--{label}",
                "step": step,
                "examples_seen": step,
                "run": f"runs/qwen3-8b-2k--{label}",
                "scored": "2026-11-02T01:34:49+00:00",
                "kept": step == 0,
                "correct": correct,
                "graded": 11,
                "by_field": by_field,
                "not_correct": {} if correct == 11 else {"wrong_line": 11 - correct},
                "fallbacks": 0,
            }
        )
    write_json(
        folder / "validation.json",
        {
            "run": "training/qwen3-8b-2k",
            "created": "2026-11-02T01:31:28+00:00",
            "base_model": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "server": "vLLM 0.32.0",
            "links": "data/links.csv",
            "releases": 2,
            "settings": {
                "prompt": "extract_v2",
                "request_parameters": {
                    "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "max_tokens": 1024,
                },
            },
            "models": models,
            "chosen": "step-00025",
        },
    )
    write(
        folder / "examples.csv",
        csv_text(
            [
                {
                    "position": position,
                    "epoch": 1,
                    "release_accession": accession,
                    "ticker": next(t[0] for t in TRAIN if t[2] == accession),
                    "kind": next(t[4] for t in TRAIN if t[2] == accession),
                    "prompt_tokens": 5900,
                    "target_tokens": 130,
                    "tokens": 6030,
                }
                for accession, position in sorted(TRAINED.items(), key=lambda x: x[1])
            ],
            [
                "position",
                "epoch",
                "release_accession",
                "ticker",
                "kind",
                "prompt_tokens",
                "target_tokens",
                "tokens",
            ],
        ),
    )
    checkpoint = folder / "checkpoints" / "step-00025"
    adapter_config = {
        "base_model_name_or_path": INSTANCE_PATH,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "peft_version": "0.21.2",
        "r": 16,
        "revision": None,
        "target_modules": lora["target_modules"],
        "task_type": "CAUSAL_LM",
    }
    write_json(checkpoint / "adapter_config.json", adapter_config)
    (checkpoint / "adapter_model.safetensors").write_bytes(safetensors_bytes())
    write_json(
        checkpoint / "state.json",
        {
            "step": 25,
            "examples_seen": EXAMPLES_SEEN,
            "epoch": 0.5,
            "learning_rate": 0.0002,
            "mean_loss_since_last_checkpoint": 0.05,
            "saved": "2026-11-01T23:24:56+00:00",
        },
    )
    write(checkpoint / "README.md", f"---\nbase_model: {INSTANCE_PATH}\n---\n")


def train_examples(root: Path) -> None:
    labels = root / "data" / "stage2" / "labels"
    lines, index = [], []
    for ticker, _cik, accession, filed, kind, form in TRAIN:
        reply = {
            "period_end": "2024-06-30",
            "quarter": {
                "revenue": {"value": 500, "unit": "millions"},
                "net_income": {"value": 40, "unit": "millions"},
                "eps_diluted": 0.31,
            },
            "year_to_date": {
                "revenue": {"value": 990, "unit": "millions"},
                "net_income": {"value": 75, "unit": "millions"},
                "eps_diluted": 0.58,
            },
        }
        user = release_text(
            f"{ticker} (train)",
            "A training release.",
            table(
                ["(in millions)", "Three Months Ended June 30, 2024"],
                [
                    ["Total revenues", "$ 500"],
                    ["Net income", "40"],
                    ["Diluted earnings per share", "$ 0.31"],
                ],
            ),
        )
        lines.append(
            json.dumps(
                {
                    "release_accession": accession,
                    "messages": [
                        {"role": "system", "content": PROMPT_V2},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": json.dumps(reply)},
                    ],
                }
            )
        )
        index.append(
            {
                "release_accession": accession,
                "ticker": ticker,
                "kind": kind,
                "linked_form": form,
                "filing_date": filed,
            }
        )
    write(labels / "train.jsonl", "\n".join(lines) + "\n")
    write(
        labels / "examples.csv",
        csv_text(
            index, ["release_accession", "ticker", "kind", "linked_form", "filing_date"]
        ),
    )


def build(root: Path) -> MiniRepo:
    """The mini repository under root (which must not exist yet or be empty)."""
    root.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE",):
        shutil.copyfile(REPO / name, root / name)
    for name in ("extract_v1.txt", "extract_v2.txt"):
        write(root / "prompts" / name, (REPO / "prompts" / name).read_text("utf-8"))
    for name in ("test_results.json", "test_protocol.md"):
        write(root / "results" / name, (REPO / "results" / name).read_text("utf-8"))
    for name in ("accuracy_vs_cost.png", "test_accuracy_vs_cost.png"):
        shutil.copyfile(REPO / "results" / name, root / "results" / name)
    write(
        root / "configs" / "serving" / "qwen3-8b-2k.toml",
        (REPO / "configs" / "serving" / "qwen3-8b-2k.toml").read_text("utf-8"),
    )
    adapter_sha256 = hashlib.sha256(safetensors_bytes()).hexdigest()
    test_config = (REPO / "configs" / "test" / "qwen3-8b-2k.toml").read_text("utf-8")
    real_sha = "b93376d91af812159f4bd3a0fb80be090808e6d3539b251dc53ed26b9d27041b"
    write(
        root / "configs" / "test" / "qwen3-8b-2k.toml",
        test_config.replace(real_sha, adapter_sha256),
    )
    write(
        root / "data" / "companies.csv",
        csv_text(
            [
                {"ticker": t, "cik": c, "name": n, "sector": s, "split": p}
                for t, c, n, s, p in COMPANIES
            ],
            ["ticker", "cik", "name", "sector", "split"],
        ),
    )
    all_specs = specs()
    stage1 = [s for s in all_specs if s.split == "validation"]
    stage2 = [s for s in all_specs if s.split != "validation"]
    for spec in stage1:
        save_release(root / "data" / "releases", spec)
    for spec in stage2:
        save_release(root / "data" / "stage2" / "releases", spec)
    write(
        root / "data" / "links.csv",
        csv_text([links_row(s) for s in stage1], link.COLUMNS),
    )
    answer_columns = answers_module.COLUMNS
    write(
        root / "data" / "answers.csv",
        csv_text([a for s in stage1 for a in s.answers], answer_columns),
    )
    write(
        root / "data" / "stage2" / "links.csv",
        csv_text([links_row(s) for s in stage2], link.COLUMNS),
    )
    write(
        root / "data" / "stage2" / "answers.csv",
        csv_text([a for s in stage2 for a in s.answers], answer_columns),
    )
    write(
        root / "data" / "stage2" / "releases.csv",
        csv_text([releases_csv_row(s) for s in all_specs], RELEASE_COLUMNS),
    )
    train_examples(root)

    # The splits as secbench.splits builds and freezes them (token counts given).
    sources = splits.Sources(
        root / "data" / "links.csv",
        root / "data" / "answers.csv",
        root / "data" / "releases",
    )
    built = splits.build_all(sources, root / "data" / "stage2", [], [])
    by_accession = {s.accession: s for s in all_specs}
    frozen_rows = []
    for split in built:
        for release in split.releases:
            spec = by_accession[release.accession]
            release.tokens = spec.tokens
            release.status = splits.LEFT_OUT if spec.left_out else splits.KEPT
        folder = root / "data" / "splits" / split.name
        for name, text in splits.split_files(split, splits.MAX_PROMPT_TOKENS).items():
            write(folder / name, text)
        frozen_rows += [r.frozen_row() for r in split.releases]
    write(
        root / "data" / "frozen_test_releases.csv",
        csv_text(frozen_rows, splits.FROZEN_COLUMNS),
    )

    # Which answers each release prints (the grader's input check).
    printed = {}
    for spec in all_specs:
        text = (
            root
            / "data"
            / ("releases" if spec.split == "validation" else "stage2/releases")
            / f"{spec.cik:010d}"
            / f"{spec.accession}.md"
        ).read_text("utf-8")
        rows = grade.input_rows(text)
        for ans in spec.answers:
            found = grade.find_in_input(Decimal(ans["value"]), ans["field"], text, rows)
            printed[(spec.accession, ans["field"], ans["period_type"])] = (
                found is not None
            )

    # Validation runs: four in comparison.md's main table, and three that aren't.
    links = root / "data" / "links.csv"
    thinking_off = {"enable_thinking": False}
    validation = [
        (
            "2026-10-14T023302_claude-haiku-5-5_extract_v1",
            "claude",
            run_info(
                "2026-10-14T023302_claude-haiku-5-5_extract_v1",
                "anthropic",
                "claude-haiku-5-5",
                "extract_v1",
                {"max_tokens": 16000, "output_config": {"effort": "medium"}},
                stage1,
                gpu=False,
            ),
        ),
        (
            "qwen3-8b-nothink-v2",
            "qwen3-nothink",
            run_info(
                "qwen3-8b-nothink-v2",
                "openai",
                "Qwen/Qwen3-8B",
                "extract_v2",
                {"temperature": 0.0, "chat_template_kwargs": thinking_off},
                stage1,
                gpu=True,
            ),
        ),
        (
            "qwen3-8b-2k--base",
            "base",
            run_info(
                "qwen3-8b-2k--base",
                "openai",
                "Qwen/Qwen3-8B",
                "extract_v2",
                {
                    "temperature": 0.0,
                    "chat_template_kwargs": thinking_off,
                    "max_tokens": 1024,
                },
                stage1,
                gpu=True,
            ),
        ),
        (
            "qwen3-8b-2k--step-00025",
            "step-00025",
            run_info(
                "qwen3-8b-2k--step-00025",
                "openai",
                "qwen3-8b-2k--step-00025",
                "extract_v2",
                {
                    "temperature": 0.0,
                    "chat_template_kwargs": thinking_off,
                    "max_tokens": 1024,
                },
                stage1,
                gpu=True,
                links=links,
            ),
        ),
    ]
    for name, key, info in validation:
        save_run(root, name, info, key, stage1, printed)
    stopped = run_info(
        "2026-11-07T104935_gpt-6.1-sol_extract_v2",
        "openai",
        "gpt-6.1-sol",
        "extract_v2",
        {"reasoning_effort": "medium"},
        [],
        gpu=False,
        status="aborted",
    )
    save_run(
        root,
        "2026-11-07T104935_gpt-6.1-sol_extract_v2",
        stopped,
        "sol",
        [],
        printed,
        graded=False,
    )
    smoke = run_info(
        "smoke-gpt-luna",
        "openai",
        "gpt-6-luna",
        "extract_v1",
        {},
        stage1[:1],
        gpu=False,
    )
    save_run(root, "smoke-gpt-luna", smoke, "gpt-6-luna", stage1[:1], printed)

    # Test runs, named as results/test_results.json names them.
    results = json.loads((root / "results" / "test_results.json").read_text("utf-8"))
    test_runs: dict[str, list[str]] = {}
    for split in ("company_test", "time_test"):
        kept = [s for s in all_specs if s.split == split and not s.left_out]
        split_links = root / "data" / "splits" / split / "links.csv"
        test_runs[split] = []
        for model in results["models"]:
            name = results["splits"][split]["runs"][model]["run"]
            gpu = model in ("step-00025", "base")
            provider = "anthropic" if model.startswith("claude") else "openai"
            model_id = {
                "step-00025": "qwen3-8b-2k--step-00025",
                "base": "Qwen/Qwen3-8B",
            }.get(model, model)
            parameters = (
                {
                    "temperature": 0.0,
                    "chat_template_kwargs": thinking_off,
                    "max_tokens": 1024,
                }
                if gpu
                else {"reasoning_effort": "medium"}
            )
            info = run_info(
                name,
                provider,
                model_id,
                "extract_v2",
                parameters,
                kept,
                gpu=gpu,
                links=split_links,
            )
            save_run(root, name, info, model, kept, printed)
            test_runs[split].append(name)
    attempt = run_info(
        "gpt-6.1-sol--time_test",
        "openai",
        "gpt-6.1-sol",
        "extract_v2",
        {"reasoning_effort": "medium"},
        [s for s in all_specs if s.split == "time_test" and not s.left_out],
        gpu=False,
        links=root / "data" / "splits" / "time_test" / "links.csv",
    )
    save_run(
        root,
        "gpt-6.1-sol--time_test--attempt-1",
        attempt,
        "gpt-6.1-sol",
        [s for s in all_specs if s.split == "time_test" and not s.left_out],
        printed,
    )

    training_files(root)
    compare.main(
        [
            "--runs-dir",
            str(root / "runs"),
            "--out",
            str(root / "results" / "comparison.md"),
            "--links",
            str(root / "data" / "links.csv"),
            "--answers",
            str(root / "data" / "answers.csv"),
            "--gpu-hourly",
            "1",
            "--report-config",
            str(root / "configs" / "report" / "missing.toml"),
        ]
    )
    return MiniRepo(
        root=root,
        specs=all_specs,
        adapter_sha256=adapter_sha256,
        validation_runs=[name for name, _, _ in validation],
        test_runs=test_runs,
        test_results=results,
    )


def serving_results() -> dict:
    """A results/serving_results.json as secbench.servereport writes it, with
    synthetic numbers and no machine paths."""
    variants = {}
    for i, variant in enumerate(("adapter", "merged", "fp8", "int4-awq")):
        levels = [
            {
                "level": level,
                "started": "2026-11-10T00:53:56+00:00",
                "requests": 128,
                "ok": 128,
                "errors": 0,
                "makespan_s": 600.0 / level + 10 * i,
                "releases_per_hour": 128 * 3600 / (600.0 / level + 10 * i),
                "prompt_tokens": 1_500_000,
                "output_tokens": 17_000,
                "prompt_tokens_per_s": 2500.0 * level,
                "output_tokens_per_s": 28.0 * level,
                "latency_s": {
                    "p50": 4.0 + level,
                    "p95": 6.0 + level,
                    "p99": 7.0 + level,
                    "max": 8.0 + level,
                },
                "ttft_s": {"p50": 0.5 + level / 10, "p95": 0.9 + level / 10},
                "decode_tokens_per_s_p50": 60.0 - level - i,
                "warmup_errors": 0,
                "cost_per_1000_usd": 2.5 * (600.0 / level + 10 * i) / 3600 / 128 * 1000,
                "cost_per_1000_at_1_usd": (600.0 / level + 10 * i) / 3600 / 128 * 1000,
                "error_messages": [],
            }
            for level in (1, 16, 64)
        ]
        best = max(levels, key=lambda x: x["releases_per_hour"])
        variants[variant] = {
            "variant": variant,
            "served": f"qwen3-8b-2k--step-00025--{variant}",
            "status": "complete",
            "error": None,
            "model": None
            if variant == "adapter"
            else f"training/qwen3-8b-2k/serving/models/{variant}",
            "server_command": ["vllm", "serve", "MODEL"],
            "startup": {
                "weights_gib": [15.4, 15.3, 8.5, 5.6][i],
                "kv_cache_gib": [60.1, 60.2, 67.0, 70.0][i],
                "kv_cache_tokens": [486720, 487000, 542000, 566000][i],
                "max_concurrency": [11.88, 11.9, 13.2, 13.8][i],
                "max_concurrency_tokens": 40960,
                "engine_init_s": 40.0,
                "cuda_graphs_s": 12.0,
                "launch_to_healthy_s": [95.0, 80.0, 110.0, 85.0][i],
            },
            "load": None,
            "levels": levels,
            "best": {
                "level": best["level"],
                "releases_per_hour": best["releases_per_hour"],
                "latency_p95_s": best["latency_s"]["p95"],
                "cost_per_1000_usd": best["cost_per_1000_usd"],
                "cost_per_1000_at_1_usd": best["cost_per_1000_at_1_usd"],
            },
            "runs": {},
            "accuracy": {
                "validation": {
                    "correct": 287 - i,
                    "graded": 293,
                    "by_field": {
                        "revenue": [102, 102],
                        "net_income": [96 - i, 102],
                        "eps_diluted": [89, 89],
                    },
                    "not_correct": {"wrong_line": 6 + i},
                    "fallbacks": 0,
                },
                "time_test": {
                    "correct": 1699 - i,
                    "graded": 1740,
                    "by_field": {
                        "revenue": [583, 591],
                        "net_income": [545 - i, 577],
                        "eps_diluted": [571, 572],
                    },
                    "not_correct": {"wrong_line": 41 + i},
                    "fallbacks": 0,
                },
            },
            "agreement": {
                split: {
                    "with_adapter": None
                    if variant == "adapter"
                    else {
                        "releases": n,
                        "figures": 6 * n,
                        "same": 6 * n - i,
                        "releases_identical": n - i,
                        "period_end_differs": 0,
                        "differences": [],
                    },
                    "with_stage2": {
                        "releases": n,
                        "figures": 6 * n,
                        "same": 6 * n - i,
                        "releases_identical": n - i,
                        "period_end_differs": 0,
                        "differences": [],
                    },
                }
                for split, n in (("validation", 60), ("time_test", 346))
            },
        }
    return {
        "run": "qwen3-8b-2k",
        "checkpoint": "step-00025",
        "config": "configs/serving/qwen3-8b-2k.toml",
        "config_sha256": "0" * 64,
        "session": {
            "created": "2026-11-10T00:53:40+00:00",
            "finished": "2026-11-10T03:07:41+00:00",
            "commit": "3624f3c62c7e0000000000000000000000000000",
            "clean": True,
            "server": "vLLM 0.32.0",
            "gpus": ["NVIDIA H100 NVL, 95830 MiB, 580.178.04"],
            "gpu_hourly_usd": 2.5,
            "base_model": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        },
        "settings": {
            "levels": [1, 16, 64],
            "prompts": 128,
            "warmup": 8,
            "load_split": "time_test",
            "prefix_caching": False,
            "max_model_len": 40960,
            "accuracy_concurrency": 16,
        },
        "variants": variants,
        "models": {
            "merged": {
                "weight_gib": 15.26,
                "timings_s": {"total": 300.0},
                "versions": {},
                "settings": {},
                "quantization_config": None,
            },
            "int4-awq": {
                "weight_gib": 5.5,
                "timings_s": {"calibrate": 2400.0, "total": 2500.0},
                "versions": {"llmcompressor": "0.15.0", "compressed-tensors": "0.20.1"},
                "settings": {
                    "recipe": "AWQModifier, then QuantizationModifier (llm-compressor)",
                    "scheme": "W4A16_ASYM",
                    "targets": ["Linear"],
                    "ignore": ["lm_head"],
                    "duo_scaling": "both",
                    "n_grid": 20,
                    "max_seq_length": 16384,
                    "num_calibration_samples": 128,
                    "load_dtype": "bfloat16",
                    "cache_offload": None,
                    "retries": [],
                },
                "quantization_config": {"format": "pack-quantized"},
            },
        },
        "reference_accuracy": {
            "validation": {"correct": 287, "graded": 293},
            "time_test": {"correct": 1699, "graded": 1740},
        },
        "overhead": {
            "setup_steps": [],
            "server_starts_s": {
                "adapter": 95.0,
                "merged": 80.0,
                "fp8": 110.0,
                "int4-awq": 85.0,
            },
            "model_builds_s": {"merged": 300.0, "int4-awq": 2500.0},
            "starts_and_builds_s": 3170.0,
            "starts_and_builds_usd": 2.2,
        },
    }


def add_serving_results(root: Path) -> dict:
    """Writes results/serving_results.json (and a chart) into the mini repo."""
    data = serving_results()
    write_json(root / "results" / "serving_results.json", data)
    shutil.copyfile(
        REPO / "results" / "accuracy_vs_cost.png",
        root / "results" / "serving_throughput_latency.png",
    )
    return data
