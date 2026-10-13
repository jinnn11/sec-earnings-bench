import json
from pathlib import Path

import pytest

from secbench.inputs import (
    ReleaseInput,
    build_input,
    has_income_statement,
    is_revenue_row,
    release_records,
)
from secbench.link import check_earnings

FIXTURES = Path(__file__).parent / "fixtures"
RELEASES = FIXTURES / "input_releases"
XOM_Q2_2026 = RELEASES / "0002115436" / "0002115436-26-000006.json"
AAPL_Q2_FY25 = RELEASES / "0000320193" / "0000320193-25-000055.json"
CAT_Q2_2025 = RELEASES / "0000018230" / "0000018230-25-000037.json"


def table(*rows: list[str]) -> str:
    """A markdown table as secbench.releases writes it; the first row is the
    header."""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * len(rows[0])]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


HEADER = ["", "Three Months Ended June 30, 2026"]
REVENUE = ["Total revenues", "116,017"]
NET_INCOME = ["Net income", "14,525"]
PER_SHARE = ["Diluted earnings per share", "3.48"]
STATEMENT = table(HEADER, REVENUE, NET_INCOME, PER_SHARE)
SUMMARY = table(HEADER, ["Earnings (U.S. GAAP)", "14,525"], PER_SHARE)


# Income statement test


@pytest.mark.parametrize(
    "label",
    [
        "Total net sales",
        "Net sales",
        "Revenues",
        "Total revenue",
        "Total revenues and other income",
        "Sales and other operating revenue",
        "Total sales and revenues",
    ],
)
def test_revenue_labels(label) -> None:
    assert is_revenue_row([label, "1,234"])


@pytest.mark.parametrize(
    "row",
    [
        ["Cost of sales", "1,234"],
        ["Cost of revenue", "1,234"],
        ["Total revenues", ""],  # no figure
        ["Gross margin", "1,234"],
    ],
)
def test_not_revenue_rows(row) -> None:
    assert not is_revenue_row(row)


def test_statement_needs_revenue_net_income_and_per_share_rows() -> None:
    assert has_income_statement(STATEMENT)
    assert not has_income_statement(table(HEADER, NET_INCOME, PER_SHARE))
    assert not has_income_statement(table(HEADER, REVENUE, PER_SHARE))
    assert not has_income_statement(table(HEADER, REVENUE, NET_INCOME))


def test_statement_rows_must_share_a_table() -> None:
    markdown = table(HEADER, REVENUE) + "\n\n" + table(HEADER, NET_INCOME, PER_SHARE)
    assert not has_income_statement(markdown)


def test_header_row_does_not_count() -> None:
    assert not has_income_statement(table(REVENUE, NET_INCOME, PER_SHARE))


def test_stricter_than_the_link_filter() -> None:
    # ExxonMobil's Q2 2026 EX-99.1 has earnings and per-share rows in a summary
    # table, which is enough to link it, but its statements are in EX-99.2.
    assert check_earnings(SUMMARY).ok
    assert not has_income_statement(SUMMARY)
    markdown = (XOM_Q2_2026.parent / "0002115436-26-000006.md").read_text()
    assert check_earnings(markdown).ok
    assert not has_income_statement(markdown)


# Building inputs


def test_xom_q2_2026_appends_ex_99_2() -> None:
    release = (XOM_Q2_2026.parent / "0002115436-26-000006.md").read_text()
    statements = (XOM_Q2_2026.parent / "0002115436-26-000006_EX-99.2.md").read_text()
    built = build_input(XOM_Q2_2026)
    assert built.exhibits == ("EX-99.1", "EX-99.2")
    assert built.text == (
        f"{release.rstrip()}\n\n# EX-99.2: INVESTOR RELATIONS DATA SUMMARY\n\n"
        f"{statements}"
    )
    assert "| Total revenues and other income | 116,017 |" in built.text


@pytest.mark.parametrize("record", [AAPL_Q2_FY25, CAT_Q2_2025])
def test_release_with_statement_is_unchanged(record) -> None:
    # CAT also has an EX-99.2; it is not added.
    release = (record.parent / f"{record.stem}.md").read_text()
    assert build_input(record) == ReleaseInput(release, ("EX-99.1",))


def write_release(folder: Path, exhibits: list[tuple[str, str, str | None]]) -> Path:
    """A saved release whose first exhibit is the chosen one; each exhibit is (type,
    description, markdown or None for a non-HTML exhibit)."""
    folder.mkdir(parents=True, exist_ok=True)
    accession = "0000000001-26-000001"
    listed = []
    for i, (kind, description, markdown) in enumerate(exhibits):
        name = None
        if markdown is not None:
            name = f"{accession}.md" if i == 0 else f"{accession}_{kind}.md"
            (folder / name).write_text(markdown)
        listed.append(
            {
                "type": kind,
                "description": description,
                "url": f"https://www.sec.gov/{kind}.htm",
                "file": name,
                "chosen": i == 0,
            }
        )
    record = folder / f"{accession}.json"
    record.write_text(
        json.dumps(
            {"accession_number": accession, "exhibit": "EX-99.1", "exhibits": listed}
        )
    )
    return record


def test_first_other_exhibit_with_a_statement_is_appended(tmp_path) -> None:
    record = write_release(
        tmp_path / "0000000001",
        [
            ("EX-99.1", "PRESS RELEASE", "Record quarter.\n\n" + SUMMARY),
            ("EX-99.2", "SLIDES", "No tables here."),
            ("EX-99.3", "CHART", None),  # not HTML: no file
            ("EX-99.4", "", STATEMENT),
            ("EX-99.5", "MORE STATEMENTS", STATEMENT),
        ],
    )
    built = build_input(record)
    assert built.exhibits == ("EX-99.1", "EX-99.4")
    assert built.text == f"Record quarter.\n\n{SUMMARY}\n\n# EX-99.4\n\n{STATEMENT}"


def test_no_exhibit_with_a_statement(tmp_path) -> None:
    record = write_release(
        tmp_path / "0000000001",
        [("EX-99.1", "PRESS RELEASE", SUMMARY), ("EX-99.2", "SLIDES", "No tables.")],
    )
    assert build_input(record) == ReleaseInput(SUMMARY, ("EX-99.1",))


def test_chosen_exhibit_must_have_a_file(tmp_path) -> None:
    record = write_release(tmp_path / "0000000001", [("EX-99.1", "PDF", None)])
    with pytest.raises(ValueError, match="no single chosen exhibit with a file"):
        build_input(record)


def test_release_records_by_accession() -> None:
    assert release_records(RELEASES) == {
        "0000018230-25-000037": CAT_Q2_2025,
        "0000320193-25-000055": AAPL_Q2_FY25,
        "0000950170-25-100226": RELEASES / "0000789019" / "0000950170-25-100226.json",
        "0000104169-25-000069": RELEASES / "0000104169" / "0000104169-25-000069.json",
        "0002115436-26-000006": XOM_Q2_2026,
    }
