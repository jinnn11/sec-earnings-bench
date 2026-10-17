import contextlib
import csv
import dataclasses
import hashlib
import io
import json
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import anthropic
import httpx
import httpx2
import pytest
from anthropic.types import Message

from secbench import answers, extract
from secbench.answers import FIELDS, FULL_YEAR, QUARTER, YEAR_TO_DATE, load_links
from secbench.edgar import EdgarClient, RateLimiter
from secbench.extract import (
    API_ERROR,
    INVALID_REPLY,
    MALFORMED,
    OK,
    OK_AFTER_RETRY,
    RAW_DOLLARS,
    RETRY_MESSAGE,
    UNITS,
    UNKNOWN_UNIT,
    WITH_UNITS,
    AnthropicChat,
    FigureProblem,
    OpenAIChat,
    ReplyError,
    RunError,
    Settings,
    SettingsError,
    default_run_name,
    describe_parameters,
    estimated_cost,
    format_run_summary,
    load_prompt,
    load_settings,
    parse_reply,
    percentile,
    run_info,
    select_releases,
)
from secbench.grade import load_answers
from secbench.inputs import build_input, release_records

FIXTURES = Path(__file__).parent / "fixtures"
PROMPTS = Path(__file__).parent.parent / "prompts"
UA = "Test Runner test@example.com"
LINKS = FIXTURES / "answers_links.csv"
COMPANYFACTS = FIXTURES / "answers_companyfacts.json"
SUBMISSIONS = FIXTURES / "link_submissions.json"
RELEASES = FIXTURES / "input_releases"

# A released prompt is never edited; a changed prompt is a new version.
RELEASED_PROMPTS = {
    "extract_v1": "f72b85f60da4e5835404b7c0826053c3d8b6f745a5049650a5c699f4e5b02ac0",
    "extract_v2": "2e9e989384563ebf94db29e78604c0d78187767bcb688645fbafa3777c9ece29",
}

AAPL = "0000320193-25-000055"  # Q2 FY25
CAT = "0000018230-25-000037"  # Q2 2025
MSFT = "0000950170-25-100226"  # FY25, a 10-K
WMT = "0000104169-25-000069"  # Q1 FY26
XOM = "0002115436-26-000006"  # Q2 2026, statements in EX-99.2
FIXTURE_RELEASES = [AAPL, CAT, MSFT, WMT, XOM]  # links.csv order

KEY = "sk-test-secret"
ENVIRON = {
    "MODEL_BASE_URL": "https://models.test/v1/",
    "MODEL_NAME": "fake-model",
    "MODEL_API_KEY": KEY,
    "MODEL_TEMPERATURE": "0",
    "MODEL_PRICE_INPUT": "3",
    "MODEL_PRICE_OUTPUT": "15",
}
ENDPOINT = "https://models.test/v1/chat/completions"
SETTINGS = Settings(
    "https://models.test/v1/",
    "fake-model",
    KEY,
    temperature=0.0,
    price_input=3.0,
    price_output=15.0,
)

REPLY = {
    "period_end": "2025-03-29",
    "quarter": {
        "revenue": 95359000000,
        "net_income": 24780000000,
        "eps_diluted": 1.65,
    },
    "year_to_date": {"revenue": None, "net_income": None, "eps_diluted": None},
}


# Prompt


def test_released_prompts_are_unchanged() -> None:
    for version, sha256 in RELEASED_PROMPTS.items():
        text = (PROMPTS / f"{version}.txt").read_bytes()
        assert hashlib.sha256(text).hexdigest() == sha256, (
            f"prompts/{version}.txt was edited; restore it and add a new version"
        )


def test_load_prompt(tmp_path) -> None:
    prompt = load_prompt("extract_v1", PROMPTS)
    assert prompt.version == "extract_v1"
    assert prompt.sha256 == RELEASED_PROMPTS["extract_v1"]
    assert prompt.text.startswith("You read a US company's earnings press release")
    assert prompt.reply_format == RAW_DOLLARS
    v2 = load_prompt("extract_v2", PROMPTS)
    assert (v2.sha256, v2.reply_format) == (RELEASED_PROMPTS["extract_v2"], WITH_UNITS)
    with pytest.raises(RunError, match="bad prompt version"):
        load_prompt("../extract_v1", PROMPTS)
    with pytest.raises(RunError, match="extract_v9.txt not found"):
        load_prompt("extract_v9", PROMPTS)
    # A prompt without a reply format can't be run: its replies couldn't be read.
    (tmp_path / "extract_v9.txt").write_text("Reply in JSON.")
    with pytest.raises(RunError) as raised:
        load_prompt("extract_v9", tmp_path)
    assert str(raised.value) == (
        "no reply format for extract_v9; add it to REPLY_FORMATS in secbench/extract.py"
    )


def test_v2_changes_only_the_output_format() -> None:
    # Revenue and net income as printed with their unit; nothing else changes.
    v1_schema = """\
 "quarter": {"revenue": ..., "net_income": ..., "eps_diluted": ...},
 "year_to_date": {"revenue": ..., "net_income": ..., "eps_diluted": ...}}
"""
    v2_schema = """\
 "quarter": {"revenue": {"value": ..., "unit": ...},
             "net_income": {"value": ..., "unit": ...},
             "eps_diluted": ...},
 "year_to_date": {"revenue": {"value": ..., "unit": ...},
                  "net_income": {"value": ..., "unit": ...},
                  "eps_diluted": ...}}
"""
    v1_rule = """\
- Raw dollars: if a table says "in millions", 12,345 means 12345000000; if it says
  "in thousands", 1,234,567 means 1234567000. EPS stays in dollars per share (1.23).
"""
    v2_rule = """\
- Revenue and net income as printed, with the table's unit: if a table says "in
  millions", 12,345 is {"value": 12345, "unit": "millions"}; if it says "in
  thousands", 1,234,567 is {"value": 1234567, "unit": "thousands"}. The unit is
  "units" (whole dollars), "thousands", "millions" or "billions". EPS stays a plain
  number in dollars per share (1.23).
"""
    v1 = load_prompt("extract_v1", PROMPTS).text
    assert v1.count(v1_schema) == 1 and v1.count(v1_rule) == 1
    assert load_prompt("extract_v2", PROMPTS).text == v1.replace(
        v1_schema, v2_schema
    ).replace(v1_rule, v2_rule)


# Settings


def test_load_settings() -> None:
    settings = load_settings(ENVIRON)
    assert settings == SETTINGS
    assert settings.endpoint == ENDPOINT
    assert settings.parameters == {"temperature": 0.0}
    warm = load_settings({**ENVIRON, "MODEL_TEMPERATURE": " 0.7 "})
    assert warm.parameters == {"temperature": 0.7}
    local = load_settings(
        {"MODEL_BASE_URL": "http://localhost:8000/v1", "MODEL_NAME": "qwen"}
    )
    assert (local.api_key, local.price_input, local.price_output) == ("", None, None)
    # No temperature set: none is sent.
    assert local.temperature is None
    assert local.parameters == {}
    assert load_settings({**ENVIRON, "MODEL_TEMPERATURE": ""}).parameters == {}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"MODEL_BASE_URL": "", "MODEL_NAME": " "},
            "MODEL_BASE_URL and MODEL_NAME not set",
        ),
        ({"MODEL_BASE_URL": "ftp://hidden.example"}, "must start with http"),
        ({"MODEL_PRICE_INPUT": "three"}, "MODEL_PRICE_INPUT is not a number"),
        ({"MODEL_PRICE_OUTPUT": "-1"}, "MODEL_PRICE_OUTPUT must be 0 or more"),
        ({"MODEL_PROVIDER": "gemini"}, "MODEL_PROVIDER must be openai or anthropic"),
        ({"MODEL_TEMPERATURE": "warm"}, "MODEL_TEMPERATURE is not a number"),
        ({"MODEL_TEMPERATURE": "-0.5"}, "MODEL_TEMPERATURE must be 0 or more"),
        ({"MODEL_TEMPERATURE": "nan"}, "MODEL_TEMPERATURE must be 0 or more"),
        (
            {"MODEL_EXTRA_BODY": "{reasoning_effort: medium}"},
            "MODEL_EXTRA_BODY is not valid JSON",
        ),
        ({"MODEL_EXTRA_BODY": '{"top_p": NaN}'}, "MODEL_EXTRA_BODY is not valid JSON"),
        ({"MODEL_EXTRA_BODY": '["medium"]'}, "MODEL_EXTRA_BODY must be a JSON object"),
        ({"MODEL_GPU_HOURLY": "cheap"}, "MODEL_GPU_HOURLY is not a number"),
        ({"MODEL_GPU_HOURLY": "-2.5"}, "MODEL_GPU_HOURLY must be 0 or more"),
        (
            {"MODEL_GPU_HOURLY": "2.5"},  # ENVIRON has token prices
            "set MODEL_GPU_HOURLY or MODEL_PRICE_INPUT and MODEL_PRICE_OUTPUT, not both",
        ),
        ({"MODEL_CONCURRENCY": "four"}, "MODEL_CONCURRENCY is not a whole number"),
        ({"MODEL_CONCURRENCY": "2.5"}, "MODEL_CONCURRENCY is not a whole number"),
        ({"MODEL_CONCURRENCY": "0"}, "MODEL_CONCURRENCY must be 1 or more"),
        ({"MODEL_CONCURRENCY": "-3"}, "MODEL_CONCURRENCY must be 1 or more"),
        (
            {"MODEL_EXTRA_BODY": '{"stream": true, "model": "other-model"}'},
            "MODEL_EXTRA_BODY can't set model or stream ",
        ),
        (
            {"MODEL_EXTRA_BODY": '{"temperature": 0.2}'},
            "MODEL_EXTRA_BODY can't set temperature ",
        ),
    ],
)
def test_bad_settings_name_the_setting_not_its_value(changes, message) -> None:
    with pytest.raises(SettingsError, match=message) as raised:
        load_settings({**ENVIRON, **changes})
    for value in changes.values():
        if value.strip():
            assert value not in str(raised.value)


def test_extra_body() -> None:
    extra = '{"reasoning_effort": "medium", "max_tokens": 4096}'
    settings = load_settings({**ENVIRON, "MODEL_EXTRA_BODY": f" {extra} "})
    assert settings.extra_body == {"reasoning_effort": "medium", "max_tokens": 4096}
    assert settings.parameters == {
        "temperature": 0.0,
        "reasoning_effort": "medium",
        "max_tokens": 4096,
    }
    assert describe_parameters("openai", settings.parameters) == (
        "temperature 0, reasoning_effort medium, max_tokens 4096"
    )
    # Blank or unset: nothing more is sent.
    for environ in (ENVIRON, {**ENVIRON, "MODEL_EXTRA_BODY": " "}):
        assert load_settings(environ).extra_body is None
        assert load_settings(environ).parameters == {"temperature": 0.0}
    vllm = load_settings(
        {
            "MODEL_BASE_URL": "http://localhost:8000/v1",
            "MODEL_NAME": "qwen",
            "MODEL_EXTRA_BODY": '{"chat_template_kwargs": {"enable_thinking": false}}',
        }
    )
    assert vllm.parameters == {"chat_template_kwargs": {"enable_thinking": False}}
    assert describe_parameters("openai", vllm.parameters) == (
        'chat_template_kwargs {"enable_thinking": false}, no temperature'
    )


def test_concurrency() -> None:
    assert load_settings(ENVIRON).concurrency == 4  # unset
    assert load_settings({**ENVIRON, "MODEL_CONCURRENCY": " "}).concurrency == 4
    assert load_settings({**ENVIRON, "MODEL_CONCURRENCY": " 16 "}).concurrency == 16
    # Both providers.
    anthropic_two = {**ANTHROPIC_ENVIRON, "MODEL_CONCURRENCY": "2"}
    assert load_settings(anthropic_two).concurrency == 2


def test_anthropic_ignores_extra_body() -> None:
    environ = {**ANTHROPIC_ENVIRON, "MODEL_EXTRA_BODY": "{not json"}
    assert load_settings(environ) == ANTHROPIC_SETTINGS


def test_env_file_is_the_only_source(tmp_path) -> None:
    env_file = tmp_path / ".env.vllm"
    env_file.write_text(
        "# a local vLLM server\n"
        'MODEL_BASE_URL="http://localhost:8000/v1"\n'
        "MODEL_NAME=qwen3-4b\n"
        'MODEL_API_KEY="key-${HOME}"\n'
        "MODEL_TEMPERATURE=\n"
    )
    # ENVIRON's settings are not used, even those the file leaves out (prices); the
    # file is read literally.
    assert load_settings(ENVIRON, env_file) == Settings(
        "http://localhost:8000/v1", "qwen3-4b", "key-${HOME}"
    )


def test_env_file_problems(tmp_path) -> None:
    with pytest.raises(SettingsError, match="none.env not found"):
        load_settings(ENVIRON, tmp_path / "none.env")
    env_file = tmp_path / ".env.openai"
    env_file.write_text("MODEL_NAME=gpt-test\n")
    with pytest.raises(
        SettingsError, match=f"MODEL_BASE_URL not set; add to {env_file} "
    ):
        load_settings(ENVIRON, env_file)


# Replies


def reply_text(**changes) -> str:
    return json.dumps({**REPLY, **changes})


@pytest.mark.parametrize(
    "content",
    [
        reply_text(),
        f"```json\n{reply_text()}\n```",
        f"```\n{reply_text()}\n```\n",
        f"<think>The quarter ends March 29. {{}}</think>\n\n{reply_text()}",
    ],
)
def test_parse_reply(content) -> None:
    parsed = parse_reply(content, AAPL)
    assert (parsed.units, parsed.problems) == ((), ())
    prediction = parsed.prediction
    assert prediction.release_accession == AAPL
    assert str(prediction.period_end) == "2025-03-29"
    # Exact decimals, not floats.
    assert prediction.quarter.eps_diluted == Decimal("1.65")
    assert str(prediction.quarter.eps_diluted) == "1.65"
    assert prediction.quarter.revenue == Decimal(95359000000)
    assert prediction.year_to_date.revenue is None


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "the reply is empty"),
        (None, "the reply is empty"),
        (
            f"Here are the figures: {reply_text()}",
            "not valid JSON (Expecting value: line 1 column 1 (char 0))",
        ),
        ("[]", "not a JSON object"),
        (
            reply_text(release_accession=AAPL),
            "release_accession: Extra inputs are not permitted",
        ),
        (
            reply_text(quarter={**REPLY["quarter"], "revenue": "95,359"}),
            'quarter.revenue: Input should be a valid decimal (got "95,359")',
        ),
        (
            reply_text().replace("95359000000", "NaN"),
            "quarter.revenue: Input should be a finite number (got nan)",
        ),
        (
            reply_text(quarter={**REPLY["quarter"], "gross_margin": 0.47}),
            "quarter.gross_margin: Extra inputs are not permitted (got 0.47)",
        ),
        (
            json.dumps({"period_end": "2025-03-29", "quarter": REPLY["quarter"]}),
            "year_to_date: Field required",
        ),
        (
            reply_text(period_end="March 29, 2025"),
            (
                "period_end: Input should be a valid date or datetime, invalid "
                'character in year (got "March 29, 2025")'
            ),
        ),
    ],
)
def test_parse_reply_errors(content, message) -> None:
    with pytest.raises(ReplyError) as raised:
        parse_reply(content, AAPL)
    assert str(raised.value) == message
    assert raised.value.salvage is None


# Replies with units (extract_v2)

V2_REPLY = {
    "period_end": "2025-03-29",
    "quarter": {
        "revenue": {"value": 95359, "unit": "millions"},
        "net_income": {"value": 24780, "unit": "millions"},
        "eps_diluted": 1.65,
    },
    "year_to_date": {"revenue": None, "net_income": None, "eps_diluted": None},
}
SHOULD_BE_FIGURE = 'should be null or {"value": <number>, "unit": <unit>}'
SHOULD_BE_UNIT = 'should be "units", "thousands", "millions" or "billions"'
Q = "quarter.revenue"


def v2_text(revenue) -> str:
    """V2_REPLY with this quarter revenue, as JSON."""
    quarter = {**V2_REPLY["quarter"], "revenue": revenue}
    return json.dumps({**V2_REPLY, "quarter": quarter})


@pytest.mark.parametrize(
    ("value", "unit", "dollars"),
    [
        ("95359000000", "units", "95359000000"),
        ("-1234.56", "units", "-1234.56"),
        ("95359000", "thousands", "95359000000"),
        ("-2179000", "thousands", "-2179000000"),
        ("1.1", "thousands", "1100"),  # 1100.0000000000002 through a float
        ("95359", "millions", "95359000000"),
        ("-95359", "millions", "-95359000000"),
        ("0", "millions", "0"),
        ("95.359", "billions", "95359000000"),
        ("-0.29", "billions", "-290000000"),
    ],
)
def test_parse_reply_with_units(value, unit, dollars) -> None:
    content = v2_text({"value": "VALUE", "unit": unit}).replace('"VALUE"', value)
    parsed = parse_reply(content, AAPL, WITH_UNITS)
    quarter = parsed.prediction.quarter
    assert isinstance(quarter.revenue, Decimal)
    assert quarter.revenue == Decimal(dollars)  # exactly
    assert quarter.net_income == Decimal(24780000000)
    assert quarter.eps_diluted == Decimal("1.65")  # a plain number, as in v1
    assert parsed.units == (unit, "millions")
    assert parsed.problems == ()


@pytest.mark.parametrize(
    "revenue",
    [
        None,
        {"value": None, "unit": "millions"},
        {"value": None},
        {"value": None, "unit": "n/a", "note": "not reported"},
    ],
)
def test_a_null_value_is_null(revenue) -> None:
    parsed = parse_reply(v2_text(revenue), AAPL, WITH_UNITS)
    assert parsed.prediction.quarter.revenue is None
    assert (parsed.units, parsed.problems) == (("millions",), ())


@pytest.mark.parametrize(
    ("revenue", "kind", "message"),
    [
        # Raw dollars, as extract_v1 asks.
        (95359000000, MALFORMED, f"{Q}: {SHOULD_BE_FIGURE} (got 95359000000)"),
        ("95,359", MALFORMED, f'{Q}: {SHOULD_BE_FIGURE} (got "95,359")'),
        (
            [95359, "millions"],
            MALFORMED,
            f'{Q}: {SHOULD_BE_FIGURE} (got [95359, "millions"])',
        ),
        (
            {"value": 95359},
            MALFORMED,
            f'{Q}: {SHOULD_BE_FIGURE} (got {{"value": 95359}})',
        ),
        (
            {"value": 95359, "unit": "millions", "label": "Total"},
            MALFORMED,
            (
                f'{Q}: {SHOULD_BE_FIGURE} (got {{"value": 95359, "unit": "millions", '
                '"label": "Total"})'
            ),
        ),
        (
            {"value": "95,359", "unit": "millions"},
            MALFORMED,
            f'{Q}.value: should be a JSON number (got "95,359")',
        ),
        (
            {"value": True, "unit": "millions"},
            MALFORMED,
            f"{Q}.value: should be a JSON number (got true)",
        ),
        (
            {"value": 95359, "unit": None},
            MALFORMED,
            f"{Q}.unit: {SHOULD_BE_UNIT} (got null)",
        ),
        (
            {"value": 95359, "unit": 1000000},
            MALFORMED,
            f"{Q}.unit: {SHOULD_BE_UNIT} (got 1000000)",
        ),
        # Units are matched exactly.
        *(
            (
                {"value": 95359, "unit": unit},
                UNKNOWN_UNIT,
                f"{Q}.unit: {SHOULD_BE_UNIT} (got {json.dumps(unit)})",
            )
            for unit in ("million", "Millions", "in millions", " millions", "USD", "")
        ),
    ],
)
def test_bad_figures(revenue, kind, message) -> None:
    with pytest.raises(ReplyError) as raised:
        parse_reply(v2_text(revenue), AAPL, WITH_UNITS)
    assert str(raised.value) == message
    # The rest of the reply is fine: salvage has it, with the bad figure null.
    salvage = raised.value.salvage
    assert salvage.problems == (
        FigureProblem("quarter.revenue", kind, message, json.dumps(revenue)),
    )
    assert salvage.prediction.quarter.revenue is None
    assert salvage.prediction.quarter.net_income == Decimal(24780000000)
    assert salvage.units == ("millions",)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_a_value_that_is_not_finite_is_malformed(constant) -> None:
    content = v2_text({"value": 0, "unit": "millions"}).replace(
        '"value": 0', f'"value": {constant}'
    )
    with pytest.raises(ReplyError) as raised:
        parse_reply(content, AAPL, WITH_UNITS)
    message = f"quarter.revenue.value: should be a JSON number (got {constant})"
    assert str(raised.value) == message
    [problem] = raised.value.salvage.problems
    assert (problem.kind, problem.replied) == (
        MALFORMED,
        f'{{"value": {constant}, "unit": "millions"}}',
    )


def test_every_bad_figure_is_reported() -> None:
    reply = {
        "period_end": "2025-03-29",
        "quarter": {
            "revenue": {"value": 95359, "unit": "million"},
            "net_income": 24780000000,
            "eps_diluted": 1.65,
        },
        "year_to_date": {
            "revenue": {"value": 219.659, "unit": "billions"},
            "net_income": {"value": 61110, "unit": "Millions"},
            "eps_diluted": 4.05,
        },
    }
    with pytest.raises(ReplyError) as raised:
        parse_reply(json.dumps(reply), AAPL, WITH_UNITS)
    assert str(raised.value) == "; ".join(
        [
            f'quarter.revenue.unit: {SHOULD_BE_UNIT} (got "million")',
            f"quarter.net_income: {SHOULD_BE_FIGURE} (got 24780000000)",
            f'year_to_date.net_income.unit: {SHOULD_BE_UNIT} (got "Millions")',
        ]
    )
    salvage = raised.value.salvage
    assert [(p.figure, p.kind) for p in salvage.problems] == [
        ("quarter.revenue", UNKNOWN_UNIT),
        ("quarter.net_income", MALFORMED),
        ("year_to_date.net_income", UNKNOWN_UNIT),
    ]
    assert salvage.prediction.year_to_date.revenue == Decimal(219659000000)
    assert salvage.units == ("billions",)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        # EPS stays a plain number.
        (
            json.dumps(
                {
                    **V2_REPLY,
                    "quarter": {
                        **V2_REPLY["quarter"],
                        "eps_diluted": {"value": 1.65, "unit": "units"},
                    },
                }
            ),
            (
                "quarter.eps_diluted: Decimal input should be an integer, float, "
                "string or Decimal object"
            ),
        ),
        (
            json.dumps(
                {**V2_REPLY, "quarter": {**V2_REPLY["quarter"], "gross_margin": 0.47}}
            ),
            "quarter.gross_margin: Extra inputs are not permitted (got 0.47)",
        ),
        # A bad figure with another problem: both are reported, nothing is salvaged.
        (
            json.dumps(
                {
                    **json.loads(v2_text({"value": 95359, "unit": "million"})),
                    "period_end": "March 29, 2025",
                }
            ),
            (
                f'quarter.revenue.unit: {SHOULD_BE_UNIT} (got "million"); period_end: '
                "Input should be a valid date or datetime, invalid character in year "
                '(got "March 29, 2025")'
            ),
        ),
        (
            json.dumps({"period_end": "2025-03-29", "quarter": V2_REPLY["quarter"]}),
            "year_to_date: Field required",
        ),
    ],
)
def test_other_problems_are_not_salvaged(content, message) -> None:
    with pytest.raises(ReplyError) as raised:
        parse_reply(content, AAPL, WITH_UNITS)
    assert str(raised.value) == message
    assert raised.value.salvage is None


def test_extract_v1_replies_have_no_units() -> None:
    # A figure with a unit is no number in extract_v1's format.
    with pytest.raises(ReplyError) as raised:
        parse_reply(json.dumps(V2_REPLY), AAPL)
    assert str(raised.value) == (
        "quarter.revenue: Decimal input should be an integer, float, string or "
        "Decimal object; quarter.net_income: Decimal input should be an integer, "
        "float, string or Decimal object"
    )
    assert raised.value.salvage is None


# Small pieces


def test_percentile() -> None:
    assert percentile([3.0, 1.0, 2.0, 4.0], 50) == 2.0
    assert percentile([3.0, 1.0, 2.0, 4.0], 95) == 4.0
    assert percentile([float(n) for n in range(1, 21)], 95) == 19.0
    assert percentile([5.0], 50) == 5.0
    assert percentile([], 50) is None


def test_estimated_cost() -> None:
    # 1,000 input tokens at $3 and 100 output tokens at $15 per million.
    assert estimated_cost(1000, 100, SETTINGS) == pytest.approx(0.0045)
    unpriced = Settings("https://models.test/v1", "fake-model", price_input=3.0)
    assert estimated_cost(1000, 100, unpriced) is None


VLLM_ENVIRON = {
    "MODEL_BASE_URL": "https://models.test/v1/",
    "MODEL_NAME": "qwen3-4b",
    "MODEL_GPU_HOURLY": "2.5",
}


def cost_info(settings: Settings, seconds: float) -> dict:
    started = datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC)
    return run_info(
        run_name="cost",
        status="complete",
        settings=settings,
        prompt=load_prompt("extract_v1", PROMPTS),
        started=started,
        finished=started + timedelta(seconds=seconds),
        selection={},
        results=[],
    )


def test_gpu_hour_cost() -> None:
    settings = load_settings(VLLM_ENVIRON)
    assert (settings.gpu_hourly, settings.price_input) == (2.5, None)
    info = cost_info(settings, 72)
    # $2.50 per hour for 72 s: 2.5 x 72 / 3,600 = $0.05.
    assert (info["cost_basis"], info["gpu_hourly_usd"], info["wall_time_s"]) == (
        "gpu_hours",
        2.5,
        72.0,
    )
    assert info["estimated_cost_usd"] == 0.05
    assert (
        "estimated cost: $0.0500 ($2.5 per GPU-hour for 72.0 s of wall-clock time)"
        in format_run_summary(info, Path("runs/cost")).splitlines()
    )
    # 90 minutes at $2.50 per hour: $3.75.
    assert cost_info(settings, 5400)["estimated_cost_usd"] == 3.75


def test_token_cost_basis() -> None:
    info = cost_info(SETTINGS, 72)
    assert (info["cost_basis"], info["gpu_hourly_usd"]) == ("tokens", None)
    assert info["estimated_cost_usd"] == 0.0  # no calls
    unpriced = cost_info(load_settings({**VLLM_ENVIRON, "MODEL_GPU_HOURLY": ""}), 72)
    assert (unpriced["cost_basis"], unpriced["estimated_cost_usd"]) == (None, None)


def test_anthropic_ignores_gpu_hourly() -> None:
    environ = {**ANTHROPIC_ENVIRON, "MODEL_GPU_HOURLY": "2.5"}  # with token prices
    assert load_settings(environ) == ANTHROPIC_SETTINGS


@pytest.mark.parametrize(
    ("model", "slug"),
    [
        ("Qwen/Qwen3-8B", "Qwen--Qwen3-8B"),
        ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama--Llama-3.1-8B-Instruct"),
        ("/models/Qwen3-8B", "models--Qwen3-8B"),  # vLLM serving a local path
        ("qwen3:8b", "qwen3-8b"),
        ("claude-opus-5-5", "claude-opus-5-5"),
        ("///", "model"),
    ],
)
def test_default_run_name(model, slug) -> None:
    now = datetime(2026, 10, 15, 12, 0, 5, tzinfo=UTC)
    name = default_run_name(model, load_prompt("extract_v1", PROMPTS), now)
    assert name == f"2026-10-15T120005_{slug}_extract_v1"
    assert extract.NAME.fullmatch(name)


def test_select_releases() -> None:
    linked = load_links(LINKS)[0]
    accessions = [r.release_accession for r in linked]
    assert [r.release_accession for r in select_releases(linked)] == accessions
    xom = select_releases(linked, tickers=["xom"])
    assert len(xom) == 6 and {r.ticker for r in xom} == {"XOM"}
    assert [r.release_accession for r in select_releases(linked, limit=2)] == (
        accessions[:2]
    )
    chosen = select_releases(linked, accessions=[XOM, AAPL])
    assert [r.release_accession for r in chosen] == [AAPL, XOM]  # links.csv order
    assert select_releases(linked, tickers=["AAPL"], accessions=[XOM, AAPL]) == [
        chosen[0]
    ]
    with pytest.raises(RunError, match="no linked releases for ZZZZ"):
        select_releases(linked, tickers=["AAPL", "ZZZZ"])
    with pytest.raises(RunError, match="not linked: 0000000000-00-000000"):
        select_releases(linked, accessions=["0000000000-00-000000"])
    with pytest.raises(RunError, match="no linked release matches every filter"):
        select_releases(linked, tickers=["AAPL"], accessions=[XOM])


# HTTP retries (the OpenAI-compatible client alone)


def completion(content: str, input_tokens=1000, output_tokens=100) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        },
    )


def scripted_client(responses: list, settings: Settings = SETTINGS):
    """An OpenAIChat whose server answers with responses in order (an exception is
    raised instead), and the delays it slept."""
    queue = list(responses)
    requests: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    chat = OpenAIChat(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=slept.append,
        wall_clock=lambda: 0.0,
    )
    return chat, requests, slept


BODY = {"model": "fake-model", "messages": [], "temperature": 0}


def test_chat_client_sends_the_body_and_key() -> None:
    chat, requests, slept = scripted_client([completion("{}")])
    exchange = chat.complete(BODY)
    assert exchange.content == "{}"
    assert exchange.tokens == (1000, 100)
    assert exchange.finish_reason == "stop"
    assert [a["status"] for a in exchange.attempts] == [200]
    assert str(requests[0].url) == ENDPOINT
    assert json.loads(requests[0].content) == BODY
    assert requests[0].headers["Authorization"] == f"Bearer {KEY}"
    assert slept == []


def test_chat_client_without_a_key_sends_no_authorization() -> None:
    local = Settings("http://localhost:8000/v1", "qwen")
    chat, requests, _ = scripted_client([completion("{}")], local)
    chat.complete(BODY)
    assert str(requests[0].url) == "http://localhost:8000/v1/chat/completions"
    assert "Authorization" not in requests[0].headers


def test_chat_client_honors_retry_after() -> None:
    chat, _, slept = scripted_client(
        [httpx.Response(429, headers={"Retry-After": "3"}), completion("{}")]
    )
    exchange = chat.complete(BODY)
    assert exchange.error == ""
    assert [a["status"] for a in exchange.attempts] == [429, 200]
    assert slept == [3.0]


def test_chat_client_backs_off_on_5xx_and_connection_errors() -> None:
    chat, _, slept = scripted_client(
        [
            httpx.Response(500),
            httpx.ConnectError("connection refused"),
            httpx.Response(503),
            completion("{}"),
        ]
    )
    exchange = chat.complete(BODY)
    assert exchange.content == "{}"
    assert [a["status"] for a in exchange.attempts] == [500, None, 503, 200]
    assert exchange.attempts[1]["error"] == "ConnectError: connection refused"
    assert len(slept) == 3
    for delay, base in zip(slept, (1, 2, 4), strict=True):
        assert base <= delay <= 1.25 * base


def test_chat_client_gives_up_after_5_retries() -> None:
    chat, requests, slept = scripted_client(
        [httpx.Response(502, json={"error": {"message": "Bad gateway"}})] * 6
    )
    exchange = chat.complete(BODY)
    assert exchange.reply is None and not exchange.abort
    assert exchange.error == "HTTP 502 (Bad gateway), after 6 attempts"
    assert len(requests) == 6 and len(slept) == 5


def test_chat_client_gives_up_on_a_long_retry_after() -> None:
    chat, requests, slept = scripted_client(
        [httpx.Response(429, headers={"Retry-After": "300"})]
    )
    exchange = chat.complete(BODY)
    assert exchange.error == "HTTP 429, and the server asked to wait 300s"
    assert len(requests) == 1 and slept == []


def test_chat_client_does_not_retry_other_4xx() -> None:
    message = "This model's maximum context length is 8192 tokens."
    chat, requests, slept = scripted_client(
        [httpx.Response(400, json={"error": {"message": message}})]
    )
    exchange = chat.complete(BODY)
    assert exchange.error == f"HTTP 400 ({message})"
    assert not exchange.abort
    assert exchange.attempts[0]["response"] == {"error": {"message": message}}
    assert len(requests) == 1 and slept == []


@pytest.mark.parametrize("status", [401, 403, 404])
def test_chat_client_aborts_on_a_wrong_key_model_or_url(status) -> None:
    body = {"error": {"message": "Incorrect API key provided: sk-test-****cret."}}
    chat, requests, slept = scripted_client([httpx.Response(status, json=body)])
    exchange = chat.complete(BODY)
    assert exchange.abort
    assert exchange.error == (
        f"HTTP {status}; check MODEL_BASE_URL, MODEL_NAME and MODEL_API_KEY"
    )
    # The body may echo part of the key, so it isn't kept.
    assert [(a["status"], a["response"]) for a in exchange.attempts] == [(status, None)]
    assert len(requests) == 1 and slept == []


def test_chat_client_rejects_a_body_without_choices() -> None:
    chat, _, _ = scripted_client([httpx.Response(200, json={"error": "overloaded"})])
    exchange = chat.complete(BODY)
    assert exchange.reply is None
    assert exchange.error == "HTTP 200 without choices[0].message"


# Runs on the input_releases fixtures, with a fake model server


def make_client(cache_dir: Path) -> EdgarClient:
    """An EDGAR client serving link_submissions.json and answers_companyfacts.json;
    other URLs 404."""
    bodies = json.loads(SUBMISSIONS.read_bytes()) | json.loads(
        COMPANYFACTS.read_bytes()
    )
    routes = {url: json.dumps(body).encode() for url, body in bodies.items()}

    def handler(request: httpx.Request) -> httpx.Response:
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


@pytest.fixture(scope="module")
def key_dir(tmp_path_factory) -> Path:
    """A folder with answers.csv built from the answers fixtures."""
    tmp = tmp_path_factory.mktemp("key")
    with contextlib.redirect_stdout(io.StringIO()):
        status = answers.main(
            ["--links", str(LINKS), "--out", str(tmp / "answers.csv")],
            client=make_client(tmp / "raw"),
        )
    assert status == 0
    return tmp


def number(value: Decimal) -> int | float:
    """A JSON number for value (1.65 is written 1.65)."""
    return int(value) if value == value.to_integral_value() else float(value)


@pytest.fixture(scope="module")
def perfect_replies(key_dir) -> dict[str, dict]:
    """Release accession -> a reply that gives every answer in the key."""
    key = load_answers(key_dir / "answers.csv")
    replies = {}
    for release in load_links(LINKS)[0]:
        if release.release_accession not in FIXTURE_RELEASES:
            continue
        year = FULL_YEAR if release.linked_form == "10-K" else YEAR_TO_DATE

        def group(period: str, accession=release.release_accession) -> dict:
            return {
                f.name: number(a.value)
                if (a := key.get((accession, f.name, period)))
                else None
                for f in FIELDS
            }

        replies[release.release_accession] = {
            "period_end": str(release.period_end),
            "quarter": group(QUARTER),
            "year_to_date": group(year),
        }
    return replies


class FakeModel:
    """An OpenAI-compatible chat completions server. script(accession, call,
    body) gives each response; call counts the release's requests from 1."""

    def __init__(self, script: Callable[[str, int, dict], httpx.Response]) -> None:
        records = release_records(RELEASES)
        self.accessions = {build_input(records[a]).text: a for a in FIXTURE_RELEASES}
        self.script = script
        self.requests: list[httpx.Request] = []
        self.calls: Counter[str] = Counter()
        self._lock = threading.Lock()

    def accession(self, request: httpx.Request) -> str:
        return self.accessions[json.loads(request.content)["messages"][1]["content"]]

    def requests_for(self, accession: str) -> list[httpx.Request]:
        return [r for r in self.requests if self.accession(r) == accession]

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and str(request.url) == ENDPOINT
        body = json.loads(request.content)
        accession = self.accession(request)
        with self._lock:
            self.requests.append(request)
            self.calls[accession] += 1
            call = self.calls[accession]
        return self.script(accession, call, body)


def run(
    tmp_path: Path,
    key_dir: Path,
    model: FakeModel | None,
    *args: str,
    environ=ENVIRON,
    anthropic_client=None,
) -> int:
    transport = httpx.MockTransport(model.handler) if model else None
    argv = [
        "--accessions",
        *FIXTURE_RELEASES,
        "--run-name",
        "test",
        "--links",
        str(LINKS),
        "--answers",
        str(key_dir / "answers.csv"),
        "--releases",
        str(RELEASES),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--prompts-dir",
        str(PROMPTS),
        *args,
    ]
    return extract.main(
        argv,
        transport=transport,
        environ=environ,
        anthropic_client=anthropic_client,
        edgar_client=make_client(tmp_path / "raw"),
        sleep=lambda s: None,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_with_a_perfect_model(tmp_path, key_dir, perfect_replies, capsys) -> None:
    model = FakeModel(lambda a, call, body: completion(json.dumps(perfect_replies[a])))
    assert run(tmp_path, key_dir, model) == 0
    out = capsys.readouterr().out
    run_dir = tmp_path / "runs" / "test"

    # Every answer for the five releases is in its input, and all are correct: AAPL,
    # CAT and XOM quarter and year to date (6 each), MSFT full year (3) and derived
    # fourth quarter (revenue, net income), WMT Q1 quarter (3).
    key = load_answers(key_dir / "answers.csv")
    answerable = sum(1 for a in key.values() if a.release_accession in FIXTURE_RELEASES)
    assert answerable == 26
    assert f"accuracy: {answerable}/{answerable} (100.0%)" in out.splitlines()
    assert "graded 5 releases" in out

    predictions = read_jsonl(run_dir / "predictions.jsonl")
    assert [p["release_accession"] for p in predictions] == FIXTURE_RELEASES
    assert predictions[0]["quarter"]["eps_diluted"] == "1.65"
    assert (run_dir / "graded.csv").is_file() and (run_dir / "summary.txt").is_file()

    info = json.loads((run_dir / "run.json").read_text())
    assert info["status"] == "complete"
    assert info["provider"] == "openai"
    assert info["model"] == "fake-model"
    assert info["env_file"] is None
    assert info["extra_body"] is None
    assert info["prompt_version"] == "extract_v1"
    assert info["prompt_sha256"] == RELEASED_PROMPTS["extract_v1"]
    assert (info["reply_format"], info["figures_by_unit"]) == (RAW_DOLLARS, None)
    assert info["format_nulls"] == []
    assert info["request_parameters"] == {"temperature": 0.0}
    for request in model.requests:
        assert json.loads(request.content)["temperature"] == 0
    assert info["releases"] == 5
    assert info["outcomes"] == {
        OK: 5,
        OK_AFTER_RETRY: 0,
        INVALID_REPLY: 0,
        API_ERROR: 0,
        "aborted": 0,
    }
    assert info["fallbacks"] == []
    assert info["tokens"] == {"input": 5000, "output": 500, "calls_without_usage": 0}
    # 5,000 input tokens at $3 and 500 output tokens at $15 per million.
    assert info["estimated_cost_usd"] == 0.0225
    assert [c["release_accession"] for c in info["calls"]] == FIXTURE_RELEASES
    assert all(c["status"] == 200 and c["http_attempts"] == 1 for c in info["calls"])
    assert all(isinstance(c["latency_s"], float) for c in info["calls"])
    assert info["latency_s"]["p50"] is not None

    raw = json.loads((run_dir / "raw" / f"{XOM}.json").read_text())
    assert raw["outcome"] == OK
    assert raw["format_nulls"] == []
    assert raw["input_exhibits"] == ["EX-99.1", "EX-99.2"]
    [call] = raw["calls"]
    records = release_records(RELEASES)
    assert call["request"] == {
        "model": "fake-model",
        "messages": [
            {"role": "system", "content": load_prompt("extract_v1", PROMPTS).text},
            {"role": "user", "content": build_input(records[XOM]).text},
        ],
        "temperature": 0,
    }
    assert call["attempts"][0]["response"]["choices"][0]["message"]["content"] == (
        json.dumps(perfect_replies[XOM])
    )
    assert "estimated cost: $0.0225 ($3 input, $15 output per million tokens)" in out
    assert "provider openai; sent with each request: temperature 0" in (
        out.splitlines()
    )
    assert (
        "runs/test: 5 releases, provider openai, prompt extract_v1, temperature 0, "
        "up to 4 calls" in out
    )
    assert "invalid replies (all-null predictions, graded as missing): 0" in out
    assert "API errors (all-null predictions, graded as missing): 0" in out
    # No units in extract_v1.
    assert not any(
        line.startswith(("figures with a unit", "format nulls"))
        for line in out.splitlines()
    )

    # The key is sent, and never saved.
    assert model.requests[0].headers["Authorization"] == f"Bearer {KEY}"
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert KEY not in path.read_text()


def test_an_invalid_reply_is_retried_with_its_errors(
    tmp_path, key_dir, perfect_replies, capsys
) -> None:
    def script(accession, call, body) -> httpx.Response:
        if accession == CAT and call == 1:
            bad = json.loads(json.dumps(perfect_replies[CAT]))
            bad["quarter"]["revenue"] = "$16.6 billion"
            return completion(json.dumps(bad))
        return completion(json.dumps(perfect_replies[accession]))

    model = FakeModel(script)
    assert run(tmp_path, key_dir, model) == 0
    run_dir = tmp_path / "runs" / "test"
    raw = json.loads((run_dir / "raw" / f"{CAT}.json").read_text())
    first, second = (call["request"] for call in raw["calls"])
    assert [json.loads(r.content) for r in model.requests_for(CAT)] == [first, second]
    error = 'quarter.revenue: Input should be a valid decimal (got "$16.6 billion")'
    assert second["messages"][:2] == first["messages"]
    assert second["messages"][2]["role"] == "assistant"
    assert '"$16.6 billion"' in second["messages"][2]["content"]
    assert second["messages"][3] == {
        "role": "user",
        "content": RETRY_MESSAGE.format(errors=error),
    }
    assert raw["outcome"] == OK_AFTER_RETRY
    info = json.loads((run_dir / "run.json").read_text())
    assert info["outcomes"][OK_AFTER_RETRY] == 1
    assert [c["call"] for c in info["calls"] if c["release_accession"] == CAT] == [1, 2]
    assert info["tokens"]["input"] == 6000
    out = capsys.readouterr().out
    assert "accuracy: 26/26 (100.0%)" in out.splitlines()


def test_two_invalid_replies_give_an_all_null_prediction(
    tmp_path, key_dir, perfect_replies, capsys
) -> None:
    def script(accession, call, body) -> httpx.Response:
        if accession == AAPL:
            return completion("The revenue was $95.4 billion.")
        return completion(json.dumps(perfect_replies[accession]))

    assert run(tmp_path, key_dir, FakeModel(script)) == 0
    run_dir = tmp_path / "runs" / "test"
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    null = {"revenue": None, "net_income": None, "eps_diluted": None}
    assert predictions[0] == {
        "release_accession": AAPL,
        "period_end": "2025-03-29",  # from links.csv
        "quarter": null,
        "year_to_date": null,
    }
    reason = "not valid JSON (Expecting value: line 1 column 1 (char 0))"
    info = json.loads((run_dir / "run.json").read_text())
    assert info["fallbacks"] == [
        {
            "ticker": "AAPL",
            "release_accession": AAPL,
            "outcome": INVALID_REPLY,
            "reason": reason,
        }
    ]
    out = capsys.readouterr().out.splitlines()
    i = out.index("invalid replies (all-null predictions, graded as missing): 1")
    assert out[i + 1] == f"  AAPL {AAPL}: {reason}"
    assert "API errors (all-null predictions, graded as missing): 0" in out
    # Graded as missing: AAPL's six answers.
    graded = (run_dir / "graded.csv").read_text().splitlines()
    aapl = [line for line in graded if AAPL in line]
    assert len(aapl) == 6 and all(",missing," in line for line in aapl)


def test_api_errors_are_missing_and_listed_for_a_rerun(
    tmp_path, key_dir, perfect_replies, capsys
) -> None:
    too_long = "This model's maximum context length is 32768 tokens."

    def script(accession, call, body) -> httpx.Response:
        if accession == CAT:
            return httpx.Response(400, json={"error": {"message": too_long}})
        if accession == XOM:
            return httpx.Response(500)
        return completion(json.dumps(perfect_replies[accession]))

    model = FakeModel(script)
    assert run(tmp_path, key_dir, model) == 1
    assert model.calls == {AAPL: 1, CAT: 1, MSFT: 1, WMT: 1, XOM: 6}
    run_dir = tmp_path / "runs" / "test"
    info = json.loads((run_dir / "run.json").read_text())
    assert info["outcomes"][API_ERROR] == 2
    assert [(f["release_accession"], f["reason"]) for f in info["fallbacks"]] == [
        (CAT, f"HTTP 400 ({too_long})"),
        (XOM, "HTTP 500, after 6 attempts"),
    ]
    xom = [c for c in info["calls"] if c["release_accession"] == XOM]
    assert [(c["status"], c["http_attempts"]) for c in xom] == [(500, 6)]
    # Tokens and cost count only the answered calls.
    assert info["tokens"] == {"input": 3000, "output": 300, "calls_without_usage": 0}
    out = capsys.readouterr().out.splitlines()
    i = out.index("API errors (all-null predictions, graded as missing): 2")
    assert out[i + 1 : i + 4] == [
        f"  CAT {CAT}: HTTP 400 ({too_long})",
        f"  XOM {XOM}: HTTP 500, after 6 attempts",
        f"rerun the API errors: uv run python -m secbench.extract --accessions {CAT} {XOM}",
    ]
    assert len(read_jsonl(run_dir / "predictions.jsonl")) == 5
    graded = (run_dir / "graded.csv").read_text().splitlines()
    assert all(",missing," in line for line in graded if CAT in line)


@pytest.mark.parametrize("temperature", ["", "0.7"])
def test_temperature_is_sent_only_when_set(
    tmp_path, key_dir, perfect_replies, capsys, temperature
) -> None:
    model = FakeModel(lambda a, call, body: completion(json.dumps(perfect_replies[a])))
    environ = {**ENVIRON, "MODEL_TEMPERATURE": temperature}
    assert run(tmp_path, key_dir, model, environ=environ) == 0
    run_dir = tmp_path / "runs" / "test"
    sent = {"temperature": 0.7} if temperature else {}
    bodies = [json.loads(r.content) for r in model.requests]
    assert len(bodies) == 5
    for body in bodies:
        assert set(body) == {"model", "messages", *sent}
        assert body.get("temperature") == sent.get("temperature")
    raw = json.loads((run_dir / "raw" / f"{AAPL}.json").read_text())
    assert set(raw["calls"][0]["request"]) == {"model", "messages", *sent}
    info = json.loads((run_dir / "run.json").read_text())
    assert info["request_parameters"] == sent
    shown = "temperature 0.7" if temperature else "no temperature"
    line = f"provider openai; sent with each request: {shown}"
    assert line in capsys.readouterr().out.splitlines()


def test_a_wrong_key_stops_the_run(tmp_path, key_dir, capsys) -> None:
    model = FakeModel(lambda a, call, body: httpx.Response(401))
    assert run(tmp_path, key_dir, model) == 1
    # The first four calls were in flight; the fifth never started.
    assert len(model.requests) == 4
    run_dir = tmp_path / "runs" / "test"
    info = json.loads((run_dir / "run.json").read_text())
    assert info["status"] == "aborted"
    assert info["outcomes"]["aborted"] == 4
    assert info["stop_reason"] == (
        "HTTP 401; check MODEL_BASE_URL, MODEL_NAME and MODEL_API_KEY"
    )
    assert len(list((run_dir / "raw").glob("*.json"))) == 4
    assert not (run_dir / "predictions.jsonl").exists()
    assert not (run_dir / "graded.csv").exists()
    out = capsys.readouterr().out
    assert "stopped, not graded: HTTP 401; check MODEL_BASE_URL" in out


@pytest.mark.parametrize(
    ("setting", "limit"), [(None, 4), ("2", 2), ("3", 3)], ids=["default", "2", "3"]
)
def test_at_most_concurrency_calls_at_once(
    tmp_path, key_dir, perfect_replies, setting, limit
) -> None:
    # The first `limit` calls wait for each other, so all must be in flight at once
    # (a smaller pool times out). They then stay in flight long enough for one more
    # to start if the pool allowed it; it may start only when one finishes.
    barrier = threading.Barrier(limit, timeout=10)
    lock = threading.Lock()
    state = {"now": 0, "max": 0, "started": 0}

    def script(accession, call, body) -> httpx.Response:
        with lock:
            state["now"] += 1
            state["started"] += 1
            state["max"] = max(state["max"], state["now"])
            first = state["started"] <= limit
        if first:
            barrier.wait()
            time.sleep(0.2)
        with lock:
            state["now"] -= 1
        return completion(json.dumps(perfect_replies[accession]))

    model = FakeModel(script)
    environ = ENVIRON if setting is None else {**ENVIRON, "MODEL_CONCURRENCY": setting}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert run(tmp_path, key_dir, model, environ=environ) == 0
    assert state["max"] == limit
    assert sorted(model.calls) == sorted(FIXTURE_RELEASES)
    info = json.loads((tmp_path / "runs" / "test" / "run.json").read_text())
    assert info["max_parallel"] == limit
    assert stdout.getvalue().splitlines()[0].endswith(f"up to {limit} calls at a time")


def refuse(accession: str, call: int, body: dict) -> httpx.Response:
    raise AssertionError("a call was made")


def test_run_with_an_env_file(tmp_path, key_dir, perfect_replies, capsys) -> None:
    env_file = tmp_path / ".env.openai"
    env_file.write_text(
        "MODEL_BASE_URL=https://models.test/v1/\nMODEL_NAME=file-model\n"
        f"MODEL_API_KEY={KEY}\n"
    )
    model = FakeModel(lambda a, call, body: completion(json.dumps(perfect_replies[a])))
    environ = {**ENVIRON, "MODEL_NAME": "environment-model"}  # not used
    assert (
        run(tmp_path, key_dir, model, "--env-file", str(env_file), environ=environ) == 0
    )
    assert {json.loads(r.content)["model"] for r in model.requests} == {"file-model"}
    # The file sets no temperature and no prices.
    assert all("temperature" not in json.loads(r.content) for r in model.requests)
    info = json.loads((tmp_path / "runs" / "test" / "run.json").read_text())
    assert (info["model"], info["env_file"]) == ("file-model", str(env_file))
    assert (info["request_parameters"], info["estimated_cost_usd"]) == ({}, None)
    out = capsys.readouterr().out
    assert out.splitlines()[0].endswith(
        f"no temperature, up to 4 calls at a time; settings from {env_file}"
    )
    assert KEY not in out and KEY not in json.dumps(info)


def test_run_with_an_extra_body(tmp_path, key_dir, perfect_replies, capsys) -> None:
    def script(accession, call, body) -> httpx.Response:
        if accession == CAT and call == 1:
            return completion("The figures are below.")  # retried
        return completion(json.dumps(perfect_replies[accession]))

    model = FakeModel(script)
    extra = '{"reasoning_effort": "medium", "max_tokens": 4096}'
    environ = {**ENVIRON, "MODEL_EXTRA_BODY": extra}
    assert run(tmp_path, key_dir, model, environ=environ) == 0
    bodies = [json.loads(r.content) for r in model.requests]
    assert len(bodies) == 6  # five releases, one retried
    for body in bodies:
        assert list(body)[:2] == ["model", "messages"]
        assert {k: v for k, v in body.items() if k not in ("model", "messages")} == {
            "temperature": 0,
            "reasoning_effort": "medium",
            "max_tokens": 4096,
        }
    info = json.loads((tmp_path / "runs" / "test" / "run.json").read_text())
    assert info["extra_body"] == {"reasoning_effort": "medium", "max_tokens": 4096}
    assert info["request_parameters"] == {
        "temperature": 0.0,
        "reasoning_effort": "medium",
        "max_tokens": 4096,
    }
    assert info["outcomes"][OK_AFTER_RETRY] == 1
    assert (
        "provider openai; sent with each request: temperature 0, reasoning_effort "
        "medium, max_tokens 4096"
    ) in capsys.readouterr().out.splitlines()


def test_run_with_gpu_hour_cost(tmp_path, key_dir, perfect_replies, capsys) -> None:
    model = FakeModel(lambda a, call, body: completion(json.dumps(perfect_replies[a])))
    environ = {**VLLM_ENVIRON, "MODEL_GPU_HOURLY": "3.6"}  # $0.001 a second
    assert run(tmp_path, key_dir, model, environ=environ) == 0
    info = json.loads((tmp_path / "runs" / "test" / "run.json").read_text())
    assert (info["cost_basis"], info["gpu_hourly_usd"]) == ("gpu_hours", 3.6)
    # From the unrounded wall time; run.json rounds it to 0.1 s.
    assert info["estimated_cost_usd"] == pytest.approx(
        info["wall_time_s"] * 0.001, abs=0.00005
    )
    assert info["tokens"] == {"input": 5000, "output": 500, "calls_without_usage": 0}
    assert any(
        line.startswith("estimated cost: $")
        and line.endswith(
            f"($3.6 per GPU-hour for {info['wall_time_s']:.1f} s of wall-clock time)"
        )
        for line in capsys.readouterr().out.splitlines()
    )


def test_default_run_name_for_a_model_id_with_a_slash(
    tmp_path, key_dir, perfect_replies
) -> None:
    model = FakeModel(lambda a, call, body: completion(json.dumps(perfect_replies[a])))
    runs = tmp_path / "runs"
    argv = [
        "--accessions",
        AAPL,
        "--links",
        str(LINKS),
        "--answers",
        str(key_dir / "answers.csv"),
        "--releases",
        str(RELEASES),
        "--runs-dir",
        str(runs),
        "--prompts-dir",
        str(PROMPTS),
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        status = extract.main(
            argv,
            transport=httpx.MockTransport(model.handler),
            environ={**ENVIRON, "MODEL_NAME": "Qwen/Qwen3-8B"},
            edgar_client=make_client(tmp_path / "raw"),
            sleep=lambda s: None,
        )
    assert status == 0
    [folder] = runs.iterdir()  # one folder, not runs/<time>_Qwen/...
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d{6}_Qwen--Qwen3-8B_extract_v1", folder.name)
    info = json.loads((folder / "run.json").read_text())
    assert (info["run_name"], info["model"]) == (folder.name, "Qwen/Qwen3-8B")
    # The model ID is sent unchanged.
    assert json.loads(model.requests[0].content)["model"] == "Qwen/Qwen3-8B"


# extract_v2: revenue and net income with their unit, converted by the runner

# A unit for each release's figures in v2_replies.
V2_UNITS = {
    AAPL: "millions",
    CAT: "thousands",
    MSFT: "billions",
    WMT: "units",
    XOM: "millions",
}


def v2_reply(reply: dict, unit: str) -> dict:
    """A raw-dollar reply with revenue and net income given in unit."""

    def figure(value: float | None) -> dict | None:
        if value is None:
            return None
        return {"value": number(Decimal(str(value)) / UNITS[unit]), "unit": unit}

    return {
        **reply,
        **{
            group: {
                **reply[group],
                "revenue": figure(reply[group]["revenue"]),
                "net_income": figure(reply[group]["net_income"]),
            }
            for group in ("quarter", "year_to_date")
        },
    }


@pytest.fixture(scope="module")
def v2_replies(perfect_replies) -> dict[str, dict]:
    """perfect_replies in extract_v2's format, a unit for each release."""
    return {a: v2_reply(perfect_replies[a], V2_UNITS[a]) for a in FIXTURE_RELEASES}


def graded_verdicts(run_dir: Path) -> dict[tuple[str, str, str], str]:
    with (run_dir / "graded.csv").open(newline="") as f:
        return {
            (row["release_accession"], row["field"], row["period_type"]): row["verdict"]
            for row in csv.DictReader(f)
        }


def test_run_with_units(tmp_path, key_dir, perfect_replies, v2_replies, capsys) -> None:
    model = FakeModel(lambda a, call, body: completion(json.dumps(v2_replies[a])))
    assert run(tmp_path, key_dir, model, "--prompt", "extract_v2") == 0
    out = capsys.readouterr().out.splitlines()
    assert "accuracy: 26/26 (100.0%)" in out
    assert model.calls == dict.fromkeys(FIXTURE_RELEASES, 1)
    run_dir = tmp_path / "runs" / "test"

    # The same predictions as extract_v1's raw dollars, in the same format.
    for prediction, accession in zip(
        read_jsonl(run_dir / "predictions.jsonl"), FIXTURE_RELEASES, strict=True
    ):
        expected = perfect_replies[accession]
        assert prediction.keys() == {"release_accession", *expected}
        for group in ("quarter", "year_to_date"):
            for name, value in expected[group].items():
                got = prediction[group][name]
                assert (got is None) == (value is None)
                if value is not None:
                    assert Decimal(got) == Decimal(str(value)), (accession, group, name)

    info = json.loads((run_dir / "run.json").read_text())
    assert info["prompt_version"] == "extract_v2"
    assert info["prompt_sha256"] == RELEASED_PROMPTS["extract_v2"]
    assert info["reply_format"] == WITH_UNITS
    # AAPL and XOM 4 each, CAT 4, MSFT's full year and fourth quarter 4, WMT's
    # quarter 2.
    assert info["figures_by_unit"] == {
        "units": 2,
        "thousands": 4,
        "millions": 8,
        "billions": 4,
    }
    assert info["format_nulls"] == []
    assert info["outcomes"][OK] == 5

    raw = json.loads((run_dir / "raw" / f"{MSFT}.json").read_text())
    [call] = raw["calls"]
    assert call["request"]["messages"][0] == {
        "role": "system",
        "content": load_prompt("extract_v2", PROMPTS).text,
    }
    # The reply as the model gave it, and the prediction converted from it.
    content = call["attempts"][0]["response"]["choices"][0]["message"]["content"]
    assert json.loads(content)["year_to_date"]["revenue"] == {
        "value": 281.724,
        "unit": "billions",
    }
    assert raw["prediction"]["year_to_date"]["revenue"] == "281724000000.000"
    assert raw["format_nulls"] == []

    assert (
        "figures with a unit: 18 converted (units 2, thousands 4, millions 8, "
        "billions 4)"
    ) in out
    assert "format nulls (figures still bad on the retry, graded as missing): 0" in out


def test_bad_figures_are_retried_then_null(
    tmp_path, key_dir, v2_replies, capsys
) -> None:
    def script(accession, call, body) -> httpx.Response:
        reply = json.loads(json.dumps(v2_replies[accession]))
        if accession == AAPL:  # an unknown unit, twice
            reply["quarter"]["revenue"]["unit"] = "Millions"
        elif accession == CAT and call == 1:  # fixed on the retry
            reply["year_to_date"]["revenue"]["unit"] = "thousand"
        elif accession == MSFT:  # a bad figure, then no JSON
            if call == 2:
                return completion("I apologize for the error.")
            reply["quarter"]["revenue"] = "76.4 billion"
        elif accession == WMT:  # raw dollars, as extract_v1 asks, twice
            reply["quarter"]["net_income"] = 4487000000
        elif accession == XOM:  # negative: graded as a sign error
            reply["quarter"]["net_income"]["value"] = -14525
        return completion(json.dumps(reply))

    model = FakeModel(script)
    assert run(tmp_path, key_dir, model, "--prompt", "extract_v2") == 0
    assert model.calls == {AAPL: 2, CAT: 2, MSFT: 2, WMT: 2, XOM: 1}
    run_dir = tmp_path / "runs" / "test"

    # The retry asks again with the problem, as for any invalid reply.
    cat = [json.loads(r.content) for r in model.requests_for(CAT)]
    cat_error = f'year_to_date.revenue.unit: {SHOULD_BE_UNIT} (got "thousand")'
    assert cat[1]["messages"][:2] == cat[0]["messages"]
    assert cat[1]["messages"][2]["role"] == "assistant"
    assert '"thousand"' in cat[1]["messages"][2]["content"]
    assert cat[1]["messages"][3] == {
        "role": "user",
        "content": RETRY_MESSAGE.format(errors=cat_error),
    }

    info = json.loads((run_dir / "run.json").read_text())
    assert info["outcomes"] == {
        OK: 1,
        OK_AFTER_RETRY: 3,
        INVALID_REPLY: 1,
        API_ERROR: 0,
        "aborted": 0,
    }
    aapl_message = f'quarter.revenue.unit: {SHOULD_BE_UNIT} (got "Millions")'
    wmt_message = f"quarter.net_income: {SHOULD_BE_FIGURE} (got 4487000000)"
    assert info["format_nulls"] == [
        {
            "ticker": "AAPL",
            "release_accession": AAPL,
            "figure": "quarter.revenue",
            "kind": UNKNOWN_UNIT,
            "message": aapl_message,
            "replied": '{"value": 95359, "unit": "Millions"}',
        },
        {
            "ticker": "WMT",
            "release_accession": WMT,
            "figure": "quarter.net_income",
            "kind": MALFORMED,
            "message": wmt_message,
            "replied": "4487000000",
        },
    ]
    # MSFT's retry wasn't JSON: all null, as in extract_v1.
    assert info["fallbacks"] == [
        {
            "ticker": "MSFT",
            "release_accession": MSFT,
            "outcome": INVALID_REPLY,
            "reason": "not valid JSON (Expecting value: line 1 column 1 (char 0))",
        }
    ]
    # AAPL 3 millions, CAT 4 thousands (on the retry), WMT 1 units, XOM 4 millions.
    assert info["figures_by_unit"] == {
        "units": 1,
        "thousands": 4,
        "millions": 7,
        "billions": 0,
    }

    raw = json.loads((run_dir / "raw" / f"{AAPL}.json").read_text())
    assert raw["outcome"] == OK_AFTER_RETRY
    assert raw["format_nulls"] == [
        {
            "figure": "quarter.revenue",
            "kind": UNKNOWN_UNIT,
            "message": aapl_message,
            "replied": '{"value": 95359, "unit": "Millions"}',
        }
    ]
    assert raw["prediction"]["quarter"]["revenue"] is None
    assert raw["prediction"]["quarter"]["net_income"] == "24780000000"
    # The replies as the model gave them.
    replies = [
        json.loads(c["attempts"][0]["response"]["choices"][0]["message"]["content"])
        for c in raw["calls"]
    ]
    assert [r["quarter"]["revenue"] for r in replies] == [
        {"value": 95359, "unit": "Millions"}
    ] * 2

    verdicts = graded_verdicts(run_dir)
    assert verdicts[(AAPL, "revenue", QUARTER)] == "missing"
    assert verdicts[(CAT, "revenue", YEAR_TO_DATE)] == "correct"
    assert verdicts[(WMT, "net_income", QUARTER)] == "missing"
    assert verdicts[(XOM, "net_income", QUARTER)] == "sign"
    assert {v for (a, _, _), v in verdicts.items() if a == MSFT} == {"missing"}
    out = capsys.readouterr().out.splitlines()
    # 26 less AAPL's and WMT's format nulls, MSFT's 5 and XOM's sign.
    assert "accuracy: 18/26 (69.2%)" in out
    i = out.index(
        "format nulls (figures still bad on the retry, graded as missing): 2 (1 "
        "unknown unit, 1 malformed)"
    )
    assert (
        out[i - 1]
        == "figures with a unit: 12 converted (units 1, thousands 4, millions 7)"
    )
    assert out[i + 1 : i + 3] == [
        f"  AAPL {AAPL} {aapl_message}",
        f"  WMT {WMT} {wmt_message}",
    ]


@pytest.mark.parametrize("version", ["extract_v1", "extract_v2"])
def test_dry_run_makes_no_calls(tmp_path, key_dir, capsys, version) -> None:
    model = FakeModel(refuse)
    assert (
        run(tmp_path, key_dir, model, "--dry-run", "--prompt", version, environ={}) == 0
    )
    assert model.requests == []
    assert not (tmp_path / "runs").exists()
    out = capsys.readouterr().out
    records = release_records(RELEASES)
    text = build_input(records[AAPL]).text
    lines = out.splitlines()
    assert lines[0] == (
        f"dry run: AAPL {AAPL} (1 of 5 selected releases); input EX-99.1, "
        f"{len(text):,} characters"
    )
    # Both providers' requests: the same system prompt and user message.
    assert lines[1:3] == [
        (
            "openai (the default): POST <MODEL_BASE_URL>/chat/completions, model "
            "<MODEL_NAME>, temperature <MODEL_TEMPERATURE> (not sent if blank), "
            "<MODEL_EXTRA_BODY> merged into the body (if set)"
        ),
        (
            "anthropic: messages.create (anthropic SDK), model <MODEL_NAME>, "
            "max_tokens 16000, effort <MODEL_EFFORT> (not sent if blank), no "
            "temperature"
        ),
    ]
    sha256 = RELEASED_PROMPTS[version][:12]
    assert re.fullmatch(
        rf"=== system: .*{version}\.txt \(sha256 {sha256}\) ===", lines[3]
    )
    prompt = load_prompt(version, PROMPTS).text
    assert (
        out
        == "\n".join(
            [
                *lines[:4],
                prompt.rstrip("\n"),
                "=== user: the release input ===",
                text.rstrip("\n"),
                "=== end of request; no call made ===",
            ]
        )
        + "\n"
    )


def test_missing_settings_stop_before_any_call(tmp_path, key_dir, capsys) -> None:
    model = FakeModel(refuse)
    assert run(tmp_path, key_dir, model, environ={"MODEL_NAME": "x"}) == 1
    assert model.requests == []
    assert capsys.readouterr().err == (
        "error: MODEL_BASE_URL not set; add to .env (see .env.example)\n"
    )


def test_an_existing_run_is_never_overwritten(tmp_path, key_dir, capsys) -> None:
    (tmp_path / "runs" / "test").mkdir(parents=True)
    model = FakeModel(refuse)
    assert run(tmp_path, key_dir, model) == 1
    assert model.requests == []
    assert "already exists; choose another --run-name" in capsys.readouterr().err


# Anthropic: a fake client, and one test through the real SDK over a mock transport

ANTHROPIC_ENVIRON = {
    "MODEL_PROVIDER": "anthropic",
    "MODEL_NAME": "claude-opus-5-5",
    "MODEL_API_KEY": KEY,
    "MODEL_PRICE_INPUT": "4",
    "MODEL_PRICE_OUTPUT": "20",
    # Left from an openai setup: not used, not checked.
    "MODEL_BASE_URL": "https://models.test/v1/",
    "MODEL_TEMPERATURE": "0",
}
ANTHROPIC_SETTINGS = Settings(
    "",
    "claude-opus-5-5",
    KEY,
    price_input=4.0,
    price_output=20.0,
    provider="anthropic",
)
MESSAGES_URL = "https://api.anthropic.com/v1/messages"
THINKING = {
    "type": "thinking",
    "thinking": "Revenue is in the first table.",
    "signature": "sig-1",
}
REFUSAL = {"type": "refusal", "category": "cyber", "explanation": "Declined."}


def message(*texts: str, stop_reason="end_turn", stop_details=None) -> Message:
    """A reply: a thinking block (summarized), then a text block per
    text."""
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [THINKING, *({"type": "text", "text": t} for t in texts)],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": {"input_tokens": 1000, "output_tokens": 100},
        }
    )


def api_error(cls, status: int, text: str = "", headers=None) -> anthropic.APIError:
    """What the SDK raises for an HTTP error."""
    request = httpx2.Request("POST", MESSAGES_URL)
    body = {"type": "error", "error": {"type": "error", "message": text}}
    response = httpx2.Response(status, headers=headers, json=body, request=request)
    return cls(f"Error code: {status}", response=response, body=body)


class FakeAnthropic:
    """Stands in for anthropic.Anthropic: messages.create returns or raises what
    script(kwargs) gives, and keeps every call's kwargs."""

    def __init__(self, script: Callable[[dict], object]) -> None:
        self.script = script
        self.messages = self
        self.calls: list[dict] = []
        self.closed = False
        self._lock = threading.Lock()

    def create(self, **kwargs) -> Message:
        with self._lock:
            self.calls.append(kwargs)
        result = self.script(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.closed = True


def scripted_anthropic(results: list):
    """An AnthropicChat whose fake client gives results in order, and the delays it
    slept."""
    queue = list(results)
    fake = FakeAnthropic(lambda kwargs: queue.pop(0))
    slept: list[float] = []
    chat = AnthropicChat(
        ANTHROPIC_SETTINGS, client=fake, sleep=slept.append, wall_clock=lambda: 0.0
    )
    return chat, fake, slept


def first_body(chat) -> dict:
    return chat.first_body(load_prompt("extract_v1", PROMPTS), "the release")


def test_load_settings_for_anthropic() -> None:
    settings = load_settings(ANTHROPIC_ENVIRON)
    assert settings == ANTHROPIC_SETTINGS
    # max_tokens, and never a temperature.
    assert settings.parameters == {"max_tokens": 16000}
    upper = {**ANTHROPIC_ENVIRON, "MODEL_PROVIDER": " Anthropic "}
    assert load_settings(upper).provider == "anthropic"
    # MODEL_BASE_URL and MODEL_TEMPERATURE aren't read.
    unused = {**ANTHROPIC_ENVIRON, "MODEL_BASE_URL": "", "MODEL_TEMPERATURE": "warm"}
    assert load_settings(unused) == ANTHROPIC_SETTINGS
    with pytest.raises(SettingsError, match="^MODEL_API_KEY not set"):
        load_settings({**ANTHROPIC_ENVIRON, "MODEL_API_KEY": ""})


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_anthropic_effort(effort) -> None:
    settings = load_settings(
        {**ANTHROPIC_ENVIRON, "MODEL_EFFORT": f" {effort.upper()} "}
    )
    assert settings.effort == effort
    assert settings.parameters == {
        "max_tokens": 16000,
        "output_config": {"effort": effort},
    }


def test_anthropic_effort_is_left_out_when_blank() -> None:
    for blank in ("", " "):
        settings = load_settings({**ANTHROPIC_ENVIRON, "MODEL_EFFORT": blank})
        assert settings.effort is None
        assert settings.parameters == {"max_tokens": 16000}
    assert load_settings(ANTHROPIC_ENVIRON).effort is None  # unset


def test_a_bad_effort_names_the_setting_not_its_value() -> None:
    with pytest.raises(SettingsError) as raised:
        load_settings({**ANTHROPIC_ENVIRON, "MODEL_EFFORT": "extreme"})
    assert str(raised.value) == ("MODEL_EFFORT must be low, medium, high, xhigh or max")


def test_openai_ignores_model_effort() -> None:
    settings = load_settings({**ENVIRON, "MODEL_EFFORT": "extreme"})
    assert settings.effort is None
    assert settings.parameters == {"temperature": 0.0}


def test_anthropic_request_and_text_blocks() -> None:
    chat, fake, slept = scripted_anthropic([message("{", '"a": 1}')])
    body = first_body(chat)
    assert body == {
        "model": "claude-opus-5-5",
        "max_tokens": 16000,
        "system": load_prompt("extract_v1", PROMPTS).text,
        "messages": [{"role": "user", "content": "the release"}],
    }
    exchange = chat.complete(body)
    assert fake.calls == [body]
    assert exchange.error == ""
    # The text blocks, joined; not the thinking block.
    assert exchange.content == '{"a": 1}'
    assert exchange.finish_reason == "end_turn"
    assert exchange.tokens == (1000, 100)
    [attempt] = exchange.attempts
    assert attempt["status"] == 200
    # The whole response is kept, thinking block included.
    assert attempt["response"]["content"] == [
        THINKING,
        {"type": "text", "text": "{"},
        {"type": "text", "text": '"a": 1}'},
    ]
    assert slept == []


def test_anthropic_retry_sends_the_reply_back_unchanged() -> None:
    chat, _, _ = scripted_anthropic([message("Here are the figures.")])
    body = first_body(chat)
    exchange = chat.complete(body)
    retry = chat.retry_body(body, exchange.reply, "not valid JSON")
    assert retry == {
        **body,
        "messages": [
            *body["messages"],
            {
                "role": "assistant",
                "content": [
                    THINKING,
                    {"type": "text", "text": "Here are the figures."},
                ],
            },
            {"role": "user", "content": RETRY_MESSAGE.format(errors="not valid JSON")},
        ],
    }
    assert len(body["messages"]) == 1  # the first request is unchanged


@pytest.mark.parametrize(
    ("details", "reason"),
    [(REFUSAL, "refusal (cyber): Declined."), (None, "refusal")],
)
def test_an_anthropic_refusal_is_an_api_error(details, reason) -> None:
    refused = message(stop_reason="refusal", stop_details=details)
    chat, fake, slept = scripted_anthropic([refused])
    exchange = chat.complete(first_body(chat))
    assert exchange.error == reason
    assert not exchange.abort
    assert exchange.finish_reason == "refusal"
    assert exchange.tokens == (1000, 100)  # billed
    # Not retried, and no other model is tried.
    assert len(fake.calls) == 1 and slept == []
    assert {c["model"] for c in fake.calls} == {"claude-opus-5-5"}


def test_anthropic_errors_go_through_the_shared_retries() -> None:
    chat, fake, slept = scripted_anthropic(
        [
            api_error(anthropic.RateLimitError, 429, headers={"retry-after": "3"}),
            api_error(anthropic.InternalServerError, 500, "Internal server error"),
            api_error(anthropic.OverloadedError, 529, "Overloaded"),
            anthropic.APITimeoutError(request=httpx2.Request("POST", MESSAGES_URL)),
            message("{}"),
        ]
    )
    exchange = chat.complete(first_body(chat))
    assert exchange.content == "{}"
    assert [a["status"] for a in exchange.attempts] == [429, 500, 529, None, 200]
    assert exchange.attempts[1]["response"]["error"]["message"] == (
        "Internal server error"
    )
    assert exchange.attempts[3]["error"].startswith(
        "APITimeoutError: Request timed out"
    )
    assert slept[0] == 3.0  # Retry-After
    for delay, base in zip(slept[1:], (2, 4, 8), strict=True):
        assert base <= delay <= 1.25 * base
    assert len(fake.calls) == 5


def test_anthropic_gives_up_after_5_retries() -> None:
    overloaded = api_error(anthropic.OverloadedError, 529, "Overloaded")
    chat, fake, slept = scripted_anthropic([overloaded] * 6)
    exchange = chat.complete(first_body(chat))
    assert exchange.error == "HTTP 529 (Overloaded), after 6 attempts"
    assert len(fake.calls) == 6 and len(slept) == 5


def test_anthropic_does_not_retry_other_4xx() -> None:
    too_long = "prompt is too long: 1200000 tokens > 1000000 maximum"
    chat, fake, slept = scripted_anthropic(
        [api_error(anthropic.BadRequestError, 400, too_long)]
    )
    exchange = chat.complete(first_body(chat))
    assert exchange.error == f"HTTP 400 ({too_long})"
    assert not exchange.abort
    assert len(fake.calls) == 1 and slept == []


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (anthropic.AuthenticationError, 401),
        (anthropic.PermissionDeniedError, 403),
        (anthropic.NotFoundError, 404),
    ],
)
def test_anthropic_aborts_on_a_wrong_key_or_model(cls, status) -> None:
    chat, fake, slept = scripted_anthropic(
        [api_error(cls, status, "invalid x-api-key")]
    )
    exchange = chat.complete(first_body(chat))
    assert exchange.abort
    assert exchange.error == f"HTTP {status}; check MODEL_NAME and MODEL_API_KEY"
    assert [(a["status"], a["response"]) for a in exchange.attempts] == [(status, None)]
    assert len(fake.calls) == 1 and slept == []


def test_anthropic_client_retries_nothing_itself() -> None:
    chat = AnthropicChat(ANTHROPIC_SETTINGS)
    assert chat._client.max_retries == 0
    assert chat._client.api_key == KEY


def test_anthropic_through_the_real_sdk() -> None:
    # The SDK, over a mock transport: what it sends is exactly what raw records,
    # and a 529 reaches the shared retries (the SDK's own are off).
    sent: list[httpx2.Request] = []
    replies = [
        httpx2.Response(
            529, json={"type": "error", "error": {"message": "Overloaded"}}
        ),
        httpx2.Response(200, json=message("Not JSON").to_dict(mode="json")),
        httpx2.Response(200, json=message(reply_text()).to_dict(mode="json")),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return replies.pop(0)

    client = anthropic.Anthropic(
        api_key=KEY,
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(
            transport=httpx2.MockTransport(handler)
        ),
    )
    slept: list[float] = []
    settings = dataclasses.replace(ANTHROPIC_SETTINGS, effort="high")
    chat = AnthropicChat(settings, client=client, sleep=slept.append)
    body = first_body(chat)
    assert body["output_config"] == {"effort": "high"}
    first = chat.complete(body)
    assert [a["status"] for a in first.attempts] == [529, 200]
    assert len(slept) == 1
    retry = chat.retry_body(body, first.reply, "not valid JSON")
    second = chat.complete(retry)
    assert second.content == reply_text()
    assert [str(r.url) for r in sent] == [MESSAGES_URL] * 3
    assert {r.headers["x-api-key"] for r in sent} == {KEY}
    assert [json.loads(r.content) for r in sent] == [body, body, retry]
    assert second.attempts[0]["response"] == message(reply_text()).to_dict(mode="json")


def by_release(script: Callable[[str, int, dict], object]) -> Callable[[dict], object]:
    """A FakeAnthropic script from script(accession, call, kwargs); call counts the
    release's requests from 1."""
    records = release_records(RELEASES)
    accessions = {build_input(records[a]).text: a for a in FIXTURE_RELEASES}
    calls: Counter[str] = Counter()
    lock = threading.Lock()

    def wrapped(kwargs: dict) -> object:
        accession = accessions[kwargs["messages"][0]["content"]]
        with lock:
            calls[accession] += 1
            call = calls[accession]
        return script(accession, call, kwargs)

    return wrapped


@pytest.mark.parametrize("effort", ["", "high"])
def test_run_with_anthropic(tmp_path, key_dir, perfect_replies, capsys, effort) -> None:
    def script(accession, call, kwargs) -> Message:
        if accession == XOM:
            return message(stop_reason="refusal", stop_details=REFUSAL)
        if accession == CAT and call == 1:
            return message("The figures are in the table below.")
        return message(json.dumps(perfect_replies[accession]))

    fake = FakeAnthropic(by_release(script))
    environ = {**ANTHROPIC_ENVIRON, "MODEL_EFFORT": effort}
    sent = {"output_config": {"effort": "high"}} if effort else {}
    # The refusal is an API error, so the run exits 1.
    assert run(tmp_path, key_dir, None, environ=environ, anthropic_client=fake) == 1
    assert fake.closed
    run_dir = tmp_path / "runs" / "test"
    info = json.loads((run_dir / "run.json").read_text())
    assert info["provider"] == "anthropic"
    assert info["model"] == "claude-opus-5-5"
    # What was sent: output_config only with an effort.
    assert info["request_parameters"] == {"max_tokens": 16000, **sent}
    assert info["outcomes"] == {
        OK: 3,
        OK_AFTER_RETRY: 1,
        INVALID_REPLY: 0,
        API_ERROR: 1,
        "aborted": 0,
    }
    assert info["fallbacks"] == [
        {
            "ticker": "XOM",
            "release_accession": XOM,
            "outcome": API_ERROR,
            "reason": "refusal (cyber): Declined.",
        }
    ]
    # Six calls, all billed (the refusal too): 6,000 input tokens at $4 and 600
    # output tokens at $20 per million.
    assert info["tokens"] == {"input": 6000, "output": 600, "calls_without_usage": 0}
    assert info["estimated_cost_usd"] == 0.036
    xom = [c for c in info["calls"] if c["release_accession"] == XOM]
    assert [(c["status"], c["finish_reason"], c["error"]) for c in xom] == [
        (200, "refusal", "refusal (cyber): Declined.")
    ]

    # One model throughout; the prompt is the system parameter; no temperature; an
    # effort only when set.
    prompt = load_prompt("extract_v1", PROMPTS).text
    assert len(fake.calls) == 6
    for kwargs in fake.calls:
        assert set(kwargs) == {"model", "max_tokens", "system", "messages", *sent}
        assert (kwargs["model"], kwargs["max_tokens"]) == ("claude-opus-5-5", 16000)
        assert kwargs["system"] == prompt
        assert kwargs.get("output_config") == sent.get("output_config")
    raw = json.loads((run_dir / "raw" / f"{AAPL}.json").read_text())
    assert raw["calls"][0]["request"].get("output_config") == sent.get("output_config")

    # CAT's retry sent its first reply back unchanged, thinking block included.
    raw = json.loads((run_dir / "raw" / f"{CAT}.json").read_text())
    first, second = raw["calls"]
    assert second["request"]["messages"][1] == {
        "role": "assistant",
        "content": first["attempts"][0]["response"]["content"],
    }
    assert second["request"]["messages"][1]["content"][0] == THINKING
    assert raw["outcome"] == OK_AFTER_RETRY

    out = capsys.readouterr().out.splitlines()
    shown = f"max_tokens 16000, {'effort high' if effort else 'default effort'}, no "
    shown += "temperature"
    assert any(
        line.endswith(
            f"runs/test: 5 releases, provider anthropic, prompt extract_v1, {shown}, "
            "up to 4 calls at a time"
        )
        for line in out
    )
    assert f"provider anthropic; sent with each request: {shown}" in out
    # XOM's six answers are missing.
    assert "accuracy: 20/26 (76.9%)" in out
    i = out.index("API errors (all-null predictions, graded as missing): 1")
    assert out[i + 1 : i + 3] == [
        f"  XOM {XOM}: refusal (cyber): Declined.",
        f"rerun the API errors: uv run python -m secbench.extract --accessions {XOM}",
    ]
