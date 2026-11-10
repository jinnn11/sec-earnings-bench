"""The results explorer, a Hugging Face Space (secbench.hfexport's part C): build_space
writes the Space's folder (dist/hf/space/) from the export's Context.

- app.py and requirements.txt: space/'s, copied byte for byte. The app imports gradio
  and the standard library only, never secbench, so the Space runs without this
  repository.
- README.md: the Space's metadata (sdk_version is the gradio version that
  space/requirements.txt pins) and a short description with links.
- data/: everything the app reads, precomputed here so that the app only looks things
  up: meta.json (links; the splits, fields, periods and verdicts described; the About
  text), leaderboard.json (results/comparison.md's main table from
  hfcommon.validation_table, and both test sets from results/test_results.json
  formatted with secbench.testreport's own functions, so every cell reads as in those
  files; a GPU run's "runs on" names its GPU, as README's tables do), runs.json (each
  split's runs, and those Explore shows first; repeated runs of one model and prompt
  are told apart by their run's name), releases.json (each scored release without its
  text; left-out ones flagged, with why in words that need no other document),
  answers.json (each answer with where the grader finds it in the text),
  predictions/<split>.json (graded.csv's figures and verdicts by run and release:
  figures only, never a model's reply text), errors.json (verdict counts by run, and
  one example of each verdict), serving.json (only when results/serving_results.json
  exists: its numbers, without server commands or paths), texts/<split>/<accession>.md
  (each release's model input, left-out ones included) and charts/ (results' PNGs).

Every number shown comes from a results file, through hfcommon's Context,
training/qwen3-8b-2k/validation.json, data/companies.csv or
results/serving_results.json, and every sentence holding one is written here, not in
app.py (which only counts the releases its filters match). The one fact no results
file holds is stage 1's GPU (STAGE1_GPU), named as README's results table names it.
JSON is compact (hfcommon.write_json without an indent) and every order is fixed, so
two builds give the same bytes.

Library only: secbench.hfexport calls build_space.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from secbench import (
    answers,
    chat,
    grade,
    hfcommon,
    link,
    servebench,
    servereport,
    serving,
    splits,
    testreport,
)
from secbench.hfcommon import (
    COMPANY_TEST,
    DATASET,
    MODEL,
    REPO_URL,
    SELF_HOSTED,
    SPACE,
    TIME_TEST,
    TRAINING_RUN,
    VALIDATION,
    Context,
    ExportError,
    Written,
)
from secbench.stage2 import COMPANY_TEST_RELEASES, REPO

SPACE_SOURCE = REPO / "space"  # app.py and requirements.txt, tracked
APP = "app.py"
REQUIREMENTS = "requirements.txt"
README = "README.md"
DATA = "data"
# data/'s files; app.py names them the same (a test checks).
META = "meta.json"
LEADERBOARD = "leaderboard.json"
RUNS = "runs.json"
RELEASES = "releases.json"
ANSWERS = "answers.json"
PREDICTIONS = "predictions"  # predictions/<split>.json
ERRORS = "errors.json"
SERVING = "serving.json"
TEXTS = "texts"  # texts/<split>/<accession>.md
CHARTS = "charts"
PREDICTION_COLUMNS = (
    "field",
    "period_type",
    "predicted",
    "verdict",
    "factor",
    "evidence",
)
EXAMPLE_COLUMNS = (
    "run",
    "verdict",
    "ticker",
    "release_accession",
    "field",
    "period_type",
    "answer",
    "predicted",
    "factor",
    "evidence",
)

# The Space's card metadata (README.md's front matter), besides what is computed.
TITLE = "SEC Earnings Explorer"
EMOJI = "📊"
COLOR_FROM, COLOR_TO = "blue", "indigo"
SHORT_DESCRIPTION = "LLMs extracting SEC earnings figures, graded by XBRL"
SHORT_DESCRIPTION_MAX = 60  # the Hub's limit
PYTHON_VERSION = "3.11"
GRADIO_PIN = re.compile(r"^gradio==(\S+)$")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

SPLIT_TITLES = {
    VALIDATION: "Validation",
    COMPANY_TEST: "Company test",
    TIME_TEST: "Time test",
}
FIELD_TITLES = {
    "revenue": "Revenue",
    "net_income": "Net income",
    "eps_diluted": "Diluted EPS",
}
PERIOD_TITLES = {
    "quarter": "Quarter",
    "year_to_date": "Year to date",
    "full_year": "Full year",
}
# The printed forms grade.printed_forms searches (grade.USD_FORMS: unit, decimals).
USD_FORM_WORDS = {
    (10**6, 0): "whole millions",
    (10**6, 1): "millions to one decimal",
    (10**3, 0): "whole thousands",
}
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
CHART_CAPTIONS = {
    "validation": (
        "Accuracy against cost per 1,000 releases on the validation releases (stage "
        "1's runs), log scale; arrows show each open model's move from prompt v1 to v2."
    ),
    "test": (
        "Accuracy against cost per 1,000 releases on both test sets, with 95% "
        "intervals; the company test's fine-tuned point is joined to its cost without "
        "the instance freeze."
    ),
    "serving": "Throughput against latency for each serving variant and load level.",
}
# Stage 1's open models all ran on this GPU, as README's results table ("1× A100
# 80GB") and the validation chart (scripts/plot_results.py) name it; no run file
# records a GPU. A validation GPU run that isn't stage 2's (TRAINING_RUN's, whose GPU
# results/test_results.json names) is stage 1's, as plot_results.py has it: a GPU run
# made elsewhere would need its GPU named here.
STAGE1_GPU = "A100 80GB"
# The serving variants (secbench.serving) as the Space's README names them.
VARIANT_WORDS = {
    serving.ADAPTER: "an adapter",
    serving.MERGED: "merged",
    serving.FP8: "FP8",
    serving.INT4_AWQ: "4-bit AWQ",
}


def build_space(ctx: Context, out: Path) -> Written:
    """Writes the Space's folder at out: app.py, requirements.txt, README.md and
    data/. Skips the serving tab's data (and says so) without
    results/serving_results.json."""
    written = Written()
    gradio_version = gradio_pin(SPACE_SOURCE / REQUIREMENTS)
    serving_results = None
    if ctx.sources.serving_results.is_file():
        serving_results = hfcommon.read_json(ctx.sources.serving_results)
    else:
        written.skipped.append(
            "Space's serving tab: results/serving_results.json not found"
        )
    for name in (APP, REQUIREMENTS):
        hfcommon.copy_file(SPACE_SOURCE / name, out / name)
        written.files.append(out / name)
    hfcommon.write_text(
        out / README, space_readme(ctx, gradio_version, serving_results)
    )
    written.files.append(out / README)

    data = out / DATA
    charts = chart_files(ctx, serving_results is not None, written)
    files: dict[str, Any] = {
        META: meta(ctx, serving_results is not None),
        LEADERBOARD: leaderboard(ctx, charts),
        RUNS: runs_data(ctx),
        RELEASES: releases_data(ctx),
        ANSWERS: answers_data(ctx),
        ERRORS: errors_data(ctx),
    }
    for split in hfcommon.SPLIT_NAMES:
        files[f"{PREDICTIONS}/{split}.json"] = predictions_data(ctx, split)
    if serving_results is not None:
        files[SERVING] = serving_data(serving_results, charts.get("serving"))
    for name, value in files.items():
        check_no_email(f"{DATA}/{name}", value)
        hfcommon.write_json(data / name, value, indent=None)
        written.files.append(data / name)
    for split, rows in ctx.releases.items():
        for row in rows:
            path = data / text_path(split, row.release_accession)
            hfcommon.write_text(path, row.text)
            written.files.append(path)
    for key, name in charts.items():
        source = {
            "validation": ctx.sources.validation_chart,
            "test": ctx.sources.test_chart,
            "serving": ctx.sources.serving_chart,
        }[key]
        hfcommon.copy_file(source, data / name)
        written.files.append(data / name)
    return written


def gradio_pin(path: Path) -> str:
    """The gradio version requirements.txt pins (gradio==X.Y.Z)."""
    if not path.is_file():
        raise ExportError(f"{path} not found")
    pins = [
        m[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if (m := GRADIO_PIN.match(line.strip()))
    ]
    if len(pins) != 1:
        raise ExportError(f"{path} must pin gradio once (gradio==X.Y.Z)")
    return pins[0]


def text_path(split: str, accession: str) -> str:
    """A release text's path under data/."""
    return f"{TEXTS}/{split}/{accession}.md"


def chart_files(ctx: Context, serving: bool, written: Written) -> dict[str, str]:
    """The charts to copy, by key, as data/charts/<name>: both result charts, and the
    serving chart when there are serving results and it exists."""
    charts = {
        "validation": f"{CHARTS}/{ctx.sources.validation_chart.name}",
        "test": f"{CHARTS}/{ctx.sources.test_chart.name}",
    }
    if serving:
        if ctx.sources.serving_chart.is_file():
            charts["serving"] = f"{CHARTS}/{ctx.sources.serving_chart.name}"
        else:
            written.skipped.append(
                "Space's serving chart: results/serving_throughput_latency.png not found"
            )
    return charts


# README.md


def space_readme(
    ctx: Context,
    gradio_version: str,
    serving_results: Mapping[str, Any] | None = None,
) -> str:
    if len(SHORT_DESCRIPTION) > SHORT_DESCRIPTION_MAX:
        raise ExportError(
            f"the Space's short_description is over {SHORT_DESCRIPTION_MAX} characters"
        )
    front = [
        "---",
        f"title: {TITLE}",
        f"emoji: {EMOJI}",
        f"colorFrom: {COLOR_FROM}",
        f"colorTo: {COLOR_TO}",
        "sdk: gradio",
        f"sdk_version: {gradio_version}",
        f'python_version: "{PYTHON_VERSION}"',
        f"app_file: {APP}",
        "pinned: false",
        "license: mit",
        f"short_description: {SHORT_DESCRIPTION}",
        "datasets:",
        f"  - {ctx.repo_id(DATASET)}",
        "models:",
        f"  - {ctx.repo_id(MODEL)}",
        "---",
    ]
    results = ctx.test_results
    api = len(results["models"]) - len({results["under_test"], results["base"]})
    body = [
        "",
        "# SEC earnings extraction: results explorer",
        "",
        (
            f"The results of [sec-earnings-bench]({REPO_URL}), a benchmark of how "
            "accurately LLMs extract GAAP revenue, net income and diluted EPS from US "
            "earnings press releases, graded automatically against XBRL in the "
            "company's later 10-Q or 10-K."
        ),
        "",
        (
            "- **Leaderboard:** every model on the validation releases, and the "
            f"fine-tuned Qwen3-8B against its base model and {count_word(api)} API "
            f"models on the {count_word(len(hfcommon.TEST_SPLITS))} held-out test "
            "sets, with 95% intervals and cost."
        ),
        (
            "- **Explore:** any release of the "
            f"{count_word(len(hfcommon.SPLIT_NAMES))} scored splits: the answer key, "
            "each model's figures and verdicts, and the release text with each answer "
            "highlighted where the grader found it."
        ),
        "- **Errors:** each model's verdicts, with an example of each.",
        *serving_bullet(serving_results),
        (
            "- **About:** how releases, answers, the grader and the splits are built, "
            "the limitations, and where the fine-tuned model fails."
        ),
        "",
        (
            f"Links: [code on GitHub]({REPO_URL}) (MIT) · "
            f"[dataset]({ctx.hub_url(DATASET)}) · "
            f"[fine-tuned model (LoRA adapter)]({ctx.hub_url(MODEL)})"
        ),
        "",
        (
            "The release texts are the companies' press releases as filed with the SEC "
            "on EDGAR. For the API models (OpenAI and Anthropic), only the predicted "
            "figures and the grader's verdicts are shown, never a model's reply. The "
            "files in `data/` are built by `secbench.hfspace` in the GitHub repository "
            "from its results."
        ),
    ]
    return "\n".join(front + body) + "\n"


def serving_bullet(results: Mapping[str, Any] | None) -> list[str]:
    """The README's line on the Serving tab, naming the variants that completed
    (servebench's status); none without serving results."""
    if results is None:
        return []
    done = [
        VARIANT_WORDS.get(name, name)
        for name, v in results["variants"].items()
        if v.get("status") == servebench.COMPLETE
    ]
    what = f", as {joined(done, 'and')}" if done else ": no variant completed"
    return [
        (
            "- **Serving:** the fine-tuned model's throughput, latency, cost and "
            f"accuracy on vLLM{what}."
        )
    ]


# meta.json


def count_word(n: int) -> str:
    """A small count in words, as prose writes it ("three API models")."""
    return NUMBER_WORDS.get(n, f"{n:,}")


def joined(items: Sequence[str], last: str = "or") -> str:
    """'a', 'a or b', 'a, b or c'."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {last} {items[-1]}"


def verdict_meanings() -> dict[str, str]:
    """A line per verdict of grade.VERDICTS, in that order; the powers of ten from the
    grader's own constants."""
    scales = sorted(int(s) for s in grade.SCALES if s > 1)
    zeros = sorted(int(f) for f in grade.FACTORS if f > 1 and f not in grade.SCALES)
    meanings = {
        grade.CORRECT: (
            "equals the answer once rounded to the precision the release prints; EPS "
            "exact to the cent"
        ),
        grade.MISSING: "no figure (null) where the release prints the answer",
        grade.SIGN: "the answer with the wrong sign",
        grade.SCALE: (
            f"off by a factor of {joined([f'{n:,}' for n in scales])}: thousands or "
            "millions applied wrongly"
        ),
        grade.ZEROS: (
            f"off by another power of ten ({joined([f'{n:,}' for n in zeros])}): "
            "zeros miscounted"
        ),
        grade.WRONG_PERIOD: (
            "another period's figure: another column of the answer's row, or another "
            "period's XBRL value"
        ),
        grade.NONCONTROLLING_INTEREST: (
            "net income including noncontrolling interests (XBRL ProfitLoss)"
        ),
        grade.ROUNDED: "the answer rounded to a coarser power of ten",
        grade.ADJUSTED: (
            "a figure on a row or line labelled adjusted, non-GAAP or excluding"
        ),
        grade.WRONG_LINE: "a figure from another line of the release",
        grade.OTHER: "none of the above",
        grade.NOT_IN_INPUT: (
            "the release doesn't print the answer in any form the grader searches; not "
            "graded"
        ),
        grade.NO_ANSWER: (
            "a figure for a period without an XBRL answer, such as fourth-quarter EPS; "
            "not graded"
        ),
    }
    if list(meanings) != list(grade.VERDICTS):
        raise ExportError("grade.VERDICTS changed: describe each verdict in hfspace")
    return meanings


def printed_forms_text() -> str:
    """The printed forms the grader searches, in words, from grade's constants."""
    usd = [USD_FORM_WORDS.get(form) for form in grade.USD_FORMS]
    eps = [NUMBER_WORDS.get(d) for d in grade.EPS_DECIMALS]
    if None in usd or None in eps:
        raise ExportError("grade's printed forms changed: describe them in hfspace")
    return f"{joined(usd)}, or EPS to {joined(eps)} decimals"


def factor_range() -> str:
    factors = sorted(int(f) for f in grade.FACTORS if f > 1)
    return f"{factors[0]:,} to {factors[-1]:,}"


def unit_slips_text() -> str:
    """What a unit slips count holds (the rows of graded.csv with a factor), as
    results/comparison.md defines it: not only figures off from the answer, since
    any check may match another figure after a power of ten."""
    return (
        "the figures the grader matched with a power-of-ten factor (scale and zeros, "
        "and the other errors matched only after multiplying or dividing the "
        f"prediction by {factor_range()})"
    )


def split_counts(ctx: Context) -> dict[str, dict[str, Any]]:
    """Each split's releases, kept and left out, companies and answers, from the rows."""
    found = {}
    for split in hfcommon.SPLIT_NAMES:
        rows = ctx.releases[split]
        kept = [r for r in rows if r.kept]
        found[split] = {
            "releases": len(rows),
            "kept": len(kept),
            "left_out": len(rows) - len(kept),
            "companies": len({r.ticker for r in kept}),
            "answers": len(ctx.answers[split]),
            "answers_in_input": sum(a.in_input for a in ctx.answers[split]),
            "first_filed": min(r.filing_date for r in rows),
            "last_filed": max(r.filing_date for r in rows),
        }
    return found


def split_descriptions(ctx: Context, counts: Mapping[str, Mapping[str, Any]]) -> dict:
    v, c, t = counts[VALIDATION], counts[COMPANY_TEST], counts[TIME_TEST]
    time_test = ctx.test_results["time_test"]
    one = t["left_out"] == 1
    left_out = (
        f" {t['left_out']} more {'is' if one else 'are'} left out for every model: "
        f"{'its prompt exceeds' if one else 'their prompts exceed'} "
        f"{splits.MAX_PROMPT_TOKENS:,} tokens (the GPU model's context minus the "
        f"reply's allowance). Explore shows {'it' if one else 'them'}, flagged."
        if t["left_out"]
        else ""
    )
    return {
        VALIDATION: (
            f"Stage 1's {v['kept']} releases from {v['companies']} companies, filed "
            f"{v['first_filed']} to {v['last_filed']}: every model of stage 1 ran "
            "on them, and they chose the fine-tuned checkpoint."
        ),
        COMPANY_TEST: (
            f"{c['kept']} releases from {c['companies']} companies never trained on "
            f"(each company's latest {COMPANY_TEST_RELEASES}), filed "
            f"{c['first_filed']} to {c['last_filed']}. The primary result is "
            "measured here."
        ),
        TIME_TEST: (
            f"{t['kept']} releases from {t['companies']} of the training companies, "
            f"filed {time_test['first_filed']} to {time_test['last_filed']}, after the "
            f"releases trained on.{left_out}"
        ),
    }


def explore_status(split: str, counts: Mapping[str, Any]) -> str:
    """Explore's line when no filter is chosen: the split's releases, with those left
    out for every model counted (the Leaderboard counts only the kept ones)."""
    n = counts["releases"]
    text = (
        f"{n:,} release{'' if n == 1 else 's'} in the "
        f"{SPLIT_TITLES[split].lower()} split"
    )
    if counts["left_out"]:
        text += f", {counts['left_out']:,} of them left out for every model"
    return text + "."


def meta(ctx: Context, serving: bool) -> dict[str, Any]:
    counts = split_counts(ctx)
    descriptions = split_descriptions(ctx, counts)
    meanings = verdict_meanings()
    return {
        "title": "SEC earnings extraction: results explorer",
        "subtitle": (
            "How accurately LLMs extract GAAP revenue, net income and diluted EPS from "
            "US earnings press releases, graded automatically against XBRL in the "
            "company's later 10-Q or 10-K."
        ),
        "links": links(ctx),
        "repos": {
            "dataset": ctx.repo_id(DATASET),
            "model": ctx.repo_id(MODEL),
            "space": ctx.repo_id(SPACE),
        },
        "splits": [
            {
                "name": split,
                "title": SPLIT_TITLES[split],
                "description": descriptions[split],
                "status": explore_status(split, counts[split]),
                **counts[split],
            }
            for split in hfcommon.SPLIT_NAMES
        ],
        "default_split": COMPANY_TEST,
        "fields": [{"name": f, "title": FIELD_TITLES[f]} for f in hfcommon.FIELD_NAMES],
        "periods": [
            {"name": p, "title": PERIOD_TITLES[p]} for p in hfcommon.PERIOD_TYPES
        ],
        "verdicts": [
            {"name": v, "meaning": meanings[v], "graded": v not in grade.UNGRADED}
            for v in grade.VERDICTS
        ],
        "correct": grade.CORRECT,
        "serving": serving,
        "about": about(ctx, counts),
    }


def links(ctx: Context) -> dict[str, str]:
    blob = f"{REPO_URL}/blob/main"
    return {
        "github": REPO_URL,
        "dataset": ctx.hub_url(DATASET),
        "model": ctx.hub_url(MODEL),
        "space": ctx.hub_url(SPACE),
        "readme": f"{REPO_URL}#readme",
        "test_results": f"{blob}/results/test_results.md",
        "deviations": f"{blob}/results/test_results.md#deviations-from-the-protocol",
        "comparison": f"{blob}/results/comparison.md",
        "protocol": f"{blob}/results/test_protocol.md",
    }


def about(ctx: Context, counts: Mapping[str, Mapping[str, Any]]) -> str:
    """The About tab, in Markdown."""
    results = ctx.test_results
    m = results["model_under_test"]
    lora, checkpoint = m["lora"], m["checkpoint"]
    under, base = results["under_test"], results["base"]
    labels = results["models"]
    chosen = results["validation"]["counts"][under]["all"]
    protocol, boot = results["protocol"], results["bootstrap"]
    url = links(ctx)
    names = company_names(ctx)
    descriptions = split_descriptions(ctx, counts)
    tags = (
        ";\n".join(
            f"- {FIELD_TITLES[f.name]}: " + ", then ".join(f"`{t}`" for t in f.tags)
            for f in answers.FIELDS
        )
        + "."
    )
    verdicts = "\n".join(
        f"| {name} | {meaning} |" for name, meaning in verdict_meanings().items()
    )
    gaps = "; ".join(
        f"the {SPLIT_TITLES[s].lower()} {gap_text(results['splits'][s]['coverage'])}"
        for s in hfcommon.TEST_SPLITS
    )
    lines = [
        "## How the benchmark works",
        "",
        (
            "**Releases.** Earnings press releases from SEC EDGAR: 8-K filings that "
            "report Item 2.02 (results of operations). The press release, usually "
            "EX-99.1, is converted from HTML to Markdown with its tables intact; if it "
            "has no income statement, the exhibit that has one is appended. That text "
            "is the model's input, the text Explore shows."
        ),
        "",
        (
            "**Answer key.** Each release is linked to the first 10-Q or 10-K filed on "
            "or after it whose fiscal period ended at most "
            f"{link.MAX_PERIOD_AGE.days} days before it. The answers are XBRL company "
            "facts reported in that filing (matched by accession number) for its "
            "period; the first tag with a value wins (some of the later ones count "
            "only under conditions the README states):"
        ),
        "",
        tags,
        "",
        (
            "A 10-Q gives the quarter and the fiscal year to date, a 10-K the full "
            "year. Fourth-quarter revenue and net income are derived as the 10-K's "
            "full year minus the third-quarter 10-Q's nine months. Fourth-quarter EPS "
            "can't be derived by subtraction, so it isn't graded."
        ),
        "",
        (
            "**Grader.** A figure is graded only if the release prints its answer, in "
            f"{printed_forms_text()}; otherwise it is not_in_input. A graded figure "
            "gets the first verdict that explains it:"
        ),
        "",
        "| verdict | meaning |",
        "| --- | --- |",
        verdicts,
        "",
        (
            f"Unit slips are {unit_slips_text()}. Accuracy is correct figures over "
            "graded figures."
        ),
        "",
        (
            "**Splits.** By company, fixed with a seed before the new companies' "
            "filings were downloaded."
        ),
        "",
        *(f"- **{SPLIT_TITLES[s]}:** {descriptions[s]}" for s in hfcommon.SPLIT_NAMES),
        "",
        (
            f"**The fine-tuned model.** {m['base_model']} with a LoRA adapter (r "
            f"{lora['r']}, alpha {lora['alpha']}) on all "
            f"{len(lora['target_modules'])} projections of every layer, thinking off. "
            "Each training label is written by code from the XBRL answer key, as the "
            "release prints the figure, never from a model's output: "
            f"{m['labelled']:,} training releases were labelled and "
            f"{m['selected']:,} drawn from the {m['within_max_tokens']:,} under "
            f"{m['max_tokens']:,} tokens. The checkpoint, {checkpoint['label']}, had "
            f"seen {checkpoint['examples_seen']:,} examples; its validation score, "
            f"{validation_score(chosen)}, is {testreport.CHOSEN_ON_VALIDATION}."
        ),
        "",
        (
            "**The protocol.** The test analysis was fixed before any test call, in "
            f"[results/test_protocol.md]({url['protocol']}) (commit "
            f"{protocol['commit'][:7]}, sha256 {protocol['sha256'][:8]}…). The primary "
            f"result is {labels[under]}'s accuracy minus {labels[base]}'s on the "
            "company test, with a 95% bootstrap interval over companies "
            f"({boot['draws']:,} draws, seed {boot['seed']}); everything else is "
            "descriptive. The deviations from it, the time test's cutoff among them, "
            f"are listed in [results/test_results.md]({url['deviations']})."
        ),
        "",
        "**Limitations.**",
        "",
        "Validation (stage 1's releases):",
        "",
        (
            "- **Small, and not held out:** "
            f"{counts[VALIDATION]['answers_in_input']:,} graded figures from "
            f"{counts[VALIDATION]['companies']} companies, and prompt v2 was designed "
            "on these releases (the test sets are held out)."
        ),
        "",
        "The test sets (stage 2):",
        "",
        cutoff_line(results, url["deviations"]),
        f"- **Answer-key gaps:** {gaps}.",
        *note_lines(ctx, names),
        *optional(drawn_line(ctx, names)),
        training_line(ctx),
        (
            "- **Survivorship:** the companies are the S&P 500's current non-financial "
            "members only."
        ),
        f"- **Abstention isn't scored:** {abstention_text(results)}",
        "",
        "Both:",
        "",
        (
            "- **Fourth-quarter EPS isn't graded:** XBRL has no fourth-quarter value, "
            "and EPS can't be derived by subtraction."
        ),
        (
            "- **Sampling:** one run per model and split. Runs without temperature 0 "
            "vary from run to run (the API models' sampling isn't fixed), and no "
            "repeat runs measure that variation."
        ),
        (
            "- **GPU cost:** each run's wall-clock time only, not model loading, setup "
            "or idle time; GPU costs and latencies compare only between runs on the "
            "same GPU."
        ),
        "",
        (
            f"**Where {labels[under]} fails** (descriptive, not pre-registered; every "
            f"miss is listed in [results/test_results.md]({url['test_results']})):"
        ),
        "",
        *failure_lines(results, names),
        "",
        (
            f"**Links.** [Code and README]({url['github']}) · "
            f"[dataset]({url['dataset']}) · [fine-tuned model]({url['model']}) · "
            f"[test results]({url['test_results']}) · "
            f"[validation comparison]({url['comparison']})"
        ),
        "",
        (
            f'Load the answers: `load_dataset("{ctx.repo_id(DATASET)}", "answers", '
            'split="company_test")`.'
        ),
    ]
    return "\n".join(lines) + "\n"


def optional(line: str | None) -> list[str]:
    return [] if line is None else [line]


def company_names(ctx: Context) -> dict[str, str]:
    """Each company's name by ticker (data/companies.csv, the drawn universe)."""
    return {r["ticker"]: r["name"] for r in hfcommon.read_csv(ctx.sources.companies)}


def name_of(names: Mapping[str, str], ticker: str) -> str:
    """'Boeing (BA)', or the ticker alone if the company isn't listed."""
    return f"{names[ticker]} ({ticker})" if ticker in names else ticker


def companies_text(tickers: Sequence[str], names: Mapping[str, str]) -> str:
    """A few companies by name, more by their count."""
    if len(tickers) > 3:
        return testreport.company_count(len(tickers))
    return joined([name_of(names, t) for t in tickers], "and")


def quarter_of(day: str) -> str:
    """The calendar quarter of an ISO date, as 'Q3 2026'."""
    year, month = int(day[:4]), int(day[5:7])
    return f"Q{(month - 1) // 3 + 1} {year}"


def cutoff_line(results: Mapping[str, Any], deviations: str) -> str:
    """The time test's cutoff, a deviation from the protocol (results/test_results.md,
    Deviations, 1), from test_results.json's time_test record and the runs' times."""
    t = results["time_test"]
    company, time_test = (results["splits"][s] for s in hfcommon.TEST_SPLITS)
    quarter = quarter_of(t["last_filed"])
    frozen = t["frozen_at"]
    when = [
        clause
        for clause, holds in (
            ("after the company test was scored", frozen > company["runs_finished"]),
            ("before any time-test call", frozen < time_test["runs_started"]),
        )
        if holds
    ]
    return (
        f"- **No {quarter} releases in the time test:** it holds the releases filed "
        f"{t['first_filed']} to {t['last_filed']}, as the {quarter} 10-Qs, which "
        "give the answer key, weren't all filed yet. Its cutoff was set on "
        f"{frozen[:10]}"
        + (f", {joined(when, 'and')}," if when else ",")
        + f" though the protocol said it would be set when the {quarter} releases "
        f"were added: a [deviation]({deviations})."
    )


def note_lines(ctx: Context, names: Mapping[str, str]) -> list[str]:
    """The test sets' noted releases (test_results.json's notes), such as one that
    isn't a results release."""
    lines = []
    for split in hfcommon.TEST_SPLITS:
        for n in ctx.test_results["splits"][split]["notes"]:
            printed = n["answers"] - n["not_in_input"]
            effect = (
                f"none of its {n['answers']} answers is printed, so it adds no graded "
                "figure"
                if not printed
                else f"{printed} of its {n['answers']} answers are printed"
            )
            company = f" ({names[n['ticker']]})" if n["ticker"] in names else ""
            lines.append(
                f"- **{SPLIT_TITLES[split]} release {n['ticker']} "
                f"{n['release_accession']}**{company} is {n['note']}; {effect}."
            )
    return lines


def drawn_line(ctx: Context, names: Mapping[str, str]) -> str | None:
    """The companies drawn for the company test (data/companies.csv) that have no
    release in it; None if every one has."""
    drawn = sorted(
        r["ticker"]
        for r in hfcommon.read_csv(ctx.sources.companies)
        if r["split"] == COMPANY_TEST
    )
    present = {r.ticker for r in ctx.releases[COMPANY_TEST]}
    absent = [t for t in drawn if t not in present]
    if not absent:
        return None
    return (
        f"- **{len(drawn) - len(absent)} of the {len(drawn)} companies drawn for the "
        f"company test:** {joined(absent, 'and')} "
        f"{'has' if len(absent) == 1 else 'have'} no linked release."
    )


def training_line(ctx: Context) -> str:
    """Where the checkpoint chosen on validation falls among those the one training
    run saved (training/qwen3-8b-2k/validation.json)."""
    record = hfcommon.read_json(ctx.sources.training / "validation.json")
    saved = [
        m["label"]
        for m in sorted(record["models"], key=lambda m: m["step"])
        if m["step"] > 0
    ]
    chosen = record["chosen"]
    if chosen not in saved:
        raise ExportError(f"validation.json: {chosen} isn't a saved checkpoint")
    place = saved.index(chosen)
    which = "the first" if place == 0 else f"number {place + 1}"
    return (
        "- **One training run, untuned:** the checkpoint chosen on validation, "
        f"{chosen}, is {which} of the {len(saved)} it saved."
    )


def failure_lines(results: Mapping[str, Any], names: Mapping[str, str]) -> list[str]:
    """The model under test's main failures on the test sets, from test_results.json's
    miss summaries (testreport.group_misses): the common shareholders' line, then the
    quarter copied into year to date; time test first, where most misses are."""
    under = results["models"][results["under_test"]]
    preferred = "releases whose filing reports preferred dividends"
    common: list[str] = []
    for split in (TIME_TEST, COMPANY_TEST):
        m = results["splits"][split]["miss_summary"]
        count = m["groups"].get(testreport.COMMON, 0)
        if not count:
            continue
        inside, outside = m["common_in_preferred"], m["common_outside_preferred"]
        where = (
            f"all are in {preferred}"
            if not outside
            else f"none is in {preferred}"
            if not inside
            else f"{inside} are in {preferred} and {outside} are not"
        )
        if outside:
            # As testreport says it: the sensitivity analysis without those releases
            # leaves these misses in.
            where += ", so the preferred-dividend sensitivity understates it"
        whose = "its" if common else f"{under}'s"
        broken = (
            f"; they include {m['broken_common']} of the {m['broken']} figures "
            "training broke" + ("" if common else " (the base model had them right)")
            if m["broken"]
            else ""
        )
        common.append(
            f"{SPLIT_TITLES[split]}: {count} of {whose} {m['misses']} misses, from "
            f"{companies_text(m['common_companies'], names)}; {where}{broken}."
        )
    copied = []
    for split in hfcommon.TEST_SPLITS:
        m = results["splits"][split]["miss_summary"]
        n = m["groups"].get(testreport.QUARTER_AS_YTD, 0)
        if n:
            copied.append(
                f"{n} figure{'' if n == 1 else 's'} on the "
                f"{SPLIT_TITLES[split].lower()}, from "
                f"{companies_text(m['quarter_as_ytd_companies'], names)}"
            )
    lines = []
    if common:
        lines.append(
            "- **The common shareholders' line:** net income read from a line naming "
            "common shareholders, stockholders, shares or stock (income after "
            "preferred dividends or participating securities), where the answer key's "
            "`NetIncomeLoss` comes before them. " + " ".join(common)
        )
    if copied:
        lines.append(
            "- **The quarter copied into year to date:** in second- or third-quarter "
            "releases whose income statement has quarter columns only, with the year "
            f"to date printed elsewhere, {under} repeated the quarter: "
            f"{'; '.join(copied)}."
        )
    return lines


def abstention_text(results: Mapping[str, Any]) -> str:
    """What the models did with the not_in_input figures on the test sets, as the
    README says it, from test_results.json's abstention counts."""
    labels = results["models"]
    qwen = [results["under_test"], results["base"]]
    api = [m for m in labels if m not in qwen]
    entries = [results["splits"][s]["abstention"] for s in hfcommon.TEST_SPLITS]
    never = all(e[m]["null"] == 0 for e in entries for m in qwen)
    empty = [
        100 * (e[m]["not_in_input"] - e[m]["answered"]) / e[m]["not_in_input"]
        for e in entries
        for m in api
        if e[m]["not_in_input"]
    ]
    parts = []
    if never:
        parts.append(
            f"neither {joined([labels[m] for m in qwen], 'nor')} ever answered null "
            "on the test sets"
        )
    if empty:
        parts.append(
            f"the API models left {min(empty):.0f}-{max(empty):.0f}% of the "
            "not_in_input figures empty: XBRL answers the grader can't find printed, "
            "which a release sometimes reports in another form"
        )
    told = ", while ".join(parts)
    rest = "Accuracy leaves those figures out, and no training label is null."
    return f"{told}. {rest}" if told else rest


def validation_score(counts: Sequence[int]) -> str:
    """'98.0% (287/293)', as results/test_results.md writes step-00025's."""
    correct, graded = counts
    return f"{testreport.num(100 * correct / graded, 1)}% ({correct}/{graded})"


def gap_text(coverage: Mapping[str, Any]) -> str:
    missing, expected = coverage["missing"], coverage["expected"]
    return (
        f"has no XBRL answer for {missing:,} of the {expected:,} figures its kept "
        f"releases should report ({testreport.num(100 * missing / expected, 1)}%)"
    )


# leaderboard.json


VALIDATION_COLUMNS = (
    ("model", "model"),
    ("runs on", "runs_on"),
    ("prompt", "prompt"),
    ("accuracy", "accuracy"),
    ("revenue", "revenue"),
    ("net_income", "net_income"),
    ("eps_diluted", "eps_diluted"),
    ("unit slips", "unit_slips"),
    ("cost / 1,000 releases", "cost_per_1000"),
    ("calls at a time", "concurrency"),
    ("latency p50", "latency_p50"),
    ("latency p95", "latency_p95"),
)


def leaderboard(ctx: Context, charts: Mapping[str, str]) -> dict[str, Any]:
    return {
        "intro": (
            "Accuracy is correct figures over graded figures: answers the release "
            "prints, graded against XBRL in the company's later 10-Q or 10-K "
            "(not_in_input and no_answer figures are left out). See About for the "
            "verdicts."
        ),
        "validation": validation_board(ctx, charts["validation"]),
        "test": test_board(ctx, charts["test"]),
    }


def gpu_name(line: str) -> str:
    """A GPU as README's tables name it: nvidia-smi's line ("NVIDIA H100 NVL, 95830
    MiB, 580.178.04") without the vendor and the details ("H100 NVL")."""
    return line.split(",")[0].strip().removeprefix("NVIDIA ")


def gpu_cell(names: Sequence[str]) -> str:
    """Each GPU with its count, as README's 'Runs on' column writes it."""
    return ", ".join(f"{n}× {name}" for name, n in Counter(names).items())


def is_stage2_run(run: hfcommon.ExportRun) -> bool:
    """A run of stage 2's training run (its base model or a checkpoint)."""
    return run.name.startswith(f"{TRAINING_RUN}--")


def runs_on(results: Mapping[str, Any], run: hfcommon.ExportRun) -> str:
    """Where a run ran: its provider's API, or for a GPU run the GPU, as README's
    tables have it. On validation, stage 2's runs ran on test_results.json's
    validation GPU (one, as testreport says it) and the others on stage 1's; on a
    test split, the GPUs that split's record lists."""
    if run.source != SELF_HOSTED:
        return run.source
    if run.split != VALIDATION:
        return gpu_cell(
            [gpu_name(g) for g in results["model_under_test"]["gpus"][run.split]]
        )
    return gpu_cell(
        [results["validation"]["gpu"] if is_stage2_run(run) else STAGE1_GPU]
    )


def validation_gpus(
    results: Mapping[str, Any], runs: Sequence[hfcommon.ExportRun]
) -> list[str]:
    """The validation GPU runs' GPUs in words: stage 1's, then stage 2's."""
    gpu_runs = [r for r in runs if r.source == SELF_HOSTED]
    found = []
    if any(not is_stage2_run(r) for r in gpu_runs):
        found.append(f"one {STAGE1_GPU} for stage 1's open models")
    stage2 = [r for r in gpu_runs if is_stage2_run(r)]
    if stage2:
        v = results["validation"]
        which = [
            *(
                [results["models"][results["base"]]]
                if any(r.name == f"{TRAINING_RUN}--{results['base']}" for r in stage2)
                else []
            ),
            *(
                ["the step-* checkpoints"]
                if any(r.name.startswith(f"{TRAINING_RUN}--step-") for r in stage2)
                else []
            ),
        ]
        found.append(
            f"one {v['gpu']} ({results['gpu_host']}, billed "
            f"${v['billed_usd_per_hour']:g}/h) for the {TRAINING_RUN} runs"
            + (f" ({joined(which, 'and')})" if which else "")
        )
    return found


def repeated_runs(runs: Sequence[hfcommon.ExportRun]) -> set[str]:
    """The runs that share their model and prompt with another run (repeated runs),
    by name."""
    shared = Counter(run_label(r) for r in runs)
    return {r.name for r in runs if shared[run_label(r)] > 1}


def with_run(label: str, run: hfcommon.ExportRun, repeated: set[str]) -> str:
    """A repeated run's label gains its run's name, which is unique."""
    return f"{label} · {run.name}" if run.name in repeated else label


def validation_board(ctx: Context, chart: str) -> dict[str, Any]:
    runs = ctx.validation_runs
    table = hfcommon.validation_table(runs)
    results = ctx.test_results
    under = results["under_test"]
    reference = results["validation"]["counts"][under]
    graded = sorted({r["graded"] for r in table})
    graded_text = (
        f"the {graded[0]:,} graded figures of each run"
        if len(graded) == 1
        else "each run's graded figures"
    )
    counts = split_counts(ctx)[VALIDATION]
    repeated = repeated_runs(runs)
    gpus = validation_gpus(results, runs)
    hourly = f"${hfcommon.GPU_HOURLY:.2f} per GPU-hour times the run's wall-clock time"
    notes = [
        (
            f"- **Accuracy:** correct figures out of {graded_text} (revenue, net "
            "income and diluted EPS, for the quarter and the fiscal year to date); the "
            "field columns count each field alone."
        ),
        f"- **Unit slips:** {unit_slips_text()}.",
        (
            "- **Runs on:** the provider's API, or the GPU an open model was served on "
            f"with vLLM: {joined(gpus, 'and')}."
            if gpus
            else "- **Runs on:** the provider's API."
        ),
        (
            f"- **Cost:** API runs at list token prices; GPU runs at {hourly}, "
            "whatever each run paid. That compares runs on the same GPU, not runs on "
            "different GPUs (a faster GPU finishes the same work sooner), so compare "
            "GPU costs and latencies only between runs on the same GPU."
            if len(gpus) > 1
            else f"- **Cost:** API runs at list token prices; GPU runs at {hourly}, "
            "so they compare with each other, whatever each run paid."
        ),
        (
            "- **Latency:** per call, median and 95th percentile, with the run's calls "
            "at a time."
        ),
        (
            "- **Prompts:** extract_v1 asks for raw dollars; extract_v2 asks for "
            "revenue and net income as printed, with their unit, and converts in code."
        ),
    ]
    if repeated:
        notes.append(
            "- **Repeated runs** of one model and prompt show their run's name beside "
            "the model's."
        )
    if reference["run"] in {r["run"] for r in table}:
        notes.append(
            f"- **{results['models'][under]}** is stage 2's fine-tuned checkpoint "
            "(Qwen3-8B with a LoRA adapter); the other step-* rows are the other "
            "checkpoints. Its validation score, "
            f"{validation_score(reference['all'])}, is "
            f"{testreport.CHOSEN_ON_VALIDATION}."
        )
    rows = [
        {
            **row,
            "model": with_run(row["model"], run, repeated),
            "runs_on": runs_on(results, run),
        }
        for run, row in zip(runs, table, strict=True)
    ]
    return {
        "title": (
            f"Validation: {counts['kept']} releases from {counts['companies']} "
            f"companies, {len(table)} runs"
        ),
        "columns": [column for column, _ in VALIDATION_COLUMNS],
        "rows": [[row[key] for _, key in VALIDATION_COLUMNS] for row in rows],
        "runs": [row["run"] for row in table],
        "notes": "\n".join(notes) + "\n",
        "source": "results/comparison.md",
        "chart": chart,
        "chart_caption": CHART_CAPTIONS["validation"],
    }


def test_columns(graded: int) -> list[str]:
    return [
        "model",
        "runs on",
        f"accuracy, {graded:,} figures [95% interval]",
        *hfcommon.FIELD_NAMES,
        "cost / 1,000 releases",
    ]


def cost_cell(split: Mapping[str, Any], key: str) -> str:
    """Cost per 1,000 releases at the price billed; for the run with the instance
    freeze, its freeze-excluded estimate beside it, as the README shows it."""
    text = testreport.money(split["cost"][key]["per_1000"])
    freeze = split.get("freeze")
    if freeze and freeze["run"] == split["runs"][key]["run"]:
        text += (
            f" (about {testreport.money(freeze['excluded_per_1000'])} without the "
            "freeze)"
        )
    return text


def test_board(ctx: Context, chart: str) -> dict[str, Any]:
    results = ctx.test_results
    labels = results["models"]
    under, base = results["under_test"], results["base"]
    company = results["splits"][COMPANY_TEST]
    primary = company["differences"][base]["all"]
    chosen = results["validation"]["counts"][under]["all"]
    tables = []
    for split in hfcommon.TEST_SPLITS:
        s = results["splits"][split]
        places = {r.key: runs_on(results, r) for r in ctx.test_runs[split]}
        if list(places) != list(labels):
            raise ExportError(f"{split}'s runs aren't test_results.json's models")
        cov = s["coverage"]
        tables.append(
            {
                "name": split,
                "title": (
                    f"{SPLIT_TITLES[split]}: {cov['kept_releases']} releases from "
                    f"{cov['kept_companies']} companies"
                    + (
                        f" ({len(s['left_out'])} more left out for every model)"
                        if s["left_out"]
                        else ""
                    )
                ),
                "columns": test_columns(cov["graded"]),
                "rows": [
                    [
                        labels[m],
                        places[m],
                        testreport.acc_cell(s["accuracy"][m]["all"]),
                        *(
                            testreport.acc_cell(s["accuracy"][m][f])
                            for f in hfcommon.FIELD_NAMES
                        ),
                        cost_cell(s, m),
                    ]
                    for m in labels
                ],
                "keys": list(labels),
                "differences": "\n".join(
                    testreport.difference_table(s, "all", labels, under, base)
                )
                + "\n",
            }
        )
    return {
        "title": "Held-out test sets (stage 2)",
        "primary": (
            "**Primary result (pre-registered).** On the company test, "
            f"{labels[under]} minus {labels[base]}: "
            f"**{testreport.num(primary['difference'], signed=True)} points "
            f"{testreport.bounds(primary['interval'], signed=True)}**, from "
            f"{testreport.num(company['accuracy'][base]['all']['accuracy'])}% to "
            f"{testreport.num(company['accuracy'][under]['all']['accuracy'])}% of "
            f"{company['coverage']['graded']:,} figures; {primary['discordant']} "
            f"discordant figures ({primary['first_right']} fixed by training, "
            f"{primary['second_right']} broken) from {primary['companies']} "
            "companies. Everything else is descriptive under the protocol."
        ),
        "summary": test_summary(results),
        "splits": tables,
        "notes": test_notes(results, chosen),
        "source": "results/test_results.md",
        "chart": chart,
        "chart_caption": CHART_CAPTIONS["test"],
    }


def test_summary(results: Mapping[str, Any]) -> str:
    """How the fine-tuned model compares with the API models, from the differences
    and costs, as the README says it (it trails them on the time test; its case is
    cost)."""
    labels = results["models"]
    under, base = results["under_test"], results["base"]
    api = [m for m in labels if m not in (under, base)]
    time_test = results["splits"][TIME_TEST]
    diffs = [time_test["differences"][m]["all"] for m in api]
    gaps = sorted(-d["difference"] for d in diffs)
    if all(d["interval"][1] < 0 for d in diffs):
        trail = (
            f"On the time test, {labels[under]} trails all {count_word(len(api))} API "
            "models, by "
            f"{testreport.num(gaps[0])} to {testreport.num(gaps[-1])} points, with 95% "
            "intervals that exclude 0."
        )
    else:
        trail = (
            f"On the time test, {labels[under]} minus each API model: "
            + "; ".join(
                f"{labels[m]} {testreport.num(d['difference'], signed=True)} "
                f"{testreport.bounds(d['interval'], signed=True)}"
                for m, d in zip(api, diffs, strict=True)
            )
            + "."
        )
    costs = time_test["cost"]
    cheaper = sorted(api, key=lambda m: costs[m]["per_1000"])
    return (
        f"{trail} Its case is cost, not accuracy: on the time test it costs "
        f"{testreport.money(costs[under]['per_1000'])} per 1,000 releases, against "
        + joined(
            [
                f"{testreport.money(costs[m]['per_1000'])} for {labels[m]}"
                for m in cheaper
            ],
            "and",
        )
        + "."
    )


def test_notes(results: Mapping[str, Any], chosen: Sequence[int]) -> str:
    m = results["model_under_test"]
    labels = results["models"]
    under = results["under_test"]
    gpus = sorted({gpu_name(g) for gs in m["gpus"].values() for g in gs})
    servers = sorted(set(m["server"].values()))
    company, time_test = (results["splits"][s] for s in hfcommon.TEST_SPLITS)
    cov = time_test["coverage"]
    freeze = company.get("freeze")
    notes = [
        (
            f"- **{labels[under]}** is {m['base_model']} with the fine-tuned LoRA "
            f"adapter. Its validation score, {validation_score(chosen)}, is "
            f"{testreport.CHOSEN_ON_VALIDATION}."
        ),
        (
            "- **Accuracy:** correct figures over graded figures, with 95% bootstrap "
            f"intervals over companies ({results['bootstrap']['draws']:,} draws). On "
            f"the time test, the {cov['releases_graded']} releases with graded figures "
            f"come from {cov['companies_graded']} companies, so there the bootstrap is "
            "in effect over releases."
        ),
        (
            f"- **Runs on:** {labels[under]} and {labels[results['base']]} on "
            f"{joined(servers, 'and')} on one {results['gpu_host']} "
            f"{joined(gpus, 'and')} per test split; the API models on their providers' "
            "APIs."
        ),
        (
            "- **Cost:** GPU runs at the price billed "
            f"(${company['billed_usd_per_hour']:g} per hour for the company test's "
            f"session, ${time_test['billed_usd_per_hour']:g} for the time test's) "
            "times the run's wall-clock time, leaving out setup, model loading and "
            f"idle time; API runs at list token prices.{recorded_price_text(results)}"
        ),
    ]
    if freeze:
        notes.append(
            f"- **The freeze:** the GPU instance froze for {freeze['window_s']:.0f} s "
            f"during the company test's {labels[under]} run, so its wall-time cost "
            "includes that time; the estimate without it is beside it."
        )
    notes.append(
        "- **Differences:** "
        f"{labels[under]} minus each other model, in points, with the discordant "
        "figures (one model correct, the other not) and the companies they come from."
    )
    return "\n".join(notes) + "\n"


def recorded_price_text(results: Mapping[str, Any]) -> str:
    """For a test split whose GPU runs record another price than the one billed, the
    costs at the recorded price, the price by the protocol's letter
    (results/test_results.md, Deviations, 2); "" if none does."""
    labels = results["models"]
    parts = []
    for split in hfcommon.TEST_SPLITS:
        s = results["splits"][split]
        gpu = {
            m: c
            for m, c in s["cost"].items()
            if c["basis"] == "gpu"
            and c["recorded_usd_per_hour"] != c["billed_usd_per_hour"]
        }
        if not gpu:
            continue
        prices = sorted({c["recorded_usd_per_hour"] for c in gpu.values()})
        why = f" ({s['recorded_price_note']})" if s.get("recorded_price_note") else ""
        costs = [
            f"{labels[m]} {testreport.money(c['per_1000_at_recorded'])}"
            for m, c in gpu.items()
        ]
        parts.append(
            f" The {SPLIT_TITLES[split].lower()}'s GPU runs record "
            f"{joined([f'${p:g}/h' for p in prices], 'and')}{why}, the price by the "
            f"protocol's letter; at it, per 1,000 releases: {joined(costs, 'and')}."
        )
    return "".join(parts)


# runs.json, releases.json, answers.json, predictions/<split>.json


def run_label(run: hfcommon.ExportRun) -> str:
    """The run's name in the app: the model, and on validation (where models ran on
    both prompts) the prompt as the README writes it (v1, v2). Repeated runs of one
    model and prompt share it; run_labels tells them apart."""
    if run.split != VALIDATION:
        return run.model
    return f"{run.model} · {run.prompt.removeprefix('extract_')}"


def run_labels(runs: Sequence[hfcommon.ExportRun]) -> list[str]:
    """Each run's label in the app, unique: a repeated run's gains its run's name
    ("Qwen3-8B (thinking) · v2 · <run>")."""
    repeated = repeated_runs(runs)
    labels = [with_run(run_label(r), r, repeated) for r in runs]
    if len(set(labels)) != len(labels):  # run names are unique folder names
        raise ExportError("two runs would show under one name")
    return labels


def default_runs(ctx: Context, split: str) -> list[str]:
    """The runs Explore shows first: on validation, the test sets' models' validation
    references (stage 2's runs) present among its runs; on a test split, all."""
    names = [r.name for r in ctx.runs(split)]
    if split != VALIDATION:
        return names
    references = [c["run"] for c in ctx.test_results["validation"]["counts"].values()]
    shown = [n for n in references if n in names]
    return shown or names


def runs_data(ctx: Context) -> dict[str, dict[str, Any]]:
    """Per split: its runs in the Context's order (comparison.md's on validation,
    test_results.json's on a test split), and the runs Explore shows first, in the
    order it shows them."""
    found = {}
    for split in hfcommon.SPLIT_NAMES:
        runs = ctx.runs(split)
        labels = run_labels(runs)
        found[split] = {
            "runs": [
                {
                    "run": r.name,
                    "key": r.key,
                    "label": label,
                    "model": r.model,
                    "model_id": r.model_id,
                    "source": r.source,
                    "prompt": r.prompt,
                    "settings": r.settings,
                }
                for r, label in zip(runs, labels, strict=True)
            ],
            "default": default_runs(ctx, split),
        }
    return found


def releases_data(ctx: Context) -> dict[str, list[dict[str, Any]]]:
    """Each split's releases (hfcommon's rows without the text), with why a left-out
    one is left out in words (left_out_text)."""
    found = {}
    for split in hfcommon.SPLIT_NAMES:
        found[split] = []
        for row in ctx.releases[split]:
            record = hfcommon.as_json(row)
            del record["split"], record["text"]
            record["left_out_reason"] = left_out_text(row)
            record["text_file"] = text_path(split, row.release_accession)
            record["text_chars"] = len(row.text)
            found[split].append(record)
    return found


def left_out_text(row: hfcommon.ReleaseRow) -> str | None:
    """Why a release is left out for every model, from its prompt's token count and
    the limit: the split's own reason (secbench.splits) cites a planning document
    that isn't published with the repository."""
    if row.kept:
        return None
    limit = splits.MAX_PROMPT_TOKENS
    if (
        row.left_out_reason != splits.left_out_reason(limit)
        or row.prompt_tokens <= limit
    ):
        raise ExportError(
            f"{row.split} {row.release_accession}: left out for a reason hfspace "
            f"doesn't describe ({row.left_out_reason!r}, {row.prompt_tokens:,} "
            "tokens)"
        )
    return (
        f"its prompt is {row.prompt_tokens:,} tokens, over the {limit:,}-token limit "
        f"(the GPU model's context of {chat.MAX_MODEL_LEN:,} tokens minus "
        f"{chat.MAX_TOKENS:,} for the reply)"
    )


def answers_data(ctx: Context) -> dict[str, dict[str, list[dict[str, Any]]]]:
    found: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in hfcommon.SPLIT_NAMES:
        by_release: dict[str, list[dict[str, Any]]] = {}
        for a in ctx.answers[split]:
            record = hfcommon.as_json(a)
            for key in ("split", "ticker", "release_accession"):
                del record[key]
            by_release.setdefault(a.release_accession, []).append(record)
        found[split] = by_release
    return found


def predictions_data(ctx: Context, split: str) -> dict[str, Any]:
    """graded.csv's figures by run and release, as rows of PREDICTION_COLUMNS."""
    runs: dict[str, dict[str, list[list[Any]]]] = {}
    for row in hfcommon.prediction_rows(ctx.runs(split)):
        record = hfcommon.as_json(row)
        runs.setdefault(row.run, {}).setdefault(row.release_accession, []).append(
            [record[c] for c in PREDICTION_COLUMNS]
        )
    return {"columns": list(PREDICTION_COLUMNS), "runs": runs}


# errors.json


def errors_data(ctx: Context) -> dict[str, Any]:
    """Per split: each run's verdict counts and unit slips (graded.csv's), and the
    first graded.csv row of each verdict per run, in the run's order."""
    found = {}
    for split in hfcommon.SPLIT_NAMES:
        counts, slips, examples = {}, {}, []
        for run in ctx.runs(split):
            tally = Counter(row["verdict"] for row in run.graded)
            unknown = set(tally) - set(grade.VERDICTS)
            if unknown:
                raise ExportError(
                    f"runs/{run.name}: unknown verdicts {sorted(unknown)}"
                )
            counts[run.name] = {v: tally[v] for v in grade.VERDICTS}
            slips[run.name] = sum(bool(row["factor"]) for row in run.graded)
            first = {}
            for row in run.graded:
                first.setdefault(row["verdict"], row)
            for verdict in grade.VERDICTS:
                if verdict in first:
                    row = first[verdict]
                    examples.append(
                        [
                            run.name,
                            verdict,
                            row["ticker"],
                            row["release_accession"],
                            row["field"],
                            row["period_type"],
                            row["answer"] or None,
                            row["predicted"] or None,
                            row["factor"] or None,
                            row["evidence"] or None,
                        ]
                    )
        found[split] = {
            "runs": [r.name for r in ctx.runs(split)],
            "verdicts": [
                v for v in grade.VERDICTS if any(c[v] for c in counts.values())
            ],
            "counts": counts,
            "unit_slips": slips,
            "example_columns": list(EXAMPLE_COLUMNS),
            "examples": examples,
        }
    return found


# serving.json


def serving_data(results: Mapping[str, Any], chart: str | None) -> dict[str, Any]:
    """results/serving_results.json's numbers for the Serving tab, formatted as
    secbench.servereport formats them: each variant's best throughput, its accuracy
    with its agreement with the adapter variant and with stage 2's runs, its memory,
    and every load level. Never its server commands, model paths, run paths or error
    messages (they can name the GPU instance's folders)."""
    session, settings = results["session"], results["settings"]
    hourly = session["gpu_hourly_usd"]
    fmt = servereport.fmt
    variants = results["variants"]
    gpus = [g.split(",")[0] for g in session.get("gpus") or []]
    summary, accuracy_rows, memory, levels = [], [], [], []
    for name, v in variants.items():
        startup = v.get("startup") or {}
        best = v.get("best")
        accuracy = v.get("accuracy") or {}
        agreement = v.get("agreement") or {}
        summary.append(
            [
                name,
                v["status"],
                "–"
                if best is None
                else (
                    f"{fmt(best['releases_per_hour'], ',.0f')} at {best['level']} in "
                    "flight"
                ),
                "–" if best is None else fmt(best["latency_p95_s"], ".2f"),
                "–" if best is None else fmt(best["cost_per_1000_usd"], ".3f"),
                "–" if best is None else fmt(best["cost_per_1000_at_1_usd"], ".3f"),
            ]
        )
        accuracy_rows.append(
            [
                name,
                *(
                    cell
                    for split in (VALIDATION, TIME_TEST)
                    for cell in (
                        accuracy_text(accuracy.get(split)),
                        agreement_cell(name, agreement.get(split)),
                        stage2_cell(agreement.get(split)),
                    )
                ),
            ]
        )
        concurrency = startup.get("max_concurrency")
        memory.append(
            [
                name,
                fmt(startup.get("weights_gib"), ".2f"),
                fmt(startup.get("kv_cache_gib"), ".2f"),
                fmt(startup.get("kv_cache_tokens"), ","),
                "–" if concurrency is None else f"{concurrency:.2f}x",
                fmt(startup.get("launch_to_healthy_s"), ".0f"),
            ]
        )
        for x in v.get("levels") or []:
            latency, first = x.get("latency_s") or {}, x.get("ttft_s") or {}
            levels.append(
                [
                    name,
                    str(x["level"]),
                    f"{x['ok']}/{x['requests']}",
                    fmt(x.get("releases_per_hour"), ",.0f"),
                    fmt(latency.get("p50"), ".2f"),
                    fmt(latency.get("p95"), ".2f"),
                    fmt(first.get("p50"), ".2f"),
                    fmt(x.get("decode_tokens_per_s_p50"), ",.0f"),
                    fmt(x.get("cost_per_1000_usd"), ".3f"),
                    fmt(x.get("cost_per_1000_at_1_usd"), ".3f"),
                ]
            )
    reference = results.get("reference_accuracy") or {}
    stage2 = [
        f"{SPLIT_TITLES[s].lower()} {accuracy_text(reference.get(s))}"
        for s in (VALIDATION, TIME_TEST)
        if reference.get(s)
    ]
    load_split = SPLIT_TITLES.get(settings["load_split"], settings["load_split"])
    unfinished = [
        f"{name} {v['status']}"
        for name, v in variants.items()
        if v["status"] != servebench.COMPLETE
    ]
    statuses = f" ({joined(unfinished, 'and')})" if unfinished else ""
    intro = (
        f"The fine-tuned checkpoint ({results['checkpoint']}) served by "
        f"{session['server']} on {joined(gpus, 'and') or 'an unrecorded GPU'} at "
        f"${hourly:g} per GPU-hour, in {len(variants)} variants, one at a "
        f"time{statuses}. Load test: the same {settings['prompts']} "
        f"{load_split.lower()} prompts at each level, after {settings['warmup']} "
        f"warm-up requests, with {joined([str(n) for n in settings['levels']])} "
        "requests in flight; prefix caching "
        f"{'on' if settings['prefix_caching'] else 'off'}, --max-model-len "
        f"{settings['max_model_len']:,}."
    )
    notes = [
        (
            "- **Variants:** adapter is the base model with the LoRA adapter, as stage "
            "2 served it; merged has the adapter merged into the weights; fp8 is "
            "merged, quantized to FP8 by vLLM as it loads; int4-awq is merged, "
            "quantized to 4-bit weights with AWQ."
        ),
        (
            "- **Best throughput:** the load level with the most releases per hour, "
            "with the latency p95 and the cost at that level. Cost per 1,000 releases "
            f"is the GPU's price for the level's makespan, at ${hourly:g}/h and at "
            "$1/h; it leaves out setup, model loading and idle time."
        ),
        (
            "- **Accuracy:** correct over graded figures on validation and on the time "
            "test (descriptive: no variant is chosen on it). Same as adapter: the "
            "predicted figures (null equals null) equal to the adapter variant's, and "
            "the releases with every figure equal; for merged, the merge check. Same "
            "as stage 2: the same against stage 2's run of the checkpoint on that "
            "split; for adapter, the check that serving it reproduces stage 2."
        ),
        (
            "- **Memory:** from vLLM's start-up log: the weights, the KV cache and the "
            "requests of the full context length it can hold at once."
        ),
    ]
    if stage2:
        notes.append(f"- **Stage 2's runs of the checkpoint:** {'; '.join(stage2)}.")
    return {
        "intro": intro,
        "summary_columns": [
            "variant",
            "status",
            "best releases/hour",
            "latency p95 (s)",
            f"$/1,000 at ${hourly:g}/h",
            "$/1,000 at $1/h",
        ],
        "summary": summary,
        "accuracy_columns": [
            "variant",
            "validation accuracy",
            "same as adapter (validation)",
            "same as stage 2 (validation)",
            "time-test accuracy",
            "same as adapter (time test)",
            "same as stage 2 (time test)",
        ],
        "accuracy": accuracy_rows,
        "memory_columns": [
            "variant",
            "weights (GiB)",
            "KV cache (GiB)",
            "KV cache (tokens)",
            f"max concurrency at {settings['max_model_len']:,} tokens",
            "launch to healthy (s)",
        ],
        "memory": memory,
        "level_columns": [
            "variant",
            "in flight",
            "ok",
            "releases/hour",
            "latency p50 (s)",
            "latency p95 (s)",
            "first token p50 (s)",
            "decode tokens/s",
            f"$/1,000 at ${hourly:g}/h",
            "$/1,000 at $1/h",
        ],
        "levels": levels,
        "notes": "\n".join(notes) + "\n",
        "chart": chart,
        "chart_caption": CHART_CAPTIONS["serving"],
    }


def accuracy_text(found: Mapping[str, Any] | None) -> str:
    if not found:
        return "–"
    return servereport.percent(found["correct"], found["graded"])


def agreement_cell(variant: str, found: Mapping[str, Any] | None) -> str:
    """A variant's agreement with the adapter variant, as servereport writes it."""
    if variant == serving.ADAPTER:
        return "(itself)"
    return servereport.agreement_text((found or {}).get("with_adapter"))


def stage2_cell(found: Mapping[str, Any] | None) -> str:
    """A variant's agreement with stage 2's run of the checkpoint on the split."""
    return servereport.agreement_text((found or {}).get("with_stage2"))


def check_no_email(name: str, value: Any) -> None:
    """Stops the build if a JSON file would hold an e-mail address: they are allowed in
    the release texts (press contacts) only, and the grader's evidence quotes release
    lines."""
    for text in json_strings(value):
        if match := EMAIL.search(text):
            masked = match[0][:2] + "…@…"
            raise ExportError(f"{name}: an e-mail address ({masked}) outside a text")


def json_strings(value: Any) -> Iterable[str]:
    """Every string in a JSON value (keys too)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield k
            yield from json_strings(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from json_strings(v)
