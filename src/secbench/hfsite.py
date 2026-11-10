"""The results explorer as a static site: secbench.hfspace writes the Space's data/
(the JSON files, the release texts and the charts), then write_site renders every page
from it into the same folder, so Hugging Face serves the Space as plain files (sdk:
static), with no server to start, sleep or pay for.

Pages: index.html (Leaderboard), explore.html (every release of the three scored
splits, narrowed to a model's misses, a verdict or a company), release/<accession>.html
(one per release: the answer key beside each model's graded figures, and the release
text with each answer highlighted where the grader found it), errors.html (each
model's verdict counts and an example of each verdict), serving.html (only when
data/serving.json exists) and about.html. assets/ holds the style sheet and the script
(space/'s, copied byte for byte by hfspace). The script only shows one split at a time
and hides the rows a filter leaves out; the pages hold every row, so without it they
show everything.

The logic is in plain functions of the loaded Data, as the Gradio app the Space was
until Hugging Face stopped hosting Gradio Spaces on its free hardware had them. Every
number shown is in data/: this module only arranges it. The build's markdown (notes,
About) is rendered with markdown-it-py: CommonMark with tables, raw HTML escaped, and
links off the site opening in a new tab (the Hub shows a Space in a frame).
Deterministic: the same data gives the same bytes.

Library only: secbench.hfspace calls load_data and write_site.
"""

from __future__ import annotations

import html
import itertools
import json
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NamedTuple

from markdown_it import MarkdownIt

# data/'s files (secbench.hfspace writes them).
DATA = "data"
META = "meta.json"
LEADERBOARD = "leaderboard.json"
RUNS = "runs.json"
RELEASES = "releases.json"
ANSWERS = "answers.json"
PREDICTIONS = "predictions"  # predictions/<split>.json
ERRORS = "errors.json"
SERVING = "serving.json"
# The site's own files, beside data/.
ASSETS = "assets"
STYLE = "style.css"
SCRIPT = "explorer.js"
RELEASE_PAGES = "release"  # release/<accession>.html

TABS = {
    "leaderboard": "Leaderboard",
    "explore": "Explore",
    "errors": "Errors",
    "serving": "Serving",
    "about": "About",
}
PAGES = {
    "leaderboard": "index.html",
    "explore": "explore.html",
    "errors": "errors.html",
    "serving": "serving.html",
    "about": "about.html",
}
DEFAULT_TAB = "leaderboard"
SERVING_TAB = "serving"
ANY = ""  # a filter that keeps everything
MISS = "miss"  # the verdict filter for any graded figure that isn't correct

# How a model's figure is coloured: green correct, orange a unit slip (an error the
# grader matched with a power-of-ten factor), red another error, grey not graded.
CORRECT, SLIP, ERROR, UNGRADED = "correct", "unit slip", "error", "not graded"
KIND_COLOURS = {  # background, text
    CORRECT: ("#dcfce7", "#14532d"),
    SLIP: ("#ffedd5", "#7c2d12"),
    ERROR: ("#fee2e2", "#7f1d1d"),
    UNGRADED: ("#f3f4f6", "#374151"),
}
FIELD_COLOURS = {
    "revenue": "#bfdbfe",
    "net_income": "#ddd6fe",
    "eps_diluted": "#fde68a",
}
MARK_TEXT = "#111827"
BORDER = "rgba(128, 128, 128, 0.35)"
EVIDENCE_CHARS = 160  # evidence shown in a cell; the whole of it on hover
EXCERPT_CHARS = 220  # an answer's line, shortened around the figure if longer
TEXT_HEIGHT = "640px"
EXAMPLES_HEIGHT = 700  # pixels: the error examples scroll in a box this tall
SHORT_CELL = 24  # characters: a table cell this short stays on one line
# A cell shown as a number (right-aligned): counts, dollars, seconds, percentages.
NUMBER = re.compile(r"[\u2212+-]?\$?(?:\d[\d,]*)?\.?\d+(?: s|%)?")
# The tab's icon: the Space's emoji, so no file is fetched for it.
FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'"
    "%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%93%8A%3C/text%3E%3C/svg%3E"
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class Data(NamedTuple):
    """data/, loaded once: the JSON files as the build wrote them, predictions as
    {split: {run: {release: [row, ...]}}} with each row a dict, and the releases
    indexed by accession."""

    folder: Path
    meta: dict
    leaderboard: dict
    runs: dict
    releases: dict
    answers: dict
    predictions: dict
    errors: dict
    serving: dict | None
    index: dict  # {split: {accession: release}}
    graded: frozenset  # the verdicts that count in accuracy


def read_json(folder: Path, name: str) -> Any:
    return json.loads((folder / name).read_text(encoding="utf-8"))


def load_data(folder: Path) -> Data:
    meta = read_json(folder, META)
    predictions = {}
    for split in split_names_of(meta):
        table = read_json(folder, f"{PREDICTIONS}/{split}.json")
        columns = table["columns"]
        predictions[split] = {
            run: {
                accession: [dict(zip(columns, row, strict=True)) for row in rows]
                for accession, rows in by_release.items()
            }
            for run, by_release in table["runs"].items()
        }
    releases = read_json(folder, RELEASES)
    return Data(
        folder=folder,
        meta=meta,
        leaderboard=read_json(folder, LEADERBOARD),
        runs=read_json(folder, RUNS),
        releases=releases,
        answers=read_json(folder, ANSWERS),
        predictions=predictions,
        errors=read_json(folder, ERRORS),
        serving=read_json(folder, SERVING) if (folder / SERVING).is_file() else None,
        index={
            split: {r["release_accession"]: r for r in rows}
            for split, rows in releases.items()
        },
        graded=frozenset(v["name"] for v in meta["verdicts"] if v["graded"]),
    )


def split_names_of(meta: Mapping[str, Any]) -> list[str]:
    return [s["name"] for s in meta["splits"]]


def read_release_text(data: Data, split: str, accession: str) -> str:
    """A release's text exactly as written (newline="" keeps every character, so the
    answers' spans line up)."""
    release = data.index[split][accession]
    with (data.folder / release["text_file"]).open(encoding="utf-8", newline="") as f:
        return f.read()


# Tabs


def available_tabs(data: Data) -> list[str]:
    return [t for t in TABS if t != SERVING_TAB or data.serving is not None]


# Names and figures


def split_meta(data: Data, split: str) -> dict:
    return next(s for s in data.meta["splits"] if s["name"] == split)


def split_title(data: Data, split: str) -> str:
    return split_meta(data, split)["title"]


def field_titles(data: Data) -> dict[str, str]:
    return {f["name"]: f["title"] for f in data.meta["fields"]}


def period_titles(data: Data) -> dict[str, str]:
    return {p["name"]: p["title"] for p in data.meta["periods"]}


def run_labels(data: Data, split: str) -> dict[str, str]:
    return {r["run"]: r["label"] for r in data.runs[split]["runs"]}


def figure(value: str | None) -> str:
    """A figure as graded (raw dollars, EPS in dollars per share) with thousands
    separators, exactly; null when there is none."""
    if value is None or value == "":
        return "null"
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    return f"{number:,f}"


def is_miss(data: Data, verdict: str) -> bool:
    return verdict != data.meta["correct"] and verdict in data.graded


def kind(data: Data, verdict: str, factor: str | None) -> str:
    """The colour class of a graded figure."""
    if verdict == data.meta["correct"]:
        return CORRECT
    if verdict not in data.graded:
        return UNGRADED
    return SLIP if factor else ERROR


# Explore: choices and filters


def split_choices(data: Data) -> list[tuple[str, str]]:
    return [(s["title"], s["name"]) for s in data.meta["splits"]]


def run_choices(data: Data, split: str) -> list[tuple[str, str]]:
    return [(r["label"], r["run"]) for r in data.runs[split]["runs"]]


def default_runs(data: Data, split: str) -> list[str]:
    return list(data.runs[split]["default"])


def model_filter_choices(data: Data, split: str) -> list[tuple[str, str]]:
    return [("All models", ANY), *run_choices(data, split)]


def verdict_filter_choices(data: Data, split: str) -> list[tuple[str, str]]:
    """Any verdict, any miss, then each verdict some run of the split has."""
    return [
        ("Any verdict", ANY),
        ("Any miss (graded, not correct)", MISS),
        *((v, v) for v in data.errors[split]["verdicts"]),
    ]


def figure_matches(data: Data, row: Mapping[str, Any], verdict: str) -> bool:
    if verdict == ANY:
        return True
    if verdict == MISS:
        return is_miss(data, row["verdict"])
    return row["verdict"] == verdict


def matching_releases(
    data: Data, split: str, model: str = ANY, verdict: str = ANY
) -> list[dict[str, Any]]:
    """The split's releases (in its order) where the model (or any model) has a
    figure with the verdict; with neither filter, every release, left-out ones too."""
    releases = data.releases[split]
    if model == ANY and verdict == ANY:
        return list(releases)
    runs = [model] if model != ANY else [r["run"] for r in data.runs[split]["runs"]]
    keep = set()
    for run in runs:
        for accession, rows in data.predictions[split].get(run, {}).items():
            if any(figure_matches(data, row, verdict) for row in rows):
                keep.add(accession)
    return [r for r in releases if r["release_accession"] in keep]


def release_order(data: Data, split: str) -> list[dict[str, Any]]:
    """The split's releases as Explore lists them: by ticker, then filing date."""
    return sorted(
        data.releases[split],
        key=lambda r: (r["ticker"], r["filing_date"], r["release_accession"]),
    )


def explore_filters(
    data: Data, split: str, order: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, list[int]]]:
    """For each model choice and each verdict choice, all models with any verdict
    aside: the positions in order (Explore's rows) of the releases that match
    (matching_releases), so the page's script only looks them up."""
    position = {r["release_accession"]: i for i, r in enumerate(order)}
    found: dict[str, dict[str, list[int]]] = {}
    for _, model in model_filter_choices(data, split):
        found[model] = {}
        for _, verdict in verdict_filter_choices(data, split):
            if model == ANY and verdict == ANY:
                continue
            found[model][verdict] = sorted(
                position[r["release_accession"]]
                for r in matching_releases(data, split, model, verdict)
            )
    return found


def release_label(release: Mapping[str, Any]) -> str:
    label = (
        f"{release['filing_date']} · {release['linked_form']}, period ended "
        f"{release['period_end']}"
    )
    return label if release["kept"] else f"{label} · left out"


# Explore: the release


def e(text: Any) -> str:
    return html.escape(str(text), quote=True)


def link(url: str, text: str) -> str:
    return f'<a href="{e(url)}" target="_blank" rel="noopener">{e(text)}</a>'


def release_header(data: Data, split: str, accession: str) -> str:
    r = data.index[split][accession]
    period = "year" if r["linked_form"] == "10-K" else "period"
    status = (
        f"Sent to every model: its prompt is {r['prompt_tokens']:,} tokens"
        if r["kept"]
        else f"Left out for every model: {r['left_out_reason']}"
    )
    lines = [
        (
            f"<h3 style='margin:0 0 4px'>{e(r['company'])} ({e(r['ticker'])})"
            f"<span style='font-weight:normal;font-size:0.8em'> · {e(r['sector'])} · "
            f"{e(split_title(data, split))}</span></h3>"
        ),
        (
            f"<div>Press release filed {e(r['filing_date'])} (8-K "
            f"{e(r['release_accession'])}; input {e(' + '.join(r['exhibits']))}), "
            f"linked to the {e(r['linked_form'])} for the {period} ended "
            f"{e(r['period_end'])} ({e(r['linked_accession'])}).</div>"
        ),
        f"<div>{e(status)}.</div>",
    ]
    if r["preferred_dividends"]:
        lines.append(
            "<div>The linked filing reports nonzero preferred dividends (the "
            "split's preferred.csv).</div>"
        )
    lines.append(
        "<div>On SEC EDGAR: "
        + " · ".join(
            [
                link(r["release_url"], "press release"),
                link(r["filing_url"], "8-K filing index"),
                link(r["linked_url"], f"{r['linked_form']} filing index"),
            ]
        )
        + "</div>"
    )
    return "\n".join(lines)


def slots(data: Data, release: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(field, period) for each figure the prompt asks for: the quarter and the
    year to date, which the grader takes as the full year in a release linked to a
    10-K."""
    second = "full_year" if release["linked_form"] == "10-K" else "year_to_date"
    return [(f, p) for f in field_titles(data) for p in ("quarter", second)]


def answers_by_slot(data: Data, split: str, accession: str) -> dict:
    return {
        (a["field"], a["period_type"]): a
        for a in data.answers[split].get(accession, [])
    }


def predictions_by_slot(data: Data, split: str, run: str, accession: str) -> dict:
    return {
        (p["field"], p["period_type"]): p
        for p in data.predictions[split].get(run, {}).get(accession, [])
    }


def cell(content: str, colours: tuple[str, str] | None = None, title: str = "") -> str:
    style = f"padding:6px 8px;border:1px solid {BORDER};vertical-align:top"
    if colours:
        style += f";background:{colours[0]};color:{colours[1]}"
    hover = f' title="{e(title)}"' if title else ""
    return f'<td style="{style}"{hover}>{content}</td>'


def small(text: str) -> str:
    return f'<div style="font-size:0.85em;opacity:0.85">{text}</div>'


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def answer_cell(answer: Mapping[str, Any] | None) -> str:
    if answer is None:
        return cell(small("no answer"), KIND_COLOURS[UNGRADED])
    parts = [f"<b>{e(figure(answer['value']))}</b>", small(e(answer["tag"]))]
    title = ""
    if answer["derived"]:
        title = (
            f"full year ({answer['full_year_accession']}) minus nine months "
            f"({answer['nine_months_accession']})"
        )
        parts.append(small("derived: full year minus nine months"))
    return cell("".join(parts), title=title)


def printed_cell(answer: Mapping[str, Any] | None) -> str:
    if answer is None:
        return cell("")
    if not answer["in_input"]:
        return cell(small("not printed: not graded"), KIND_COLOURS[UNGRADED])
    places = len(answer["spans"])
    more = small(f"found in {places} places") if places > 1 else ""
    return cell(f"{e(answer['printed_as'])}{more}")


class FigureCell(NamedTuple):
    run: str
    prediction: dict | None  # its graded.csv row; None: it answered null, no answer
    kind: str  # CORRECT, SLIP, ERROR or UNGRADED


class FigureRow(NamedTuple):
    field: str
    period: str
    answer: dict | None  # None: no XBRL answer for this period
    cells: list[FigureCell]


def figure_rows(
    data: Data, split: str, accession: str, runs: Sequence[str]
) -> list[FigureRow]:
    """The figures table's rows: each field and period of the release, with its
    answer and each run's graded figure and colour; none for a left-out release."""
    release = data.index[split][accession]
    if not release["kept"]:
        return []
    labels = run_labels(data, split)
    runs = [run for run in runs if run in labels]
    answers = answers_by_slot(data, split, accession)
    predictions = {
        run: predictions_by_slot(data, split, run, accession) for run in runs
    }
    rows = []
    for field, period in slots(data, release):
        cells = []
        for run in runs:
            p = predictions[run].get((field, period))
            colour = UNGRADED if p is None else kind(data, p["verdict"], p["factor"])
            cells.append(FigureCell(run, p, colour))
        rows.append(FigureRow(field, period, answers.get((field, period)), cells))
    return rows


def prediction_cell(figure_cell: FigureCell) -> str:
    prediction = figure_cell.prediction
    if prediction is None:
        return cell(small("null; no answer"), KIND_COLOURS[UNGRADED])
    verdict, factor = prediction["verdict"], prediction["factor"]
    label = verdict + (f" (×{factor})" if factor else "")
    evidence = prediction["evidence"] or ""
    content = f"<b>{e(figure(prediction['predicted']))}</b>{small(e(label))}"
    if evidence:
        content += small(e(shorten(evidence, EVIDENCE_CHARS)))
    title = f"{verdict}: {evidence}" if evidence else verdict
    return cell(content, KIND_COLOURS[figure_cell.kind], title)


def figures_table(data: Data, split: str, accession: str, runs: Sequence[str]) -> str:
    """Rows: each field and period; columns: the answer key (value and XBRL tag,
    where it is printed), then each run's prediction coloured by its verdict, with
    the grader's evidence in the cell and on hover."""
    release = data.index[split][accession]
    if not release["kept"]:
        return (
            "<p>Left out for every model, so there are no answers or predictions: "
            f"{e(release['left_out_reason'])}.</p>"
        )
    fields, periods = field_titles(data), period_titles(data)
    labels = run_labels(data, split)
    rows = figure_rows(data, split, accession, runs)
    shown = [c.run for c in rows[0].cells] if rows else []
    head = "".join(
        f'<th style="padding:6px 8px;border:1px solid {BORDER};text-align:left">'
        f"{e(name)}</th>"
        for name in ["Figure", "Answer (XBRL)", "Printed as", *map(labels.get, shown)]
    )
    body = "".join(
        "<tr>"
        + cell(f"<b>{e(fields[row.field])}</b>{small(e(periods[row.period]))}")
        + answer_cell(row.answer)
        + printed_cell(row.answer)
        + "".join(prediction_cell(c) for c in row.cells)
        + "</tr>"
        for row in rows
    )
    return (
        legend()
        + '<div style="overflow-x:auto"><table style="border-collapse:collapse;'
        f'width:100%;font-size:13px"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def legend() -> str:
    swatches = " ".join(
        f'<span style="background:{bg};color:{fg};padding:1px 6px;border-radius:4px">'
        f"{e(name)}</span>"
        for name, (bg, fg) in KIND_COLOURS.items()
    )
    return f'<div style="margin:4px 0 8px;font-size:13px">{swatches}</div>'


def line_of(text: str, start: int, end: int) -> tuple[int, str, int, int]:
    """The line holding text[start:end]: its number (from 1), its text, and the
    span's offsets within it."""
    first = text.rfind("\n", 0, start) + 1
    last = text.find("\n", end)
    if last == -1:
        last = len(text)
    return text.count("\n", 0, start) + 1, text[first:last], start - first, end - first


def excerpt(line: str, start: int, end: int, limit: int = EXCERPT_CHARS) -> str:
    """The line as HTML with the figure in bold, shortened around it if long."""
    left, right = 0, len(line)
    if right > limit:
        room = max(limit - (end - start), 0) // 2
        left = max(0, start - room)
        right = min(len(line), end + room)
    return (
        ("…" if left > 0 else "")
        + e(line[left:start])
        + f"<b>{e(line[start:end])}</b>"
        + e(line[end:right])
        + ("…" if right < len(line) else "")
    )


def shown_span(text: str, spans: Sequence[Sequence[int]]) -> tuple[int, int]:
    """Which of an answer's places the list above the text shows: the first on a
    table row (a line starting with "|"), else the first. Every place is highlighted
    in the text either way."""
    for start, end in spans:
        if line_of(text, start, end)[1].lstrip().startswith("|"):
            return start, end
    start, end = spans[0]
    return start, end


def answer_lines(data: Data, split: str, accession: str, text: str) -> str:
    """Above the text: each answer, how the release prints it and a line it is
    found on (a table row's when there is one), so a reader needn't scroll."""
    if not data.index[split][accession]["kept"]:
        return ""
    fields, periods = field_titles(data), period_titles(data)
    items = []
    for a in sorted(
        data.answers[split].get(accession, []),
        key=lambda a: (
            list(fields).index(a["field"]),
            list(periods).index(a["period_type"]),
        ),
    ):
        name = f"{fields[a['field']]}, {periods[a['period_type']].lower()}"
        swatch = (
            f'<span style="background:{FIELD_COLOURS[a["field"]]};color:{MARK_TEXT};'
            f'padding:0 4px;border-radius:3px">{e(name)}</span>'
        )
        if not a["in_input"]:
            found = "not printed in any form the grader searches (not graded)"
        else:
            number, line, start, end = line_of(text, *shown_span(text, a["spans"]))
            others = len(a["spans"]) - 1
            more = f" (and {others} more)" if others else ""
            found = (
                f"printed as <b>{e(a['printed_as'])}</b>, line {number}{more}: "
                f"<code>{excerpt(line, start, end)}</code>"
            )
        items.append(f"<li>{swatch} {e(figure(a['value']))}: {found}</li>")
    if not items:
        return "<p>No answers for this release.</p>"
    return f'<ul style="font-size:13px;margin:4px 0">{"".join(items)}</ul>'


def text_marks(
    data: Data, split: str, accession: str
) -> list[tuple[int, int, str, str]]:
    """Each place the grader finds an answer: (start, end, field, description)."""
    fields, periods = field_titles(data), period_titles(data)
    marks = []
    for a in data.answers[split].get(accession, []):
        what = (
            f"{fields[a['field']]}, {periods[a['period_type']].lower()}: "
            f"{figure(a['value'])} ({a['tag']})"
        )
        marks += [(start, end, a["field"], what) for start, end in a["spans"]]
    return marks


def highlighted_text(text: str, marks: Sequence[tuple[int, int, str, str]]) -> str:
    """The text, escaped, with each mark highlighted in its field's colour (the
    answer named on hover), in a scrollable box; overlapping marks are split into
    pieces that each name every answer they belong to."""
    cuts = sorted({0, len(text), *(m[0] for m in marks), *(m[1] for m in marks)})
    pieces = []
    for a, b in itertools.pairwise(cuts):
        covering = [m for m in marks if m[0] <= a and b <= m[1]]
        chunk = e(text[a:b])
        if covering:
            colour = FIELD_COLOURS.get(covering[0][2], "#e5e7eb")
            title = "; ".join(dict.fromkeys(m[3] for m in covering))
            chunk = (
                f'<mark style="background:{colour};color:{MARK_TEXT};'
                f'padding:0 1px;border-radius:2px" title="{e(title)}">{chunk}</mark>'
            )
        pieces.append(chunk)
    return (
        f'<div style="max-height:{TEXT_HEIGHT};overflow:auto;white-space:pre-wrap;'
        "word-break:break-word;font-family:ui-monospace,SFMono-Regular,Menlo,"
        f"Consolas,monospace;font-size:12px;line-height:1.5;padding:12px;"
        f'border:1px solid {BORDER};border-radius:8px">{"".join(pieces)}</div>'
    )


def field_legend(data: Data) -> str:
    return " ".join(
        f'<span style="background:{FIELD_COLOURS[f]};color:{MARK_TEXT};'
        f'padding:1px 6px;border-radius:4px">{e(title)}</span>'
        for f, title in field_titles(data).items()
    )


class ReleaseView(NamedTuple):
    header: str
    figures: str
    answers: str
    text: str


def release_view(
    data: Data, split: str, accession: str, runs: Sequence[str]
) -> ReleaseView:
    text = read_release_text(data, split, accession)
    marks = text_marks(data, split, accession)
    caption = (
        f"<div style='margin:8px 0 4px;font-size:13px'>The release text, the model's "
        f"input; answers highlighted where the grader found them: "
        f"{field_legend(data)}</div>"
        if marks
        else "<div style='margin:8px 0 4px;font-size:13px'>The release text, the "
        "model's input.</div>"
    )
    return ReleaseView(
        header=release_header(data, split, accession),
        figures=figures_table(data, split, accession, runs),
        answers=answer_lines(data, split, accession, text),
        text=caption + highlighted_text(text, marks),
    )


# Errors


def verdict_table(data: Data, split: str) -> tuple[list[str], list[list[Any]]]:
    """Each run's verdict counts (graded.csv's), the verdicts some run has, and its
    unit slips."""
    errors = data.errors[split]
    labels = run_labels(data, split)
    shown = errors["verdicts"]
    rows = [
        [
            labels[run],
            *(errors["counts"][run][v] for v in shown),
            errors["unit_slips"][run],
        ]
        for run in errors["runs"]
    ]
    return ["model", *shown, "unit slips"], rows


EXAMPLE_HEADER = [
    "model",
    "verdict",
    "ticker",
    "release",
    "field",
    "period",
    "answer",
    "predicted",
    "factor",
    "evidence",
]


def example_rows(
    data: Data, split: str, model: str = ANY, verdict: str = ANY
) -> list[tuple[dict[str, Any], list[Any]]]:
    """One example per run and verdict (the first in the run's graded.csv),
    optionally one run's or one verdict's: each as its errors.json row and as the
    examples table shows it."""
    errors = data.errors[split]
    labels = run_labels(data, split)
    columns = errors["example_columns"]
    found = []
    for values in errors["examples"]:
        row = dict(zip(columns, values, strict=True))
        if model != ANY and row["run"] != model:
            continue
        if verdict != ANY and not figure_matches(data, row, verdict):
            continue
        shown = [
            labels[row["run"]],
            row["verdict"],
            row["ticker"],
            row["release_accession"],
            row["field"],
            row["period_type"],
            "" if row["answer"] is None else figure(row["answer"]),
            figure(row["predicted"]),
            row["factor"] or "",
            row["evidence"] or "",
        ]
        found.append((row, shown))
    return found


def examples_table(
    data: Data, split: str, model: str = ANY, verdict: str = ANY
) -> tuple[list[str], list[list[Any]]]:
    """One example per run and verdict (the first in the run's graded.csv),
    optionally one run's or one verdict's."""
    return list(EXAMPLE_HEADER), [
        shown for _, shown in example_rows(data, split, model, verdict)
    ]


# Tables


def is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, int | float) or bool(NUMBER.fullmatch(str(value)))


def keep_together(text: str) -> str:
    """text with no-break spaces inside "n/m (p%)" and "[a, b]", so a long cell
    such as "657/662 (99.24%) [98.05, 100.00]" wraps only between the two."""
    text = re.sub(r"(\d) \(", "\\1\u00a0(", text)
    return re.sub(r"\[[^\]]*\]", lambda m: m[0].replace(" ", "\u00a0"), text)


def attributes(values: Mapping[str, str] | None) -> str:
    """HTML attributes, escaped, in the given order."""
    return "".join(f' {name}="{e(value)}"' for name, value in (values or {}).items())


def html_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_height: int | None = None,
    row_attributes: Sequence[Mapping[str, str]] | None = None,
) -> str:
    """A table as HTML that fits the page: headers break only at spaces and
    underscores; numbers, cells without a space (accession numbers, verdicts) and
    cells up to SHORT_CELL characters stay on one line, while longer ones may wrap,
    never inside "n/m (p%)" or an interval "[a, b]" (keep_together); columns of
    numbers are right-aligned; with max_height the table scrolls in a box that tall,
    its header kept in view; row_attributes, one mapping per row, are each row's
    attributes (the script filters rows by them)."""
    if not rows:
        return "<p>Nothing to show.</p>"
    if row_attributes is not None and len(row_attributes) != len(rows):
        raise ValueError("row_attributes needs one mapping per row")
    numeric = [
        all(is_number(row[i]) for row in rows if row[i] not in ("", None))
        for i in range(len(columns))
    ]
    edge = f"padding:4px 8px;border:1px solid {BORDER}"
    head = "".join(
        f'<th style="{edge};text-align:{"right" if right else "left"};'
        "vertical-align:bottom;position:sticky;top:0;z-index:1;"
        f'background:var(--background-fill-primary,#fff)">'
        f"{e(name).replace('_', '_<wbr>')}</th>"
        for name, right in zip(columns, numeric, strict=True)
    )
    body = []
    for i, row in enumerate(rows):
        cells = []
        for value, right in zip(row, numeric, strict=True):
            text = "" if value is None else str(value)
            style = f"{edge};vertical-align:top"
            if right:
                style += ";text-align:right"
            if right or len(text) <= SHORT_CELL or not any(c.isspace() for c in text):
                style += ";white-space:nowrap"
            else:
                style += ";white-space:normal"
                text = keep_together(text)
            cells.append(f'<td style="{style}">{e(text)}</td>')
        extra = attributes(row_attributes[i]) if row_attributes is not None else ""
        body.append(f"<tr{extra}>{''.join(cells)}</tr>")
    box = "overflow-x:auto"
    if max_height:
        box += f";max-height:{max_height}px;overflow-y:auto"
    return (
        f'<div style="{box}"><table style="border-collapse:collapse;width:100%;'
        f'font-size:13px"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


# Leaderboard and serving


def validation_board(data: Data) -> tuple[list[str], list[list[str]]]:
    board = data.leaderboard["validation"]
    return board["columns"], board["rows"]


def test_boards(data: Data) -> list[dict[str, Any]]:
    return data.leaderboard["test"]["splits"]


def serving_table(data: Data, name: str) -> tuple[list[str], list[list[str]]]:
    """One of the Serving tab's tables: summary, accuracy, memory or level."""
    serving = data.serving or {}
    rows = serving.get("levels" if name == "level" else name, [])
    return serving.get(f"{name}_columns", []), rows


def verdict_glossary(data: Data) -> str:
    return "\n".join(
        ["| verdict | meaning |", "| --- | --- |"]
        + [f"| {v['name']} | {v['meaning']} |" for v in data.meta["verdicts"]]
    )


# Markdown


def _off_site_link(self, tokens, idx, options, env) -> str:
    """markdown-it's link_open with a link off the site opening in a new tab."""
    token = tokens[idx]
    if str(token.attrGet("href") or "").startswith(("https://", "http://")):
        token.attrSet("target", "_blank")
        token.attrSet("rel", "noopener")
    return self.renderToken(tokens, idx, options, env)


MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable("table")
MARKDOWN.add_render_rule("link_open", _off_site_link)


def markdown(text: str) -> str:
    """Markdown as HTML, in a block the style sheet gives tables and code."""
    return f'<div class="md">\n{MARKDOWN.render(text)}</div>'


def markdown_inline(text: str) -> str:
    """One line of markdown as HTML, with no paragraph around it."""
    return MARKDOWN.renderInline(text)


# Pages


def png_size(path: Path) -> tuple[int, int]:
    """A PNG's width and height, from its header, so the page keeps the chart's room
    before it loads."""
    with path.open("rb") as f:
        head = f.read(24)
    if head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        raise ValueError(f"{path} isn't a PNG")
    width, height = struct.unpack(">II", head[16:24])
    return width, height


def chart(data: Data, path: str, caption: str) -> str:
    """A chart from data/ (path relative to it), as a figure with its caption."""
    width, height = png_size(data.folder / path)
    src = e(f"{DATA}/{path}")
    return (
        f'<figure><a href="{src}" target="_blank" rel="noopener">'
        f'<img src="{src}" width="{width}" height="{height}" alt="{e(caption)}" '
        f'loading="lazy"></a><figcaption>{markdown_inline(caption)}</figcaption>'
        "</figure>"
    )


def page(data: Data, tab: str, title: str, body: str, root: str = "") -> str:
    """A whole page: its head, the site's header with the tabs (tab is the current
    one), the body and the footer; root is the way back to the site's top folder
    ("" or "../")."""
    meta, links = data.meta, data.meta["links"]
    tabs = "".join(
        f'<a href="{root}{PAGES[name]}"'
        + (' aria-current="page"' if name == tab else "")
        + f">{e(TABS[name])}</a>"
        for name in available_tabs(data)
    )
    elsewhere = " · ".join(
        [
            link(links["github"], "GitHub"),
            link(links["dataset"], "dataset"),
            link(links["model"], "model"),
        ]
    )
    footer = (
        "The release texts are the companies' press releases as filed with the SEC on "
        "EDGAR. For the API models (OpenAI and Anthropic), only the predicted figures "
        "and the grader's verdicts are shown, never a model's reply. Built from the "
        f"results of {link(links['github'], 'sec-earnings-bench')} (MIT)."
    )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{e(title)}</title>",
            f'<meta name="description" content="{e(meta["subtitle"])}">',
            f'<link rel="icon" href="{FAVICON}">',
            f'<link rel="stylesheet" href="{root}{ASSETS}/{STYLE}">',
            "<script>document.documentElement.classList.add('js')</script>",
            f'<script src="{root}{ASSETS}/{SCRIPT}" defer></script>',
            "</head>",
            "<body>",
            '<header class="site">',
            f'<h1><a href="{root}{PAGES[DEFAULT_TAB]}">{e(meta["title"])}</a></h1>',
            f'<p class="subtitle">{e(meta["subtitle"])} {elsewhere}</p>',
            f'<nav class="tabs" aria-label="Pages">{tabs}</nav>',
            "</header>",
            "<main>",
            body,
            "</main>",
            f"<footer>{footer}</footer>",
            "</body>",
            "</html>",
            "",
        ]
    )


def page_title(data: Data, name: str) -> str:
    return f"{name} · {data.meta['title']}"


def leaderboard_page(data: Data) -> str:
    board, links = data.leaderboard, data.meta["links"]
    validation, test = board["validation"], board["test"]
    parts = [
        markdown(board["intro"]),
        f"<h2>{e(validation['title'])}</h2>",
        html_table(*validation_board(data)),
        markdown(validation["notes"]),
        chart(data, validation["chart"], validation["chart_caption"]),
        f"<h2>{e(test['title'])}</h2>",
        markdown(test["primary"]),
        markdown(test["summary"]),
    ]
    for split_board in test_boards(data):
        parts += [
            f"<h3>{e(split_board['title'])}</h3>",
            html_table(split_board["columns"], split_board["rows"]),
            markdown(split_board["differences"]),
        ]
    parts += [
        markdown(test["notes"]),
        chart(data, test["chart"], test["chart_caption"]),
        markdown(
            f"Every number here is from [results/comparison.md]"
            f"({links['comparison']}) and [results/test_results.md]"
            f"({links['test_results']})."
        ),
    ]
    return page(
        data, "leaderboard", page_title(data, TABS["leaderboard"]), _join(parts)
    )


def split_tabs(data: Data) -> str:
    """The buttons that show one split at a time (the script's; hidden without it)."""
    default = data.meta["default_split"]
    buttons = "".join(
        f'<button type="button" role="tab" data-split="{e(name)}" aria-selected='
        f'"{"true" if name == default else "false"}">{e(title)}</button>'
        for title, name in split_choices(data)
    )
    return (
        f'<div class="split-tabs" role="tablist" aria-label="Split" '
        f'data-default="{e(default)}">{buttons}</div>'
    )


def split_section(data: Data, split: str, body: str) -> str:
    """One split's part of a page; the script shows one at a time (the default
    split's first), and without it every split shows."""
    current = " current" if split == data.meta["default_split"] else ""
    return (
        f'<section class="split{current}" data-split="{e(split)}">\n'
        f"<h2>{e(split_title(data, split))}</h2>\n{body}\n</section>"
    )


def options(choices: Iterable[tuple[str, str]]) -> str:
    return "".join(
        f'<option value="{e(value)}">{e(label)}</option>' for label, value in choices
    )


def script_json(value: Any) -> str:
    """JSON for a <script> element: compact, with "<" escaped so no text in it can
    end the element."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).replace(
        "<", "\\u003c"
    )


def release_path(accession: str) -> str:
    return f"{RELEASE_PAGES}/{accession}.html"


def explore_page(data: Data) -> str:
    intro = (
        "<p>Pick a split, then narrow its releases to one model's misses, to a "
        "verdict or to a company. Each release opens on its own page: the answer key "
        "beside each model's figures (hover a cell for the grader's evidence), and "
        "the release text with each answer highlighted where the grader found it.</p>"
    )
    sections = [
        split_section(data, s, explore_split(data, s))
        for s in split_names_of(data.meta)
    ]
    return page(
        data,
        "explore",
        page_title(data, TABS["explore"]),
        _join([intro, split_tabs(data), *sections]),
    )


def explore_split(data: Data, split: str) -> str:
    """A split's releases as a list, with the filters and what they match."""
    meta = split_meta(data, split)
    order = release_order(data, split)
    filters = (
        f'<div class="filters js-only" data-miss="{e(MISS)}">'
        "<label>Releases where"
        f'<select data-filter="model">{options(model_filter_choices(data, split))}'
        "</select></label>"
        "<label>has a figure with"
        f'<select data-filter="verdict">{options(verdict_filter_choices(data, split))}'
        "</select></label>"
        '<label>Company<input type="search" data-filter="company" '
        'placeholder="Ticker or name" autocomplete="off"></label>'
        "</div>"
    )
    rows = []
    for i, r in enumerate(order):
        tokens = f"{r['prompt_tokens']:,} tokens"
        prompt = tokens if r["kept"] else f"{tokens} · left out"
        title = "" if r["kept"] else f' title="{e(r["left_out_reason"])}"'
        search = f"{r['ticker']} {r['company']}".lower()
        rows.append(
            f'<tr data-i="{i}" data-ticker="{e(r["ticker"])}" data-search="{e(search)}">'
            f'<td><a href="{release_path(r["release_accession"])}">'
            f"{e(r['ticker'])} · {e(r['company'])}</a></td>"
            f"<td>{e(r['sector'])}</td>"
            f'<td class="nowrap">{e(r["filing_date"])}</td>'
            f"<td>{e(r['linked_form'])}, period ended {e(r['period_end'])}</td>"
            f'<td class="num"{title}>{e(prompt)}</td>'
            "</tr>"
        )
    table = (
        '<div class="table-box"><table class="list"><thead><tr><th>Company</th>'
        "<th>Sector</th><th>Filed</th><th>Linked filing</th><th>Prompt</th></tr>"
        f"</thead><tbody>{''.join(rows)}</tbody></table></div>"
    )
    return _join(
        [
            f'<p class="split-note">{e(meta["description"])}</p>',
            filters,
            f'<p class="status" data-default="{e(meta["status"])}">{e(meta["status"])}</p>',
            table,
            (
                '<script type="application/json" class="filter-data">'
                f"{script_json(explore_filters(data, split, order))}</script>"
            ),
        ]
    )


def release_page(
    data: Data, split: str, order: Sequence[Mapping[str, Any]], i: int
) -> str:
    """The page of order[i], a release of split, with the way back to its split and
    to the releases before and after it in Explore's list."""
    release = order[i]
    accession = release["release_accession"]
    shown = default_runs(data, split)
    view = release_view(data, split, accession, shown)
    crumbs = [
        (
            f'<a href="../{PAGES["explore"]}#split={e(split)}">'
            f"← {e(split_title(data, split))} releases</a>"
        )
    ]
    if i > 0:
        before = order[i - 1]
        crumbs.append(
            f'<a href="../{release_path(before["release_accession"])}" rel="prev">'
            f"‹ {e(before['ticker'])} {e(before['filing_date'])}</a>"
        )
    if i + 1 < len(order):
        after = order[i + 1]
        crumbs.append(
            f'<a href="../{release_path(after["release_accession"])}" rel="next">'
            f"{e(after['ticker'])} {e(after['filing_date'])} ›</a>"
        )
    parts = [
        f'<p class="crumbs">{"".join(f"<span>{c}</span>" for c in crumbs)}</p>',
        view.header,
    ]
    parts.append("<h3>Figures</h3>")
    parts.append(view.figures)
    every = [run for _, run in run_choices(data, split)]
    if release["kept"] and len(every) > len(shown):
        parts.append(
            f"<details><summary>Every model on this split ({len(every)})</summary>"
            f"{figures_table(data, split, accession, every)}</details>"
        )
    if view.answers:
        parts += ["<h3>Where the release prints each answer</h3>", view.answers]
    parts += ["<h3>The release</h3>", view.text]
    title = f"{release['ticker']} {release['filing_date']} · {data.meta['title']}"
    return page(data, "explore", title, _join(parts), root="../")


def errors_page(data: Data) -> str:
    intro = markdown(
        "Each graded figure gets the first verdict that explains it (see "
        f"[About]({PAGES['about']})). Unit slips count the figures matched with a "
        "power-of-ten factor; not_in_input and no_answer figures aren't graded. The "
        "examples are the first figure with each verdict in each run, in the run's "
        "order."
    )
    glossary = (
        "<details><summary>What each verdict means</summary>"
        f"{markdown(verdict_glossary(data))}</details>"
    )
    sections = [
        split_section(data, s, errors_split(data, s)) for s in split_names_of(data.meta)
    ]
    return page(
        data,
        "errors",
        page_title(data, TABS["errors"]),
        _join([intro, glossary, split_tabs(data), *sections]),
    )


def errors_split(data: Data, split: str) -> str:
    """A split's verdict counts, and its examples with their filters."""
    found = example_rows(data, split)
    attrs = [
        {
            "data-run": row["run"],
            "data-verdict": row["verdict"],
            "data-miss": "1" if is_miss(data, row["verdict"]) else "0",
        }
        for row, _ in found
    ]
    filters = (
        f'<div class="filters js-only" data-miss="{e(MISS)}">'
        "<label>Model"
        f'<select data-filter="example-model">{options(model_filter_choices(data, split))}'
        "</select></label>"
        "<label>Verdict"
        '<select data-filter="example-verdict">'
        f"{options(verdict_filter_choices(data, split))}</select></label>"
        "</div>"
    )
    return _join(
        [
            "<h3>Verdicts by model</h3>",
            html_table(*verdict_table(data, split)),
            "<h3>An example of each verdict</h3>",
            filters,
            html_table(
                EXAMPLE_HEADER,
                [shown for _, shown in found],
                max_height=EXAMPLES_HEIGHT,
                row_attributes=attrs,
            ),
            '<p class="empty" hidden>No example matches these filters.</p>',
        ]
    )


def serving_page(data: Data) -> str:
    serving = data.serving or {}
    parts = [
        markdown(serving["intro"]),
        "<h3>Best throughput</h3>",
        html_table(*serving_table(data, "summary")),
        "<h3>Accuracy</h3>",
        html_table(*serving_table(data, "accuracy")),
        "<h3>Memory</h3>",
        html_table(*serving_table(data, "memory")),
        "<h3>Every load level</h3>",
        html_table(*serving_table(data, "level")),
        markdown(serving["notes"]),
    ]
    if serving.get("chart"):
        parts.append(chart(data, serving["chart"], serving["chart_caption"]))
    return page(data, SERVING_TAB, page_title(data, TABS[SERVING_TAB]), _join(parts))


def about_page(data: Data) -> str:
    return page(
        data, "about", page_title(data, TABS["about"]), markdown(data.meta["about"])
    )


def _join(parts: Iterable[str]) -> str:
    return "\n".join(part for part in parts if part)


def site_pages(data: Data) -> dict[str, str]:
    """Every page, by its path in the site."""
    pages = {
        PAGES["leaderboard"]: leaderboard_page(data),
        PAGES["explore"]: explore_page(data),
        PAGES["errors"]: errors_page(data),
    }
    if data.serving is not None:
        pages[PAGES[SERVING_TAB]] = serving_page(data)
    pages[PAGES["about"]] = about_page(data)
    for split in split_names_of(data.meta):
        order = release_order(data, split)
        for i, release in enumerate(order):
            path = release_path(release["release_accession"])
            if path in pages:
                raise ValueError(
                    f"release {release['release_accession']} is in two splits"
                )
            pages[path] = release_page(data, split, order, i)
    return pages


def write_site(data: Data, out: Path) -> list[Path]:
    """Writes every page under out (the Space's folder, which holds data/ already);
    returns their paths."""
    written = []
    for name, text in site_pages(data).items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)
    return written
