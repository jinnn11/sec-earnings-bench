"""The results explorer of sec-earnings-bench
(https://github.com/jinnn11/sec-earnings-bench): how accurately LLMs extract GAAP
revenue, net income and diluted EPS from US earnings press releases, graded
automatically against XBRL in the company's later 10-Q or 10-K.

It reads data/ beside this file, which the repository's secbench.hfspace builds from
its results: every number shown is there, and this file only arranges it. The logic
is in plain functions of the loaded Data, which the repository's tests run without
gradio; build_demo imports gradio and wires them to five tabs: Leaderboard, Explore,
Errors, Serving (only when data/serving.json exists) and About. A URL query opens a
tab (?tab=errors) or a release in Explore (?tab=explore&release=<accession>, with
&split=<split> if wanted).

Run it with python app.py, as Hugging Face Spaces does. It needs gradio (and the
pandas gradio brings) and the standard library only. There is no
`from __future__ import annotations` here: gradio finds the request parameter of the
page-load handler by its annotation at run time.
"""

import html
import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NamedTuple

DATA_DIR = Path(__file__).resolve().parent / "data"
# data/'s files (secbench.hfspace writes them under the same names).
META = "meta.json"
LEADERBOARD = "leaderboard.json"
RUNS = "runs.json"
RELEASES = "releases.json"
ANSWERS = "answers.json"
PREDICTIONS = "predictions"  # predictions/<split>.json
ERRORS = "errors.json"
SERVING = "serving.json"

TABS = {
    "leaderboard": "Leaderboard",
    "explore": "Explore",
    "errors": "Errors",
    "serving": "Serving",
    "about": "About",
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


def load_data(folder: Path = DATA_DIR) -> Data:
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


def tab_from_query(query: Mapping[str, str], tabs: Sequence[str]) -> str:
    """The tab a URL's ?tab= names, if the app has it; otherwise the first tab."""
    tab = str(query.get("tab", "")).strip().lower()
    return tab if tab in tabs else DEFAULT_TAB


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


def company_choices(releases: Iterable[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Each company with a release among releases, by ticker."""
    names = {r["ticker"]: r["company"] for r in releases}
    return [(f"{t} · {names[t]}", t) for t in sorted(names)]


def release_label(release: Mapping[str, Any]) -> str:
    label = (
        f"{release['filing_date']} · {release['linked_form']}, period ended "
        f"{release['period_end']}"
    )
    return label if release["kept"] else f"{label} · left out"


def release_choices(
    releases: Iterable[Mapping[str, Any]], ticker: str | None
) -> list[tuple[str, str]]:
    """The company's releases among releases, by filing date."""
    mine = sorted(
        (r for r in releases if r["ticker"] == ticker),
        key=lambda r: (r["filing_date"], r["release_accession"]),
    )
    return [(release_label(r), r["release_accession"]) for r in mine]


class ExploreState(NamedTuple):
    """Everything Explore shows for one choice of split, filters, company and
    release, with the choices each dropdown offers."""

    split: str
    run_choices: list[tuple[str, str]]
    runs: list[str]  # the models shown in the figures table
    model_choices: list[tuple[str, str]]
    model: str
    verdict_choices: list[tuple[str, str]]
    verdict: str
    company_choices: list[tuple[str, str]]
    ticker: str | None
    release_choices: list[tuple[str, str]]
    accession: str | None
    status: str  # how many releases match the filters


def find_split(data: Data, accession: str | None) -> str | None:
    for split in split_names_of(data.meta):
        if accession in data.index[split]:
            return split
    return None


def explore_state(
    data: Data,
    split: str | None = None,
    model: str = ANY,
    verdict: str = ANY,
    ticker: str | None = None,
    accession: str | None = None,
    runs: Sequence[str] | None = None,
) -> ExploreState:
    """A consistent state from what the user chose: unknown values fall back to the
    defaults; a release asked for (a link) clears filters that would hide it."""
    if accession and accession not in data.index.get(split or "", {}):
        split = find_split(data, accession) or split
    if split not in data.releases:
        split = data.meta["default_split"]
    run_names = [r["run"] for r in data.runs[split]["runs"]]
    model = model if model in run_names else ANY
    verdict_values = [v for _, v in verdict_filter_choices(data, split)]
    verdict = verdict if verdict in verdict_values else ANY
    matched = matching_releases(data, split, model, verdict)
    if accession in data.index[split] and accession not in {
        r["release_accession"] for r in matched
    }:
        model, verdict = ANY, ANY
        matched = matching_releases(data, split)
    if accession in data.index[split]:
        ticker = data.index[split][accession]["ticker"]
    companies = company_choices(matched)
    tickers = [t for _, t in companies]
    if ticker not in tickers:
        ticker = tickers[0] if tickers else None
    releases = release_choices(matched, ticker)
    accessions = [a for _, a in releases]
    if accession not in accessions:
        accession = accessions[0] if accessions else None
    shown = [r for r in (runs or []) if r in run_names] or default_runs(data, split)
    total = len(data.releases[split])
    if model == ANY and verdict == ANY:
        status = split_meta(data, split)["status"]
    else:
        status = (
            f"{len(matched)} of the {total} releases match, from {len(companies)} "
            f"{'company' if len(companies) == 1 else 'companies'}."
        )
    return ExploreState(
        split=split,
        run_choices=run_choices(data, split),
        runs=shown,
        model_choices=model_filter_choices(data, split),
        model=model,
        verdict_choices=verdict_filter_choices(data, split),
        verdict=verdict,
        company_choices=companies,
        ticker=ticker,
        release_choices=releases,
        accession=accession,
        status=status,
    )


def verdict_after_model_change(model: str, verdict: str) -> str:
    """Picking a model with no verdict chosen shows that model's misses."""
    return MISS if model != ANY and verdict == ANY else verdict


# Explore: the release


def e(text: Any) -> str:
    return html.escape(str(text), quote=True)


def link(url: str, text: str) -> str:
    return f'<a href="{e(url)}" target="_blank" rel="noopener">{e(text)}</a>'


def release_header(data: Data, split: str, accession: str) -> str:
    r = data.index[split][accession]
    period = "year" if r["linked_form"] == "10-K" else "period"
    status = (
        f"kept: {r['prompt_tokens']:,} prompt tokens"
        if r["kept"]
        else f"left out for every model: {r['left_out_reason']}"
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
        f"<div>Status: {e(status)}.</div>",
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


def answer_lines(data: Data, split: str, accession: str, text: str) -> str:
    """Above the text: each answer, how the release prints it and the line it is
    first found on, so a reader needn't scroll."""
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
            number, line, start, end = line_of(text, *a["spans"][0])
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
    data: Data, split: str, accession: str | None, runs: Sequence[str]
) -> ReleaseView:
    if accession is None:
        return ReleaseView("<p>No release matches these filters.</p>", "", "", "")
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


def examples_table(
    data: Data, split: str, model: str = ANY, verdict: str = ANY
) -> tuple[list[str], list[list[Any]]]:
    """One example per run and verdict (the first in the run's graded.csv),
    optionally one run's or one verdict's."""
    errors = data.errors[split]
    labels = run_labels(data, split)
    columns = errors["example_columns"]
    rows = []
    for values in errors["examples"]:
        row = dict(zip(columns, values, strict=True))
        if model != ANY and row["run"] != model:
            continue
        if verdict != ANY and not figure_matches(data, row, verdict):
            continue
        rows.append(
            [
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
        )
    header = [
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
    return header, rows


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


# The app


def build_demo(data: Data):
    """The Gradio app over data (gradio is imported here only)."""
    import gradio as gr
    import pandas as pd

    tabs = available_tabs(data)
    meta, board = data.meta, data.leaderboard
    links = meta["links"]

    def frame(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> Any:
        return pd.DataFrame(list(rows), columns=list(columns))

    def table(columns, rows, **kwargs):
        return gr.Dataframe(
            value=frame(columns, rows), interactive=False, wrap=True, **kwargs
        )

    def chart(path: str, caption: str) -> None:
        gr.Image(
            value=str(data.folder / path),
            show_label=False,
            interactive=False,
            buttons=["download", "fullscreen"],
        )
        gr.Markdown(f"*{caption}*")

    def explore_updates(state: ExploreState) -> list[Any]:
        view = release_view(data, state.split, state.accession, state.runs)
        return [
            gr.Radio(value=state.split),
            gr.Dropdown(choices=state.run_choices, value=state.runs, multiselect=True),
            gr.Dropdown(choices=state.model_choices, value=state.model),
            gr.Dropdown(choices=state.verdict_choices, value=state.verdict),
            gr.Dropdown(choices=state.company_choices, value=state.ticker),
            gr.Dropdown(choices=state.release_choices, value=state.accession),
            state.status,
            view.header,
            view.figures,
            view.answers,
            view.text,
        ]

    first = explore_state(data)
    first_view = release_view(data, first.split, first.accession, first.runs)
    title = f"# {meta['title']}\n\n{meta['subtitle']} " + " · ".join(
        [
            f"[GitHub]({links['github']})",
            f"[dataset]({links['dataset']})",
            f"[model]({links['model']})",
        ]
    )
    with gr.Blocks(title=meta["title"], analytics_enabled=False) as demo:
        gr.Markdown(title)
        with gr.Tabs(selected=DEFAULT_TAB) as tab_set:
            with gr.Tab(TABS["leaderboard"], id="leaderboard"):
                gr.Markdown(board["intro"])
                validation = board["validation"]
                gr.Markdown(f"## {validation['title']}")
                table(*validation_board(data), max_height=900)
                gr.Markdown(validation["notes"])
                chart(validation["chart"], validation["chart_caption"])
                test = board["test"]
                gr.Markdown(f"## {test['title']}\n\n{test['primary']}")
                gr.Markdown(test["summary"])
                for split_board in test_boards(data):
                    gr.Markdown(f"### {split_board['title']}")
                    table(split_board["columns"], split_board["rows"])
                    gr.Markdown(split_board["differences"])
                gr.Markdown(test["notes"])
                chart(test["chart"], test["chart_caption"])
                gr.Markdown(
                    f"Every number here is from [results/comparison.md]"
                    f"({links['comparison']}) and [results/test_results.md]"
                    f"({links['test_results']})."
                )
            with gr.Tab(TABS["explore"], id="explore"):
                gr.Markdown(
                    "Pick a split, a company and a release, or narrow the choices to "
                    "one model's misses or to one verdict. The table compares each "
                    "model's figures with the answer key (hover a cell for the "
                    "grader's evidence); below it, the release text with each answer "
                    "highlighted where the grader found it."
                )
                with gr.Row():
                    split = gr.Radio(
                        choices=split_choices(data),
                        value=first.split,
                        label="Split",
                    )
                    runs = gr.Dropdown(
                        choices=first.run_choices,
                        value=first.runs,
                        multiselect=True,
                        label="Models in the table",
                    )
                with gr.Row():
                    model = gr.Dropdown(
                        choices=first.model_choices,
                        value=first.model,
                        label="Releases where this model…",
                    )
                    verdict = gr.Dropdown(
                        choices=first.verdict_choices,
                        value=first.verdict,
                        label="…has a figure with this verdict",
                    )
                with gr.Row():
                    company = gr.Dropdown(
                        choices=first.company_choices,
                        value=first.ticker,
                        label="Company",
                    )
                    release = gr.Dropdown(
                        choices=first.release_choices,
                        value=first.accession,
                        label="Release",
                    )
                status = gr.Markdown(first.status)
                header = gr.HTML(first_view.header, js_on_load=None)
                figures = gr.HTML(first_view.figures, js_on_load=None)
                answer_list = gr.HTML(first_view.answers, js_on_load=None)
                text = gr.HTML(first_view.text, js_on_load=None)
            with gr.Tab(TABS["errors"], id="errors"):
                gr.Markdown(
                    "Each graded figure gets the first verdict that explains it (see "
                    "About). Unit slips count the figures matched with a power-of-ten "
                    "factor; not_in_input and no_answer figures aren't graded. The "
                    "examples are the first figure with each verdict in each run, in "
                    "the run's order."
                )
                error_split = gr.Radio(
                    choices=split_choices(data),
                    value=meta["default_split"],
                    label="Split",
                )
                counts = table(*verdict_table(data, meta["default_split"]))
                with gr.Accordion("What each verdict means", open=False):
                    gr.Markdown(verdict_glossary(data))
                with gr.Row():
                    example_model = gr.Dropdown(
                        choices=model_filter_choices(data, meta["default_split"]),
                        value=ANY,
                        label="Model",
                    )
                    example_verdict = gr.Dropdown(
                        choices=verdict_filter_choices(data, meta["default_split"]),
                        value=ANY,
                        label="Verdict",
                    )
                examples = table(
                    *examples_table(data, meta["default_split"]), max_height=700
                )
            if SERVING_TAB in tabs:
                serving = data.serving
                with gr.Tab(TABS["serving"], id=SERVING_TAB):
                    gr.Markdown(serving["intro"])
                    gr.Markdown("### Best throughput")
                    table(*serving_table(data, "summary"))
                    gr.Markdown("### Accuracy")
                    table(*serving_table(data, "accuracy"))
                    gr.Markdown("### Memory")
                    table(*serving_table(data, "memory"))
                    gr.Markdown("### Every load level")
                    table(*serving_table(data, "level"))
                    gr.Markdown(serving["notes"])
                    if serving.get("chart"):
                        chart(serving["chart"], serving["chart_caption"])
            with gr.Tab(TABS["about"], id="about"):
                gr.Markdown(meta["about"])

        explore_outputs = [
            split,
            runs,
            model,
            verdict,
            company,
            release,
            status,
            header,
            figures,
            answer_list,
            text,
        ]

        def on_split(chosen):
            return explore_updates(explore_state(data, chosen))

        def on_model(chosen_split, chosen_model, chosen_verdict, shown):
            chosen_verdict = verdict_after_model_change(chosen_model, chosen_verdict)
            state = explore_state(
                data, chosen_split, chosen_model, chosen_verdict, runs=shown
            )
            return explore_updates(state)

        def on_filter(chosen_split, chosen_model, chosen_verdict, shown):
            state = explore_state(
                data, chosen_split, chosen_model, chosen_verdict, runs=shown
            )
            return explore_updates(state)

        def on_company(chosen_split, chosen_model, chosen_verdict, ticker, shown):
            state = explore_state(
                data, chosen_split, chosen_model, chosen_verdict, ticker, runs=shown
            )
            return explore_updates(state)

        def on_release(chosen_split, accession, shown):
            if accession not in data.index.get(chosen_split, {}):
                return gr.skip()
            view = release_view(data, chosen_split, accession, shown or [])
            return [view.header, view.figures, view.answers, view.text]

        def on_runs(chosen_split, accession, shown):
            if accession not in data.index.get(chosen_split, {}):
                return gr.skip()
            return figures_table(data, chosen_split, accession, shown or [])

        split.input(on_split, inputs=split, outputs=explore_outputs)
        model.input(
            on_model, inputs=[split, model, verdict, runs], outputs=explore_outputs
        )
        verdict.input(
            on_filter, inputs=[split, model, verdict, runs], outputs=explore_outputs
        )
        company.input(
            on_company,
            inputs=[split, model, verdict, company, runs],
            outputs=explore_outputs,
        )
        release.input(
            on_release,
            inputs=[split, release, runs],
            outputs=[header, figures, answer_list, text],
        )
        runs.input(on_runs, inputs=[split, release, runs], outputs=figures)

        def on_error_split(chosen):
            return [
                frame(*verdict_table(data, chosen)),
                gr.Dropdown(choices=model_filter_choices(data, chosen), value=ANY),
                gr.Dropdown(choices=verdict_filter_choices(data, chosen), value=ANY),
                frame(*examples_table(data, chosen)),
            ]

        def on_example_filter(chosen_split, chosen_model, chosen_verdict):
            return frame(
                *examples_table(data, chosen_split, chosen_model, chosen_verdict)
            )

        error_split.input(
            on_error_split,
            inputs=error_split,
            outputs=[counts, example_model, example_verdict, examples],
        )
        for control in (example_model, example_verdict):
            control.input(
                on_example_filter,
                inputs=[error_split, example_model, example_verdict],
                outputs=examples,
            )

        def on_load(request: gr.Request):
            query = dict(request.query_params) if request is not None else {}
            state = explore_state(
                data, query.get("split"), accession=query.get("release")
            )
            return [gr.Tabs(selected=tab_from_query(query, tabs))] + explore_updates(
                state
            )

        demo.load(on_load, inputs=None, outputs=[tab_set, *explore_outputs])
    return demo


if __name__ == "__main__":
    build_demo(load_data()).launch()
