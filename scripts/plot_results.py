"""Plot accuracy against cost per 1,000 releases from results/comparison.md.

Reads the table of runs on all linked releases that secbench.compare writes (run it
with --gpu-hourly so GPU runs share one price) and draws one point per run: x is cost
per 1,000 releases (log scale), y is accuracy. Arrows join each open model's prompt
v1 run to its v2 run. Needs matplotlib (a dev dependency).

Usage: uv run python scripts/plot_results.py [--comparison results/comparison.md]
           [--out results/accuracy_vs_cost.png]
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

FULL_RUNS = "## Runs on all"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Categorical slots 1-3 of the reference palette, validated all-pairs (light mode).
GROUPS = {
    "Anthropic API": "#2a78d6",
    "OpenAI API": "#eb6834",
    "Open model, one A100 80GB": "#1baf7a",
}
NAMES = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "gpt-6.1-sol": "GPT-6.1 Sol",
    "gpt-6-luna": "GPT-6 Luna",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "google/gemma-3-12b-it": "Gemma-3-12B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}
# Label offsets in points from the labelled point, and alignment, by display name;
# the three top-right labels are staggered so they don't collide.
LABELS = {
    "Claude Opus 5.5": (0, -12, "center", "top"),
    "Claude Sonnet 5.5": (0, -26, "center", "top"),
    "GPT-6.1 Sol": (-4, -12, "right", "top"),
    "Claude Haiku 5.5": (9, 2, "left", "center"),
    "GPT-6 Luna": (-9, 2, "right", "center"),
    "gpt-oss-20b": (9, 0, "left", "center"),
    "Gemma-3-12B": (9, 0, "left", "center"),
    "Qwen3-8B, thinking": (9, -2, "left", "center"),
    "Qwen3-8B, no thinking": (9, -4, "left", "center"),
    "Llama-3.1-8B": (9, 0, "left", "center"),
}


@dataclass(frozen=True)
class Point:
    name: str
    group: str
    prompt: str
    accuracy: float  # percent
    cost: float  # dollars per 1,000 releases


def table_rows(text: str) -> list[dict[str, str]]:
    """The rows of the table under the "## Runs on all ..." heading."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(FULL_RUNS))
    table = []
    for line in lines[start + 1 :]:
        if line.startswith("|"):
            table.append([cell.strip() for cell in line.strip("|").split(" | ")])
        elif table:
            break
    header, _, *rows = table
    for row in rows:
        assert len(row) == len(header), row
    return [dict(zip(header, row, strict=True)) for row in rows]


def point(row: dict[str, str]) -> Point:
    correct, graded = map(int, re.match(r"(\d+)/(\d+)", row["accuracy"]).groups())
    name = NAMES[row["model"]]
    if row["model"] == "Qwen/Qwen3-8B":
        thinking = '"enable_thinking": false' not in row["settings sent"]
        name += ", thinking" if thinking else ", no thinking"
    if row["cost basis"].startswith("GPU"):
        group = "Open model, one A100 80GB"
    else:
        group = "Anthropic API" if row["provider"] == "anthropic" else "OpenAI API"
    cost = float(row["cost / 1,000 releases"].lstrip("$").replace(",", ""))
    return Point(name, group, row["prompt"], 100 * correct / graded, cost)


def plot(points: list[Point], out: Path) -> None:
    fig = Figure(figsize=(9, 6), dpi=200)
    ax = fig.subplots()
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")

    # Arrows from each open model's v1 run to its v2 run.
    by_name: dict[str, dict[str, Point]] = {}
    for p in points:
        by_name.setdefault(p.name, {})[p.prompt] = p
    for runs in by_name.values():
        v1, v2 = runs.get("extract_v1"), runs.get("extract_v2")
        if v1 and v2 and v1.group.startswith("Open"):
            ax.annotate(
                "",
                xy=(v2.cost, v2.accuracy),
                xytext=(v1.cost, v1.accuracy),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": GROUPS[v1.group],
                    "lw": 1.5,
                    "shrinkA": 5,
                    "shrinkB": 5,
                    "mutation_scale": 12,
                },
                zorder=2,
            )

    # Hollow v1 markers above the filled v2 ones, so an overlapping pair shows both.
    for p in points:
        color = GROUPS[p.group]
        filled = p.prompt == "extract_v2"
        ax.scatter(
            p.cost,
            p.accuracy,
            s=64,
            facecolor=color if filled else SURFACE,
            edgecolor=color,
            linewidth=2,
            zorder=3 if filled else 4,
        )

    # One label per model and set-up: at its v1 point (open models spread out more
    # there), else at its only point.
    for name, runs in by_name.items():
        p = runs.get("extract_v1") or runs["extract_v2"]
        dx, dy, ha, va = LABELS[name]
        ax.annotate(
            name,
            (p.cost, p.accuracy),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=9,
            color=TEXT,
            zorder=5,
        )

    ax.set_xlabel(
        "Cost per 1,000 releases, USD (log scale; GPU at $1 per GPU-hour)",
        color=TEXT_SECONDARY,
    )
    ax.set_ylabel("Accuracy, % of graded figures", color=TEXT_SECONDARY)
    ax.set_title(
        "Accuracy vs cost per 1,000 releases",
        color=TEXT,
        fontsize=12,
        loc="left",
        pad=22,
    )
    ax.text(
        0,
        1.02,
        "Hollow: prompt v1. Filled: prompt v2. Arrows join an open model's v1 and v2 "
        "runs.",
        transform=ax.transAxes,
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:g}"))
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
    low = min(p.cost for p in points)
    high = max(p.cost for p in points)
    ax.set_xlim(low / 1.6, high * 2.2)
    ax.set_ylim(40, 102)
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=color,
            label=group,
        )
        for group, color in GROUPS.items()
    ] + [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            markeredgewidth=2,
            markerfacecolor=face,
            markeredgecolor=TEXT_SECONDARY,
            label=label,
        )
        for face, label in ((SURFACE, "Prompt v1"), (TEXT_SECONDARY, "Prompt v2"))
    ]
    ax.legend(
        handles=legend,
        loc="lower right",
        frameon=False,
        labelcolor=TEXT,
        fontsize=9,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--comparison", type=Path, default=Path("results/comparison.md")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/accuracy_vs_cost.png")
    )
    args = parser.parse_args(argv)
    points = [point(row) for row in table_rows(args.comparison.read_text())]
    plot(points, args.out)
    print(f"wrote {args.out} ({len(points)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
