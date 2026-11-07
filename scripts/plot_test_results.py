"""Plot accuracy against cost per 1,000 releases on stage 2's test sets.

Reads results/test_results.json (secbench.testreport) and draws one panel per test
set, one point per model: x is cost per 1,000 releases (log scale), y is accuracy
over the graded figures, with its 95% bootstrap interval as a vertical bar. GPU runs
are at the price billed for their session (not $1/h, as results/comparison.md and
the stage 1 chart show them), wall time only. The company test's step-00025 run
includes an instance freeze: a hollow marker shows its freeze-excluded estimate,
joined to the run's own cost. Needs matplotlib (a dev dependency).

Usage: uv run python scripts/plot_test_results.py [--results results/test_results.json]
           [--out results/test_accuracy_vs_cost.png]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Categorical slots 1-3 of the reference palette, validated all-pairs (light mode),
# in the same roles as the stage 1 chart.
ANTHROPIC, OPENAI, QWEN = "Anthropic API", "OpenAI API", "Qwen3-8B, one H100 NVL"
GROUPS = {ANTHROPIC: "#2a78d6", OPENAI: "#eb6834", QWEN: "#1baf7a"}
NAMES = {"step-00025": "step-00025 (fine-tuned)"}
SPLITS = {"company_test": "Company test", "time_test": "Time test"}
# Label offsets in points, and alignment, by (split, model); offsets from the point,
# or (LOW) from the bottom of its interval bar.
LOW = "low"
LABELS: dict[tuple[str, str], tuple] = {
    ("company_test", "step-00025"): (8, -7, "left", "top"),
    ("company_test", "base"): (8, 0, "left", "center"),
    ("company_test", "gpt-6.1-sol"): (0, -9, "center", "top"),
    ("company_test", "claude-haiku-5-5"): (0, 9, "center", "bottom"),
    ("company_test", "gpt-6-luna"): (0, -4, "center", "top", LOW),
    ("time_test", "step-00025"): (8, 0, "left", "center"),
    ("time_test", "base"): (8, 0, "left", "center"),
    ("time_test", "gpt-6.1-sol"): (0, -9, "center", "top"),
    ("time_test", "claude-haiku-5-5"): (8, -6, "left", "top"),
    ("time_test", "gpt-6-luna"): (-8, 6, "right", "bottom"),
}


@dataclass(frozen=True)
class Point:
    model: str
    name: str
    group: str
    accuracy: float  # percent
    low: float
    high: float
    cost: float  # dollars per 1,000 releases


def points(report: dict[str, Any], split: str) -> list[Point]:
    s = report["splits"][split]
    result = []
    for model, label in report["models"].items():
        entry = s["accuracy"][model]["all"]
        cost = s["cost"][model]
        if cost["basis"] == "gpu":
            group = QWEN
        else:
            group = ANTHROPIC if s["runs"][model]["provider"] == "anthropic" else OPENAI
        result.append(
            Point(
                model,
                NAMES.get(model, label),
                group,
                entry["accuracy"],
                *entry["interval"],
                cost["per_1000"],
            )
        )
    return result


def subtitle(report: dict[str, Any]) -> str:
    prices = ", ".join(
        f"${report['splits'][split]['billed_usd_per_hour']:g}/h ({title.lower()})"
        for split, title in SPLITS.items()
    )
    return (
        f"GPU runs at the price billed, {prices}, times wall time; API runs at token "
        "prices. Bars: 95% intervals over companies."
    )


def plot(report: dict[str, Any], out: Path) -> None:
    fig = Figure(figsize=(11, 5.6), dpi=200)
    axes = fig.subplots(1, 2, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    all_costs = []
    for ax, (split, title) in zip(axes, SPLITS.items(), strict=True):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        s = report["splits"][split]
        freeze = s.get("freeze")
        for p in points(report, split):
            color = GROUPS[p.group]
            all_costs.append(p.cost)
            ax.vlines(p.cost, p.low, p.high, color=color, linewidth=1.5, zorder=2)
            ax.scatter(
                p.cost,
                p.accuracy,
                s=64,
                facecolor=color,
                edgecolor=SURFACE,
                linewidth=2,
                zorder=4,
            )
            dx, dy, ha, va, *anchor = LABELS[split, p.model]
            ax.annotate(
                p.name,
                (p.cost, p.low if anchor == [LOW] else p.accuracy),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=9,
                color=TEXT,
                zorder=5,
            )
            if freeze and p.model == report["under_test"]:
                excluded = freeze["excluded_per_1000"]
                all_costs.append(excluded)
                ax.annotate(
                    "",
                    xy=(excluded, p.accuracy),
                    xytext=(p.cost, p.accuracy),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "lw": 1.5,
                        "linestyle": (0, (3, 2)),
                        "shrinkA": 6,
                        "shrinkB": 6,
                        "mutation_scale": 12,
                    },
                    zorder=3,
                )
                ax.scatter(
                    excluded,
                    p.accuracy,
                    s=64,
                    facecolor=SURFACE,
                    edgecolor=color,
                    linewidth=2,
                    zorder=4,
                )
                ax.annotate(
                    "without the\ninstance freeze",
                    (excluded, p.accuracy),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=TEXT_SECONDARY,
                    zorder=5,
                )
        cov = s["coverage"]
        ax.set_title(
            f"{title}: {cov['kept_releases']} releases, {cov['graded']:,} figures",
            color=TEXT,
            fontsize=11,
            loc="left",
        )
        ax.set_xlabel("Cost per 1,000 releases, USD (log scale)", color=TEXT_SECONDARY)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:g}"))
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY)
    low, high = min(all_costs), max(all_costs)
    for ax in axes:
        ax.set_xlim(low / 1.8, high * 2.4)
    axes[0].set_ylim(82, 101)
    axes[0].set_ylabel("Accuracy, % of graded figures", color=TEXT_SECONDARY)
    fig.suptitle(
        "Stage 2 test sets: accuracy vs cost per 1,000 releases",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=12,
    )
    # "$" escaped: matplotlib reads text between two as math.
    fig.text(
        0.01,
        0.905,
        subtitle(report).replace("$", r"\$"),
        color=TEXT_SECONDARY,
        fontsize=9,
    )
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
            markerfacecolor=SURFACE,
            markeredgecolor=GROUPS[QWEN],
            label="freeze-excluded estimate",
        )
    ]
    axes[1].legend(
        handles=legend,
        loc="lower right",
        frameon=False,
        labelcolor=TEXT,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results", type=Path, default=Path("results/test_results.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/test_accuracy_vs_cost.png")
    )
    args = parser.parse_args(argv)
    report = json.loads(args.results.read_text(encoding="utf-8"))
    plot(report, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
