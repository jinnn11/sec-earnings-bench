"""Plot the serving benchmark's throughput against its p95 latency.

Reads results/serving_results.json (secbench.servereport) and draws one line per
variant across the concurrency levels: x is the p95 end-to-end latency in seconds
(log scale), y is releases per hour (requests without an error, per hour of
makespan). Each line runs from 1 request in flight (left) to the most (right); the
levels are written beside the line that reaches the most releases per hour. Colors
are the reference palette's categorical slots 1-4 in the variants' fixed order
(validated on adjacent pairs, light mode), with a marker shape per variant and its
name at the line's end, so no variant is told by color alone. Needs matplotlib (a dev
dependency).

Usage: uv run python scripts/plot_serving.py [--results results/serving_results.json]
           [--out results/serving_throughput_latency.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Variant -> (color, marker), in the variants' fixed order.
STYLES = {
    "adapter": ("#2a78d6", "o"),
    "merged": ("#eb6834", "s"),
    "fp8": ("#1baf7a", "D"),
    "int4-awq": ("#eda100", "^"),
}
NAMES = {
    "adapter": "adapter (bf16 + LoRA)",
    "merged": "merged (bf16)",
    "fp8": "fp8",
    "int4-awq": "int4-awq",
}


def lines(results: dict[str, Any]) -> dict[str, list[tuple[int, float, float]]]:
    """Each variant's (level, p95 latency, releases per hour), levels without an
    error-free request left out."""
    found = {}
    for variant, v in results["variants"].items():
        points = [
            (x["level"], x["latency_s"]["p95"], x["releases_per_hour"])
            for x in v["levels"]
            if x["latency_s"]["p95"] and x["releases_per_hour"]
        ]
        if points:
            found[variant] = points
    return found


LABEL_GAP = 11  # points between the centres of two names at the lines' ends


def spread(ys: list[float], gap: float) -> list[float]:
    """Positions for labels wanted at ys (ascending), at least gap apart: labels that
    would overlap form a group, stacked gap apart and centred on their wanted mean."""
    groups = [[i] for i in range(len(ys))]

    def place(group: list[int]) -> list[float]:
        centre = sum(ys[i] for i in group) / len(group)
        start = centre - gap * (len(group) - 1) / 2
        return [start + gap * k for k in range(len(group))]

    merged = True
    while merged:
        merged = False
        for g in range(len(groups) - 1):
            if place(groups[g + 1])[0] - place(groups[g])[-1] < gap:
                groups[g] += groups.pop(g + 1)
                merged = True
                break
    out = [0.0] * len(ys)
    for group in groups:
        for i, y in zip(group, place(group), strict=True):
            out[i] = y
    return out


def end_labels(ax: Any, fig: Figure, ends: dict[str, tuple[float, float]]) -> None:
    """Each variant's short name beside its line's last point (the legend gives the
    long one), moved apart vertically where two lines end close together (merged and
    int4-awq end within a release per hour of each other on the H100 run)."""
    to_points = 72 / fig.dpi
    wanted = sorted(
        (ax.transData.transform(xy)[1] * to_points, variant)
        for variant, xy in ends.items()
    )
    placed = spread([y for y, _ in wanted], LABEL_GAP)
    for (y, variant), at in zip(wanted, placed, strict=True):
        ax.annotate(
            variant,
            ends[variant],
            xytext=(8, at - y),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=TEXT,
        )


def plot(results: dict[str, Any], out: Path) -> None:
    found = lines(results)
    if not found:
        raise SystemExit("no variant has load-test results to plot")
    fig = Figure(figsize=(9, 5.6), dpi=200)
    ax = fig.subplots()
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    for variant, points in found.items():
        color, marker = STYLES.get(variant, (TEXT_SECONDARY, "o"))
        xs, ys = [p[1] for p in points], [p[2] for p in points]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.5,
            solid_joinstyle="round",
            solid_capstyle="round",
            marker=marker,
            markersize=7,
            markerfacecolor=color,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
    top = max(found, key=lambda v: max(p[2] for p in found[v]))
    for level, x, y in found[top]:
        ax.annotate(
            str(level),
            (x, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=TEXT_SECONDARY,
        )
    session, settings = results["session"], results["settings"]
    gpus = ", ".join(g.split(",")[0] for g in session.get("gpus") or []) or "one GPU"
    fig.suptitle(
        f"Throughput against latency: {results['run']} {results['checkpoint']} on "
        f"{gpus}",
        x=0.06,
        ha="left",
        fontsize=12,
        color=TEXT,
    )
    ax.set_title(
        f"{settings['prompts']} {settings['load_split'].replace('_', ' ')} prompts per "
        f"level after {settings['warmup']} warm-up; left to right, "
        f"{', '.join(map(str, settings['levels']))} requests in flight, numbered on "
        f"{top}.",
        loc="left",
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )
    ax.set_xlabel("Latency p95, seconds (log scale)", color=TEXT_SECONDARY)
    ax.set_ylabel("Releases per hour", color=TEXT_SECONDARY)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    highest = max(p[2] for points in found.values() for p in points)
    ax.set_ylim(0, highest * 1.15)  # room for the level numbers above the top line
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    handles = [
        Line2D(
            [],
            [],
            color=STYLES.get(v, (TEXT_SECONDARY, "o"))[0],
            marker=STYLES.get(v, (TEXT_SECONDARY, "o"))[1],
            markeredgecolor=SURFACE,
            linewidth=1.5,
            markersize=7,
            label=NAMES.get(v, v),
        )
        for v in found
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=8.5,
        labelcolor=TEXT,
    )
    x_low, x_high = ax.get_xlim()
    ax.set_xlim(x_low, x_high * 1.6)  # room for the names at the lines' ends
    fig.subplots_adjust(left=0.1, right=0.97, top=0.87, bottom=0.11)
    end_labels(ax, fig, {v: points[-1][1:] for v, points in found.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results", type=Path, default=Path("results/serving_results.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/serving_throughput_latency.png")
    )
    args = parser.parse_args(argv)
    plot(json.loads(args.results.read_text(encoding="utf-8")), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
