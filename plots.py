"""Plot communication-adjusted reward against message cost for every condition."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order, validated for color-vision deficiency. Every
# figure uses the same color for the same condition.
CONDITIONS = (
    ("silent", "Silent", "#257de0"),
    ("always_send", "Always send", "#f47230"),
    ("conflict_rule", "Heuristic", "#00b894"),
    ("learned", "Learned", "#f2a600"),
)
LAYOUT_NAMES = {
    "cramped_room": "Cramped Room",
    "coordination_ring": "Coordination Ring",
}
# Always-send pays for 400 messages per episode and falls far below the others.
# It is drawn but left off scale so the selective conditions stay readable.
OFF_SCALE = {"always_send"}
TEXT = "#1f1f1f"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"


def spread_labels(positions, gap):
    """Move label heights apart by at least ``gap``, keeping their order."""
    order = sorted(range(len(positions)), key=lambda index: positions[index])
    placed = list(positions)
    for previous, current in zip(order, order[1:]):
        placed[current] = max(placed[current], placed[previous] + gap)
    return placed


def plot_cost_sweep(results, output):
    """Draw one panel per layout: IQM adjusted reward against message cost.

    Reads the ``cost_sweep`` rows written by ``experiments.py learned``. The
    y-range fits the selective conditions and their intervals. Conditions in
    OFF_SCALE are drawn anyway and labeled where they leave the panel.
    """
    layouts = list(results["layouts"])
    figure, axes = plt.subplots(
        1, len(layouts), figsize=(5.2 * len(layouts), 4.2), constrained_layout=True
    )
    axes = [axes] if len(layouts) == 1 else list(axes)
    for axis, layout in zip(axes, layouts):
        sweep = results["layouts"][layout]["cost_sweep"]
        costs = [row["message_cost"] for row in sweep]
        series = {}
        for key, label, color in CONDITIONS:
            summaries = [row["adjusted_return"][key] for row in sweep]
            series[key] = (
                [summary["iqm"] for summary in summaries],
                [summary["iqm_95_percent_interval"][0] for summary in summaries],
                [summary["iqm_95_percent_interval"][1] for summary in summaries],
            )
        on_scale = [values for key, values in series.items() if key not in OFF_SCALE]
        bottom = min(min(low) for _, low, _ in on_scale)
        top = max(max(high) for _, _, high in on_scale)
        margin = 0.1 * (top - bottom)
        bottom, top = bottom - margin, top + margin

        end_labels = []
        for key, label, color in CONDITIONS:
            iqms, low, high = series[key]
            axis.fill_between(costs, low, high, color=color, alpha=0.12, linewidth=0)
            axis.plot(
                costs, iqms, color=color, linewidth=2, marker="o", markersize=5,
                label=label,
            )
            if key in OFF_SCALE:
                # Label where the line leaves the bottom of the panel.
                for (x0, y0), (x1, y1) in zip(zip(costs, iqms), zip(costs[1:], iqms[1:])):
                    if y0 >= bottom > y1:
                        exit_cost = x0 + (y0 - bottom) / (y0 - y1) * (x1 - x0)
                        axis.annotate(
                            f"{label} (off scale)", (exit_cost, bottom),
                            xytext=(6, 8), textcoords="offset points",
                            fontsize=9, color=TEXT,
                        )
                        break
                else:
                    end_labels.append((iqms[-1], label))
            else:
                end_labels.append((iqms[-1], label))
        heights = spread_labels([height for height, _ in end_labels], 0.06 * (top - bottom))
        for height, (_, label) in zip(heights, end_labels):
            axis.annotate(
                label, (costs[-1], height), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=TEXT,
            )
        axis.set_ylim(bottom, top)
        axis.set_title(LAYOUT_NAMES.get(layout, layout), color=TEXT, fontsize=11, loc="left")
        axis.set_xlabel("Cost per message", color=MUTED)
        axis.set_xticks(costs)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=MUTED, labelsize=9)
        axis.set_xlim(costs[0], costs[-1] + 0.3 * (costs[-1] - costs[0]))
    axes[0].set_ylabel("Adjusted reward (IQM, 95% interval)", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, frameon=False, fontsize=9, loc="outside upper left",
        ncols=len(CONDITIONS),
    )
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("learned-results.json"))
    parser.add_argument("--output", type=Path, default=Path("figures/cost-sweep.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_cost_sweep(json.loads(args.results.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
