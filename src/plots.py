"""Static charts for the README, written next to results.md."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluate import gap_rows, question_types, summarise, type_label  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Categorical slots 1-6 of the validated reference palette, fixed per pipeline.
COLORS = {"Chunk": "#2a78d6", "Graph": "#eb6834", "Graph (hub-pruned)": "#1baf7a",
          "Hybrid (fusion)": "#eda100", "Hybrid (routed)": "#e87ba4", "Graph v2": "#008300"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _grouped(ax, groups, runs_vals, title, ylabel, floor=None):
    names = list(runs_vals)
    width = 0.8 / len(names)
    x = np.arange(len(groups))
    for i, name in enumerate(names):
        vals = runs_vals[name]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width * 0.92,
                      color=COLORS.get(name, INK_2), label=name, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=8, color=INK_2)
    if floor is not None:
        ax.axhline(floor, color=INK_2, lw=1, ls="--", zorder=2, label=f"No-context floor ({floor:.0f})")
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 108)  # headroom so a label on a bar near 100 clears the title
    ax.set_yticks(range(0, 101, 20))
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(length=0)


def scores_chart(questions, runs, no_context, path):
    types = question_types(questions)
    groups = ["All"] + [type_label(t).capitalize() for t in types]
    summ = {n: summarise(r["scores"], questions) for n, r in runs.items()}
    recall = {n: [100 * s["recall"]] + [100 * s["by_type"][t]["recall"] for t in types] for n, s in summ.items()}
    f1 = {n: [100 * s["f1"]] + [100 * s["by_type"][t]["f1"] for t in types] for n, s in summ.items()}
    floor = 100 * summarise(no_context, questions)["f1"] if no_context else None

    fig, axes = plt.subplots(1, 2, figsize=(4 + 2.4 * len(groups), 4.2))
    _grouped(axes[0], groups, recall, "Support recall@5 (retrieval)", "%")
    _grouped(axes[1], groups, f1, "Answer F1", "%", floor)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gaps_chart(questions, runs, cfg, path):
    rows = [g for g in gap_rows(questions, runs, cfg) if g["metric"] in ("recall", "f1")]
    names = [n for n in runs if n != "Chunk"]
    subsets = ["all", *question_types(questions)]
    labels = [(metric, subset) for metric in ("recall", "f1") for subset in subsets]

    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.32 * len(labels) * len(names)))
    y = np.arange(len(labels))[::-1]
    step = 0.6 / len(names)
    for i, name in enumerate(names):
        for yy, (metric, subset) in zip(y, labels):
            g = next(r for r in rows if r["pipeline"] == name and r["metric"] == metric and r["subset"] == subset)
            yi = yy + ((len(names) - 1) / 2 - i) * step
            ax.plot([100 * g["lo"], 100 * g["hi"]], [yi, yi], color=COLORS[name], lw=2, zorder=3)
            ax.plot(100 * g["diff"], yi, "o", ms=8, color=COLORS[name], mec=SURFACE, mew=2, zorder=4,
                    label=name if (metric, subset) == labels[0] else None)
    ax.axvline(0, color=INK_2, lw=1, zorder=2)
    names_map = {"recall": "Recall@5", "f1": "F1"}
    ax.set_yticks(y, [f"{names_map[m]} · {type_label(s)}" for m, s in labels])
    ax.set_xlabel("Difference from chunk, points, with paired bootstrap 95% interval\n← chunk better · other pipeline better →")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(names))
    ax.set_title("Where each pipeline beats chunk, and where it loses", loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cost_chart(runs, path):
    # Rows that reuse another row's index (pruning, routing) would repeat its cost.
    names = [n for n in runs if n not in ("Graph (hub-pruned)", "Hybrid (routed)")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.6 + 0.65 * len(names)))
    for ax, key, title, fmt in (
        (axes[0], "index_seconds", "Index build time (s)", "{:,.0f} s"),
        (axes[1], "index_tokens", "Index LLM tokens", "{:,.0f}"),
    ):
        vals = [runs[n][key] for n in names]
        bars = ax.barh(names[::-1], vals[::-1], color=[COLORS[n] for n in names[::-1]], height=0.55, zorder=3)
        for b, v in zip(bars, vals[::-1]):
            ax.text(b.get_width() + max(vals) * 0.02, b.get_y() + b.get_height() / 2, fmt.format(v),
                    va="center", fontsize=9, color=INK_2)
        ax.set_xlim(0, max(vals) * 1.3 or 1)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(length=0)
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_all(questions, runs, cfg, no_context, out):
    scores_chart(questions, runs, no_context, out / "scores.png")
    if len(runs) > 1:
        gaps_chart(questions, runs, cfg, out / "gaps.png")
        cost_chart(runs, out / "cost.png")
