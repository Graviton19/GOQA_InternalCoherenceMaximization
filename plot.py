import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_figure1(
    results: Dict[str, float],
    output_path: str = "outputs/figure1_aggregate.png",
    title: str = "GlobalOpinionQA — ICM Aggregate Results",
):
    conditions = [
        "Zero-shot",
        "Zero-shot (Chat)",
        "Prompt-Golden",
        "Prompt-ICM (Ours)",
    ]

    colors = ["#C39BD3", "#BDC3C7", "#F5B041", "#5DADE2"]
    accuracies = [results.get(c, 0) * 100 for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, accuracies, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.6)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{acc:.1f}%",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 1 saved to: {output_path}")


def plot_figure1_per_persona(
    per_persona_results: Dict[str, Dict[str, float]],
    output_path: str = "outputs/figure1_per_persona.png",
):
    conditions = [
        "Zero-shot",
        "Zero-shot (Chat)",
        "Prompt-Golden",
        "Prompt-ICM (Ours)",
    ]
    colors = ["#C39BD3", "#BDC3C7", "#F5B041", "#5DADE2"]

    personas = list(per_persona_results.keys())
    n_personas = len(personas)
    n_conditions = len(conditions)

    fig, ax = plt.subplots(figsize=(max(10, n_personas * 2), 6))
    x = np.arange(n_personas)
    width = 0.8 / n_conditions

    for i, cond in enumerate(conditions):
        offsets = x + (i - n_conditions / 2 + 0.5) * width
        values = [per_persona_results[p].get(cond, 0) * 100 for p in personas]
        ax.bar(offsets, values, width * 0.9, color=colors[i],
               edgecolor="black", linewidth=0.5, label=cond)

    ax.set_title("GlobalOpinionQA — Per-Persona Results",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    short_names = [p.split("(")[0].strip()[:15] for p in personas]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-persona figure saved to: {output_path}")


def plot_figure2(
    results: Dict[str, Dict[int, float]],
    output_path: str = "outputs/figure2_accuracy_vs_shots.png",
    title: str = "GlobalOpinionQA — Accuracy vs. Number of In-Context Examples",
):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    style_map = {
        "Gold":   {"color": "#F5B041", "marker": "s", "linestyle": "-",
                   "label": "Gold Labels"},
        "ICM":    {"color": "#5DADE2", "marker": "o", "linestyle": "-",
                   "label": "ICM Labels (Ours)"},
        "Random": {"color": "#BDC3C7", "marker": "^", "linestyle": "--",
                   "label": "Random Labels"},
    }

    for method in ["Gold", "ICM", "Random"]:
        if method not in results:
            continue
        data = results[method]
        shots = sorted(data.keys())
        accs = [data[s] * 100 for s in shots]
        style = style_map[method]
        ax.plot(shots, accs,
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=2,
                markersize=7, label=style["label"])

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of In-Context Examples", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc="lower right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 2 saved to: {output_path}")


def plot_figure2_per_persona(
    per_persona_shots: Dict[str, Dict[str, Dict[int, float]]],
    output_path_prefix: str = "outputs/figure2",
):
    for country, results in per_persona_shots.items():
        safe_name = country.replace(" ", "_").replace("(", "").replace(")", "")
        plot_figure2(
            results,
            output_path=f"{output_path_prefix}_{safe_name}.png",
            title=f"Accuracy vs. Examples — {country}",
        )

    if len(per_persona_shots) > 1:
        combined = {}
        for method in ["ICM", "Random", "Gold"]:
            shot_accs = {}
            for country, results in per_persona_shots.items():
                if method in results:
                    for n_shots, acc in results[method].items():
                        if n_shots not in shot_accs:
                            shot_accs[n_shots] = []
                        shot_accs[n_shots].append(acc)
            combined[method] = {s: sum(a) / len(a) for s, a in shot_accs.items()}

        plot_figure2(
            combined,
            output_path=f"{output_path_prefix}_aggregate.png",
            title="Accuracy vs. Examples — Aggregated",
        )
