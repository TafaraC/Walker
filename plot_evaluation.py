import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_POLICIES = (
    "oracle", "bc", "dagger", "dagger_mixed",
    "gail_dagger", "gail_bc", "gail_random"
)
COLORS = {
    "oracle": "#333333",
    "bc": "#4C78A8",
    "dagger": "#F58518",
    "dagger_mixed": "#ECA82C",
    "gail_dagger": "#54A24B",
    "gail_bc": "#E45756",
    "gail_random": "#B279A2",
}


def metric(result, name):
    return result["metrics"][name]


def asymmetric_error(stats):
    return np.array([[stats["mean"] - stats["ci95_low"]],
                     [stats["ci95_high"] - stats["mean"]]])


def summary_table(ax, aggregates, labels, display):
    """Render the principal Phase 1 results beneath the plot dashboard."""
    rows = (
        ("Normalized return (95% CI)", lambda result: (
            lambda stats: f'{stats["mean"]:.3f} [{stats["ci95_low"]:.3f}, {stats["ci95_high"]:.3f}]'
        )(metric(result, "normalized_return"))),
        ("Expert-test Action MSE", lambda result: f'{result["expert_test_action_mse"]:.4f}'),
        ("Trajectory Wasserstein", lambda result: f'{metric(result, "trajectory_sliced_wasserstein")["mean"]:.4f}'),
        ("Forward velocity (m/s)", lambda result: f'{metric(result, "mean_forward_velocity_mps")["mean"]:.3f}'),
        ("Falls / active min", lambda result: f'{metric(result, "falls_per_active_minute")["mean"]:.2f}'),
        ("Oracle queries", lambda result: f'{result["resources"]["oracle_queries"]:,}'),
        ("Environment steps", lambda result: f'{result["resources"]["environment_steps"]:,}'),
    )
    cell_text = [[formatter(aggregates[label]) for label in labels]
                 for _, formatter in rows]
    table = ax.table(
        cellText=cell_text,
        rowLabels=[name for name, _ in rows],
        colLabels=[display(label) for label in labels],
        cellLoc="center",
        rowLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    for column, label in enumerate(labels):
        header = table[(0, column)]
        header.set_facecolor(COLORS[label])
        header.set_text_props(color="white", weight="bold")
    for row in range(1, len(rows) + 1):
        table[(row, -1)].set_text_props(weight="bold")
        if row % 2 == 0:
            for column in range(len(labels)):
                table[(row, column)].set_facecolor("#F2F2F2")
    ax.axis("off")
    ax.set_title("Phase 1 quantitative summary", pad=8)


def plot_report(report_path, output_path, pdf_path):
    report = json.loads(Path(report_path).read_text())
    aggregates = report["aggregates"]
    labels = [label for label in DEFAULT_POLICIES if label in aggregates]
    learned = [label for label in DEFAULT_POLICIES if label != "oracle" and label in aggregates]
    colors = [COLORS[label] for label in labels]

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "figure.dpi": 140,
    })
    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.72))
    axes = np.array([
        [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
        [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
    ])
    table_ax = fig.add_subplot(grid[2, :])

    # Algorithmic performance
    ax = axes[0, 0]
    returns = [metric(aggregates[label], "normalized_return") for label in labels]
    for index, (label, stats) in enumerate(zip(labels, returns)):
        ax.bar(index, stats["mean"], color=COLORS[label], width=0.7)
        ax.errorbar(index, stats["mean"], yerr=asymmetric_error(stats),
                    fmt="none", ecolor="black", capsize=4, linewidth=1)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Oracle parity")
    display = lambda label: label.upper().replace("_", "-")
    ax.set_xticks(range(len(labels)), [display(label) for label in labels], rotation=15)
    ax.set_ylabel("Normalized return")
    ax.set_title("Task performance (mean and 95% CI)")
    ax.legend(frameon=False, fontsize=8)

    # Statistical imitation fidelity
    ax = axes[0, 1]
    x = np.arange(len(learned))
    action_mse = [aggregates[label]["expert_test_action_mse"] for label in learned]
    wasserstein = [metric(aggregates[label], "trajectory_sliced_wasserstein")["mean"]
                   for label in learned]
    width = 0.36
    ax.bar(x - width / 2, action_mse, width, label="Expert-test action MSE",
           color="#72B7B2")
    ax.bar(x + width / 2, wasserstein, width, label="Trajectory Wasserstein",
           color="#E45756")
    ax.set_xticks(x, [display(label) for label in learned], rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("Error (log scale; lower is better)")
    ax.set_title("Imitation fidelity")
    ax.legend(frameon=False, fontsize=8)

    # Biomechanical performance
    ax = axes[1, 0]
    velocity = [metric(aggregates[label], "mean_forward_velocity_mps")["mean"]
                for label in labels]
    falls = [metric(aggregates[label], "falls_per_active_minute")["mean"]
             for label in labels]
    ax.bar(np.arange(len(labels)) - width / 2, velocity, width,
           color=colors, label="Forward velocity (m/s)")
    ax.set_xticks(range(len(labels)), [display(label) for label in labels], rotation=15)
    ax.set_ylabel("Forward velocity (m/s)")
    ax2 = ax.twinx()
    ax2.bar(np.arange(len(labels)) + width / 2, falls, width,
            color=colors, alpha=0.35, hatch="//", label="Falls/min")
    ax2.set_ylabel("Falls per active minute")
    ax.set_title("Locomotion and stability")
    handles = [ax.patches[0], ax2.patches[0]]
    ax.legend(handles, ["Forward velocity", "Falls/min"], frameon=False, fontsize=8)

    # Resource complexity: separate axes because the resources are not equivalent.
    ax = axes[1, 1]
    queries = [aggregates[label]["resources"]["oracle_queries"] for label in learned]
    env_steps = [aggregates[label]["resources"]["environment_steps"] for label in learned]
    ax.bar(x - width / 2, queries, width, color="#B279A2", label="Oracle queries")
    ax.bar(x + width / 2, env_steps, width, color="#FF9DA6", label="Environment steps")
    ax.set_xticks(x, [display(label) for label in learned], rotation=15)
    ax.set_ylabel("Cumulative samples")
    ax.set_title("Supervisory and environment cost")
    ax.legend(frameon=False, fontsize=8)

    summary_table(table_ax, aggregates, labels, display)

    fig.suptitle("FC Portugal imitation-learning evaluation", fontsize=14)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if pdf_path:
        pdf = Path(pdf_path)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Figure: {output}")
    if pdf_path:
        print(f"[Plots] PDF:    {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the imitation evaluation report")
    parser.add_argument("--report", default="evaluation/comparison.json")
    parser.add_argument("--output", default="evaluation/evaluation_summary.png")
    parser.add_argument("--pdf", default="evaluation/evaluation_summary.pdf")
    args = parser.parse_args()
    plot_report(args.report, args.output, args.pdf)
