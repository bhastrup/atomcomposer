from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


from src.performance.cumulative.investigator import CummulativeInvestigator
from src.performance.cumulative.cum_io import CumulativeIO


FONT_TYPE = "serif"
FONT_SIZE = 16
TITLE_FONT_SIZE = FONT_SIZE + 3
figsize = (4, 5)


def load_discovery_metrics_data(
    base_dir: Path, run_names_map: dict, n_seeds: int, step_max: int, mol_dataset: str, sample_tag: str
) -> dict:
    """Load data .json files for each run and seed"""
    run_names = list(run_names_map.keys())
    run_names_short = list(run_names_map.values())

    total_dm = {}
    for run_name_short, run_name in zip(run_names_short, run_names):
        total_dm[run_name_short] = {}

        for seed in range(n_seeds):
            inv = CummulativeInvestigator(
                CumulativeIO(save_dir=base_dir / f"{run_name}/seed_{seed}" / "results", batched=True),
                data_dir=Path.cwd() / "data",
                step_max=step_max,
                mol_dataset=mol_dataset,
            )
            dm = inv.load_discovery_metrics(sample_tag, mol_dataset)
            total_dm[run_name_short][seed] = inv._aggregate_discovery_metrics(dm)

    return total_dm


def compute_mean_and_std(dm_all_seeds: dict, ratio: bool) -> dict:
    stats = {"rediscovered_means": [], "novel_means": [], "old_data_means": [], "rediscovered_std": [], "novel_std": []}

    for dms in dm_all_seeds.values():
        stats["rediscovered_means"].append(
            np.mean([dm["rediscovery_ratio"] if ratio else dm["rediscovered"] for dm in dms.values()])
        )
        stats["novel_means"].append(np.mean([dm["expansion_ratio"] if ratio else dm["novel"] for dm in dms.values()]))
        stats["old_data_means"].append(np.mean([1.0 if ratio else dm["old_data"] for dm in dms.values()]))
        stats["rediscovered_std"].append(
            np.std([dm["rediscovery_ratio"] if ratio else dm["rediscovered"] for dm in dms.values()], ddof=1)
        )
        stats["novel_std"].append(
            np.std([dm["expansion_ratio"] if ratio else dm["novel"] for dm in dms.values()], ddof=1)
        )

    return stats


def plot_cum_discovery_seeds(
    dm_all_seeds: dict, mol_dataset: str = "QM7", undiscovered_gap: bool = False, ratio: bool = False
):
    """Plot the cumulative discovery metrics averaged over seeds"""
    n_bars = len(dm_all_seeds)

    fig_width = 0.8 * n_bars
    fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))

    # Compute means
    stats = compute_mean_and_std(dm_all_seeds, ratio)

    categorical_positions = range(n_bars)  # positions for the bars
    bottom = stats["old_data_means"] if undiscovered_gap else stats["rediscovered_means"]
    ax.bar(categorical_positions, stats["rediscovered_means"], label="Rediscovered")
    ax.bar(categorical_positions, stats["novel_means"], bottom=bottom, label="Novel")

    # Old data
    for i, y in enumerate(categorical_positions):
        ax.plot(
            [y - 0.5, y + 0.5],
            [stats["old_data_means"][i], stats["old_data_means"][i]],
            color="black",
            linestyle="-",
            linewidth=5,
            label=f"{mol_dataset}" if i == 0 else "",
        )

    ax.set_xticks(categorical_positions)
    ax.set_xticklabels(dm_all_seeds.keys())

    # Set y-axis to log scale
    ax.set_yscale("log")

    if not ratio:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # thicker spine
    spine_width = 1.5
    ax.spines["top"].set_linewidth(spine_width)
    ax.spines["right"].set_linewidth(spine_width)
    ax.spines["bottom"].set_linewidth(spine_width)
    ax.spines["left"].set_linewidth(spine_width)

    count_label = "Expansion/Discovery Counts" if not ratio else "Rediscovery and Expansion Ratio"
    dataset_label = f"{mol_dataset}" if not ratio else f"{mol_dataset}"
    ax.set_ylabel(count_label, fontsize=FONT_SIZE, fontfamily=FONT_TYPE)
    ax.set_xlabel(dataset_label, fontsize=FONT_SIZE, fontfamily=FONT_TYPE)

    ax.grid(axis="y", linestyle="--")
    ax.legend(fontsize=FONT_SIZE, prop={"family": FONT_TYPE})

    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 2)
    plt.setp(ax.get_xticklabels(), family=FONT_TYPE)
    plt.setp(ax.get_yticklabels(), family=FONT_TYPE)

    fig.tight_layout()

    return fig, ax


def plot_cum_discovery_seeds_2_subplots(dm_all_seeds: dict, mol_dataset: str = "QM7", ratio: bool = False):
    """Plot the cumulative discovery metrics averaged over seeds"""
    n_bars = len(dm_all_seeds)
    fig_width = 2 + 0.5 * (n_bars - 1)
    colors = plt.cm.tab10.colors  # colors = plt.cm.Set2.colors

    # Create figure with two subplots, stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 4), height_ratios=[1, 1])

    # Compute means
    stats = compute_mean_and_std(dm_all_seeds, ratio)
    categorical_positions = range(n_bars)

    # Top subplot for expansion/novel (use different colors)
    # Use different colors for each bar in top subplot
    ax1.bar(categorical_positions, stats["novel_means"], label="Novel", color=colors[:n_bars])
    ax1.errorbar(
        categorical_positions,
        stats["novel_means"],
        yerr=stats["novel_std"],
        fmt="none",
        ecolor="black",
        capsize=5,
        elinewidth=2,
        capthick=2,
    )

    ax2.bar(categorical_positions, stats["rediscovered_means"], label="Rediscovered", color=colors[:n_bars])
    ax2.errorbar(
        categorical_positions,
        stats["rediscovered_means"],
        yerr=stats["rediscovered_std"],
        fmt="none",
        ecolor="black",
        capsize=5,
        elinewidth=2,
        capthick=2,
    )

    # Configure both subplots
    for ax in [ax1, ax2]:
        ax.set_xticks(categorical_positions)
        ax.set_xticklabels(dm_all_seeds.keys(), fontsize=FONT_SIZE, fontfamily=FONT_TYPE)

        if not ratio:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Styling
        spine_width = 1.5
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

        ax.grid(axis="y", linestyle="-")
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    # Remove x-axis labels from top subplot
    ax1.set_xticklabels([])
    ax1.set_xlabel("")

    # Set labels
    count_label = "Ratio" if ratio else "Counts"
    ax1.set_title(f"Expansion {count_label}", fontsize=TITLE_FONT_SIZE, fontfamily=FONT_TYPE)
    ax2.set_title(f"Rediscovery {count_label}", fontsize=TITLE_FONT_SIZE, fontfamily=FONT_TYPE)

    fig.tight_layout()
    return fig, (ax1, ax2)
