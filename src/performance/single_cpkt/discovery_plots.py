import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_rediscovery_novelty(formula_metrics: dict, mol_dataset: str = "QM7", undiscovered_gap: bool = False):
    formulas = list(formula_metrics.keys())
    rediscovered = [formula_metrics[formula]["rediscovered"] for formula in formulas]
    novel = [formula_metrics[formula]["n_novel"] for formula in formulas]
    old_data = [formula_metrics[formula]["old_data_size"] for formula in formulas]

    # Sort by old_data_size
    formulas, rediscovered, novel, old_data = zip(
        *sorted(zip(formulas, rediscovered, novel, old_data), key=lambda x: x[3])
    )

    fig, ax = plt.subplots()
    y_positions = range(len(formulas))  # Y-positions for the bars
    ax.barh(y_positions, rediscovered, label="Rediscovered")
    ax.barh(y_positions, novel, left=old_data if undiscovered_gap else rediscovered, label="Novel")

    # Old data
    for i, y in enumerate(y_positions):
        ax.plot(
            [old_data[i], old_data[i]],
            [y - 0.5, y + 0.5],
            color="black",
            linestyle="-",
            linewidth=5,
            label=f"{mol_dataset}" if i == 0 else "",
        )
    # for i, y in enumerate(y_positions):
    #     ax.hlines(y, 0, old_data[i], color='black', linestyle='-', linewidth=5, label='Old Data' if i == 0 else '')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(formulas)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Counts")
    ax.set_ylabel("Formula")
    ax.set_title("Rediscovered and expansion of reference dataset ({})".format(mol_dataset))
    ax.legend()
    ax.grid(axis="x", linestyle="--")

    return fig, ax


def plot_global_rediscovery_novelty(
    df_global_metrics: dict,
    mol_dataset: str = "QM7",
    undiscovered_gap: bool = False,
    vertical: bool = False,
    ratio: bool = False,
):
    n_sets = 1

    rediscovered = df_global_metrics["rediscovery_ratio"] if ratio else df_global_metrics["rediscovered"]
    novel = df_global_metrics["expansion_ratio"] if ratio else df_global_metrics["novel"]
    old_data = [1.0] if ratio else [df_global_metrics["old_data_size"]]

    fig, ax = plt.subplots()
    categorical_positions = range(n_sets)  # positions for the bars
    bottom = old_data if undiscovered_gap else rediscovered
    if vertical:
        ax.bar(categorical_positions, rediscovered, label="Rediscovered")
        ax.bar(categorical_positions, novel, bottom=bottom, label="Novel")
    else:
        ax.barh(categorical_positions, rediscovered, label="Rediscovered")
        ax.barh(categorical_positions, novel, left=bottom, label="Novel")

    # Old data
    for i, y in enumerate(categorical_positions):
        if vertical:
            ax.plot(
                [y - 0.5, y + 0.5],
                [old_data[i], old_data[i]],
                color="black",
                linestyle="-",
                linewidth=5,
                label=f"{mol_dataset}" if i == 0 else "",
            )
        else:
            ax.plot(
                [old_data[i], old_data[i]],
                [y - 0.5, y + 0.5],
                color="black",
                linestyle="-",
                linewidth=5,
                label=f"{mol_dataset}" if i == 0 else "",
            )

    if vertical:
        ax.set_xticks([])
    else:
        ax.set_yticks([])

    if not ratio:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)) if vertical else ax.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )

    # thicker spine
    spine_width = 1.5
    ax.spines["top"].set_linewidth(spine_width)
    ax.spines["right"].set_linewidth(spine_width)
    ax.spines["bottom"].set_linewidth(spine_width)
    ax.spines["left"].set_linewidth(spine_width)

    count_label = "Expansion/Discovery Counts" if not ratio else "Rediscovery and Expansion Ratio"
    dataset_label = f"{mol_dataset}" if not ratio else f"{mol_dataset}"
    ax.set_ylabel(count_label) if vertical else ax.set_xlabel(count_label)
    ax.set_xlabel(dataset_label) if vertical else ax.set_ylabel(dataset_label)

    ax.set_title("Rediscovered and expansion of reference dataset ({})".format(mol_dataset))
    ax.grid(axis="y" if vertical else "x", linestyle="--")
    ax.legend()

    return fig, ax
