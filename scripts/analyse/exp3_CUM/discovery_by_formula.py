import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.performance.cumulative.investigator import CummulativeInvestigator, select_top_n_formulas


def plot_rediscovery_novelty(
    inv: CummulativeInvestigator,
    tag: str = "in_sample",
    num_formulas: int = 10,
    mol_dataset: str = "qm7",
    undiscovered_gap: bool = False,
) -> None:

    # Fetch the discovery metrics
    dm = inv.get_discovery_metrics(tag=tag, mol_dataset=mol_dataset)
    num_formulas = min(num_formulas, len(dm["formulas"])) if num_formulas is not None else len(dm["formulas"])
    formulas, rediscovered, novel, old_data = select_top_n_formulas(n=num_formulas, dm=dm)

    # Plotting
    fig_height = 6 + 0.15 * num_formulas
    fig, ax = plt.subplots(figsize=(10, fig_height))
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
            linewidth=3,
            label=f"{mol_dataset}" if i == 0 else "",
        )

        if i == num_formulas - 1:
            break

    ax.set_yticks(y_positions)
    ax.set_yticklabels(formulas)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Counts")
    ax.set_ylabel("Formula")
    ax.set_title("Rediscovered and expansion of reference dataset ({})".format(mol_dataset))
    ax.legend()
    ax.grid(axis="x", linestyle="--")

    # Save the plot
    plot_path = inv.plot_dir / "discovery_by_formula.png"
    plt.savefig(plot_path)
    print(f"Discovery by formula plot saved to {plot_path}")
