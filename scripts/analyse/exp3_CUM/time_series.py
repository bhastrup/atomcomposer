import copy
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from src.performance.energetics import EnergyUnit
from src.data.reference_dataloader import ReferenceDataLoader
from src.performance.cumulative.investigator import CummulativeInvestigator
from src.performance.cumulative.cum_io import SmilesCounterType, sort_formulas_by_terminal_count


def normalize_counts(smiles_counter: SmilesCounterType, ref_counts: Dict, tag: str) -> SmilesCounterType:
    """Normalize the counts by the reference counts"""
    for formula in smiles_counter[tag]:
        ref_count = ref_counts[formula]
        smiles_counter[tag][formula] = [
            (step, diff, count / ref_count) for (step, diff, count) in smiles_counter[tag][formula]
        ]

    return smiles_counter


def plot_cumulative_counts(
    inv: CummulativeInvestigator,
    data_dir: Path,
    tag: str = "in_sample",
    num_formulas: int = 50,
    mol_dataset: str = None,
    log_scale: bool = False,
    aggregated: bool = True,
    rediscovery: bool = False,
) -> None:
    """
    Makes a plot of the cumulative number of molecules found (as a function of steps).
    This function is monotonically increasing of course.
    We can plot the rising count for each bag individually, but we can also plot the
    total count averaged over all bags.
    """

    if rediscovery:
        smiles_counter = copy.deepcopy(inv.output["smiles_counter_rediscovery"])
    else:
        smiles_counter = copy.deepcopy(inv.output["smiles_counter"])

    if mol_dataset is not None:
        ref_smiles = (
            ReferenceDataLoader(data_dir=data_dir).load_and_polish(mol_dataset, EnergyUnit.EV, fetch_df=False).smiles
        )

        ref_counts = {f: len(ref_smiles[f]) for f in ref_smiles}

        # Normalize the counts in smiles_counter by the reference counts
        smiles_counter = normalize_counts(smiles_counter, ref_counts, tag)
        smiles_counter = sort_formulas_by_terminal_count(smiles_counter)

    num_formulas = min(num_formulas, len(smiles_counter[tag])) if num_formulas is not None else len(smiles_counter[tag])

    fn = plot_aggregated if aggregated else plot_individual
    fn(inv, smiles_counter, tag, num_formulas, log_scale, mol_dataset, rediscovery)


def plot_individual(
    inv: CummulativeInvestigator,
    smiles_counter: SmilesCounterType,
    tag: str,
    num_formulas: int,
    log_scale: bool,
    mol_dataset: str,
    rediscovery: bool,
) -> None:
    """Plot the cumulative number of molecules found for each formula"""

    # Plotting
    plt.figure(figsize=(10, 8))
    colors = plt.cm.hsv(np.linspace(0, 0.85, num_formulas))
    for idx, (formula, data) in enumerate(smiles_counter[tag].items()):
        steps, counts = zip(*data)
        if log_scale:  # Logarithmic scale, base 10
            counts = np.log10(np.array(counts) + [0.001 if cc == 0 else 0 for cc in counts])
        plt.step(steps, counts, where="post", color=colors[idx], label=formula, alpha=0.6)
        if idx >= num_formulas - 1:
            break

    plt.title(
        f"{'Relative ' if mol_dataset else ''}Number of SMILES Discovered During "
        f"Training{' (log scale)' if log_scale else ''}"
    )

    # plt.ylabel('Cumulative Number of SMILES')
    plt.xlabel("Steps")
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_path = inv.plot_dir / "cumulative_individual{}.png".format("_rediscovery" if rediscovery else "")
    plt.savefig(plot_path)
    print(f"Cumulative plot saved to {plot_path}")


def plot_aggregated(
    inv: CummulativeInvestigator,
    smiles_counter: SmilesCounterType,
    tag: str,
    num_formulas: int,
    log_scale: bool,
    mol_dataset: str,
    rediscovery: bool,
) -> None:
    """Plot the cumulative number of molecules found averaged over all formulas"""

    # Plotting
    plt.figure(figsize=(10, 8))

    # Aggregate the counts over all formulas
    all_obs = [(step, diff) for formula in smiles_counter[tag] for (step, diff, _) in smiles_counter[tag][formula]]
    all_obs = sorted(all_obs, key=lambda x: x[0])
    steps, diffs = zip(*all_obs)
    counts = np.cumsum(diffs)

    if log_scale:  # Logarithmic scale, base 10
        counts = np.log10(np.array(counts) + [0.001 if cc == 0 else 0 for cc in counts])
    plt.step(steps, counts, where="post", color="blue", label="Aggregated", alpha=0.6)

    plt.title(
        f"{'Relative ' if mol_dataset else ''}Number of SMILES Discovered During "
        f"Training{' (log scale)' if log_scale else ''}"
    )

    # plt.ylabel('Cumulative Number of SMILES')
    plt.xlabel("Steps")
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_path = inv.plot_dir / "cumulative_aggregated{}.png".format("_rediscovery" if rediscovery else "")
    plt.savefig(plot_path)
    print(f"Cumulative plot saved to {plot_path}")

    # Save the data
    np.savez_compressed(
        inv.plot_dir / "cumulative_aggregated{}.npz".format("_rediscovery" if rediscovery else ""),
        steps=steps,
        counts=counts,
    )
