import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ase import Atoms
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from src.performance.energetics import XTBOptimizer, EnergyUnit


FONT_TYPE = "serif"

TITLE_SIZE = 24
SUBTITLE_SIZE = 18


def plot_rae_distributions(formula_dfs: Dict[str, pd.DataFrame], column_name: str = "rae_relaxed"):
    """Plot the distribution of the Relative Absolute Error (RAE) for each formula."""

    x_min, x_max = [-0.15, 1.0]

    n_cols = 4
    row_height = 2

    # Calculate number of rows needed based on the number of formulas
    num_formulas = len(formula_dfs)
    num_rows = (num_formulas + 2) // n_cols  # add 2 for integer division adjustment to handle all items

    fig, axes = plt.subplots(nrows=num_rows, ncols=n_cols, figsize=(15, num_rows * row_height))
    axes = axes.flatten()  # Flatten the 2D array of axes into 1D for easier iteration
    bins = np.linspace(x_min, x_max, 100)

    for i, (formula, df) in enumerate(formula_dfs.items()):
        rae = df[column_name].values

        # Plot on the ith subplot
        axes[i].hist(rae, bins=bins, alpha=1.0, label=formula)
        axes[i].set_xlim([x_min, x_max])
        axes[i].legend(loc="upper right", prop={"size": SUBTITLE_SIZE, "family": FONT_TYPE})
        axes[i].axvline(x=0, color="red", linestyle="--")

    # If the number of formulas is not a multiple of 3, turn off the unused axes
    for j in range(i + 1, num_rows * n_cols):
        axes[j].axis("off")

    plt.tight_layout()
    return fig, axes


def plot_rae_one_figure(formula_dfs: Dict[str, pd.DataFrame], column_name: str = "rae_relaxed"):
    """Plot all histograms of the Relative Absolute Error (RAE) in a single figure."""
    x_min, x_max = [-0.15, 1.0]
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(x_min, x_max, 100)

    for formula, df in formula_dfs.items():
        rae = df[column_name].values
        ax.hist(rae, bins=bins, alpha=0.5, label=formula)

    ax.set_xlim([x_min, x_max])
    ax.legend()
    ax.set_title("RAE Distributions", fontsize=TITLE_SIZE, fontname=FONT_TYPE)
    return fig, ax


def plot_rae_aggregated(formula_dfs: Dict[str, pd.DataFrame], column_name: str = "rae_relaxed"):
    """Plot a histogram of the aggregated Relative Absolute Error (RAE)"""
    x_min, x_max = [-0.15, 1.0]
    n_bins = 30

    fig, ax = plt.subplots(figsize=(10, 6))

    values = []
    for formula, df in formula_dfs.items():
        rae = df[column_name].values
        values.extend(rae)

    ax.hist(values, bins=np.linspace(x_min, x_max, n_bins), alpha=1.0, rwidth=0.8)
    ax.set_xlim([x_min, x_max])
    ax.set_title("Aggregated RAE Distribution", fontsize=TITLE_SIZE, fontname=FONT_TYPE)
    return fig, ax


def plot_rae_against_ref(
    formula_dfs: Dict[str, pd.DataFrame], ref_rae_values: List[float], column_name: str = "rae_relaxed"
):
    """Plot the RAE values of the generated data against the reference data (aggregated)"""
    x_min, x_max = [-0.15, 1.0]
    n_bins = 50
    bins = np.linspace(x_min, x_max, n_bins)
    bin_width = np.diff(bins)

    # Calculate reference histogram without plotting
    ref_hist, ref_bin_edges = np.histogram(ref_rae_values, bins=bins)

    # Collect all generated RAE values from formula_dfs
    values = []
    for formula, df in formula_dfs.items():
        rae = df[column_name].values
        values.extend(rae)

    prop_factor = len(values) / len(ref_rae_values)

    # Calculate histogram for generated values without plotting
    gen_hist, gen_bin_edges = np.histogram(values, bins=bins)

    # Renormalize the RL generated RAE values
    normalized_gen_hist = gen_hist / prop_factor

    fig, ax = plt.subplots(figsize=(10, 6))

    rwidth = 0.96
    ax.bar(ref_bin_edges[:-1], ref_hist, width=bin_width * rwidth, alpha=1.0, label="Reference")
    ax.bar(gen_bin_edges[:-1], normalized_gen_hist, width=bin_width * rwidth, alpha=0.65, label="Generated")

    ax.set_xlim([bins[0], bins[-1]])
    ax.set_title("RAE Distributions", fontsize=TITLE_SIZE, fontname=FONT_TYPE)
    ax.legend(loc="upper right", prop={"size": SUBTITLE_SIZE, "family": FONT_TYPE})

    ax.tick_params(axis="both", which="major", labelsize=SUBTITLE_SIZE)

    return fig, ax


################################# ETKDG ########################################


def get_etkdg_energies(
    smiles: List[str], n_confs: int = 10, relax: bool = False, fmax: int = 0.05, step_max: int = 150
) -> List[float]:
    """
    https://greglandrum.github.io/rdkit-blog/posts/2021-02-22-etkdg-and-distance-constraints.html
    """

    calc = XTBOptimizer(method="GFN2-xTB", energy_unit=EnergyUnit.EV)
    ps = rdDistGeom.ETKDGv3()
    ps.randomSeed = 0xF00D

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMultipleConfs(mol, n_confs, ps)
    confs = mol.GetConformers()

    atoms_objects = []

    for conf in confs:
        pos = conf.GetPositions()
        atoms = Atoms(symbols=[atom.GetSymbol() for atom in mol.GetAtoms()], positions=pos)
        atoms_objects.append(atoms)

    if relax:
        etkdg_energies = [
            calc.optimize_atoms(atoms, max_steps=step_max, fmax=fmax)["energy_after"] for atoms in atoms_objects
        ]
    else:
        etkdg_energies = [calc.calc_potential_energy(atoms) for atoms in atoms_objects]

    return etkdg_energies


def get_rl_energies(
    atoms_list: List[Atoms],
    relax: bool = False,
    df: pd.DataFrame = None,
    fmax: int = 0.05,
    step_max: int = 150,
    energy_name: str = "e_relaxed",
) -> List[float]:
    calc = XTBOptimizer(method="GFN2-xTB", energy_unit=EnergyUnit.EV)

    # If we need to relax the atoms further
    if relax:
        return [calc.optimize_atoms(atoms, max_steps=step_max, fmax=fmax)["energy_after"] for atoms in atoms_list]

    # If energies are available in the dataframe
    if df is not None and energy_name in df.columns and not df[energy_name].isnull().values.all():
        return [df[energy_name].values[i] for i in range(len(atoms_list))]

    # Else calculate the energies
    return [calc.calc_potential_energy(atoms) for atoms in atoms_list]


def get_etkdg_dict(
    formula_dfs: Dict[str, pd.DataFrame],
    rollouts: Dict[str, List[Atoms]],
    n_confs: int = 10,
    fmax: int = 0.05,
    step_max: int = 150,
    n_smiles_max: int = 100,
    max_smiles_per_formula: int = 10,
) -> dict:

    smiles_count = 0
    results = {}
    for formula_iter, (formula, atoms_list) in enumerate(rollouts.items()):
        df = formula_dfs[formula]
        assert len(atoms_list) > 0, f"No atoms found for formula {formula}"
        assert len(atoms_list) == len(df), (
            f"Number of atoms {len(atoms_list)} does not match the number of rows in the dataframe {len(df)}"
        )

        smiles_set = set(smiles for smiles in df["SMILES"].values if smiles is not None)

        for smiles_iter, smiles in enumerate(smiles_set):
            df_smiles = df[df["SMILES"] == smiles]
            isomer_rollouts = [atoms_list[i] for i in df_smiles.index.values]

            # Obtain RL and ETKDG generated energies
            rl_energies = get_rl_energies(isomer_rollouts, relax=True, df=df_smiles, fmax=fmax, step_max=step_max)
            try:
                etkdg_energies = get_etkdg_energies(smiles, n_confs, relax=True, fmax=fmax, step_max=step_max)
            except Exception:
                logging.warning("Failed to fetch ETKDG energies")
                continue

            n_atoms = len(atoms_list[0])
            best_rl_energy = min(rl_energies)
            etkdg_deltas = [(best_rl_energy - energy) / n_atoms for energy in etkdg_energies]

            # Store metrics
            results[smiles] = {}
            results[smiles]["etkdg_deltas"] = etkdg_deltas
            results[smiles]["better_fraction"] = sum(1 for delta in etkdg_deltas if delta < 0) / len(etkdg_deltas)

            # Break conditions (for testing)
            smiles_count += 1
            if smiles_count >= n_smiles_max:
                return results
            if smiles_iter >= max_smiles_per_formula:
                break

    return results


def plot_etkdg(results: Dict[str, dict]):
    """Compares the RL generated energies with the ETKDG generated energies."""

    n_bins = 50
    fig, ax = plt.subplots(figsize=(10, 6))

    values = []
    for smiles, smiles_stats in results.items():
        values.extend(smiles_stats["etkdg_deltas"])

    ax.hist(values, bins=n_bins, alpha=1.0, rwidth=0.8)

    ax.legend()
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_title("ETKDG Distributions", fontsize=TITLE_SIZE, fontname=FONT_TYPE)

    return fig, ax
