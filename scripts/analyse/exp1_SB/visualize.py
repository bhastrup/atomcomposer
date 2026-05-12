import time
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


from ase import Atoms
from ase.io import read
import pandas as pd

from src.tools import util
from src.performance.energetics import XTBOptimizer, EnergyUnit
from src.performance.utils import get_max_view_positions


import subprocess
import json
import shlex


def submit_job(params, script_path: str = "analysis/catalysis/version-1/script.py"):

    # Convert dictionary to a JSON string
    params_json = json.dumps(params)
    quoted_params_json = shlex.quote(params_json)

    print(f"quoted_params_json: {quoted_params_json}")

    # Construct the command correctly
    cmd = [
        "chimerax",
        "--nogui",
        "--offscreen",
        "--script",
        f'"{script_path} {quoted_params_json}"',  # Properly quote the JSON
    ]

    print("Executing:", " ".join(cmd))  # Debugging
    subprocess.run(" ".join(cmd), shell=True, check=True)  # Run as a single shell command


def get_path(tag: str, run_name: str, seed: int, formula: str) -> Path:
    return Path(f"{base_dir}/{run_name}/seed_{seed}/results/{tag}/{formula}")


def fix_eval_formulas(eval_formulas: List[str]) -> List[str]:
    return [util.bag_tuple_to_str_formula(util.str_formula_to_bag_tuple(f)) for f in eval_formulas]


def get_all_dfs(eval_formulas: List[str], n_seeds: int, tag: str, run_name: str) -> Dict[str, pd.DataFrame]:
    dfs = {}
    trajs = {}

    for formula in tqdm(eval_formulas, desc="Loading data"):
        dfs[formula] = []
        trajs[formula] = []

        # Load metrics from each seed
        paths = [get_path(tag, run_name, seed, formula) for seed in range(n_seeds)]
        print(paths)
        existing_paths = [path for path in paths if path.exists()]

        if not len(existing_paths) == n_seeds:
            print(f"Missing {n_seeds - len(existing_paths)} paths for {run_name} {formula}")
            continue

        for seed_path in tqdm(existing_paths, desc="Loading data"):
            # load df
            df = pd.read_csv(os.path.join(seed_path, "df.csv"))
            dfs[formula].append(df)

            # load atoms.traj
            atoms_traj_path = os.path.join(seed_path, "atoms.traj")
            atoms_list = read(atoms_traj_path, index=":")
            trajs[formula].extend(atoms_list)

    all_dfs = {formula: pd.concat(dfs[formula]).reset_index(drop=True) for formula in eval_formulas}

    return all_dfs, trajs


def filter_data(
    all_dfs: Dict[str, pd.DataFrame],
    trajs: Dict[str, List[Atoms]],
    eval_formulas: List[str],
    sorting_key: str,
    smiles_col: str,
    stratify_on_smiles: bool,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[Atoms]]]:

    merged_dfs = all_dfs.copy()
    merged_trajs = trajs.copy()

    trajs_filtered = {}
    dfs_filtered = {}

    for formula in eval_formulas:
        df = merged_dfs[formula].copy().reset_index(drop=True)
        # Sort by sorting_key
        df = df.sort_values(by=sorting_key)

        # Drop rows where SMILES is None
        df = df[df[smiles_col].notna()]

        # Drop rows where relax_stable is False
        # df = df[df['relax_stable']]

        # Keep only the best of each SMILES
        if stratify_on_smiles:
            df = df.drop_duplicates(subset=[smiles_col], keep="first")

        # Update the trajectories
        idx = df.index.copy()
        trajs_filtered[formula] = [merged_trajs[formula][i] for i in idx]
        dfs_filtered[formula] = df.reset_index(drop=True)

    return dfs_filtered, trajs_filtered


def find_candidate_mols(
    dfs_filtered: Dict[str, pd.DataFrame],
    trajs_filtered: Dict[str, List[Atoms]],
    eval_formulas: List[str],
    n_query: int,
    n_non_query: int,
) -> Dict[str, List[Atoms]]:
    """Select the most promising molecules."""
    best_mols = {}
    for formula in eval_formulas:
        # Select molecules with rings and more than 4 atoms in the largest ring
        query = "n_rings > 0 and n_atoms_ring_max > 4"
        queried_idx = dfs_filtered[formula].query(query).index
        queried_mols = [trajs_filtered[formula][i] for i in queried_idx[:n_query]].copy()

        # Other promising molecules based on energy (complementary to query)
        non_query_idx = dfs_filtered[formula].index.difference(queried_idx)
        non_query_mols = [trajs_filtered[formula][i] for i in non_query_idx[:n_non_query]].copy()

        # Combine the two lists
        best_mols[formula] = queried_mols + non_query_mols

    return best_mols


def optimize_and_sort_mols(
    best_mols: Dict[str, List[Atoms]],
    eval_formulas: List[str],
) -> Tuple[Dict[str, List[Atoms]], Dict[str, List[float]]]:
    calc = XTBOptimizer(method="GFN2-xTB", energy_unit=EnergyUnit.EV, use_mp=False)

    energies_relaxed = {}
    mols_relaxed = {}
    for formula in tqdm(eval_formulas, desc="Optimizing and sorting molecules"):
        print(f"Optimizing and sorting molecules for {formula}")
        e_relaxed = []
        m_relaxed = []
        for mol in tqdm(best_mols[formula], desc="Relaxing molecules"):
            opt_info = calc.optimize_atoms(mol, max_steps=1000, fmax=0.05, redirect_logfile=False)
            e_relaxed.append(opt_info["energy_after"])
            m_relaxed.append(opt_info["new_atoms"])

        # Sort based on relaxed energy
        e_relaxed, m_relaxed = zip(*sorted(zip(e_relaxed, m_relaxed), key=lambda pair: pair[0]))
        energies_relaxed[formula] = list(e_relaxed)
        mols_relaxed[formula] = list(m_relaxed)

    return mols_relaxed, energies_relaxed


def rotate_mols(mols: Dict[str, List[Atoms]]) -> Dict[str, List[Atoms]]:
    """Rotate of the molecule to get a better view."""
    for formula, mols_to_view in mols.items():
        for mol in mols_to_view:
            mol.set_positions(get_max_view_positions(mol.get_positions()))


def launch_chimerax_jobs(mols_to_view: Dict[str, List[Atoms]], save_dir: str, bg_color_str: str):

    formulas = list(mols_to_view.keys())

    for formula in formulas:
        # Finally, write molecules to pdb and create Chimerax visualization (png).
        mol_paths = [os.path.join(save_dir, f"{formula}_{i}.pdb") for i in range(len(mols_to_view))]
        save_paths = [os.path.join(save_dir, f"{formula}_{i}.png") for i in range(len(mols_to_view))]

        for i, mol in enumerate(mols_to_view[formula]):
            mol.write(mol_paths[i])

        # Create Chimerax visualization (png).
        for i, (mol_path, save_path) in enumerate(zip(mol_paths, save_paths)):
            params = {
                "pdb_path": os.path.join(os.getcwd(), mol_path),
                "image_path": os.path.join(os.getcwd(), save_path),
                "movie_path": None,
                "atoms_to_deselect": [],  # 0, 1, 2, 3]
                "selected_action": "",
                "bg_color": bg_color_str,
            }

            print(f"params: {params}")
            submit_job(params)

    return


eval_formulas_before = ["C3H5NO3", "C4H7N", "C3H8O", "C7H10O2", "C7H8N2O2"]

if __name__ == "__main__":
    # Define the experiment parameters

    n_seeds = 1
    tag = "EXP1_30000"

    eval_formulas = fix_eval_formulas(eval_formulas_before)

    run_names = ["entropy-schedule-A"]
    base_dir = "from_niflheim/digital_discovery"

    n_query = 3  # n_mols matching the query
    n_non_query = 3  # n_mols not matching the query

    n_mols = 5  # number of molecules to view

    stratify_on_smiles = True

    bg_color_str = "black"

    show_relaxed = True
    sorting_key = "e_relaxed" if show_relaxed else "abs_energy"
    smiles_col = "SMILES" if show_relaxed else "NEW_SMILES"

    illustration_dir_name = f"exp1_{bg_color_str}_{sorting_key}_no_stable_TEST"

    for run_name in run_names:
        save_dir = os.path.join(base_dir, run_name, illustration_dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    for run_name in run_names:
        # Get all data
        start_time = time.time()
        all_dfs, trajs = get_all_dfs(eval_formulas, n_seeds, tag, run_name)
        print(f"a) Time taken to get all data: {time.time() - start_time} seconds")

        # Filter data
        start_time = time.time()
        dfs_filtered, trajs_filtered = filter_data(
            all_dfs, trajs, eval_formulas, sorting_key, smiles_col, stratify_on_smiles
        )
        print(f"b) Time taken to filter data: {time.time() - start_time} seconds")

        # Find candidate molecules
        start_time = time.time()
        best_mols = find_candidate_mols(dfs_filtered, trajs_filtered, eval_formulas, n_query, n_non_query)
        print(f"c) Time taken to find candidate molecules: {time.time() - start_time} seconds")

        # Optimize and sort molecules
        start_time = time.time()
        if show_relaxed:
            mols_to_view, energies_relaxed = optimize_and_sort_mols(best_mols, eval_formulas)
        else:
            mols_to_view = best_mols
            energies_relaxed = None
        print(f"d) Time taken to optimize and sort molecules: {time.time() - start_time} seconds")

        # Rotate molecules
        start_time = time.time()
        rotate_mols(mols_to_view)
        print(f"e) Time taken to rotate molecules: {time.time() - start_time} seconds")

        # Visualize molecules
        start_time = time.time()
        launch_chimerax_jobs(mols_to_view, save_dir, bg_color_str)
        print(f"e) Time taken to launch Chimerax jobs: {time.time() - start_time} seconds")
