import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
from ase import Atoms
from ase.io import read

from src.performance.utils import get_max_view_positions

from scripts.analyse.exp5_scaffold.visuals.visualize import (
    submit_job,
    filter_data,
    find_candidate_mols,
)

from ase.units import Debye
from src.performance.energetics import XTBOptimizer, EnergyUnit


def compute_and_cache_unrelaxed_dipoles(
    base_dir: str,
    data_folder: str,
    eval_formulas: List[str],
) -> Dict[str, np.ndarray]:
    """Compute dipoles from unrelaxed atoms.traj and cache to dipoles_unrelaxed.npy.

    Always reads atoms.traj (never atoms_relaxed.traj). Results are index-aligned
    with df.csv. Nothing in the existing result directory is overwritten.
    """
    calc = XTBOptimizer(method="GFN2-xTB", energy_unit=EnergyUnit.EV)
    all_dipoles = {}

    for formula in tqdm(eval_formulas, desc="Computing unrelaxed dipoles"):
        formula_dir = Path(f"{base_dir}/results/{data_folder}/{formula}")
        cache_path = formula_dir / "dipoles_unrelaxed.npy"

        if cache_path.exists():
            print(f"{formula}: loading cached unrelaxed dipoles")
            all_dipoles[formula] = np.load(cache_path)
            continue

        traj_path = formula_dir / "atoms.traj"
        if not traj_path.exists():
            print(f"{formula}: atoms.traj not found, skipping")
            continue

        atoms_list = read(str(traj_path), index=":")
        dipoles = []
        for atoms in tqdm(atoms_list, desc=f"  {formula}", leave=False):
            try:
                d = calc.calc_dipole(atoms)
            except Exception:
                d = np.nan
            dipoles.append(d if d is not None else np.nan)

        arr = np.array(dipoles, dtype=float)
        np.save(cache_path, arr)
        print(f"{formula}: saved {len(arr)} unrelaxed dipoles to {cache_path}")
        all_dipoles[formula] = arr

    return all_dipoles


def load_cached_unrelaxed_dipoles(
    base_dir: str,
    data_folder: str,
    eval_formulas: List[str],
) -> Dict[str, np.ndarray]:
    """Load pre-computed dipoles_unrelaxed.npy files. Returns dict formula -> array."""
    result = {}
    for formula in eval_formulas:
        cache_path = Path(f"{base_dir}/results/{data_folder}/{formula}/dipoles_unrelaxed.npy")
        if cache_path.exists():
            result[formula] = np.load(cache_path)
        else:
            print(f"Warning: no cached unrelaxed dipoles for {formula} at {cache_path}")
    return result


def get_data_folder(base_dir: str, data_folder: str, formula: str) -> Path:
    return Path(f"{base_dir}/results/{data_folder}/{formula}")


def get_all_dfs(base_dir: str, eval_formulas: List[str], data_folder: str) -> Dict[str, pd.DataFrame]:
    dfs = {}
    trajs = {}

    for formula in tqdm(eval_formulas, desc="Loading data"):
        dfs[formula] = []
        trajs[formula] = []

        # Load metrics
        formula_folder_path = get_data_folder(base_dir, data_folder, formula)
        print(formula_folder_path)
        existing_path = formula_folder_path if formula_folder_path.exists() else None

        if not existing_path:
            print(f"Missing path for {formula}")
            continue

        # load df
        df = pd.read_csv(os.path.join(existing_path, "df.csv"))
        dfs[formula].append(df)

        # prefer relaxed atoms; fall back to raw rollout atoms for older run dirs
        relaxed_path = os.path.join(existing_path, "atoms_relaxed.traj")
        traj_path = relaxed_path if os.path.exists(relaxed_path) else os.path.join(existing_path, "atoms.traj")
        atoms_list = read(traj_path, index=":")
        trajs[formula].extend(atoms_list)

    valid_formulas = [f for f in eval_formulas if dfs.get(f)]
    all_dfs = {formula: pd.concat(dfs[formula]).reset_index(drop=True) for formula in valid_formulas}
    trajs = {formula: trajs[formula] for formula in valid_formulas}

    return all_dfs, trajs


def select_best_mols(
    dfs_filtered: Dict[str, pd.DataFrame], trajs_filtered: Dict[str, List[Atoms]], eval_formulas: List[str], n_mols: int
) -> Tuple[Dict[str, List[Atoms]], Dict[str, pd.DataFrame]]:
    """Just select the best n_mols of each formula, according to the sorting_key."""
    best_mols = {}
    best_df = {}
    for formula in eval_formulas:
        best_mols[formula] = trajs_filtered[formula][:n_mols]
        best_df[formula] = dfs_filtered[formula].iloc[:n_mols]
    return best_mols, best_df


@dataclass
class Molecule:
    smiles: str
    abs_energy: float
    e_relaxed_f10: float
    e_relaxed_f5: float
    dipole_relaxed_f5: float
    atoms: Atoms


@dataclass
class AgentData:
    tag: str
    visuals_dir: str
    save_dir: str
    top_k: Dict[str, List[Molecule]]
    all_dfs: Dict[str, pd.DataFrame]
    all_trajs: Dict[str, List[Atoms]]


def build_sorted_candidates(
    best_trajs: Dict[str, List[Atoms]],
    best_df: Dict[str, pd.DataFrame],
    eval_formulas: List[str],
    smiles_col: str,
) -> Dict[str, List[Molecule]]:
    """Build Molecule objects from pre-computed df.csv values and pre-relaxed atoms."""
    sorted_candidates = {}
    for formula in eval_formulas:
        df = best_df[formula].reset_index(drop=True)
        trajs = best_trajs[formula]

        # Sort by relaxed dipole (largest first), fall back to unrelaxed dipole
        sort_col = "dipole_relaxed" if "dipole_relaxed" in df.columns else "dipole"
        order = df[sort_col].fillna(0.0).argsort()[::-1].values

        sorted_candidates[formula] = [
            Molecule(
                smiles=df.iloc[i][smiles_col],
                abs_energy=df.iloc[i]["abs_energy"],
                e_relaxed_f10=df.iloc[i].get("e_relaxed", None),
                e_relaxed_f5=df.iloc[i].get("e_relaxed", None),
                dipole_relaxed_f5=df.iloc[i].get("dipole_relaxed", df.iloc[i].get("dipole", 0.0)),
                atoms=trajs[i],
            )
            for i in order
        ]

    return sorted_candidates


def rotate_mols(mols: Dict[str, List[Atoms]]) -> Dict[str, List[Atoms]]:
    """Rotate of the molecule to get a better view."""
    for formula, mols_to_view in mols.items():
        for mol in mols_to_view:
            mol.set_positions(get_max_view_positions(mol.get_positions()))


def launch_chimerax_jobs(mols_to_view: Dict[str, List[Atoms]], save_dir: str, bg_color_str: str):
    rotate_mols(mols_to_view)

    formulas = list(mols_to_view.keys())

    for formula in formulas:
        mols = mols_to_view[formula]

        # Finally, write molecules to pdb and create Chimerax visualization (png).
        pdb_paths = [os.path.join(save_dir, f"{formula}_{i}.pdb") for i in range(len(mols))]
        png_paths = [os.path.join(save_dir, f"{formula}_{i}.png") for i in range(len(mols))]

        # Write molecules to pdb.
        for i, mol in enumerate(mols):
            mol.write(pdb_paths[i])

        # Create Chimerax visualization (png).
        for i, (pdb_path, png_path) in enumerate(zip(pdb_paths, png_paths)):
            params = {
                "pdb_path": os.path.join(os.getcwd(), pdb_path),
                "image_path": os.path.join(os.getcwd(), png_path),
                "movie_path": None,
                "atoms_to_deselect": [],  # 0, 1, 2, 3]
                "selected_action": "",
                "bg_color": bg_color_str,
            }

            print(f"params: {params}")
            submit_job(params)

    return


def default_eval_formulas() -> List[str]:
    return [
        "H4C3O3",  # Bad - Ethylene carbonate (EC)
        "H6C3O3",  # Good - Dimethyl carbonate (DMC)
        "H6C4O3",  # Bad - Propylene carbonate (PC)
        "H8C4O3",  # Good - Ethyl methyl carbonate (EMC)
        "H6C5O3",  # Bad -Diethenylcarbonate (DMC) (doulbe bonded carbons)
        "H8C5O3",  # Bad - 1,2-Butylene carbonate (ring-like)
        "H10C5O3",  # Good - Diethyl carbonate (DEC) / Methyl propyl carbonate (MPC),
        # Others
        "H10C4O2",  # 1,2-Dimethoxyethane (DME), (glyme),
        "H6C2OS",  # Dimethyl sulfoxide (DMSO),
        "H2C3O3",  # vinylene carbonate (VC)
    ]


FORMULA_COMMON_NAMES = {
    "H4C3O3": "Ethylene carbonate (EC)",
    "H6C3O3": "Dimethyl carbonate (DMC)",
    "H6C4O3": "Propylene carbonate (PC)",
    "H8C4O3": "Ethyl methyl carbonate (EMC)",
    "H6C5O3": "Diethenyl carbonate",
    "H8C5O3": "1,2-Butylene carbonate",
    "H10C5O3": "Diethyl carbonate (DEC)",
    "H10C4O2": "1,2-Dimethoxyethane (DME)",
    "H6C2OS": "Dimethyl sulfoxide (DMSO)",
    "H2C3O3": "Vinylene carbonate (VC)",
}


def build_eval_formulas_pretty() -> List[str]:
    formulas = default_eval_formulas()
    # In the text string, flip the order of hydrogens and carbon to match convention.
    # Remember their counts as well.
    # Use regex to do this.

    pretty_formulas = []
    for formula in formulas:
        # Use regex to find matches for element+count, e.g., H4, C3, O3
        matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
        # We'll reorder so that C comes first, then H, then anything else alphabetically
        # Build a dict; count is optional (absent means 1, e.g. O and S in H6C2OS)
        elem_dict = {elem: int(count) if count else 1 for elem, count in matches if elem}
        order = ["C", "H"] + sorted([e for e in elem_dict.keys() if e not in ("C", "H")])
        # Only include element if present
        pretty = "".join(f"{e}{elem_dict[e]}" if elem_dict[e] > 1 else e for e in order if e in elem_dict)
        pretty_formulas.append(pretty)

    return pretty_formulas


def build_comparison_image_name(bg_color_str: str, model_names: Dict[str, Dict[str, str]], prefix: str = "") -> str:
    """Build output image name from compared model keys."""
    model_keys = list(model_names.keys())
    safe_keys = [str(k).replace(" ", "_") for k in model_keys]
    return f"{prefix}{bg_color_str}_{'_vs_'.join(safe_keys)}.png"


def get_agent_data(
    base_dir: str,
    model_names: Dict[str, Dict[str, str]],
    eval_formulas: List[str],
    stratify_on_smiles: bool,
    search_for_candidates: bool,
    n_query: int,
    n_non_query: int,
    n_mols: int,
    n_mols_optimize: int,
    sorting_key: str,
    smiles_col: str,
) -> Dict[str, AgentData]:

    results = {}
    for model_name in model_names:
        model_data_folder = model_names[model_name]["data_folder"]

        # Get all data
        start_time = time.time()
        all_dfs, trajs = get_all_dfs(base_dir, eval_formulas, model_data_folder)
        print(f"a) Time taken to get all data: {time.time() - start_time} seconds")

        # Filter data
        start_time = time.time()
        valid_formulas = list(all_dfs.keys())
        dfs_filtered, trajs_filtered = filter_data(
            all_dfs, trajs, valid_formulas, sorting_key, smiles_col, stratify_on_smiles
        )
        print(f"b) Time taken to filter data: {time.time() - start_time} seconds")

        # Find candidate molecules
        if search_for_candidates:
            best_trajs, best_df = find_candidate_mols(
                dfs_filtered, trajs_filtered, valid_formulas, n_query, n_non_query
            )
        else:
            best_trajs, best_df = select_best_mols(dfs_filtered, trajs_filtered, valid_formulas, n_mols_optimize)

        for formula in best_trajs:
            print(f"Number of molecules for {formula}: {len(best_trajs[formula])}")

        sorted_candidates = build_sorted_candidates(best_trajs, best_df, valid_formulas, smiles_col)
        sorted_candidates = {formula: mols[:n_mols] for formula, mols in sorted_candidates.items()}

        # Save images
        # launch_chimerax_jobs(mols_to_view, save_dir, bg_color_str)

        results[model_name] = AgentData(
            tag=model_name,
            visuals_dir=model_names[model_name]["visuals_dir"],
            save_dir=model_data_folder,
            top_k=sorted_candidates,
            all_dfs=all_dfs,
            all_trajs=trajs,
        )

    return results, valid_formulas


def get_visuals_dir(base_dir: str, visuals_dir: str) -> Path:
    return Path(f"{base_dir}/{visuals_dir}")


class ImageGridExp1:
    color_tuples = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "grey": (200, 200, 200),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
    }

    def __init__(
        self,
        agent_data: List[AgentData],
        n_mols: int,
        save_dir: str,
        eval_formulas: List[str],
        bg_color_str: str,
        smiles_col: str,
        unit: str,
        hist_label: str,
    ):
        self.colors = ["#00bfff", "red"]  # Using deep sky blue instead of regular blue

        self.save_dir = save_dir
        self.agent_data = agent_data
        self.eval_formulas = eval_formulas
        self.n_mols = n_mols
        self.n_rows = len(self.eval_formulas)
        self.n_cols = n_mols * 2 + 1
        self.unit = unit
        self.hist_label = hist_label

        self.smiles_col = smiles_col
        self.img_width = 600
        self.img_height = 600
        self.line_width = 2
        self.gap_height = 20
        self.top_offset = 10

        self.font = ImageFont.truetype(
            "DejaVuSans.ttf", int(np.round(32 * (self.img_width / 300)))
        )  # Increased font size
        self.font_small = ImageFont.truetype("DejaVuSans.ttf", int(np.round(20 * (self.img_width / 300))))
        self.big_bold_font = ImageFont.truetype("DejaVuSans.ttf", int(np.round(40 * (self.img_width / 300))))

        line_color_str = "white" if bg_color_str == "black" else "black"
        self.bg_color_str = bg_color_str
        self.bg_color = self.color_tuples[bg_color_str]  # switch to tuple format
        self.line_color = self.color_tuples[line_color_str]
        self.opposite_line_color = (255 - self.line_color[0], 255 - self.line_color[1], 255 - self.line_color[2])

        self.image_grid = Image.new(
            "RGB",
            (self.n_cols * self.img_width, self.n_rows * (self.img_height + self.gap_height)),
            color=self.bg_color,
        )
        self.draw = ImageDraw.Draw(self.image_grid)

    @staticmethod
    def get_image(path_name):
        return Image.open(path_name)

    def insert_image_into_grid(self, j, i, img, offset):
        img = img.resize((self.img_width, self.img_height))
        self.image_grid.paste(img, (j * self.img_width, i * (self.img_height + self.gap_height) + offset))

    def insert_black_image_into_grid(self, j, i):
        img = Image.new("RGB", (self.img_width, self.img_height), color=self.bg_color)
        self.image_grid.paste(img, (j * self.img_width, i * (self.img_height + self.gap_height) + self.top_offset))

    def draw_smiles_text(self, j, i, smiles_text, offset=0):
        """Write SMILES on image"""
        x = j * self.img_width + 0.03 * self.img_width
        y = (i - 0.02) * (self.img_height + self.gap_height) + offset
        self.draw.text((x, y), smiles_text, fill=self.line_color, font=self.font_small)

    def draw_text(self, j, i, value, offset=0):
        """Write value on image"""
        text = f"p: {value:.3f} {self.unit}"
        debye_value = value / Debye
        text = text + f"={debye_value:.3f} D"

        factor = self.img_width / 300

        # First draw a small rectangle background for the energy text
        box_width = len(text) * 11 * factor
        box_start_x = j * self.img_width + 0.03 * self.img_width

        box_height = 20 * factor
        box_start_y = (i - 0.02) * (self.img_height + self.gap_height) + self.top_offset + 24 * factor + offset
        self.draw.rectangle(
            (box_start_x, box_start_y, box_start_x + box_width, box_start_y + box_height),
            fill=(*self.opposite_line_color, 128),
        )

        # Then write the text on top
        text_x = j * self.img_width + 0.03 * self.img_width
        text_y = (i - 0.02) * (self.img_height + self.gap_height) + self.top_offset + 20 * factor + offset
        self.draw.text((text_x, text_y), text, fill=self.line_color, font=self.font_small)

    def draw_lines(self, i):
        """Draw horizontal line between rows."""
        if i < self.n_rows - 1:
            self.draw.line(
                [
                    (0, (i + 1 - 0.05) * (self.img_height + self.gap_height) + self.top_offset),
                    (
                        self.n_cols * self.img_width,
                        (i + 1 - 0.05) * (self.img_height + self.gap_height) + self.top_offset,
                    ),
                ],
                width=self.line_width,
                fill=self.line_color,
            )

    def plot_molecules(self):
        for i in range(self.n_rows):
            formula = self.eval_formulas[i]

            start_col = 0
            for agent_data in self.agent_data:
                for j, j_col in enumerate(range(start_col, start_col + self.n_mols)):
                    path_name = get_visuals_dir(self.save_dir, agent_data.visuals_dir) / f"{formula}_{j}.png"
                    if os.path.exists(path_name):
                        self.insert_image_into_grid(j_col, i, self.get_image(path_name), self.top_offset + 20)

                        value_offset = 20 if i == 0 else 0
                        self.draw_text(
                            j_col, i, value=agent_data.top_k[formula][j].dipole_relaxed_f5, offset=value_offset
                        )

                        smiles_offset = 20 if i == 0 else 0
                        self.draw_smiles_text(
                            j_col, i, smiles_text=agent_data.top_k[formula][j].smiles, offset=smiles_offset
                        )
                    else:
                        self.insert_black_image_into_grid(j, i)

                start_col += self.n_mols + 1

            self.draw_lines(i)

    def save_image_grid(self, file_name: str):
        self.image_grid.save(f"{self.save_dir}/{file_name}", dpi=(300, 300), quality=95)

    def plot_histograms(self):
        import matplotlib.pyplot as plt
        import seaborn as sns  # only if you want nicer style defaults

        sns.set_context("talk", font_scale=0.9)
        dark = self.bg_color_str == "black"
        plt.style.use("dark_background" if dark else "default")

        text_color = "white" if dark else "black"

        j = self.n_mols  # Place histogram in center column

        for i in range(self.n_rows):
            formula = self.eval_formulas[i]
            pretty_formula = build_eval_formulas_pretty()[i]

            # Create figure with high-quality resolution
            fig_bg = "black" if dark else "white"
            fig, ax = plt.subplots(figsize=(6, 5), dpi=300, facecolor=fig_bg)
            ax.set_facecolor(fig_bg)

            # Define monochrome colors for elegance
            alphas = [1.0, 0.70]

            # Get all energies to determine common bin range
            # Use relaxed dipole if available, fall back to unrelaxed.
            # Only include molecules that passed the SMILES filter (NEW_SMILES not null),
            # i.e. the same population that can actually be selected for display.
            dipole_col = (
                "dipole_relaxed" if "dipole_relaxed" in self.agent_data[0].all_dfs[formula].columns else "dipole"
            )
            all_values = np.concatenate(
                [
                    agent_data.all_dfs[formula][agent_data.all_dfs[formula][self.smiles_col].notna()][
                        dipole_col
                    ].dropna()
                    for agent_data in self.agent_data
                ]
            )

            # Create common bins based on all data
            min_value = min(all_values)
            max_value = max(all_values)
            bins = np.linspace(min_value, max_value, 51)  # 20 bins + 1 edge

            for idx, agent_data in enumerate(self.agent_data):
                df_valid = agent_data.all_dfs[formula][agent_data.all_dfs[formula][self.smiles_col].notna()]
                dipole = df_valid[dipole_col].dropna()

                ax.hist(
                    dipole,
                    bins=bins,
                    edgecolor="black",
                    alpha=alphas[idx],
                    color=self.colors[idx],
                    label=agent_data.tag,
                )
            # Set descriptive labels and stylish title
            if i == self.n_rows - 1:
                ax.set_xlabel(self.hist_label, fontsize=24, color=text_color, labelpad=10)

            ax.set_ylabel("Frequency", fontsize=24, color=text_color, labelpad=10)
            common_name = FORMULA_COMMON_NAMES.get(formula, "")
            title = f"{pretty_formula} \u2014 {common_name}" if common_name else pretty_formula
            ax.set_title(title, fontsize=26, color=text_color, pad=15)

            # Enhance ticks and spines visibility
            ax.tick_params(colors=text_color, direction="out", length=6, width=1.5)
            for spine in ax.spines.values():
                spine.set_edgecolor(text_color)

            # Refined legend appearance
            legend = ax.legend(frameon=False, fontsize=18, loc="upper left")
            for text in legend.get_texts():
                text.set_color(text_color)

            # Ensure neat layout
            plt.tight_layout(pad=2.0)

            # Save histogram as image with transparency
            path_name = os.path.join(self.save_dir, f"hist_{formula}.png")
            plt.savefig(path_name, facecolor=fig_bg, bbox_inches="tight")

            # Insert image into grid
            self.insert_image_into_grid(j, i, self.get_image(path_name), offset=-10 if i != 0 else 5)

    @staticmethod
    def _best_dipole_per_smiles(df: pd.DataFrame, smiles_col: str, dipole_col: str) -> pd.Series:
        """Return the single best (max) dipole for each unique SMILES."""
        valid = df[df[smiles_col].notna()]
        return valid.groupby(smiles_col)[dipole_col].max()

    def save_histogram_figure(self, file_name: str, unrelaxed_dipoles: List[Dict[str, np.ndarray]] = None):
        """Save a standalone 3-column figure per formula row:
          col 0 – unrelaxed dipoles (from cached npy, valid samples only)
          col 1 – relaxed dipoles   (df.dipole_relaxed, valid samples only)
          col 2 – best relaxed dipole per unique SMILES
        If unrelaxed_dipoles is None, falls back to a 2-column figure (cols 1 & 2).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_context("talk", font_scale=0.9)
        dark = self.bg_color_str == "black"
        plt.style.use("dark_background" if dark else "default")
        text_color = "white" if dark else "black"
        fig_bg = "black" if dark else "white"

        have_unrelaxed = unrelaxed_dipoles is not None
        n_cols_fig = 3 if have_unrelaxed else 2
        col_titles = (
            ["Unrelaxed dipoles", "Relaxed dipoles", "Best dipole per isomer"]
            if have_unrelaxed
            else ["Relaxed dipoles", "Best dipole per isomer"]
        )

        fig, axes = plt.subplots(
            self.n_rows,
            n_cols_fig,
            figsize=(6 * n_cols_fig, 2.25 * self.n_rows),
            dpi=200,
            facecolor=fig_bg,
        )
        if self.n_rows == 1:
            axes = axes[np.newaxis, :]

        alphas = [1.0, 0.70]
        pretty_formulas = build_eval_formulas_pretty()

        for i, formula in enumerate(self.eval_formulas):
            dipole_col = (
                "dipole_relaxed" if "dipole_relaxed" in self.agent_data[0].all_dfs[formula].columns else "dipole"
            )
            common_name = FORMULA_COMMON_NAMES.get(formula, "")
            title_base = f"{pretty_formulas[i]} \u2014 {common_name}" if common_name else pretty_formulas[i]

            # Build shared bin range from all data sources and agents
            all_values = []
            for a_idx, ad in enumerate(self.agent_data):
                df = ad.all_dfs[formula]
                valid_mask = df[self.smiles_col].notna()
                all_values.append(df.loc[valid_mask, dipole_col].dropna().values)
                if have_unrelaxed:
                    unrel = unrelaxed_dipoles[a_idx].get(formula)
                    if unrel is not None:
                        all_values.append(unrel[valid_mask.values][~np.isnan(unrel[valid_mask.values])])
            combined = np.concatenate(all_values)
            bins = np.linspace(combined.min(), combined.max(), 31)

            for col_idx, col_title in enumerate(col_titles):
                ax = axes[i, col_idx]
                ax.set_facecolor(fig_bg)

                for a_idx, ad in enumerate(self.agent_data):
                    df = ad.all_dfs[formula]
                    valid_mask = df[self.smiles_col].notna()

                    if col_title == "Unrelaxed dipoles":
                        unrel = unrelaxed_dipoles[a_idx].get(formula)
                        if unrel is None:
                            continue
                        values = pd.Series(unrel)[valid_mask.values].dropna()
                    elif col_title == "Relaxed dipoles":
                        values = df.loc[valid_mask, dipole_col].dropna()
                    else:  # Best dipole per isomer
                        values = self._best_dipole_per_smiles(df, self.smiles_col, dipole_col)

                    ax.hist(
                        values,
                        bins=bins,
                        edgecolor="black",
                        alpha=alphas[a_idx],
                        color=self.colors[a_idx],
                        label=ad.tag,
                    )

                mid = n_cols_fig // 2
                if i == 0:
                    ax.set_title(col_title, fontsize=22, fontweight="bold", color=text_color, pad=30)
                    if col_idx == mid:
                        ax.text(
                            0.5,
                            1.01,
                            title_base,
                            transform=ax.transAxes,
                            ha="center",
                            va="bottom",
                            fontsize=15,
                            color=text_color,
                        )
                else:
                    if col_idx == mid:
                        ax.set_title(title_base, fontsize=15, color=text_color, pad=8)
                    else:
                        ax.set_title("", pad=8)
                if i == self.n_rows - 1:
                    ax.set_xlabel(self.hist_label, fontsize=13, color=text_color)
                if col_idx == 0:
                    ax.set_ylabel("Frequency", fontsize=13, color=text_color)
                ax.tick_params(colors=text_color, direction="out", length=5, width=1.2)
                for spine in ax.spines.values():
                    spine.set_edgecolor(text_color)
                legend = ax.legend(frameon=False, fontsize=12, loc="upper left")
                for t in legend.get_texts():
                    t.set_color(text_color)

        plt.tight_layout(pad=2.5)
        out_path = os.path.join(self.save_dir, file_name)
        plt.savefig(out_path, facecolor=fig_bg, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"Saved histogram figure to {out_path}")

    def draw_outer_box(self):

        define_manally = False
        if define_manally:
            num_pixels_x = self.n_cols * self.img_width
            num_pixels_y = self.n_rows * (self.img_height + self.gap_height) + self.top_offset
        else:
            num_pixels_x = self.image_grid.size[0]
            num_pixels_y = self.image_grid.size[1]

        self.draw.rectangle((0, 0, num_pixels_x, num_pixels_y), fill=None, outline=self.line_color, width=7)

    def draw_boxes_around_agents(self):
        left1 = 0
        right1 = self.n_mols * self.img_width

        left2 = (self.n_mols + 1) * self.img_width
        right2 = self.image_grid.size[0]

        bottom = self.image_grid.size[1]

        self.draw.rectangle((left1, 0, right1, bottom), fill=None, outline=self.colors[0], width=15)
        self.draw.rectangle((left2, 0, right2, bottom), fill=None, outline=self.colors[1], width=15)

    def write_agent_names(self):

        # Just above bottom
        y = self.image_grid.size[1] - 125

        x1 = (self.n_mols * self.img_width) / 2

        x_end = self.image_grid.size[0]
        x2 = x_end - x1 - 200

        for idx, (agent_data, x_pos) in enumerate(zip(self.agent_data, [x1, x2])):
            self.draw.text((x_pos, y), agent_data.tag, fill=self.colors[idx], font=self.big_bold_font)


if __name__ == "__main__":
    # Define the experiment parameters
    base_dir = "runs/A-30k-Fixed/seed_0"
    tag = "EXP5_30000"
    n_seeds = 1
    model_names = {
        "Pretrained": {
            "data_folder": "monday-30k-dip-30000",
            "model_obj": "pretrain_run-0_steps-30000.model",
            "visuals_dir": "exp5_white_dipole_relaxed_monday-30k-dip-30000",
        },
        "Finetuned": {
            "data_folder": "monday-30k-dip-30900",
            "model_obj": "pretrain_run-0_steps-30900.model",
            "visuals_dir": "exp5_white_dipole_relaxed_monday-30k-dip-30900",
        },
    }
    eval_formulas = default_eval_formulas()

    stratify_on_smiles = True  # absolutely necessary
    bg_color_str = "white"

    # Which molecules to view?
    search_for_candidates = False
    n_query = None  # n_mols matching the query
    n_non_query = None  # n_mols not matching the query
    n_mols = 3  # Number of molecules shown in the plot
    n_mols_optimize = 20  # Larger pool relaxed before selecting top n_mols

    sorting_key = "dipole_relaxed"
    smiles_col = "NEW_SMILES"

    # Create save directory
    illustration_dir_name = "exp5_grid_figure"

    save_dir = os.path.join(base_dir, illustration_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results, eval_formulas = get_agent_data(
        base_dir=base_dir,
        model_names=model_names,
        eval_formulas=eval_formulas,
        stratify_on_smiles=stratify_on_smiles,
        search_for_candidates=search_for_candidates,
        n_query=n_query,
        n_non_query=n_non_query,
        n_mols=n_mols,
        n_mols_optimize=n_mols_optimize,
        sorting_key=sorting_key,
        smiles_col=smiles_col,
    )

    # Compute and cache unrelaxed dipoles for all model folders
    for model_name, model_cfg in model_names.items():
        print(f"\n--- Unrelaxed dipoles: {model_name} ---")
        compute_and_cache_unrelaxed_dipoles(base_dir, model_cfg["data_folder"], eval_formulas)

    agent_list = [agent_data for agent_data in results.values()]

    use_dipole = True
    if use_dipole:
        unit = "e·Å"
        hist_label = r"Dipole ($e\,\mathrm{\AA}$)"
    else:
        unit = "eV / atom"
        hist_label = "Relaxed Energy (eV / atom)"

    grid = ImageGridExp1(
        agent_data=agent_list,
        n_mols=n_mols,
        save_dir=base_dir,
        eval_formulas=eval_formulas,
        bg_color_str=bg_color_str,
        smiles_col=smiles_col,
        unit=unit,
        hist_label=hist_label,
    )
    grid.plot_histograms()
    grid.plot_molecules()
    grid.draw_outer_box()
    unrelaxed_dipoles_by_agent = [
        load_cached_unrelaxed_dipoles(base_dir, model_cfg["data_folder"], eval_formulas)
        for model_cfg in model_names.values()
    ]
    hist_fig_name = build_comparison_image_name(prefix="hist_fig_", bg_color_str=bg_color_str, model_names=model_names)
    grid.save_histogram_figure(file_name=hist_fig_name, unrelaxed_dipoles=unrelaxed_dipoles_by_agent)
    # grid.write_agent_names()
    # grid.draw_boxes_around_agents()
    output_file_name = build_comparison_image_name(
        prefix="supplementary_fig", bg_color_str=bg_color_str, model_names=model_names
    )
    grid.save_image_grid(file_name=output_file_name)
