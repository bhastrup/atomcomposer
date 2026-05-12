from typing import List, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from ase import Atoms

from src.performance.xyz2mol import xyz2mol
from src.performance.energetics import EnergyUnit, XTBOptimizer
from src.tools.util import symbols_to_str_formula
from src.performance.utils import no_print_wrapper


def get_compact_smiles(smiles: str) -> str:
    """Get the compact smiles representation"""
    new_mol_pos_free = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(new_mol_pos_free, isomericSmiles=False, canonical=True)


def check_charge_neutrality(mol: Chem.Mol) -> bool:
    """Check if the RDKit molecule object has any charged or radical atoms."""
    if mol:
        for atom in mol.GetAtoms():
            formal_charge = atom.GetFormalCharge()
            num_radical_electrons = atom.GetNumRadicalElectrons()
            if formal_charge != 0 or num_radical_electrons != 0:
                return False
    return True


def _get_mol(atoms: Atoms, use_huckel: bool = True) -> dict:
    coordinates = atoms.get_positions()
    atom_nums = atoms.get_atomic_numbers().tolist()

    try:
        molecule = xyz2mol(
            atom_nums,
            coordinates,
            charge=0,
            allow_charged_fragments=False,
            use_graph=True,
            use_huckel=use_huckel,
            embed_chiral=True,
            use_atom_maps=False,
        )
    except Exception:
        return {"mol": None, "info": "crashed"}

    if not molecule:
        return {"mol": None, "info": "failed"}

    if len(Chem.GetMolFrags(molecule[0])) != 1:
        return {"mol": None, "info": "fragmented"}

    if not check_charge_neutrality(molecule[0]):
        return {"mol": None, "info": "charged_or_radical"}

    return {"mol": molecule[0], "info": "valid"}


def get_mol(atoms: Atoms, use_huckel: bool = True) -> dict:
    """Wrapper around _get_mol to try first with Huckel and then without (unless Huckel==False)"""
    if use_huckel:
        mol_info = no_print_wrapper(_get_mol, atoms, use_huckel=True)
        if mol_info["mol"]:
            return mol_info

    return no_print_wrapper(_get_mol, atoms, use_huckel=False)


def calc_rae(energy: np.ndarray, benchmark_energy: float, n_atoms: int) -> float:
    return (energy - benchmark_energy) / n_atoms


class MoleculeAnalyzer:
    """Handles molecular analysis and conversion operations."""

    def __init__(self, use_huckel: bool = True):
        self.use_huckel = use_huckel

    @staticmethod
    def get_compact_smiles(smiles: str) -> str:
        mol_pos_free = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol_pos_free, isomericSmiles=False, canonical=True)

    @staticmethod
    def check_charge_neutrality(mol: Chem.Mol) -> bool:
        if not mol:
            return False
        return all(atom.GetFormalCharge() == 0 and atom.GetNumRadicalElectrons() == 0 for atom in mol.GetAtoms())

    def get_mol(self, atoms: Atoms) -> dict:
        """Tries to convert Atoms to RDKit molecule, first with Huckel if enabled."""
        if self.use_huckel:
            mol_info = no_print_wrapper(self._get_mol, atoms, use_huckel=True)
            if mol_info["mol"]:
                return mol_info
        return no_print_wrapper(self._get_mol, atoms, use_huckel=False)

    def _get_mol(self, atoms: Atoms, use_huckel: bool) -> dict:
        coordinates = atoms.get_positions()
        atom_nums = atoms.get_atomic_numbers().tolist()

        try:
            molecule = xyz2mol(
                atom_nums,
                coordinates,
                charge=0,
                allow_charged_fragments=False,
                use_graph=True,
                use_huckel=use_huckel,
                embed_chiral=True,
                use_atom_maps=False,
            )
        except Exception:
            return {"mol": None, "info": "crashed"}

        if not molecule:
            return {"mol": None, "info": "failed"}

        if len(Chem.GetMolFrags(molecule[0])) != 1:
            return {"mol": None, "info": "fragmented"}

        if not self.check_charge_neutrality(molecule[0]):
            return {"mol": None, "info": "charged_or_radical"}

        return {"mol": molecule[0], "info": "valid"}


class MoleculeProcessor:
    """Handles processing of molecular structures and energy calculations."""

    def __init__(self, use_huckel: bool = True):
        self.analyzer = MoleculeAnalyzer(use_huckel)
        self.features_dict = {
            "valid": bool,
            "SMILES": str,
            "n_rings": int,
            "n_atoms_ring_max": int,
            "charge_fail": bool,
            "abs_energy": float,
            "dipole": float,
            "rae": float,
            "NEW_SMILES": str,
            "relax_stable": bool,
            "basin_distance": float,
            "RMSD": float,
            "e_relaxed": float,
            "rae_relaxed": float,
            "dipole_relaxed": float,
        }

    @staticmethod
    def calc_rae(energy: np.ndarray, benchmark_energy: float, n_atoms: int) -> float:
        return (energy - benchmark_energy) / n_atoms

    def do_calc_rae_fn(self, args: dict) -> bool:
        return "benchmark_energies" in args and args["benchmark_energies"] is not None

    def process_atoms(self, args: dict) -> Tuple[dict, dict]:
        energy_unit = EnergyUnit.EV

        atoms = args["atoms"] if "atoms" in args else args["atoms_object_list"][args["index"]].copy()
        perform_optimization = args["perform_optimization"]
        calc_dipole = args["calc_dipole"]
        f_max = args["f_max"]

        mol_info = self.analyzer.get_mol(atoms)
        valid = mol_info["info"] == "valid"
        mol = mol_info["mol"]
        charge_fail = True if mol_info["info"] == "charged_or_radical" else False

        # Get the number of rings and the size of the biggest ring
        n_rings = 0
        biggest_ring = 0
        if mol is not None:
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            n_rings = len(atom_rings)
            num_atoms_in_rings = [len(ring) for ring in atom_rings]
            biggest_ring = max(num_atoms_in_rings) if num_atoms_in_rings else 0

        # Calculate the absolute energy and RAE and dipole
        calc = XTBOptimizer(method="GFN2-xTB", energy_unit=energy_unit)
        abs_energy = calc.calc_potential_energy(atoms)
        rae = None
        do_calc_rae = self.do_calc_rae_fn(args) and abs_energy is not None
        if do_calc_rae:
            benchmark_energy = args["benchmark_energies"][symbols_to_str_formula(atoms.symbols)]
            rae = self.calc_rae(energy=abs_energy, benchmark_energy=benchmark_energy, n_atoms=len(atoms))
        dipole = None
        if calc_dipole and not perform_optimization:
            dipole = calc.calc_dipole(atoms)

        # Get the SMILES representation
        SMILES = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True) if mol else None
        mol_pos_free = Chem.MolFromSmiles(SMILES) if SMILES is not None else None
        SMILES_Compact = Chem.MolToSmiles(mol_pos_free, isomericSmiles=False, canonical=True) if mol_pos_free else None

        # Perform optimization
        NEW_SMILES = None
        new_SMILES_Compact = None
        new_mol = None
        new_atoms = None
        relax_stable = None
        basin_distance = None
        RMSD = None
        e_relaxed = None
        rae_relaxed = None
        dipole_relaxed = None

        if valid and perform_optimization:
            opt_info = calc.optimize_atoms(atoms, max_steps=200, fmax=f_max)
            new_atoms = opt_info["new_atoms"]

            new_mol_info = self.analyzer.get_mol(new_atoms)
            new_valid = new_mol_info["info"] == "valid"
            new_mol = new_mol_info["mol"]

            NEW_SMILES = Chem.MolToSmiles(new_mol, canonical=True) if new_mol else None
            relax_stable = True if new_valid and NEW_SMILES == SMILES else False

            new_mol_pos_free = Chem.MolFromSmiles(NEW_SMILES) if NEW_SMILES is not None else None
            new_SMILES_Compact = (
                Chem.MolToSmiles(new_mol_pos_free, isomericSmiles=False, canonical=True) if new_mol_pos_free else None
            )

            e_relaxed = opt_info["energy_after"]
            if do_calc_rae:
                rae_relaxed = self.calc_rae(
                    energy=opt_info["energy_after"], benchmark_energy=benchmark_energy, n_atoms=len(atoms)
                )
            if calc_dipole and new_atoms is not None:
                dipole_relaxed = calc.calc_dipole(new_atoms)
            if valid and relax_stable and opt_info["converged"]:
                basin_distance = abs(opt_info["energy_after"] - opt_info["energy_before"])
                RMSD = Chem.rdMolAlign.GetBestRMS(mol, new_mol)

        del calc

        return {
            "valid": valid,
            "SMILES": SMILES_Compact,
            "n_rings": n_rings,
            "n_atoms_ring_max": biggest_ring,
            "charge_fail": charge_fail,
            "abs_energy": abs_energy,
            "dipole": dipole,
            "rae": rae,
            "NEW_SMILES": new_SMILES_Compact,
            "relax_stable": relax_stable,
            "basin_distance": basin_distance,
            "RMSD": RMSD,
            "e_relaxed": e_relaxed,
            "rae_relaxed": rae_relaxed,
            "dipole_relaxed": dipole_relaxed,
        }, {
            "mol": mol,
            "new_mol": new_mol,
            "new_atoms": new_atoms,
        }

    def atom_list_to_df(
        self,
        atoms_object_list: List[Atoms] = None,
        benchmark_energies: dict = None,
        perform_optimization: bool = False,
        calc_dipole: bool = False,
        f_max: float = 0.025,
    ) -> Tuple[pd.DataFrame, List[dict]]:

        feature_cols = list(self.features_dict.keys())
        df = pd.DataFrame(columns=feature_cols)

        for col, dtype in self.features_dict.items():
            df[col] = df[col].astype(dtype)

        if not atoms_object_list:
            return df, []

        stats_list = []
        extra_data_list = []

        # Process each atom object sequentially
        for atoms in atoms_object_list:
            stats, extra_data = self.process_atoms(
                {
                    "atoms": atoms,
                    "benchmark_energies": benchmark_energies,
                    "perform_optimization": perform_optimization,
                    "use_huckel": self.analyzer.use_huckel,
                    "calc_dipole": calc_dipole,
                    "f_max": f_max,
                }
            )
            stats_list.append(stats)
            extra_data_list.append(extra_data)

        # Combine the statistics into a DataFrame
        df = pd.concat([pd.DataFrame([list(stats.values())], columns=feature_cols) for stats in stats_list])

        return df, extra_data_list
