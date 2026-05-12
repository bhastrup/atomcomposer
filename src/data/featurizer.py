from typing import Dict, Any, Tuple

import numpy as np
from rdkit import Chem
from ase import Atoms

from src.performance.energetics import EnergyUnit, XTBOptimizer
from src.performance.metrics import get_mol
from src.data.data_util import get_con_upper_triangular, elements_to_symbols, syms_to_count_dict, sym_count_to_bag_repr


def get_both_SMILES(mol) -> Tuple[str, str]:
    if mol is not None:
        SMILES = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        mol_pos_free = Chem.MolFromSmiles(SMILES)
        smiles_compact = Chem.MolToSmiles(mol_pos_free, isomericSmiles=False, canonical=True)
        return SMILES, smiles_compact
    return None, None


class Featurizer:
    def __init__(
        self,
        include_energy: bool,
        include_smiles: bool,
        include_connectivity: bool,
        use_huckel: bool,
        smiles_compatible: bool = True,
    ):
        self.include_energy = include_energy
        self.include_smiles = include_smiles
        self.include_connectivity = include_connectivity
        self.use_huckel = use_huckel
        self.smiles_compatible = smiles_compatible
        self.energy_unit = EnergyUnit.EV

        self.meta_data = {
            "include_energy": include_energy,
            "include_smiles": include_smiles,
            "include_connectivity": include_connectivity,
            "use_huckel": use_huckel,
            "energy_unit": self.energy_unit.value,
        }

    def calc_features(self, i: int, data: dict) -> Tuple[Dict[str, Any], str]:
        atom_pos = data["atom_pos"]
        atom_nums = data["atom_nums"]
        atom_syms = data["atom_syms"]

        used_keys = ["atom_pos", "atom_nums", "atom_syms"]

        # Bag info
        elements_unique = list(sorted(set(atom_nums)))
        symbols_sorted = elements_to_symbols(elements_unique)
        counts_sorted = syms_to_count_dict(atom_syms, symbols_sorted)
        bag_repr = sym_count_to_bag_repr(counts_sorted)

        # Energy
        atoms = Atoms(atom_syms, positions=atom_pos)
        calc = XTBOptimizer(energy_unit=self.energy_unit)
        energy_GFN2 = calc.calc_potential_energy(atoms) if self.include_energy else None

        # Molecule info
        if self.smiles_compatible:
            mol_info = get_mol(atoms, use_huckel=self.use_huckel)
            mol = mol_info["mol"]
        else:
            mol = None
            mol_info = {"info": "valid"} if energy_GFN2 is not None else {"info": "failed_energy"}

        con_upper_triangular, SMILES, SMILES_Compact = None, None, None
        if mol is not None:
            if self.include_connectivity:
                con_upper_triangular = get_con_upper_triangular(mol)

            if self.include_smiles:
                SMILES, SMILES_Compact = get_both_SMILES(mol)
                # assert smiles0 == SMILES, f'SMILES mismatch: {smiles0} != {SMILES}'

        data_dict = {
            "molecule_id": i,
            "atomic_symbols": np.array(atom_syms),
            "atomic_nums": atom_nums,
            "pos": atom_pos,
            "elements_unique": elements_unique,
            "symbols_sorted": symbols_sorted,
            "bag_repr": bag_repr,
            "n_atoms": len(atom_syms),
            "energy_GFN2": energy_GFN2,
            "SMILES": SMILES_Compact,
            "smiles": SMILES,
            "con_upper_triangular": con_upper_triangular,
        }

        # Extend data_dict with other keys from data
        data_dict.update({k: v for k, v in data.items() if k not in used_keys})

        return data_dict, mol_info["info"]
