from typing import List
import streamlit as st

from ase import Atoms
from ase.visualize import view
from rdkit import Chem

from src.rl.reward import InteractionReward

from src.performance.metrics import get_mol
from rdkit.Chem import Draw


def get_single_energy(atoms):
    return InteractionReward()._calculate_energy(atoms)


def view_atoms_from_list(index, atoms_list):
    return view(atoms_list[index], viewer="ase")


def row_to_atoms(row):
    return Atoms(symbols=row["atomic_symbols"], positions=row["pos"])


def row_to_energy(row):
    return get_single_energy(row_to_atoms(row))


def view_rdkit_mol(index: int, atoms_list: List[Atoms]):
    atoms = atoms_list[index]

    mol = get_mol(atoms)["mol"]
    if mol is None:
        st.write("No molecule found")
        return None

    cols = st.columns([1, 1])
    with cols[0]:
        img = Draw.MolToImage(mol)
        st.image(img, use_column_width=True)
    with cols[1]:
        SMILES = Chem.MolToSmiles(mol) if mol else None
        mol_pos_free = Chem.MolFromSmiles(SMILES) if SMILES is not None else None
        img = Draw.MolToImage(mol_pos_free)
        st.image(img, use_column_width=True)


def view_atoms_from_list_sequence(index, atoms_list: List[Atoms]):
    """Select the atoms object and decompose it into a list of atoms objects,
    starting from the first atom and ending at the full molecules."""
    atoms = atoms_list[index]
    atoms_sequence = [atoms[:i] for i in range(1, len(atoms) + 1)]
    view(atoms_sequence, viewer="ase")


def view_bag_conformers_fn(index: int, fn_args: dict, sort_by_energy: bool = True):
    bag_df = fn_args["bag_df"]
    df = fn_args["df"]
    sort_by_energy = fn_args["sort_by_energy"]

    bag_repr = bag_df.iloc[index]["bag_repr"]
    df_filtered = df[df["bag_repr"] == bag_repr].copy()

    if sort_by_energy:
        if "energy_GFN2" not in df_filtered.columns:
            df_filtered["energy_GFN2"] = df_filtered.apply(row_to_energy, axis=1)
        df_filtered.sort_values(by="energy_GFN2", inplace=True)

    all_atoms = []
    for i in range(len(df_filtered)):
        atoms = Atoms(symbols=df_filtered.iloc[i]["atomic_symbols"], positions=df_filtered.iloc[i]["pos"])
        all_atoms.append(atoms)

    view(all_atoms)
