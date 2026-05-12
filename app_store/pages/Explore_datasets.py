from typing import List, Optional, Tuple

import torch
import streamlit as st
import pandas as pd

from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import chemical_symbols, atomic_numbers
from ase.visualize import view
from millify import prettify

from src.data.reference_dataloader import ReferenceDataLoader
from src.performance.energetics import EnergyUnit
from src.pretraining.action_decom import recenter, pos_seq_to_actions, pos_seq_to_actions_emma, decompose_pos
from app_store.pages.tools.images.show_logo import show_logo
from app_store.pages.tools.app_utils import make_multi_button_columns, initialize_session_state
from app_store.pages.tools.visualize import view_bag_conformers_fn
from app_store.pages.tools.visualize import row_to_energy, get_single_energy, row_to_atoms
from app_store.pages.tools.visualize import view_atoms_from_list, view_atoms_from_list_sequence, view_rdkit_mol

# Set page config
st.set_page_config(page_title="Explore dataset", page_icon=":memo:", layout="wide", initial_sidebar_state="expanded")

# Add CSS to change font size
st.markdown(
    """
    <style>
    body {
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_dataset(name: str) -> Tuple[Optional[dict], Optional[pd.DataFrame]]:
    """Load dataset using the reference dataloader."""
    if name not in st.session_state.datasets or name + "_df" not in st.session_state.datasets:
        with st.spinner(f"Loading {name} dataset from file"):
            try:
                loader = ReferenceDataLoader(data_dir="data")
                ref_data = loader.load_and_polish(
                    mol_dataset=name,
                    new_energy_unit=EnergyUnit.EV,
                    fetch_df=True,
                )
                df_full = ref_data.df
                df_full = df_full.reset_index(drop=True)
                dataset_dict = df_full.to_dict("index") if df_full is not None else None
                if df_full is None:
                    return None, None
                st.session_state.datasets[name] = dataset_dict
                st.session_state.datasets[name + "_df"] = df_full
            except Exception:
                return None, None
    else:
        dataset_dict = st.session_state.datasets[name]
        df_full = st.session_state.datasets[name + "_df"]

    return dataset_dict, df_full


def get_bag_df(df: pd.DataFrame):
    """Extract dataframe of unique bags"""
    bag_df = (
        df["bag_repr"].value_counts().rename("isomers").to_frame().reset_index().rename(columns={"index": "bag_repr"})
    )
    bag_df["n_atoms"] = bag_df.bag_repr.map(df.groupby("bag_repr")["n_atoms"].mean())
    return bag_df


def mol_is_already_loaded(dataset_name: str, index: int):
    return True if index in st.session_state.mols[dataset_name] else False


def add_remove_molecule(index: int, df, dataset_name: str):
    if dataset_name not in st.session_state.mols:
        st.session_state.mols[dataset_name] = {}

    if not mol_is_already_loaded(dataset_name, index):
        st.session_state.mols[dataset_name][index] = df.iloc[index]


def get_sum_of_energies(symbols: List[str]):
    energies = {}
    for atom_sym in symbols:
        if atom_sym not in energies.keys():
            atom = Atoms(symbols=[atom_sym], positions=[[0, 0, 0]])
            energies[atom_sym] = get_single_energy(atom)
    sum_energies = sum([energies[sym] for sym in symbols])
    return sum_energies


def df_to_single_atoms(index: int, df: pd.DataFrame):
    row = df.iloc[index]
    return Atoms(symbols=row["atomic_symbols"], positions=row["pos"])


def explore_mols_set_bag_index(index: int, **kwargs) -> None:
    """
    Below has been implemented to mitigate this issue:
        Watch out. Will display something wrong once we change the bag_df.
        Btw, think about storing a dict in session_state which contains various indices and objects in a hierarchy.
        Ex. here, we would store the index in session_state.index_dict.explore_datasets.bag_index.
    """
    bag_df = kwargs["bag_df"]
    df = kwargs["df"]
    bag_repr = bag_df.iloc[index]["bag_repr"]

    df_filtered = df[df["bag_repr"] == bag_repr]
    st.session_state.explore_datasets_mols_indices = df_filtered.index.tolist()

    if "current_dataset_name" not in st.session_state:
        st.session_state.current_dataset_name = kwargs["dataset_name"]

    return None


def explore_mols_decom(index: int, **kwargs) -> None:
    if "loaded_mols_decom" not in st.session_state:
        st.session_state.loaded_mols_decom = pd.DataFrame()

    mol_id = kwargs["df_shown"].iloc[index]["mol_id"]
    new_row = kwargs["df_full"].loc[[mol_id]]
    new_row["dataset_name"] = st.session_state.current_dataset_name
    st.session_state.loaded_mols_decom = pd.concat([st.session_state.loaded_mols_decom, new_row])

    return None


def explore_mols_load_mol(index: int, **kwargs) -> None:
    mol_id = kwargs["df_shown"].iloc[index]["mol_id"]

    # Load into dict
    # if 'loaded_mols' not in st.session_state:
    #     st.session_state.loaded_mols = {}
    # Add to dict
    # st.session_state.loaded_mols[mol_id] = kwargs['atoms_list'][index]

    # Load into df
    if "loaded_mols" not in st.session_state:
        st.session_state.loaded_mols = pd.DataFrame()
        # st.write(f"---- type of st.session_state.loaded_mols: {type(st.session_state.loaded_mols)}")

    # Concat to df
    # st.write(f"type of st.session_state.loaded_mols: {type(st.session_state.loaded_mols)}")
    # st.write(f"type of kwargs['df_full'].loc[[mol_id]]]: {type(kwargs['df_full'].loc[[mol_id]])}")

    new_row = kwargs["df_full"].loc[[mol_id]]

    new_row["dataset_name"] = st.session_state.current_dataset_name

    st.session_state.loaded_mols = pd.concat([st.session_state.loaded_mols, new_row])

    with kwargs["col"]:
        st.balloons()  # st.snow()
        st.success(f"Molecule {mol_id} has been loaded. Find it in PartialCanvasEnv builder", icon="✅")

    return None


def display_bag_mols_df(df: pd.DataFrame, bag_df: pd.DataFrame, sort_by_energy: bool = True, cols=None):

    # The small block just below must be rewritten using st.session_state.explore_datasets_mols_indices

    if "explore_datasets_mols_indices" not in st.session_state:
        st.session_state.explore_datasets_mols_indices = None

    if st.session_state.explore_datasets_mols_indices is None:
        return None

    df_filtered = df.loc[st.session_state.explore_datasets_mols_indices]

    #########
    # index = st.session_state.index_in_explore_mols
    # if index is None:
    #     return None

    # bag_repr = bag_df.iloc[index]['bag_repr']
    # df_filtered = df[df['bag_repr'] == bag_repr]

    ##########

    # st.write(f'Bag {index} with bag_repr {bag_repr} has {len(df_filtered)} conformers')

    if sort_by_energy:
        if "energy_GFN2" not in df_filtered.columns:
            df_filtered["energy_GFN2"] = df_filtered.apply(row_to_energy, axis=1)
        df_filtered.sort_values(by="energy_GFN2", inplace=True)

    symbols = df_filtered["atomic_symbols"].iloc[0]
    df_filtered["delta_E"] = df_filtered["energy_GFN2"].copy() - get_sum_of_energies(symbols)

    with cols[0]:
        # Check if 'SMILES' column exists before accessing it
        if "SMILES" in df_filtered.columns:
            # st.dataframe(df_filtered[['bag_repr', 'n_atoms', 'energy_GFN2', 'delta_E', 'SMILES']])
            # atoms_list ... issue here is that we don't have access to positions and atomic_symbols in this function.
            # Quick fix is to just
            # TODO: Make an ExploreDataset class which contains all the data and functions we need to display and interact with the dataset.

            df_shown = (
                df_filtered[["bag_repr", "n_atoms", "energy_GFN2", "delta_E", "SMILES"]]
                .reset_index(drop=False)
                .rename(columns={"index": "mol_id"})
            )
        else:
            df_shown = (
                df_filtered[["bag_repr", "n_atoms", "energy_GFN2", "delta_E"]]
                .reset_index(drop=False)
                .rename(columns={"index": "mol_id"})
            )

        df_pos_and_symbols = (
            df_filtered[["pos", "atomic_symbols"]].reset_index(drop=False).rename(columns={"index": "mol_id"})
        )
        atoms_list = [row_to_atoms(row) for _, row in df_pos_and_symbols.iterrows()]

        def view_rdkit_on_the_right_wrapper(index, atoms_list, col):
            with col:
                view_rdkit_mol(index, atoms_list)

        make_multi_button_columns(
            df=df_shown,
            button_configs=[
                # ('View', lambda index, df: view(df_to_single_atoms(index, df)), {'df': df_pos_and_symbols}),
                ("View", view_atoms_from_list, {"atoms_list": atoms_list}),
                ("View Seq", view_atoms_from_list_sequence, {"atoms_list": atoms_list}),
                ("Decom", explore_mols_decom, {"df_shown": df_shown, "df_full": df}),
                ("View RdKit", view_rdkit_on_the_right_wrapper, {"atoms_list": atoms_list, "col": cols[1]}),
                (
                    "Load",
                    explore_mols_load_mol,
                    {"df_shown": df_shown, "df_full": df, "atoms_list": atoms_list, "col": cols[1]},
                ),
            ],
            unique_key="explore_datasets_mols_df",
        )


def view_decom_single(
    index, atoms_list, decom_params: dict, decom_method="bfs", cutoff=1.5, shuffle=True, mega_shuffle=False
):

    action_decomposition_fn = decom_params["act_fn"]
    hydrogen_delay = decom_params["hydrogen_delay"]
    no_hydrogen_focus = decom_params["no_hydrogen_focus"]
    show_focus = decom_params["show_focus"]

    atoms = atoms_list[index]

    pos = atoms.get_positions()
    elements = atoms.get_atomic_numbers()

    st.write(f"elements: {elements}")
    st.write(f"pos: {pos}")
    pos_tensor = torch.tensor(pos, dtype=torch.float32)
    st.write(f"pos_tensor: {pos_tensor}")

    pos = recenter(pos, elements, None, st.session_state.current_dataset_name, heavy_first=False)
    sorted_indices = decompose_pos(
        elements,
        pos,
        decom_method=decom_method,
        cutoff=cutoff,
        shuffle=shuffle,
        mega_shuffle=mega_shuffle,
        hydrogen_delay=hydrogen_delay,
    )

    # reorder atoms
    pos = pos[sorted_indices]
    # pos = rotate_to_axis(pos, atom_index=sorted_indices[1], axis='z')
    elements = elements[sorted_indices]
    atoms = Atoms(symbols=elements, positions=pos)

    atoms_sequence = [atoms[:i] for i in range(1, len(atoms) + 1)]

    if show_focus:
        if st.session_state.current_dataset_name == "QM7":
            zs = [0, 1, 6, 7, 8, 16]
        elif st.session_state.current_dataset_name == "QM9":
            zs = [0, 1, 6, 7, 8, 9]
        else:
            st.write("Must be either QM7 or QM9")
            return None

        actions = action_decomposition_fn(pos, elements, zs, no_hydro_focus=no_hydrogen_focus)

        st.write(f"focuses: {actions[:, 1]}")

        for i, atoms in enumerate(atoms_sequence):
            if i == 0:
                continue
            focus_atom = actions[i - 1, 1]
            new_constraint = FixAtoms(mask=[True if i == focus_atom else False for i in range(len(atoms))])
            atoms.set_constraint(new_constraint)

    view(atoms_sequence, viewer="ase")


def view_decom_all(
    index, atoms_list, decom_params: dict, decom_method="bfs", cutoff=1.5, shuffle=True, mega_shuffle=False
):

    hydrogen_delay = decom_params["hydrogen_delay"]

    if st.session_state.update_multi_decom_view:
        # Find all decompositions (sorted_indices)
        atoms = atoms_list[index]

        pos = atoms.get_positions()
        elements = atoms.get_atomic_numbers()

        sorted_indices_dict = {}
        max_count = 100
        for i in range(max_count):
            pos = recenter(pos, elements, None, st.session_state.current_dataset_name, heavy_first=False)
            sorted_indices = decompose_pos(
                elements,
                pos,
                decom_method=decom_method,
                cutoff=cutoff,
                shuffle=shuffle,
                mega_shuffle=mega_shuffle,
                hydrogen_delay=hydrogen_delay,
            )
            if sorted_indices is None:
                continue

            sorted_indices = tuple(sorted_indices)

            if sorted_indices not in sorted_indices_dict.keys():
                sorted_indices_dict[sorted_indices] = 1
            else:
                sorted_indices_dict[sorted_indices] += 1

        # Create df with sorted_indices and counts
        df = (
            pd.DataFrame(sorted_indices_dict.items(), columns=["sorted_indices", "count"])
            .sort_values(by="count", ascending=False)
            .reset_index(drop=True)
        )

        st.session_state.sorted_indices_df = df
        st.session_state.multi_decom_old_index = index
        st.session_state.update_multi_decom_view = False

    else:
        if "sorted_indices_df" not in st.session_state:
            st.session_state.sorted_indices_df = pd.DataFrame()
            return None

        if len(st.session_state.sorted_indices_df) == 0:
            return None
        df = st.session_state.sorted_indices_df.copy()
        atoms = atoms_list[st.session_state.multi_decom_old_index]
        pos = atoms.get_positions()
        elements = atoms.get_atomic_numbers()

    # Create atoms list
    atoms_list_new = []
    for sorted_indices in df["sorted_indices"].tolist():
        pos_new = pos[list(sorted_indices)]
        elements_new = elements[list(sorted_indices)]
        atoms_new = Atoms(symbols=elements_new, positions=pos_new)
        atoms_list_new.append(atoms_new)

    st.write(f"Found {len(atoms_list_new)} unique decompositions using {decom_method}.")

    make_multi_button_columns(
        df=df,
        button_configs=[
            ("View Seq", view_atoms_from_list_sequence, {"atoms_list": atoms_list_new}),
        ],
        unique_key="explore_datasets_decom_all_df",
    )

    return None


def investigate_decom(df: pd.DataFrame) -> None:
    st.header("Investigate decomposition scheme ")

    if "loaded_mols_decom" not in st.session_state:
        st.session_state.loaded_mols_decom = pd.DataFrame()
        return None
    if len(st.session_state.loaded_mols_decom) == 0:
        return None
    if "decom_view_params" not in st.session_state:
        st.session_state.decom_view_params = {}
    if "run_single_view" not in st.session_state:
        st.session_state.run_single_view = False
    if "update_multi_decom_view" not in st.session_state:
        st.session_state.update_multi_decom_view = False
    if "multi_decom_old_index" not in st.session_state:
        st.session_state.multi_decom_old_index = None

    df_filtered = st.session_state.loaded_mols_decom.copy()
    symbols = df_filtered["atomic_symbols"].iloc[0]
    df_filtered["delta_E"] = df_filtered["energy_GFN2"].copy() - get_sum_of_energies(symbols)
    df_shown = (
        df_filtered[["bag_repr", "n_atoms", "energy_GFN2", "delta_E", "SMILES"]]
        .reset_index(drop=False)
        .rename(columns={"index": "mol_id"})
    )
    df_pos_and_symbols = (
        df_filtered[["pos", "atomic_symbols"]].reset_index(drop=False).rename(columns={"index": "mol_id"})
    )
    atoms_list = [row_to_atoms(row) for _, row in df_pos_and_symbols.iterrows()]

    hydrogen_delay = st.checkbox("Hydrogen Delay", key="hydrogen_delay")
    no_hydrogen_focus = st.checkbox("No hydrogen focus", key="no_hydrogen_focus")
    show_focus = st.checkbox("Show focus", key="show_focus")
    action_reconstruction_fn = st.radio(
        r"$\textsf{{\large Select action reconstruction function}}$",
        ["pos_seq_to_actions", "pos_seq_to_actions_emma"],
        key="action_reconstruction_fn",
        horizontal=True,
    )

    all_decoms = st.checkbox("View all decompositions", key="all_decoms")

    act_fn_dict = {"pos_seq_to_actions": pos_seq_to_actions, "pos_seq_to_actions_emma": pos_seq_to_actions_emma}

    decom_params = {
        "hydrogen_delay": hydrogen_delay,
        "no_hydrogen_focus": no_hydrogen_focus,
        "show_focus": show_focus,
        "act_fn": act_fn_dict[action_reconstruction_fn],
    }

    def set_decom_params(index: int, **kwargs) -> None:

        decom_method = kwargs["decom_method"]
        decom_params = kwargs["decom_params"]
        all_decoms = kwargs["all_decoms"]

        # st.session_state.explore_decom_params = decom_params
        # st.session_state.explore_decom_method = decom_method
        # st.session_state.explore_decom_index = index

        st.session_state.decom_view_params = {
            "decom_method": decom_method,
            "decom_params": decom_params,
            "index": index,
        }

        if all_decoms:
            st.session_state.run_single_view = False
            st.session_state.update_multi_decom_view = True
        else:
            st.session_state.run_single_view = True
            st.session_state.update_multi_decom_view = False

        return None

    make_multi_button_columns(
        df=df_shown,
        button_configs=[
            ("View", view_atoms_from_list, {"atoms_list": atoms_list}),
            ("View Seq Original", view_atoms_from_list_sequence, {"atoms_list": atoms_list}),
            # ('View BFS', view_fn, {'atoms_list': atoms_list, 'decom_method': 'bfs', 'decom_params': decom_params}),
            # ('View DFS', view_fn, {'atoms_list': atoms_list, 'decom_method': 'dfs', 'decom_params': decom_params})
            (
                "View BFS",
                set_decom_params,
                {"decom_method": "bfs", "decom_params": decom_params, "all_decoms": all_decoms},
            ),
            (
                "View DFS",
                set_decom_params,
                {"decom_method": "dfs", "decom_params": decom_params, "all_decoms": all_decoms},
            ),
        ],
        unique_key="explore_datasets_decom_df",
    )

    if st.session_state.decom_view_params == {}:
        return None

    # st.write(f" st.session_state.decom_view_params: {st.session_state.decom_view_params}")

    view_decom_all(
        st.session_state.decom_view_params["index"],
        atoms_list,
        st.session_state.decom_view_params["decom_params"],
        st.session_state.decom_view_params["decom_method"],
    )

    if st.session_state.run_single_view:
        view_decom_single(
            st.session_state.decom_view_params["index"],
            atoms_list,
            st.session_state.decom_view_params["decom_params"],
            st.session_state.decom_view_params["decom_method"],
        )
        st.session_state.run_single_view = False


def main():
    show_logo()
    initialize_session_state()

    st.sidebar.title("Load dataset")
    datasets = ["QM7", "QM9", "tmQM"]
    dataset = st.sidebar.selectbox("Select dataset", datasets)

    if "current_dataset_name" not in st.session_state:
        st.session_state.current_dataset_name = dataset
    elif st.session_state.current_dataset_name != dataset:
        # switch_detected
        st.session_state.explore_datasets_mols_indices = None
    st.session_state.current_dataset_name = dataset

    # st.sidebar.title('Upcoming datasets')

    # upcoming_datasets = ['PC9', 'QMugs', 'GEOM']
    # for upcoming_dataset in upcoming_datasets:
    #     st.sidebar.write(upcoming_dataset)

    data_dict = None
    data_dict, df_full = load_dataset(name=dataset)

    if data_dict is None:
        return None

    if dataset + "_symbols" not in st.session_state:
        # syms = list(set([atom for molecule in data_dict.values() for atom in molecule['atomic_symbols']]))
        zs = list(set([atom for molecule in data_dict.values() for atom in molecule["elements_unique"]]))
        symbols_sorted = [chemical_symbols[z] for z in zs]
        st.session_state[dataset + "_symbols"] = symbols_sorted

    st.header(f"Explore {dataset} dataset")
    col1, col2, dead_col = st.columns([1, 2, 5])
    col1.metric(label="#Molecules", value=prettify(len(df_full)), delta=None, delta_color="normal", help=None)
    col2.metric("#Bags", prettify(len(df_full["bag_repr"].unique())), None)

    with st.expander(r"$\textsf{{\large Filter molecule dataframe}}$", expanded=False):
        # Filter molecules based on n_atoms
        n_atoms_max_data = max(data_dict.values(), key=lambda x: x["n_atoms"])["n_atoms"]
        n_atoms_min, n_atoms_max = st.slider("n_atoms", 0, n_atoms_max_data, (5, min(10, n_atoms_max_data)))
        df = df_full[(df_full["n_atoms"] >= n_atoms_min) & (df_full["n_atoms"] <= n_atoms_max)]

        all_elements = [atomic_numbers[sym] for sym in st.session_state[dataset + "_symbols"]]

        # Filter molecules based on atomic elements
        st.write("_______________________________________________")

        def stringify(i: int = 0) -> str:
            return ["Not present", "Optional", "Must be present"][i]

        dict_of_slides = {}
        for element in all_elements:
            _, col_write, _ = st.columns([5, 1, 5])
            col_write.write(chemical_symbols[element])
            dict_of_slides[element] = st.select_slider(
                chemical_symbols[element],
                options=[0, 1, 2],
                value=1,
                format_func=stringify,
                label_visibility="collapsed",
            )

        def filter_based_on_sliders(row):
            for element, slider in dict_of_slides.items():
                if slider == 0 and element in row["atomic_nums"]:
                    return False
                if slider == 2 and element not in row["atomic_nums"]:
                    return False
            return True

        df = df[df.apply(filter_based_on_sliders, axis=1)]
        st.write("_______________________________________________")

        # Filter molecules based on bag_repr
        bag_repr = st.text_input(
            r"$\textsf{{\small Chemical formula}}$",
            key="bag_repr",
            placeholder="H5C4NO2 (notice that symbols must be ordered by atomic number, TODO: fix this)",
        )  # value='H5C4NO2'
        if bag_repr != "":
            df = df[df["bag_repr"] == bag_repr]

    st.subheader("Bag statistics")

    cols = st.columns([3, 5, 2])
    with cols[0]:
        bag_df = get_bag_df(df)
        sort_by_energy = True  # st.checkbox('Sort by energy in viewer', key='sort_dataset_bags_by_energy_{dataset}')

        make_multi_button_columns(
            df=bag_df,
            button_configs=[
                (
                    "GUI",
                    view_bag_conformers_fn,
                    {"fn_args": {"df": df, "bag_df": bag_df, "sort_by_energy": sort_by_energy}},
                ),
                # ('List', display_bag_mols_df, {'df': df, 'bag_df': bag_df, 'sort_by_energy': sort_by_energy, 'col': cols[1]})
                ("List", explore_mols_set_bag_index, {"df": df, "bag_df": bag_df, "dataset_name": dataset}),
            ],
            unique_key=f"explore_datasets_bag_df_{dataset}",
        )

    # if st.session_state.current_dataset_name == 'tmQM':
    #    return None

    display_bag_mols_df(df=df_full, bag_df=bag_df, sort_by_energy=sort_by_energy, cols=cols[1:])

    investigate_decom(df=df_full)


if __name__ == "__main__":
    main()
