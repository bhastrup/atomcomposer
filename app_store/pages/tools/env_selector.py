import itertools
from typing import List, Tuple
from copy import deepcopy

from ase import Atoms
from ase.data import atomic_numbers
from ase.visualize import view

import numpy as np
import pandas as pd
import streamlit as st

from src.rl.spaces import ActionSpace, ObservationSpace
from src.rl.reward import InteractionReward
from src.rl.envs.environment import tmqmEnv
from src.rl.env_container import SimpleEnvContainer
from src.tools import util
from src.rl.envs.env_no_reward import HeavyFirstNoReward
from src.rl.envs.env_partial_canvas import (
    ObservationType,
    FormulaType,
    AbstractMolecularEnvironment,
)

# class LoadedEnv(SimpleEnvContainer):
#     """A LoadedEnv should contain a SimpleEnvContainer and a name"""
#     def __init__(self, env_container: SimpleEnvContainer, name: str):
#         self.env_container.name = name


def configure_env():
    """
    Configure and load an env into the session state.
    Select between
        - From scratch environments (using the HeavyFirst env) and
        - partial canvas environments (using the PartialCanvasEnv env) that must read a particular input molecule
          and remove atoms from it, in order to place them again.
    """
    # options = ['From scratch', 'PartialCanvasEnv']
    # switch = st.radio("", options, index=0, key='env_type_selector', horizontal=True, label_visibility='collapsed')

    options = (("From scratch", single_bag_selector), ("PartialCanvasEnv", partial_canvas_selector))

    config_tabs = st.tabs([op[0] for op in options])
    config_fns = [op[1] for op in options]
    for config_tab, config_fn in zip(config_tabs, config_fns):
        with config_tab:
            config_fn()


def df_to_single_atoms(index: int, df: pd.DataFrame):
    row = df.iloc[index]
    return Atoms(symbols=row["atomic_symbols"], positions=row["pos"])


def partial_canvas_selector():
    if len(st.session_state.loaded_mols) == 0:
        st.caption("No molecules loaded yet. Go to 'Explore datasets' tab to find an interesting molecule.")
        return
    else:
        st.caption("Go to 'Explore datasets' tab to find more interesting molecules.")

    # mol_ids = list(st.session_state.loaded_mols.keys())
    mol_ids = list(st.session_state.loaded_mols.index)
    mol_dataset_names = list(st.session_state.loaded_mols["dataset_name"])
    # st.write(f"mol_ids: {mol_ids}")

    atom_list = [df_to_single_atoms(i, st.session_state.loaded_mols) for i in range(len(mol_ids))]
    atom_list_dict = {k: v for k, v in zip(mol_ids, atom_list)}

    # st.write(f"atom_list: {atom_list}")

    values = [atoms.get_chemical_formula() for atoms in atom_list]  # st.session_state.loaded_mols.values()]
    options = [f"{dataset_name}, {str(k)}, {v}" for dataset_name, k, v in zip(mol_dataset_names, mol_ids, values)]

    selected_mol = st.selectbox("Select molecule:", options=options, index=0, key="mol_selector")
    selected_mol_dataset_name = selected_mol.split(",")[0]
    selected_mol_id = int(selected_mol.split(",")[1])
    # atoms = st.session_state.loaded_mols[selected_mol_id]
    atoms = atom_list_dict[selected_mol_id]

    full_mol_id = f"{selected_mol_dataset_name}-{selected_mol_id}"

    st.markdown("---")

    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.caption(f"MOL: {selected_mol}")

    with cols[1]:
        view_button = st.button("View full", key="view_mol")
        if view_button:
            view(atoms, viewer="ase")

    with cols[2]:
        remove_button = st.button("Remove ☠️", key="remove_atoms")
        if remove_button:
            # remove row from session state df
            st.session_state.loaded_mols = st.session_state.loaded_mols.drop(selected_mol_id)
            st.rerun()

    atoms_to_remove = st.text_input("Type in atoms to remove (ex. 0,1,2,3):", placeholder="ex. 1,2,3", value="0")
    atoms_to_remove = [int(i) for i in atoms_to_remove.split(",")] if atoms_to_remove else []
    atoms_to_remove = set([i for i in atoms_to_remove if i < len(atoms)])
    atoms_to_remove = list(sorted(atoms_to_remove, reverse=False))

    pos = atoms.get_positions()
    new_pos = [p for i, p in enumerate(pos) if i not in atoms_to_remove]
    atomic_symbols = atoms.get_chemical_symbols()
    new_symbols = [s for i, s in enumerate(atomic_symbols) if i not in atoms_to_remove]
    removed_symbols = [s for i, s in enumerate(atomic_symbols) if i in atoms_to_remove]
    atoms = Atoms(symbols=new_symbols, positions=new_pos)

    cols_new = st.columns([3, 1])
    with cols_new[0]:
        cols_new2 = st.columns([1, 1])
        with cols_new2[0]:
            st.write(f"Removed: {atoms_to_remove}")

        with cols_new2[1]:
            view_button = st.button("View Canvas", key="view_partial_mol")
            if view_button:
                view(atoms, viewer="ase")

        original_bag = st.session_state.loaded_mols.loc[selected_mol_id]["formulas"]

        new_bag = tuple([(el, 0) for (el, z) in deepcopy(original_bag)])
        canvas_bag = deepcopy(original_bag)
        for sym in removed_symbols:
            new_bag = util.add_atom_to_formula(new_bag, atomic_numbers[sym])
            canvas_bag = util.remove_atom_from_formula(canvas_bag, atomic_numbers[sym])

        # st.write(f"original_bag: {original_bag}")
        # st.write(f"canvas_bag: {canvas_bag}")
        # st.write(f"new_bag: {new_bag}")
        # st.write(f"removed_symbols: {removed_symbols}")

    ############################ Customize bag ############################
    updated_bag = deepcopy(new_bag)

    def reset_bag():
        updated_bag = deepcopy(new_bag)
        for sym in st.session_state[selected_mol_dataset_name + "_symbols"]:
            el = atomic_numbers[sym]
            # if f"custom_bag_{full_mol_id}_{sym}" in st.session_state:
            st.session_state[f"custom_bag_{full_mol_id}_{sym}"] = util.find_count(updated_bag, el)
        return updated_bag

    change_bag = st.checkbox("Customize bag", key="change_bag", value=False, on_change=reset_bag)
    if change_bag:
        # reset_bag()

        for sym in st.session_state[selected_mol_dataset_name + "_symbols"]:
            el = atomic_numbers[sym]
            count = util.find_count(updated_bag, el)
            # st.write(f"count: {count}")

            sym_cols = st.columns([1, 10, 1])
            with sym_cols[0]:
                st.write(rf"$\textsf{{\small {sym}}}$")

            with sym_cols[1]:
                # my_num = st.number_input(f"{sym}", min_value=0, max_value=100, value=count, key=f"num_{s}", label_visibility='collapsed', step=1)
                new_count = st.slider(
                    f"{sym}",
                    min_value=0,
                    max_value=7,
                    value=count
                    if f"custom_bag_{full_mol_id}_{sym}" not in st.session_state
                    else None,  # st.session_state[f"custom_bag_{full_mol_id}_{sym}"],
                    key=f"custom_bag_{full_mol_id}_{sym}",
                    label_visibility="collapsed",
                    step=1,
                )
                updated_bag = util.update_count(updated_bag, el, new_count)

        reset_bag = st.button("Align with removed", key="reset_bag", on_click=reset_bag)

    st.write(f"updated bag: {updated_bag}")

    with cols_new[1]:
        make_env = st.button("Load env", key="make_partial_env")
        if make_env:
            df = st.session_state.loaded_mols.loc[[selected_mol_id]]
            df["atoms_to_remove"] = [atoms_to_remove]
            envs = make_partial_canvas_env(df=df, num_envs=1, mol_dataset="QM7", new_bag=updated_bag)
            # name = f"Partial_{selected_mol_id}_{str(atoms_to_remove)}"
            updated_formula = util.bag_tuple_to_str_formula(updated_bag)
            # st.write(''.join(f"{chemical_symbols[z] if count > 0 else ''}{count if count > 1 else ''}" for z, count in updated_bag))
            name = (
                f"Partial-{selected_mol_dataset_name}-{selected_mol_id}-{str(atoms_to_remove)}-{str(updated_formula)}"
            )

            if not is_already_loaded_envs(name):
                st.session_state.loaded_envs[name] = envs
            else:
                st.write(f"Env {name} is already loaded.")


class PartialCanvasEnvFixed(AbstractMolecularEnvironment):
    def __init__(self, molecule_df: pd.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # benchmark_energy: List[float] = [None]
        # self.eval = eval

        assert len(molecule_df) == 1, "Only one molecule at this point, but can be extended to multiple"

        self.molecule_df = molecule_df
        self.formulas = [molecule_df["formulas"].iloc[0]]
        self.new_bag = molecule_df["new_bag"].iloc[0]
        indices = np.arange(len(self.molecule_df))
        self.index_cycle = itertools.cycle(indices)

        self.obs_reset = self.reset()

    def reset(self) -> ObservationType:
        """Reset the environment."""

        index = next(self.index_cycle)
        mol = self.molecule_df.iloc[index]
        # st.write(f"mol: {mol}")

        pos, elements, _, _ = mol["pos"], mol["atomic_nums"], mol["atomic_symbols"], mol["formulas"]
        atoms_to_remove = mol["atoms_to_remove"]

        sorted_indices = np.array([int(i) for i in np.arange(len(elements)) if i not in atoms_to_remove])
        sorted_indices = np.concatenate((sorted_indices, atoms_to_remove)).astype(int)

        obs_tuple = self._construct_observation_tuple(
            pos, elements, sorted_indices, len(atoms_to_remove), new_bag=deepcopy(self.new_bag)
        )

        self.current_atoms, self.current_formula = obs_tuple

        # st.write(f"self.current_atoms: {self.current_atoms}")
        # st.write(f"self.current_formula: {self.current_formula}")

        return self.observation_space.build(self.current_atoms, self.current_formula)

    def _construct_observation_tuple(
        self,
        pos,
        elements,
        sorted_indices,
        num_atoms_to_place: int = 1,
        new_bag: FormulaType = None,
    ) -> Tuple[Atoms, FormulaType]:
        """Construct observation tuples from decomposed molecule."""

        num_core = len(elements) - num_atoms_to_place
        elements = np.array(elements)

        pos_new = pos[sorted_indices[:num_core]]
        zs_new = elements[sorted_indices[:num_core]]
        canvas_atoms = Atoms(numbers=zs_new, positions=pos_new)

        # bag_formula = util.zs_to_formula(elements[sorted_indices[num_core:]])
        bag_formula = new_bag

        # print(f"obs tuples: {(canvas_atoms, bag_formula)}")

        return (canvas_atoms, bag_formula)


def make_partial_canvas_env(df: pd.DataFrame, num_envs: int = 1, mol_dataset: str = None, new_bag: FormulaType = None):
    """
    Creates a PartialCanvasEnv from a given Atoms object.
    """

    if mol_dataset == "QM7":
        zs = [0, 1, 6, 7, 8, 16]
        canvas_size = 23

    action_space = ActionSpace(zs=zs)
    observation_space = ObservationSpace(canvas_size=canvas_size, zs=zs)

    reward_coefs = {"rew_abs_E": 1.0, "rew_valid": 3.0}
    reward = InteractionReward(reward_coefs=reward_coefs)
    # TODO: Don't even calc rew, since we do it also in metrics

    min_atomic_distance = 0.6
    max_solo_distance = 2.0
    min_reward = -3

    assert len(df) == 1, "Only one molecule at this point, but can be extended to multiple"
    df["new_bag"] = [new_bag]

    envs = SimpleEnvContainer(
        [
            PartialCanvasEnvFixed(
                reward=reward,
                observation_space=observation_space,
                action_space=action_space,
                molecule_df=df,
                min_atomic_distance=min_atomic_distance,
                max_solo_distance=max_solo_distance,
                min_reward=min_reward,
            )
            for _ in range(num_envs)
        ]
    )

    return envs


def single_bag_selector():
    """
    Specifies and loads a 'From scratch' environment that is played from an empty canvas. Here the focus is to discover the best isomers.
    """
    # Setting calculator is only relevant for further training, not for evaluation
    # Maybe we should have a boolean decision whether or not to even calculate reward at each env.step()?

    eval_envs = None
    eval_formula = None
    options_list = None

    formulas_origin = st.radio(
        "Select formulas:",
        options=["From QM7", "Custom", "MolGym 2", "cG-SchNet", "Alanine-Dipeptide"],
        index=0,
        key="formula_selector",
        horizontal=True,
    )

    if formulas_origin == "From QM7":
        eval_formula = None
        col0, col1 = st.columns([1, 4])
        with col0:
            num_envs = st.number_input("Number of formulas:", min_value=1, max_value=10, value=1)
            # TODO: Should be able to select multiple formulas, currently this is not possible

        if st.button("Load env", key="load_QM7_env"):
            if not is_already_loaded_envs(eval_formula):
                eval_envs, eval_formulas = make_molgym_env(eval_formulas=None, num_envs=num_envs, mol_dataset="QM7")
                st.session_state.loaded_envs[eval_formulas[0]] = eval_envs
            else:
                st.write(f"Env {eval_formula} is already loaded.")

    elif formulas_origin == "Custom":
        eval_formula = st.text_input(
            "Type in bag of atoms here (or go to 'Explore datasets' tab to find an interesting test bag):",
            placeholder="ex. H2C5O2 (stick with H, C, N, O, S for now)",
            value="C3H5NO3",
        )

    elif formulas_origin == "MolGym 2":
        single_bags = [None, "C3H5NO3", "C4H7N", "C3H8O", "C7H10O2", "C7H8N2O2"]
        stochastic_bags = [None, "C7H10O2", "C7H8N2O2"]
        formula_type = st.radio(
            "Select bag type:", options=["Single", "Stochastic"], index=0, key="bag_type_selector", horizontal=True
        )
        options_list = single_bags if formula_type == "Single" else stochastic_bags

    elif formulas_origin == "cG-SchNet":
        options_list = [None, "C7N1O1H11"]

    elif formulas_origin == "Alanine-Dipeptide":
        options_list = [None, "C6N2O2H12"]
        # https://pubchem.ncbi.nlm.nih.gov/compound/5484387
        # Target SMILES: CC(C(=O)NC)NC(=O)C

    if options_list:
        eval_formula = st.selectbox("Select a bag:", options=options_list, index=0, key="bag_selector")

    if eval_formula:
        if st.button("Load env", key=f"load_{formulas_origin}_env"):
            if not is_already_loaded_envs(eval_formula):
                eval_envs, _ = make_molgym_env(eval_formulas=[eval_formula])
                st.session_state.loaded_envs[eval_formula] = eval_envs
            else:
                st.write(f"Env {eval_formula} is already loaded.")


@st.cache_data
def make_molgym_env(eval_formulas: List = None, num_envs: int = 1, mol_dataset: str = None):
    # TODO: Fix arguments

    # TODO: Fix num_envs. Should be determined at deployment.
    # num_envs = 1 # len(eval_formulas) if eval_formulas is not None else num_envs
    min_atomic_distance = 0.6
    max_solo_distance = 2.0
    min_reward = -3

    if mol_dataset == "QM7":
        if eval_formulas is None:
            raise NotImplementedError(
                "Automatic formula loading from pretraining data is not supported; pass eval_formulas explicitly."
            )
        zs = [0, 1, 6, 7, 8, 16]
        canvas_size = 23
    else:
        # TODO: Pretty sure it can cause trouble if the actionspace does not match the actionspace of the trained agent
        bag_tuples = [util.string_to_formula(f) for f in eval_formulas]
        n_atoms = [util.get_formula_size(bag_tuple) for bag_tuple in bag_tuples]
        canvas_size = max(n_atoms)
        elements = sorted(set([e for bag_tuple in bag_tuples for (e, _) in bag_tuple]))
        zs = [0] + elements

    # st.write(f'eval_formulas: {eval_formulas}')

    # TODO: else: should mol_dataset just refer to the training set name?
    # Should we change zs to include all elements, to facilitate finetuning on new elements?

    # benchmark_energies, _ = get_benchmark_energies(eval_formulas, mol_dataset)
    action_space = ActionSpace(zs=zs)
    observation_space = ObservationSpace(canvas_size=canvas_size, zs=zs)
    RLEnvironment = tmqmEnv if mol_dataset == "TMQM" else HeavyFirstNoReward  # HeavyFirst

    reward_coefs = {"rew_abs_E": 1.0}  # Should obviously not be hardcoded here
    # either we should have
    reward = InteractionReward(reward_coefs)

    eval_envs = SimpleEnvContainer(
        [
            RLEnvironment(
                reward=reward,
                observation_space=observation_space,
                action_space=action_space,
                formulas=[util.string_to_formula(f) for f in eval_formulas],
                min_atomic_distance=min_atomic_distance,
                max_solo_distance=max_solo_distance,
                min_reward=min_reward,
                # benchmark_energy=benchmark_energies,
            )
        ]
    )

    return eval_envs, eval_formulas


def is_already_loaded_envs(name: str):
    return True if name in st.session_state.loaded_envs else False


def display_loaded_envs():

    if st.session_state.loaded_envs:
        for name, loaded_env in st.session_state.loaded_envs.items():
            agent_col0, agent_col1, agent_col2 = st.columns([3, 2, 3])
            with agent_col0:
                st.write(name)

            with agent_col1:
                if st.button("Remove", key=f"unload_env_{name}"):
                    st.session_state.loaded_envs.pop(name)
                    st.rerun()

            with agent_col2:
                if st.button("To Playgrounds", key=f"to_playgrounds_{name}"):
                    env = deepcopy(loaded_env)
                    env.name = name
                    st.session_state.pm.add_env_to_pgs(env)
                    st.rerun()
    else:
        st.write("No envs loaded yet.")
