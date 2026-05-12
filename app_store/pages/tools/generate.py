from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.rl.env_container import SimpleEnvContainer
from src.rl.buffer_container import PPOBufferContainerDeploy
from src.rl.rollouts import rollout_n_eps_per_env
from src.tools.util import bag_tuple_to_str_formula

# from src.performance.metrics import atom_list_to_df
from src.data.reference_dataloader import ReferenceDataLoader, ReferenceData
from src.performance.energetics import EnergyUnit
from app_store.pages.tools.playground import Playground


def generate_samples(
    ac,
    eval_envs,
    mode: str = "stochastic",
    num_samples: int = 1,
    render: bool = False,
    render_bonds: bool = False,
    n_workers: int = None,
):
    if mode not in ["stochastic", "argmax"]:
        raise ValueError("mode must be either stochastic or argmax")
    if mode == "argmax":
        num_samples = 1

    ac = deepcopy(ac)

    assert eval_envs.get_size() == 1, "Expected n_envs to be 1 when entering generate_samples()"
    for env in eval_envs.environments:
        assert len(env.formulas) == 1, "Only one formula per environment supported"

    eval_envs = SimpleEnvContainer([deepcopy(eval_envs.environments[0]) for _ in range(n_workers)])
    n_envs = eval_envs.get_size()
    num_samples_per_env = int(np.ceil(num_samples / n_envs))

    # st.write(f"num_workers: {n_envs},   samples_per_worker: {num_samples_per_env} \n",
    #          f"needed: {num_samples},   total: {n_envs * num_samples_per_env}",
    #          f"redundancy: {n_envs * num_samples_per_env - num_samples}",
    #          f"redundancy_ratio: {(n_envs * num_samples_per_env - num_samples) / num_samples}")

    with torch.no_grad():
        ac.training = False if mode == "argmax" else True
        eval_container = PPOBufferContainerDeploy(size=n_envs, gamma=1.0, lam=0.97)
        # st.write(f"Render bonds: {render_bonds}")
        rollout = rollout_n_eps_per_env(
            ac,
            deepcopy(eval_envs),
            buffer_container=eval_container,
            num_episodes=num_samples_per_env,
            output_trajs=True,
            render=render,
            render_bonds=render_bonds,
            num_episodes_combined=num_samples,
        )

    # TODO: Merge trajs
    formula = bag_tuple_to_str_formula(eval_envs.environments[0].formulas[0])
    merged_atoms_list = []
    for atoms_list in rollout["rollout_trajs"]:
        merged_atoms_list.extend(atoms_list)
    rollout["rollout_trajs"] = {formula: merged_atoms_list}

    if len(merged_atoms_list) != num_samples:
        st.warning(
            f"Number of samples ({len(merged_atoms_list)}) does not match number of requested samples ({num_samples})",
            icon="⚠️",
        )

    # TODO: Potentially concatenate rollouts from on the same formula but different environments

    return rollout


def append_rollouts(old_rollouts, new_rollouts):
    if old_rollouts is None:
        return new_rollouts
    for formula, rollout in new_rollouts["rollout_trajs"].items():
        if formula in old_rollouts["rollout_trajs"]:
            old_rollouts["rollout_trajs"][formula].extend(rollout)
        else:
            old_rollouts["rollout_trajs"][formula] = rollout
    return old_rollouts


def save_to_formula_dict(formula_dict: dict, mode: str, formula: str, new_df: pd.DataFrame, data: List[Dict]):
    new_df.reset_index(inplace=True, drop=True)

    if formula in formula_dict and mode == "stochastic":
        assert "df" in formula_dict[formula], "formula_dict[formula] should have a df key"
        formula_dict[formula]["df"] = pd.concat([formula_dict[formula]["df"], new_df])
        formula_dict[formula]["df"].reset_index(inplace=True, drop=True)

        assert "data" in formula_dict[formula], "formula_dict[formula] should have a data key"
        formula_dict[formula]["data"].extend(data)

    else:
        formula_dict[formula] = {}
        formula_dict[formula]["df"] = new_df
        formula_dict[formula]["data"] = data


def extract_bag_repr(pg: Playground) -> str:
    assert len(pg.envs.environments[0].formulas) == 1, "Only one formula per environment supported"
    bag_tuple = pg.envs.environments[0].formulas[0]
    bag_tuple_sorted = sorted(bag_tuple, key=lambda x: x[0])
    return bag_tuple_to_str_formula(bag_tuple_sorted)


def select_dataset(matching_datasets: List[str], key: str = None) -> str:

    if len(matching_datasets) == 1:
        dataset_name = matching_datasets[0]
    else:  # len(matching_datasets) > 1:
        default_choice = "Choose one of the datasets below"
        matching_datasets = [default_choice] + matching_datasets

        dataset_name = st.selectbox(
            label="Select dataset for energy comparison (RAE)",
            options=matching_datasets,
            key=f"ref_dataset_selection_{key}",
        )

        if dataset_name == default_choice:
            st.warning("Please select a dataset from the list.")
            st.session_state.trapped_in_ref_selector = True
            st.stop()
        else:
            st.session_state.trapped_in_ref_selector = False

    return dataset_name


def get_ref_energies(bag_repr: Union[str, List[str]], key: str = None) -> Dict[str, ReferenceData]:
    loader = ReferenceDataLoader(data_dir="data")

    if isinstance(bag_repr, str):
        bag_repr = [bag_repr]

    matching_datasets = set()
    for bag in bag_repr:
        matching_datasets.update(loader.finder.find_matching_datasets(bag))
    matching_datasets = list(matching_datasets)

    if len(matching_datasets) == 0:
        st.write(f"No matching dataset found for {bag_repr}")
        return None

    dataset_name = select_dataset(matching_datasets, key)

    if dataset_name in matching_datasets:
        ref_data = loader.load_and_polish(mol_dataset=dataset_name, new_energy_unit=EnergyUnit.EV, fetch_df=False)

        st.success(f"Succesfully loaded dataset {dataset_name}")

        return {dataset_name: ref_data}


def write_reference_energies_into_pg(pg) -> None:
    bag_repr = extract_bag_repr(pg)
    st.write(f"Searching for reference data for playground {pg.name} with bag {bag_repr}")

    energy_dict = get_ref_energies(bag_repr)
    if energy_dict is None:
        return

    pg.bag_energies = energy_dict["bag_energies"]
    pg.bag_energy_mean = energy_dict["bag_energy_mean"]


def write_reference_energies_into_da(da) -> None:
    bag_repr = extract_bag_repr(da.pg)
    st.write(f"Searching for reference data for double agent {da.name} with bag {bag_repr}")

    energy_dict = get_ref_energies(bag_repr)
    if energy_dict is None:
        return

    da.bag_energies = energy_dict["bag_energies"]
    da.bag_energy_mean = energy_dict["bag_energy_mean"]


def rollout_pg(
    pg,
    mode="stochastic",
    num_samples=1,
    render=False,
    render_bonds=False,
    num_workers=None,
    perform_relaxation=False,
    calc_dipole=False,
):

    # write_reference_energies_into_pg(pg)

    if hasattr(pg, "bag_energy_mean"):
        benchmark_energies = pg.bag_energy_mean
    else:
        benchmark_energies = None

    names = [model.name for model in pg.agents]

    for name, da in zip(names, pg.double_agents):
        with st.spinner(f"Generating {mode} rollouts for {name} on {da.envs.environments[0].formulas}"):
            rollout = generate_samples(
                da.ac,
                da.envs,
                mode=mode,
                num_samples=num_samples,
                render=render,
                render_bonds=render_bonds,
                n_workers=num_workers,
            )
        if mode == "stochastic":
            da.stoch_rollouts = append_rollouts(da.stoch_rollouts, rollout)
        else:
            da.argmax_rollouts = rollout

        for formula, atoms_list in rollout["rollout_trajs"].items():
            if len(atoms_list) > 0:
                new_df, data = st.session_state.mol_processor.atom_list_to_df(
                    atoms_list,
                    benchmark_energies=benchmark_energies,
                    perform_optimization=perform_relaxation,
                    calc_dipole=calc_dipole,
                    # use_huckel=True
                )
                save_to_formula_dict(
                    da.stoch_dict if mode == "stochastic" else da.argmax_dict, mode, formula, new_df, data
                )


def generate_buttons():
    if "trapped_in_ref_selector" not in st.session_state:
        st.session_state.trapped_in_ref_selector = False

    toggle_cols = st.columns([1, 1, 1, 1])
    with toggle_cols[0]:
        render = st.toggle("Render", key="render_during_generation")
    with toggle_cols[1]:
        render_bonds = st.toggle("with bonds", disabled=not render, key="render_during_generation_with_bonds")
    with toggle_cols[2]:
        perform_relaxation = st.toggle("Relax", key="relax_results")
    with toggle_cols[3]:
        calc_dipole = st.toggle("Calculate dipole", key="calc_dipole_of_results")

    argmax_col, stoch_col, n_samples_col = st.columns([3, 3, 4])
    with argmax_col:
        generate_argmax_button = st.button("Generate Argmax Rollout ⛰️")
        if generate_argmax_button:
            st.session_state.generate_button = True

            for pg in st.session_state.pm.playgrounds:
                if pg.deployable:
                    rollout_pg(
                        pg,
                        mode="argmax",
                        num_samples=1,
                        num_workers=1,
                        perform_relaxation=perform_relaxation,
                        calc_dipole=calc_dipole,
                    )

    with n_samples_col:
        num_samples = st.number_input("#samples:", min_value=1, max_value=10000, value=3)
        num_workers = st.number_input("#workers:", min_value=1, max_value=128, value=1)

    with stoch_col:
        _ = st.button("Generate Stochastic Rollouts 🎲", key="generate_stoch_button")
        # render_cols = st.columns([1, 1])
        # with render_cols[0]:
        #     render = st.toggle("Render")
        # with render_cols[1]:

        if st.session_state.generate_stoch_button:  #  or st.session_state.trapped_in_ref_selector:
            for pg in st.session_state.pm.playgrounds:
                if pg.deployable:
                    rollout_pg(
                        pg,
                        mode="stochastic",
                        num_samples=num_samples,
                        render=render,
                        render_bonds=render_bonds,
                        num_workers=num_workers,
                        perform_relaxation=perform_relaxation,
                        calc_dipole=calc_dipole,
                    )
