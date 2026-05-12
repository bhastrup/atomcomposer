import numpy as np
import pandas as pd
import streamlit as st
from ase.visualize import view

from src.performance.single_cpkt.stats import single_formula_metrics
from src.tools.util import str_formula_to_size
from app_store.pages.tools.playground import DoubleAgent
from app_store.pages.tools.app_utils import make_multi_button_columns
from app_store.pages.tools.generate import get_ref_energies
from app_store.pages.tools.visualize import view_atoms_from_list, view_atoms_from_list_sequence, view_rdkit_mol
from app_store.pages.tools.plots.rmsd import plot_basin_RMSD
from app_store.pages.tools.plots.jackknife import jackknife_plot
from app_store.pages.tools.plots.energy_histogram import energy_histogram_numpy, energy_histogram
from app_store.pages.tools.plots.rae_vs_rae_relaxed import plot_rae_vs_rae_relaxed
from src.performance.metrics import calc_rae


def rae_missing_from_df(df: pd.DataFrame) -> bool:
    rae_col = "rae_relaxed"
    if rae_col not in df.columns:
        return True
    elif df[rae_col].isnull().all():
        return True
    # if for some ROWS the RAE is None but the SMILES is not None
    elif (df[rae_col].isnull() & df["SMILES"].notnull()).any():
        return True
    return False


def do_show_argmax(da: DoubleAgent, formula) -> bool:
    if da.argmax_rollouts:
        if formula in da.argmax_rollouts["rollout_trajs"].keys():
            if da.argmax_rollouts["rollout_trajs"][formula]:
                return True
    return False


def do_show_stoch(da: DoubleAgent, formula) -> bool:
    if da.stoch_rollouts:
        if formula in da.stoch_rollouts["rollout_trajs"].keys():
            if da.stoch_rollouts["rollout_trajs"][formula]:
                return True
    return False


class MetricsDisplay:
    def __init__(self):
        self.metric_groups = {
            "Discovery metrics": [
                "total_samples",
                "n_valid",
                "valid_per_sample",
                "n_unique",
                "unique_per_sample",
                "unique_per_valid",
            ],
            "Rediscovery/Expansion metrics": [
                "old_data_size",
                "rediscovered",
                "rediscovery_ratio",
                "n_novel",
                "expansion_ratio",
            ],
            "Structure/energy metrics": [
                "abs_energy_avg",
                "dipole_avg",
                "relax_stable",
                "RMSD_avg",
                "rae_relaxed_avg",
                "n_rings_avg",
                "ring3+_ratio",
                "ring4+_ratio",
                "ring5+_ratio",
            ],
        }
        self.used_keys = set()

    @classmethod
    def display(cls, formula_metrics):
        instance = cls()
        for header_name, columns in instance.metric_groups.items():
            st.markdown("---")
            instance.display_metrics(header_name, columns, formula_metrics)

        instance.display_remaining_metrics(formula_metrics)

    def display_metrics(self, header_name, columns, formula_metrics):
        st.header(header_name)
        metrics = {k: formula_metrics.get(k, None) for k in columns}
        self.used_keys.update(columns)
        self.display_metric_columns(metrics)

    def display_remaining_metrics(self, formula_metrics):
        remaining_metrics = {k: v for k, v in formula_metrics.items() if k not in self.used_keys}
        if remaining_metrics:
            st.header("Other Metrics")
            self.display_metric_columns(remaining_metrics)

    @staticmethod
    def display_metric_columns(metrics):
        n_metrics = len(metrics)
        metric_cols = st.columns(n_metrics)
        for col, (metric_name, metric_value) in zip(metric_cols, metrics.items()):
            value = np.round(metric_value, 3) if metric_value is not None else None
            col.metric(metric_name, value)


def display_results_tabs():
    st.header("Results 📊")
    if len(st.session_state.pm.playgrounds) == 0:
        return None

    # pg has results if any double agent has results, i.e. if either argmax_rollouts or stoch_rollouts is not None
    pgs_with_results = [
        pg
        for pg in st.session_state.pm.playgrounds
        if any([da.argmax_rollouts or da.stoch_rollouts for da in pg.double_agents])
    ]
    if not pgs_with_results:
        return None

    pg_names = [pg.name for pg in pgs_with_results]
    # st.write(f'pg_names: {pg_names}')

    pg_tabs = st.tabs(pg_names)
    for tab, pg in zip(pg_tabs, pgs_with_results):
        with tab:
            css = """
            <style>
                .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size:1.2rem;
                }
            </style>
            """

            st.markdown(css, unsafe_allow_html=True)
            playground_results_tab(pg)


def playground_results_tab(pg):

    # write_reference_energies_into_pg(pg)

    # st.write(f"Results for {pg.name}")
    das, names = zip(
        *[(da, agent.name) for da, agent in zip(pg.double_agents, pg.agents) if da.argmax_rollouts or da.stoch_rollouts]
    )
    # st.write(f"len(das): {len(das)}")
    da_tabs = st.tabs(names)
    for i, (tab, da, name) in enumerate(zip(da_tabs, das, names)):
        with tab:
            agent_results_tab(da, name, pg_name=pg.name)


def agent_results_tab(da: DoubleAgent, name: str, pg_name: str):
    # TODO: Merge argmax and stochastic rollouts, i.e. align by formula

    # st.write(f"Agent: {name}")
    # get all formulas (in both argmax and stochastic rollouts)
    formulas = set()
    if da.argmax_rollouts:
        formulas.update(da.argmax_rollouts["rollout_trajs"].keys())
    if da.stoch_rollouts:
        formulas.update(da.stoch_rollouts["rollout_trajs"].keys())
    formulas = list(formulas)

    col0, col1, deadcol = st.columns([1, 1, 5])
    with col0:
        do_load_ref_data = st.checkbox("Load reference data", key=f"load_ref{pg_name}_{name}")
        if do_load_ref_data:
            ref_energies = get_ref_energies(bag_repr=formulas, key=f"{pg_name}_{name}")
            if ref_energies is not None:
                da.ref_data_collection.update(ref_energies)

    for formula in formulas:
        st.subheader(f"Formula: {formula}")
        show_argmax = do_show_argmax(da, formula)
        show_stoch = do_show_stoch(da, formula)
        df_stoch = da.stoch_dict[formula]["df"] if show_stoch else None

        if not da.ref_data_collection:
            SMILES_db = None
        else:
            with col1:
                ref_data_name = st.radio(
                    "Reference dataset",
                    options=da.ref_data_collection.keys(),
                    key=f"ref_dataset_{pg_name}_{name}_{str(formula)}",
                    horizontal=True,
                )
                SMILES_db = da.ref_data_collection[ref_data_name].smiles[formula]
                if rae_missing_from_df(df_stoch):
                    do_enhance_with_rae = st.checkbox(
                        "Enhance with RAE", key=f"enhance_rae_{pg_name}_{name}_{str(formula)}"
                    )
                    if do_enhance_with_rae:
                        df_stoch["rae_relaxed"] = calc_rae(
                            energy=df_stoch["e_relaxed"],
                            benchmark_energy=np.mean(da.ref_data_collection[ref_data_name].energies[formula]),
                            n_atoms=str_formula_to_size(formula),
                        )
                        df_stoch["rae"] = calc_rae(
                            energy=df_stoch["abs_energy"],
                            benchmark_energy=np.mean(da.ref_data_collection[ref_data_name].energies[formula]),
                            n_atoms=str_formula_to_size(formula),
                        )

        if show_argmax:
            st.caption("Argmax Rollout")
            arg_max_atoms_list = da.argmax_rollouts["rollout_trajs"][formula]
            df_argmax = da.argmax_dict[formula]["df"]
            n_obs = len(da.argmax_dict[formula]["data"])
            argmax_atoms_list_opt = [da.argmax_dict[formula]["data"][i]["new_atoms"] for i in range(n_obs)]

            make_multi_button_columns(
                df=df_argmax,
                button_configs=[
                    ("View", view_atoms_from_list, {"atoms_list": arg_max_atoms_list}),
                    ("View Seq", view_atoms_from_list_sequence, {"atoms_list": arg_max_atoms_list}),
                    ("View RdKit", view_rdkit_mol, {"atoms_list": arg_max_atoms_list}),
                    ("View Opt", view_atoms_from_list, {"atoms_list": argmax_atoms_list_opt}),
                ],
                unique_key=f"view_argmax_{pg_name}_{name}_{str(formula)}",
            )

        if show_stoch:
            st.caption("Stochastic Rollout")

            n_obs_stoch = len(da.stoch_dict[formula]["data"])
            stoch_atoms_list = da.stoch_rollouts["rollout_trajs"][formula]
            stoch_atoms_list_opt = [da.stoch_dict[formula]["data"][i]["new_atoms"] for i in range(n_obs_stoch)]

            # View buttons
            view_cols = st.columns([1, 1, 1, 1, 7])
            with view_cols[0]:
                if st.button("View all", key=f"view_all_{pg_name}_{name}_{str(formula)}"):
                    view(stoch_atoms_list)
            with view_cols[1]:
                if st.button("View all valid", key=f"view_all_valid_{pg_name}_{name}_{str(formula)}"):
                    all_valid = []
                    for i, atoms in enumerate(stoch_atoms_list):
                        if df_stoch.loc[i, "valid"]:
                            all_valid.append(atoms)
                    view(all_valid)

            with view_cols[3]:
                if st.button("View all valid RELAXED", key=f"view_all_relaxed_{pg_name}_{name}_{str(formula)}"):
                    view([atoms for atoms in stoch_atoms_list_opt if atoms is not None])

            # Show df of unique smiles

            # Select one and see all its conformers

            # View dataframe of stats
            make_multi_button_columns(
                df=df_stoch,
                button_configs=[
                    ("View", view_atoms_from_list, {"atoms_list": stoch_atoms_list}),
                    ("View Seq", view_atoms_from_list_sequence, {"atoms_list": stoch_atoms_list}),
                    ("View RdKit", view_rdkit_mol, {"atoms_list": stoch_atoms_list}),
                    ("View Opt", view_atoms_from_list, {"atoms_list": stoch_atoms_list_opt}),
                ],
                unique_key=f"view_stoch_{pg_name}_{name}_{str(formula)}",
            )

            # Metrics
            formula_metrics = single_formula_metrics(df_stoch, SMILES_db=SMILES_db)

            with st.expander(r"$\textsf{\Huge Metrics}$", expanded=True):
                MetricsDisplay.display(formula_metrics)

            # Plots
            plot_energy_histogram_pretty = st.checkbox(
                "Energy histogram pretty", key=f"energy_hist_{pg_name}_{name}_{str(formula)}"
            )
            if plot_energy_histogram_pretty:
                energy_histogram(df_stoch)

            plot_energy_histogram_numpy = st.checkbox(
                "Energy histogram numpy", key=f"energy_hist_numpy_{pg_name}_{name}_{str(formula)}"
            )
            if plot_energy_histogram_numpy:
                energy_histogram_numpy(df_stoch)

            plot_rae_energy = st.checkbox("RAE (Relaxed)", key=f"rae_{pg_name}_{name}_{str(formula)}")
            if plot_rae_energy:
                energy_histogram_numpy(df_stoch, "rae")

            plot_basin_rmsd = st.checkbox("Basin/RMSD scatter plot", key=f"basin_rmsd_{pg_name}_{name}_{str(formula)}")
            if plot_basin_rmsd:
                plot_basin_RMSD(df_stoch)

            do_jackknife = st.checkbox("Jackknife plot", key=f"jackknife_{pg_name}_{name}_{str(formula)}")
            if do_jackknife:
                jackknife_plot(df_stoch)

            do_rae_vs_rae_relaxed = st.checkbox(
                "RAE vs RAE relaxed", key=f"rae_vs_rae_relaxed_{pg_name}_{name}_{str(formula)}"
            )
            if do_rae_vs_rae_relaxed:
                plot_rae_vs_rae_relaxed(df_stoch)

        st.markdown("---")
