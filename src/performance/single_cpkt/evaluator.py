import os
import sys
import json
import logging
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import submitit

from ase import Atoms
from ase.io import read, write

from src.tools import util
from src.agents.base import AbstractActorCritic
from src.rl.rollouts import rollout_stoch
from src.performance.metrics import MoleculeProcessor
from src.performance.single_cpkt.stats import single_formula_metrics, get_global_metrics


from src.tools.launch_utils import submit_jobs


def get_num_episodes(
    smiles: Dict[str, List[str]], num_episodes_const: int = None, prop_factor: int = None, formulas: List[str] = None
) -> List[int]:

    assert (num_episodes_const is None) != (prop_factor is None), (
        "One and only one of num_episodes_const and prop_factor must be provided."
    )

    if num_episodes_const:
        num_episodes = [num_episodes_const] * len(formulas)
    elif prop_factor:
        num_episodes = (np.array([len(v) for _, v in smiles.items()]) * prop_factor).tolist()

    episode_dict = {f: n for f, n in zip(formulas, num_episodes)}

    logging.info(f"Number of episodes: {episode_dict}")
    print(f"Number of episodes: {episode_dict}")

    return num_episodes


class EvaluatorIO:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def save_smiles_to_json(self, smiles: Dict[str, List[str]]):
        for formula, smiles_list in smiles.items():
            formula_dir = os.path.join(self.base_dir, formula)
            os.makedirs(formula_dir, exist_ok=True)
            with open(os.path.join(formula_dir, "smiles.json"), "w") as f:
                json.dump(smiles_list, f)

    def save_atoms_as_traj_file(self, final_atoms: Dict[str, List[Atoms]]):
        print(f"Saving atoms as traj files to {self.base_dir}")
        for formula, atom_list in final_atoms.items():
            formula_dir = os.path.join(self.base_dir, formula)
            os.makedirs(formula_dir, exist_ok=True)
            write(os.path.join(formula_dir, "atoms.traj"), atom_list)

    def save_relaxed_atoms_as_traj_file(self, formula: str, atom_list: List[Atoms]):
        formula_dir = os.path.join(self.base_dir, formula)
        os.makedirs(formula_dir, exist_ok=True)
        write(os.path.join(formula_dir, "atoms_relaxed.traj"), atom_list)

    def read_atoms_as_traj_file(self, formulas: List[str]) -> Dict[str, List[Atoms]]:
        final_atoms = {}
        for formula in formulas:
            formula_path = os.path.join(self.base_dir, formula, "atoms.traj")
            if not os.path.exists(formula_path):
                raise FileNotFoundError(f"No trajectory file found at {formula_path} for formula {formula}")
            atom_list = read(formula_path, index=":")
            final_atoms[formula] = [atom_list] if not isinstance(atom_list, list) else atom_list
        return final_atoms

    def save_df_to_csv(self, df: pd.DataFrame, formula: str):
        formula_dir = os.path.join(self.base_dir, formula)
        os.makedirs(formula_dir, exist_ok=True)
        df.to_csv(os.path.join(formula_dir, "df.csv"), index=False)

    def save_dfs_to_csv(self, dfs: Dict[str, pd.DataFrame]):
        logging.warning("This method is deprecated. Use save_df_to_csv instead.")
        for formula, df in dfs.items():
            formula_dir = os.path.join(self.base_dir, formula)
            os.makedirs(formula_dir, exist_ok=True)
            df.to_csv(os.path.join(formula_dir, "df.csv"), index=False)

    def load_dfs_from_csv(self, formulas: List[str]) -> Dict[str, pd.DataFrame]:
        formula_dfs = {}
        for formula in formulas:
            formula_dir = os.path.join(self.base_dir, formula)
            df_path = os.path.join(formula_dir, "df.csv")
            if os.path.exists(df_path):
                formula_dfs[formula] = pd.read_csv(df_path)
            else:
                warning_text = f"No feature DataFrame found at {df_path} for formula {formula}"
                logging.warning(warning_text)
                print(warning_text)
        return formula_dfs

    def load_stats_and_smiles(self, formulas: List[str]) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
        formula_dfs = {}
        for formula in formulas:
            formula_dir = os.path.join(self.base_dir, formula)
            df_path = os.path.join(formula_dir, "df.csv")
            smiles_path = os.path.join(formula_dir, "smiles.json")

            if not os.path.exists(df_path):
                raise FileNotFoundError(f"No feature DataFrame found at {df_path} for formula {formula}")
            if not os.path.exists(smiles_path):
                raise FileNotFoundError(f"No smiles found at {smiles_path} for formula {formula}")

            df = pd.read_csv(os.path.join(formula_dir, "df.csv"))
            smiles = json.load(open(os.path.join(formula_dir, "smiles.json"), "r"))
            formula_dfs[formula] = (df, smiles)
        return formula_dfs

    def save_formula_metrics(self, formula_metrics: Dict[str, Dict[str, Any]]):
        for formula, metrics in formula_metrics.items():
            formula_dir = os.path.join(self.base_dir, formula)
            os.makedirs(formula_dir, exist_ok=True)
            with open(os.path.join(formula_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)

    def save_global_metrics(self, global_metrics: Dict[str, Any]):
        with open(os.path.join(self.base_dir, "global_metrics.json"), "w") as f:
            json.dump(global_metrics, f)


class SingleCheckpointEvaluator:
    def __init__(
        self,
        eval_envs,
        reference_smiles: Dict[str, list] = None,
        benchmark_energies: Dict[str, float] = None,
        io_handler: EvaluatorIO = None,
        wandb_run: Any = None,
        num_episodes_const: int = 10,
        prop_factor: int = 100,
        calc_dipole: bool = False,
    ):

        self.eval_envs = eval_envs
        self.io = io_handler
        self.wandb_run = wandb_run
        self.reference_smiles = reference_smiles
        self.benchmark_energies = benchmark_energies
        self.calc_dipole = calc_dipole
        self.eval_formulas = util.get_str_formulas_from_vecenv(eval_envs)
        self.num_episodes = get_num_episodes(reference_smiles, num_episodes_const, prop_factor, self.eval_formulas)

        if reference_smiles is not None:
            self.benchmark_energies_eval = {f: e for f, e in benchmark_energies.items() if f in self.eval_formulas}
            assert (
                len(reference_smiles)
                == len(self.eval_formulas)
                == len(self.benchmark_energies_eval)
                == eval_envs.get_size()
                == len(self.num_episodes)
            ), (
                "Inconsistent number of formulas, reference smiles, \
                benchmark energies, environments, and num_episodes."
            )
        else:
            self.benchmark_energies_eval = None
            assert len(self.eval_formulas) == eval_envs.get_size() == len(self.num_episodes), (
                "Inconsistent number of formulas, environments, and num_episodes."
            )

    def reset(self, new_base_dir: str):
        if self.io:
            self.io.base_dir = new_base_dir
            if self.reference_smiles:
                self.io.save_smiles_to_json(self.reference_smiles)

        self._data = {
            "rollouts": None,
            "formula_dfs": None,
            "formula_metrics": None,
            "global_metrics": None,
        }

        return self

    @property
    def data(self):
        return self._data

    def _rollout(self, ac: AbstractActorCritic, rollouts_from_file: bool = False) -> Dict[str, List[Atoms]]:
        """Rollout trajectories for each formula."""
        if rollouts_from_file:
            self.data["rollouts"] = self.io.read_atoms_as_traj_file(self.eval_formulas)
        else:
            self.data["rollouts"] = rollout_stoch(ac, self.eval_envs, self.num_episodes)
            if self.io:
                self.io.save_atoms_as_traj_file(self.data["rollouts"])

        return self.data["rollouts"]

    def _calc_features(
        self, features_from_file: bool = False, perform_optimization: bool = False, f_max: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """Calculate features for each formula."""
        formula_dfs = {}

        # Load features from file if specified
        if features_from_file:
            formula_dfs = self.io.load_dfs_from_csv(self.eval_formulas)

        # Identify missing formulas
        loaded_formulas = set(formula_dfs.keys())
        print(f"Loaded the following formulas: {loaded_formulas}")
        all_formulas = set(self.data["rollouts"].keys())  # self.eval_formulas
        missing_formulas = all_formulas - loaded_formulas
        print(f"Missing formulas: {missing_formulas}")

        # Process missing formulas
        if missing_formulas:
            processor = MoleculeProcessor()
            for formula in missing_formulas:
                print(f"Calc features for {formula}")
                atom_list = self.data["rollouts"][formula]
                df, extra_data_list = processor.atom_list_to_df(
                    atoms_object_list=atom_list,
                    benchmark_energies=self.benchmark_energies_eval,
                    perform_optimization=perform_optimization,
                    calc_dipole=self.calc_dipole,
                    f_max=f_max,
                )
                formula_dfs[formula] = df
                if self.io:
                    self.io.save_df_to_csv(df, formula)
                    if perform_optimization:
                        # Fall back to original atoms for invalid/unrelaxed molecules so the
                        # traj file stays index-aligned with df.csv and atoms.traj.
                        relaxed_atoms = [
                            d["new_atoms"] if d["new_atoms"] is not None else orig
                            for d, orig in zip(extra_data_list, atom_list)
                        ]
                        self.io.save_relaxed_atoms_as_traj_file(formula, relaxed_atoms)

        # Store in data and return
        self.data["formula_dfs"] = formula_dfs
        return formula_dfs

    def _get_metrics_by_formula(self, dfs_from_file: bool = False) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics for each formula."""
        if dfs_from_file:
            stats_and_smiles = self.io.load_stats_and_smiles(self.eval_formulas)
        else:
            assert self.data["formula_dfs"] is not None, "Features must be calculated before metrics."
            stats_and_smiles = {
                formula: (df, self.reference_smiles[formula] if self.reference_smiles else None)
                for formula, df in self.data["formula_dfs"].items()
            }

        self.data["formula_metrics"] = {
            formula: single_formula_metrics(df, SMILES_db=smiles_list)
            for formula, (df, smiles_list) in stats_and_smiles.items()
        }

        if self.io:
            self.io.save_formula_metrics(self.data["formula_metrics"])
        return self.data["formula_metrics"]

    def _calc_global_metrics(self, size_weighted: bool = True) -> Dict[str, Any]:
        """Aggregates metrics across all formulas."""
        self.data["global_metrics"] = get_global_metrics(
            self.data["formula_metrics"].copy(), size_weighted=size_weighted
        )
        if self.io:
            self.io.save_global_metrics(self.data["global_metrics"].copy())
        return self.data["global_metrics"]

    def evaluate(self, args: dict):
        self._rollout(args["ac"], args["rollouts_from_file"])
        self._calc_features(args["features_from_file"], self.calc_dipole, self.f_max)
        self._get_metrics_by_formula()
        self._calc_global_metrics()

        logging.info(f"Global metrics: {self.data['global_metrics']}")

        if self.wandb_run:
            self.data["global_metrics"].update({"total_num_steps": args["total_num_steps"]})
            self.wandb_run.log(self.data["global_metrics"])


def launch_eval_jobs(
    ac: AbstractActorCritic,
    evaluator: SingleCheckpointEvaluator,
    n_batches: int = None,
    batch_size: int = 512,
    cf: dict = None,
):

    # Update basedir for evaluator_io
    total_num_steps = n_batches * batch_size  # counts the number of (single environment) env.steps()
    # Note that it is currently not compatible with pure pretraining step counting
    # cf['pretrain_every_k_iter'] can tell us how much it has been pretrained.

    eval_save_dir = os.path.join(cf["results_dir"], "BIG_EVAL", f"{total_num_steps}")
    evaluator.reset(eval_save_dir)

    job_args_list = [
        {
            "ac": ac,
            "total_num_steps": total_num_steps,
            "rollouts_from_file": False,
            "features_from_file": False,
        }
    ]

    if not cf["launch_eval_new_job"]:
        evaluator.evaluate(job_args_list[0])
        return None

    executor = submitit.AutoExecutor(folder="sublogs/" + cf["wandb_group"] + "_BIG_EVAL_" + str(total_num_steps))
    if executor._executor.__class__.__name__ == "SlurmExecutor":
        executor._command = "/home/energy/bjaha/miniconda3/envs/delight-rl/bin/python"
        executor.update_parameters(
            slurm_partition="sm3090",
            slurm_num_gpus=1,
            cpus_per_task=8,
            tasks_per_node=1,
            slurm_nodes=1,
            slurm_time="0-06:00:00",
        )
    elif executor._executor.__class__.__name__ == "LocalExecutor":
        executor.update_parameters(
            timeout_min=5,  # 5 mins
            gpus_per_node=0,
            cpus_per_task=2,
            nodes=1,
            tasks_per_node=1,
            mem_gb=4,
        )
    else:
        print(f"Unknown executor type: {executor._executor.__class__.__name__}")
        sys.exit(1)

    submit_jobs(
        submit_fn=evaluator.evaluate,
        executor=executor,
        parameter_dicts=job_args_list,
        ask_permission=False,
        use_submitit=True,
    )
