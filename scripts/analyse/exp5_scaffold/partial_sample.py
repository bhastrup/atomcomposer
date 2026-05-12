"""
Scaffold-functionalization task: use a trained agent to complete a partial molecule
(core scaffold + bag of atoms to place). Define the scaffold and run in CONFIG below.
"""

import itertools
import time
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import chemical_symbols

from src.tools import util
from src.data.io_handler import IOHandler
from src.data.reference_dataloader import ReferenceDataLoader
from src.performance.energetics import EnergyUnit
from src.performance.single_cpkt.utils import (
    PathHelper,
    process_config,
    get_model,
)
from src.performance.single_cpkt.evaluator import (
    EvaluatorIO,
    SingleCheckpointEvaluator,
)
from src.rl.env_container import SimpleEnvContainer
from src.rl.envs.env_partial_canvas import AbstractMolecularEnvironment
from src.rl.reward import InteractionReward
from src.rl.envs.env_partial_canvas import FormulaType


class PartialCanvasEnvFixed(AbstractMolecularEnvironment):
    """Single scaffold + fixed bag to place; one formula per env."""

    def __init__(self, molecule_df: pd.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(molecule_df) == 1
        self.molecule_df = molecule_df
        # Full molecule formula (bag tuple) for evaluator save key
        self.formulas = [molecule_df["formulas"].iloc[0]]
        self.new_bag = molecule_df["new_bag"].iloc[0]
        indices = np.arange(len(self.molecule_df))
        self.index_cycle = itertools.cycle(indices)
        self.obs_reset = self.reset()

    def reset(self):
        index = next(self.index_cycle)
        mol = self.molecule_df.iloc[index]
        pos = np.array(mol["pos"])
        elements = np.array(mol["atomic_nums"])
        atoms_to_remove = mol["atoms_to_remove"]

        sorted_indices = np.array([i for i in range(len(elements)) if i not in atoms_to_remove])
        sorted_indices = np.concatenate((sorted_indices, np.array(atoms_to_remove))).astype(int)

        obs_tuple = self._construct_observation_tuple(
            pos, elements, sorted_indices, len(atoms_to_remove), new_bag=deepcopy(self.new_bag)
        )
        self.current_atoms, self.current_formula = obs_tuple
        return self.observation_space.build(self.current_atoms, self.current_formula)

    def _construct_observation_tuple(
        self,
        pos,
        elements,
        sorted_indices,
        num_atoms_to_place: int = 1,
        new_bag: FormulaType = None,
    ) -> Tuple[Atoms, FormulaType]:
        num_core = len(elements) - num_atoms_to_place
        elements = np.array(elements)
        pos_new = pos[sorted_indices[:num_core]]
        zs_new = elements[sorted_indices[:num_core]]
        canvas_atoms = Atoms(numbers=zs_new, positions=pos_new)
        return (canvas_atoms, new_bag)


def load_reference_df(data_dir: str, mol_dataset: str = "QM7") -> pd.DataFrame:
    """Load and polish reference molecule dataframe. Reuse for many write_atoms_to_remove calls."""
    loader = ReferenceDataLoader(data_dir=data_dir)
    ref_data = loader.load_and_polish(
        mol_dataset=mol_dataset,
        new_energy_unit=EnergyUnit.EV,
        fetch_df=True,
    )
    ref_data.df = ref_data.df.reset_index(drop=True)
    return ref_data.df


def dict_to_bag(spec: dict, zs: List[int]) -> FormulaType:
    """Convert dict of symbol -> count to FormulaType (same z order as space). zs e.g. [0,1,6,7,8,16]."""
    out = []
    for z in zs:
        if z == 0:
            out.append((0, 0))
        else:
            sym = chemical_symbols[z]
            out.append((z, spec.get(sym, 0)))
    return tuple(out)


def build_scaffold_tasks(cfg: dict, zs: List[int]):
    """
    Build env-ready molecule_dfs from CONFIG['scaffold_tasks'].
    Config: scaffold_tasks[dataset_name] = [ task, ... ]; each task has mol_index, atoms_to_remove, new_bags.
    Each yielded df has atoms_to_remove, new_bag, and formulas (canvas + new_bag) set for PartialCanvasEnvFixed.
    """
    data_dir = cfg.get("data_dir", "data")

    for mol_dataset, tasks in cfg["scaffold_tasks"].items():
        ref_df = load_reference_df(data_dir=data_dir, mol_dataset=mol_dataset)
        for task in tasks:
            mol_index = task["mol_index"]
            atoms_to_remove = task["atoms_to_remove"]
            new_bags_specs = task.get("new_bags") or []  # list of N dicts e.g. {"H": 2, "C": 1}

            n_atoms = len(ref_df.iloc[mol_index]["atomic_symbols"])
            valid = sorted(i for i in atoms_to_remove if 0 <= i < n_atoms)
            molecule_df = ref_df.iloc[[mol_index]].copy()
            molecule_df["atoms_to_remove"] = [valid]

            for spec in new_bags_specs:
                mdf = molecule_df.copy()
                mdf["new_bag"] = [dict_to_bag(spec, zs)]
                yield mdf


CONFIG = {
    # Model
    "run_dir": "runs/nat-com-training/A/seed_0",
    "model_name": "pretrain_run-0_CP-8_steps-20000.model",
    "log_name": "pretrain_run-0.json",
    "tag": "scaffold_eval",
    "data_dir": "data",
    "scaffold_tasks": {
        # "QM7": [
        #     {
        #         "mol_index": 369, # Benzene: 0-5 is carbon, 6-11 is hydrogen
        #         "atoms_to_remove": [6],
        #         "new_bags": [
        #             {"H": 1},
        #             {"H": 2, "O": 1},
        #         ],
        #     },
        # ],
        "QM9": [
            {
                "mol_index": 450,
                "atoms_to_remove": [3, 4, 6, 7, 8, 9],
                "new_bags": [
                    {"H": 4, "C": 2},
                    {"H": 6, "C": 2},
                    {"H": 6, "C": 3},
                    {"H": 8, "C": 3},
                    {"H": 6, "C": 4},
                    {"H": 8, "C": 4},
                    {"H": 10, "C": 4},
                ],
            },
        ],
    },
    # Sampling
    "num_episodes": 5,
    "num_envs": 1,
    "relax": True,
    "out_dir": None,
}

if __name__ == "__main__":
    cfg = CONFIG

    # (a) Model
    ph = PathHelper(cfg["run_dir"], cfg["model_name"], cfg["log_name"], tag=cfg["tag"])
    config = process_config(IOHandler.read_json(ph.log_path))
    model, start_num_steps = get_model(config, ph)
    print(f"Successfully loaded model from {ph.cp_path}")

    util.set_seeds(seed=config["seed"])
    action_space = model.action_space
    observation_space = model.observation_space
    zs_model = model.action_space.zs
    if zs_model != [0, 1, 6, 7, 8, 16]:
        raise ValueError(f"Model action space zs {zs_model} does not match QM7 zs [0, 1, 6, 7, 8, 16]")
    print(f"Model action space zs: {zs_model}")

    # (b) Build one env per (molecule_df, new_bag, label) from scaffold tasks
    reward_coefs = {"rew_abs_E": 1.0, "rew_valid": 3.0}
    reward = InteractionReward(reward_coefs=reward_coefs)

    # One env per (scaffold, new_bag); each mdf is env-ready (atoms_to_remove, new_bag, formulas).
    env_list = []
    for mdf in build_scaffold_tasks(cfg, zs=zs_model):
        env = PartialCanvasEnvFixed(
            reward=reward,
            observation_space=observation_space,
            action_space=action_space,
            molecule_df=mdf,
            min_atomic_distance=0.6,
            max_solo_distance=2.0,
            min_reward=-3,
        )
        env_list.append(env)
    eval_envs = SimpleEnvContainer(env_list)

    print(f"Number of environments: {len(eval_envs.environments)}")

    evaluator = SingleCheckpointEvaluator(
        eval_envs=eval_envs,
        reference_smiles=None,
        benchmark_energies=None,
        io_handler=EvaluatorIO(base_dir=ph.eval_save_dir),
        num_episodes_const=cfg.get("num_episodes", 50),
        prop_factor=None,
    )
    evaluator.reset(ph.eval_save_dir)
    print(f"Created evaluator with IO base directory: {evaluator.io.base_dir}")

    # (c) Sample and relax
    t0 = time.time()
    evaluator._rollout(ac=model, rollouts_from_file=False)
    print(f"Rollout done in {time.time() - t0:.1f}s")
    t0 = time.time()
    evaluator._calc_features(features_from_file=False, perform_optimization=cfg["relax"])
    print(f"Features (relax={cfg['relax']}) done in {time.time() - t0:.1f}s")
    evaluator._get_metrics_by_formula(dfs_from_file=False)

    print(f"Metrics done in {time.time() - t0:.1f}s")

    # Best by energy (relaxed if available)
    for formula, df in evaluator.data["formula_dfs"].items():
        df = df.dropna(subset=["abs_energy"])
        if len(df) == 0:
            print(f"No valid energies for {formula}")
            continue
        energy_col = "abs_energy_relaxed" if cfg["relax"] and "abs_energy_relaxed" in df.columns else "abs_energy"
        df_sorted = df.sort_values(energy_col)
        best = df_sorted.iloc[0]
        print(f"Formula {formula}: best {energy_col}={best[energy_col]:.4f} eV, SMILES={best.get('SMILES', 'N/A')}")
        out_csv = cfg["out_dir"] / formula / "df.csv"
        if out_csv.parent.exists():
            df.to_csv(out_csv, index=False)
    print(f"Results under {cfg['out_dir']}")
