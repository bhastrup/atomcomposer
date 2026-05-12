from typing import List
from pathlib import Path

import numpy as np
from rdkit import Chem

from src.tools import util
from src.rl.spaces import ObservationType
from src.rl.buffer import DynamicPPOBuffer
from src.agents.base import AbstractActorCritic
from src.rl.env_container import SimpleEnvContainer
from src.performance.metrics import get_compact_smiles
from src.performance.cumulative.cum_io import CumulativeIO
from src.performance.cumulative.performance_summary import Logger
from src.performance.cumulative.storage import FormulaData, MolCandidate


def obs_to_positions(obs: ObservationType) -> np.ndarray:
    canvas = obs[0]
    return np.array([pos for _, pos in canvas if _ > 0])


def obs_to_elements(obs: ObservationType, zs: List[int]) -> List[int]:
    canvas = obs[0]
    elements, _ = zip(*[(zs[index], position) for index, position in canvas if index > 0])
    return elements


def observation_to_bag_repr(obs: ObservationType, zs: List[int]) -> str:
    return util.elements_to_str_formula(obs_to_elements(obs, zs))


def get_termination_flag(info: dict) -> str:
    if info.get("termination_info") == "invalid_action":
        return "invalid_action"
    elif info.get("termination_info") == "full_formula":
        return info.get("mol_info", {}).get("info")


def extract_bag_reprs(env_container: SimpleEnvContainer) -> List[str]:
    all_bags = set()
    for env in env_container.environments:
        for f in env.formulas:
            bag_repr = util.bag_tuple_to_str_formula(f)
            if bag_repr not in all_bags:
                all_bags.add(bag_repr)
    return list(all_bags)


class CumulativeDiscoveryTracker(Logger):
    """
    Logger class for tracking the cumulative discovery for a multibag agent across training.

    Logs all completed episodes throughout training and stores every single valid molecule into a
    a data object for that formula. This object keeps track of all unique SMILES strings discovered so far.
    Under each SMILES string it then stores the molecule (3D position, energy, and rewards and so on).


    """

    def __init__(
        self,
        cf: dict,
        model: AbstractActorCritic,
        env_container_train: SimpleEnvContainer,
        env_container_eval: SimpleEnvContainer,
        start_num_iter: int = 0,
    ):
        super().__init__(cf, model, start_num_iter)  # initializes wandb_run

        # Get spaces for reconstruction of molecules
        self.action_space = env_container_train.environments[0].action_space
        self.observation_space = env_container_train.environments[0].observation_space
        self.zs = self.action_space.zs

        # Initialize IO handler
        self.cum_io = CumulativeIO(save_dir=Path.cwd() / cf["results_dir"])
        # Initialize storage structures
        self.formulas_is = extract_bag_reprs(env_container_train)  # in sample formulas
        self.formulas_oos = extract_bag_reprs(env_container_eval)  # out of sample formulas
        self.cum_io.dump_bag_reprs(dict(in_sample=self.formulas_is, oos_sample=self.formulas_oos))
        self.reset_db_big()
        self.reset_db_small()

        self.RL_info = []

    def reset_db_small(self) -> None:
        # Dump data into big db
        if self.get_num_steps() > 0 and hasattr(self, "db_small"):
            self.db_big.update({self.get_num_steps(): self.db_small})

        # Reset small db
        self.db_small = dict(
            in_sample={formula: FormulaData() for formula in self.formulas_is},
            oos_sample={formula: FormulaData() for formula in self.formulas_oos},
        )
        self.n_eps_current_rollout = 0

    def reset_db_big(self) -> None:
        self.db_big = {}

    def save_episode_RL(self, state: ObservationType, total_reward: float, info: dict, name: str) -> None:
        self.n_eps_current_rollout += 1
        termination_info = get_termination_flag(info)
        self.RL_info.append(termination_info)
        # if self.n_eps_current_rollout == 2:
        #     exit()
        if not termination_info == "valid":
            return  # Skip invalid molecules

        if name == "train":
            db = self.db_small["in_sample"]
        elif name == "eval":
            db = self.db_small["oos_sample"]

        # Get formula and formula data
        formula = observation_to_bag_repr(state, self.zs)
        if formula not in self.formulas_is and formula not in self.formulas_oos:
            print(f"Formula {formula} not found in formulas_is or formulas_oos.")
            print(f"info: {info}")
            print(f"in sample formulas: {self.formulas_is}")
            print(f"out of sample formulas: {self.formulas_oos}")
            print("exiting...")
            exit()
        else:
            formula_data = db[formula]

        # Get SMILES string
        smiles = get_compact_smiles(Chem.MolToSmiles(info["mol_info"]["mol"], canonical=True, isomericSmiles=False))

        # if smiles not in formula_data.unique_smiles:
        #     formula_data.unique_smiles.add(smiles)
        #     formula_data.molecules[smiles] = []

        if smiles not in formula_data.molecules:
            formula_data.molecules[smiles] = []

        # Append candidate to formula data
        candidate = MolCandidate(
            num_env_steps=self.get_num_steps(),
            elements=obs_to_elements(state, self.zs),
            pos=obs_to_positions(state),
            reward=total_reward,
            energy=info["metrics"]["final:AE"],
            energy_relaxed=info.get("energy_relaxed", None),
            rae=info["metrics"]["final:RAE"],
        )
        formula_data.molecules[smiles].append(candidate)

    def save_rollout_and_info(
        self,
        info_saver: util.InfoSaver,
        rollout_saver: util.RolloutSaver,
        save_rollout: bool,
        rollout: dict,
        buffer: DynamicPPOBuffer,
        name: str,
        total_num_iter: int,
    ):
        assert total_num_iter == self.total_num_iter, "total_num_iter != self.total_num_iter"
        super().save_rollout_and_info(
            info_saver=info_saver,
            rollout_saver=rollout_saver,
            save_rollout=False,
            rollout=rollout,
            buffer=buffer,
            name=name,
            total_num_iter=total_num_iter,
        )

        self.reset_db_small()

        if self.total_num_iter % 100 == 0:
            self.cum_io.dump_current_db(db_big=self.db_big, steps=self.get_num_steps())
            self.reset_db_big()
