import logging
from typing import Tuple, List, Dict

import pandas as pd

from src.performance.energetics import EnergyUnit
from src.rl.reward import InteractionReward
from src.rl.env_container import SimpleEnvContainer
from src.rl.envs.environment import tmqmEnv, HeavyFirst
from src.rl.envs.env_no_reward import HeavyFirstNoReward
from src.rl.spaces import ObservationSpace, ActionSpace

from src.rl.reward import RaeReward
from src.tools import util

from src.data.reference_dataloader import ReferenceDataLoader
from src.data.io_handler import IOHandler
from src.data.data_util import get_benchmark_energies_from_df


class EnvMaker:
    """Class for creating the environments based on an external molecule dataset."""

    def __init__(self, cf: dict, split_method: str = "random", eval_formulas: list = None):
        self.cf = cf
        if "energy_unit" not in self.cf:
            self.cf["energy_unit"] = EnergyUnit.EV

        self.ref_data_loader = ReferenceDataLoader(data_dir="data")
        self.df = self._get_df()
        self._set_spaces()

        # Splitting procedure
        assert split_method in ["hardcoded", "random", "formula_statified", "read_split_from_disk", "eval_on_train"]
        if eval_formulas is not None or split_method == "hardcoded":
            assert split_method == "hardcoded" and eval_formulas is not None  # bi-implication

        self.split_method = split_method
        self.disjoint: bool = True

        self.eval_formulas = eval_formulas  # If formulas are hardcoded

        # Filters for formula_statified split
        self.eval_formula_criteria = {"n_atoms_min": 12, "n_atoms_max": 23, "min_isomers": 100, "max_isomers": 200}

        self.n_formulas_test: int = 10

    def _set_spaces(self) -> None:
        if self.cf["mol_dataset"] == "QM7":
            self.zs = [0, 1, 6, 7, 8, 16]
            self.canvas_size = 23
        elif self.cf["mol_dataset"] == "QM9":
            self.zs = [0, 1, 6, 7, 8, 9]
            self.canvas_size = 29
        else:
            raise ValueError(f"Unknown molecule dataset: {self.cf['mol_dataset']}")

        self.action_space = ActionSpace(zs=self.zs)
        self.observation_space = ObservationSpace(canvas_size=self.canvas_size, zs=self.zs)

    def get_spaces(self) -> Tuple[ObservationSpace, ActionSpace]:
        if hasattr(self, "observation_space") and hasattr(self, "action_space"):
            return self.observation_space, self.action_space
        else:
            self._set_spaces()
            return self.observation_space, self.action_space

    def _get_df(self) -> pd.DataFrame:
        if hasattr(self, "ref_data"):
            if hasattr(self.ref_data, "df"):
                return self.ref_data.df

        self.ref_data = self.ref_data_loader.load_and_polish(
            mol_dataset=self.cf["mol_dataset"], new_energy_unit=self.cf["energy_unit"], fetch_df=True
        )
        return self.ref_data.df

    def get_formula_data(self) -> dict:
        if hasattr(self, "formula_data"):
            return self.formula_data
        else:
            return self._make_data_dict()

    def make_envs(
        self,
    ) -> Tuple[SimpleEnvContainer, SimpleEnvContainer]:
        """Main class method which creates the training and evaluation environments."""

        self.formula_data = self._make_data_dict()

        training_envs, eval_envs, eval_envs_big = self._build_envs(self.formula_data)
        return training_envs, eval_envs, eval_envs_big

    def get_training_reward(self) -> InteractionReward:
        assert hasattr(self, "training_reward"), "Call make_envs() before get_training_reward()"
        return self.training_reward

    def get_reference_smiles(self, formulas: list) -> Dict[str, list]:
        """Returns a dictionary with formulas as keys and a list of SMILES as values."""
        df = self._get_df()
        return {f: df[(df["bag_repr"] == f)]["SMILES"].values.tolist() for f in formulas}

    def _make_data_dict(self) -> dict:
        df = self._get_df()
        df_train, df_eval = self._split_dataset(df)
        benchmark_energies_train = self._get_benchmark_energies(df_train)
        benchmark_energies_eval = self._get_benchmark_energies(df_eval)

        train_formulas = self._get_list_of_bag_reprs(df_train)
        eval_formulas = self._get_list_of_bag_reprs(df_eval)

        return dict(
            df_train=df_train,
            df_eval=df_eval,
            benchmark_energies_train=benchmark_energies_train,
            benchmark_energies_eval=benchmark_energies_eval,
            train_formulas=train_formulas,
            eval_formulas=eval_formulas,
        )

    def _get_list_of_bag_reprs(self, df: pd.DataFrame) -> list:
        return df["bag_repr"].unique().tolist()

    def _get_benchmark_energies(self, df: pd.DataFrame) -> Dict[str, float]:
        return get_benchmark_energies_from_df(df)

    def _split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataset into training and evaluation sets. TODO: Clean up this function."""

        if self.split_method == "random":
            # Select n_test eval molecules randomly
            df_eval = df.sample(self.n_formulas_test)
            df_train = df.drop(df_eval.index)

        elif self.split_method == "hardcoded":
            # eval_formulas = ['H5C4NO2', 'H4C4N2O', 'H8C6O', 'H6C4N2O']
            df_eval = df[df["bag_repr"].isin(self.eval_formulas)]
            df_train = df[~df["bag_repr"].isin(self.eval_formulas)] if self.disjoint else df

        elif self.split_method == "formula_statified":
            n_atoms_min = self.eval_formula_criteria["n_atoms_min"]
            n_atoms_max = self.eval_formula_criteria["n_atoms_max"]
            min_isomers = self.eval_formula_criteria["min_isomers"]
            max_isomers = self.eval_formula_criteria["max_isomers"]

            # Select subset viable for evaluation (n_atoms_min <= n_atoms <= n_atoms_max)
            df_filtered = df[(df["n_atoms"] >= n_atoms_min) & (df["n_atoms"] <= n_atoms_max)]

            # We sort them by isomer count to keep as many structures in the training set as possible
            # Sorting by isomer count yields highly unsaturated molecules. We want rings also, so we sample rather than taking tail.
            isomer_count = df_filtered["bag_repr"].value_counts()
            isomer_count = isomer_count[(isomer_count >= min_isomers) & (isomer_count <= max_isomers)]
            # Beware of the case where there are not enough isomers fulfilling the criteria in the filters above.
            isomer_count = isomer_count.sample(self.n_formulas_test)  # pandas.core.series.Series

            logging.info(f"eval_isomer_count: {isomer_count}")
            eval_formulas = pd.DataFrame(isomer_count).index.values.tolist()

            # Create train and eval sets
            df_eval = df_filtered[df_filtered["bag_repr"].isin(eval_formulas)]
            df_train = df[~df["bag_repr"].isin(eval_formulas)] if self.disjoint else df

        elif self.split_method == "read_split_from_disk":
            split = IOHandler.read_json(f"data/{self.cf['mol_dataset'].lower()}/processed/split.json")
            df_eval = df[df["bag_repr"].isin(split["test"])]
            df_train = df[~df["bag_repr"].isin(split["test"])] if self.disjoint else df

        return df_train, df_eval

    def _build_envs(self, data_dict: dict) -> Tuple[SimpleEnvContainer, SimpleEnvContainer, SimpleEnvContainer]:
        """Builds the environments and returns them in a SimpleEnvContainer."""

        benchmark_energies_train = data_dict["benchmark_energies_train"]
        benchmark_energies_eval = data_dict["benchmark_energies_eval"]
        train_formulas = data_dict["train_formulas"]
        eval_formulas = data_dict["eval_formulas"]

        all_benchmark_energies = benchmark_energies_train.copy()
        all_benchmark_energies.update(benchmark_energies_eval)
        safe_xtb = self.cf.get("safe_xtb", False)
        if "rew_rae" in self.cf["reward_coefs"]:
            rewards = RaeReward(
                reward_coefs=self.cf["reward_coefs"],
                relax_steps_final=self.cf["relax_steps_final"],
                benchmark_energies=all_benchmark_energies,
                energy_unit=self.cf["energy_unit"],
                xtb_mp=safe_xtb,
            )
            self.training_reward = rewards
        else:
            reward = InteractionReward(
                reward_coefs=self.cf["reward_coefs"],
                relax_steps_final=self.cf["relax_steps_final"],
                energy_unit=self.cf["energy_unit"],
                n_workers=self.cf["num_envs"],
                xtb_mp=safe_xtb,
            )
            eval_reward = InteractionReward(
                reward_coefs=self.cf["reward_coefs"],
                relax_steps_final=self.cf["relax_steps_final"],
                energy_unit=self.cf["energy_unit"],
            )
            eval_big_reward = InteractionReward(
                reward_coefs=self.cf["reward_coefs"],
                relax_steps_final=self.cf["relax_steps_final"],
                energy_unit=self.cf["energy_unit"],
            )
            self.training_reward = reward

        if self.cf["mol_dataset"] == "TMQM":
            RLEnvironment = tmqmEnv
        elif self.cf["mol_dataset"] == "QM7" or self.cf["mol_dataset"] == "QM9":
            RLEnvironment = HeavyFirst if self.cf.get("calc_rew", True) else HeavyFirstNoReward
        else:
            raise ValueError(f"Unknown molecule dataset: {self.cf['mol_dataset']}")

        online_formulas = [util.string_to_formula(f) for f in train_formulas]

        training_envs = []
        for i in range(self.cf["num_envs"]):
            training_envs.append(
                RLEnvironment(
                    reward=reward,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    formulas=online_formulas,
                    min_atomic_distance=self.cf["min_atomic_distance"],
                    max_solo_distance=self.cf["max_solo_distance"],
                    min_reward=self.cf["min_reward"],
                    energy_unit=self.cf["energy_unit"],
                    worker_id=i,
                )
            )
        training_envs = SimpleEnvContainer(training_envs)
        logging.info(f"Number of training bags: {len(online_formulas)}")

        benchmark_energies_eval_list = [benchmark_energies_eval[f] for f in eval_formulas]
        eval_envs = SimpleEnvContainer(
            [
                RLEnvironment(
                    reward=eval_reward,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    formulas=[util.string_to_formula(eval_formulas[i])],
                    min_atomic_distance=self.cf["min_atomic_distance"],
                    max_solo_distance=self.cf["max_solo_distance"],
                    min_reward=self.cf["min_reward"],
                    energy_unit=self.cf["energy_unit"],
                    benchmark_energy=[benchmark_energies_eval_list[i]],
                )
                for i in range(len(eval_formulas))
            ]
        )
        logging.info(f"Number of evaluation environments: {eval_envs.get_size()}")
        logging.info(f"eval formulas: {eval_formulas}")

        eval_envs_big = SimpleEnvContainer(
            [
                HeavyFirstNoReward(
                    reward=eval_big_reward,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    formulas=[util.string_to_formula(eval_formulas[i])],
                    min_atomic_distance=self.cf["min_atomic_distance"],
                    max_solo_distance=self.cf["max_solo_distance"],
                    min_reward=self.cf["min_reward"],
                    energy_unit=self.cf["energy_unit"],
                    benchmark_energy=[benchmark_energies_eval_list[i]],
                )
                for i in range(len(eval_formulas))
            ]
        )

        return training_envs, eval_envs, eval_envs_big


class EnvMakerNoRef:
    """Class for creating the environments from formulas only (no dataset load).

    - deploy=True: only eval envs, returns (None, eval_envs). Use for inference.
    - deploy=False: train + eval envs from the same formula list (finetuning), returns
      (training_envs, eval_envs, eval_envs_big). Formulas need not belong to any dataset.
    """

    def __init__(
        self,
        cf: dict,
        train_formulas: List[str] = None,
        eval_formulas: List[str] = None,
        eval_only: bool = False,
        action_space: ActionSpace = None,
        observation_space: ObservationSpace = None,
    ):
        self.cf = cf
        if "energy_unit" not in self.cf:
            self.cf["energy_unit"] = EnergyUnit.EV

        self.train_formulas = train_formulas or []
        self.eval_formulas = eval_formulas or train_formulas or []
        self.eval_only = eval_only
        self.action_space = action_space
        self.observation_space = observation_space

    def make_envs(self) -> Tuple[SimpleEnvContainer, SimpleEnvContainer, SimpleEnvContainer]:
        reward = InteractionReward(
            reward_coefs=self.cf["reward_coefs"],
            relax_steps_final=self.cf["relax_steps_final"],
            energy_unit=self.cf["energy_unit"],
        )

        eval_envs = SimpleEnvContainer(
            [
                HeavyFirstNoReward(
                    reward=reward,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    formulas=[util.string_to_formula(f)],
                    min_atomic_distance=self.cf["min_atomic_distance"],
                    max_solo_distance=self.cf["max_solo_distance"],
                    min_reward=self.cf["min_reward"],
                    energy_unit=self.cf["energy_unit"],
                )
                for f in self.eval_formulas
            ]
        )

        if self.eval_only:
            return None, eval_envs, None

        # Finetuning: build both train and eval envs from the same formula list
        formulas = self.train_formulas if self.train_formulas else self.eval_formulas
        if not formulas:
            raise ValueError("EnvMakerNoRef with deploy=False requires train_formulas or eval_formulas.")

        num_envs = self.cf.get("num_envs", 1)
        calc_rew = self.cf.get("calc_rew", True)
        RLEnvironment = HeavyFirst if calc_rew else HeavyFirstNoReward

        reward_train = InteractionReward(
            reward_coefs=self.cf["reward_coefs"],
            relax_steps_final=self.cf["relax_steps_final"],
            energy_unit=self.cf["energy_unit"],
            n_workers=num_envs,
            xtb_mp=self.cf.get("safe_xtb", False),
        )
        eval_reward = InteractionReward(
            reward_coefs=self.cf["reward_coefs"],
            relax_steps_final=self.cf["relax_steps_final"],
            energy_unit=self.cf["energy_unit"],
        )
        eval_big_reward = InteractionReward(
            reward_coefs=self.cf["reward_coefs"],
            relax_steps_final=self.cf["relax_steps_final"],
            energy_unit=self.cf["energy_unit"],
        )

        online_formulas = [util.string_to_formula(f) for f in formulas]
        training_envs = SimpleEnvContainer(
            [
                RLEnvironment(
                    reward=reward_train,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    formulas=online_formulas,
                    min_atomic_distance=self.cf["min_atomic_distance"],
                    max_solo_distance=self.cf["max_solo_distance"],
                    min_reward=self.cf["min_reward"],
                    energy_unit=self.cf["energy_unit"],
                    worker_id=i,
                )
                for i in range(num_envs)
            ]
        )
        logging.info(f"Number of training bags (formulas only): {len(formulas)}")

        eval_formula_list = self.eval_formulas if self.eval_formulas else formulas

        eval_envs = SimpleEnvContainer(
            [
                RLEnvironment(
                    reward=eval_reward,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    formulas=[util.string_to_formula(eval_formula_list[i])],
                    min_atomic_distance=self.cf["min_atomic_distance"],
                    max_solo_distance=self.cf["max_solo_distance"],
                    min_reward=self.cf["min_reward"],
                    energy_unit=self.cf["energy_unit"],
                    benchmark_energy=[None],
                )
                for i in range(len(eval_formula_list))
            ]
        )
        eval_envs_big = SimpleEnvContainer(
            [
                HeavyFirstNoReward(
                    reward=eval_big_reward,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    formulas=[util.string_to_formula(eval_formula_list[i])],
                    min_atomic_distance=self.cf["min_atomic_distance"],
                    max_solo_distance=self.cf["max_solo_distance"],
                    min_reward=self.cf["min_reward"],
                    energy_unit=self.cf["energy_unit"],
                    benchmark_energy=[None],
                )
                for i in range(len(eval_formula_list))
            ]
        )
        logging.info(f"Number of evaluation environments (formulas only): {len(eval_formula_list)}")
        logging.info(f"eval formulas: {eval_formula_list}")
        return training_envs, eval_envs, eval_envs_big
