import abc
import itertools
import logging
from typing import Tuple, List

import ase.data
import gymnasium as gym
import numpy as np
from ase import Atoms, Atom

from src.rl.reward import InteractionReward
from src.rl.spaces import ActionSpace, ObservationSpace, ActionType, ObservationType, FormulaType
from src.tools.util import remove_atom_from_formula, get_formula_size


class AbstractMolecularEnvironment(gym.Env, abc.ABC):
    # Negative reward should be on the same order of magnitude as the positive ones.
    # Memory agent on QM9: mean 0.26, std 0.13, min -0.54, max 1.23 (negative reward indeed possible
    # but avoidable and probably due to PM6)

    def __init__(
        self,
        reward: InteractionReward,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        min_atomic_distance=0.6,  # Angstrom
        max_solo_distance=2.0,  # Angstrom
        min_reward=-0.6,  # Hartree
        seed=0,
        energy_unit="eV",
        worker_id=None,
    ):

        self.constant_reward = 0.1

        self.reward = reward
        self.observation_space = observation_space
        self.action_space = action_space

        self.random_state = np.random.RandomState(seed=seed)

        self.min_atomic_distance = min_atomic_distance
        self.max_solo_distance = max_solo_distance
        self.min_reward = min_reward
        self.energy_unit = energy_unit

        self.current_atoms = Atoms()
        self.current_formula: FormulaType = tuple()

        self.use_intermediate_rewards = np.any(
            [name in set(reward.intermediate_rew_terms) for name in reward.reward_names]
        )

        self.worker_id = worker_id

        # if self.use_intermediate_rewards:
        #     intermediate_rew_coefs = reward.reward_coefs.copy()
        #     intermediate_rew_coefs.pop('rew_valid')
        #     self.intermediate_reward = InteractionReward(
        #         reward_coefs=intermediate_rew_coefs,
        #         relax_steps_final=reward.relax_steps_final,
        #         energy_unit=reward.energy_unit
        #     )

    @abc.abstractmethod
    def reset(self) -> ObservationType:
        raise NotImplementedError

    def _reward_logic(
        self, new_atom: Atom, terminal: bool, use_intermediate_rewards: bool = False
    ) -> Tuple[float, bool, dict]:
        done = True if terminal else False
        if not use_intermediate_rewards and not terminal:
            return self.constant_reward, done, {}

        reward, info = self.reward.calculate(self.current_atoms, new_atom, terminal, self.worker_id)
        if terminal:
            info["termination_info"] = "full_formula"

        if reward is None:
            return self.min_reward, True, info

        # if we are considering an intermediate reward
        # TODO: experiment with allowing one or more steps after invalid action,
        # i.e. initialize counter
        # reward = reward / 5

        return reward, done, info

    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, dict]:
        atomic_number_index, _ = action
        atomic_number = self.action_space.zs[atomic_number_index]
        done = atomic_number == 0

        if done:
            return (
                self.observation_space.build(self.current_atoms, self.current_formula),
                0.0,
                done,
                {"termination_info": "stop_token"},
            )

        new_atom = self.action_space.to_atom(action)
        if not self._is_valid_action(current_atoms=self.current_atoms, new_atom=new_atom):
            return (
                self.observation_space.build(self.current_atoms, self.current_formula),
                self.min_reward,
                True,
                {"termination_info": "invalid_action"},
            )

        is_terminal = self._is_terminal(lag=1)
        reward, done, info = self._reward_logic(new_atom, is_terminal, self.use_intermediate_rewards)

        self.current_atoms.append(new_atom)
        self.current_formula = remove_atom_from_formula(self.current_formula, atomic_number)

        # Check if state is terminal
        if self._is_terminal(lag=0):
            done = True

        return self.observation_space.build(self.current_atoms, self.current_formula), reward, done, info

    def _is_terminal(self, lag: int = 1) -> bool:
        return (
            len(self.current_atoms) + lag == self.observation_space.canvas_space.size
            or get_formula_size(self.current_formula) == lag
        )

    def _is_valid_action(self, current_atoms: Atoms, new_atom: Atom) -> bool:
        if self._is_too_close(current_atoms, new_atom):
            return False

        return self._all_covered(current_atoms, new_atom)

    def _is_too_close(self, existing_atoms: Atoms, new_atom: Atom) -> bool:
        # Check distances between new and old atoms
        for existing_atom in existing_atoms:
            if np.linalg.norm(existing_atom.position - new_atom.position) < self.min_atomic_distance:
                logging.debug("Atoms are too close")
                return True

        return False

    # def _calculate_reward(self, new_atom: Atom) -> Tuple[float, dict]:
    #     return self.reward.calculate(self.current_atoms, new_atom)

    # def _calculate_intermediate_reward(self, new_atom: Atom) -> Tuple[float, dict]:
    #     reward, info = self.intermediate_reward.calculate(self.current_atoms, new_atom)
    #     return (reward if reward else self.min_reward, info)

    def _all_covered(self, existing_atoms: Atoms, new_atom: Atom) -> bool:
        # Ensure that certain atoms are not too far away from the nearest heavy atom to avoid H2, F2,... formation
        candidates = ["H", "F", "Cl", "Br"]
        if len(existing_atoms) == 0 or new_atom.symbol not in candidates:
            return True

        for existing_atom in existing_atoms:
            if existing_atom.symbol in candidates:
                continue

            distance = np.linalg.norm(existing_atom.position - new_atom.position)
            if distance < self.max_solo_distance:
                return True

        logging.debug("There is a single atom floating around")
        return False

    def render(self, mode="human"):
        pass

    def seed(self, seed=None) -> int:
        seed = seed or np.random.randint(int(1e5))
        self.random_state = np.random.RandomState(seed)
        return seed


class MolecularEnvironment(AbstractMolecularEnvironment):
    def __init__(self, formulas: List[FormulaType], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.formulas = formulas
        self.formula_cycle = itertools.cycle(self.formulas)
        self.reset()

    def reset(self) -> ObservationType:
        self.current_atoms = Atoms()
        self.current_formula = next(self.formula_cycle)

        self.step()
        return self.observation_space.build(self.current_atoms, self.current_formula)


class HeavyFirst(AbstractMolecularEnvironment):
    def __init__(self, formulas: List[FormulaType], benchmark_energy: List[float] = [None], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.formulas = formulas
        self.formula_cycle = itertools.cycle(self.formulas)

        self.benchmark_energies = benchmark_energy
        self.benchmark_energy_cycle = itertools.cycle(self.benchmark_energies)

        self.formula_counter = 0
        self.reshuffle_formula_list()

        self.first_atom = "heavy_first"  # 'any_non_hydro', None
        self.obs_reset = self.reset()

    def reset(self) -> ObservationType:
        if self.formula_counter == len(self.formulas):
            self.reshuffle_formula_list()

        self.current_atoms = Atoms()
        self.current_formula = next(self.formula_cycle)
        self.benchmark_energy = next(self.benchmark_energy_cycle)

        self.reward.reset_old_energies(self.worker_id)  # self.reward.reset_old_energy()

        elements = [z for (z, _) in self.current_formula]

        if self.first_atom is None:
            return self.observation_space.build(self.current_atoms, self.current_formula)
        elif self.first_atom == "heavy_first":
            first_atom_index = self.action_space.zs.index(max(elements))
        elif self.first_atom == "any_non_hydro":
            first_atom_index = np.random.choice([i for i, z in enumerate(elements) if z != 1])
        else:
            raise ValueError(f"Unknown first atom type: {self.first_atom}")

        # Take a step to add the first atom on the canvas
        obs, _, _, _ = self.step(action=(first_atom_index, (0, 0, 0)))

        return obs

    def _is_valid_action(self, current_atoms: Atoms, new_atom: Atom) -> bool:
        if self._is_too_close(current_atoms, new_atom):
            return False
        return True

    def reshuffle_formula_list(self):
        self.formula_counter = 0
        random_permutation = np.random.permutation(len(self.formulas))
        self.formulas = [self.formulas[i] for i in random_permutation]
        self.formula_cycle = itertools.cycle(self.formulas)

        if self.benchmark_energies[0] is not None:
            self.benchmark_energies = [self.benchmark_energies[i] for i in random_permutation]
            self.benchmark_energy_cycle = itertools.cycle(self.benchmark_energies)


class tmqmEnv(AbstractMolecularEnvironment):
    transition_metals = [
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]

    transition_metal_numbers = [ase.data.atomic_numbers[tm] for tm in transition_metals]

    def __init__(self, formulas: List[FormulaType], benchmark_energy: List[float] = [None], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.formulas = formulas
        self.formula_cycle = itertools.cycle(self.formulas)
        self.benchmark_energy_cycle = itertools.cycle(benchmark_energy)

        self.obs_reset = self.reset()

    def reset(self) -> ObservationType:
        self.current_atoms = Atoms()
        self.current_formula = next(self.formula_cycle)
        self.benchmark_energy = next(self.benchmark_energy_cycle)
        self.reward.reset_old_energies(self.worker_id)  # self.reward.reset_old_energy()

        # Take a step to add the TM to the canvas
        tm_element = [z for (z, count) in self.current_formula if z in self.transition_metal_numbers][0]
        tm_atomic_number_index = self.action_space.zs.index(tm_element)
        obs, _, _, _ = self.step(action=(tm_atomic_number_index, (0, 0, 0)))

        return obs

    def _is_valid_action(self, current_atoms: Atoms, new_atom: Atom) -> bool:
        if self._is_too_close(current_atoms, new_atom):
            return False
        return True
