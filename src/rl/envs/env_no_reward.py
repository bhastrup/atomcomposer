from typing import Tuple


from ase import Atoms

from src.rl.spaces import ActionType, ObservationType
from src.tools.util import remove_atom_from_formula
from src.rl.envs.environment import HeavyFirst


class HeavyFirstNoReward(HeavyFirst):
    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, dict]:
        atomic_number_index, position = action
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

        self.current_atoms.append(new_atom)
        self.current_formula = remove_atom_from_formula(self.current_formula, atomic_number)

        # Check if state is terminal
        if self._is_terminal(lag=0):
            done = True

        reward = 0
        info = {}

        return self.observation_space.build(self.current_atoms, self.current_formula), reward, done, info

    def reset(self) -> ObservationType:
        if self.formula_counter == len(self.formulas):
            self.reshuffle_formula_list()

        self.current_atoms = Atoms()
        self.current_formula = next(self.formula_cycle)
        self.benchmark_energy = next(self.benchmark_energy_cycle)
        # self.reward.reset_old_energies(self.worker_id)

        # Take a step to add the heaviest atom on the canvas
        heaviest = max([z for (z, _) in self.current_formula])
        heaviest_number_index = self.action_space.zs.index(heaviest)
        obs, _, _, _ = self.step(action=(heaviest_number_index, (0, 0, 0)))

        return obs
