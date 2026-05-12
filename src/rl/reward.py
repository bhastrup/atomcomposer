import abc
import time
from typing import Tuple, Dict, Union, Optional

import numpy as np
from ase import Atoms, Atom

from src.data.reference_dataloader import ReferenceDataLoader
from src.performance.energetics import EnergyUnit, EnergyConverter, XTBOptimizer
from src.performance.metrics import MoleculeAnalyzer

from src.tools.util import symbols_to_str_formula


class MolecularReward(abc.ABC):
    @abc.abstractmethod
    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        raise NotImplementedError

    @staticmethod
    def get_minimum_spin_multiplicity(atoms: Atoms) -> int:
        return atoms.numbers.sum() % 2 + 1

    @staticmethod
    def get_spin(atoms) -> int:
        return MolecularReward.get_minimum_spin_multiplicity(atoms) - 1


class InteractionReward(MolecularReward):
    intermediate_rew_terms = ["rew_formation"]
    terminal_rew_terms = [
        "rew_formation",
        "rew_atomisation",
        "rew_valid",
        "rew_basin",
        "rew_rae",
        "rew_dipole",
    ]

    reward_functions_available = {
        "rew_formation": "_formation",
        "rew_atomisation": "_atomisation",
        "rew_valid": "_validity_rew",
        "rew_basin": "_basin_dist_rew",
        "rew_dipole": "_dipole_rew",
    }

    def __init__(
        self,
        reward_coefs: Dict = {},
        calculator: str = "GFN2-xTB",
        relax_steps_final: int = 0,
        energy_unit: str = EnergyUnit.EV,
        use_exponential: bool = True,
        n_workers: int = 1,
        xtb_mp: bool = False,
    ) -> None:
        # Sometimes XTB crashes when calculating the energy of single nitrogen atom
        E_nitrogen_eV = -71.006818
        E_nitrogen = EnergyConverter.convert(E_nitrogen_eV, EnergyUnit.EV, energy_unit)
        self.atom_energies: Dict[str, float] = {"N": E_nitrogen}

        self.calculator = calculator
        self.energy_unit = energy_unit
        self.relax_steps_final = relax_steps_final

        self.reward_coefs = reward_coefs
        self.reward_names = list(reward_coefs.keys())
        self.use_exponential = use_exponential

        self.calc = XTBOptimizer(energy_unit=self.energy_unit, use_mp=xtb_mp)
        self.analyzer = MoleculeAnalyzer(use_huckel=True)

        self.old_energies = {f"{n}": 0.0 for n in range(n_workers)}
        self.old_energies["eval"] = 0.0

        self.benchmark_energies = (
            ReferenceDataLoader().load_and_polish("qm7", energy_unit, fetch_df=False).get_mean_energies()
        )

    def reset_old_energies(self, worker_id):
        self.old_energies[worker_id] = 0.0

    def calculate(self, atoms: Atoms, new_atom: Atom, terminal: bool = False, worker_id: int = 1) -> Tuple[float, dict]:
        info = {}
        start = time.time()
        all_atoms = atoms.copy()
        all_atoms.append(new_atom)
        n_atoms = len(all_atoms)

        # Intersection of self.reward_names and either intermediate_rew_terms or terminal_rew_terms
        rew_names = list(
            set(self.reward_names) & set(self.terminal_rew_terms if terminal else self.intermediate_rew_terms)
        )

        if terminal and self.relax_steps_final > 0:
            opt_info = self._optimize_atoms(all_atoms, max_steps=self.relax_steps_final)
            all_atoms = opt_info["new_atoms"]
            e_tot = opt_info["energy_after"]
        else:
            e_tot = self._calculate_energy(all_atoms)

        if e_tot is None:
            return (None, {})

        # Prepare arguments for reward functions
        args = {"atoms": all_atoms, "e_tot": e_tot}
        if terminal or "rew_valid" in rew_names:
            mol_info = self.analyzer.get_mol(all_atoms)
            args["mol_info"] = mol_info
            info["mol_info"] = mol_info
        if "rew_formation" in rew_names:
            args["worker_id"] = worker_id

        # Calculate reward
        new_rewards = {}
        for reward_name in rew_names:
            method_name = self.reward_functions_available[reward_name]
            reward_function = getattr(self, method_name)
            new_rewards[reward_name] = reward_function(args)

        reward = sum([self.reward_coefs[k] * new_rewards[k] for k in new_rewards.keys()])

        info.update(
            {
                "rew_calc_time": time.time() - start,
                "new_rewards": new_rewards,
            }
        )

        if terminal:
            # For terminal molecules, save chemically insightful metrics; reuse from new_rewards when available
            atomization_energy = (self._sum_of_atomic_energies(all_atoms) - e_tot) / n_atoms  # Cheap
            rae = new_rewards["rew_rae"] if "rew_rae" in new_rewards else self._calc_rae(all_atoms, e_tot)  # Cheap
            dipole = (
                new_rewards["rew_dipole"] if "rew_dipole" in new_rewards else self._dipole_rew(args)
            )  # Cheap if available
            validity = new_rewards["rew_valid"] if "rew_valid" in new_rewards else self._validity_rew(args)  # Cheap

            metrics = {
                "final:AE": atomization_energy,
                "final:Valid": validity,
                "final:RAE": rae,
                "final:Dipole": dipole,
            }
            info.update({"metrics": metrics})

        return reward, info

    def reduce_validity_reward(self, factor: float) -> None:
        assert "rew_valid" in self.reward_names, "rew_valid must be in reward_names"
        assert (factor > 0) and (factor <= 1), "factor must be between 0 and 1"

        min_value = 0.1
        new_value = self.reward_coefs["rew_valid"] * factor
        self.reward_coefs["rew_valid"] = max(new_value, min_value)

    # Reward calculation methods
    def _atomisation(self, args: Dict) -> float:
        """The energy of the molecule minus the sum of the atomic energies. Often used at the end of the episode."""
        atoms = args["atoms"]
        e_tot = args["e_tot"]
        if e_tot is None:
            return 0.0

        e_parts = self._sum_of_atomic_energies(atoms)
        neg_delta_e = -1 * (e_tot - e_parts)

        # Convert to hartree so it's closer to unity
        neg_delta_e = EnergyConverter.convert(neg_delta_e, self.energy_unit, EnergyUnit.HARTREE)

        x = neg_delta_e

        if x < 0:
            reward = x
        else:
            # Polynomial reward for good energies
            reward = x + 0.5 * x**2

        return reward

    def _formation(self, args: Dict) -> float:
        """The perStep reward for the formation of a new atom."""
        atoms = args["atoms"]
        e_tot = args["e_tot"]
        if e_tot is None:
            return 0.0
        worker_id = args["worker_id"]

        # Get old energy
        e_old = self.old_energies[worker_id]

        e_parts = e_old + self._calculate_atomic_energy(atoms[-1])
        neg_delta_e = -1 * (e_tot - e_parts)
        neg_delta_e = EnergyConverter.convert(neg_delta_e, self.energy_unit, EnergyUnit.HARTREE)

        # Update old energy
        self.old_energies[worker_id] = e_tot

        return neg_delta_e

    def _basin_dist_rew(self, args: Dict) -> float:
        atoms = args["atoms"]
        opt_info = self._optimize_atoms(atoms, max_steps=100)
        energy_before = opt_info["energy_before"]
        energy_after = opt_info["energy_after"]
        return max(energy_after - energy_before, -1) if energy_before and energy_after else -1

    def _validity_rew(self, args: Dict) -> float:
        flag = args["mol_info"]["info"]
        if flag == "valid":
            return 1.0
        elif flag == "charged_or_radical":
            return 0.5
        elif flag == "fragmented":
            return 0.1
        elif flag == "failed":
            return 0.05
        else:
            return 0.0

    # Energy calculation methods
    def _calculate_energy(self, atoms: Atoms) -> Optional[float]:
        if len(atoms) == 0:
            return 0.0
        return self.calc.calc_potential_energy(atoms)

    def _calculate_atomic_energy(self, atom: Union[Atom, Atoms]) -> float:
        atom = atom[0] if isinstance(atom, Atoms) else atom
        if atom.symbol not in self.atom_energies:
            atoms = Atoms()
            atoms.append(atom)
            energy = self._calculate_energy(atoms)
            if energy is None:
                return 0.0  # xTB failed for single atom; fallback, not cached so it retries next call
            self.atom_energies[atom.symbol] = energy
        return self.atom_energies[atom.symbol]

    def _sum_of_atomic_energies(self, atoms: Atoms) -> float:
        return sum([self._calculate_atomic_energy(a) for a in atoms])

    def _optimize_atoms(self, atoms: Atoms, max_steps: int = 50) -> dict:
        calc = XTBOptimizer(energy_unit=self.energy_unit)
        return calc.optimize_atoms(atoms, max_steps=max_steps, fmax=0.10)

    # RAE calculation methods
    def _calc_rae(self, atoms: Atoms, e_tot: Optional[float] = None) -> float:
        bag_repr = self._get_formula(atoms)
        if bag_repr not in self.benchmark_energies:
            return None
        mean_reference_energy = self.benchmark_energies[bag_repr]
        e_tot = e_tot if e_tot is not None else self._calculate_energy(atoms)
        rae = e_tot - mean_reference_energy
        rae = rae / len(atoms)
        return rae

    def _get_formula(self, atoms: Atoms) -> str:
        assert hasattr(self, "benchmark_energies"), "Benchmark energies must be set to calculate RAE"
        assert self.benchmark_energies is not None, "Benchmark energies cannot be None"
        bag_repr = symbols_to_str_formula([a.symbol for a in atoms])
        return bag_repr

    def _dipole_rew(self, args: Dict) -> float:
        """Dipole magnitude (expensive XTB call); prefer reusing from new_rewards['rew_dipole'] when available."""
        dipole = self.calc.calc_dipole(args["atoms"])
        if dipole is None:
            return 0.0
        return dipole


class RaeReward(InteractionReward):
    """Calculates the Relative Atomic Energy (RAE) of the molecule."""

    def __init__(self, benchmark_energies: Dict[str, float], use_exponential: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)

        self.benchmark_energies = benchmark_energies
        self.use_exponential = use_exponential

        # Add RAE to reward functions
        self.reward_functions_available["rew_rae"] = "_rae_rew"

    def _rae_rew(self, args: Dict) -> float:
        """Calculates the Relative Atomic Energy (RAE) of the molecule."""
        rae = self._calc_rae(args["atoms"], args["e_tot"])

        reward = -1 * rae

        if self.use_exponential:
            reward = np.exp(3 * (reward + 0.5))

        return reward

    def _calc_rae(self, atoms: Atoms, e_tot: Optional[float] = None) -> float:
        mean_reference_energy = self.benchmark_energies[self._get_formula(atoms)]
        e_tot = e_tot if e_tot is not None else self._calculate_energy(atoms)
        rae = e_tot - mean_reference_energy
        rae = rae / len(atoms)
        return rae

    def _get_formula(self, atoms: Atoms) -> str:
        assert hasattr(self, "benchmark_energies"), "Benchmark energies must be set to calculate RAE"
        assert self.benchmark_energies is not None, "Benchmark energies cannot be None"
        bag_repr = symbols_to_str_formula([a.symbol for a in atoms])
        assert bag_repr in self.benchmark_energies, f"Formula {bag_repr} not in benchmark energies"
        return bag_repr
