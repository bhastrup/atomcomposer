import logging
import multiprocessing as mp
import signal
from enum import Enum
from typing import Optional

import numpy as np
from ase import Atoms
from ase.optimize import FIRE
from xtb.ase.calculator import XTB


def _xtb_energy(atoms: Atoms, method: str) -> float:
    atoms = atoms.copy()
    atoms.calc = XTB(method=method)
    return atoms.get_potential_energy()


def _xtb_dipole(atoms: Atoms, method: str):
    atoms = atoms.copy()
    calc = XTB(method=method)
    calc.calculate(atoms=atoms, properties=["dipole"])
    return calc.results["dipole"]


def _xtb_worker_loop(conn, method: str) -> None:
    """Runs in a forked subprocess; loops receiving tasks and sending results.

    When xTB calls C-level exit(), this process dies and the OS closes the pipe,
    which the parent detects as EOFError — a catchable Python exception.

    We clear all inherited atexit handlers immediately so that when xTB calls
    C-level exit(), Python's inherited cleanup (wandb, logging, etc.) does NOT
    run in this child — preventing corruption of the parent's shared resources.
    """
    import atexit

    atexit._clear()

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break

        task_type, atoms = msg
        if task_type == "stop":
            break

        if task_type == "energy":
            try:
                result = _xtb_energy(atoms, method)
                conn.send(("ok", result))
            except Exception:
                conn.send(("error", None))

        elif task_type == "dipole":
            try:
                result = _xtb_dipole(atoms, method)
                conn.send(("ok", result))
            except Exception:
                conn.send(("error", None))

    try:
        conn.close()
    except Exception:
        pass


class EnergyUnit(Enum):
    HARTREE = "hartree"
    KCAL_MOL = "kcal/mol"
    EV = "eV"


def str_to_EnergyUnit(value: str) -> EnergyUnit:
    """Converts a string to an EnergyUnit enum, case-insensitively."""
    value = value.lower()
    for unit in EnergyUnit:
        if unit.value.lower() == value:
            return unit
    raise ValueError(f"'{value}' is not a valid EnergyUnit")


class EnergyConverter:
    conversion_factors = {
        (EnergyUnit.HARTREE, EnergyUnit.KCAL_MOL): 627.50960,
        (EnergyUnit.HARTREE, EnergyUnit.EV): 27.21140,
        (EnergyUnit.KCAL_MOL, EnergyUnit.HARTREE): 1 / 627.50960,
        (EnergyUnit.EV, EnergyUnit.HARTREE): 1 / 27.21140,
        (EnergyUnit.KCAL_MOL, EnergyUnit.EV): 27.21140 / 627.50960,
        (EnergyUnit.EV, EnergyUnit.KCAL_MOL): 627.50960 / 27.21140,
    }

    @staticmethod
    def convert(values: np.ndarray, old_unit: EnergyUnit, new_unit: EnergyUnit) -> float:
        """Convert value from old_unit to new_unit."""
        if old_unit == new_unit:
            return values

        try:
            factor = EnergyConverter.conversion_factors[(old_unit, new_unit)]
            return values * factor
        except KeyError:
            raise ValueError(f"Unknown conversion from {old_unit} to {new_unit}")


class XTBOptimizer:
    old_energy_unit = EnergyUnit.EV

    def __init__(
        self,
        method: str = "GFN2-xTB",
        energy_unit: EnergyUnit = EnergyUnit.HARTREE,
        use_mp: bool = False,
        timeout: int = 60,
    ):
        self.method = method
        self.energy_unit = energy_unit
        self.use_mp = use_mp
        self.timeout = timeout
        self._parent_conn = None
        self._worker_process = None
        if use_mp:
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)
            self._spawn_worker()

    def _spawn_worker(self) -> None:
        """Start a forked subprocess for safe xTB execution."""
        ctx = mp.get_context("fork")
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(target=_xtb_worker_loop, args=(child_conn, self.method), daemon=True)
        process.start()
        child_conn.close()  # Close child's copy of the connection in the parent
        self._parent_conn = parent_conn
        self._worker_process = process
        logging.info(f"XTB worker spawned (PID={process.pid})")

    def _restart_worker(self) -> None:
        """Kill and restart the worker after a crash."""
        logging.error("XTB worker died (xTB called exit()); restarting subprocess.")
        if self._worker_process is not None and self._worker_process.is_alive():
            self._worker_process.terminate()
        if self._worker_process is not None:
            self._worker_process.join(timeout=5)
        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except Exception:
                pass
        self._spawn_worker()

    def _run_xtb_task(self, task_type: str, atoms: Atoms) -> Optional[object]:
        """Send a task to the worker and return the result, or None on failure.

        When xTB calls C-level exit(), the OS closes the pipe and the parent
        receives EOFError or BrokenPipeError — both are caught here.
        """
        try:
            self._parent_conn.send((task_type, atoms))
            if self._parent_conn.poll(self.timeout):
                status, result = self._parent_conn.recv()
                return result  # None if the worker caught a Python exception
            else:
                logging.error("XTB calculation timed out; restarting worker.")
                self._restart_worker()
                return None
        except (EOFError, BrokenPipeError, OSError) as e:
            logging.error(f"XTB worker pipe error ({type(e).__name__}: {e}); restarting worker.")
            self._restart_worker()
            return None

    def get_max_force(self, atoms: Atoms) -> float:
        forces = atoms.get_forces()
        max_force = ((forces**2).sum(axis=1) ** 0.5).max()
        return EnergyConverter.convert(max_force, self.old_energy_unit, self.energy_unit)

    def calc_dipole(self, atoms: Atoms) -> Optional[float]:
        try:
            if self.use_mp:
                dipole_vector = self._run_xtb_task("dipole", atoms)
            else:
                dipole_vector = _xtb_dipole(atoms, self.method)
            if dipole_vector is None:
                return None
            return np.linalg.norm(dipole_vector)
        except Exception as e:
            logging.error(f"Error calculating dipole moment: {e}")
            return None

    def calc_potential_energy(self, atoms: Atoms) -> Optional[float]:
        try:
            if self.use_mp:
                raw_energy = self._run_xtb_task("energy", atoms)
            else:
                raw_energy = _xtb_energy(atoms, self.method)
            if raw_energy is None:
                return None
            return EnergyConverter.convert(raw_energy, self.old_energy_unit, self.energy_unit)
        except Exception as e:
            logging.error(f"Error in XTB calculation: {e}")
            return None

    def optimize_atoms(
        self, atoms: Atoms, max_steps: int = 50, fmax: float = 0.10, redirect_logfile: bool = True
    ) -> dict:
        atoms = atoms.copy()
        atoms.calc = XTB(method=self.method)

        energy_before = self.calc_potential_energy(atoms)

        if energy_before is not None:
            try:
                dyn = FIRE(atoms, logfile="temp_ASE_optimizer_log_file" if redirect_logfile else None)
                dyn.run(fmax=fmax, steps=max_steps)
                energy_after = self.calc_potential_energy(atoms)
            except Exception as e:
                logging.error(f"Error during optimization: {e}")
                energy_after = None
        else:
            energy_after = None

        try:
            fmax_final = self.get_max_force(atoms)
        except Exception as e:
            logging.error(f"Error in calculating max force: {e}")
            fmax_final = fmax

        converged = fmax_final < fmax

        atoms.calc = None

        opt_info = dict(
            energy_before=energy_before,
            energy_after=energy_after,
            new_atoms=atoms,
            fmax_final=fmax_final,
            converged=converged,
            nsteps=dyn.nsteps if converged else max_steps,
        )

        return opt_info

    def close_pool(self) -> None:
        """Shut down the worker subprocess."""
        if self._parent_conn is not None and self._worker_process is not None and self._worker_process.is_alive():
            try:
                self._parent_conn.send(("stop", None))
                self._worker_process.join(timeout=5)
            except Exception:
                pass
        if self._worker_process is not None and self._worker_process.is_alive():
            self._worker_process.terminate()
        if self._worker_process is not None:
            self._worker_process.join(timeout=2)
        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except Exception:
                pass
        self._parent_conn = None
        self._worker_process = None
