import collections
import json
import logging
import os
import pickle
import re
import sys
from typing import List, Iterable, Dict, Union


import ase.data
import ase.formula
from ase.formula import Formula
from ase.data import atomic_numbers, chemical_symbols

import numpy as np
import scipy.signal
import torch

from src.rl.spaces import FormulaType
from src.rl.env_container import SimpleEnvContainer


def get_str_formulas_from_vecenv(envs: SimpleEnvContainer) -> List[str]:
    """Only meant to extract eval formulas. One for each environment."""
    for env in envs.environments:
        assert len(env.formulas) == 1
    return [bag_tuple_to_str_formula(env.formulas[0]) for env in envs.environments]


def str_formula_to_size(formula: str) -> int:
    # Regular expression to find elements and their counts
    element_pattern = re.compile(r"([A-Z][a-z]?)(\d*)")

    total_atoms = 0

    for element, count in element_pattern.findall(formula):
        # If count is empty, it means there's only one atom of this element
        count = int(count) if count else 1
        total_atoms += count

    return total_atoms


def str_formula_to_bag_tuple(formula: str) -> FormulaType:
    # Regular expression to find elements and their counts
    element_pattern = re.compile(r"([A-Z][a-z]?)(\d*)")

    bag = []

    for element, count in element_pattern.findall(formula):
        # If count is empty, it means there's only one atom of this element
        count = int(count) if count else 1
        bag.append((ase.data.atomic_numbers[element], count))

    return tuple(bag)


def symbols_to_str_formula(atom_syms: list) -> str:
    elements_unique = set([atomic_numbers[s] for s in atom_syms])
    symbols_sorted = [chemical_symbols[atomic_number] for atomic_number in sorted(elements_unique)]
    counts = dict(collections.Counter(atom_syms))
    counts_sort = {k: counts[k] for k in symbols_sorted}
    return "".join([f"{k}" if v == 1 else f"{k}{v}" for k, v in counts_sort.items()])


def elements_to_str_formula(elements: List[Union[int, str]]) -> str:
    if all(isinstance(e, int) for e in elements):
        symbols = [chemical_symbols[e] for e in elements]
    elif all(isinstance(e, str) for e in elements):
        symbols = elements
    else:
        raise ValueError("All elements in the list must be of the same type (all integers or all strings)")

    counts = collections.Counter(symbols)
    sorted_symbols = sorted(counts.keys(), key=lambda x: atomic_numbers[x])
    return "".join([f"{s}" if counts[s] == 1 else f"{s}{counts[s]}" for s in sorted_symbols])


def bag_tuple_to_str_formula(bag_tuple: FormulaType, sort: bool = True) -> str:
    if sort:
        bag_tuple = sorted(bag_tuple, key=lambda x: x[0])
    return "".join(
        f"{ase.data.chemical_symbols[z] if count > 0 else ''}{count if count > 1 else ''}" for z, count in bag_tuple
    )


def string_to_formula(string: str) -> FormulaType:
    d = Formula(string).count().items()
    return tuple((ase.data.atomic_numbers[symbol], count) for symbol, count in d)


def zs_to_formula(zs: List[int]) -> FormulaType:
    counter: Dict[int, int] = collections.Counter()
    for z in zs:
        counter[z] += 1
    return tuple(counter.items())


def remove_atom_from_formula(formula: FormulaType, atomic_number: int) -> FormulaType:
    copy = list(formula)
    for i, (z, count) in enumerate(formula):
        if z == atomic_number and count >= 1:
            copy[i] = (z, count - 1)
            return tuple(copy)

    raise RuntimeError(f"Could not remove atomic number {atomic_number} from bag {formula}")


def add_atom_to_formula(formula: FormulaType, atomic_number: int) -> FormulaType:
    copy = list(formula)
    for i, (z, count) in enumerate(formula):
        if z == atomic_number:
            copy[i] = (z, count + 1)
            return tuple(copy)

    copy.append((atomic_number, 1))
    return tuple(copy)


def find_count(formula: FormulaType, atomic_number: int) -> int:
    for z, count in formula:
        if z == atomic_number:
            return count
    return 0


def update_count(formula: FormulaType, atomic_number: int, count: int) -> FormulaType:
    copy = list(formula)
    for i, (z, _) in enumerate(formula):
        if z == atomic_number:
            copy[i] = (z, count)
            return tuple(copy)

    copy.append((atomic_number, count))
    return tuple(copy)


def get_formula_size(formula: FormulaType) -> int:
    return sum(count for z, count in formula)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def count_vars(module: torch.nn.Module) -> int:
    return sum(np.prod(p.shape) for p in module.parameters())


def compute_gradient_norm(parameters: Iterable[torch.nn.Parameter], norm_type: int = 2) -> float:
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if len(parameters) == 0:
        return 0.0
    device = parameters[0].grad.device  # type: ignore
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),  # type: ignore
        norm_type,
    )
    return total_norm.item()


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def set_seeds(seed: int) -> None:
    np.random.seed(seed)  # Sets seed for numpy
    torch.manual_seed(seed)  # Sets seed for PyTorch
    if hasattr(torch, "cuda"):  # Sets seed for CUDA (if using GPU)
        torch.cuda.manual_seed_all(seed)


def get_tag(config: dict) -> str:
    return "{exp}_run-{seed}".format(exp=config["name"], seed=config["seed"])


def save_config(config: dict, directory: str, tag: str, verbose=True):
    formatted = json.dumps(config, indent=4, sort_keys=True)

    if verbose:
        logging.info(formatted)

    path = os.path.join(directory, tag + ".json")
    with open(file=path, mode="w") as f:
        f.write(formatted)


def create_directories(directories: List[str]):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


class SafeStreamHandler(logging.StreamHandler):
    def flush(self):
        try:
            super().flush()
        except BrokenPipeError:
            logging.getLogger().removeHandler(self)


def setup_logger(config: dict, directory, tag: str):
    logger = logging.getLogger()
    logger.setLevel(config["log_level"])

    formatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = SafeStreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    path = os.path.join(directory, tag + ".log")
    fh = logging.FileHandler(path)
    fh.setFormatter(formatter)

    logger.addHandler(fh)


class RolloutSaver:
    def __init__(self, directory: str, tag: str):
        self.directory = directory
        self.tag = tag
        self._suffix = ".pkl"

    def save(self, obj: object, num_steps: int, info: str):
        added = f"steps-{num_steps}"

        path = os.path.join(self.directory, self.tag + "_" + added + "_" + info + self._suffix)
        logging.debug(f"Saving rollout: {path}")
        with open(path, mode="wb") as f:
            pickle.dump(obj, f)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class InfoSaver:
    def __init__(self, directory: str, tag: str):
        self.directory = directory
        self.tag = tag
        self._suffix = ".txt"

    def save(self, obj: object, name: str):
        path = os.path.join(self.directory, self.tag + "_" + name + self._suffix)
        logging.debug(f"Saving info: {path}")
        with open(path, mode="a") as f:
            f.write(json.dumps(obj, cls=_NumpyEncoder))
            f.write("\n")


def init_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        assert torch.cuda.is_available(), "No CUDA device available!"
        logging.info("CUDA Device: {}".format(torch.cuda.current_device()))
        torch.cuda.init()
        return torch.device("cuda")
    else:
        logging.info("Using CPU")
        return torch.device("cpu")


def fibonacci_sphere(samples=15):
    points = np.zeros((samples, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i] = [x, y, z]

    return points
