from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class MolCandidate:
    num_env_steps: int  # total number of training steps taken in the environment
    elements: List[int]
    pos: np.ndarray
    reward: float
    energy: float
    energy_relaxed: float = None
    rae: float = None


@dataclass
class FormulaData:
    # keys are SMILES strings, values are lists of MolCandidates
    molecules: Dict[str, List[MolCandidate]] = field(default_factory=dict)
