import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from src.data.io_handler import IOHandler
from src.performance.cumulative.storage import FormulaData


SmilesCounterType = Dict[str, Dict[str, List[Tuple[int, int]]]]
MergedDBType = Dict[str, Dict[str, FormulaData]]
RawDBType = Dict[int, Dict[str, Dict[str, FormulaData]]]


def process_raw_data(
    db_raw: RawDBType, bag_reprs: dict, ref_smiles: dict
) -> Dict[str, Union[MergedDBType, SmilesCounterType]]:
    # Initialize a counter for the number of smiles found at each step
    smiles_counter = {
        tag: {
            formula: [
                (0, 0, 0),
            ]
            for formula in formulas
        }
        for tag, formulas in bag_reprs.items()
    }
    smiles_counter_r = {
        tag: {
            formula: [
                (0, 0, 0),
            ]
            for formula in formulas
        }
        for tag, formulas in bag_reprs.items()
    }

    # Initialize a full database to store all the data
    full_db = {tag: {formula: FormulaData() for formula in formulas} for tag, formulas in bag_reprs.items()}

    rediscovered_smiles = {tag: {formula: set() for formula in formulas} for tag, formulas in bag_reprs.items()}

    # Merge all the data
    for steps, rollout in tqdm(db_raw.items(), desc="Raw data -> (MergedDBType SmilesCounterType)"):
        for in_oos_tag, formula_dict in rollout.items():
            for formula, formula_data in formula_dict.items():
                for smiles, mol_candidate_list in formula_data.molecules.items():
                    # First, check if smiles is not already in the FormulaData object
                    if smiles not in full_db[in_oos_tag][formula].molecules:
                        full_db[in_oos_tag][formula].molecules[smiles] = []
                    # Then add the mol_candidates to the full_db
                    full_db[in_oos_tag][formula].molecules[smiles].extend(mol_candidate_list)

                    if smiles in ref_smiles[formula]:
                        rediscovered_smiles[in_oos_tag][formula].add(smiles)

                # Count the number of molecules found up until this point (steps)
                num_smiles = len(full_db[in_oos_tag][formula].molecules)
                # only append if the number of smiles has changed
                if num_smiles != smiles_counter[in_oos_tag][formula][-1][2]:
                    diff = num_smiles - smiles_counter[in_oos_tag][formula][-1][2]
                    smiles_counter[in_oos_tag][formula].append((steps, diff, num_smiles))

                # Count the number of rediscovered molecules up until this point (steps)
                num_rediscovered = len(rediscovered_smiles[in_oos_tag][formula])
                if num_rediscovered != smiles_counter_r[in_oos_tag][formula][-1][2]:
                    diff = num_rediscovered - smiles_counter_r[in_oos_tag][formula][-1][2]
                    smiles_counter_r[in_oos_tag][formula].append((steps, diff, num_rediscovered))

    smiles_counter = append_last_step(smiles_counter)
    smiles_counter_r = append_last_step(smiles_counter_r)
    smiles_counter = sort_formulas_by_terminal_count(smiles_counter)
    smiles_counter_r = sort_formulas_by_terminal_count(smiles_counter_r)

    print(
        f"avg length of smiles_list: \
            {np.mean([len(smiles_list) for smiles_list in smiles_counter['in_sample'].values()])}"
    )

    print(
        f"avg length of rediscovered smiles_list: \
            {np.mean([len(smiles_list) for smiles_list in smiles_counter_r['in_sample'].values()])}"
    )

    return dict(full_db=full_db, smiles_counter=smiles_counter, smiles_counter_rediscovery=smiles_counter_r)


def append_last_step(smiles_counter: SmilesCounterType) -> SmilesCounterType:
    """Append one extra step obs to the counter with count equal to the last count"""
    for tag in smiles_counter:
        max_step = max([step for formula in smiles_counter[tag] for (step, _, _) in smiles_counter[tag][formula]])
        for formula in smiles_counter[tag]:
            last_step, last_diff, last_count = smiles_counter[tag][formula][-1]
            smiles_counter[tag][formula].append((max_step, 0, last_count))

    return smiles_counter


def sort_formulas_by_terminal_count(smiles_counter: SmilesCounterType) -> SmilesCounterType:
    """Sort formulas by the number of molecules found at the end of the training"""
    for tag in smiles_counter:
        smiles_counter[tag] = {
            formula: counts_list
            for formula, counts_list in sorted(
                smiles_counter[tag].items(), key=lambda item: item[1][-1][2], reverse=True
            )
        }

    return smiles_counter


def find_steps_and_paths(save_dir: Path, step_max: int) -> Dict[int, str]:
    """Find all paths to the stored data"""
    paths = {}
    for file in os.listdir(save_dir):
        if file.endswith(".pkl"):
            steps = int(file.split("steps-")[1].split(".pkl")[0])
            paths[steps] = os.path.join(save_dir, file)

    paths = {steps: path for steps, path in sorted(paths.items(), key=lambda item: item[0])}

    if step_max is not None:
        return {steps: path for steps, path in paths.items() if steps <= step_max}

    return paths


class CumulativeIO:
    def __init__(self, save_dir: Path, batched: bool = False) -> None:
        self.save_dir = save_dir / "discovery"
        self.file_name = "cumulative_discovery"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.batched = batched

    def dump_bag_reprs(self, bag_reprs: Dict[str, List[str]]) -> None:
        IOHandler.write_json(bag_reprs, os.path.join(self.save_dir, "bag_reprs.json"))

    def load_bag_reprs(self) -> Dict[str, List[str]]:
        return IOHandler.read_json(os.path.join(self.save_dir, "bag_reprs.json"))

    def _new_db_path(self, steps: int) -> str:
        return os.path.join(self.save_dir, f"{self.file_name}-steps-{steps}.pkl")

    def dump_current_db(self, db_big: Any, steps: int) -> None:
        # To save space on disk, remove FormulaData objects that have no molecules.
        for step, db_small in db_big.items():
            for db in db_small.values():
                db_copy = db.copy()
                for formula, formula_data in db_copy.items():
                    if not formula_data.molecules:
                        del db[formula]

        IOHandler.write_pickle(db_big, self._new_db_path(steps))

    def load_all_dbs(self, step_max: int) -> Dict[int, Any]:
        if self.batched:
            big_paths = find_steps_and_paths(self.save_dir, step_max)

            loaded_files = {
                big_step: IOHandler.read_pickle(big_path)
                for big_step, big_path in tqdm(big_paths.items(), desc="Loading big DB files")
            }

            return {
                small_step: db_small
                for big_step, big_db in loaded_files.items()
                for small_step, db_small in big_db.items()
            }
        else:
            paths = find_steps_and_paths(self.save_dir)

            return {steps: IOHandler.read_pickle(path) for steps, path in tqdm(paths.items(), desc="Loading DB files")}
