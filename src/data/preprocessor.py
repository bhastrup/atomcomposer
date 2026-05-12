import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

import numpy as np

from ase.visualize import view
from ase import Atoms

from src.data.io_handler import IOHandler
from src.data.parser import Parser
from src.data.featurizer import Featurizer
from src.data.reference_dataloader import FileType

from tqdm import tqdm


class Preprocessor:
    """Preprocessing class for molecule data.

    example usage:

    parser = QM7Parser(tag = 'qm7_test1')
    featurizer = Featurizer(
        include_energy=include_energy,
        include_smiles=include_smiles,
        include_connectivity=include_connectivity,
        use_huckel=use_huckel
    )

    processor = Preprocessor(parser, featurizer)
    processor.load_data(n_mols=None)
    processor.preprocess()
    processor.print_success_ratio()
    processor.view_bad_molecules()
    processor.save_data()


    The data is stored in self.data, which is a dictionary with keys corresponding to the status of the molecule.
    The load_data() method doesn't necessarily load the actual molecule data, but creates an iterable of the
    molecules for the preprocess method.
    """

    # status_flags = ['valid', 'charged_or_radical', 'fragmented', 'failed', 'crashed']

    def __init__(self, parser: Parser, featurizer: Featurizer, max_workers: int = 1):
        self.parser = parser
        self.featurizer = featurizer
        self.max_workers = max_workers

        self.data = {}

        self.processed_dir_path = os.path.join(self.parser.data_dir, self.parser.tag, "processed")

    def load_data(self, n_mols: int = None) -> None:
        """Create the self.mol_iterable using the parser."""
        self.mol_iterable = self.parser.load_data(n_mols=n_mols)

    def preprocess(self) -> None:
        max_workers = min(self.max_workers, len(self.mol_iterable))
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self._process_molecule_return, i, raw) for i, raw in enumerate(self.mol_iterable)]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                results.append(fut.result())

        # Deterministic ordering
        results.sort(key=lambda x: x[0])  # sort by i

        self.data = {}
        for i, feature_dict, status in results:
            self.data.setdefault(status, []).append(feature_dict)

    def _process_molecule_return(self, i: int, raw_data: Any):
        mol_data = self.parser.read_atoms(i, raw_data)
        feature_dict, status = self.featurizer.calc_features(i, mol_data)
        return i, feature_dict, status

    def _store_molecule(self, feature_dict: dict, status: str) -> None:
        if status not in self.data.keys():
            self.data[status] = []
        self.data[status].append(feature_dict)

    def get_dict_of_type(self, type: str) -> Dict[int, Dict[str, Any]]:
        """Converts list mol dicts to a dictionary with molecule_id as key."""
        if type not in self.data.keys():
            return {}
        return {
            mol_dict["molecule_id"]: {k: v for k, v in mol_dict.items() if k != "molecule_id"}
            for mol_dict in self.data[type]
        }

    def save_data(self) -> None:
        """Save the preprocessed data to disk."""
        os.makedirs(self.processed_dir_path, exist_ok=True)

        # Save meta data (json)
        meta_data = {}
        if hasattr(self.featurizer, "meta_data"):
            meta_data.update(self.featurizer.meta_data)
        meta_data.update({"bag_list": self._get_bag_list()})
        IOHandler.write_json(meta_data, os.path.join(self.processed_dir_path, FileType.METADATA.value))

        # Save valid and discarded molecules
        valid_data = self.get_dict_of_type("valid")
        IOHandler.write_pickle(valid_data, os.path.join(self.processed_dir_path, FileType.DF.value))

        discarded = {status: self.get_dict_of_type(status) for status in self.data.keys() if status != "valid"}
        IOHandler.write_pickle(discarded, os.path.join(self.processed_dir_path, "discarded_mols.pkl"))

        # Save smiles for each bag_repr
        bag_repr_to_smiles = self._get_bag_repr_to_values(valid_data, "SMILES")
        IOHandler.write_json(bag_repr_to_smiles, os.path.join(self.processed_dir_path, FileType.SMILES.value))

        # From valid_data, save dict of bag_repr keys with array of energies
        bag_energy_dict = self._get_bag_repr_to_values(valid_data, "energy_GFN2")
        IOHandler.write_json(bag_energy_dict, os.path.join(self.processed_dir_path, FileType.ENERGIES.value))

    def _get_bag_list(self) -> list:
        if "valid" not in self.data.keys():
            return []
        return np.unique([mol_dict["bag_repr"] for mol_dict in self.data["valid"]]).tolist()

    def _get_bag_repr_to_values(self, valid_data: dict, column_name: str) -> dict:
        bag_repr_to_values = {}
        for mol_dict in valid_data.values():
            bag_repr = mol_dict["bag_repr"]
            value = mol_dict[column_name]
            if bag_repr not in bag_repr_to_values:
                bag_repr_to_values[bag_repr] = []
            bag_repr_to_values[bag_repr].append(value)
        return bag_repr_to_values

    def has_mol_iterable(self) -> bool:
        return hasattr(self, "mol_iterable")

    @property
    def n_mols_max(self) -> int:
        if not self.has_mol_iterable():
            logging.info("No data loaded. Run load_data() first.")
            return 0
        return len(self.mol_iterable)

    def n_mols_processed(self) -> int:
        return sum([len(v) for k, v in self.data.items()])

    def n_mols_success(self) -> int:
        return len(self.data["valid"]) if "valid" in self.data.keys() else 0

    def success_ratio(self) -> float:
        return self.n_mols_success() / self.n_mols_processed() * 100

    def print_success_ratio(self) -> None:
        print(f"Successes {self.n_mols_success()} / {self.n_mols_processed()}   |   Rate {self.success_ratio():.2f}%")

    def view_bad_molecules(self):
        for k, v in self.data.items():
            if k == "valid":
                continue
            print(f"{k}: {len(v)}")
            print(f"{[v[i]['SMILES'] for i in range(len(v))]}")
            atoms_list = [Atoms(v[i]["atomic_symbols"], positions=v[i]["pos"]) for i in range(len(v))]
            view(atoms_list)
