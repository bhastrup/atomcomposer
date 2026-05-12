import glob
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum

import numpy as np
import pandas as pd

from src.tools import util
from src.data.io_handler import IOHandler
from src.data.data_util import unpack_upper_triangular_to_full
from src.performance.energetics import EnergyConverter, EnergyUnit, str_to_EnergyUnit


class FileType(Enum):
    METADATA = "meta_data.json"
    DF = "dataframe.pkl"
    SMILES = "bags_and_smiles.json"
    ENERGIES = "bags_and_energies.json"


@dataclass
class ReferenceData:
    metadata: Dict[str, Any]

    df: pd.DataFrame = None
    energies: Dict[str, List[float]] = None
    smiles: Dict[str, List[str]] = None

    def get_mean_energies(self) -> Dict[str, float]:
        return {k: np.mean(v) for k, v in self.energies.items()}


BenchmarkEnergyType = Dict[str, float]


class ReferenceDataFinder:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def _build_path(self, dataset_name: str, file_type: FileType) -> str:
        return os.path.join(self.data_dir, dataset_name.lower(), "processed", file_type.value)

    def _dataset_is_good(self, dataset_name: str) -> bool:
        required_files = [self._build_path(dataset_name, file_type) for file_type in FileType]
        return all(os.path.exists(file) for file in required_files)

    def find_good_datasets(self) -> List[str]:
        candidates = [os.path.basename(f) for f in glob.glob(os.path.join(self.data_dir, "*")) if os.path.isdir(f)]
        good_datasets = [dataset_name for dataset_name in candidates if self._dataset_is_good(dataset_name)]
        logging.info(f"Found the following datasets: {good_datasets}")
        return good_datasets

    def get_good_datasets(self) -> List[str]:
        if not hasattr(self, "good_datasets"):
            self.good_datasets = self.find_good_datasets()
        return self.good_datasets

    def get_reference_bags(self) -> Dict[str, List[str]]:
        good_datasets = self.get_good_datasets()
        bag_dict = {}
        for dataset_name in good_datasets:
            bag_dict[dataset_name] = self.get_reference_bags_for_dataset(dataset_name)
        return bag_dict

    def get_reference_bags_for_dataset(self, dataset_name: str) -> List[str]:
        return IOHandler.read_json(self._build_path(dataset_name, FileType.SMILES))

    def find_matching_datasets(self, bag_repr: str) -> List[str]:
        bags = self.get_reference_bags()
        matching_datasets = [name for name in bags.keys() if bag_repr in set(bags[name])]
        return matching_datasets


class ReferenceDataLoader:
    """Class for loading and post-processing the reference dataset."""

    energy_column = "energy_GFN2"

    def __init__(self, data_dir: str = "data"):
        data_dir = os.path.join(os.getcwd(), data_dir)
        self.finder = ReferenceDataFinder(data_dir)

    def _has_smiles(self, metadata: Dict[str, Any]) -> bool:
        return metadata["include_smiles"]

    def load_and_polish(self, mol_dataset: str, new_energy_unit: EnergyUnit, fetch_df: bool = True) -> ReferenceData:
        """Load the reference data and polish it."""

        logging.info(f"Loading reference data {mol_dataset}. Using energy_unit: {new_energy_unit}")

        ref_data = self._load_data(mol_dataset, fetch_df=fetch_df)
        ref_data = self._change_energy_unit(ref_data, new_energy_unit)

        if ref_data.df is not None:
            ref_data.df = self._add_columns(ref_data.df)
            ref_data.df = self._remove_invalid_mols(ref_data)

        return ref_data

    def get_benchmark_energies(self, reference_data: ReferenceData) -> BenchmarkEnergyType:
        """Get the benchmark energies from the dataframe."""
        return reference_data.get_mean_energies()

    def _load_data(self, mol_dataset: str, fetch_df: bool = True) -> ReferenceData:
        """Load the metadata and data from disk."""
        metadata = IOHandler.read_json(self.finder._build_path(mol_dataset, FileType.METADATA))

        if fetch_df:
            data = IOHandler.read_pickle(self.finder._build_path(mol_dataset, FileType.DF))
            df = pd.DataFrame.from_dict(data, orient="index")
        else:
            df = None

        energies = IOHandler.read_json(self.finder._build_path(mol_dataset, FileType.ENERGIES))
        smiles = IOHandler.read_json(self.finder._build_path(mol_dataset, FileType.SMILES))

        return ReferenceData(metadata=metadata, df=df, energies=energies, smiles=smiles)

    def _change_energy_unit(self, ref_data: ReferenceData, new_energy_unit: EnergyUnit) -> ReferenceData:
        """Change the energy unit of the dataframe and update the metadata."""

        old_energy_unit = str_to_EnergyUnit(ref_data.metadata["energy_unit"])

        if ref_data.df is not None:
            ref_data.df[self.energy_column] = EnergyConverter.convert(
                values=ref_data.df[self.energy_column].values, old_unit=old_energy_unit, new_unit=new_energy_unit
            )

        ref_data.energies = {
            k: EnergyConverter.convert(v, old_energy_unit, new_energy_unit) for k, v in ref_data.energies.items()
        }

        ref_data.metadata["energy_unit"] = new_energy_unit

        return ref_data

    def _add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["formulas"] = df["bag_repr"].apply(lambda x: util.string_to_formula(x))
        if "con_upper_triangular" in df.columns and np.all(df["con_upper_triangular"].notnull()):
            df["con_mat"] = df.apply(
                lambda x: unpack_upper_triangular_to_full(x["con_upper_triangular"], x["n_atoms"]), axis=1
            )
        return df.drop(columns=["con_upper_triangular"])

    def _remove_invalid_mols(self, ref_data: ReferenceData) -> pd.DataFrame:
        df = ref_data.df

        if self._has_smiles(ref_data.metadata):
            """ Filter out mols that couldn't be converted to RdKit mol object (SMILES = None) """
            logging.info(f"SMILES == None for : {len(df[df['SMILES'].isnull()])} / {len(df)}")
            df = df[~df["SMILES"].isnull()]

        return df
