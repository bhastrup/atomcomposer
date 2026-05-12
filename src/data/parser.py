import logging
import os
import tarfile
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple
import gzip

import numpy as np
import requests
import scipy.io
from tqdm import tqdm

from ase.data import atomic_numbers, chemical_symbols


class Parser(ABC):
    urls: List[str]
    file_names: List[str]

    def __init__(self, tag: str):
        self.tag = tag

        self.data_dir = "data"
        self.base_data_dir = os.path.join(os.getcwd(), self.data_dir)

        self.raw_dir_path = os.path.join(self.base_data_dir, self.tag, "raw")
        os.makedirs(self.raw_dir_path, exist_ok=True)

    @property
    def raw_file_paths(self) -> List[str]:
        return [os.path.join(self.raw_dir_path, file_name) for file_name in self.file_names]

    def load_data(self, n_mols: int = None) -> Iterable[dict]:
        # Ensure all raw data files are downloaded or already exist
        for file_name, raw_file_path in zip(self.file_names, self.raw_file_paths):
            if not os.path.exists(raw_file_path):
                self.download_raw_data(file_name)
            else:
                logging.debug(f"File {raw_file_path} already exists. Loading data from disk.")

        # Load raw data from files
        data = {
            file_name: self.load_raw_data(raw_file_path)
            for file_name, raw_file_path in zip(self.file_names, self.raw_file_paths)
        }

        return self.make_mol_iterable(data, n_mols)

    def download_raw_data(self, file_name: str) -> None:
        """Download raw data from URLs and save to multiple files."""
        for url in self.urls:
            logging.debug(f"Downloading {self.tag} dataset from {url} to {file_name}")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                # Get the total length of the content
                total_length = int(response.headers.get("content-length", 0))

                raw_file_path = os.path.join(self.raw_dir_path, file_name)
                with open(raw_file_path, "wb") as file:
                    # Initialize the tqdm progress bar
                    progress_bar = tqdm(total=total_length, unit="iB", unit_scale=True)
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            # Update the progress bar
                            progress_bar.update(len(chunk))
                    progress_bar.close()

                logging.info(f"Download completed successfully: {raw_file_path}")
                break
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to download data from {url}: {e}")
                continue

    @abstractmethod
    def load_raw_data(self, file_name: str) -> Any:
        """Load raw data from file. Must be implemented by subclasses."""

    @abstractmethod
    def make_mol_iterable(self, data: Dict[str, Any], n_mols: int = None) -> Iterable[dict]:
        """Create an iterable of molecules from raw data (implement limit on number of mols). Must be implemented by subclasses."""

    @abstractmethod
    def read_atoms(self, i: int, data: dict) -> dict:
        """Read atoms information from raw data. Must be implemented by subclasses."""


class QM7Parser(Parser):
    urls = ["http://quantum-machine.org/data/qm7.mat"]
    file_names = ["qm7.mat"]

    def __init__(self, tag: str = "qm7"):
        super().__init__(tag)

    def load_raw_data(self, file_name: str) -> dict:
        return scipy.io.loadmat(file_name)

    def make_mol_iterable(self, data: dict, n_mols: int = None) -> Iterable[dict]:
        data = data[self.file_names[0]]  # Single url

        if n_mols is None:
            n_mols = data["R"].shape[0]  # use all molecules

        positions = [data["R"][i] for i in range(n_mols)]
        atomic_nums = [data["Z"][i] for i in range(n_mols)]

        return [{"atom_pos": positions[i], "atom_nums": atomic_nums[i]} for i in range(n_mols)]

    def read_atoms(self, i: int, raw_data: Any) -> dict:
        # convert to angstrom
        factor = np.float32(0.52917721092)
        atom_pos = raw_data["atom_pos"].astype("float32") * factor

        atom_nums = raw_data["atom_nums"].astype(int).tolist()
        num_atoms = sum([True if atom_num != 0 else False for atom_num in atom_nums])

        atom_syms = [chemical_symbols[atom_num] for atom_num in atom_nums[:num_atoms]]
        atom_pos = atom_pos[:num_atoms]
        atom_nums = atom_nums[:num_atoms]

        assert len(atom_syms) == len(atom_nums) == atom_pos.shape[0] == num_atoms, (
            f"Lengths do not match: {len(atom_syms)}, {len(atom_nums)}, {atom_pos.shape[0]}, {num_atoms}"
        )

        return dict(
            num_atoms=num_atoms,
            atom_pos=atom_pos,
            atom_syms=atom_syms,
            atom_nums=atom_nums,
        )


class QM9Parser(Parser):
    urls = ["https://figshare.com/ndownloader/files/3195389"]
    file_names = ["dsgdb9nsd.xyz.tar.bz2"]

    def __init__(self, tag: str = "qm9"):
        super().__init__(tag)

        self.xyz_folder = os.path.join(self.raw_dir_path, "xyz")
        os.makedirs(self.xyz_folder, exist_ok=True)

    def download_raw_data(self, file_name: str) -> None:
        """Download and then unpack the tar file."""
        super().download_raw_data(file_name)
        with tarfile.open(os.path.join(self.raw_dir_path, file_name), "r:bz2") as tar:
            tar.extractall(self.xyz_folder)

    def load_raw_data(self, file_name: str) -> List[str]:
        xyz_files = sorted([f for f in os.listdir(self.xyz_folder) if f.endswith(".xyz")])
        return xyz_files

    def make_mol_iterable(self, data: Dict[str, List[str]], n_mols: int = None) -> List[str]:
        xyz_files = data[self.file_names[0]]  # Single url

        if n_mols is None:
            n_mols = len(xyz_files)

        start_id = 0  # increase to use larger molecules
        xyz_files = xyz_files[start_id : start_id + n_mols]

        return xyz_files if isinstance(xyz_files, list) else [xyz_files]

    def read_atoms(self, i: int, raw_data: Any) -> dict:
        lines = self._read_lines(xyz_file=raw_data)

        num_atoms = int(lines[0])
        coords = [line.split() for line in lines[2 : num_atoms + 2]]
        atom_syms = [coord[0] for coord in coords]
        atom_nums = [atomic_numbers[sym] for sym in atom_syms]
        atom_pos = self._read_coordinates(coords)

        assert len(atom_syms) == len(atom_nums) == atom_pos.shape[0] == num_atoms, (
            f"Lengths do not match: {len(atom_syms)}, {len(atom_nums)}, {atom_pos.shape[0]}, {num_atoms}"
        )

        smiles_from_file, _ = self._read_smiles(lines, num_atoms=num_atoms)

        return dict(
            num_atoms=num_atoms,
            atom_pos=atom_pos,
            atom_syms=atom_syms,
            atom_nums=atom_nums,
            smiles_from_file=smiles_from_file,
        )

    def _read_lines(self, xyz_file: str) -> List[str]:
        with open(os.path.join(self.xyz_folder, xyz_file), "r") as f:
            lines = f.readlines()
        return lines

    def _read_coordinates(self, coords: List[str]) -> np.ndarray:
        atom_pos = []
        for coord in coords:
            pos = [float(val.replace("*^", "e").replace("*", "")) for val in coord[1:4]]
            atom_pos.append(pos)
        return np.array(atom_pos, dtype=np.float32)

    def _read_smiles(self, lines: List[str], num_atoms: int) -> Tuple[str, str]:
        smiles0, smiles1 = lines[num_atoms + 3].split()
        if self._contains_charged_atoms(smiles0):
            print(f"Charged atoms detected in SMILES: {smiles0}")
        return smiles0, smiles1

    def _contains_charged_atoms(self, smiles_str: str) -> bool:
        return "+" in smiles_str or "-" in smiles_str


class tmQMParser(Parser):
    """
    Transition metal complexes from paper: https://pubs.acs.org/doi/10.1021/acs.jcim.0c01041
    Alternatively, use this data source for download: https://www.openqdc.io/datasets/tmqm
    """

    urls = [
        "https://raw.githubusercontent.com/uiocompcat/tmQM/refs/heads/master/old_tmQM/old_tmQM_X1.xyz.gz",
        "https://raw.githubusercontent.com/uiocompcat/tmQM/refs/heads/master/old_tmQM/old_tmQM_X2.xyz.gz",
    ]

    file_names = ["old_tmQM_X1.xyz.gz", "old_tmQM_X2.xyz.gz"]

    def __init__(self, tag: str = "tmqm"):
        super().__init__(tag)

    def _raw_file_path(self, file_name: str) -> str:
        return os.path.join(self.raw_dir_path, file_name)

    def load_raw_data(self, file_name: str) -> List[str]:
        """Load raw data from gzipped XYZ files."""
        with gzip.open(self._raw_file_path(file_name), "rt") as f:
            content = f.read()
        splits = content.split("\n\n")
        return [mol for mol in splits if mol]  # Remove empty entries

    def make_mol_iterable(self, data: Dict[str, List[str]], n_mols: int = None) -> List[dict]:
        """We expect 2 files so they must first be merged."""
        xyz_files = data[self.file_names[0]] + data[self.file_names[1]]
        print(f"Number of molecules in merged tmQM dataset: {len(xyz_files)}")
        return xyz_files[:n_mols]

    def read_atoms(self, i: int, raw_data: Any) -> dict:
        """Read atoms information from raw data."""
        blocks = raw_data.split("\n")

        # Block 0: n_atoms
        n_atoms = int(blocks[0])

        # Block 1: mol_info
        mol_info = blocks[1].split("|")
        CSD_code = mol_info[0].split()[-1]
        charge = mol_info[1].split()[-1]
        spin = mol_info[2].split()[-1]
        stoichiometry = mol_info[3].split()[-1]
        mnd = mol_info[4].split()[-1]

        # Block 2: position_block
        position_block = blocks[2 : n_atoms + 2]
        atomic_symbols = []
        pos = []
        for atom_line in position_block:
            atomic_symbol, x, y, z = atom_line.split()
            atomic_symbols.append(atomic_symbol)
            pos.append([float(x), float(y), float(z)])
        pos = np.array(pos, dtype=np.float32)
        assert len(atomic_symbols) == len(pos) == n_atoms, (
            f"Lengths do not match: {len(atomic_symbols)}, {len(pos)}, {n_atoms}"
        )

        atomic_nums = [atomic_numbers[atomic_symbol] for atomic_symbol in atomic_symbols]

        return dict(
            num_atoms=n_atoms,
            atom_pos=pos,
            atom_syms=atomic_symbols,
            atom_nums=atomic_nums,
            CSD_code=CSD_code,
            charge=charge,
            spin=spin,
            stoichiometry=stoichiometry,
            mnd=mnd,
        )


class TMQMParserSmall(tmQMParser):
    """Only include molecules with less than max_atoms."""

    def __init__(self, tag: str = "tmqm-max50", max_atoms: int = 10):
        super().__init__(tag=tag)
        self.max_atoms = max_atoms

    def make_mol_iterable(self, data: Dict[str, List[str]], n_mols: int = None) -> List[dict]:
        xyz_files = data[self.file_names[0]] + data[self.file_names[1]]
        print(f"Number of molecules in merged tmQM dataset: {len(xyz_files)}")

        # Filter out molecules with more than max_atoms
        print(f"Filtering out molecules with more than {self.max_atoms} atoms")
        mol_sizes = []
        for xyz_file in xyz_files:
            blocks = xyz_file.split("\n")
            mol_sizes.append(int(blocks[0]))

        xyz_files = [xyz_file for (xyz_file, mol_size) in zip(xyz_files, mol_sizes) if mol_size <= self.max_atoms]
        print(f"Number of molecules in filtered tmQM dataset: {len(xyz_files)}")

        if n_mols:
            if len(xyz_files) < n_mols:
                print(f"Warning: Only {len(xyz_files)} rather than {n_mols} molecules passed the filter.")

        return xyz_files[:n_mols]
