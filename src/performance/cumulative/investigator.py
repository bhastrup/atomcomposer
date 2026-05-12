import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.data.io_handler import IOHandler
from src.data.reference_dataloader import ReferenceDataLoader
from src.performance.energetics import EnergyUnit
from src.performance.metrics import get_compact_smiles
from src.performance.cumulative.cum_io import CumulativeIO, process_raw_data


DiscoveryCountType = Dict[str, List[int]]
AggregateDiscoveryCountType = Dict[str, DiscoveryCountType]


class CummulativeInvestigator:
    def __init__(self, cum_io: CumulativeIO, data_dir: Path, step_max: int = None, mol_dataset: str = "qm7") -> None:
        # Currently in a slightly awkward state where the metrics are indexed by tag and mol_dataset
        # but the rest of the reference data corresponds to a single mol_dataset.

        self.data_dir = data_dir
        self.step_max = step_max

        self.cum_io = cum_io

        # Directories
        self.plot_dir = (cum_io.save_dir / ".." / "discovery_plots").resolve()
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_dir = (cum_io.save_dir / ".." / "metrics").resolve()
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Load the bag representations
        # Temporary hack: change key name from 'out_of_sample' to 'oos_sample' in the bag_repr dict
        self.bag_reprs = cum_io.load_bag_reprs()
        self.bag_reprs = {
            tag if tag != "out_of_sample" else "oos_sample": formulas for tag, formulas in self.bag_reprs.items()
        }

        self.ref_data = ReferenceDataLoader(data_dir=data_dir).load_and_polish(
            mol_dataset, EnergyUnit.EV, fetch_df=False
        )

        self.discovery_metrics = {}
        self.aggregate_discovery_metrics = {}

    def dm_file_path(self, tag: str, mol_dataset: str, aggregate: bool = False) -> Path:
        """Get the file path for the discovery metrics"""
        name_tag = self._get_name_tag(tag, mol_dataset)
        if aggregate:
            return self.metrics_dir / f"aggregate_discovery_metrics_{name_tag}_{self.step_max}.json"
        else:
            return self.metrics_dir / f"discovery_metrics_{name_tag}_{self.step_max}.json"

    def _get_name_tag(self, tag: str, mol_dataset: str) -> str:
        return f"{tag}_{mol_dataset}"

    @staticmethod
    def _aggregate_discovery_metrics(dm: DiscoveryCountType) -> AggregateDiscoveryCountType:
        """Aggregate the discovery metrics"""

        data = dict(
            old_data=sum(dm["old_data"]),
            rediscovered=sum(dm["rediscovered"]),
            rediscovery_ratio=sum(dm["rediscovered"]) / sum(dm["old_data"]),
            novel=sum(dm["novel"]),
            expansion_ratio=sum(dm["novel"]) / sum(dm["old_data"]),
        )

        logging.debug(f"old_data: {data['old_data']}")
        logging.debug(f"rediscovered: {data['rediscovered']} | rediscovery ratio: {data['rediscovery_ratio']:.3f}")
        logging.debug(f"novel: {data['novel']} | expansion ratio: {data['expansion_ratio']:.3f}")

        return data

    def load_and_process_raw_data(self, step_max: int = None) -> None:
        """Load and process the raw data"""
        if self.cum_io is not None:
            self.db_raw = self.cum_io.load_all_dbs(step_max=step_max)
            print("Done loading raw data.")
            self.output = process_raw_data(self.db_raw, self.bag_reprs, self.ref_data.smiles)

    def get_discovery_metrics(self, tag: str = "in_sample", mol_dataset: str = "qm7") -> DiscoveryCountType:
        """Get the discovery metrics"""
        name_tag = self._get_name_tag(tag, mol_dataset)
        if name_tag in self.discovery_metrics:
            return self.discovery_metrics[name_tag]
        else:
            self.discovery_metrics[name_tag] = make_discovery_metrics(
                full_db=self.output["full_db"], data_dir=self.data_dir, tag=tag, mol_dataset=mol_dataset
            )
            return self.discovery_metrics[name_tag]

    def save_discovery_metric(self, tag: str, mol_dataset: str) -> None:
        """Save the discovery metrics"""

        dm = self.get_discovery_metrics(tag, mol_dataset)
        dm_agg = self._aggregate_discovery_metrics(dm)

        IOHandler.write_json(data=dm, file_path=self.dm_file_path(tag, mol_dataset))
        IOHandler.write_json(data=dm_agg, file_path=self.dm_file_path(tag, mol_dataset, aggregate=True))

    def load_discovery_metrics(self, tag: str, mol_dataset: str) -> DiscoveryCountType:
        """Load the discovery metrics"""
        name_tag = self._get_name_tag(tag, mol_dataset)
        self.discovery_metrics[name_tag] = IOHandler.read_json(self.dm_file_path(tag, mol_dataset))
        return self.discovery_metrics[name_tag]


def make_discovery_metrics(
    full_db: dict, data_dir: Path, tag: str = "in_sample", mol_dataset: str = "qm7"
) -> DiscoveryCountType:

    db = full_db[tag]

    ref_data = ReferenceDataLoader(data_dir=data_dir).load_and_polish(mol_dataset, EnergyUnit.EV, fetch_df=False)
    formulas = list(db.keys())

    discovery_counts = {formula: {"rediscovered": 0, "novel": 0, "old_data_size": 0} for formula in formulas}

    for formula in tqdm(formulas, desc="Calculating discovery metrics"):
        if formula not in ref_data.smiles:
            raise ValueError(f"Formula {formula} not found in the reference data.")

        # Discovered smiles
        new_set = set(db[formula].molecules.keys())

        new_set_fixed = set()
        for smiles in new_set:
            new_SMILES_Compact = get_compact_smiles(smiles)
            new_set_fixed.add(new_SMILES_Compact)

        # Reference smiles
        ref_set = set(ref_data.smiles[formula])

        # Rediscovered smiles
        rediscovered_set = new_set_fixed.intersection(ref_set)

        # Novel smiles
        novel_set = new_set - rediscovered_set

        # Update the metrics
        discovery_counts[formula]["rediscovered"] = len(rediscovered_set)
        discovery_counts[formula]["novel"] = len(novel_set)
        discovery_counts[formula]["old_data_size"] = len(ref_set)

    # To list for plotting
    rediscovered = [discovery_counts[formula]["rediscovered"] for formula in formulas]
    novel = [discovery_counts[formula]["novel"] for formula in formulas]
    old_data = [discovery_counts[formula]["old_data_size"] for formula in formulas]

    # Sort by old_data_size
    formulas, rediscovered, novel, old_data = zip(
        *sorted(zip(formulas, rediscovered, novel, old_data), key=lambda x: x[3], reverse=True)
    )

    return dict(formulas=formulas, rediscovered=rediscovered, novel=novel, old_data=old_data)


def select_top_n_formulas(n: int = 10, dm: dict = None) -> Tuple[List[str], List[int], List[int], List[int]]:
    """Reduce the number of formulas to plot"""
    formulas = dm["formulas"][:n]
    rediscovered = dm["rediscovered"][:n]
    novel = dm["novel"][:n]
    old_data = dm["old_data"][:n]

    return formulas, rediscovered, novel, old_data
