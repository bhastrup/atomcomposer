import argparse
import logging

from src.data.parser import (
    QM7Parser,
    QM9Parser,
)
from src.data.featurizer import Featurizer
from src.data.preprocessor import Preprocessor


def parse_cmd():
    parser = argparse.ArgumentParser(description="Preprocess molecular datasets for training.")
    parser.add_argument(
        "--mol_dataset", type=str, choices=["qm7", "qm9"], default="qm7", help="Dataset to preprocess (default: qm7)"
    )
    parser.add_argument("--n_mols", type=int, default=None, help="Number of molecules to process (default: all)")
    return parser.parse_args()


def get_parser_and_featurizer(dataset):
    qm_kwargs = {
        "include_energy": True,
        "include_smiles": True,
        "include_connectivity": True,
        "use_huckel": True,
        "smiles_compatible": True,
    }

    if dataset == "qm7":
        return QM7Parser(tag="qm7-test1"), Featurizer(**qm_kwargs)
    elif dataset == "qm9":
        return QM9Parser(tag="qm9"), Featurizer(**qm_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = parse_cmd()

    parser, featurizer = get_parser_and_featurizer(args.mol_dataset)

    processor = Preprocessor(parser, featurizer)
    processor.load_data(n_mols=args.n_mols)
    processor.preprocess()
    processor.print_success_ratio()
    processor.save_data()
    # processor.view_bad_molecules()
