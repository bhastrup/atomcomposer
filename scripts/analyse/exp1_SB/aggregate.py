from typing import List
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools import util
from src.data.io_handler import IOHandler


# use fewer decimal places
pd.set_option("display.float_format", lambda x: f"{x:.2f}" if isinstance(x, float) else str(x))


def fix_eval_formulas(eval_formulas: List[str]) -> List[str]:
    return [util.bag_tuple_to_str_formula(util.str_formula_to_bag_tuple(f)) for f in eval_formulas]


if __name__ == "__main__":
    n_seeds = 3
    tag = "EXP1_27500"
    eval_formulas = fix_eval_formulas(["C3H5NO3", "C4H7N", "C3H8O", "C7H10O2", "C7H8N2O2"])
    print(eval_formulas)

    run_names = ["final-ent15-AV", "final-ent15-A", "final-ent15-F", "final-ent15-FV", "final-ent15-AFV"]

    def get_path(run_name: str, seed: int, formula: str) -> Path:
        return Path(f"runs/{run_name}/seed_{seed}/results/{tag}/{formula}/metrics.json")

    # Calc mean and std over seeds
    data = {}
    for run_name in run_names:
        data[run_name] = {}

        for formula in eval_formulas:
            # Load metrics from each seed
            paths = [get_path(run_name, seed, formula) for seed in range(n_seeds)]
            existing_paths = [path for path in paths if path.exists()]

            if not len(existing_paths) == n_seeds:
                print(f"Missing {n_seeds - len(existing_paths)} paths for {run_name} {formula}")
                continue
            seed_metrics = [IOHandler.read_json(path) for path in existing_paths]

            # Initialize formula dict
            data[run_name][formula] = {}

            # Get metrics from first seed to know what metrics exist
            metrics = seed_metrics[0].keys()

            # Calculate mean and std for each metric across seeds
            for metric in metrics:
                metric_values = [d[metric] for d in seed_metrics]
                # Filter out None values before calculating statistics
                valid_values = [v for v in metric_values if v is not None]
                if valid_values:
                    data[run_name][formula][metric] = np.mean(valid_values)
                    data[run_name][formula][metric + "_std"] = np.std(valid_values)
                else:
                    data[run_name][formula][metric] = None
                    data[run_name][formula][metric + "_std"] = None

    # Write pretty table
    keep_cols = ["n_unique", "n_unique_std"]
    for run_name in list(data.keys()):
        if data[run_name]:
            print(run_name, ":")
            df = pd.DataFrame(data[run_name]).T
            df = df.loc[:, keep_cols]
            print(df)
        else:
            print(f"Nothing to display for {run_name}")
