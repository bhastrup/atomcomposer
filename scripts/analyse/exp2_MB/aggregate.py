from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io_handler import IOHandler


# use fewer decimal places
pd.set_option("display.float_format", lambda x: f"{x:.2f}" if isinstance(x, float) else str(x))


if __name__ == "__main__":
    n_seeds = 3
    tag = "EXP2_p100_RELAX"

    run_names_map = {
        "final-ent15-A": "MB-A",
        "final-ent15-AV": "MB-AV",
        "final-ent15-F": "MB-F (\\textsc{MolGym} rew)",
        "final-ent15-FV": "MB-FV",
        "final-ent15-AFV": "MB-AFV",
    }
    run_names = list(run_names_map.keys())

    def get_path(run_name: str, seed: int) -> Path:
        return Path(f"runs/{run_name}/seed_{seed}/results/{tag}/global_metrics.json")

    # Calc mean and std over seeds
    data = {}
    for run_name in run_names:
        paths = [get_path(run_name, seed) for seed in range(n_seeds)]
        existing_paths = [path for path in paths if path.exists()]

        if not len(existing_paths) == n_seeds:
            print(f"Missing {n_seeds - len(existing_paths)} paths for {run_name}")
            continue

        data[run_name] = {}

        # Load metrics from each seed
        seed_metrics = [IOHandler.read_json(path) for path in existing_paths]

        # Get metrics from first seed to know what metrics exist
        metrics = seed_metrics[0].keys()

        # Calculate mean and std for each metric across seeds
        for metric in metrics:
            metric_values = [d[metric] for d in seed_metrics]
            # Filter out None values before calculating statistics
            valid_values = [v for v in metric_values if v is not None]
            if valid_values:
                data[run_name][metric] = np.mean(valid_values)
                data[run_name][metric + "_std"] = np.std(valid_values)
            else:
                data[run_name][metric] = None
                data[run_name][metric + "_std"] = None

    # Write pretty table
    # Map long column names to shorter ones
    col_map = {
        "valid_per_sample": "valid",
        "valid_per_sample_std": "valid_std",
        "rediscovery_ratio": "redisc",
        "rediscovery_ratio_std": "redisc_std",
        "expansion_ratio": "expand",
        "expansion_ratio_std": "expand_std",
        "rae_relaxed_avg": "rae",
        "rae_relaxed_avg_std": "rae_std",
    }

    keep_cols = list(col_map.keys())

    for run_name in list(data.keys()):
        df = pd.DataFrame([data[run_name]]).loc[:, keep_cols].rename(columns=col_map)

        # Map to latex (value +- _std value)
        row = run_names_map[run_name] if run_name in run_names_map else run_name
        for col in df.columns:
            if col.endswith("_std"):
                continue
            value = df[col][0]
            std_dev = df[col + "_std"][0] if df[col + "_std"][0] is not None else None
            if value is not None:
                if std_dev is not None:
                    row += f" & \\uncertainty{{{value:.2f}}}{{{std_dev:.2f}}}"
                else:
                    row += f" & {value:.2f}"
            else:
                row += " & -"

        # Add one dummy column in expectation of an extra metric to be inserted later
        row += " & 0.00 $\pm$ 0.00 \\\\"
        print(row)
