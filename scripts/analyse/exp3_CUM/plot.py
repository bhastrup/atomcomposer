import argparse
from pathlib import Path

import numpy as np
import torch

from src.performance.cumulative.cum_io import CumulativeIO
from src.performance.cumulative.investigator import (
    CummulativeInvestigator,
)

from scripts.analyse.exp3_CUM.discovery_by_formula import (
    plot_rediscovery_novelty,
)
from scripts.analyse.exp3_CUM.time_series import plot_cumulative_counts

from src.tools.config import str2bool


def parse_cmd():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--run_dir", type=str, help="Name of the run to analyse")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory", default="data")
    parser.add_argument("--mol_dataset", type=str, help="Name of the molecule dataset", default="qm7")

    parser.add_argument("--step_max", type=str, default="None", help="Maximum number of steps to consider")
    parser.add_argument("--make_discovery_by_formula_plot", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--make_time_series_plot", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "--aggregate_across_formulas", help="Avg. over formulas", type=str2bool, nargs="?", const=True, default=False
    )

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    set_seed(seed=42)

    args = parse_cmd()
    step_max = None if args.step_max == "None" else int(args.step_max)

    tag = "in_sample"

    investigator = CummulativeInvestigator(
        CumulativeIO(save_dir=Path.cwd() / f"{args.run_dir}/results", batched=True),
        data_dir=Path.cwd() / args.data_dir,
        step_max=step_max,
        mol_dataset=args.mol_dataset,
    )
    investigator.load_and_process_raw_data(step_max)
    investigator.get_discovery_metrics(tag, args.mol_dataset)
    investigator.save_discovery_metric(tag, args.mol_dataset)

    if args.make_discovery_by_formula_plot:
        plot_rediscovery_novelty(
            inv=investigator, tag=tag, num_formulas=None, mol_dataset=args.mol_dataset, undiscovered_gap=True
        )

    if args.make_time_series_plot:
        for rediscovery in [True, False]:
            plot_cumulative_counts(
                inv=investigator,
                data_dir=Path.cwd() / args.data_dir,
                tag=tag,
                num_formulas=None,
                mol_dataset=args.mol_dataset,
                log_scale=False,
                aggregated=args.aggregate_across_formulas,
                rediscovery=rediscovery,
            )
