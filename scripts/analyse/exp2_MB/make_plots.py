import argparse
from pathlib import Path

from src.tools import util
from src.data.io_handler import IOHandler
from src.tools.env_util import EnvMaker
from src.performance.single_cpkt.utils import (
    PathHelper,
    process_config,
    get_model,
    get_evaluator_prop,
)
from src.performance.single_cpkt.discovery_plots import plot_rediscovery_novelty, plot_global_rediscovery_novelty
from src.performance.single_cpkt.energy_plots import (
    plot_rae_distributions,
    plot_rae_aggregated,
    plot_rae_one_figure,
    plot_rae_against_ref,
    get_etkdg_dict,
    plot_etkdg,
)
from src.performance.metrics import calc_rae

from src.data.reference_dataloader import ReferenceDataLoader
from src.performance.energetics import EnergyUnit

from src.tools.config import str2bool


def parse_cmd():
    parser = argparse.ArgumentParser(description="Script for generating molecular evaluation plots.")

    parser.add_argument("--run_dir", type=str, required=True, help="Directory path for the run.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model file.")
    parser.add_argument("--tag", type=str, default="Exp2-Eval", help="Tag for the evaluation.")
    parser.add_argument("--log_name", type=str, required=True, help="Name of the log file.")
    parser.add_argument("--prop_factor", type=int, default=100, help="Proportion factor for the evaluator.")

    parser.add_argument("--make_discovery_plots", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--make_energy_plots", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--rae_name", type=str, default="rae", help="Name of the RAE column in the dataframe.")
    parser.add_argument("--make_etkdg_plots", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--n_confs", type=int, default=5, help="Number of conformers to generate.")
    parser.add_argument("--f_max", type=float, default=0.10, help="Force threshold for relaxation.")
    parser.add_argument("--step_max", type=int, default=100, help="Maximum number of steps for relaxation.")

    parser.add_argument("--mol_dataset", type=str, default="QM7", help="Name of the molecular dataset.")
    parser.add_argument("--energy_unit", type=str, default=EnergyUnit.EV, help="Energy unit to use.")
    parser.add_argument(
        "--ref_data_dir",
        type=str,
        default="/home/bjaha/Documents/RL-catalyst/delight-rl/data",
        help="Directory path for reference data.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cmd()

    # Initialize paths and load model
    ph = PathHelper(args.run_dir, args.model_name, args.log_name, tag=args.tag)
    config = process_config(IOHandler.read_json(ph.log_path))
    model, start_num_steps = get_model(config, ph)
    util.set_seeds(seed=config["seed"])

    # Build evaluator and generate rollouts
    env_maker = EnvMaker(config, split_method="read_split_from_disk")
    evaluator = get_evaluator_prop(env_maker, ph, args.prop_factor)
    evaluator.reset(ph.eval_save_dir)

    # evaluator._rollout(ac=model, rollouts_from_file=True)
    evaluator._calc_features(features_from_file=True)
    evaluator._get_metrics_by_formula(dfs_from_file=False)
    evaluator._calc_global_metrics()

    # Generate discovery plots if flag is set
    if args.make_discovery_plots:
        # Per formula
        fig, ax = plot_rediscovery_novelty(
            evaluator.data["formula_metrics"], mol_dataset=config["mol_dataset"], undiscovered_gap=True
        )
        fig.savefig(ph.eval_save_dir / "rediscovery_novelty.png", dpi=300)

        # Global
        evaluator._calc_global_metrics()
        fig, ax = plot_global_rediscovery_novelty(
            evaluator.data["global_metrics"],
            mol_dataset=config["mol_dataset"],
            undiscovered_gap=True,
            vertical=True,
            ratio=True,
        )
        fig.savefig(ph.eval_save_dir / "global_rediscovery_novelty.png", dpi=300)

    # Generate energy plots if flag is set
    if args.make_energy_plots:
        rae_name = args.rae_name

        # Enhance with RAE if not already done
        for formula, df in evaluator.data["formula_dfs"].items():
            if "rae" not in df.columns or df["rae"].isnull().values.all():
                energy = df["abs_energy"].values
                benchmark_energy = evaluator.benchmark_energies[formula]
                n_atoms = util.str_formula_to_size(formula)
                df["rae"] = calc_rae(energy, benchmark_energy, n_atoms)

        # Plot RAE distributions
        fig, ax = plot_rae_distributions(evaluator.data["formula_dfs"], column_name=rae_name)
        fig.savefig(ph.eval_save_dir / "rae_distributions.png", dpi=300)

        # Aggregate RAE values
        fig, ax = plot_rae_aggregated(evaluator.data["formula_dfs"], column_name=rae_name)
        fig.savefig(ph.eval_save_dir / "rae_aggregate.png", dpi=300)

        # Plot RAE values in one figure (each formula is visualized separately)
        fig, ax = plot_rae_one_figure(evaluator.data["formula_dfs"], column_name=rae_name)
        fig.savefig(ph.eval_save_dir / "rae_one_figure.png", dpi=300)

        # Calculate RAE against reference data
        ref_data_loader = ReferenceDataLoader(data_dir=Path(args.ref_data_dir))
        ref_data = ref_data_loader.load_and_polish(
            mol_dataset=args.mol_dataset, new_energy_unit=args.energy_unit, fetch_df=False
        )

        test_formulas = list(evaluator.data["formula_dfs"].keys())
        energy_dict = {formula: ref_data.energies[formula] for formula in test_formulas}
        ref_rae_values = []
        for formula, formula_energies in energy_dict.items():
            ref_rae_values.extend(
                calc_rae(
                    energy=formula_energies,
                    benchmark_energy=ref_data.get_mean_energies()[formula],
                    n_atoms=util.str_formula_to_size(formula),
                )
            )

        fig, ax = plot_rae_against_ref(evaluator.data["formula_dfs"], ref_rae_values, column_name=rae_name)
        fig.savefig(ph.eval_save_dir / "rae_against_ref.png", dpi=300)

    # Generate ETKDG plots if flag is set
    if args.make_etkdg_plots:
        rollouts = evaluator._rollout(ac=None, rollouts_from_file=True)
        results = get_etkdg_dict(
            formula_dfs=evaluator.data["formula_dfs"],
            rollouts=rollouts,
            n_confs=args.n_confs,
            fmax=args.f_max,
            step_max=args.step_max,
            n_smiles_max=5,
            max_smiles_per_formula=1,
        )

        fig, ax = plot_etkdg(results)
        fig.savefig(ph.eval_save_dir / "etkdg_hist.png", dpi=300)
