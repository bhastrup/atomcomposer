import argparse
import time

from src.tools import util
from src.data.io_handler import IOHandler
from src.tools.env_util import EnvMaker
from src.performance.single_cpkt.utils import PathHelper, process_config, get_model, get_evaluator_prop

from src.tools.config import str2bool


def parse_cmd():
    parser = argparse.ArgumentParser(description="Script for sampling molecules.")
    parser.add_argument(
        "--run_dir",
        type=str,
        default="runs/Atom/name_Atom",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pretrain_run-0_CP-4_steps-10000.model",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="Single_ckpt_eval",
    )
    parser.add_argument(
        "--prop_factor",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="pretrain_run-0.json",
    )
    # perform optimization
    parser.add_argument("--relax", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--rollouts_from_file", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--features_from_file", type=str2bool, nargs="?", const=True, default=False)
    return parser.parse_args()


def get_args():
    args = parse_cmd()
    return (
        args.run_dir,
        args.model_name,
        args.log_name,
        args.prop_factor,
        args.tag,
        args.relax,
        args.rollouts_from_file,
        args.features_from_file,
    )


if __name__ == "__main__":
    run_dir, model_name, log_name, prop_factor, tag, relax, rollouts_from_file, features_from_file = get_args()

    ph = PathHelper(run_dir, model_name, log_name, tag=tag)
    config = process_config(IOHandler.read_json(ph.log_path))
    model, start_num_steps = get_model(config, ph)
    util.set_seeds(seed=config["seed"])

    # Build evaluator and generate rollouts
    env_maker = EnvMaker(config, split_method="read_split_from_disk")
    evaluator = get_evaluator_prop(env_maker, ph, prop_factor)
    evaluator.reset(ph.eval_save_dir)

    start_time_rollout = time.time()
    evaluator._rollout(ac=model, rollouts_from_file=rollouts_from_file)
    print(f"Done generating rollouts in {time.time() - start_time_rollout}")

    start_time_features = time.time()
    evaluator._calc_features(features_from_file=features_from_file, perform_optimization=relax)
    print(f"Done calculating features {time.time() - start_time_features}")

    start_time_global = time.time()
    evaluator._get_metrics_by_formula(dfs_from_file=False)

    evaluator._calc_global_metrics()
    print(f"Done calculating global features {time.time() - start_time_global}")
