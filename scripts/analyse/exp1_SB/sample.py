import argparse
import time

from src.tools import util
from src.data.io_handler import IOHandler
from src.tools.env_util import EnvMakerNoRef
from src.performance.single_cpkt.utils import (
    PathHelper,
    process_config,
    get_model,
)
from src.performance.single_cpkt.evaluator import EvaluatorIO, SingleCheckpointEvaluator


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
        "--num_episodes_const",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="pretrain_run-0.json",
    )
    return parser.parse_args()


def get_args():
    args = parse_cmd()
    return args.run_dir, args.model_name, args.log_name, args.num_episodes_const, args.tag


if __name__ == "__main__":
    run_dir, model_name, log_name, num_episodes_const, tag = get_args()

    split_method = "hardcoded"
    eval_formulas = ["C3H5NO3", "C4H7N", "C3H8O", "C7H10O2", "C7H8N2O2"]

    ph = PathHelper(run_dir, model_name, log_name, tag=tag)
    config = process_config(IOHandler.read_json(ph.log_path))
    model, start_num_steps = get_model(config, ph)
    util.set_seeds(seed=config["seed"])

    env_maker = EnvMakerNoRef(
        cf=config,
        train_formulas=None,
        eval_formulas=eval_formulas,
        eval_only=True,
        action_space=model.action_space,
        observation_space=model.observation_space,
    )
    _, eval_envs, _ = env_maker.make_envs()

    evaluator = SingleCheckpointEvaluator(
        eval_envs=eval_envs,
        reference_smiles=None,
        benchmark_energies=None,
        io_handler=EvaluatorIO(base_dir=ph.eval_save_dir),
        num_episodes_const=num_episodes_const,
        prop_factor=None,
    )

    evaluator.reset(ph.eval_save_dir)

    start_time_rollout = time.time()
    evaluator._rollout(ac=model, rollouts_from_file=False)
    print(f"Done generating rollouts in {time.time() - start_time_rollout}")

    start_time_features = time.time()
    evaluator._calc_features(
        features_from_file=False,
        perform_optimization=False,
    )
    print(f"Done calculating features {time.time() - start_time_features}")

    start_time_metrics = time.time()
    evaluator._get_metrics_by_formula(dfs_from_file=False)
    print(f"Done calculating metrics {time.time() - start_time_metrics}")

    exit()

    evaluator._calc_global_metrics(size_weighted=False)
    print("Done calculating global metrics")
    print(f"global_metrics: {evaluator.data['global_metrics']}")

    print(f"Total time {time.time() - start_time_rollout}")
