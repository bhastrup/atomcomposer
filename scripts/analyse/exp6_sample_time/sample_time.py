"""
Exp 6: Pure sampling time benchmark.

Measures wall-clock time to generate N molecules for each of the 20 QM7 test
formulas, with no energy, validity, or any post-processing calculations.

All 20 formulas sampled simultaneously in one vectorised call.
"""

import argparse
import json
import os
import time


from src.tools import util
from src.data.io_handler import IOHandler
from src.rl.envs.env_no_reward import HeavyFirstNoReward
from src.rl.env_container import SimpleEnvContainer
from src.performance.single_cpkt.utils import PathHelper, process_config, get_model
from src.rl.rollouts import rollout_stoch
from src.rl.reward import InteractionReward


def parse_cmd():
    parser = argparse.ArgumentParser(description="Sampling time benchmark (no energy/validity).")
    parser.add_argument("--run_dir", type=str, default="runs/nat-com-training/A/seed_0")
    parser.add_argument("--model_name", type=str, default="pretrain_run-0_CP-12_steps-30000.model")
    parser.add_argument("--log_name", type=str, default="pretrain_run-0.json")
    parser.add_argument("--tag", type=str, default="exp6_sample_time")
    parser.add_argument("--n_molecules", type=int, default=100, help="Number of molecules to sample per formula.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cmd()

    # ------------------------------------------------------------------ setup
    ph = PathHelper(args.run_dir, args.model_name, args.log_name, tag=args.tag)
    config = process_config(IOHandler.read_json(ph.log_path))
    model, _ = get_model(config, ph)
    util.set_seeds(seed=config["seed"])

    print(f"Model loaded from: {ph.cp_path}")
    print(f"Sampling {args.n_molecules} molecules per formula\n")

    # -------------------------------------------------------- parallel timing

    experiment = "exp1_SB"

    if experiment == "exp1_SB":
        eval_formulas = ["C3H5NO3", "C4H7N", "C3H8O", "C7H10O2", "C7H8N2O2"]
        n_extension = 20
        # Extend each base formula by repeated inclusion, e.g., 5 times each for benchmarking
        eval_formulas_extended = []
        for formula in eval_formulas:
            eval_formulas_extended.extend([formula] * n_extension)

        eval_formulas = eval_formulas_extended
    elif experiment == "hold_out_20":
        split = IOHandler.read_json(f"data/{config['mol_dataset'].lower()}/processed/split.json")
        eval_formulas = split["test"]

    eval_envs = SimpleEnvContainer(
        [
            HeavyFirstNoReward(
                reward=InteractionReward(reward_coefs={}),
                observation_space=model.observation_space,
                action_space=model.action_space,
                formulas=[util.string_to_formula(eval_formulas[i])],
                min_atomic_distance=config["min_atomic_distance"],
                max_solo_distance=config["max_solo_distance"],
                min_reward=config["min_reward"],
                energy_unit=config["energy_unit"],
            )
            for i in range(len(eval_formulas))
        ]
    )

    test_formulas = util.get_str_formulas_from_vecenv(eval_envs)
    print(f"Test formulas ({len(test_formulas)}): {test_formulas}\n")

    print("=== Phase 1: Parallel (all 20 formulas in one call) ===")
    t_par_start = time.perf_counter()
    rollout_stoch(model, eval_envs, num_episodes=args.n_molecules)
    t_par_total = time.perf_counter() - t_par_start
    print(f"Total wall time (parallel): {t_par_total:.2f}s  ({t_par_total / len(test_formulas):.2f}s per formula)\n")

    results = {
        "run_dir": args.run_dir,
        "model_name": args.model_name,
        "n_molecules": args.n_molecules,
        "n_formulas": len(test_formulas),
        "test_formulas": test_formulas,
        "parallel_total_s": round(t_par_total, 4),
        "parallel_per_formula_s": round(t_par_total / len(test_formulas), 4),
    }

    out_dir = os.path.join(args.run_dir, "results", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "timing.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {out_path}")
