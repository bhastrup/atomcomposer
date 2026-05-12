import os
from pathlib import Path

import torch

from src.tools import util
from src.performance.energetics import str_to_EnergyUnit, EnergyUnit
from src.tools.model_util import ModelIO
from src.tools.env_util import EnvMaker
from src.performance.single_cpkt.evaluator import EvaluatorIO, SingleCheckpointEvaluator


class PathHelper:
    def __init__(self, run_dir: str, model_name: str, log_name: str = "pretrain_run-0.json", tag: str = "local_eval"):
        run_path = Path(os.getcwd()) / run_dir

        self.results_dir = run_path / "results"
        self.model_dir = run_path / "models"
        self.log_dir = run_path / "logs"
        self.eval_save_dir = self.results_dir / tag

        self.cp_path = self.model_dir / model_name
        self.log_path = self.log_dir / log_name


def process_config(cf: dict) -> dict:
    cf["calc_rew"] = False
    cf["energy_unit"] = str_to_EnergyUnit(cf["energy_unit"]) if "energy_unit" in cf else EnergyUnit.EV
    return cf


def get_model(cf: dict, ph: PathHelper, tag: str = None):
    device = util.init_device("cuda" if torch.cuda.is_available() else "cpu")
    model_handler = ModelIO(directory=ph.model_dir, tag=tag)
    model, start_num_steps = model_handler.load(device=device, path=ph.cp_path)
    model.set_device(device)
    return model, start_num_steps


def get_evaluator_prop(env_maker: EnvMaker, ph: PathHelper, prop_factor: int = 2) -> SingleCheckpointEvaluator:
    """Build Evaluator from EnvMaker"""
    _, _, eval_envs_big = env_maker.make_envs()
    benchmark_energies = env_maker.ref_data.get_mean_energies()
    eval_formulas = util.get_str_formulas_from_vecenv(eval_envs_big)

    # Build evaluator
    return SingleCheckpointEvaluator(
        eval_envs=eval_envs_big,
        reference_smiles=env_maker.get_reference_smiles(eval_formulas),
        benchmark_energies=benchmark_energies,
        io_handler=EvaluatorIO(base_dir=ph.eval_save_dir),
        wandb_run=None,
        num_episodes_const=None,
        prop_factor=prop_factor,
    )
