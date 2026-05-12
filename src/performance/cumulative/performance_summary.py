import logging
from datetime import datetime
from typing import List, Dict

import numpy as np

from src.rl.spaces import ObservationType
from src.tools import util
from src.rl.buffer import DynamicPPOBuffer, compute_buffer_stats

from src.agents.base import AbstractActorCritic
from src.agents.painn.agent import PainnAC
from src.tools.util import count_vars


class Logger:
    """Base class for logging training and evaluation metrics"""

    def __init__(self, cf: dict = None, model: AbstractActorCritic = None, start_num_iter: int = 0) -> None:

        self.cf = cf
        self._init_wandb(cf, model)

        # Number of steps per iteration
        if "num_steps_per_iter" in cf:
            self.num_steps_per_iter = cf["num_steps_per_iter"]
        elif "config_ft" in cf and "num_steps_per_iter" in cf["config_ft"]:
            self.num_steps_per_iter = cf["config_ft"]["num_steps_per_iter"]
        else:
            raise ValueError("num_steps_per_iter not found in config")

        # Initialize counters
        self.n_eps_current_rollout = 0
        self.total_num_iter = start_num_iter
        # self.num_env_steps = self.num_steps_per_iter * start_num_iter

    def increment_total_num_iter(self):
        self.total_num_iter += 1

    def get_num_steps(self):
        return self.num_steps_per_iter * self.total_num_iter

    # def _increase_step_count(self, num_steps: int) -> None:
    #     self.num_env_steps += num_steps

    def _init_wandb(self, cf: dict, model: AbstractActorCritic) -> None:
        var_counts = count_vars(model)
        logging.info(f"Number of parameters: {var_counts}")

        if cf["save_to_wandb"]:
            import wandb

            now = datetime.now()
            dt_string = now.strftime("%d_%m_%H_%M")
            cf["wandb_name"] = cf["wandb_name"] + "_" + dt_string

            wandb_run = wandb.init(
                config=cf,
                mode=cf["wandb_mode"],
                entity=cf["entity"],
                project=cf["wandb_project"],
                group=cf["wandb_group"],
                name=cf["wandb_name"],
                job_type=cf["wandb_job_type"],
                reinit=True,
            )

            if cf["wandb_watch_model"] and model is not None:
                wandb_run.watch(model, log="gradients", log_freq=1000)

        self.wandb_run = wandb_run if cf["save_to_wandb"] else None

    def _simple_rollout_logging(self, rollout, name: str = "train"):
        if name == "train":
            rollout_name = "Training rollout"
        elif name == "eval":
            rollout_name = "Evaluation rollout"

        logging.info(
            f"{rollout_name}: return={rollout['return_mean']:.3f} ({rollout['return_std']:.1f}), "
            f"episode length={rollout['episode_length_mean']:.1f}"
        )

        if self.wandb_run:
            rollout = {f"{k}_{name}": v for k, v in rollout.items()}
            rollout.update({"total_num_iter": self.total_num_iter, "num_env_steps": self.get_num_steps()})
            self.wandb_run.log(rollout)

    def save_rollout_and_info(
        self,
        info_saver: util.InfoSaver,
        rollout_saver: util.RolloutSaver,
        save_rollout: bool,
        rollout: dict,
        buffer: DynamicPPOBuffer,
        name: str,
        total_num_iter: int,
    ):

        total_num_steps = total_num_iter * self.num_steps_per_iter

        self._simple_rollout_logging(rollout, name=name)

        if info_saver:
            rollout["total_num_steps"] = total_num_steps
            rollout.update(compute_buffer_stats(buffer))
            info_saver.save(rollout, name=name)

        if rollout_saver and save_rollout:
            rollout_saver.save(buffer, num_steps=total_num_steps, info=name)

    def save_optimization_data(self, info_saver: util.InfoSaver, opt_info: dict, total_num_steps: int):
        if info_saver:
            opt_info["total_num_steps"] = total_num_steps
            info_saver.save(opt_info, name="opt")
            if self.wandb_run is not None:
                try:
                    self.wandb_run.log(opt_info)
                except Exception as e:
                    print(f"An error occurred with wandb log opt_info: {e}")

    def save_episode_RL(self, state: ObservationType, total_reward: float, info: dict, name: str) -> None:
        pass

    def update_and_log_data(self, total_num_iter: int) -> None:
        pass


class MultibagLogger(Logger):
    def __init__(self, cf: dict, model: AbstractActorCritic) -> None:
        super().__init__(cf, model)

        self.num_episodes_rollout = 0

    def save_episode_RL(self, state: ObservationType, total_reward: float, info: dict, name: str) -> None:
        self.num_episodes_rollout += 1
        if "new_rewards" in info:
            info["new_rewards"] = {f"{k}_{name}": v for k, v in info["new_rewards"].items()}
        if "mol_info" in info:
            info["mol_info"].pop("mol")
        if self.wandb_run is not None:
            self.wandb_run.log(info)

    def log_after_rollout(self, info: dict, reward_terms: List[Dict[str, float]], name: str) -> None:
        """Log the reward terms and other info after full rollout, i.e. many episodes."""
        # List of dicts to dict of lists
        reward_terms = {k: np.mean([d[k] for d in reward_terms]) for k in reward_terms[0]}
        # reward_terms = {k: np.mean(v) for k, v in reward_terms.items()}
        # reward_terms = {f'{k}_{name}': v for k, v in reward_terms.items()}
        info.update(reward_terms)

        # if self.wandb_run is not None:
        #    self.wandb_run.log(info)

    def log_gaussian_stds(self, ac: AbstractActorCritic, total_num_iter: int) -> None:
        if self.wandb_run is None:
            return None
        if isinstance(ac, PainnAC):
            stds = ac.log_stds.exp().detach().cpu().numpy()
            self.wandb_run.log(
                {
                    "total_num_steps": total_num_iter,
                    "distance_std": stds[0],
                    "angle_std": stds[1],
                    "dihedral_std": stds[2],
                }
            )
