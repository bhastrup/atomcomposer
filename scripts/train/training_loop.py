from typing import Optional

import torch
from torch.optim.optimizer import Optimizer

from src.tools.model_util import ModelIO
from src.tools.util import RolloutSaver, InfoSaver
from src.agents.base import AbstractActorCritic
from src.rl.buffer_container import PPOBufferContainer
from src.rl.env_container import VecEnv
from src.rl.rollouts import batch_rollout_with_logging
from src.rl.losses import (
    train,
    EntropySchedule,
    RewardCoefficientSchedule,
)
from src.rl.reward import InteractionReward
from src.performance.cumulative.performance_summary import Logger


def training_loop(
    total_num_iter: int,
    ac: AbstractActorCritic,
    optimizer_online: Optimizer,
    mini_batch_size: int,
    device: torch.device,
    save_freq: int = 50,
    model_handler: Optional[ModelIO] = None,
    config: dict = None,
    config_ft: dict = None,
    train_envs_online: VecEnv = None,
    rl_algo_online: str = "PPO",
    logger: Logger = None,
    rollout_saver: Optional[RolloutSaver] = None,
    info_saver: Optional[InfoSaver] = None,
    entropy_schedule: EntropySchedule = None,
    reward_coef_schedule: RewardCoefficientSchedule = None,
    reward: InteractionReward = None,
):
    assert rl_algo_online == "PPO", "Unknown online RL algorithm"

    num_training_iterations = config_ft["max_num_steps"] // config_ft["num_steps_per_iter"]
    for i in range(num_training_iterations):
        infos = {}

        # Update scheduled reward coefficient
        if reward_coef_schedule is not None and reward is not None:
            reward.reward_coefs.update(reward_coef_schedule.calculate(total_num_iter))

        # Save model (big checkpoint)
        if model_handler:
            if total_num_iter in model_handler._checkpoints:
                model_handler.save_if_checkpoint(ac, total_num_iter)

        ####  Collect data online
        if rl_algo_online:
            online_container = PPOBufferContainer(size=train_envs_online.get_size(), gamma=1.0, lam=0.97)
            online_rollout = batch_rollout_with_logging(
                ac=ac,
                envs=train_envs_online,
                buffer_container=online_container,
                num_steps=config_ft["num_steps_per_iter"],
                logger=logger,
            )
            online_buffer = online_container.merge()
            if logger:
                logger.save_rollout_and_info(
                    info_saver=info_saver,
                    rollout_saver=rollout_saver,
                    rollout=online_rollout,
                    save_rollout=False,
                    buffer=online_buffer,
                    name="train",
                    total_num_iter=total_num_iter,
                )
            online_data = online_buffer.get_data()

        # Optimize
        if online_rollout["return_std"] != 0.0:
            print(f"starting optimization at step {total_num_iter}")
            opt_info_online = train(
                ac=ac,
                optimizer=optimizer_online,
                data=online_data,
                mini_batch_size=mini_batch_size,
                clip_ratio=config_ft["clip_ratio"],
                vf_coef=config_ft["vf_coef"],
                entropy_coef=entropy_schedule.calculate(step=total_num_iter)
                if entropy_schedule
                else config_ft["entropy_coef"],
                target_kl=config_ft["target_kl"],
                gradient_clip=config_ft["gradient_clip"],
                max_num_steps=config_ft["max_num_train_iters"],
                device=device,
                rl_algo="PPO",
            )
            infos.update(opt_info_online)

        if logger and infos:
            total_num_steps = total_num_iter * config_ft["num_steps_per_iter"]
            logger.save_optimization_data(
                info_saver=info_saver,
                opt_info=infos,
                total_num_steps=total_num_steps,
            )

        # Save model (perpetually)
        if model_handler and (total_num_iter % save_freq == 0):
            model_handler.save(ac, num_steps=total_num_iter)

        total_num_iter += 1
        if logger:
            logger.increment_total_num_iter()
        if total_num_iter >= model_handler._checkpoints[-1]:
            break

    return total_num_iter
