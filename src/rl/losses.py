import time
import logging
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from src.agents.base import AbstractActorCritic
from src.rl.buffer import get_batch_generator, collect_data_batch, compute_mean_dict
from src.tools.util import compute_gradient_norm, to_numpy


def compute_loss(
    ac: AbstractActorCritic,
    data: dict,
    clip_ratio: float,
    vf_coef: float,
    entropy_coef: float,
    device=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred = ac.step(data["obs"], data["act"])

    old_logp = torch.as_tensor(data["logp"], device=device)
    adv = torch.as_tensor(data["adv"], device=device)
    ret = torch.as_tensor(data["ret"], device=device)

    # Policy loss
    ratio = torch.exp(pred["logp"] - old_logp)
    obj = ratio * adv
    clipped_obj = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    policy_loss = -torch.min(obj, clipped_obj).mean()

    # Entropy loss
    entropy_loss = -entropy_coef * pred["ent"].mean()

    # Value loss
    vf_loss = vf_coef * (pred["v"] - ret).pow(2).mean()

    # Total loss
    loss = policy_loss + entropy_loss + vf_loss

    # Approximate KL for early stopping
    approx_kl = (old_logp - pred["logp"]).mean()

    # Extra info
    clipped = ratio.lt(1 - clip_ratio) | ratio.gt(1 + clip_ratio)
    clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean()

    info = dict(
        policy_loss=to_numpy(policy_loss).item(),
        entropy_loss=to_numpy(entropy_loss).item(),
        vf_loss=to_numpy(vf_loss).item(),
        total_loss_ppo=to_numpy(loss).item(),
        approx_kl=to_numpy(approx_kl).item(),
        clip_fraction=to_numpy(clip_fraction).item(),
    )

    return loss, info


# Train policy with multiple steps of gradient descent
def train(
    ac: AbstractActorCritic,
    optimizer: Optimizer,
    data: Dict[str, Sequence],
    mini_batch_size: int,
    clip_ratio: float,
    target_kl: float,
    vf_coef: float,
    entropy_coef: float,
    gradient_clip: float,
    max_num_steps: int,
    device=None,
    rl_algo: str = "PPO",
) -> dict:
    infos = {}

    start_time = time.time()

    num_epochs = 0

    for i in range(max_num_steps):
        optimizer.zero_grad()

        batch_infos = []
        batch_generator = get_batch_generator(indices=np.arange(len(data["obs"])), batch_size=mini_batch_size)
        for batch_indices in batch_generator:
            data_batch = collect_data_batch(data, indices=batch_indices)

            batch_loss, batch_info = compute_loss(
                ac,
                data=data_batch,
                clip_ratio=clip_ratio,
                vf_coef=vf_coef,
                entropy_coef=entropy_coef,
                device=device,
            )

            batch_loss.backward(retain_graph=False)  # type: ignore
            batch_infos.append(batch_info)

        loss_info = compute_mean_dict(batch_infos)
        loss_info["grad_norm"] = compute_gradient_norm(ac.parameters())

        # Check KL
        if rl_algo == "PPO" and loss_info["approx_kl"] > 1.5 * target_kl:
            logging.debug(f"Early stopping at step {i} for reaching max KL.")
            break

        # Take gradient step
        logging.debug("Taking gradient step")
        torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=gradient_clip)
        optimizer.step()
        optimizer.zero_grad()

        num_epochs += 1

        # Logging
        logging.debug(f"Online {rl_algo} loss {i}: {loss_info}")
        infos.update(loss_info)

    infos["num_opt_steps"] = num_epochs
    infos["time"] = time.time() - start_time

    # if num_epochs > 0:
    #     logging.info(f'Optimization: policy loss={infos["policy_loss"]:.3f}, vf loss={infos["vf_loss"]:.3f}, '
    #                  f'entropy loss={infos["entropy_loss"]:.3f}, total loss={infos["total_loss"]:.3f}, '
    #                  f'num steps={num_epochs}')
    return infos


class EntropySchedule:
    def __init__(self, start_value: float, final_value: float, start_iter: int, end_iter: int):
        """
        Linearly ramps the entropy coefficient between start_iter and end_iter
        (both inclusive).  Outside that interval values are clamped.

        :param start_value: The entropy value at the start of the ramp.
        :param final_value: The entropy value at the end of the ramp.
        :param start_iter: Iteration at which the ramp begins.
        :param end_iter: Iteration at which the ramp ends.
        """
        self.start_value = start_value
        self.final_value = final_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def calculate(self, step: int) -> float:
        """Return the entropy coefficient at the given step."""
        if step <= self.start_iter:
            t = 0.0
        elif step >= self.end_iter:
            t = 1.0
        else:
            t = (step - self.start_iter) / (self.end_iter - self.start_iter)
        return t * (self.final_value - self.start_value) + self.start_value


class RewardCoefficientSchedule:
    def __init__(self, schedules: dict, start_iter: int, end_iter: int):
        """
        Linearly ramps one or more reward coefficients between start_iter and
        end_iter (both inclusive).  Outside that interval values are clamped.

        :param schedules: Mapping of reward name -> (start_value, final_value).
            Example: {'rew_dipole': (0.0, 2.0), 'rew_basin': (1.0, 0.0)}
        :param start_iter: Iteration at which the ramp begins.
        :param end_iter: Iteration at which the ramp ends.
        """
        self.schedules = schedules
        self.start_iter = start_iter
        self.end_iter = end_iter

    def calculate(self, step: int) -> dict:
        """Return {reward_name: current_value} for all scheduled terms."""
        if step <= self.start_iter:
            t = 0.0
        elif step >= self.end_iter:
            t = 1.0
        else:
            t = (step - self.start_iter) / (self.end_iter - self.start_iter)
        return {name: t * (sv[1] - sv[0]) + sv[0] for name, sv in self.schedules.items()}
