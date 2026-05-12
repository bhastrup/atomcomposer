import time
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch

from ase.gui.gui import GUI  # from ase.gui.view import View

from src.agents.base import AbstractActorCritic
from src.rl.buffer_container import PPOBufferContainer
from src.rl.env_container import VecEnv
from src.tools.util import to_numpy
from src.tools import util

from src.performance.cumulative.performance_summary import Logger
from src.rl.buffer_container import PPOBufferContainerDeploy


def batch_rollout(
    ac: AbstractActorCritic,
    envs: VecEnv,
    buffer_container: PPOBufferContainer,
    num_steps: int = None,
    num_episodes: int = None,
) -> dict:
    assert num_steps is not None or num_episodes is not None

    if num_steps is not None:
        assert num_steps % envs.get_size() == 0
        num_iters = num_steps // envs.get_size()
    else:
        num_iters = np.inf

    if num_episodes is not None:
        assert envs.get_size() == 1
    else:
        num_episodes = np.inf

    start_time = time.time()

    counter = 0
    observations = envs.reset()

    while counter < num_iters and buffer_container.get_num_episodes() < num_episodes:
        predictions = ac.step(observations)

        next_observations, rewards, terminals, _ = envs.step(predictions["actions"])

        buffer_container.store(
            observations=observations,
            actions=to_numpy(predictions["a"]),
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            values=to_numpy(predictions["v"]),
            logps=to_numpy(predictions["logp"]),
        )

        # Reset environment if state is terminal to get valid next observation
        observations = envs.reset_if_terminal(next_observations, terminals)

        if counter == num_iters - 1:
            # Note: finished trajectories will not be affected by this
            predictions = ac.step(observations)
            buffer_container.finish_paths(to_numpy(predictions["v"]))

        counter += 1

    info = {
        "time": time.time() - start_time,
        "return_mean": np.mean(buffer_container.episodic_returns).item(),
        "return_std": np.std(buffer_container.episodic_returns).item(),
        "episode_length_mean": np.mean(buffer_container.episode_lengths).item(),
        "episode_length_std": np.std(buffer_container.episode_lengths).item(),
    }

    return info


class GUI_focus(GUI):
    def focus(self, x=None):
        cell = self.window["toggle-show-unit-cell"] and self.images[0].cell.any()
        if len(self.atoms) == 0 and not cell:
            self.scale = 20.0
            self.center = np.zeros(3)
            self.draw()
            return

        # Get the min and max point of the projected atom positions
        # including the covalent_radii used for drawing the atoms
        P = np.dot(self.X, self.axes)
        n = len(self.atoms)
        covalent_radii = self.get_covalent_radii()
        P[:n] -= covalent_radii[:, None]
        P1 = P.min(0)
        P[:n] += 2 * covalent_radii[:, None]
        P2 = P.max(0)
        # self.center = np.dot(self.axes, (P1 + P2) / 2)
        # self.center += self.atoms.get_celldisp().reshape((3,)) / 2

        # Add 30% of whitespace on each side of the atoms
        S = 1.3 * (P2 - P1)
        w, h = self.window.size
        if S[0] * h < S[1] * w:
            self.scale = h / S[1]
        elif S[0] > 0.0001:
            self.scale = w / S[0]
        else:
            self.scale = 1.0

        self.center = np.array([0, 0, 0])
        self.scale = 30

        self.draw()


def rollout_n_eps_per_env(
    ac: AbstractActorCritic,
    envs: VecEnv,
    buffer_container: PPOBufferContainerDeploy,
    num_episodes: Union[int, List[int]] = None,
    output_trajs: bool = False,
    render: bool = False,
    render_bonds: bool = False,
    num_episodes_combined: int = None,
) -> dict:
    """
    This function circumvents the limitation that num_episodes can only be specified when
    len(envs) == 1.
    """

    start_time = time.time()

    counter = 0
    observations = envs.reset()

    if render:
        render_id = np.random.randint(len(observations))
        print(f"render_id: {render_id}")
        atoms = ac.observation_space.canvas_space.to_atoms(observations[render_id][0])
        gui = GUI_focus(show_bonds=render_bonds)
        gui.images.initialize([atoms])
        gui.set_frame()
        gui.draw()

        # viewer = View(rotations='0.0x,0.0y,0.0z')
        # viewer.set_atoms(atoms)

    num_envs = envs.get_size()
    num_eps_total = num_envs * num_episodes if isinstance(num_episodes, int) else sum(num_episodes)
    max_steps = 100 * num_episodes if isinstance(num_episodes, int) else 100 * max(num_episodes)

    for i in range(num_envs):
        envs.environments[i].n_eps = 0
    buffer_container.ep_ret_unmixed = [[] for _ in range(num_envs)]
    buffer_container.active_buffers = [i for i in range(num_envs)]

    rollout_trajs = [[] for _ in range(num_envs)]

    while counter < max_steps and buffer_container.get_num_episodes() < num_eps_total:
        predictions = ac.step(observations)
        next_observations, rewards, terminals, _ = envs.step(predictions["actions"])
        buffer_container.store(
            observations=observations,
            actions=to_numpy(predictions["a"]),
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            values=to_numpy(predictions["v"]),
            logps=to_numpy(predictions["logp"]),
        )

        # Render one of the environments
        if render:
            # get atoms object (cannot naively take the first one, since we might have discarded some envs)
            if render_id in buffer_container.active_buffers:
                updated_id = buffer_container.active_buffers.index(render_id)
                atoms = ac.observation_space.canvas_space.to_atoms(next_observations[updated_id][0])
                gui.images.initialize([atoms])
                # gui.set_frame()
                gui.reset_view("x")
                # overwrite focus method
                # gui.scale = 40
                # gui.focus()
                # gui._do_zoom(0.9)
                # gui.draw()

                # viewer.set_atoms(atoms)
                # print('rendering')

            # else:
            #    gui.exit()

        for i, (env, term, i_original, next_obs) in enumerate(
            zip(envs.environments, terminals, buffer_container.active_buffers, next_observations)
        ):
            if term:
                env.n_eps += 1
                bag_or_int = next_obs[1]
                atoms_to_go = bag_or_int if isinstance(bag_or_int, int) else sum(bag_or_int)
                if atoms_to_go == 0:
                    final_atoms = ac.observation_space.canvas_space.to_atoms(next_obs[0])
                    print(f"Successfully ---------------------- : {final_atoms}")
                    rollout_trajs[i_original].append(final_atoms)
                else:
                    print(f"----------------------------- Failed: {atoms_to_go} atoms remaining")
                if num_episodes_combined:
                    if sum([len(rollout_traj) for rollout_traj in rollout_trajs]) >= num_episodes_combined:
                        break

        if num_episodes_combined:
            if sum([len(rollout_traj) for rollout_traj in rollout_trajs]) >= num_episodes_combined:
                break

        # Bookkeeping for discarding envs that have completed num_episodes
        if isinstance(num_episodes, int):
            rollouts_done = [env.n_eps >= num_episodes for env in envs.environments]
        elif isinstance(num_episodes, list):
            rollouts_done = [
                env.n_eps >= num_episodes[i] for env, i in zip(envs.environments, buffer_container.active_buffers)
            ]
        envs.environments = [env for env, done in zip(envs.environments, rollouts_done) if not done]
        next_observations = [obs for obs, done in zip(next_observations, rollouts_done) if not done]
        terminals = [term for term, done in zip(terminals, rollouts_done) if not done]
        buffer_container.active_buffers = [
            buffer_container.active_buffers[i] for i, done in enumerate(rollouts_done) if not done
        ]

        # Reset environment if state is terminal to get valid next observation
        observations = envs.reset_if_terminal(next_observations, terminals)

        counter += 1
        if envs.environments == []:
            break

    # max_returns = [max(buffer_container.ep_ret_unmixed[i]) for i in range(num_envs)]
    # assert len(max_returns) == num_envs, f'len(max_returns)={len(max_returns)}, num_envs={num_envs}'

    # info = {
    #     'return_mean': np.mean(buffer_container.episodic_returns).item(),
    #     'return_std': np.std(buffer_container.episodic_returns).item(),
    #     'episode_length_mean': np.mean(buffer_container.episode_lengths).item(),
    #     'episode_length_std': np.std(buffer_container.episode_lengths).item(),
    #     'max_returns': max_returns,
    #     'rollout_trajs': rollout_trajs,
    #     'relaxed_energies': relaxed_energies,
    #     'zero_shot_energies': zero_shot_energies
    # }

    if render:
        gui.exit()

    info = {
        "rollout_time": time.time() - start_time,
        "return_mean": np.mean(buffer_container.episodic_returns).item(),
        "return_std": np.std(buffer_container.episodic_returns).item(),
        "episode_length_mean": np.mean(buffer_container.episode_lengths).item(),
        "episode_length_std": np.std(buffer_container.episode_lengths).item(),
    }

    if output_trajs:
        info.update(
            {
                "rollout_trajs": rollout_trajs,
                "buf_con": buffer_container,
            }
        )

    return info


def batch_rollout_with_logging(
    ac: AbstractActorCritic,
    envs: VecEnv,
    buffer_container: PPOBufferContainer,
    num_steps: int = None,
    num_episodes: int = None,
    logger: Logger = None,
    eval: bool = False,
) -> dict:

    assert num_steps is not None or num_episodes is not None

    if num_steps is not None:
        assert num_steps % envs.get_size() == 0
        num_iters = num_steps // envs.get_size()
    else:
        num_iters = np.inf

    if num_episodes is not None:
        assert envs.get_size() == 1
    else:
        num_episodes = np.inf

    start_time = time.time()

    counter = 0
    observations = envs.reset()
    reward_total = np.zeros(envs.get_size())
    gaussian_stats_rollout = {}
    reward_terms = []
    reward_names = set()
    metrics = []
    metrics_names = set()
    while counter < num_iters and buffer_container.get_num_episodes() < num_episodes:
        predictions = ac.step(observations)

        if "gaussian_stats" in predictions.keys():
            for key in predictions["gaussian_stats"]:
                if key not in gaussian_stats_rollout.keys():
                    gaussian_stats_rollout[key] = []
                gaussian_stats_rollout[key].extend(predictions["gaussian_stats"][key])

        next_observations, rewards, terminals, step_info = envs.step(predictions["actions"])
        reward_total += np.array(rewards)

        # Log trajectories if terminal
        for i, (next_ob, total_env_reward, terminal, done_info) in enumerate(
            zip(next_observations, reward_total, terminals, step_info)
        ):
            if terminal:
                if logger is not None:
                    # print(f"worker {i} finished episode at step {counter}")
                    # print(f"done_info: {done_info}")
                    logger.save_episode_RL(
                        state=next_ob, total_reward=total_env_reward, info=done_info, name="eval" if eval else "train"
                    )
                if "new_rewards" in done_info:
                    reward_terms.append(done_info["new_rewards"])
                    reward_names.update(done_info["new_rewards"].keys())
                if "metrics" in done_info:
                    # pop Nones from dict
                    for k in done_info["metrics"].copy():
                        if done_info["metrics"][k] is None:
                            done_info["metrics"].pop(k)

                    metrics.append(done_info["metrics"])
                    metrics_names.update(done_info["metrics"].keys())

                reward_total[i] = 0

        buffer_container.store(
            observations=observations,
            actions=to_numpy(predictions["a"]),
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            values=to_numpy(predictions["v"]),
            logps=to_numpy(predictions["logp"]),
        )

        # Reset environment if state is terminal to get valid next observation
        observations = envs.reset_if_terminal(next_observations, terminals)

        if counter == num_iters - 1:
            # Note: finished trajectories will not be affected by this
            predictions = ac.step(observations)
            buffer_container.finish_paths(to_numpy(predictions["v"]))

        counter += 1

    info = {
        "rollout_time": time.time() - start_time,
        "return_mean": np.mean(buffer_container.episodic_returns).item(),
        "return_std": np.std(buffer_container.episodic_returns).item(),
        "episode_length_mean": np.mean(buffer_container.episode_lengths).item(),
        "episode_length_std": np.std(buffer_container.episode_lengths).item(),
    }
    if reward_terms != []:
        info.update({k: np.mean([rew_dict[k] for rew_dict in reward_terms if k in rew_dict]) for k in reward_names})
    if metrics != []:
        info.update({k: np.mean([met_dict[k] for met_dict in metrics if k in met_dict]) for k in metrics_names})

    # import logging
    # logging.info(f'buffer_container.episodic_returns length: {len(buffer_container.episodic_returns)}')
    # logging.info(f'buffer_container.episodic_returns mean: {np.mean(buffer_container.episodic_returns).item()}')
    # new_rews = {}
    # for k in reward_terms[0]:
    #     new_rews[k] = np.mean([d[k] for d in reward_terms])
    # total_rew = sum([new_rews[k] for k in new_rews])
    # logging.info(f'new rewards: {new_rews}')
    # logging.info(f'total reward: {total_rew}')

    if gaussian_stats_rollout != {}:
        for key in gaussian_stats_rollout.keys():
            info[key + "_mean"] = np.mean(gaussian_stats_rollout[key]).item()
            info[key + "_std"] = np.std(gaussian_stats_rollout[key]).item()

    return info


def to_dict_of_lists(data: List[List[Any]], names: List[str]) -> Dict[str, List[Any]]:
    """Convert a list of lists to a dict of lists."""
    assert len(data) == len(names), "The length of data and names must be the same."
    return {name: data[i] for i, name in enumerate(names)}


def merge_rollouts_dicts(rollouts1: dict, rollouts2: dict) -> dict:
    """Merge two dicts of rollouts into one dict of rollouts."""
    # TODO: Add buf_con to merge_rollouts()
    assert rollouts1.keys() == rollouts2.keys()
    merged_rollouts = {}
    for key in rollouts1.keys():
        merged_rollouts[key] = rollouts1[key] + rollouts2[key]
    return merged_rollouts


def rollout_argmax_and_stoch(ac: AbstractActorCritic, eval_envs: VecEnv, num_episodes: int = 20):
    formulas = util.get_str_formulas_from_vecenv(eval_envs)
    n_envs = eval_envs.get_size()
    gamma = lam = 1.0

    with torch.no_grad():
        # Argmax rollout
        ac.training = False
        eval_container = PPOBufferContainerDeploy(size=n_envs, gamma=gamma, lam=lam)
        argmax_rollout = rollout_n_eps_per_env(
            ac, deepcopy(eval_envs), buffer_container=eval_container, num_episodes=1, output_trajs=True
        )
        argmax_rollout["rollout_trajs"] = to_dict_of_lists(argmax_rollout["rollout_trajs"], names=formulas)

        # Stochastic rollout
        ac.training = True
        eval_container = PPOBufferContainerDeploy(size=n_envs, gamma=gamma, lam=lam)
        stoch_rollout = rollout_n_eps_per_env(
            ac, deepcopy(eval_envs), buffer_container=eval_container, num_episodes=num_episodes, output_trajs=True
        )
        stoch_rollout["rollout_trajs"] = to_dict_of_lists(stoch_rollout["rollout_trajs"], names=formulas)

    # Merge rollouts
    final_atoms = merge_rollouts_dicts(argmax_rollout["rollout_trajs"], stoch_rollout["rollout_trajs"])

    return final_atoms


def rollout_stoch(ac: AbstractActorCritic, eval_envs: VecEnv, num_episodes: int = 20):
    formulas = util.get_str_formulas_from_vecenv(eval_envs)
    n_envs = eval_envs.get_size()
    gamma = lam = 1.0

    with torch.no_grad():
        ac.training = True
        eval_container = PPOBufferContainerDeploy(size=n_envs, gamma=gamma, lam=lam)
        stoch_rollout = rollout_n_eps_per_env(
            ac, deepcopy(eval_envs), buffer_container=eval_container, num_episodes=num_episodes, output_trajs=True
        )
        stoch_rollout["rollout_trajs"] = to_dict_of_lists(stoch_rollout["rollout_trajs"], names=formulas)

    return stoch_rollout["rollout_trajs"]
