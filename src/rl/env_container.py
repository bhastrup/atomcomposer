import concurrent.futures
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import List, Tuple

import gymnasium as gym
import numpy as np

# The class is based on: Baselines https://github.com/openai/baselines.
from src.rl.spaces import ObservationType


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    @abstractmethod
    def reset(self) -> List[ObservationType]:
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        raise NotImplementedError

    @abstractmethod
    def step_async(self, actions) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        raise NotImplementedError

    @abstractmethod
    def step_wait(self) -> Tuple[List[ObservationType], np.ndarray, np.ndarray, List[dict]]:
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        raise NotImplementedError

    def step(self, actions) -> Tuple[List[ObservationType], np.ndarray, np.ndarray, List[dict]]:
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        raise NotImplementedError

    @abstractmethod
    def get_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset_if_terminal(self, observations: List[ObservationType], terminals: List[bool]):
        raise NotImplementedError


# This class is based on: DeepRL https://github.com/ShangtongZhang/DeepRL.

# Update: Attempted simple threading for step_wait() method, but if we intend to do CPU heavy tasks like relaxations,
# we should use multiprocessing instead. See this one:
# https://github.com/ZimmermanGroup/conformer-rl/blob/master/src/conformer_rl/environments/environment_wrapper.py
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py


class SimpleEnvContainer(VecEnv):
    def __init__(self, environments: List[gym.Env]):
        super().__init__()
        self.environments = environments

        self.actions = None

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    # def step_wait(self):
    #     assert self.actions and len(self.environments) == len(self.actions)

    #     def step(env, action):
    #         obs, reward, done, info = env.step(action)
    #         return obs, reward, done, info

    #     use_threading = False
    #     if use_threading:
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             results = [executor.submit(step, env, action) \
    #                     for env, action in zip(self.environments, self.actions)]
    #             obs_list, rewards, done_list, infos = zip(*[result.result() for result in results])
    #     else:
    #         results = [step(env, action) for env, action in zip(self.environments, self.actions)]
    #         obs_list, rewards, done_list, infos = zip(*results)

    #     return obs_list, np.array(rewards), np.array(done_list), infos

    def step_wait(self):
        assert self.actions and len(self.environments) == len(self.actions)

        def step(env, action):
            obs, reward, done, info = env.step(action)
            return obs, reward, done, info

        use_multiprocessing = False  # Set to True to enable multiprocessing
        if use_multiprocessing:
            with mp.Pool(processes=mp.cpu_count()) as pool:  # Use all available cores
                results = pool.starmap(step, zip(self.environments, self.actions))
            obs_list, rewards, done_list, infos = zip(*results)
        else:
            # Sequential or threaded approach
            use_threading = False
            if use_threading:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = [
                        executor.submit(step, env, action) for env, action in zip(self.environments, self.actions)
                    ]
                    obs_list, rewards, done_list, infos = zip(*[result.result() for result in results])
            else:
                results = [step(env, action) for env, action in zip(self.environments, self.actions)]
                obs_list, rewards, done_list, infos = zip(*results)

        return obs_list, np.array(rewards), np.array(done_list), infos

    def reset(self):
        return [env.reset() for env in self.environments]

    def reset_if_terminal(self, observations: List[ObservationType], terminals: List[bool]) -> List[ObservationType]:
        assert len(self.environments) == len(observations) == len(terminals)

        new_observations = []
        for env, observation, terminal in zip(self.environments, observations, terminals):
            if terminal:
                new_observations.append(env.reset())
            else:
                new_observations.append(observation)

        return new_observations

    def get_size(self) -> int:
        return len(self.environments)

    def close(self):
        pass

    def render(self, mode="human"):
        raise NotImplementedError
