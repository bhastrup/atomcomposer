import logging
import os
import re
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence

import torch

from src.rl.spaces import ObservationSpace, ActionSpace
from src.tools.util import count_vars

from src.agents.base import AbstractActorCritic
from src.agents.painn.agent import PainnAC


@dataclass
class ModelPathInfo:
    path: str
    tag: str
    num_steps: int


class ModelIO:
    def __init__(self, directory: str, tag: str, keep: bool = False, checkpoints: Sequence[int] = None) -> None:
        """Make sure to distinguish between number of env steps and number of training steps (training iters)."""
        self.directory = directory
        self.tag = tag
        self.keep = keep
        self.old_path: Optional[str] = None

        self._steps_string = "_steps-"
        self._suffix = ".model"
        self._iter_suffix = ".txt"

        if checkpoints:
            if 0 not in checkpoints:
                checkpoints = [0] + list(checkpoints)
        self._checkpoints = checkpoints
        self._checkpoints_string = "_CP-"

    def _get_model_filename(self, num_steps: int) -> str:
        return self.tag + self._steps_string + str(num_steps) + self._suffix

    def _list_file_paths(self) -> Sequence[str]:
        all_paths = [os.path.join(self.directory, f) for f in os.listdir(self.directory)]
        return [path for path in all_paths if os.path.isfile(path)]

    def _parse_model_path(self, path: str) -> Optional[ModelPathInfo]:
        filename = os.path.basename(path)
        regex = re.compile(rf"(?P<tag>.*){self._steps_string}(?P<num_steps>\d+){self._suffix}")
        match = regex.match(filename)
        if not match:
            return None

        return ModelPathInfo(
            path=path,
            tag=match.group("tag"),
            num_steps=int(match.group("num_steps")),
        )

    def save(self, module: AbstractActorCritic, num_steps: int) -> None:
        if not self.keep and self.old_path:
            logging.debug(f"Deleting old model: {self.old_path}")
            os.remove(self.old_path)

        filename = self._get_model_filename(num_steps)
        path = os.path.join(self.directory, filename)
        logging.debug(f"Saving model: {path}")
        torch.save(obj=module, f=path)
        self.old_path = path

    def load(self, device: torch.device, path: str) -> Tuple[AbstractActorCritic, int]:
        model_info = self._parse_model_path(path)

        if model_info is None:
            raise RuntimeError(f"Cannot find model '{path}'")

        logging.info(f"Loading model: {model_info.path}")
        model = torch.load(f=model_info.path, map_location=device)

        return model, model_info.num_steps

    def load_latest(self, device: torch.device) -> Tuple[AbstractActorCritic, int]:
        all_file_paths = self._list_file_paths()
        model_infos = [self._parse_model_path(path) for path in all_file_paths]
        selected_model_infos = [info for info in model_infos if info and info.tag == self.tag]

        if len(selected_model_infos) == 0:
            raise RuntimeError(f"Cannot find model to load in '{self.directory}'")

        latest_model_info = max(selected_model_infos, key=lambda info: info.num_steps)

        logging.info(f"Loading model: {latest_model_info.path}")
        model = torch.load(f=latest_model_info.path, map_location=device)
        model.set_device(device)

        return model, latest_model_info.num_steps

    def save_after_full_replica(self, module: AbstractActorCritic, num_steps: int, epochs: int) -> None:
        filename = self.tag + "_epoch_" + str(epochs) + self._steps_string + str(num_steps) + self._suffix
        path = os.path.join(self.directory, filename)
        logging.debug(f"Saving agent after epoch {epochs} at: {path}")
        torch.save(obj=module, f=path)

    def save_if_checkpoint(self, module: AbstractActorCritic, num_steps: int) -> None:
        if num_steps in self._checkpoints:
            cp_num = str(self._checkpoints.index(num_steps))
            filename = self.tag + self._checkpoints_string + cp_num + self._steps_string + str(num_steps) + self._suffix
            path = os.path.join(self.directory, filename)
            logging.debug(f"Saving agent at checkpoint {cp_num} at step number {str(num_steps)} at: {path}")
            torch.save(obj=module, f=path)

            return path

    def save_at_skill_change(self, module: AbstractActorCritic, num_steps: int, skill: int) -> None:
        """When using curriculum learning, we want to save the model at the point where skill changes."""
        filename = self._get_model_filename(num_steps)
        filename += "-skill-" + str(skill)

        path = os.path.join(self.directory, filename)
        logging.debug(f"Saving model: {path}")
        torch.save(obj=module, f=path)


def get_model(
    config: dict, observation_space: ObservationSpace, action_space: ActionSpace, device: torch.device, tag: str
) -> Tuple[AbstractActorCritic, int, ModelIO, int]:

    model_handler = ModelIO(
        directory=config["model_dir"], tag=tag, keep=config["keep_models"], checkpoints=config["checkpoints"]
    )
    if config["load_latest"]:
        model, start_num_steps = model_handler.load_latest(device=device)
    elif config["load_model"] is not None:
        model, start_num_steps = model_handler.load(device=device, path=config["load_model"])
    else:
        model = build_model(config, observation_space=observation_space, action_space=action_space, device=device)
        start_num_steps = 0
    model.set_device(device)

    var_counts = count_vars(model)
    logging.info(f"Number of parameters: {var_counts}")

    return model, start_num_steps, model_handler, var_counts


def build_model(
    config: dict, observation_space: ObservationSpace, action_space: ActionSpace, device: torch.device
) -> AbstractActorCritic:
    if config["model"] == "painn":
        return PainnAC(
            observation_space=observation_space,
            action_space=action_space,
            min_max_distance=(config["min_mean_distance"], config["max_mean_distance"]),
            network_width=config["network_width"],
            num_interactions=config["num_interactions"],
            cutoff=config["cutoff"],
            hydrogen_delay=config["hydrogen_delay"],
            no_hydrogen_focus=config["no_hydrogen_focus"],
            rms_norm_update=config["rms_norm_update"],
            device=device,
        )
    else:
        raise RuntimeError(f"Model '{config['model']}' is not available.")
