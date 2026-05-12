import json
import pickle
import logging
from typing import Any, Dict


class IOHandler:
    @staticmethod
    def read_json(file_path: str) -> Dict:
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from file: {file_path}")
            return {}

    @staticmethod
    def write_json(data: Dict, file_path: str) -> None:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def read_pickle(file_path: str) -> Any:
        try:
            with open(file_path, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except pickle.PickleError:
            logging.error(f"Error reading pickle file: {file_path}")

    @staticmethod
    def write_pickle(data: Any, file_path: str) -> None:
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
