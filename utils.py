import glob
import json
import os

import pandas as pd

from constants import NEURONPEDIA_API_KEY_ENV_VAR


def save_df(df: pd.DataFrame, foldername: str, filename: str) -> str:
    """
    Save a DataFrame to a CSV file.
    """
    os.makedirs(foldername, exist_ok=True)
    save_path = os.path.join(foldername, filename)
    df.to_csv(save_path, index=False)
    return save_path

def save_json(json_dict: dict, save_path: str) -> None:
    """
    Save a dictionary to a JSON file with pretty formatting.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as json_file:
        json.dump(json_dict, json_file, indent=4)

def load_json(save_path: str) -> dict:
    """
    Load a dictionary from a JSON file.
    """
    with open(save_path, "r") as json_file:
        json_dict = json.load(json_file)
    return json_dict

def get_api_key() -> str:
    """
    Retrieve the Neuronpedia API key from environment variables.
    """
    assert os.environ.get(NEURONPEDIA_API_KEY_ENV_VAR), f"Must set {NEURONPEDIA_API_KEY_ENV_VAR} in .env"
    return os.environ.get(NEURONPEDIA_API_KEY_ENV_VAR)

def get_most_recent_file(directory_path: str, pattern: str = "*") -> str | None:
    """
    Get the most recently modified file in a directory matching a pattern.
    """
    full_pattern = os.path.join(directory_path, pattern)
    list_of_files = glob.glob(full_pattern)

    if not list_of_files:
        return None

    # Sort files by their modification time (getmtime) in ascending order
    # The last element after sorting will be the most recent
    list_of_files.sort(key=os.path.getmtime)
    return list_of_files[-1]