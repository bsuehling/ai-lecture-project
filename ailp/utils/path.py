import os
import pathlib
import pickle
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn


def save_model(
    model: dict | nn.Module, directory: str | os.PathLike, epoch: int | None = None
) -> None:
    """Save the model in the specified directory and as the latest model.

    Parameters
    ----------
    model : dict | nn.Module
        The model to save.
    directory : str | os.PathLike
        In which direcory to save the model
    epoch : int | None = None
        The epoch in which the model was created.
    """
    is_torch_model = isinstance(model, nn.Module)
    file_extension = "pt" if is_torch_model else "pickle"
    file_name = "model" if epoch is None else f"model{epoch:03d}"
    file_path = os.path.join(directory, f"{file_name}.{file_extension}")

    latest_dir = os.path.join("data", "models", "latest")
    latest_torch = os.path.join(latest_dir, "model.pt")
    latest_pickle = os.path.join(latest_dir, "model.pickle")
    os.makedirs(latest_dir, exist_ok=True)
    pathlib.Path.unlink(latest_torch, missing_ok=True)
    pathlib.Path.unlink(latest_pickle, missing_ok=True)

    if is_torch_model:
        state_dict = model.state_dict()
        torch.save(state_dict, file_path)
        torch.save(state_dict, latest_torch)
    else:
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        with open(latest_pickle, "wb") as f:
            pickle.dump(model, f)


def load_model(model_path: str | os.PathLike) -> dict[str, Any]:
    """Load a model from the given model path.

    Parameters
    ----------
    model_path : str | os.PathLike
        The path from where to save the model, including filename.

    Returns
    -------
    Any
        The loaded model.
    """
    if model_path.endswith(".pt"):
        return torch.load(model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def create_unique_dir(approach: str) -> str:
    """Use the current datetime to create a new unique directory for the given approach.

    Parameters
    ----------
    approach : str
        Which approach to create the directory for.

    Returns
    -------
    str
        The path to the new directory.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    path = os.path.join("data", "models", approach, timestamp)
    os.makedirs(path, exist_ok=True)
    return path


def get_latest_path() -> Any:
    """Load the latest saved model.

    Returns
    -------
    Any
        The model.
    """
    latest_dir = os.path.join("data", "models", "latest")
    latest_torch = os.path.join(latest_dir, "model.pt")
    latest_pickle = os.path.join(latest_dir, "model.pickle")
    if os.path.exists(latest_torch):
        return latest_torch
    return latest_pickle


def get_model_path(approach: str, unique_dir: str, epoch: int | None = None):
    """Find the path from which to load a model.

    Parameters
    ----------
    approach : str
        For which approach to load the model.
    unique_dir : str
        For which unique dir, i.e., which run to load the model.
    epoch : int | None = None
        From which epoch to load the model

    Returns
    -------
    str
        The model path.

    Raises
    ------
    FileNotFoundError
        If model path deos not exist.
    """
    # file_extension = "pt" if is_torch_model else "pickle"
    directory = os.path.join("data", "models", approach, unique_dir)
    file_name = "model" if epoch is None else f"model{epoch:03d}"
    file_path_pt = os.path.join(directory, f"{file_name}.pt")
    file_path_pkl = os.path.join(directory, f"{file_name}.pickle")

    if os.path.exists(file_path_pt):
        return file_path_pt
    if os.path.exists(file_path_pkl):
        return file_path_pkl
    raise FileNotFoundError("Could not find the desired model")
