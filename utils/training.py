import datetime
import os

import torch

from sys_config import BASE_DIR


def save_checkpoint(state, name, path=None, timestamp=False, tag=None,
                    verbose=False):
    """
    Save a trained model, along with its optimizer, in order to be able to
    resume training
    Args:
        path (str): the directory, in which to save the checkpoints
        timestamp (bool): whether to keep only one model (latest), or keep every
            checkpoint

    Returns:

    """
    now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")

    if tag is not None:
        if isinstance(tag, str):
            name += "_{}".format(tag)
        elif isinstance(tag, list):
            for t in tag:
                name += "_{}".format(t)
        else:
            raise ValueError("invalid tag type!")

    if timestamp:
        name += "_{}".format(now)

    name += ".pt"

    if path is None:
        path = os.path.join(BASE_DIR, "checkpoints")

    file = os.path.join(path, name)

    if verbose:
        print("saving checkpoint:{} ...".format(name))

    torch.save(state, file)

    return name


def load_checkpoint(name, path=None, device=None):
    """
    Load a trained model, along with its optimizer
    Args:
        name (str): the name of the model
        path (str): the directory, in which the model is saved

    Returns:
        model, optimizer

    """
    if path is None:
        path = os.path.join(BASE_DIR, "checkpoints")

    model_fname = os.path.join(path, "{}.pt".format(name))

    print("Loading checkpoint `{}` ...".format(model_fname), end=" ")

    with open(model_fname, 'rb') as f:
        state = torch.load(f, map_location="cpu")

    print("done!")

    return state
