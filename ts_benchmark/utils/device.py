# -*- coding: utf-8 -*-
import torch


def get_device() -> torch.device:
    """Return the best available compute device: CUDA, then MPS (Apple Silicon), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
