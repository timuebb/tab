# -*- coding: utf-8 -*-
"""Utility helpers for PyTorch device handling.

Using this module keeps device selection consistent across the codebase:
GPU (CUDA) is used when available; otherwise the code falls back to CPU.
This allows the benchmarks to run in CPU-only environments such as
OpenShift pods without an NVIDIA GPU.
"""

import torch


def get_torch_device() -> torch.device:
    """Return the best available PyTorch device.

    Returns a CUDA device when a GPU is available, otherwise returns the CPU
    device.  All model and tensor allocations should use this function instead
    of hard-coding ``"cuda"`` so that the code runs correctly in CPU-only
    environments.

    :return: ``torch.device("cuda")`` or ``torch.device("cpu")``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
