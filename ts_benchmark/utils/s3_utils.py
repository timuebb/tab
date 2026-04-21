"""Thin utilities for S3-backed checkpoint and dataset access.

Model files throughout the codebase can use :func:`get_checkpoint_path` to
transparently resolve checkpoint paths: when the ``USE_S3_CHECKPOINTS``
environment variable is set the checkpoint is downloaded from S3 on first use;
otherwise the original relative path is returned unchanged.
"""

import os


def get_checkpoint_path(local_relative_path: str) -> str:
    """Returns the path to a model checkpoint, downloading from S3 when needed.

    When the environment variable ``USE_S3_CHECKPOINTS`` is set to ``1``,
    ``true``, or ``yes``, this function delegates to
    :func:`scripts.dih_connect.s3_data_access.get_checkpoint_path` which
    downloads the checkpoint to ``/tmp/tab_checkpoints/`` and returns the
    local path.

    Otherwise *local_relative_path* is returned unchanged so that existing
    local workflows are not affected.

    :param local_relative_path: Path relative to the project root, e.g.
        ``"ts_benchmark/baselines/LLM/checkpoints/gpt2"``.
    :return: Resolved path as a string suitable for passing to model loaders.
    """
    if os.environ.get("USE_S3_CHECKPOINTS", "").lower() not in ("1", "true", "yes"):
        return local_relative_path
    from scripts.dih_connect.s3_data_access import get_checkpoint_path as _s3_get
    return _s3_get(local_relative_path)
