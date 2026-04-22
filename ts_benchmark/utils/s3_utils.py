"""Thin utilities for S3-backed checkpoint, dataset, and result access.

Model files throughout the codebase can use :func:`get_checkpoint_path` to
transparently resolve checkpoint paths: when the ``USE_S3_CHECKPOINTS``
environment variable is set the checkpoint is downloaded from S3 on first use;
otherwise the original relative path is returned unchanged.

Result files can be mirrored to S3 via :func:`upload_result`: when the
``USE_S3_RESULTS`` environment variable is set the local result file is
uploaded to S3 under the prefix configured by ``DIH_S3_RESULTS_PREFIX``
(default ``data/tab/results``).
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


def upload_result(local_file_path: str) -> None:
    """Uploads a local result file to S3 when ``USE_S3_RESULTS`` is enabled.

    When the environment variable ``USE_S3_RESULTS`` is set to ``1``, ``true``,
    or ``yes``, this function uploads *local_file_path* to the S3 bucket
    configured via the standard ``DIH_S3_*`` environment variables.  The S3
    key is built by taking the path relative to ``{ROOT_PATH}/result/`` and
    prepending the prefix from ``DIH_S3_RESULTS_PREFIX`` (default
    ``data/tab/results``).

    If ``USE_S3_RESULTS`` is not set this function is a no-op so that
    existing local workflows are not affected.

    :param local_file_path: Absolute path to the local result file.
    """
    if os.environ.get("USE_S3_RESULTS", "").lower() not in ("1", "true", "yes"):
        return

    from scripts.dih_connect.s3_data_access import S3DataAccess
    from ts_benchmark.common.constant import ROOT_PATH

    access = S3DataAccess()
    result_root = os.path.join(ROOT_PATH, "result")
    # Build an S3 key relative to the result root directory
    rel_path = os.path.relpath(local_file_path, result_root)
    s3_key = "/".join(
        [access.results_prefix.strip("/")]
        + rel_path.replace("\\", "/").split("/")
    )
    access.upload_result_file(local_file_path, s3_key)
